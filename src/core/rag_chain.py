from typing import Dict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph

from src.config.settings import vector_store, llm
from src.core.advanced_rag import RelevanceScorer
from src.core.types import AdvancedState
from src.evaluation.metrics import RAGEvaluator

RAG_PROMPT = PromptTemplate.from_template("""
You are **Rocky**, the AI assistant of the EPFL Rocket Team (ERT). Your job is to provide **accurate, technical, and complete answers** based **strictly** on the documentation and prior conversation.

However, when the user is simply greeting you, asking how you are, or engaging in small talk (e.g., “Hey Rocky”, “Good morning”, “How’s it going?”), you may respond **naturally and politely** — no citation or technical context is needed in such cases. Keep it light and friendly.

## Chat History (Most Recent First)
{chat_history}

## Retrieved Context
{context}

## Current Question
{question}

## Answering Rules — You MUST follow all 12:
1. **Ground your answer strictly** in the retrieved context or prior chat history. Do not use external knowledge or make assumptions.
2. If the answer is **not clearly found** in the context or chat history, respond: `"I don’t have enough information to answer that question."`
3. When referring to facts, numbers, specifications, or names, **cite the source** in `[filename]` format immediately after the statement.
4. Use **Markdown formatting**:
   - Bullet points or numbered lists for structured data
   - Code blocks for code
   - Tables in Markdown when data is tabular
   - **Bold** for key terms or values
5. When a **table is referenced**, extract and interpret values properly. Perform calculations if necessary to clarify the answer.
6. Resolve **pronouns** ("it", "they", "this") or references using the most recent relevant part of the chat history.
7. Include **units** (e.g., N, mm, bar, °C) with all numerical values.
8. For each technical term, define it **briefly in parentheses** the first time it appears (unless already explained).
9. Be **precise, factual, and technical** — no vague language or speculation.
10. When summarizing long explanations, **highlight key insights first**, then add detail if necessary.
11. If asked to compare options, **explain trade-offs**, not just differences.
12. Keep the response **as concise as possible**, while ensuring full clarity and correctness.

Your response must be helpful, self-contained, and grounded in the provided context. Never assume or fabricate any information.
""")


# This function generates an answer based on the context
def generate(state):
    """Generates an answer using the LLM based on the retrieved context.

    Args:
        state (dict): Current state dictionary containing:
            - context (list): List of Document objects with relevant content
            - question (str): The user's question

    Returns:
        dict: Dictionary containing the generated answer
    """
    context_str = "\n\n".join(doc.page_content for doc in state["context"])
    chat_history_str = "\n".join([f"Human: {h}\nAssistant: {a}" for h, a in state.get("chat_history", [])])
    response = llm.invoke(
        RAG_PROMPT.format(
            context=context_str,
            question=state["question"],
            chat_history=chat_history_str)
    )
    return {"answer": response.content}


def query_rewriter(state):
    """Rewrites the search query to include technical synonyms and related terms.

    Args:
        state (dict): Current state dictionary containing:
            - question (str): Original user query
            - rewritten_query (str): Field to store primary rewritten query
            - expanded_queries (list): Field to store additional query variations

    Returns:
        dict: Updated state with rewritten and expanded queries
    """
    expand_prompt = PromptTemplate.from_template(
        """Given the original query, rewrite it using technical terms and suggest related variations.
        Format your response exactly as shown:
        Primary: [primary rewritten query]
        Variants:
        - [variant 1]
        - [variant 2]
        - [variant 3]

        Original query: {question}"""
    )

    try:
        expanded = llm.invoke(expand_prompt.format(question=state["question"]))
        content = expanded.content.strip().split('\n')

        # Extract primary query
        primary = content[0].replace('Primary:', '').strip()

        # Extract variants (skip header line)
        variants = [
            line.replace('-', '').strip()
            for line in content[2:]
            if line.strip() and line.strip() != 'Variants:'
        ]

        state["rewritten_query"] = primary if primary else state["question"]
        state["expanded_queries"] = variants[:3]  # Limit to top 3 variants

        # Debug logging
        print(f"Original: {state['question']}")
        print(f"Rewritten: {state['rewritten_query']}")
        print(f"Variants: {state['expanded_queries']}")

    except Exception as e:
        print(f"Error in query rewriting: {e}")
        state["rewritten_query"] = state["question"]
        state["expanded_queries"] = []

    return state


def assess_query_need_for_retrieval(state):
    """
    Determines whether the user's question requires external information retrieval.
    """
    chat_history = state.get('chat_history', '')
    assessment_prompt = (
        "You are Rocky, an AI assistant for the EPFL Rocket Team (ERT). "
        "You have access to general knowledge, natural conversation skills, and recent chat history. "
        "You do not have access to the ERT documentation unless it is explicitly quoted or summarized in the conversation. "
        "You should be able to answer any general, social, or conversational message (e.g., greetings, clarifications, general knowledge) using only your internal knowledge. "
        "Only respond 'no' if the question requires specific details from the ERT documentation that were not provided earlier. "
        "Be generous in assessing what you can answer based on general reasoning or prior messages. "
        "Respond strictly with 'yes' or 'no' — do not explain.\n\n"
        f"Chat History:\n{chat_history}\n\n"
        f"Question: {state['question']}"
    )

    response = llm.invoke(assessment_prompt)
    answer = response.content.strip().lower()
    print(f"Assessment response: {answer}")

    # Fix: Only do retrieval if the model says 'no'
    state["needs_retrieval"] = answer == 'no'
    return state


def multi_query_retrieval(state):
    """Performs a hybrid search using both primary and expanded queries.

    Args:
        state (dict): Current state dictionary containing:
            - rewritten_query (str): Primary search query
            - expanded_queries (list): Additional search variations
            - context (list): Field to store retrieved documents

    Returns:
        dict: Updated state with combined search results in context
    """
    all_results = []
    for query in [state["rewritten_query"]] + state["expanded_queries"]:
        results = vector_store.similarity_search(query=query, k=8)
        all_results.extend(results)

    # Deduplicate results based on page content
    state["context"] = list({doc.page_content: doc for doc in all_results}.values())
    return state


class AdvancedRAGChain:
    """Advanced RAG chain implementation with query rewriting and document re-ranking.

    This class extends the basic RAGChain by adding query expansion and
    relevance scoring capabilities for improved retrieval accuracy.
    """

    def __init__(self):
        """Initializes the advanced RAG chain with enhanced components."""
        self.relevance_scorer = RelevanceScorer(llm)  # Add this
        self.graph = self._build_advanced_graph()  # Use this instead

    def query_with_eval(self, question: str, reference_answer: str = None) -> Dict:
        """Query with evaluation if reference answer is provided."""
        result = self.graph.invoke({
            "question": question,
            "context": [],
            "answer": "",
            "search_kwargs": {"k": 5}
        })

        print(result["answer"])

        if reference_answer:
            evaluator = RAGEvaluator(llm)
            metrics = evaluator.evaluate_generation(
                predicted=result["answer"],
                reference=reference_answer,
                context=[doc.page_content for doc in result["context"]]
            )
            result["metrics"] = metrics

        return result

    def stream_response(self, question: str, chat_history=None) -> str:
        """Manually run each step of the graph except generation, then stream LLM output."""

        # Step 1: Initialize state
        if chat_history is None:
            chat_history = []
        state = {
            "question": question,
            "context": [],
            "answer": "",
            "chat_history": chat_history,
            "search_kwargs": {"k": 5}
        }

        # Manually run the graph steps (same as in _build_advanced_graph)
        state = self.initialize_state(state)
        state = assess_query_need_for_retrieval(state)

        if state["needs_retrieval"]:
            state = query_rewriter(state)
            state = multi_query_retrieval(state)
            state = self.relevance_scorer.rerank_documents(state)

        # Step 2: Format final prompt
        context_str = "\n\n".join(
            f"{doc.page_content}\n\nSource: [{doc.metadata.get('source', 'Unknown')}]"
            for doc in state["context"]
        )
        chat_history_str = "\n".join([f"Human: {h}\nAssistant: {a}" for h, a in state.get("chat_history", [])])

        stream = llm.stream(
            RAG_PROMPT.format(
                context=context_str,
                question=question,
                chat_history=chat_history_str
            )
        )

        # Yield token by token
        for chunk in stream:
            if chunk.content:
                yield chunk.content

    def initialize_state(self, state):
        """Initializes additional state fields for advanced processing.

        Args:
            state (dict): Current state dictionary

        Returns:
            dict: State with initialized advanced fields
        """
        state.update({
            "rewritten_query": state["question"],
            "expanded_queries": [],
            "scored_context": [],
            "chat_history": state.get("chat_history", []),
            "needs_retrieval": False
        })
        return state

    def _build_advanced_graph(self):
        """Builds and compiles an advanced RAG workflow graph.

        The graph includes additional nodes for query rewriting, hybrid search,
        and document re-ranking.

        Returns:
            Callable: Compiled StateGraph for the advanced RAG workflow
        """

        graph = StateGraph(AdvancedState)

        # Add nodes individually
        graph.add_node("initialize", self.initialize_state)
        graph.add_node("assess_retrieval", assess_query_need_for_retrieval)
        graph.add_node("rewrite", query_rewriter)
        graph.add_node("search", multi_query_retrieval)
        graph.add_node("rerank", self.relevance_scorer.rerank_documents)
        graph.add_node("generate", generate)

        # Add edges to connect the nodes
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "assess_retrieval")
        graph.add_conditional_edges(
            "assess_retrieval",
            lambda state: "rewrite" if state["needs_retrieval"] else "generate",
        )
        graph.add_edge("rewrite", "search")
        graph.add_edge("search", "rerank")
        graph.add_edge("rerank", "generate")

        return graph.compile()
