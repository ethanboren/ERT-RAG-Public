import re
from src.utils.helpers import logger

class RelevanceScorer:
    def __init__(self, llm):
        self.llm = llm

    def rerank_documents(self, state):
        docs = state["context"]
        question = state["question"]

        logger.info(f"\nReranking {len(docs)} documents for question: {question}")

        # Step 1: Create batch prompts
        prompts = [
            f"""Rate the relevance of this document to the question on a scale of 0-1. 
                Return ONLY a number between 0 and 1.
            
                Question: {question}
                Document: {doc.page_content}
            
                Score (number only):"""
            for doc in docs
        ]

        # Step 2: Call the LLM in batch
        responses = self.llm.batch(prompts)

        # Step 3: Parse scores from responses
        scored_docs = []
        for doc, response in zip(docs, responses):
            content = response.content.strip()
            match = re.search(r'([0-9]*[.])?[0-9]+', content)
            if match:
                score = float(match.group())
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
            else:
                logger.warning(f"No valid score found in response: {content}")
                score = 0.0
            scored_docs.append((doc, score))

        # Step 4: Sort and return
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        logger.info("\nRanked documents:")
        for i, (doc, score) in enumerate(scored_docs, 1):
            logger.info(f"\nRank {i}/{len(docs)} (Score: {score:.3f}):")
            logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")

        state["context"] = [doc for doc, _ in scored_docs]
        return state
