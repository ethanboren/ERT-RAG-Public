from dataclasses import dataclass
from typing import List, Dict

import evaluate


@dataclass
class RAGMetrics:
    bleu: float
    rouge: Dict[str, float]
    meteor: float
    context_relevance: float
    answer_faithfulness: float


class RAGEvaluator:
    def __init__(self, llm):

        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.meteor = evaluate.load('meteor')
        self.llm = llm

    def evaluate_generation(self,
                            predicted: str,
                            reference: str,
                            context: List[str]) -> RAGMetrics:
        """Evaluate RAG output using multiple metrics."""
        # Standard NLG metrics
        bleu_score = self.bleu.compute(
            predictions=[predicted],
            references=[reference]
        )['bleu']

        rouge_scores = self.rouge.compute(
            predictions=[predicted],
            references=[reference],
            use_stemmer=True
        )

        meteor_score = self.meteor.compute(
            predictions=[predicted],
            references=[reference]
        )['meteor']

        # RAG-specific metrics
        context_relevance = self._evaluate_context_relevance(predicted, context)
        answer_faithfulness = self._evaluate_faithfulness(predicted, context)

        return RAGMetrics(
            bleu=bleu_score,
            rouge=rouge_scores,
            meteor=meteor_score,
            context_relevance=context_relevance,
            answer_faithfulness=answer_faithfulness
        )

    def _evaluate_context_relevance(self, answer: str, context: List[str]) -> float:
        """Evaluate how relevant the retrieved context is to the answer."""
        prompt = """Rate how relevant the context is to the given answer on a scale of 0-1.
        Consider:
        1. Does the context contain the information needed for the answer?
        2. Is there unnecessary context?
    
        Answer: {answer}
        Context: {context}
    
        Respond with only a number between 0 and 1:"""
        return self._evaluate_with_prompt(prompt, answer, context)

    def _evaluate_faithfulness(self, answer: str, context: List[str]) -> float:
        """Evaluate if the answer is faithful to the given context."""
        prompt = """Rate how faithful the answer is to the given context on a scale of 0-1.
        Consider:
        1. Does the answer contain claims not supported by context?
        2. Does it contradict the context?
    
        Answer: {answer}
        Context: {context}
    
        Respond with only a number between 0 and 1:"""
        return self._evaluate_with_prompt(prompt, answer, context)

    def _extract_score_from_response(self, response: str) -> float:
            """Extract a numeric score from LLM response and clamp it between 0 and 1."""
            try:
                # First try direct conversion
                score = float(response.strip())
                return min(max(score, 0.0), 1.0)
            except ValueError:
                try:
                    # Fall back to regex if direct conversion fails
                    import re
                    numbers = re.findall(r"([0-9]*[.])?[0-9]+", response)
                    if numbers:
                        score = float(numbers[0])
                        return min(max(score, 0.0), 1.0)
                except:
                    pass
            return 0.0

    def _evaluate_with_prompt(self, prompt: str, answer: str, context: List[str]) -> float:
        """Evaluate using a given prompt template."""
        print(f"Debug - Context received: {context}")  # Debug line
        response = self.llm.invoke(
            prompt.format(
                answer=answer,
                context="\n".join(context)
            )
        ).content
        print(f"Debug - LLM response: {response}")  # Debug line
        score = self._extract_score_from_response(response)
        print(f"Debug - Extracted score: {score}")  # Debug line
        return score