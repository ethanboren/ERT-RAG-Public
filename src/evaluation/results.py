from dataclasses import dataclass
from typing import List

from src.evaluation.metrics import RAGMetrics


@dataclass
class TestResult:
    question: str
    answer: str
    reference: str
    metrics: RAGMetrics


class TestResultsCollector:
    def __init__(self):
        self.results: List[TestResult] = []

    def add_result(self, question: str, answer: str, reference: str, metrics: RAGMetrics):
        self.results.append(TestResult(question, answer, reference, metrics))

    def get_results(self) -> List[TestResult]:
        return self.results
