import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.results import TestResultsCollector
from src.config.settings import llm
from src.core.rag_chain import AdvancedRAGChain
from src.evaluation.metrics import RAGEvaluator


def analyze_results():
    """Analyze and print summary statistics of test results."""
    df = pd.read_csv('tests/test_results.csv')
    print(f"Average scores across all questions:")
    print(f"BLEU: {df['BLEU'].mean():.3f}")
    print(f"ROUGE-1: {df['ROUGE-1'].mean():.3f}")
    print(f"METEOR: {df['METEOR'].mean():.3f}")


class TestRAGEvaluation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.evaluator = RAGEvaluator(llm)
        self.rag = AdvancedRAGChain()
        self.results_collector = TestResultsCollector()

        # Load test data
        test_data_path = Path('tests/data/test_qa_pairs.json')
        with open(test_data_path) as f:
            self.test_data = json.load(f)

    def test_rag_evaluation_pipeline(self):
        """Test full RAG evaluation pipeline."""
        for test_case in self.test_data:
            result = self.rag.query_with_eval(
                question=test_case["question"],
                reference_answer=test_case["reference_answer"]
            )

            self.results_collector.add_result(
                question=test_case["question"],
                answer=result["answer"],
                reference=test_case["reference_answer"],
                metrics=result["metrics"]
            )

            # Export results to CSV after all tests and analyze
            self._export_results()
            analyze_results()

    def _export_results(self):
        """Export test results to CSV, including column averages at the end."""
        output_path = Path('tests/test_results.csv')
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Question', 'BLEU', 'ROUGE-1', 'METEOR', 'Context Relevance', 'Answer Faithfulness'])

            bleu_list = []
            rouge1_list = []
            meteor_list = []
            context_rel_list = []
            answer_faith_list = []

            for result in self.results_collector.results:
                bleu = result.metrics.bleu
                rouge1 = result.metrics.rouge['rouge1']
                meteor = result.metrics.meteor
                context_rel = result.metrics.context_relevance
                answer_faith = result.metrics.answer_faithfulness

                writer.writerow([
                    result.question,
                    bleu,
                    rouge1,
                    meteor,
                    context_rel,
                    answer_faith
                ])

                bleu_list.append(bleu)
                rouge1_list.append(rouge1)
                meteor_list.append(meteor)
                context_rel_list.append(context_rel)
                answer_faith_list.append(answer_faith)

            writer.writerow([
                'Average',
                sum(bleu_list) / len(bleu_list) if bleu_list else '',
                sum(rouge1_list) / len(rouge1_list) if rouge1_list else '',
                sum(meteor_list) / len(meteor_list) if meteor_list else '',
                sum(context_rel_list) / len(context_rel_list) if context_rel_list else '',
                sum(answer_faith_list) / len(answer_faith_list) if answer_faith_list else '',
            ])
