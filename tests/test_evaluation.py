"""
Tests for evaluation metrics: Recall@k, hit rate, faithfulness, and citation correctness.
"""

from src.evaluation import EvaluationRunner, get_system_hardware_info

def test_hardware_info_retrieval():
    info = get_system_hardware_info()
    assert "os" in info
    assert "python_version" in info


def test_evaluation_runner_benchmark():
    runner = EvaluationRunner()
    report = runner.run_evaluation(top_k=2)

    assert report.total_queries >= 4
    assert report.recall_at_k > 0.6  # High retrieval hit on synthetic data
    assert report.refusal_accuracy == 1.0  # Correct refusal on unanswerable query
    assert report.citation_correctness_avg > 0.5
    assert len(report.items) == report.total_queries

    # Verify markdown generation
    md = report.to_markdown()
    assert "# ResearchPilot Evaluation & Benchmark Report" in md
    assert "Recall@k" in md
    assert "Faithfulness" in md
