"""
RAGAS evaluation suite.
Evaluates the RAG pipeline on faithfulness, answer relevancy,
context recall, and context precision.
Logs all results to MLflow and saves a detailed CSV report.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from loguru import logger
import yaml


def load_eval_set(path: str) -> List[Dict[str, Any]]:
    """Load evaluation QA set from JSONL.
    Expected keys: question, ground_truth, answer, contexts (list of str).
    """
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} eval samples from {path}")
    return records


def run_ragas_evaluation(
    eval_records: List[Dict[str, Any]],
    experiment_name: str,
    run_name: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Run RAGAS evaluation and return a DataFrame of per-sample scores.
    Also logs aggregate metrics to MLflow.
    """
    dataset = Dataset.from_list(eval_records)

    logger.info("Running RAGAS evaluation...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores_df = results.to_pandas()
    logger.info(f"\nAggregate RAGAS Scores:\n{results}")

    # Save detailed CSV
    output_path = Path(output_dir) / "ragas_scores.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path, index=False)
    logger.info(f"Per-sample scores saved to {output_path}")

    # Log to MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics({
            "ragas_faithfulness":        results["faithfulness"],
            "ragas_answer_relevancy":    results["answer_relevancy"],
            "ragas_context_recall":      results["context_recall"],
            "ragas_context_precision":   results["context_precision"],
        })
        mlflow.log_artifact(str(output_path))
        logger.info("RAGAS metrics logged to MLflow")

    return scores_df


def compare_systems(
    baseline_records: List[Dict[str, Any]],
    system_records: List[Dict[str, Any]],
    output_dir: str,
) -> pd.DataFrame:
    """
    Compare two systems (e.g. GPT-4 baseline vs fine-tuned RAG).
    Returns a comparison DataFrame.
    """
    logger.info("Evaluating baseline system...")
    baseline_ds = Dataset.from_list(baseline_records)
    baseline_results = evaluate(
        baseline_ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    logger.info("Evaluating fine-tuned RAG system...")
    system_ds = Dataset.from_list(system_records)
    system_results = evaluate(
        system_ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    comparison = pd.DataFrame({
        "metric": metrics,
        "baseline": [baseline_results[m] for m in metrics],
        "rag_system": [system_results[m] for m in metrics],
    })
    comparison["improvement_%"] = (
        (comparison["rag_system"] - comparison["baseline"]) / comparison["baseline"] * 100
    ).round(1)

    output_path = Path(output_dir) / "system_comparison.csv"
    comparison.to_csv(output_path, index=False)

    logger.info(f"\nSystem Comparison:\n{comparison.to_string(index=False)}")
    logger.info(f"Comparison saved to {output_path}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--eval-file", default="data/processed/eval_results.jsonl",
                        help="JSONL with question/ground_truth/answer/contexts")
    parser.add_argument("--output-dir", default="results/ragas")
    parser.add_argument("--run-name", default="ragas-eval-v1")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    eval_records = load_eval_set(args.eval_file)
    scores_df = run_ragas_evaluation(
        eval_records,
        experiment_name=config["mlflow"]["experiment_name"],
        run_name=args.run_name,
        output_dir=args.output_dir,
    )

    print("\n=== Sample Scores (first 5) ===")
    print(scores_df[["question", "faithfulness", "answer_relevancy", "context_recall"]].head())
