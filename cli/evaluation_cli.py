import argparse
import json
import os

from lib.utils import PROJECT_ROOT
from hybrid_search_cli import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    print(f"k={limit}")

    with open(
        os.path.join(PROJECT_ROOT, "data", "golden_dataset.json"), "r", encoding="utf8"
    ) as f:
        test_data = json.load(f)

    for test_case in test_data["test_cases"]:
        results = rrf_search_command(test_case["query"], k=60, limit=limit)
        print(len(results))
        result_titles = [r["title"] for r in results]

        precision_k = precision(result_titles, test_case["relevant_docs"])
        recall_k = recall(result_titles, test_case["relevant_docs"])
        f1 = 2 * (precision_k * recall_k) / (precision_k + recall_k)

        print(
            f"- Query: {test_case["query"]}\n\t- Precision@{limit}: {precision_k:.4f}\n\t- Recall@{limit}: {recall_k:.4f}\n\t- F1 Score: {f1:.4f}\n\t- Retrieved: {result_titles}\n\t- Relevant: {test_case["relevant_docs"]}"
        )


def precision(results: list[str], targets: list[str]):
    relevant_retrieved = sum(r in targets for r in results)
    return relevant_retrieved / len(results)


def recall(results: list[str], targets: list[str]):
    relevant_retrieved = sum(r in targets for r in results)
    return relevant_retrieved / len(targets)


if __name__ == "__main__":
    main()
