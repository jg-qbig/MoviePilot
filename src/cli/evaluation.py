import argparse

from src.lib.utils import SEARCH_LIMIT
from src.lib.evaluation import evaluate


def main(args_list=None) -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "search_method",
        type=str,
        choices=["keyword", "semantic", "hybrid"],
        help="Which search method to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use llm expert to evaluate search results.",
    )

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    evaluate(args.search_method, args.limit, args.llm)


if __name__ == "__main__":
    main()
