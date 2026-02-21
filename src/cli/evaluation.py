import argparse

from src.lib.utils import SEARCH_LIMIT
from src.lib.evaluation import evaluate


def setup_subparser(subparser: argparse._SubParsersAction) -> None:
    eval_parser = subparser.add_parser("evaluate", help="Search Evaluation CLI")
    eval_parser.add_argument(
        "search_method",
        type=str,
        choices=["keyword", "semantic", "hybrid"],
        help="Which search method to evaluate",
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    eval_parser.add_argument(
        "--llm",
        action="store_true",
        help="Use llm expert to evaluate search results.",
    )

    eval_parser.set_defaults(func=execute, subparser=eval_parser)


def execute(args: argparse.Namespace) -> None:
    evaluate(args.search_method, args.limit, args.llm)
