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


def execute(args: argparse.Namespace, unknown_args: list) -> None:
    extra_args = {
        key.lstrip("-"): value
        for key, value in zip(unknown_args[::2], unknown_args[1::2])
    }
    evaluate(args.search_method, args.limit, args.llm, **extra_args)
