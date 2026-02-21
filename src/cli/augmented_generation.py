import argparse

from src.lib.utils import SEARCH_LIMIT
from src.cli.hybrid_search import rrf_search_command
from src.lib.augmented_generation import (
    generate,
    summarize,
    summarize_with_citations,
    question_answering,
)


def setup_subparser(subparser: argparse._SubParsersAction) -> None:
    rag_parser = subparser.add_parser("rag", help="Retrieval Augmented Generation CLI")
    rag_subparser = rag_parser.add_subparsers(
        dest="command", help="Available commands", required=False
    )

    ### QA
    qa_parser = rag_subparser.add_parser(
        "qa", help="Answer user question based on retrieved search results."
    )
    qa_parser.add_argument("question", type=str, help="Question to be answered.")
    qa_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to be retrieved.",
    )

    ### Summarize with citations
    citation_parser = rag_subparser.add_parser(
        "cite", help="Generate LLM summary with citations"
    )
    citation_parser.add_argument("query", type=str, help="Search query.")
    citation_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to be retrieved.",
    )

    ### Summarization
    summarize_parser = rag_subparser.add_parser(
        "summarize", help="Generate an LLM summary of the retrieved search results."
    )
    summarize_parser.add_argument("query", type=str, help="Search query.")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to be retrieved",
    )

    ### Generate based on search results
    gen_parser = rag_subparser.add_parser(
        "generate", help="Generate LLM response based on retrieved search results."
    )
    gen_parser.add_argument("query", type=str, help="Search query.")

    rag_parser.set_defaults(func=execute, subparser=rag_parser)


def execute(args: argparse.Namespace) -> None:
    match args.command:
        case "qa":
            results = rrf_search_command(args.question, limit=args.limit)
            response = question_answering(args.question, results)
            print(f"LLM Response:\n{response}")
        case "cite":
            results = rrf_search_command(args.query, limit=args.limit)
            response = summarize_with_citations(args.query, results)
            print(f"LLM Response:\n{response}")
        case "summarize":
            results = rrf_search_command(args.query, limit=args.limit)
            response = summarize(args.query, results)
            print(f"LLM Response:\n{response}")
        case "generate":
            results = rrf_search_command(args.query)
            response = generate(args.query, results)
            print(f"LLM Response:\n{response}")
        case _:
            args.subparser.print_help()
