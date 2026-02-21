import argparse
import sys

from src.cli import (
    keyword_search,
    semantic_search,
    hybrid_search,
    augmented_generation,
    multimodal_search,
    evaluation,
)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Movie Search Engine CLI",
    )

    subparsers = parser.add_subparsers(dest="tool", help="Available tools")

    subparsers.add_parser("keyword", help="Perform keyword-based search (BM25)")
    subparsers.add_parser("semantic", help="Perform vector-based semantic search")
    subparsers.add_parser(
        "hybrid", help="Return combined keyword- and vector-based search results"
    )
    subparsers.add_parser(
        "multimodal", help="Search based on a multimodal query (image + text)"
    )
    subparsers.add_parser(
        "rag", help="Generate LLM response informed by search results"
    )
    subparsers.add_parser("evaluate", help="Simple evaluation of search results")

    # Only parse first argument, rest is passed on to sub-script
    tool_name = sys.argv[1:2]
    sub_args = sys.argv[2:]

    args = parser.parse_args(tool_name)

    # Map tool to sub-script
    tool_map = {
        "keyword": keyword_search.main,
        "semantic": semantic_search.main,
        "hybrid": hybrid_search.main,
        "rag": augmented_generation.main,
        "multimodal": multimodal_search.main,
        "evaluate": evaluation.main,
    }

    if args.tool in tool_map:
        # Call sub-script main function
        tool_map[args.tool](sub_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
