import argparse

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

    keyword_search.setup_subparser(subparsers)
    semantic_search.setup_subparser(subparsers)
    hybrid_search.setup_subparser(subparsers)
    augmented_generation.setup_subparser(subparsers)
    multimodal_search.setup_subparser(subparsers)
    evaluation.setup_subparser(subparsers)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
