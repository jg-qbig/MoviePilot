import argparse

from src.lib.utils import load_movies, print_results
from src.cli.hybrid_search import rrf_search_command
from src.lib.multimodal_search import (
    MultimodalSearch,
    multimodal_prompt_gemini,
    verify_image_embedding,
)


def main(args_list=None):
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ### Image search
    search_parser = subparsers.add_parser("search", help="Search movie based on image.")
    search_parser.add_argument("--path", type=str, help="Path to image.")

    ### Image description
    describe_parser = subparsers.add_parser(
        "augment", help="Let model describe image as text."
    )
    describe_parser.add_argument("--query", type=str, help="Search query.")
    describe_parser.add_argument("--path", type=str, help="Path to image.")

    ### Verify embeddings
    verify_parser = subparsers.add_parser(
        "verify", help="Check if image embeddings are valid."
    )
    verify_parser.add_argument("--path", type=str, help="Path to image.")

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    match args.command:
        case "search":
            data = load_movies()
            index = MultimodalSearch(data)
            results = index.search(args.path)
            print_results(results, score_label="Cosine Similarity")
        case "augment":
            query = multimodal_prompt_gemini(args.query, args.path)
            print(f"Augmented Prompt: {query}")
            rrf_search_command(query)
        case "verify":
            verify_image_embedding(args.path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
