import argparse

from src.lib.multimodal_search import (
    describe_image,
    verify_image_embedding,
    image_search_command,
)


def main(args_list=None):
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Check if image embeddings are valid"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to image to embed")

    describe_parser = subparsers.add_parser(
        "describe_image", help="Let model describe image as text."
    )
    describe_parser.add_argument("--image", type=str, help="Path to image")
    describe_parser.add_argument("--query", type=str, help="Search query")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movie based on image"
    )
    image_search_parser.add_argument("image_path", type=str, help="Path to image")

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    match args.command:
        case "image_search":
            results = image_search_command(args.image_path)
            for i, r in enumerate(results):
                print(
                    f"{i}. {r["title"]} (similarity: {r["similarity"]:.3f})\n\t{r["description"][:100]}"
                )
        case "describe_image":
            print(f"Rewritten query: {describe_image(args.query, args.image)}")
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
