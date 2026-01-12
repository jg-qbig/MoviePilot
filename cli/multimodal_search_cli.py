import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Check if image embeddings are valid"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to image to embed")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movie based on image"
    )
    image_search_parser.add_argument("image_path", type=str, help="Path to image")

    args = parser.parse_args()

    match args.command:
        case "image_search":
            results = image_search_command(args.image_path)
            for i, r in enumerate(results):
                print(
                    f"{i}. {r["title"]} (similarity: {r["similarity"]:.3f})\n\t{r["description"][:100]}"
                )
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
