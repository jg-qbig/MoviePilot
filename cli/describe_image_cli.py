import argparse

from lib.image_search import describe_image


def main():
    parser = argparse.ArgumentParser(description="Multimodal search")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--query", type=str, help="Search query")

    args = parser.parse_args()

    response = describe_image(args.query, args.image)
    print(f"Rewritten query: {response}")


if __name__ == "__main__":
    main()
