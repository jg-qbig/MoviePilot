import argparse

from src.lib.utils import SEARCH_LIMIT, HYBRID_ALPHA, RRF_K, load_movies, print_results
from src.cli.hybrid_search import rrf_search_command
from src.lib.multimodal_search import (
    MultimodalSearch,
    multimodal_prompt_gemini,
)


def setup_subparser(subparser: argparse._SubParsersAction) -> None:
    multimodal_parser = subparser.add_parser(
        "multimodal", help="Multimodal (Image + Text) Search CLI"
    )
    multimodal_subparser = multimodal_parser.add_subparsers(
        dest="command", help="Available commands"
    )

    ### Image search
    search_parser = multimodal_subparser.add_parser(
        "search", help="Search movie based on query + image embeddings."
    )
    search_parser.add_argument("query", type=str, help="Search query.")
    search_parser.add_argument("path", type=str, help="Path to image.")
    search_parser.add_argument(
        "--alpha",
        type=float,
        default=HYBRID_ALPHA,
        help="Alpha parameter to control weighting towards text or image search.",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to show.",
    )

    ### Image search
    img_search_parser = multimodal_subparser.add_parser(
        "image-search", help="Search movie based on image."
    )
    img_search_parser.add_argument("path", type=str, help="Path to image.")
    img_search_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to show.",
    )

    ### Image description
    describe_parser = multimodal_subparser.add_parser(
        "augment", help="Let model describe image as text."
    )
    describe_parser.add_argument("query", type=str, help="Search query.")
    describe_parser.add_argument("path", type=str, help="Path to image.")
    describe_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to show.",
    )

    ### Build dataset embeddings
    multimodal_subparser.add_parser(
        "build",
        help="Embed movie data in multimodal vector space and store embeddings on disk.",
    )

    ### Verify embeddings
    verify_img_parser = multimodal_subparser.add_parser(
        "verify-image", help="Check if image embeddings are valid."
    )
    verify_img_parser.add_argument("path", type=str, help="Path to image.")
    multimodal_subparser.add_parser(
        "verify-text", help="Check if stored text embeddings are valid."
    )

    multimodal_parser.set_defaults(func=execute, subparser=multimodal_parser)


def execute(args: argparse.Namespace, unknown_args: list) -> None:
    extra_args = {
        key.lstrip("-"): value
        for key, value in zip(unknown_args[::2], unknown_args[1::2])
    }
    k = int(extra_args.get("k", RRF_K))
    enhance = str(extra_args.get("enhance", ""))
    rerank = str(extra_args.get("rerank", ""))

    if args.command == "augment":
        query = multimodal_prompt_gemini(args.query, args.path)
        print(f"Augmented Prompt: {query}")
        results = rrf_search_command(
            query, limit=args.limit, k=k, enhance=enhance, rerank=rerank
        )
        return

    data = load_movies()
    index = MultimodalSearch()

    match args.command:
        case "search":
            index.load_or_create_embeddings(data)
            results = index.search_multi(
                args.query, args.path, limit=args.limit, alpha=args.alpha
            )
            print_results(results, score_label="Dot Product Similarity")
        case "image-search":
            index.load_or_create_embeddings(data)
            results = index.search(args.path, limit=args.limit)
            print_results(results, score_label="Cosine Similarity")
        case "build":
            index.build_embeddings(data)
        case "verify-image":
            index.load_or_create_embeddings(data)
            embedding = index.generate_embedding(args.path)
            print(f"Embedding shape: {embedding.shape[0]} dimensions")
        case "verify-text":
            embeddings = index.load_or_create_embeddings(data)
            print(f"Number of docs:   {len(data)}")
            print(
                f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
            )
        case _:
            args.subparser.print_help()
