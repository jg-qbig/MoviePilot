import argparse

from src.lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    chunk_semantic,
    fixed_size_chunking,
)
from src.lib.utils import (
    CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    MAX_SEMANTIC_CHUNK_SIZE,
    SEARCH_LIMIT,
    load_movies,
    print_results,
)


def setup_subparser(subparser: argparse._SubParsersAction) -> None:
    semantic_parser = subparser.add_parser("semantic", help="Semantic Search CLI")
    semantic_subparser = semantic_parser.add_subparsers(
        dest="command", help="Available Commands"
    )

    ### Semantic search based on semantically chunked embeddings
    search_chunked_parser = semantic_subparser.add_parser(
        "search_chunked",
        help="Semantic search based on semantically chunked embeddings.",
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to show.",
    )

    ### Build semantically chunked embeddings
    semantic_subparser.add_parser(
        "build_chunk_embeddings",
        help="Generate semantically chunked embeddings and store them on disk.",
    )

    ### Create semantic chunks
    chunk_semantic_parser = semantic_subparser.add_parser(
        "chunk_semantic", help="Create semantic chunks from a single input text."
    )
    chunk_semantic_parser.add_argument("text", type=str, help="Text to chunk.")
    chunk_semantic_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=MAX_SEMANTIC_CHUNK_SIZE,
        help="Max size of each semantic chunk in number of sentences.",
    )
    chunk_semantic_parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Overlap between semantic chunks in number of sentences.",
    )

    ### Create simple chunks
    chunk_parser = semantic_subparser.add_parser(
        "chunk", help="Split text into simple chunks."
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk.")
    chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=MAX_CHUNK_SIZE,
        help="Max size of each chunk in number of words",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Overlap between chunks in number of words.",
    )

    ### Semantic search without chunks
    search_parser = semantic_subparser.add_parser(
        "search", help="Search movies using semantic search and single embeddings."
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to show."
    )

    ### Embed movie descriptions
    semantic_subparser.add_parser(
        "build_embeddings",
        help="Generate single embeddings from movie descriptions and store them on disk.",
    )

    ### Verify embeddings
    semantic_subparser.add_parser(
        "verify_embeddings", help="Verify simple embeddings for the movie dataset."
    )

    ### Embed search query
    embed_query_parser = semantic_subparser.add_parser(
        "embed", help="Generate embedding for a single text input."
    )
    embed_query_parser.add_argument("text", type=str, help="Text to embed.")

    ### Verify embedding model
    semantic_subparser.add_parser(
        "verify_model", help="Check if embedding model is loaded correctly."
    )

    semantic_parser.set_defaults(func=execute, subparser=semantic_parser)


def execute(args: argparse.Namespace) -> None:
    match args.command:
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case "build_chunk_embeddings":
            build_chunk_embeddings_command()
        case "chunk_semantic":
            print(f"Semantically chunking {len(args.text)} characters.")
            chunks = chunk_semantic(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "chunk":
            print(f"Chunking {len(args.text)} characters.")
            chunks = fixed_size_chunking(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "search":
            search_command(args.query, args.limit)
        case "build_embeddings":
            build_embeddings_command()
        case "verify_embeddings":
            verify_embeddings_command()
        case "embed":
            embed_command(args.text)
        case "verify_model":
            verify_model_command()
        case _:
            args.subparser.print_help()


def search_chunked_command(query: str, limit: int = SEARCH_LIMIT):
    index = ChunkedSemanticSearch()
    data = load_movies()
    index.load_or_create_chunk_embeddings(data)
    results = index.search_chunks(query, limit)
    print_results(results, score_label="Cosine Similarity")
    return results


def build_chunk_embeddings_command():
    index = ChunkedSemanticSearch()
    data = load_movies()
    embeddings = index.build_chunk_embeddings(data)
    print(f"Generated {len(embeddings)} semantically chunked embeddings")


def search_command(query, limit=SEARCH_LIMIT):
    index = SemanticSearch()
    data = load_movies()
    index.load_or_create_embeddings(data)
    print_results(index.search(query, limit), score_label="Cosine Similarity")


def build_embeddings_command():
    index = SemanticSearch()
    data = load_movies()
    embeddings = index.build_embeddings(data)
    print(f"Generated {len(embeddings)} simple embeddings")


def verify_embeddings_command():
    index = SemanticSearch()
    data = load_movies()
    embeddings = index.load_or_create_embeddings(data)
    print(f"Number of docs:   {len(data)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_command(text):
    index = SemanticSearch()
    embedding = index.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Vector Dimensions: {embedding.shape[0]}")


def verify_model_command():
    index = SemanticSearch()
    print(f"Model loaded: {index.model}")
    print(f"Max sequence length: {index.model.max_seq_length}")
