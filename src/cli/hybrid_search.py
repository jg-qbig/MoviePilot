import argparse

from src.lib.utils import (
    HYBRID_ALPHA,
    RRF_K,
    SEARCH_LIMIT,
    SEARCH_LIMIT_MULTIPLIER,
    load_movies,
    print_results,
)
from src.lib.hybrid_search import min_max_normalize, HybridSearch
from src.lib.query_enhancement import enhance_query
from src.lib.result_reranking import rerank_results


def main(args_list=None) -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ### Hybrid search using reciprocal rank fusion to combine keyword and semantic search results
    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="Search using reciprocal rank fusion to combine keyword and semantic search results.",
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query.")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=RRF_K,
        help="Parameter to control influence of high and low ranked results.",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to show.",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="LLM based query enhancement method.",
    )
    rrf_search_parser.add_argument(
        "--rerank",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM based method to rerank most relevant search results.",
    )
    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", help="Use an LLM to evaluate search results."
    )

    ### Hybrid search using min/max normalization and weighting to combine keyword and semantic search results
    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Search using weighted results from keyword and semantic search.",
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=HYBRID_ALPHA,
        help="Alpha parameter to control weighting towards keyword or semantic search",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to show.",
    )

    ### Min/max normalization
    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a given array of values using Min/Max normalization.",
    )
    normalize_parser.add_argument(
        "array", nargs="+", type=float, help="Array to normalize."
    )

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    match args.command:
        case "rrf-search":
            rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank
            )
        case "weighted-search":
            data = load_movies()
            index = HybridSearch(data)
            results = index.weighted_search(args.query, args.alpha, args.limit)
            print_results(results, score_label="Weighted Score")
        case "normalize":
            print(min_max_normalize(args.array))
        case _:
            parser.print_help()


def rrf_search_command(
    query: str,
    k: int = RRF_K,
    limit: int = SEARCH_LIMIT,
    enhance: str = "",
    rerank: str = "",
):
    enhanced_query = enhance_query(query, method=enhance)

    data = load_movies()
    index = HybridSearch(data)

    if rerank:
        results = index.rrf_search(enhanced_query, k, limit * SEARCH_LIMIT_MULTIPLIER)
    else:
        results = index.rrf_search(enhanced_query, k, limit)

    results = rerank_results(query, results, method=rerank, limit=limit)

    print_results(results, score_label="RRF Score")
    return results


if __name__ == "__main__":
    main()
