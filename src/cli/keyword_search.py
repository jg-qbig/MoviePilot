import argparse

from src.lib.utils import SEARCH_LIMIT, BM25_K1, BM25_B, print_results
from src.lib.keyword_search import match_tokens, InvertedIndex


def setup_subparser(subparsers: argparse._SubParsersAction) -> None:
    keyword_parser = subparsers.add_parser("keyword", help="Keyword Search CLI")
    keyword_subparsers = keyword_parser.add_subparsers(
        dest="command", help="Available Commands", required=False
    )

    ### BM25 Keyword search
    bm25search_parser = keyword_subparsers.add_parser(
        "bm25search",
        help="Return the top n {--limit} matches for a query {query} according to Okapi BM25 search algorithm.",
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to show",
    )

    ### BM25 inverse document frequency
    bm25idf_parser = keyword_subparsers.add_parser(
        "bm25idf",
        help="Return the IDF score of a word {term} using Okapi BM25 search algorithm.",
    )
    bm25idf_parser.add_argument(
        "term", type=str, help="Search term (must be a single word)"
    )

    ### BM25 term frequency
    bm25tf_parser = keyword_subparsers.add_parser(
        "bm25tf",
        help="Return the TF score of a word {term} using Okapi BM25 search algorithm.",
    )
    bm25tf_parser.add_argument(
        "doc_id", type=int, help="Document ID of the document to search"
    )
    bm25tf_parser.add_argument(
        "term", type=str, help="Search term (must be a single word)"
    )
    bm25tf_parser.add_argument(
        "--k1",
        type=float,
        default=BM25_K1,
        help="Parameter to control frequency saturation. Uses diminishing returns for single words appearing multiple times.",
    )
    bm25tf_parser.add_argument(
        "--b",
        type=float,
        default=BM25_B,
        help="Parameter to control document length normalization (0 <= b <= 1). Helps to account for longer documents having higher term-frequency score due to simply containing more words.",
    )

    ### TF_IDF search
    tfidf_search_parser = keyword_subparsers.add_parser(
        "search",
        help="Return the top n {--limit} matches for a query {query} based on TF-IDF scores.",
    )
    tfidf_search_parser.add_argument("query", type=str, help="Search query")
    tfidf_search_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of search results to show",
    )

    ### Term frequency - inverse document frequency score
    tfidf_parser = keyword_subparsers.add_parser(
        "tfidf",
        help="Return the TF-IDF score of a word {term}.",
    )
    tfidf_parser.add_argument(
        "doc_id", type=int, help="Document ID of the document to search"
    )
    tfidf_parser.add_argument(
        "term", type=str, help="Search term (must be a single word)"
    )

    ### Inverse document frequency
    idf_parser = keyword_subparsers.add_parser(
        "idf",
        help="Return the inverse document frequency of a word {term}.",
    )
    idf_parser.add_argument(
        "term", type=str, help="Search term (must be a single word)"
    )

    ### Term frequency
    tf_parser = keyword_subparsers.add_parser(
        "tf",
        help="Return the number of occurrences of a word {term} in a document {doc_id}",
    )
    tf_parser.add_argument(
        "doc_id", type=int, help="Document ID of the document to search"
    )
    tf_parser.add_argument("term", type=str, help="Search term (must be a single word)")

    ### Build inverted search index
    keyword_subparsers.add_parser(
        "build", help="Build the inverted search index and save it to disk"
    )

    ### Search with keyword matching
    match_parser = keyword_subparsers.add_parser(
        "match", help="Search movies using keyword matching in title and description"
    )
    match_parser.add_argument("query", type=str, help="Search query")

    keyword_parser.set_defaults(func=execute, subparser=keyword_parser)


def execute(args: argparse.Namespace) -> None:
    match args.command:
        case "bm25search":
            index = InvertedIndex()
            index.load()
            print_results(
                index.search(args.query, args.limit, bm25=True),
                score_label="BM25 Score",
            )
        case "bm25idf":
            index = InvertedIndex()
            index.load()
            bm25idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            index = InvertedIndex()
            index.load()
            bm25tf = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "search":
            index = InvertedIndex()
            index.load()
            print_results(
                index.search(args.query, args.limit, bm25=False),
                score_label="BM25 Score",
            )
        case "tfidf":
            index = InvertedIndex()
            index.load()
            tfidf = index.tfidf(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}"
            )
        case "idf":
            index = InvertedIndex()
            index.load()
            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(
                f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf:.2f}"
            )
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "match":
            results = match_tokens(args.query)
            for result in results:
                print(f"({result["id"]}) {result["title"]}")
        case _:
            args.subparser.print_help()
