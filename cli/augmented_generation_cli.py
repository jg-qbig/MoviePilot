import argparse

from hybrid_search_cli import rrf_search_command
from lib.augmented_generation import (
    augment_prompt,
    summarize,
    summarize_with_citations,
    question_answering,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate an LLM summary of the search results."
    )
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to summarize"
    )

    citation_parser = subparsers.add_parser(
        "citations", help="LLM summary with citations"
    )
    citation_parser.add_argument("query", type=str, help="Search query")
    citation_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to summarize"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer questions for retrieved documents"
    )
    question_parser.add_argument("question", type=str, help="Question to be answered")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to be retrieved"
    )

    args = parser.parse_args()

    match args.command:
        case "question":
            results = rrf_search_command(args.question, limit=args.limit)
            formatted_results = [f"{r["title"]}\n{r["document"]}\n" for r in results]
            response = question_answering(args.question, formatted_results)
            titles = "".join(f"\t- {r["title"]}\n" for i, r in enumerate(results))
            print(f"Search Results:\n{titles}LLM Answer:\n{response}")
        case "citations":
            results = rrf_search_command(args.query, limit=args.limit)
            formatted_results = [f"{r["title"]}\n{r["document"]}\n" for r in results]
            response = summarize_with_citations(args.query, formatted_results)
            titles = "".join(f"\t- [{i}] {r["title"]}\n" for i, r in enumerate(results))
            print(f"Search Results:\n{titles}LLM Summary:\n{response}")
        case "summarize":
            results = rrf_search_command(args.query, limit=args.limit)
            formatted_results = [f"{r["title"]}\n{r["document"]}\n" for r in results]
            response = summarize(args.query, formatted_results)
            titles = "".join(f"\t- {r["title"]}\n" for r in results)
            print(f"Search Results:\n{titles}LLM Summary:\n{response}")
        case "rag":
            results = rrf_search_command(args.query)
            response = augment_prompt(args.query, results)
            titles = "".join(f"\t- {r["title"]}\n" for r in results)
            print(f"Search Results:\n{titles}RAG Response:\n{response}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
