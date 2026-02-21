import re
import json

from src.lib.utils import (
    MAX_DESCRIPTION,
    SEARCH_LIMIT,
    load_test_cases,
    prompt_gemini,
    print_results,
)
from src.lib.keyword_search import InvertedIndex
from src.cli.semantic_search import search_chunked_command
from src.cli.hybrid_search import rrf_search_command


def evaluate(search_method: str, limit: int = SEARCH_LIMIT, use_llm: bool = False):
    test_data = load_test_cases()

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for test in test_data:
        print(f"{"-" * 50}")
        print("### Search Results ###")
        if search_method == "keyword":
            index = InvertedIndex()
            index.load()
            results = index.search(test["query"], limit=limit, bm25=True)
            print_results(results, score_label="BM25 Score")
        elif search_method == "semantic":
            results = search_chunked_command(test["query"], limit=limit)
        elif search_method == "hybrid":
            results = rrf_search_command(test["query"], limit=limit)
        else:
            raise ValueError("No valid search method.")

        if use_llm:
            print("\n### LLM Evaluation ###")
            llm_eval(test["query"], results)

        result_titles = [r["title"] for r in results]

        precision_k = precision(result_titles, test["relevant_docs"])
        recall_k = recall(result_titles, test["relevant_docs"])
        f1 = (
            2 * (precision_k * recall_k) / (precision_k + recall_k)
            if (precision_k + recall_k)
            else 0.0
        )

        total_precision += precision_k
        total_recall += recall_k
        total_f1 += f1

        print("\n### Evaluation Metrics ###")
        print(
            f"Query: {test["query"]}\nRelevant: {test["relevant_docs"]}\nPrecision@{limit}: {precision_k:.3f}\nRecall@{limit}: {recall_k:.3f}\nF1@{limit}: {f1:.3f}"
        )
        print(f"{"-" * 50}")

    print(f"Average Precision@{limit}: {total_precision/len(test_data):.3f}")
    print(f"Average Recall@{limit}: {total_recall/len(test_data):.3f}")
    print(f"Average F1@{limit}: {total_f1/len(test_data):.3f}")
    print(f"{"-" * 50}")


def precision(results: list[str], targets: list[str]) -> float:
    relevant_retrieved = sum(r in targets for r in results)
    return relevant_retrieved / len(results)


def recall(results: list[str], targets: list[str]) -> float:
    relevant_retrieved = sum(r in targets for r in results)
    return relevant_retrieved / len(targets)


def llm_eval(query: str, results: list[dict]):
    results_str = "\n".join(
        [
            f"{i}. {res["title"]} - {res["document"][:MAX_DESCRIPTION]}"
            for i, res in enumerate(results)
        ]
    )

    prompt = f"""
    You are an experienced movie expert and are especially skilled at recommending movies to people based on a given query.
    For the movies given below, sate how relevant each movie is to the search query on a scale from 1 to 5 where 1 equals not relevant at all and 5 equals highly relevant:

    Query: "{query}"

    Movies: {results_str}

    Scale:
    - 5: Highly relevant
    - 4: Relevant
    - 3: Neutral
    - 2: Not relevant
    - 1: Not relevant at all

    Do NOT return any other numbers besides 1, 2, 3, 4 and 5.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]
    """

    response = prompt_gemini(prompt)
    response = re.search(r"\[[\d,\s]+\]", response)
    if response:
        scores = json.loads(response.group())
    else:
        return []

    for result, score in zip(results, scores):
        result["llm_rank"] = score

    results = sorted(results, key=lambda x: x["llm_rank"], reverse=True)

    for res in results:
        print(f"({res["id"]}) {res["title"]} - LLM Relevance: {res["llm_rank"]}")
