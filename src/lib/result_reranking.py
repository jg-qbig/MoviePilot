import re
import json
import time

from sentence_transformers import CrossEncoder

from src.lib.utils import MAX_DESCRIPTION, SEARCH_LIMIT, print_results, prompt_gemini


def rate_individual(query: str, results: list[dict]) -> list[dict]:
    for result in results:
        prompt = f"""
        Given a user search query and a movie result.
        Rate how well the returned movie matches the search query out of 10.
        Consider direct relevance to the query, the users intent and appropriateness.

        Query: "{query}"

        Movie: {result["title"]} - {result["document"][:MAX_DESCRIPTION]}

        Rate movies from 0 to 10 where 10 equals a perfect match.
        Follow this format: "Rating: {{rating}}/10"
        """

        response = prompt_gemini(prompt)

        response = re.search(r"Rating:\s*(.*?)/10", response)
        if response:
            score = int(response.group(1))
        else:
            score = 0

        result["rerank_score"] = score
        time.sleep(3)

    reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results


def rate_batch(query: str, results: list[dict]) -> list:
    docmap = {doc["id"]: doc for doc in results}

    prompt = f"""
    Given a search query and a list of matched movies with their descriptions, rank the movies by relevance to the search query.
    Consider direct relevance to the query, the users intent and appropriateness.

    Query: "{query}"

    Movies: {[f"{result["id"]}, {result["title"]}, {result["document"][:MAX_DESCRIPTION]}" for result in results]}

    Return ONLY the IDs in order of relevance, starting with the best match. Return a valid JSON list, nothing else. For example:

    [75, 12, 34, 2, 1]

    Your response should start with '[' and end with ']'. Provide no other information besides the JSON list.

    Ranking:
    """

    response = prompt_gemini(prompt)

    response = re.search(r"\[[\d,\s]+\]", response)
    if response:
        reranked_ids = json.loads(response.group())
    else:
        return []

    for rank, doc_id in enumerate(reranked_ids, 1):
        docmap[doc_id]["rank"] = rank

    reranked_results = sorted(docmap.values(), key=lambda x: x["rank"])

    return reranked_results


def rate_cross_encoder(query: str, results: list[dict]):
    pairs = [
        [
            query,
            f"{result["title"]} - {result["document"]}",
        ]
        for result in results
    ]

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    for result, score in zip(results, scores):
        result["cross_score"] = score

    results = sorted(results, key=lambda x: x["cross_score"], reverse=True)
    return results


def rerank_results(
    query: str,
    results: list[dict],
    method: str = "",
    limit: int = SEARCH_LIMIT,
) -> list[dict]:
    print("Reranking Results ...")
    print_results(results)

    if method == "individual":
        return rate_individual(query, results)[:limit]

    if method == "batch":
        return rate_batch(query, results)[:limit]

    if method == "cross_encoder":
        return rate_cross_encoder(query, results)[:limit]

    return results
