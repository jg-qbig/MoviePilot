import os

from src.lib.keyword_search import InvertedIndex
from src.lib.semantic_search import ChunkedSemanticSearch
from src.lib.utils import (
    HYBRID_ALPHA,
    RRF_K,
    SEARCH_LIMIT,
    format_results,
)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_index = ChunkedSemanticSearch()
        self.semantic_index.load_or_create_chunk_embeddings(documents)

        self.inverted_index = InvertedIndex()
        if not os.path.exists(self.inverted_index.index_path):
            self.inverted_index.build()
            self.inverted_index.save()

    def __bm25_search(self, query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
        self.inverted_index.load()
        return self.inverted_index.search(query, limit, bm25=True)

    def weighted_search(
        self, query: str, alpha: float = HYBRID_ALPHA, limit: int = SEARCH_LIMIT
    ) -> list[dict]:
        results_bm25 = self.__bm25_search(query, limit * 500)
        results_semantic = self.semantic_index.search_chunks(query, limit * 500)
        norm_bm25 = min_max_normalize([r["score"] for r in results_bm25])
        norm_semantic = min_max_normalize([r["score"] for r in results_semantic])
        for result, score in zip(results_bm25, norm_bm25):
            result["norm_score"] = score
        for result, score in zip(results_semantic, norm_semantic):
            result["norm_score"] = score

        combined_results = {}
        for result in results_bm25:
            doc_id = result["id"]
            combined_results[doc_id] = {
                "title": result["title"],
                "description": result["document"],
                "bm25_score": result["norm_score"],
                "semantic_score": 0.0,
            }

        for result in results_semantic:
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "title": result["title"],
                    "description": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": result["norm_score"],
                }
            elif result["norm_score"] > combined_results[doc_id]["semantic_score"]:
                combined_results[doc_id]["semantic_score"] = result["norm_score"]

        for doc_id, doc in combined_results.items():
            final_score = weighted_score(
                doc["bm25_score"], doc["semantic_score"], alpha
            )
            doc["weighted_score"] = final_score

        scores = sorted(
            combined_results.items(), key=lambda x: x[1]["weighted_score"], reverse=True
        )

        results = []
        for doc_id, doc in scores[:limit]:
            results.append(
                format_results(
                    doc_id=doc_id,
                    title=doc["title"],
                    document=doc["description"],
                    score=doc["weighted_score"],
                )
            )
        return results

    def rrf_search(
        self, query: str, k: int = RRF_K, limit: int = SEARCH_LIMIT
    ) -> list[dict]:
        results_bm25 = self.__bm25_search(query, (limit * 500))
        results_semantic = self.semantic_index.search_chunks(query, (limit * 500))

        combined_results = {}
        for rank, result in enumerate(results_bm25, 1):
            doc_id = result["id"]
            combined_results[doc_id] = {
                "title": result["title"],
                "description": result["document"],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, result in enumerate(results_semantic, 1):
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "title": result["title"],
                    "description": result["document"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }
            elif combined_results[doc_id]["semantic_rank"] is None:
                combined_results[doc_id]["semantic_rank"] = rank
                combined_results[doc_id]["rrf_score"] += rrf_score(rank, k)

        scores = sorted(
            combined_results.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )

        results = []
        for doc_id, doc in scores[:limit]:
            results.append(
                format_results(
                    doc_id=doc_id,
                    title=doc["title"],
                    document=doc["description"],
                    score=doc["rrf_score"],
                    bm25_rank=doc["bm25_rank"],
                    semantic_rank=doc["semantic_rank"],
                )
            )
        return results


def min_max_normalize(array: list) -> list:
    if not array:
        return []

    min_score = min(array)
    max_score = max(array)

    if min_score == max_score:
        return [1.0] * len(array)

    norm = max_score - min_score
    result = [(score - min_score) / norm for score in array]

    return result


def weighted_score(
    bm25_score: float, semantic_score: float, alpha: float = HYBRID_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1 / (k + rank)
