import os
import json
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

SEARCH_LIMIT = 5
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75

MAX_CHUNK_SIZE = 100
MAX_SEMANTIC_CHUNK_SIZE = 4
CHUNK_OVERLAP = 1

HYBRID_ALPHA = 0.5
SEARCH_LIMIT_MULTIPLIER = 5
RRF_K = 60


def load_movies() -> list[dict]:
    with open(
        MOVIES_PATH,
        "r",
        encoding="utf8",
    ) as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r", encoding="utf8") as f:
        stopwords = f.read()
    return stopwords.splitlines()


def format_results(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


def print_results(results: list[dict], score_label: str = "Score"):
    for result in results:
        print(f"({result["id"]}) {result["title"]} - {score_label}: {result["score"]}")
