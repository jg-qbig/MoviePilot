import math
import os
import pickle
from string import punctuation
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from src.lib.utils import (
    SEARCH_LIMIT,
    CACHE_PATH,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords,
    format_results,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap: dict[int, dict] = {}
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_counts: dict[int, Counter] = defaultdict(Counter)
        self.term_counts_path = os.path.join(CACHE_PATH, "term_counts.pkl")
        self.doc_lengths: dict[int, int] = {}
        self.doc_length_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.__tokenize(text)
        for tok in set(tokens):
            self.index[tok].add(doc_id)
        self.term_counts[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if self.doc_lengths:
            return sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return 0.0

    def __tokenize(self, term: str | list[str]) -> list[str]:
        if isinstance(term, str):
            tokens = tokenize(term)
        else:
            tokens = term
        return tokens

    def build(self):
        data = load_movies()
        for movie in data:
            content = movie["title"] + "\n" + movie["description"]
            self.__add_document(movie["id"], content)
            self.docmap[movie["id"]] = movie

    def get_documents(self, term: str | list[str]) -> list[int]:
        tokens = self.__tokenize(term)
        is_single_token(tokens)
        token = tokens[0]
        return sorted(list(self.index[token]))

    def get_tf(self, doc_id: int, term: str | list[str]) -> int:
        tokens = self.__tokenize(term)
        is_single_token(tokens)
        token = tokens[0]
        return self.term_counts[doc_id][token]

    def get_idf(self, term: str | list[str]) -> float:
        n_total = len(self.docmap)
        n_matches = len(self.get_documents(term))
        return math.log((n_total + 1) / (n_matches + 1))

    def tfidf(self, doc_id: int, term: str | list[str]) -> float:
        tokens = self.__tokenize(term)
        tf = self.get_tf(doc_id, tokens)
        idf = self.get_idf(tokens)
        return tf * idf

    def get_bm25_tf(
        self, doc_id: int, term: str | list[str], k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_bm25_idf(self, term: str | list[str]) -> float:
        n_total = len(self.docmap)
        n_matches = len(self.get_documents(term))
        return math.log((n_total - n_matches + 0.5) / (n_matches + 0.5) + 1)

    def bm25(self, doc_id: int, term: str | list[str]) -> float:
        tokens = self.__tokenize(term)
        return self.get_bm25_tf(doc_id, tokens) * self.get_bm25_idf(tokens)

    def search(self, query: str, limit: int = SEARCH_LIMIT, bm25: bool = False) -> list:
        tokens = tokenize(query)

        scores = {}
        for doc_id in self.docmap:
            total_score = 0
            for tok in tokens:
                if bm25:
                    total_score += self.bm25(doc_id, [tok])
                else:
                    total_score += self.tfidf(doc_id, [tok])
            scores[doc_id] = total_score
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_n, score in scores[:limit]:
            doc = self.docmap[doc_n]
            results.append(
                format_results(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                )
            )
        return results

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_counts_path, "wb") as f:
            pickle.dump(self.term_counts, f)
        with open(self.doc_length_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_counts_path, "rb") as f:
            self.term_counts = pickle.load(f)
        with open(self.doc_length_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def tokenize(text: str) -> list[str]:
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    word_stems = []
    for word in text.split():
        if word not in stopwords:
            word_stems.append(stemmer.stem(word))
    return word_stems


def is_single_token(tokens: list[str]) -> None:
    if len(tokens) != 1:
        raise ValueError(f"Term must be a single word, got: {tokens}")


def match_tokens(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()
    seen, result = set(), []
    processed_query = tokenize(query)
    for token in processed_query:
        matching_doc_ids = index.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            result.append(index.docmap[doc_id])
            if len(result) >= limit:
                return result
    return result
