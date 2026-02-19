import json
import os
import re
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from src.lib.utils import (
    CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    MAX_SEMANTIC_CHUNK_SIZE,
    SEARCH_LIMIT,
    format_results,
    CACHE_PATH,
)

EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CHUNK_EMBEDDINGS_META_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")
HASH_PATH = os.path.join(CACHE_PATH, "dataset.hash")


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.documents = []

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Text is empty.")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        contents = []
        for doc in documents:
            contents.append(f"{doc["title"]}: {doc["description"]}")
        self.embeddings = self.model.encode(contents, show_progress_bar=True)

        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)

        curr_hash = get_titles_hash(self.documents)
        with open(HASH_PATH, "w", encoding="utf8") as f:
            f.write(curr_hash)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        curr_hash = get_titles_hash(documents)

        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(HASH_PATH):
            with open(HASH_PATH, "r", encoding="utf8") as f:
                stored_hash = f.read().strip()

            if stored_hash == curr_hash:
                self.embeddings = np.load(EMBEDDINGS_PATH)
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = SEARCH_LIMIT):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings or documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarities.append(
                (
                    cosine_similarity(query_embedding, embedding),
                    self.documents[i],
                )
            )

        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                format_results(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                )
            )
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunks = []
        chunks_meta = []
        for doc_idx, doc in enumerate(documents):
            # content = f"{doc['title']}. {doc['description']}"
            content = doc.get("description", "")
            if not content.strip():
                continue

            doc_chunks = chunk_semantic(
                content, max_chunk_size=MAX_SEMANTIC_CHUNK_SIZE, overlap=CHUNK_OVERLAP
            )
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                chunks_meta.append(
                    {
                        "doc_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(doc_chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_meta

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_EMBEDDINGS_META_PATH, "w", encoding="utf8") as f:
            json.dump({"chunks": chunks_meta, "total_chunks": len(chunks)}, f, indent=2)

        curr_hash = get_titles_hash(self.documents)
        with open(HASH_PATH, "w", encoding="utf8") as f:
            f.write(curr_hash)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        curr_hash = get_titles_hash(documents)

        if (
            os.path.exists(CHUNK_EMBEDDINGS_PATH)
            and os.path.exists(CHUNK_EMBEDDINGS_META_PATH)
            and os.path.exists(HASH_PATH)
        ):
            with open(HASH_PATH, "r", encoding="utf8") as f:
                stored_hash = f.read().strip()

            if stored_hash == curr_hash:
                self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
                with open(CHUNK_EMBEDDINGS_META_PATH, "r", encoding="utf8") as f:
                    data = json.load(f)
                    self.chunk_metadata = data["chunks"]
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = SEARCH_LIMIT):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "doc_idx": self.chunk_metadata[idx]["doc_idx"],
                    "chunk_idx": self.chunk_metadata[idx]["chunk_idx"],
                    "score": similarity,
                }
            )

        doc_scores = {}
        for chunk in chunk_scores:
            doc_idx = chunk["doc_idx"]
            if doc_idx not in doc_scores or chunk["score"] > doc_scores[doc_idx]:
                doc_scores[doc_idx] = chunk["score"]

        doc_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in doc_scores[:limit]:
            if idx is None:
                continue
            doc = self.documents[idx]
            results.append(
                format_results(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                )
            )
        return results


def get_titles_hash(documents: list[dict]) -> str:
    titles = "|".join([doc["title"] for doc in documents])
    titles_hash = hashlib.sha256(titles.encode())
    return titles_hash.hexdigest()


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return float(0)

    return float(dot_product / (norm1 * norm2))


def fixed_size_chunking(
    text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
):
    words = text.split()
    chunks = []
    i = 0
    n_words = len(words)
    while i < n_words:
        chunk_words = words[i : i + max_chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i = i + max_chunk_size - overlap

    return chunks


def chunk_semantic(
    text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        curr_chunk = []
        for sent in chunk_sentences:
            curr_chunk.append(sent.strip())
        if not curr_chunk:
            continue

        chunks.append(" ".join(curr_chunk))
        i += max_chunk_size - overlap
    return chunks
