import os
import mimetypes

from google.genai.types import Part
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

from src.lib.query_enhancement import prompt_gemini
from src.lib.semantic_search import SemanticSearch, cosine_similarity
from src.lib.utils import CACHE_PATH, HYBRID_ALPHA, SEARCH_LIMIT, format_results


MULTIMODAL_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "multimodal_embeddings.npy")


class MultimodalSearch(SemanticSearch):
    def __init__(self, model_name="clip-ViT-B-32"):
        super().__init__(model_name=model_name)
        self.model = SentenceTransformer(
            model_name,
            tokenizer_kwargs={"use_fast": True},
        )
        self.embeddings_path = MULTIMODAL_EMBEDDINGS_PATH

    def generate_embedding(self, input: str):
        if not input.strip():
            raise ValueError("Input is empty.")

        is_image = os.path.isfile(input) and input.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        )

        if is_image:
            img_content = Image.open(input)
            embedding = self.model.encode([img_content], show_progress_bar=True)
            return embedding[0]
        return self.model.encode([input])[0]

    def search_multi(
        self,
        query: str,
        img_path: str,
        limit: int = SEARCH_LIMIT,
        alpha: float = HYBRID_ALPHA,
    ):
        query_embedding = normalize(self.generate_embedding(query))
        img_embedding = normalize(self.generate_embedding(img_path))

        combined_embedding = alpha * query_embedding + (1 - alpha) * img_embedding

        for embedding, doc in zip(self.embeddings, self.documents):
            # Seems to work much better (less bias) than cosine similarity
            doc["similarity"] = np.dot(normalize(embedding), combined_embedding)

        sorted_docs = sorted(
            self.documents, key=lambda x: x["similarity"], reverse=True
        )

        results = []
        for doc in sorted_docs[:limit]:
            results.append(
                format_results(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=doc["similarity"],
                )
            )
        return results


def normalize(embedding: np.ndarray) -> np.ndarray:
    return np.linalg.norm(embedding, axis=-1, keepdims=True)


def multimodal_prompt_gemini(query: str, img_path: str) -> str:
    mime, _ = mimetypes.guess_type(img_path)
    mime = mime or "image/jpeg"

    with open(img_path, "rb") as f:
        img_content = f.read()

    prompt = f"""
    You are a universal image & OCR assistant.
    Your task is to infer an optimal search query for a movie search engine based on the input image.

    You should:
    - Describe relevant object, people and settings
    - Focus on completeness while being as concise as possible
    - Transcribe any visible text in order

    Your final query should be optimized to retrieve the best results in a hybrid search engine that uses keyword and vectorized semantic search.

    Query:
    """

    parts = [
        prompt,
        Part.from_bytes(data=img_content, mime_type=mime),
        query.strip(),
    ]

    return prompt_gemini(parts)
