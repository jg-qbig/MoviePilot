import mimetypes

from google.genai.types import Part
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.lib.query_enhancement import prompt_gemini
from src.lib.semantic_search import cosine_similarity
from src.lib.utils import SEARCH_LIMIT, format_results, load_movies


class MultimodalSearch:
    def __init__(self, documents: list, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"use_fast": True},
            tokenizer_kwargs={"use_fast": True},
            config_kwargs={"use_fast": True},
        )
        self.documents = documents
        self.texts = [f"{d['title']}: {d['description']}" for d in documents]
        self.text_embeddings = self.model.encode(self.texts)

    def embed_image(self, img_path: str):
        img_content = Image.open(img_path)
        embedding = self.model.encode([img_content], show_progress_bar=True)
        return embedding[0]

    def search(self, img_path: str, limit: int = SEARCH_LIMIT):
        img_embedding = self.embed_image(img_path)
        results = []
        for embedding, doc in zip(self.text_embeddings, self.documents):
            doc["similarity"] = cosine_similarity(embedding, img_embedding)

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


def multimodal_prompt_gemini(query: str, img_path: str) -> str:
    mime, _ = mimetypes.guess_type(img_path)
    mime = mime or "image/jpeg"

    with open(img_path, "rb") as f:
        img_content = f.read()

    prompt = f"""
    From the provided query and image, improve the query considering the additional information from the image.
    Synthesize visual and textual information.
    Return only the improved query.
    """

    parts = [
        prompt,
        Part.from_bytes(data=img_content, mime_type=mime),
        query.strip(),
    ]

    return prompt_gemini(parts)


def verify_image_embedding(img_path: str):
    data = load_movies()
    index = MultimodalSearch(data)
    embedding = index.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
