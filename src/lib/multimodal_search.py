import mimetypes

from google.genai.types import Part
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.lib.query_enhancement import prompt_gemini
from src.lib.semantic_search import cosine_similarity
from src.lib.utils import load_movies


class MultimodalSearch:
    def __init__(self, documents: list, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{d['title']}: {d['description']}" for d in documents]
        self.text_embeddings = self.model.encode(self.texts)

    def embed_image(self, img_path: str):
        img_content = Image.open(img_path)
        embedding = self.model.encode([img_content], show_progress_bar=True)
        return embedding[0]

    def search_with_image(self, img_path: str):
        img_embedding = self.embed_image(img_path)
        for embedding, d in zip(self.text_embeddings, self.documents):
            d["similarity"] = cosine_similarity(embedding, img_embedding)

        results = sorted(self.documents, key=lambda x: x["similarity"], reverse=True)

        return results[:5]


def image_search_command(img_path):
    data = load_data()
    data = data["movies"]
    index = MultimodalSearch(data)
    results = index.search_with_image(img_path)
    return results


def verify_image_embedding(img_path: str):
    index = MultimodalSearch()
    embedding = index.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def describe_image(query: str, img_path: str):
    mime, _ = mimetypes.guess_type(img_path)
    mime = mime or "image/jpeg"

    with open(img_path, "rb") as f:
        img_content = f.read()

    prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    parts = [
        prompt,
        Part.from_bytes(data=img_content, mime_type=mime),
        query.strip(),
    ]

    return prompt_gemini(parts)
