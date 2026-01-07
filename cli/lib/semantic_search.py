from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


def verify_model():
    index = SemanticSearch()
    print(f"Model loaded: {index.model}")
    print(f"Max sequence length: {index.model.max_seq_length}")
