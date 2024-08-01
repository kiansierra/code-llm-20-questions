from dataclasses import dataclass
from typing import List

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import pipeline

__all__ = ["QuestionClassifier", "SimilarityFilterer"]


@dataclass
class QuestionClassifier:
    """Classify questions into categories."""

    model: str
    treshold: float = 0.9

    def __enter__(self) -> "QuestionClassifier":
        """Initialize the model."""
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.pipeline = pipeline("zero-shot-classification", model=self.model, device=device)
        return self

    def filter(self, questions: List[str]) -> List[int]:
        candidate_labels = ["question", "incomplete question", "not question"]
        output = self.pipeline(questions, candidate_labels=candidate_labels)
        drop_idxs = [num for num, elem in enumerate(output) if not (elem["labels"][0] == "question" and elem["scores"][0] > self.treshold)]
        return drop_idxs

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Close the model."""
        del self.pipeline
        torch.cuda.empty_cache()


@dataclass
class SimilarityFilterer:
    model: str
    treshold: float = 0.85

    def __enter__(self) -> "SimilarityFilterer":
        self.encoder = SentenceTransformer(self.model)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.encoder
        torch.cuda.empty_cache()

    def filter(self, sentences: list[str]) -> list[int]:
        embeddings = self.encoder.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        logger.info(f"Embeddings shape: {embeddings.shape}")
        similarity_matrix = pairwise_cosine_similarity(embeddings)
        x, _ = torch.where(similarity_matrix.tril_(-1) > self.treshold)
        drop_idx = list(set(x.tolist()))
        return drop_idx
