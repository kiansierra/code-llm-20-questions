from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

__all__ = ["SentenceTransformerRag", "fix_prompt_rag"]


def fix_prompt_rag(model_name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:

    def fix_nomic(df: pd.DataFrame) -> pd.DataFrame:
        # https://huggingface.co/nomic-ai/nomic-embed-text-v1#usage
        if "query" in df.columns:
            df["query"] = "search_query: " + df["query"]
        if "prompt" in df.columns:
            df["prompt"] = "search_document: " + df["prompt"]
        return df

    if "nomic" in model_name:
        return fix_nomic

    return lambda df: df


class SentenceTransformerRag:

    def __init__(
        self,
        model_name_or_path: str | Path,
        dataframe: pd.DataFrame,
        embed_column: str = "prompt",
        embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.model = SentenceTransformer(str(model_name_or_path), trust_remote_code=True, **kwargs)
        self.dataframe = dataframe
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = self._build_embeddings(self.dataframe[embed_column].tolist())

        self.filter_embedding = self.embeddings
        self.filter_df = self.dataframe

    def to(self, device: str):
        self.model.to(device)
        self.embeddings = self.embeddings.to(device)
        return self

    def _build_embeddings(self, sentences: list[str]) -> list[np.ndarray]:
        return self.model.encode(sentences, normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True)

    def to_folder(self, folder_path: str):
        logger.info(f"Saving embeddings to {folder_path}")
        torch.save(self.embeddings, f"{folder_path}/embeddings.pt")
        self.dataframe.to_parquet(f"{folder_path}/documents.parquet")
        logger.info("Saved embeddings and dataframe to folder")
        self.model.save_pretrained(str(folder_path))

    @classmethod
    def from_folder(cls, folder_path: str) -> "SentenceTransformerRag":
        logger.info(f"Loading embeddings from {folder_path}")
        embeddings = torch.load(f"{folder_path}/embeddings.pt")
        logger.info(f"Loading documents from {folder_path}")
        dataframe = pd.read_parquet(f"{folder_path}/documents.parquet")
        return cls(folder_path, dataframe, embeddings=embeddings)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True, device=self.embeddings.device)
        scores = torch.nn.functional.cosine_similarity(query_embedding, self.embeddings)  # pylint: disable=not-callable
        sorted_indices = torch.argsort(scores, descending=True)
        top_k_indices = sorted_indices[:top_k].tolist()
        return self.dataframe.iloc[top_k_indices], scores[top_k_indices].tolist()

    def filter(
        self, query: str, top_k: Optional[int] = None, top_p: Optional[float] = None, direction: Literal["top", "bottom"] = "top"
    ) -> pd.DataFrame:
        if top_k is None and top_p is None:
            raise ValueError("Either top_k or top_p must be provided")
        if direction not in ["top", "bottom"]:
            raise ValueError("direction must be either 'top' or 'bottom'")
        if top_k is not None:
            top_k_use = top_k
        if top_k is not None and top_p is not None:
            logger.warning("Both top_k and top_p are provided, using minimum of both")
            top_k_p = int(self.filter_embedding.size(0) * top_p)
            top_k_use = min(top_k, top_k_p)
        if top_k is None:
            top_k_use = int(self.filter_embedding.size(0) * top_p)
        query_embedding = self.model.encode(query,
                                            convert_to_tensor=True,
                                            normalize_embeddings=True,
                                            device=self.filter_embedding.device,
                                            show_progress_bar=False)
        scores = torch.nn.functional.cosine_similarity(query_embedding, self.filter_embedding)  # pylint: disable=not-callable
        descending = direction == "top"
        sorted_indices = torch.argsort(scores, descending=descending)
        selected_indices = sorted_indices[:top_k_use].tolist()
        self.filter_embedding = self.filter_embedding[selected_indices]
        self.filter_df = self.filter_df.iloc[selected_indices].reset_index(drop=True)
        if len(self.filter_df) == 0:
            logger.warning("No documents found, Resseting Index")
            self.reset()
            self.filter(query, top_k, top_p, direction)
        return self.filter_df

    def reset(self):
        self.filter_embedding = self.embeddings
        self.filter_df = self.dataframe
