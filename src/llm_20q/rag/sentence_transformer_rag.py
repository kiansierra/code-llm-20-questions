import json
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pydantic import BaseModel, PositiveFloat, PositiveInt, field_validator
from sentence_transformers import SentenceTransformer

__all__ = ["SentenceTransformerRag",  "RagConfig"]


def fix_dataframe_rag(embedding_type: str) -> Callable[[pd.DataFrame], pd.DataFrame]:

    def fix_nomic(df: pd.DataFrame) -> pd.DataFrame:
        # https://huggingface.co/nomic-ai/nomic-embed-text-v1#usage
        if "query" in df.columns:
            df["query_fixed"] = "search_query: " + df["query"]
        if "prompt" in df.columns:
            df["prompt_fixed"] = "search_document: " + df["prompt"]
        return df
    
    def default(df: pd.DataFrame) -> pd.DataFrame:
        # https://huggingface.co/nomic-ai/nomic-embed-text-v1#usage
        if "query" in df.columns:
            df["query_fixed"] =  df["query"]
        if "prompt" in df.columns:
            df["prompt_fixed"] =  df["prompt"]
        return df

    if "nomic" in embedding_type:
        return fix_nomic

    return default

def fix_query_rag(embedding_type: str) -> Callable[[str], str]:

    def fix_nomic(query: str) -> str:
        # https://huggingface.co/nomic-ai/nomic-embed-text-v1#usage
        return "search_query: " + query
    if "nomic" in embedding_type:
        return fix_nomic

    return lambda x: x


class RagConfig(BaseModel):
    embedding_type: str
    min_kw: PositiveInt
    top_p: PositiveFloat
    bottom_p: PositiveFloat
    
    @field_validator('top_p')
    @classmethod
    def top_p_below_1(cls, v: PositiveFloat) -> PositiveFloat:
        if  v >= 1.0:
            raise ValueError("top_p must be below 1.0")
        return v
    
    @field_validator('bottom_p')
    @classmethod
    def bottom_p_below_1(cls, v: PositiveFloat) -> PositiveFloat:
        if  v >= 1.0:
            raise ValueError("bottom_p must be below 1.0")
        return v
    


class SentenceTransformerRag:

    def __init__(
        self,
        model_name_or_path: str | Path,
        config: RagConfig,
        dataframe: pd.DataFrame,
        embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):  
        self.config = config
        self.model = SentenceTransformer(str(model_name_or_path), trust_remote_code=True, **kwargs)
        self.dataframe = fix_dataframe_rag(config.embedding_type)(dataframe)
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = self._build_embeddings(self.dataframe['prompt_fixed'].tolist())

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
        with open(f"{folder_path}/config.json", "w", encoding='utf-8') as file:
            json.dump(self.config.model_dump_json(), file)

    @classmethod
    def from_folder(cls, folder_path: str) -> "SentenceTransformerRag":
        logger.info(f"Loading embeddings from {folder_path}")
        embeddings = torch.load(f"{folder_path}/embeddings.pt")
        logger.info(f"Loading documents from {folder_path}")
        dataframe = pd.read_parquet(f"{folder_path}/documents.parquet")
        with open(f"{folder_path}/config.json", "r", encoding='utf-8') as file:
            json_config = json.load(file)
        config = RagConfig(**json_config)
        return cls(folder_path, config, dataframe, embeddings=embeddings)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_embedding = self.model.encode(query,
                                            convert_to_tensor=True,
                                            normalize_embeddings=True,
                                            device=self.embeddings.device)
        scores = torch.nn.functional.cosine_similarity(query_embedding, self.embeddings)  # pylint: disable=not-callable
        sorted_indices = torch.argsort(scores, descending=True)
        top_k_indices = sorted_indices[:top_k].tolist()
        return self.dataframe.iloc[top_k_indices], scores[top_k_indices].tolist()

    def filter(
        self,
        query: str,
        direction: Literal["top", "bottom"] = "top"
    ) -> None:
        query = fix_query_rag(self.config.embedding_type)(query)
        if direction not in ["top", "bottom"]:
            raise ValueError("direction must be either 'top' or 'bottom'")
        top_p = self.config.top_p if direction == 'top' else self.config.bottom_p
        top_k = int(self.filter_embedding.size(0) * top_p)
        top_k = max(top_k, self.config.min_kw)
        query_embedding = self.model.encode(query,
                                            convert_to_tensor=True,
                                            normalize_embeddings=True,
                                            device=self.filter_embedding.device,
                                            show_progress_bar=False)
        scores = torch.nn.functional.cosine_similarity(query_embedding, self.filter_embedding)  # pylint: disable=not-callable
        descending = direction == "top"
        sorted_indices = torch.argsort(scores, descending=descending)
        selected_indices = sorted_indices[:top_k].tolist()
        self.filter_embedding = self.filter_embedding[selected_indices]
        self.filter_df = self.filter_df.iloc[selected_indices].reset_index(drop=True)
        if len(self.filter_df) == 0:
            logger.warning("No documents found, Resseting Index")
            self.reset()
            self.filter(query, direction)

    
    def retrieve_options(self, num_options: int) -> list[str]:
        return self.filter_df['keyword'].tolist()[:num_options]
    
    def remove_guess(self, keyword: str) -> None:
        if keyword not in self.filter_df["keyword"].tolist():
            return
        self.filter_df = self.filter_df.query(f"keyword != '{keyword}'")
        self.filter_embedding = self.embeddings[self.filter_df.index]
        self.filter_df.reset_index(drop=True, inplace=True)
        if len(self.filter_df) == 0:
            logger.warning("No documents found, Resseting Index")
            self.reset()
            self.remove_guess(keyword)

    def reset(self) -> None:
        self.filter_embedding = self.embeddings
        self.filter_df = self.dataframe
