from typing import Callable, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

__all__ = ["SentenceTransformerRag", "fix_prompt_rag"]

def fix_prompt_rag(model_name:str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    def fix_nomic(df: pd.DataFrame) -> pd.DataFrame:
        # https://huggingface.co/nomic-ai/nomic-embed-text-v1#usage
        df['query'] = 'search_query: ' + df['query']
        df['prompt'] = 'search_document: ' + df['prompt']
        return df
    
    if "nomic" in model_name:
        return fix_nomic
    
    return lambda df: df
    

class SentenceTransformerRag:

    def __init__(self, model_name_or_path: str | Path,
                 dataframe: pd.DataFrame,
                 embed_column: str = "prompt",
                 embeddings:Optional[torch.Tensor] = None,
                 **kwargs):
        self.model = SentenceTransformer(str(model_name_or_path), **kwargs)
        self.dataframe = dataframe
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = self._build_embeddings(self.dataframe[embed_column].tolist())
            
    def to(self, device: str):
        self.model.to(device)
        self.embeddings = self.embeddings.to(device)
        return self


    def _build_embeddings(self, sentences: list[str]) -> list[np.ndarray]:
        return self.model.encode(sentences, normalize_embeddings=True,
                                 show_progress_bar=True,
                                 convert_to_tensor=True)
    
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
        query_embedding = self.model.encode(query,
                                            convert_to_tensor=True,
                                            normalize_embeddings=True,
                                            device=self.embeddings.device)
        scores = torch.nn.functional.cosine_similarity(query_embedding, self.embeddings) # pylint: disable=not-callable
        sorted_indices = torch.argsort(scores, descending=True)
        top_k_indices = sorted_indices[:top_k].tolist()
        return self.dataframe.iloc[top_k_indices], scores[top_k_indices].tolist()
