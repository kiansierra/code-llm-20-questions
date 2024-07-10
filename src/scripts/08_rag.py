import os
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

import wandb
from llm_20q import SentenceTransformerRag, extract_last_checkpoint

load_dotenv()


@hydra.main(config_path="../llm_20q/configs/knowledge", config_name="rag-model", version_base=None)
def main(config):
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(**config.wandb_init, config=raw_config)
    artifact = run.use_artifact(f"{config.input_artifact.name}:latest", type=config.input_artifact.type)
    artifact_dir = Path(artifact.download())
    artifact_file = artifact_dir / config.file_name
    knowledge_df = pd.read_parquet(artifact_file)

    knowledge_df.reset_index(drop=True, inplace=True)
    model_artifact = run.use_artifact(**config.input_model)
    model_dir = Path(model_artifact.download())
    rag = SentenceTransformerRag( dataframe=knowledge_df, model_name_or_path=extract_last_checkpoint(model_dir))
    logger.info(f"Generated {len(knowledge_df)=} Unique questions")
    rag_dir = Path(f"../input/rag/{config.model_name}")
    rag_dir.mkdir(exist_ok=True, parents=True)
    rag.to_folder(rag_dir)
    assert os.path.exists(rag_dir / "embeddings.pt"), "Embeddings not saved"
    assert os.path.exists(rag_dir / "documents.parquet"), "Documents not saved"

    artifact = wandb.Artifact(**raw_config['output_artifact'])
    artifact.add_dir(str(rag_dir))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
