from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupKFold

import wandb
from llm_20q import SentenceTransformerRag, fix_prompt_rag

load_dotenv()

INPUT_DATASET_NAME = "replay-records"
OUTPUT_DATASET_NAME = "replay-dataframe"
DATASET_TYPE = "replay-games"


def build_folds(df, n_splits=5):
    group_kfold = GroupKFold(n_splits=n_splits)
    groups = df["EpisodeId"]
    df["split"] = "train"
    for num, (_, test_index) in enumerate(group_kfold.split(df, groups=groups)):
        df.loc[test_index, "split"] = "validation" if num == 0 else "train"

    return df


@hydra.main(config_path="../../llm_20q/configs/replays", config_name="nomic-embed-text-v1", version_base=None)
def main(config: DictConfig) -> None:
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(**config.wandb_init, config=raw_config)
    knowledge_artifact = run.use_artifact(**config.input_artifact_knowledge)
    knowledge_artifact_dir = knowledge_artifact.download()
    knowledge_df = pd.read_parquet(f"{knowledge_artifact_dir}/{config.file_name_knowledge}")
    knowledge_df["prompt"] = "# Keyword: " + knowledge_df["keyword"] + "\n" + "# Knowledge: " + knowledge_df["knowledge"]
    knowledge_df = fix_prompt_rag(model_name=config.model_name_or_path)(knowledge_df)

    replay_artifacts = run.use_artifact(**config.input_artifact_games)
    replay_artifacts_dir = replay_artifacts.download()
    guess_df = pd.read_parquet(f"{replay_artifacts_dir}/{config.file_name_games}")

    rag = SentenceTransformerRag(model_name_or_path=config.model_name_or_path, dataframe=knowledge_df)

    yes_kwargs = {"direction": "top", "top_p": 0.3}
    no_kwargs = {"direction": "bottom", "top_p": 0.9}

    new_records = []
    guess_df = guess_df.query('turnType == "ask"')
    logger.info(f"Processing {len(guess_df)} records")
    for episode_id in guess_df["EpisodeId"].unique():
        rag.reset()
        records = guess_df.query(f"EpisodeId == {episode_id}").to_dict("records")
        for record in records:
            question = record["questions"][-1]
            answer = record["answers"][-1]
            keyword = record["keyword"]

            kwargs = yes_kwargs if answer == "yes" else no_kwargs
            rag.filter(query=f"search_query: {question}", **kwargs)
            if keyword not in rag.filter_df["keyword"].tolist():
                break
            new_records.append({"options": rag.filter_df["keyword"].unique().tolist(), **record})

    rag.reset()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    rag.to_folder(config.output_dir)
    new_records_df = pd.DataFrame(new_records)
    logger.info(f"Processing {len(new_records_df)} records")
    new_records_df.to_parquet(f"{config.output_dir}/{config.output_file_name}")
    artifact = wandb.Artifact(**raw_config["output_artifact"])
    artifact.add_dir(config.output_dir)
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
