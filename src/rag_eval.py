import wandb
from llm_20q import SentenceTransformerRag, RagConfig
from dotenv import load_dotenv
import pandas as pd 
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

load_dotenv()

@hydra.main(config_path="llm_20q/configs/replays", config_name="nomic-embed-text-v1", version_base=None)
def main(config):
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, **config.wandb_init)
    knowledge_artifact = run.use_artifact(**config.input_artifact_knowledge)
    knowledge_artifact_dir = knowledge_artifact.download()
    knowledge_df = pd.read_parquet(f"{knowledge_artifact_dir}/{config.file_name_knowledge}")
    knowledge_df["prompt"] = "# Keyword: " + knowledge_df["keyword"] + "\n" + "# Knowledge: " + knowledge_df["knowledge"]
    replay_artifacts = run.use_artifact(**config.input_artifact_games)
    replay_artifacts_dir = replay_artifacts.download()
    guess_df = pd.read_parquet(f"{replay_artifacts_dir}/{config.file_name_games}")
    
    rag_config = RagConfig(**config.rag_config)
    
    rag = SentenceTransformerRag(model_name_or_path=config.model_name_or_path,
                                 config=rag_config,
                                 dataframe=knowledge_df)
    new_records = []
    logger.info(f"Processing {len(guess_df)} records")
    for episode_id in guess_df["EpisodeId"].unique():
        rag.reset()
        records = guess_df.query(f"EpisodeId == {episode_id}").to_dict("records")
        for record in records:
            question = record["questions"][-1]
            answer = record["answers"][-1]
            keyword = record["keyword"]
            direction="top" if answer == "yes" else "bottom"
            rag.filter(query=question, direction=direction)
            remaining_kw = rag.filter_df.drop_duplicates(subset=['keyword'])['keyword'].nunique()
            total_kw = rag.dataframe['keyword'].nunique()
            is_kw_available = keyword in rag.filter_df["keyword"].tolist()
            if is_kw_available:
                position_kw = rag.filter_df.drop_duplicates(subset=['keyword'])['keyword'].tolist().index(keyword)
            else:
                position_kw = -1
            data = {"position_kw": position_kw,
                    "remaining_kw": remaining_kw,
                    "remaining_fraction": remaining_kw / total_kw,
                    "relative_pos": position_kw / remaining_kw if position_kw != -1 else -1,
                    "is_kw_available": is_kw_available}
            new_records.append({"options": rag.filter_df["keyword"].unique().tolist(), **record, **data})
            rag.remove_guess(record["guess"])
            if not is_kw_available:
                break

    rag.reset()
    
    
if __name__ == "__main__":
    main()