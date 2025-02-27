import json
import uuid
from pathlib import Path

import hydra
import torch
from kaggle_environments import make
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import BitsAndBytesConfig, pipeline

import wandb
from llm_20q.prompts.prompt_templates import prepare_answer_messages, prepare_ask_messages, prepare_guess_messages
from llm_20q.utils import extract_last_checkpoint
from llm_20q.agent import LLM20Q

OUTPUT_DATASET_NAME = "self-play-records"
DATASET_TYPE = "self-play"


@hydra.main(config_path="../llm_20q/configs/selfplay", config_name="llama3-8b-inst", version_base=None)
def main(config: DictConfig) -> None:
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, **config.wandb_init)
    artifact = run.use_artifact(**config.pipeline_artifact)
    artifact_dir = artifact.download()
    llm20q = LLM20Q.from_folder(Path(artifact_dir))
    save_folder = Path(f"../input/self-play/{config.model_name}")
    save_folder.mkdir(parents=True, exist_ok=True)
    
    def dumb_agent_fn(obs, cfg):
        return "yes"


    for _ in range(config.num_games):
        env = make("llm_20_questions", debug=True)
        game = env.run([llm20q.agent_fn, llm20q.agent_fn, dumb_agent_fn, dumb_agent_fn])
        game_id = uuid.uuid4()
        with open(save_folder / f"{game_id}.json", "w", encoding="utf-8") as f:
            save_game = {"steps": game, "info": {"model": config.model_name}}
            json.dump(save_game, f)
    artifact = wandb.Artifact(**raw_config['output_artifact'])
    artifact.add_dir(save_folder.absolute())
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
