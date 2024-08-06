import json
import uuid
from pathlib import Path

import hydra
from kaggle_environments import make
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb.sdk.wandb_run import Run
from llm_20q.agent import LLM20Q, dumb_agent_fn
from llm_20q.players import LLMPlayer, OpenaiPlayer, Player

# FIXME: Update LLM20Q to load with agent and rag
# TODO: Configuragle different agents for self play

def build_player(config: DictConfig, run:Run) -> Player:
    if config.player_type == "openai":
        return OpenaiPlayer(**config.player_kwargs)
    artifact = run.use_artifact(**config.pipeline_artifact)
    artifact_dir = artifact.download()
    llm20q = LLMPlayer.from_folder(artifact_dir)
    return llm20q


@hydra.main(config_path="../llm_20q/configs/selfplay", config_name="llama3-8b-inst", version_base=None)
def main(config: DictConfig) -> None:
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, **config.wandb_init)
    player = build_player(config, run)
    llm20q = LLM20Q(player)
    save_folder = Path(f"../input/self-play/{config.model_name}")
    save_folder.mkdir(parents=True, exist_ok=True)


    for _ in range(config.num_games):
        env = make("llm_20_questions", debug=True)
        game = env.run([llm20q.agent_fn, llm20q.agent_fn, dumb_agent_fn, dumb_agent_fn])
        game_id = uuid.uuid4()
        with open(save_folder / f"{game_id}.json", "w", encoding="utf-8") as f:
            save_game = {"steps": game, "info": {"model": config.model_name}}
            json.dump(save_game, f)
    artifact = wandb.Artifact(**raw_config["output_artifact"])
    artifact.add_dir(save_folder.absolute())
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
