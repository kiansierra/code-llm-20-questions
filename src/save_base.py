import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

load_dotenv()


@hydra.main(config_path="llm_20q/configs", config_name="llama3-8b-inst", version_base=None)
def main(config: DictConfig) -> None:
    model_id = config.model.pretrained_model_name_or_path
    model_name = config.model_name
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, tags=["upload", model_name])
    save_dir = Path(config.output_dir).parent
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(**config.model)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    model_artifact = wandb.Artifact(f"{model_name}", type="model")
    model_artifact.add_dir(save_dir)
    run.log_artifact(model_artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
