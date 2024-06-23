import os

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import llm_20q.resolvers  # noqa: F401
import wandb

load_dotenv()


@hydra.main(config_path="llm_20q/configs", config_name="llama3-8b-inst", version_base=None)
def main(config: DictConfig) -> None:
    # login(token=os.environ["HF_TOKEN"])
    model_id = config.model.pretrained_model_name_or_path
    model_name = config.model_name
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, tags=["upload", model_name])
    save_dir = "model"
    ask_artifact = run.use_artifact(f"sft-ask-{model_name}:latest", type="model-sft")
    _ = ask_artifact.download(f"{save_dir}/ask")
    logger.info("Downloaded ask model")
    guess_artifact = run.use_artifact(f"sft-guess-{model_name}:latest", type="model-sft")
    _ = guess_artifact.download(f"{save_dir}/guess")
    logger.info("Downloaded guess model")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    model_cfg.pop("attn_implementation", None)
    bnb_config = BitsAndBytesConfig(**config.quantization)
    model = AutoModelForCausalLM.from_pretrained(**model_cfg, quantization_config=bnb_config)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
