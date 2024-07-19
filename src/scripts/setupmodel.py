from pathlib import Path
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import hydra
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold
from kaggle_environments import make
import wandb
from llm_20q.data import build_game_records
from llm_20q.model import LLM20Q
from omegaconf import OmegaConf
from loguru import logger
load_dotenv()

INPUT_DATASET_NAME = "self-play-records"
DATASET_TYPE = "self-play"



@hydra.main(config_path="../llm_20q/configs/pipeline", config_name="llama3-8b-inst", version_base=None)
def main(config) -> None:
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, **config.wandb_init)
    ask_artifact = run.use_artifact(**config.ask_artifact)
    save_dir = Path(config.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config, save_dir / "config.yaml")
    _ = ask_artifact.download(save_dir / 'ask')
    guess_artifact = run.use_artifact(**config.guess_artifact)
    _ = guess_artifact.download(save_dir / 'guess')
    rag_artifact = run.use_artifact(**config.rag_artifact)
    _ = rag_artifact.download(save_dir / 'rag')
    bnb_config = BitsAndBytesConfig(**config.quantization)
    model_params = {**config.model, "device_map": 'auto'}
    model_params["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(**model_params)
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    pipe = pipeline(**config.pipeline, model=model, tokenizer=tokenizer)
    model = LLM20Q(pipe, config, save_dir)
    
    def dumb_agent_fn(obs, cfg):
        return "yes"
    
    def agent_fn(obs, cfg):
        return model.agent_fn(obs, cfg)
    
    for num in range(2):
        logger.info(f"*** Starting Game {num} ***")
        env = make("llm_20_questions", debug=True)
        game = env.run([agent_fn, agent_fn, dumb_agent_fn, dumb_agent_fn])
        if game[-1][0].reward > 0:
            logger.info(f"*** Game won: {game[-1][0].reward} ***")
            
    artifact = wandb.Artifact(**raw_config["output_artifact"])
    artifact.add_dir(str(save_dir))
    run.log_artifact(artifact)
    run.finish()

    # print(game)
        

if __name__ == "__main__":
    main()
