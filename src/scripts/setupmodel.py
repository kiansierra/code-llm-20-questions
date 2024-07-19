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
    model_name = config.model_name
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(job_type="setup-full-model")
    ask_artifact = run.use_artifact(**config.ask_artifact)
    save_dir = Path(config.output_dir)
    ask_dir = ask_artifact.download(save_dir / 'ask')
    guess_artifact = run.use_artifact(**config.guess_artifact)
    guess_dir = guess_artifact.download(save_dir / 'guess')
    rag_artifact = run.use_artifact(**config.rag_artifact)
    rag_dir = rag_artifact.download(save_dir / 'rag')
    bnb_config = BitsAndBytesConfig(**config.quantization)
    model_params = {**config.model, "device_map": 'auto'}
    model_params["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(**model_params)
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    model = LLM20Q(pipe, config, save_dir)
    
    def dumb_agent_fn(obs, cfg):
        return "yes"
    
    def agent_fn(obs, cfg):
    # if agent is guesser and turnType is "ask"
        if obs.turnType == "ask":
            response =  model.ask(obs, cfg)
            logger.info(f"ask: {response}")
        # if agent is guesser and turnType is "guess"
        elif obs.turnType == "guess":
            response = model.guess(obs, cfg)
            logger.info(f"guess: {response}")
        # if agent is the answerer
        elif obs.turnType == "answer":
            # pipe.model.disable_adapters()
            response = model.answer(obs, cfg)
            still_available = obs.keyword in model.rag.filter_df.keyword.tolist()
            logger.info(f"answer: {response} -- keyword: {obs.keyword} {still_available=}")
        return response
    
    env = make("llm_20_questions", debug=True)
    game = env.run([agent_fn, agent_fn, dumb_agent_fn, dumb_agent_fn])
    # print(game)
        

if __name__ == "__main__":
    main()
