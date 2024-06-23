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
from llm_20q.model import prepare_answer_messages, prepare_ask_messages, prepare_guess_messages
from llm_20q.utils import extract_last_checkpoint

OUTPUT_DATASET_NAME = "self-play-records"
DATASET_TYPE = "self-play"


@hydra.main(config_path="../llm_20q/configs", config_name="llama3-8b-inst", version_base=None)
def main(config: DictConfig) -> None:

    bnb_config = BitsAndBytesConfig(**config.quantization)
    model_name = config.model_name
    pipe = pipeline(
        "conversational",
        model=config.model.pretrained_model_name_or_path,
        model_kwargs={
            "torch_dtype": config.model.torch_dtype,
            "quantization_config": bnb_config,
            "attn_implementation": config.model.attn_implementation,
        },
        device_map="auto",
    )
    logger.info("Loaded Model succesfully")
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, tags=["generation", model_name], job_type="self-play")
    ask_artifact = run.use_artifact(f"sft-ask-{model_name}:latest", type="model-sft")
    ask_dir = ask_artifact.download()
    guess_artifact = run.use_artifact(f"sft-guess-{model_name}:latest", type="model-sft")
    guess_dir = guess_artifact.download()
    pipe.model.load_adapter(extract_last_checkpoint(Path(ask_dir)), adapter_name="ask")
    pipe.model.load_adapter(extract_last_checkpoint(Path(guess_dir)), adapter_name="guess")
    logger.info("Loaded Adapters succesfully")
    ask_terminators = [pipe.tokenizer.eos_token_id, *pipe.tokenizer.convert_tokens_to_ids(["<|eot_id|>", "?", "?."])]

    def agent_fn(obs, _):
        # if agent is guesser and turnType is "ask"
        if obs.turnType == "ask":
            pipe.model.set_adapter("ask")
            conversation = prepare_ask_messages(obs.questions, obs.answers, obs.guesses)
            output = pipe(conversation, eos_token_id=ask_terminators)
            response = output[-1]["content"]
        # if agent is guesser and turnType is "guess"
        elif obs.turnType == "guess":
            pipe.model.set_adapter("guess")
            conversation = prepare_guess_messages(obs.questions, obs.answers, obs.guesses)
            output = pipe(conversation)
            response = output[-1]["content"]
        # if agent is the answerer
        elif obs.turnType == "answer":
            pipe.model.disable_adapters()
            yesno_words = ["yes", "no"]
            yes_no_ids = pipe.tokenizer.convert_tokens_to_ids(yesno_words)
            conversation = prepare_answer_messages(
                keyword=obs["keyword"], category=obs["category"], questions=obs.questions, answers=obs.answers
            )
            input_ids = pipe.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            with torch.no_grad():
                logits = pipe.model(input_ids).logits
            position = logits[0, -1, yes_no_ids].argmax()
            response = yesno_words[position]
        return response

    save_folder = Path(f"../input/self-play/{model_name}")
    save_folder.mkdir(parents=True, exist_ok=True)

    env = make("llm_20_questions", debug=True)
    game = env.run([agent_fn, agent_fn, agent_fn, agent_fn])

    for _ in range(10):
        env = make("llm_20_questions", debug=True)
        game = env.run([agent_fn, agent_fn, agent_fn, agent_fn])
        game_id = uuid.uuid4()
        with open(save_folder / f"{game_id}.json", "w", encoding="utf-8") as f:
            save_game = {"steps": game, "info": {"model": model_name}}
            json.dump(save_game, f)
    artifact = wandb.Artifact(f"{OUTPUT_DATASET_NAME}-{model_name}", type=DATASET_TYPE)
    artifact.add_dir(save_folder.absolute())
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
