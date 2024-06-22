import os
from typing import Literal

import hydra
import pandas as pd
from accelerate import PartialState
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, PreTrainedTokenizer,
                          TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import wandb
from llm_20q.data import build_df, build_game_records,TaskType
from llm_20q.model import (prepare_answer_messages, prepare_ask_messages,
                           prepare_guess_messages)

load_dotenv()

INPUT_DATASET_NAME = "replay-dataframe"
INPUT_DATASET_TYPE = "replay-games"

def generate_prompt(tokenizer: PreTrainedTokenizer, task:TaskType):

    def make_question_row(row):
        data = {
            "questions": row["questions"] + [row["question"]],
            "answers": row["answers"],
            "guesses": row["guesses"],
        }
        conversation = prepare_ask_messages(**data)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        return prompt
    
    def make_answer_row(row):
        data = {
            "questions": row["questions"],
            "answers": row["answers"] + [row["answer"]],
            "keyword": row["keyword"],
            "category": row["category"],
        }
        conversation = prepare_answer_messages(**data)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        return prompt
    
    def make_guess_row(row):
        data = {
            'questions': row['questions'],
            'answers': row['answers'] ,
            'guesses': row['guesses'] + [row['guess']],   
        }
        conversation = prepare_guess_messages(**data)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        return prompt
    
    match task:
        case 'ask':
            return make_question_row
        case 'answer':
            return make_answer_row
        case 'guess':
            return make_guess_row
        case _:
            raise ValueError(f"Invalid task type: {task}")
        


@hydra.main(config_path="llm_20q/configs", config_name="llama3-8b-inst", version_base=None)
def main(config: DictConfig) -> None:
    state = PartialState()
    model_id = config.model.pretrained_model_name_or_path
    model_name = config.model_name
    task = config.task
    artifact_dir = f'./artifacts/replay-df-{task}'
    if state.is_main_process:
        raw_config = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(config=raw_config, tags=[task, model_name], job_type="train-sft")
        artifact = run.use_artifact(f'{INPUT_DATASET_NAME}-{task}:latest', type=INPUT_DATASET_TYPE)
        artifact_dir = artifact.download(artifact_dir)
    state.wait_for_everyone()
    games_df = pd.read_parquet(f"{artifact_dir}/{task}.parquet")
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    games_df["prompt"] = games_df.apply(generate_prompt(tokenizer, task), axis=1)
    # QLoRA config
    bnb_config = BitsAndBytesConfig(**config.quantization)

    # LoRA config
    lora_config = OmegaConf.to_container(config.lora, resolve=True)
    peft_config = LoraConfig(**lora_config)
    model = AutoModelForCausalLM.from_pretrained(**config.model,
                                                 quantization_config=bnb_config,
                                                 device_map={"": state.process_index})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collate_fn = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, pad_to_multiple_of=8, response_template=response_template
    )
    dataset = Dataset.from_pandas(games_df[["prompt"]])
    dataset = dataset.map(lambda x: tokenizer(x["prompt"]))
    dataset = dataset.map(lambda x: {'input_length': len(x['input_ids'])})
    # dataset = dataset.filter(lambda x: x['input_length'] <= 1024)
    args = TrainingArguments(**config.trainer)

    state.wait_for_everyone()
    trainer = SFTTrainer(
        model=model,
        data_collator=collate_fn,
        peft_config=peft_config,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=args
    )
    trainer.train()
    if state.is_main_process:
        model_artifact = wandb.Artifact(f"{task}-{model_name}", type="model", metadata={"task": task})
        model_artifact.add_dir(config.output_dir)
        run.log_artifact(model_artifact)
        run.finish()
    state.wait_for_everyone()
    


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
