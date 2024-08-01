import os

import hydra
import pandas as pd
from accelerate import PartialState
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import wandb
from llm_20q import generate_prompt

load_dotenv()


@hydra.main(config_path="llm_20q/configs", config_name="llama3.1-8b-inst-ask", version_base=None)
def main(config: DictConfig) -> None:
    state = PartialState()
    model_id = config.model.pretrained_model_name_or_path
    task = config.task
    artifact_dir = f"./artifacts/replay-df-{task}"
    if state.is_main_process:
        raw_config = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(config=raw_config, **config.wandb_init)
        artifact = run.use_artifact(**config.input_artifact)
        artifact_dir = artifact.download(artifact_dir)
    state.wait_for_everyone()
    games_df = pd.read_parquet(f"{artifact_dir}/{task}.parquet")
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    games_df["prompt"] = games_df.apply(generate_prompt(tokenizer, task), axis=1)
    # QLoRA config
    quantize = config.get("quantization", None)
    if quantize:
        bnb_config = BitsAndBytesConfig(**config.quantization)

    # LoRA config
    lora_config = OmegaConf.to_container(config.lora, resolve=True)
    peft_config = LoraConfig(**lora_config)
    model_params = {**config.model, "device_map": {"": state.process_index}}
    if quantize:
        model_params["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(**model_params)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    collate_fn = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, **config.collator_kwargs)
    train_df = games_df.query("split == 'train'").reset_index(drop=True)
    val_df = games_df.query("split == 'validation'").reset_index(drop=True)

    datasets = DatasetDict({"train": Dataset.from_pandas(train_df[["prompt"]]), "validation": Dataset.from_pandas(val_df[["prompt"]])})
    datasets = datasets.map(lambda x: tokenizer(x["prompt"]))
    datasets = datasets.map(lambda x: {"input_length": len(x["input_ids"])})
    max_seq_length = max(datasets["train"]["input_length"] + datasets["validation"]["input_length"])
    args = SFTConfig(**config.trainer, max_seq_length=max_seq_length)

    state.wait_for_everyone()
    trainer = SFTTrainer(
        model=model,
        data_collator=collate_fn,
        peft_config=peft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        args=args,
    )
    trainer.train()
    if state.is_main_process:
        model_artifact = wandb.Artifact(**raw_config["output_artifact"])
        model_artifact.add_dir(config.output_dir)
        run.log_artifact(model_artifact)
        run.finish()
    state.wait_for_everyone()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
