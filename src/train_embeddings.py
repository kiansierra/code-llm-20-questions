import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.training_args import BatchSamplers

import wandb
from llm_20q.data import build_corpus

KNOWLEDGE_DATASET_NAME = "base-knowledge"
BASE_QUESTIONS_DATASET_NAME = "base-questions"
OPENAI_DATASET_NAME = "openai-questions"


def knowledge_template(keyword, category, knowledge, **kwargs):
    return f"# Keyword: {keyword}, # Categories: {', '.join(category)}, # Description: {knowledge}"


def main():
    run = wandb.init()
    knowledge_ds = run.use_artifact(f"{KNOWLEDGE_DATASET_NAME}:latest")
    knowledge_dir = knowledge_ds.download()
    knowledge_df = pd.read_parquet(f"{knowledge_dir}/base.parquet")
    corpus_df = build_corpus()
    corpus_knowledge_df = corpus_df.merge(knowledge_df, on="keyword")
    corpus_knowledge_df["prompt"] = corpus_knowledge_df.apply(lambda x: knowledge_template(**x), axis=1)

    questions_ds = run.use_artifact(f"{BASE_QUESTIONS_DATASET_NAME}:latest")
    questions_dir = questions_ds.download()
    questions_df = pd.read_parquet(f"{questions_dir}/base.parquet")

    questions_openai_ds = run.use_artifact(f"{OPENAI_DATASET_NAME}:latest")
    questions_openai_dir = questions_openai_ds.download()
    questions_openai_df = pd.read_parquet(f"{questions_openai_dir}/openai.parquet")
    all_questions_df = pd.concat([questions_openai_df, questions_df])[["keyword", "question", "similarity"]]

    all_questions_df = all_questions_df.merge(corpus_knowledge_df, on="keyword")
    all_questions_df.rename(columns={"similarity": "score"}, inplace=True)

    dataset = Dataset.from_pandas(all_questions_df[["question", "prompt", "score"]])

    # Load a model to train/finetune
    model = SentenceTransformer("xlm-roberta-base")

    # Initialize the CoSENTLoss
    # This loss requires pairs of text and a float similarity score as a label
    loss = CoSENTLoss(model)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/mpnet-base-all-nli-triplet",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()


if __name__ == "__main__":
    main()
