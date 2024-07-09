import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.training_args import BatchSamplers
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from llm_20q.data import build_corpus

KNOWLEDGE_DATASET_NAME = "base-knowledge"
BASE_QUESTIONS_DATASET_NAME = "base-questions"
OPENAI_DATASET_NAME = "openai-questions"


def knowledge_template(keyword, category, knowledge, **kwargs):
    return f"# Keyword: {keyword}, # Categories: {', '.join(category)}, # Description: {knowledge}"

def preprocess_answers(df:pd.DataFrame) -> pd.DataFrame:
    df = df.query('answer != "drop"')
    df['score'] = 1
    opposite_df = df.copy()
    opposite_df['answer'] = opposite_df['answer'].map({'yes':'no', 'no':'yes'})
    opposite_df['score'] = -1
    concat_df = pd.concat([df, opposite_df], ignore_index=True)
    concat_df['query'] = concat_df['question'] + " " + concat_df['answer']
    return concat_df

def generate_multy_questions(df:pd.DataFrame, num_samples:int=10) -> pd.DataFrame:
    questions_agg = []
    for num in range(num_samples):
        sample_df = df.groupby('keyword').sample(n=2 + num % 3,replace=False)
        sample_df = sample_df.groupby('keyword').agg({'score':'mean', 'query': "\n".join, 'prompt':'first'})
        questions_agg.append(sample_df)
    
    return pd.concat(questions_agg, ignore_index=True)

@hydra.main(config_path="llm_20q/configs/rag", config_name="xlm-roberta-base", version_base=None)
def main(config: DictConfig) -> None:
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, **config.wandb_init)
    artifact = run.use_artifact(**config.input_artifact_answers)
    artifact_dir = artifact.download()

    answers_df = pd.read_parquet(f"{artifact_dir}/{config.file_name_answers}")
    answers_df = preprocess_answers(answers_df)
    artifact = run.use_artifact(**config.input_artifact_knowledge)
    artifact_dir = artifact.download()
    knowledge_df = pd.read_parquet(f"{artifact_dir}/{config.file_name_knowledge}")

    single_question_df = answers_df.merge(knowledge_df, on="keyword")
    multi_questions = generate_multy_questions(single_question_df)
    selected_df = pd.concat([single_question_df[["query", "prompt", "score"]], multi_questions[["query", "prompt", "score"]]], ignore_index=True)
    selected_df = selected_df.rename(columns={"query": "sentence1", "prompt": "sentence2"})
    train_dataset = Dataset.from_pandas(selected_df)
    train_dataset = train_dataset.shuffle(42)
    eval_df = single_question_df[["query", "prompt", "score"]].rename(columns={"query": "sentence1", "prompt": "sentence2"})
    eval_df = eval_df.sample(n=1_000, random_state=42).reset_index(drop=True)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Load a model to train/finetune
    model = SentenceTransformer(config.model_name)

    # Initialize the CoSENTLoss
    # This loss requires pairs of text and a float similarity score as a label
    loss = CoSENTLoss(model)

    args = SentenceTransformerTrainingArguments(**config.trainer) # Will be used in W&B if `wandb` is installed

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
