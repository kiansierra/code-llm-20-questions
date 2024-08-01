import hydra
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CoSENTLoss
from sklearn.model_selection import StratifiedKFold

import wandb
from llm_20q import fix_prompt_rag


def knowledge_template(keyword, category, knowledge, **kwargs):
    return f"# Keyword: {keyword}, # Categories: {', '.join(category)}, # Description: {knowledge}"


def preprocess_answers(knowledge_df: pd.DataFrame, answers_df: pd.DataFrame) -> pd.DataFrame:
    knowledge_df = knowledge_df.drop(columns="question").explode("question_ids").reset_index()
    knowledge_df = knowledge_df.rename(columns={"question_ids": "question_id", "index": "document_id"})
    answers_df = answers_df.query('answer != "drop"').drop(columns="raw_answer")
    qa_df = answers_df.merge(knowledge_df, on=["keyword", "answer", "question_id"])
    qa_df["score"] = qa_df["answer"].map({"yes": 1, "no": -1})
    qa_df["query"] = qa_df["question"]
    qa_df["prompt"] = "# Keyword: " + qa_df["keyword"] + "\n" + "# Knowledge: " + qa_df["knowledge"]

    return qa_df


def generate_multy_questions(df: pd.DataFrame, num_samples: int = 15) -> pd.DataFrame:
    questions_agg = []
    for num in range(num_samples):
        sample_df = df.groupby("keyword").sample(n=2 + num, replace=False, random_state=num)
        sample_df = sample_df.groupby("keyword").agg({"score": "mean", "query": "\n".join, "prompt": "first"})
        questions_agg.append(sample_df)

    return pd.concat(questions_agg, ignore_index=True)


def build_stratified_folds(df: pd.DataFrame, num_folds: int = 5) -> pd.DataFrame:
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (_, test_index) in enumerate(kfold.split(df, df["keyword"])):
        df.loc[test_index, "split"] = "train" if fold != 0 else "validation"
    return df


@hydra.main(config_path="llm_20q/configs/rag", config_name="nomic-embed-text-v1", version_base=None)
def main(config: DictConfig) -> None:
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, **config.wandb_init)
    answers_artifact = run.use_artifact(**config.input_artifact_answers)
    answers_artifact_dir = answers_artifact.download()
    answers_df = pd.read_parquet(f"{answers_artifact_dir}/{config.file_name_answers}")
    knowledge_artifact = run.use_artifact(**config.input_artifact_knowledge)
    knowledge_artifact_dir = knowledge_artifact.download()
    knowledge_df = pd.read_parquet(f"{knowledge_artifact_dir}/{config.file_name_knowledge}")
    single_question_df = preprocess_answers(knowledge_df, answers_df)

    multi_questions = generate_multy_questions(single_question_df)
    single_question_df = build_stratified_folds(single_question_df)
    train_single_question_df = single_question_df.query('split == "train"').reset_index(drop=True)
    validation_single_question_df = single_question_df.query('split == "validation" and answer == "yes"')
    validation_single_question_df = validation_single_question_df.reset_index(drop=True)
    selected_df = pd.concat(
        [train_single_question_df[["query", "prompt", "score"]], multi_questions[["query", "prompt", "score"]]], ignore_index=True
    )

    selected_df = fix_prompt_rag(model_name=config.model_name_or_path)(selected_df)
    # selected_df = selected_df.rename(columns={"query": "sentence1", "prompt": "sentence2"})
    query_id_mappings = validation_single_question_df.groupby("question_id")["document_id"].agg(set).to_dict()
    train_dataset = Dataset.from_pandas(selected_df)
    train_dataset = train_dataset.shuffle(42)
    eval_df = validation_single_question_df[["query", "prompt", "score", "question_id", "document_id"]]
    eval_df = fix_prompt_rag(model_name=config.model_name_or_path)(eval_df)
    # eval_df = eval_df.rename(columns={"query": "sentence1", "prompt": "sentence2"})
    # eval_df = eval_df.sample(n=1_000, random_state=42).reset_index(drop=True)
    eval_dataset = Dataset.from_pandas(eval_df[["query", "prompt", "score"]])

    corpus = eval_df[["document_id", "prompt"]].drop_duplicates()
    queries = eval_df[["question_id", "query"]].drop_duplicates()
    corpus = dict(zip(corpus["document_id"], corpus["prompt"]))  # Our corpus (cid => document)
    queries = dict(zip(queries["question_id"], queries["query"]))  # Our queries (qid => question)

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=query_id_mappings,
        name="Knowledge-Retrieval-Evaluator",
    )

    # Load a model to train/finetune
    model = SentenceTransformer(config.model_name_or_path, trust_remote_code=True)

    # Initialize the CoSENTLoss
    # This loss requires pairs of text and a float similarity score as a label
    loss = CoSENTLoss(model)

    args = SentenceTransformerTrainingArguments(**config.trainer)  # Will be used in W&B if `wandb` is installed

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=ir_evaluator,
    )
    trainer.train()
    model_artifact = wandb.Artifact(**raw_config["output_artifact"])
    model_artifact.add_dir(config.output_dir)
    run.log_artifact(model_artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
