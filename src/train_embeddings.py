from typing import Callable

import hydra
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CoSENTLoss
from sklearn.model_selection import StratifiedKFold

import wandb
from llm_20q import fix_prompt_rag


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

def generate_multy_questions(df:pd.DataFrame, num_samples:int=15) -> pd.DataFrame:
    questions_agg = []
    for num in range(num_samples):
        sample_df = df.groupby('keyword').sample(n=2 + num,
                                                 replace=False,
                                                 random_state=num)
        sample_df = sample_df.groupby('keyword').agg({'score':'mean', 'query': "\n".join, 'prompt':'first'})
        questions_agg.append(sample_df)
    
    return pd.concat(questions_agg, ignore_index=True)

def build_stratified_folds(df:pd.DataFrame, num_folds:int=5) -> pd.DataFrame:
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (_, test_index) in enumerate(kfold.split(df, df['keyword'])):
        df.loc[test_index, 'split'] = 'train' if fold != 0 else 'validation'
    return df


@hydra.main(config_path="llm_20q/configs/rag", config_name="nomic-embed-text-v1", version_base=None)
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
    knowledge_df = knowledge_df.reset_index().rename(columns={"index": "document_id"})
    single_question_df = answers_df.merge(knowledge_df, on="keyword")
    multi_questions = generate_multy_questions(single_question_df)
    single_question_df = build_stratified_folds(single_question_df)
    train_single_question_df = single_question_df.query('split == "train"').reset_index(drop=True)
    validation_single_question_df = single_question_df.query('split == "validation"').reset_index(drop=True)
    selected_df = pd.concat([train_single_question_df[["query", "prompt", "score"]],
                             multi_questions[["query", "prompt", "score"]]
                             ],
                            ignore_index=True)
    
    selected_df = fix_prompt_rag(model_name=config.model_name_or_path)(selected_df)
    # selected_df = selected_df.rename(columns={"query": "sentence1", "prompt": "sentence2"})
    query_id_mappings = validation_single_question_df.groupby('question_id')['document_id'].agg(set).to_dict()
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

    args = SentenceTransformerTrainingArguments(**config.trainer) # Will be used in W&B if `wandb` is installed

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=ir_evaluator,
    )
    trainer.train()
    model_artifact = wandb.Artifact(**raw_config['output_artifact'])
    model_artifact.add_dir(config.output_dir)
    run.log_artifact(model_artifact)
    run.finish()



if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
