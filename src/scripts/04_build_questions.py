from llm_20q.data import build_corpus, build_category_questions, build_letter_based_questions
import pandas as pd
from pathlib import Path
import wandb

DATASET_NAME = "base-questions"
DATASET_TYPE = "questions-dataset"

def main():
    corpus_df = build_corpus()
    category_df = build_category_questions()
    letter_df = build_letter_based_questions()
    corpus_df['key'] = 1
    category_df['key'] = 1
    letter_df['key'] = 1
    category_questions_df = corpus_df.merge(category_df, on='key').drop(columns='key')
    category_questions_df['similarity'] = category_questions_df.apply(lambda x: (x['query_category'] in x['category']) * x['score'], axis=1)
    letter_questions_df = corpus_df.merge(letter_df, on='key').drop(columns='key')
    letter_questions_df['similarity'] = letter_questions_df.apply(lambda x: (x['query_letter'].lower() == x['letter'].lower()) * x['score'], axis=1)
    keep_columns = ['question', 'similarity', 'keyword', 'category', 'alts']
    letter_df = letter_questions_df[keep_columns]
    category_df = category_questions_df[keep_columns]
    questions_df = pd.concat([letter_df, category_df], ignore_index=True)
    questions_dir = Path('../input/questions')
    questions_dir.mkdir(exist_ok=True, parents=True)
    questions_df.to_parquet(questions_dir/'base.parquet')
    run = wandb.init(job_type="upload-questions")
    artifact = wandb.Artifact(DATASET_NAME, type=DATASET_TYPE)
    artifact.add_dir(str(questions_dir))
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == '__main__':
    main()
    
    
