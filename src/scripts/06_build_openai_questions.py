from llm_20q.data import generate_questions, build_corpus, generate_questions
import pandas as pd
from pathlib import Path
import wandb
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

DATASET_NAME = "openai-questions"
DATASET_TYPE = "questions-dataset"

def main():
    corpus_df = build_corpus()
    keywords = corpus_df['keyword'].unique()
    client = OpenAI()
    responses = []
    for keyword in tqdm(keywords):
        response = generate_questions(keyword, client, answer='yes')
        responses.append({'keyword': keyword, 'questions': response, 'answer': 'yes'})
        response = generate_questions(keyword, client, answer='no')
        responses.append({'keyword': keyword, 'questions': response, 'answer': 'no'})
    questions_df = pd.DataFrame(responses)
    questions_df['questions'] =  questions_df['questions'].str.split('\n')
    questions_df = questions_df.explode('questions')
    questions_dir = Path('../input/questions')
    questions_dir.mkdir(exist_ok=True, parents=True)
    questions_df['similarity'] = 1
    questions_df.to_parquet(questions_dir/'openai.parquet')
    run = wandb.init(job_type="upload-openai-questions")
    artifact = wandb.Artifact(DATASET_NAME, type=DATASET_TYPE)
    artifact.add_dir(str(questions_dir))
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == '__main__':
    main()
    
    
