from llm_20q.data import generate_questions, build_corpus, generate_questions, generate_questions_async
import pandas as pd
from pathlib import Path
import wandb
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
from loguru import logger

load_dotenv()

DATASET_NAME = "openai-questions"
DATASET_TYPE = "questions-dataset"

async def generate_questions_async_data(keyword:str, client:AsyncOpenAI, answer:str):
    response = await generate_questions_async(keyword, client, answer=answer)
    return {'keyword': keyword, 'question': response, 'answer': answer}

async def generate_questions_async_data_all(keywords:list[str], client:AsyncOpenAI):
    response_yes = await asyncio.gather(*[generate_questions_async_data(keyword, client, 'yes') for keyword in keywords])
    logger.info('Finished generating yes questions')
    response_no = await asyncio.gather(*[generate_questions_async_data(keyword, client, 'no') for keyword in keywords])
    logger.info('Finished generating No questions')
    return response_yes + response_no

def main():
    corpus_df = build_corpus()
    keywords = corpus_df['keyword'].unique()
    client = AsyncOpenAI(timeout=60, max_retries=100)
    with asyncio.Runner() as runner:
        responses = runner.run(generate_questions_async_data_all(keywords, client))
    questions_df = pd.DataFrame(responses)
    questions_df['question'] =  questions_df['question'].str.split('\n')
    questions_df = questions_df.explode('question')
    questions_df['question'] = questions_df['question'].str.strip() + " " + questions_df['answer'].str.strip()
    questions_dir = Path('../input/questions')
    questions_dir.mkdir(exist_ok=True, parents=True)
    questions_df['similarity'] = 1
    file_path = questions_dir/'openai.parquet'
    questions_df.to_parquet(file_path)
    run = wandb.init(job_type="upload-openai-questions")
    artifact = wandb.Artifact(DATASET_NAME, type=DATASET_TYPE)
    artifact.add_file(str(file_path), name=file_path.name)
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == '__main__':
    main()
    
    
