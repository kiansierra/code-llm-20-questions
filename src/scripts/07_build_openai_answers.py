from llm_20q.data import generate_questions, build_corpus, generate_questions, generate_answers_async
import pandas as pd
from pathlib import Path
import wandb
from omegaconf import OmegaConf
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
from loguru import logger
import hydra
import uuid
import itertools
load_dotenv()

DATASET_NAME = "openai-questions"
DATASET_TYPE = "questions-dataset"

async def generate_answers_async_data(client:AsyncOpenAI, keyword:str, question:str, **kwargs) -> str:
    choices = await generate_answers_async(client, keyword, question, **kwargs)
    return [{'answer': choice.message.content, 'keyword':keyword, 'question':question}  for choice in choices]
     

async def generate_answers_async_data_all(question_pairs:list[str], client:AsyncOpenAI, **kwargs):
    questions = await asyncio.gather(*[generate_answers_async_data(client, **elem, **kwargs) for elem in question_pairs])
    questions = list(itertools.chain(*questions))
    logger.info('Finished generating yes questions')
    return questions

@hydra.main(config_path="../llm_20q/configs/openai", config_name='openai-answers')
def main(config):
    corpus_df = build_corpus()
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(**config.wandb_init, config=raw_config)
    artifact = run.use_artifact(f"{config.input_dataset_name}:latest", type=config.input_dataset_type)
    artifact_dir = Path(artifact.download())
    artifact_file = artifact_dir/ config.input_file_name
    questions_df = pd.read_parquet(artifact_file)
    questions_df['key'] = 0
    corpus_df['key'] = 0
    question_corpus_df = corpus_df.merge(questions_df, on='key', how='outer').drop('key', axis=1)
    records = question_corpus_df[['keyword', 'question']].to_dict('records')
    logger.info(f'{len(records)=}')
    client = AsyncOpenAI(timeout=60, max_retries=100)
    with asyncio.Runner() as runner:
        responses = runner.run(generate_answers_async_data_all(records, client, **config.generate_kwargs))
    answers_df = pd.DataFrame(responses)
    answers_df.reset_index(drop=True, inplace=True)
    logger.info(f'Generated {len(answers_df)=} Unique questions')
    answers_dir = Path('../input/answers')
    answers_dir.mkdir(exist_ok=True, parents=True)
    file_path = answers_dir/ config.file_name
    questions_df.to_parquet(file_path)
    
    artifact = wandb.Artifact(config.dataset_name, type=config.dataset_type)
    artifact.add_file(str(file_path), name=file_path.name)
    table = wandb.Table(dataframe=questions_df)
    run.log({'questions': table})
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == '__main__':
    main()
    
    
