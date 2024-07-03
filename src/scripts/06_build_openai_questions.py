from llm_20q.data import generate_questions, build_corpus, generate_questions, generate_questions_async
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

async def generate_questions_async_data(client:AsyncOpenAI, **kwargs) -> str:
    choices = await generate_questions_async(client, **kwargs)
    return [{'question': choice.message.content, 'question_id':str(uuid.uuid4())}  for choice in choices]
     

async def generate_questions_async_data_all(num_questions:list[str], client:AsyncOpenAI, **kwargs):
    questions = await asyncio.gather(*[generate_questions_async_data(client, **kwargs) for num in range(num_questions)])
    questions = list(itertools.chain(*questions))
    logger.info('Finished generating yes questions')
    return questions

@hydra.main(config_path="../llm_20q/configs/questions", config_name='openai-questions')
def main(config):
    client = AsyncOpenAI(timeout=60, max_retries=100)
    with asyncio.Runner() as runner:
        responses = runner.run(generate_questions_async_data_all(config.num_questions, client, **config.generate_kwargs))
    questions_df = pd.DataFrame(responses)
    questions_df.drop_duplicates(subset=['question'],inplace=True)
    questions_df.reset_index(drop=True, inplace=True)
    logger.info(f'Generated {len(questions_df)} Unique questions')
    questions_dir = Path('../input/questions')
    questions_dir.mkdir(exist_ok=True, parents=True)
    file_path = questions_dir/ config.file_name
    questions_df.to_parquet(file_path)
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(**config.wandb_init, config=raw_config)
    artifact = wandb.Artifact(config.dataset_name, type=config.dataset_type)
    artifact.add_file(str(file_path), name=file_path.name)
    table = wandb.Table(dataframe=questions_df)
    run.log({'questions': table})
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == '__main__':
    main()
    
    
