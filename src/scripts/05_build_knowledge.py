from pathlib import Path
import hydra
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import itertools
import wandb
from llm_20q.data import build_corpus, generate_knowledge, generate_knowledge_async
from omegaconf import OmegaConf, DictConfig
import asyncio 
from loguru import logger

load_dotenv()


async def genereate_knowledge_record(keyword: str, client: AsyncOpenAI, **kwargs):
    choices = await generate_knowledge_async(keyword, client, **kwargs)
    return [{"keyword": keyword, "knowledge": choice.message.content} for choice in choices]

async def generate_questions_async_data_all(keywords: list[str], client: AsyncOpenAI, **kwargs):
    knowledge = await asyncio.gather(
        *[genereate_knowledge_record(keyword, client, **kwargs) for keyword in keywords]
    )
    knowledge = list(itertools.chain(*knowledge))
    logger.info("Finished generating Knowledge")
    return knowledge

@hydra.main(config_path="../llm_20q/configs/knowledge", config_name="openai-knowledge", version_base=None)
def main(config: DictConfig):
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, **config.wandb_init)
    artifact = wandb.Artifact(**raw_config['output_artifact'])
    corpus_df = build_corpus()
    keywords = corpus_df["keyword"].unique()
    client = AsyncOpenAI(timeout=60, max_retries=100)
    responses = []
    with asyncio.Runner() as runner:
        responses = runner.run(
            generate_questions_async_data_all(keywords, client, **config.generate_kwargs)
        )
    knowledge_df = pd.DataFrame(responses)
    knowledge_df['prompt'] = knowledge_df['knowledge']
    knowledge_dir = Path("../input/knowledge")
    knowledge_dir.mkdir(exist_ok=True, parents=True)
    file_path = knowledge_dir / config.file_name
    knowledge_df.to_parquet(file_path)
    table = wandb.Table(dataframe=knowledge_df)
    artifact.add_file(str(file_path), name=file_path.name)
    artifact.add(table, "knowledge")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
