import asyncio
import itertools
import logging
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from tqdm import tqdm

import wandb
from llm_20q.data import ALL_KEYWORDS, QUESTION_GENERATOR_PROMPT, USER_QUESTION_GENERATING_PROMPT

load_dotenv()

# Only show warnings
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def sanitize_question(keyword: str, question: str) -> str:
    question = question.replace(keyword, "keyword")
    question = question.replace(keyword.lower(), "keyword")
    question = question.strip()
    return question


async def generate_questions_async_data(keyword: str, client: AsyncOpenAI, **kwargs) -> str:
    completion = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": QUESTION_GENERATOR_PROMPT},
            {"role": "user", "content": USER_QUESTION_GENERATING_PROMPT.format(keyword=keyword)},
        ],
        **kwargs,
    )
    choices = completion.choices
    return [{"question": sanitize_question(keyword, choice.message.content), "keyword": keyword} for choice in choices]


async def generate_questions_async_data_all(keywords: list[str], client: AsyncOpenAI, batch_size: int = 10, **kwargs):
    all_questions = []
    for i in tqdm(range(0, len(keywords), batch_size), desc="Generating Questions"):
        questions = await asyncio.gather(
            *[generate_questions_async_data(keyword, client, **kwargs) for keyword in keywords[i : i + batch_size]]
        )
        all_questions.extend(questions)
    all_questions = list(itertools.chain(*all_questions))
    logger.info("Finished generating questions")
    return all_questions


@hydra.main(config_path="../../llm_20q/configs/openai", config_name="openai-questions", version_base=None)
def main(config):
    client = AsyncOpenAI(timeout=60, max_retries=100)
    with asyncio.Runner() as runner:
        responses = runner.run(generate_questions_async_data_all(ALL_KEYWORDS, client, **config.generate_kwargs))
    questions_df = pd.DataFrame(responses)
    questions_df["question"] = questions_df["question"].str.split("\n")
    questions_df = questions_df.explode("question")
    questions_df.drop_duplicates(subset=["question"], inplace=True)
    questions_df.reset_index(drop=True, inplace=True)
    logger.info(f"Generated {len(questions_df)} Unique questions")
    questions_dir = Path("../input/questions")
    questions_dir.mkdir(exist_ok=True, parents=True)
    file_path = questions_dir / config.output_file_name
    questions_df.to_parquet(file_path)
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(**config.wandb_init, config=raw_config)
    artifact = wandb.Artifact(**raw_config["output_artifact"])
    artifact.add_file(str(file_path), name=file_path.name)
    table = wandb.Table(dataframe=questions_df)
    run.log({"questions": table})
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
