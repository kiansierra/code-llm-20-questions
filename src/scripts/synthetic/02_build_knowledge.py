import asyncio
import itertools
import logging
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI
from tqdm import tqdm

import wandb

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

KNOWLEDGE_GENERATOR_PROMPT = """
Generate knowledge about the keyword {keyword}, that is relevant to the questions.
Do not provide the answer to the questions.
Provide knowledge that is relevant to answer all the questions.
Summarize the knowledge in a few sentences.
Do not mention the keyword in the knowledge.
"""


async def generate_knowledge_async_data(client: AsyncOpenAI, keyword: str, answer: str, question: str, question_id: set, **kwargs) -> str:
    choices = completion = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": KNOWLEDGE_GENERATOR_PROMPT.format(keyword=keyword)},
            {"role": "user", "content": f"{question}"},
        ],
        **kwargs,
    )
    choices = completion.choices
    return [
        {"knowledge": choice.message.content, "answer": answer, "keyword": keyword, "question": question, "question_ids": question_id}
        for choice in choices
    ]


async def generate_knowledge_async_data_all(records: list[str], client: AsyncOpenAI, batch_size: int = 20, **kwargs) -> pd.DataFrame:
    all_knowledge = []
    for i in tqdm(range(0, len(records), batch_size), desc="Generating Answers"):
        answers = await asyncio.gather(*[generate_knowledge_async_data(client, **elem, **kwargs) for elem in records[i : i + batch_size]])
        all_knowledge.extend(answers)
    all_knowledge = list(itertools.chain(*all_knowledge))
    logger.info("Finished generating Answers")
    return pd.DataFrame(all_knowledge)


def build_prompt(row):
    return f"# Keyword: {row['keyword']} \n# Knowledge {row['knowledge']}"


@hydra.main(config_path="../../llm_20q/configs/openai", config_name="openai-knowledge", version_base=None)
def main(config: DictConfig):
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(config=raw_config, **config.wandb_init)

    artifact = run.use_artifact(**config.input_artifact_answers)
    artifact_dir = artifact.download()
    answers_df = pd.read_parquet(f"{artifact_dir}/{config.input_file_name}")
    answers_df = answers_df.query('answer != "drop"').drop(columns=["raw_answer"])
    answers_df["split"] = answers_df["question_id"] % 5
    grouped_answers_df = answers_df.groupby(["keyword", "answer", "split"], as_index=False).agg({"question": "\n".join, "question_id": set})
    records = grouped_answers_df.drop(columns="split").to_dict("records")

    client = AsyncOpenAI(timeout=60, max_retries=100)
    with asyncio.Runner() as runner:
        knowledge_df = runner.run(generate_knowledge_async_data_all(records, client, **config.generate_kwargs))
    knowledge_dir = Path("../input/knowledge")
    knowledge_dir.mkdir(exist_ok=True, parents=True)
    file_path = knowledge_dir / config.output_file_name
    knowledge_df.to_parquet(file_path)
    artifact = wandb.Artifact(**raw_config["output_artifact"])
    table = wandb.Table(dataframe=knowledge_df)
    artifact.add_file(str(file_path), name=file_path.name)
    artifact.add(table, "knowledge")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
