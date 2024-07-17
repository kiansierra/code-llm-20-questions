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
from llm_20q.data import ANSWER_GENERATOR_PROMPT, build_corpus
from llm_20q.processors.question_classifier import QuestionClassifier, SimilarityFilterer

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def sanitize_answer(answer: str) -> str:
    answer = answer.strip().lower()
    if "yes" in answer and "no" in answer:
        return "drop"
    if "yes" in answer:
        return "yes"
    if "no" in answer:
        return "no"
    return "drop"


async def generate_answers_async_data(client: AsyncOpenAI, keyword: str, question: str, question_id: str, **kwargs) -> str:
    choices = completion = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": ANSWER_GENERATOR_PROMPT.format(keyword=keyword)},
            {"role": "user", "content": question},
        ],
        **kwargs,
    )
    choices = completion.choices
    return [
        {
            "answer": sanitize_answer(choice.message.content),
            "raw_answer": choice.message.content,
            "keyword": keyword,
            "question": question,
            "question_id": question_id,
        }
        for choice in choices
    ]


async def generate_answers_async_data_all(question_pairs: list[str], client: AsyncOpenAI, batch_size: int = 15, **kwargs) -> pd.DataFrame:
    all_answers = []
    for i in tqdm(range(0, len(question_pairs), batch_size), desc="Generating Answers"):
        answers = await asyncio.gather(
            *[generate_answers_async_data(client, **elem, **kwargs) for elem in question_pairs[i : i + batch_size]]
        )
        all_answers.extend(answers)
    all_answers = list(itertools.chain(*all_answers))
    logger.info("Finished generating Answers")
    return pd.DataFrame(all_answers)


@hydra.main(config_path="../../llm_20q/configs/openai", config_name="openai-answers", version_base=None)
def main(config):
    corpus_df = build_corpus()
    raw_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(**config.wandb_init, config=raw_config)
    artifact = run.use_artifact(**config.input_artifact)
    artifact_dir = Path(artifact.download())
    artifact_file = artifact_dir / config.input_file_name
    questions_df = pd.read_parquet(artifact_file)
    questions_df = questions_df.query("question.str.len() >= 10").reset_index(drop=True)
    questions_df = questions_df.query("question.str.endswith('?')").reset_index(drop=True)
    # Fiter out questions that are of bad quality
    with QuestionClassifier(**config.question_classifier) as qclassifier:
        drop_idx = qclassifier.filter(questions_df["question"].tolist())
    logger.info(f"Removed {len(drop_idx)=} questions for bad quality")
    questions_df = questions_df.drop(drop_idx).reset_index(drop=True)
    # Fiter out questions that are too similar
    with SimilarityFilterer(**config.similarity_filter) as similarity_filterer:
        drop_idx = similarity_filterer.filter(questions_df["question"].tolist())
    logger.info(f"Removed {len(drop_idx)=} questions for similarity")
    questions_df = questions_df.drop(drop_idx).reset_index(drop=True)
    questions_df = questions_df.reset_index()
    questions_df.rename(columns={"index": "question_id", "keyword": "question_keyword"}, inplace=True)
    questions_df["key"] = 0
    corpus_df["key"] = 0
    question_corpus_df = corpus_df.merge(questions_df, on="key", how="outer").drop("key", axis=1)
    question_corpus_df = question_corpus_df.groupby("question_id").sample(
        n=config.num_samples, replace=False, random_state=config.random_state
    )
    keep_cols = ["keyword", "question", "question_id"]
    question_corpus_df = question_corpus_df[keep_cols]
    questions_df = questions_df.rename(columns={"question_keyword": "keyword"})[keep_cols]
    questions_df = pd.concat([questions_df, question_corpus_df], ignore_index=True)
    records = questions_df[keep_cols].to_dict("records")
    logger.info(f"{len(records)=}")
    client = AsyncOpenAI(timeout=60, max_retries=100)
    with asyncio.Runner() as runner:
        answers_df = runner.run(generate_answers_async_data_all(records, client, **config.generate_kwargs))
    answers_df.reset_index(drop=True, inplace=True)
    logger.info(f"Generated {len(answers_df)=} Unique questions")
    answers_dir = Path("../input/answers")
    answers_dir.mkdir(exist_ok=True, parents=True)
    file_path = answers_dir / config.output_file_name
    answers_df.to_parquet(file_path)

    artifact = wandb.Artifact(**raw_config["output_artifact"])
    artifact.add_file(str(file_path), name=file_path.name)
    table = wandb.Table(dataframe=answers_df)
    run.log({"answers": table})
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
