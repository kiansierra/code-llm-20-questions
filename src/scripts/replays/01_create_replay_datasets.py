from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold

import wandb
from llm_20q.data import build_df, build_game_records

load_dotenv()

INPUT_DATASET_NAME = "replay-records"
OUTPUT_DATASET_NAME = "replay-dataframe"
DATASET_TYPE = "replay-games"


def build_folds(df, n_splits=5):
    group_kfold = GroupKFold(n_splits=n_splits)
    groups = df["EpisodeId"]
    df["split"] = "train"
    for num, (_, test_index) in enumerate(group_kfold.split(df, groups=groups)):
        df.loc[test_index, "split"] = "validation" if num == 0 else "train"

    return df


def main() -> None:
    run = wandb.init(job_type="create-game-datasets")
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest", type=DATASET_TYPE)
    artifact_dir = artifact.download()
    game_records = build_game_records(folder=artifact_dir)
    save_folder = Path("../input/replay-dataframes")
    save_folder.mkdir(exist_ok=True, parents=True)
    for task in ["ask", "answer", "guess"]:
        task_df = build_df(game_records, task)
        task_df = build_folds(task_df)
        logger.info(f"{task=} dataset size {len(task_df)=}")
        save_path = save_folder / f"{task}.parquet"
        task_df.to_parquet(save_path)
        artifact = wandb.Artifact(f"{OUTPUT_DATASET_NAME}-{task}", type=DATASET_TYPE)
        artifact.add_file(save_path.absolute(), f"{task}.parquet")
        run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
