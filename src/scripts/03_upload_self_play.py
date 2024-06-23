from pathlib import Path

import hydra
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold

import wandb
from llm_20q.data import build_game_records

load_dotenv()

INPUT_DATASET_NAME = "self-play-records"
DATASET_TYPE = "self-play"



def build_folds(df, n_splits=5):
    group_kfold = GroupKFold(n_splits=n_splits)
    groups = df["EpisodeId"]
    df["split"] = "train"
    for num, (_, test_index) in enumerate(group_kfold.split(df, groups=groups)):
        df.loc[test_index, "split"] = "validation" if num == 0 else "train"

    return df



@hydra.main(config_path="../llm_20q/configs", config_name="llama3-8b-inst", version_base=None)
def main(config) -> None:
    model_name = config.model_name
    run = wandb.init(job_type="upload-replay-records")
    record_artifact = run.use_artifact(f"{INPUT_DATASET_NAME}-{model_name}:latest", type=DATASET_TYPE)
    record_dir = record_artifact.download()
    path = Path(record_dir)
    assert list(path.glob("*.json")), f"No game records found in {path} directory."
    game_records = build_game_records(folder=str(path), reward=False)
    print(game_records)

if __name__ == "__main__":
    main()
