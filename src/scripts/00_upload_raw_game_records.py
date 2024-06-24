from pathlib import Path

import wandb

DATASET_TYPE = "replay-games"
DATASET_NAME = "replay-records"


def main() -> None:
    path = Path("../input/games")
    assert list(path.glob("*.json")), f"No game records found in {path} directory."
    run = wandb.init(job_type="upload-replay-records")
    artifact = wandb.Artifact(DATASET_NAME, type=DATASET_TYPE)
    artifact.add_dir(str(path))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
