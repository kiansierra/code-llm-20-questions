import wandb
from dotenv import load_dotenv
from pathlib import Path

DATASET_NAME = "raw_games"
DATASET_TYPE = "game_records"

def main() -> None:
    path = Path('../input/games')
    assert list(path.glob('*.json')), f"No game records found in {path} directory."
    run = wandb.init(job_type="upload_game_records")
    artifact = wandb.Artifact(DATASET_NAME, type=DATASET_TYPE)
    artifact.add_dir(str(path))
    run.log_artifact(artifact)
    run.finish()
    
    
if __name__ == "__main__":
    main()