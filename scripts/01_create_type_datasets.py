import wandb
from dotenv import load_dotenv

INPUT_DATASET_NAME = "raw_games"
OUTPUT_DATASET_NAME = "games"
DATASET_TYPE = "game_records"

def main() -> None:
    run = wandb.init(job_type="create_game_datasets")
    artifact = run.use_artifact(f'{INPUT_DATASET_NAME}:latest', type=DATASET_TYPE)
    artifact_dir = artifact.download()
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type=DATASET_TYPE)
    artifact.add_dir("./data/games")
    run.log_artifact(artifact)
    run.finish()
    
    
if __name__ == "__main__":
    main()