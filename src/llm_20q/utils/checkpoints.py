from pathlib import Path

def extract_last_checkpoint(checkpoint_dir: Path) -> Path:
    """Extract the last checkpoint from a directory."""
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise ValueError("No checkpoints found in directory")
    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.stem.split("-")[1]))
    return sorted_checkpoints[-1]