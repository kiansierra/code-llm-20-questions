from llm_20q.data import generate_knowledge, build_corpus
import pandas as pd
from pathlib import Path
import wandb
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

DATASET_NAME = "base-knowledge"
DATASET_TYPE = "knowledge-dataset"

def main():
    corpus_df = build_corpus()
    keywords = corpus_df['keyword'].unique()
    client = OpenAI()
    responses = []
    for keyword in tqdm(keywords):
        answer = generate_knowledge(keyword, client)
        responses.append({'keyword': keyword, 'knowledge': answer})
    knowledge_df = pd.DataFrame(responses)
    knowledge_dir = Path('../input/knowledge')
    knowledge_dir.mkdir(exist_ok=True, parents=True)
    knowledge_df.to_parquet(knowledge_dir/'base.parquet')
    run = wandb.init(job_type="upload-knowledge")
    artifact = wandb.Artifact(DATASET_NAME, type=DATASET_TYPE)
    artifact.add_dir(str(knowledge_dir))
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == '__main__':
    main()
    
    
