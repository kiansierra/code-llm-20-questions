# %%
from kaggle_environments import make
from transformers import  pipeline, BitsAndBytesConfig
from llm_20q.model import prepare_ask_messages, prepare_answer_messages, prepare_guess_messages
import torch
from llm_20_questions.keywords import KEYWORDS_JSON
import json
import pandas as pd 
import numpy as np
import string
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
# %%
keywords = eval(KEYWORDS_JSON)

# %%
dfs = []
for elem in keywords:
    df = pd.DataFrame(elem['words'])
    df['category'] = elem['category']
    dfs.append(df)
keyword_df = pd.concat(dfs)
categories = keyword_df['category'].unique()
keyword_df = keyword_df.explode('alts')
keyword_df['alts'] = keyword_df['alts'].fillna('')
keyword_df['letter'] = keyword_df['keyword'].apply(lambda x:x[0])
keyword_df['prompt'] = keyword_df.apply(lambda x: "category: {category}, keyword: {keyword}".format(**x) + (f" alts: {x['alts']}" if x['alts']  else ''), axis=1)

# %%
CATEGORY_BASE_QUESTIONS = "Is the Keyword a {category}? {answer}"
category_questions_positive = [{'category':category, 'question': CATEGORY_BASE_QUESTIONS.format(category=category, answer='yes'), 'score': 1} for category in categories ]
category_questions_negative = [{'category':category, 'question': CATEGORY_BASE_QUESTIONS.format(category=category, answer='no'), 'score': -1} for category in categories ]
keyword_questions = pd.DataFrame(category_questions_positive + category_questions_negative)

# %%
keyword_questions_df = keyword_questions.merge(keyword_df, on='category')
keyword_questions_df.head()

# %%
LETTER_BASE_QUESTIONS = "Does the Keyword start with {letter}? {answer}"
letter_questions_positive = [{'letter':letter, 'question': LETTER_BASE_QUESTIONS.format(letter=letter, answer='yes'), 'score': 1} for letter in string.ascii_lowercase ]
letter_questions_negative = [{'letter':letter, 'question': LETTER_BASE_QUESTIONS.format(letter=letter, answer='no'), 'score': -1} for letter in string.ascii_lowercase ]
letter_questions = pd.DataFrame(letter_questions_positive + letter_questions_negative)

# %%
letter_questions_df = letter_questions.merge(keyword_df, on='letter')
letter_questions_df.head()

# %%
all_questions_df = pd.concat([keyword_questions_df, letter_questions_df], ignore_index=True)

# %%
dataset = Dataset.from_pandas(all_questions_df[['question', 'prompt', 'score']])

# %%

# Load a model to train/finetune
model = SentenceTransformer("xlm-roberta-base")

# Initialize the CoSENTLoss
# This loss requires pairs of text and a float similarity score as a label
loss = CoSENTLoss(model)

# %%
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
)

# %%
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=loss,
)
trainer.train()

# %%



