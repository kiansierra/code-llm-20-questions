# flake8: noqa: E402
import importlib
import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "false"
os.environ["USE_FLAX"] = "false"

agent_path = Path("/kaggle_simulations/agent")

if agent_path.exists():
    lib_path = agent_path / "libs"
    sys.path.insert(0, str(lib_path.absolute()))

else:
    agent_path = Path(__file__).parent
    lib_path = agent_path / "libs"
    sys.path.insert(0, str((agent_path / "libs").absolute()))

importlib.invalidate_caches()

import torch
import transformers

transformers.utils.import_utils._bitsandbytes_available = True
transformers.utils.import_utils._peft_available = True
importlib.reload(transformers.modeling_utils)

from llm_20q.model import prepare_answer_messages, prepare_ask_messages, prepare_guess_messages
from llm_20q.utils.checkpoints import extract_last_checkpoint

# print("Loaded Imports succesfully")
# print(f"{peft.__version__=}")


model_dir = (agent_path / "model").absolute()
pipe = transformers.pipeline(
    "conversational",
    model=model_dir,
    device_map="auto",
)
print("Loaded Model succesfully")
pipe.model.load_adapter(extract_last_checkpoint(model_dir / "ask"), adapter_name="ask")
pipe.model.load_adapter(extract_last_checkpoint(model_dir / "guess"), adapter_name="guess")
print("Loaded Adapters succesfully")
ask_terminators = [pipe.tokenizer.eos_token_id, *pipe.tokenizer.convert_tokens_to_ids(["<|eot_id|>", "?", "?."])]


def agent_fn(obs, cfg):
    # if agent is guesser and turnType is "ask"
    if obs.turnType == "ask":
        pipe.model.set_adapter("ask")
        conversation = prepare_ask_messages(obs.questions, obs.answers, obs.guesses)
        output = pipe(conversation, eos_token_id=ask_terminators)
        response = output[-1]["content"]
    # if agent is guesser and turnType is "guess"
    elif obs.turnType == "guess":
        pipe.model.set_adapter("guess")
        conversation = prepare_guess_messages(obs.questions, obs.answers, obs.guesses)
        output = pipe(conversation)
        response = output[-1]["content"]
    # if agent is the answerer
    elif obs.turnType == "answer":
        # pipe.model.disable_adapters()
        yesno_words = ["yes", "no"]
        yes_no_ids = pipe.tokenizer.convert_tokens_to_ids(yesno_words)
        conversation = prepare_answer_messages(
            keyword=obs["keyword"], category=obs["category"], questions=obs.questions, answers=obs.answers
        )
        input_ids = pipe.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = pipe.model(input_ids).logits
        position = logits[0, -1, yes_no_ids].argmax()
        response = yesno_words[position]
    return response
