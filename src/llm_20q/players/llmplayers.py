from pathlib import Path
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TextGenerationPipeline, pipeline

from ..utils import generate_options, extract_last_checkpoint
from ..prompts import prepare_answer_messages, prepare_ask_messages, prepare_guess_messages
from ..types import AnswerType, Observation


class LLMPlayer:

    def __init__(self, pipe: TextGenerationPipeline, config: DictConfig, folder: Path):
        self.pipe = pipe
        self.eoq_tokens = pipe.tokenizer.convert_tokens_to_ids(["<|eot_id|>", "?", "?."])
        pipe.model.load_adapter(extract_last_checkpoint(folder / "ask"), adapter_name="ask")
        pipe.model.load_adapter(extract_last_checkpoint(folder / "guess"), adapter_name="guess")
        self.config = config

    @classmethod
    def from_folder(cls, folder: Path):
        config = OmegaConf.load(folder / "config.yaml")
        pipe = pipeline(**config.pipeline, model=str(folder), tokenizer=str(folder))
        return cls(pipe, config, folder)

    def ask(self, obs: Observation) -> str:
        # if agent is guesser and turnType is "ask"
        self.pipe.model.set_adapter("ask")
        conversation = prepare_ask_messages(obs.questions, obs.answers, obs.guesses)
        text = self.pipe.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        output = self.pipe(text, eos_token_id=self.eoq_tokens)
        response = output[0]["generated_text"].replace(text, "").strip()
        return response

    def guess(self, obs: Observation, options: Optional[List[str]] = None) -> str:
        # if agent is guesser and turnType is "guess"
        self.pipe.model.set_adapter("guess")
        conversation = prepare_guess_messages(obs.questions, obs.answers, obs.guesses, options=options)
        options_input_ids = self.pipe.tokenizer(options, add_special_tokens=False).input_ids
        input_ids = self.pipe.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        output = generate_options(self.pipe.model, sequences_ids=options_input_ids, input_ids=input_ids)
        response = self.pipe.tokenizer.decode(output)
        return response

    # if agent is the answerer
    @torch.no_grad()
    def answer(self, obs: Observation) -> AnswerType:
        pipe = self.pipe
        pipe.model.disable_adapters()
        yesno_words = ["yes", "no"]
        yes_no_ids = pipe.tokenizer.convert_tokens_to_ids(yesno_words)
        conversation = prepare_answer_messages(
            keyword=obs["keyword"], category=obs["category"], questions=obs.questions, answers=obs.answers
        )
        input_ids = pipe.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        logits = pipe.model(input_ids).logits
        position = logits[0, -1, yes_no_ids].argmax()
        response = yesno_words[position]
        return response
