from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import torch
from transformers import TextGenerationPipeline, pipeline

from llm_20q import (SentenceTransformerRag, prepare_answer_messages,
                     prepare_ask_messages, prepare_guess_messages)
from llm_20q.utils.checkpoints import extract_last_checkpoint
from llm_20q import generate_options
from loguru import logger

class LLM20Q:

    def __init__(self, pipe: TextGenerationPipeline, config:DictConfig, folder: Path):
        self.pipe = pipe
        self.eoq_tokens = pipe.tokenizer.convert_tokens_to_ids(["<|eot_id|>", "?", "?."])
        pipe.model.load_adapter(extract_last_checkpoint(folder / "ask"), adapter_name="ask")
        pipe.model.load_adapter(extract_last_checkpoint(folder / "guess"), adapter_name="guess")
        self.rag = SentenceTransformerRag.from_folder(folder / 'rag')
        self.config = config

    @classmethod
    def from_folder(cls, folder: Path):
        config = OmegaConf.load(folder / "config.yaml")
        pipe = pipeline(**config.pipeline, model=str(folder),  tokenizer=str(folder))
        return cls(pipe, config, folder)

    def ask(self, obs, cfg):
        # if agent is guesser and turnType is "ask"
        self.pipe.model.set_adapter("ask")
        conversation = prepare_ask_messages(obs.questions, obs.answers, obs.guesses)
        text = self.pipe.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        output = self.pipe(text, eos_token_id=self.eoq_tokens)
        response = output[0]['generated_text'].replace(text, '').strip()
        return response

    def guess(self, obs, cfg):
        # if agent is guesser and turnType is "guess"
        self.pipe.model.set_adapter("guess")
        question = obs.questions[-1]
        answer = obs.answers[-1]
        query = f"search_query: {question}"
        kwargs = self.config.yes_kwargs if answer == "yes" else self.config.no_kwargs
        options_df = self.rag.filter(query, **kwargs)
        options = options_df.drop_duplicates(subset='keyword')['keyword'].tolist()[:10]
        conversation = prepare_guess_messages(obs.questions, obs.answers, obs.guesses, options=options)
        options_input_ids = self.pipe.tokenizer(options, add_special_tokens=False).input_ids
        input_ids = self.pipe.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        output = generate_options(self.pipe.model, sequences_ids=options_input_ids, input_ids=input_ids)
        response = self.pipe.tokenizer.decode(output)
        return response

    # if agent is the answerer
    @torch.no_grad()
    def answer(self, obs, cfg):
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
    
    def agent_fn(self, obs, cfg):
        # if agent is guesser and turnType is "ask"
        if obs.turnType == "ask":
            response =  self.ask(obs, cfg)
            logger.info(f"{obs.step=} ask: {response}")
        # if agent is guesser and turnType is "guess"
        elif obs.turnType == "guess":
            response = self.guess(obs, cfg)
            logger.info(f"{obs.step=} guess: {response}")
        # if agent is the answerer
        elif obs.turnType == "answer":
            # pipe.model.disable_adapters()
            response = self.answer(obs, cfg)
            still_available = obs.keyword in self.rag.filter_df.keyword.tolist()
            logger.info(f"{obs.step=} answer: {response} -- keyword: {obs.keyword} {still_available=}")
        return response
