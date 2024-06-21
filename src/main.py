from pathlib import Path
import sys
current_folder = Path(__file__).parent
sys.path.append(str((current_folder/ 'libs').absolute()))
from transformers import  pipeline, BitsAndBytesConfig
from llm_20q.model import prepare_ask_messages, prepare_answer_messages, prepare_guess_messages
import torch

quantization_config = BitsAndBytesConfig(**{
    'load_in_4bit': True,
    'bnb_4bit_quant_type': "nf4",
    'bnb_4bit_compute_dtype': torch.bfloat16,
    'bnb_4bit_use_double_quant': True
})

model_dir = (current_folder / 'model').absolute()
pipe = pipeline(
            "conversational",
            model=model_dir,
            model_kwargs={"torch_dtype": torch.bfloat16,
                          'quantization_config': quantization_config},
            device_map="auto",
        )
pipe.model.load_adapter("{model_dir}/ask/checkpoint-80", adapter_name="ask")
pipe.model.load_adapter("{model_dir}/guess/checkpoint-70", adapter_name="guess")

ask_terminators = [pipe.tokenizer.eos_token_id,
               *pipe.tokenizer.convert_tokens_to_ids(["<|eot_id|>", "?", "?."])]

def llm_agent(obs, cfg):
    # if agent is guesser and turnType is "ask"
    if obs.turnType == "ask":
        pipe.model.set_adapter('ask')
        conversation = prepare_ask_messages(obs.questions, obs.answers, obs.guesses)
        output = pipe(conversation, eos_token_id=ask_terminators)
        response = output[-1]['content']
    # if agent is guesser and turnType is "guess"
    elif obs.turnType == "guess":
        pipe.model.set_adapter('guess')
        conversation = prepare_guess_messages(obs.questions, obs.answers, obs.guesses)
        output = pipe(conversation)
        response = output[-1]['content']
    # if agent is the answerer
    elif obs.turnType == "answer":
        pipe.model.disable_adapters()
        yesno_words = ["yes", "no"]
        yes_no_ids = pipe.tokenizer.convert_tokens_to_ids(yesno_words)
        conversation = prepare_answer_messages(keyword=obs['keyword'],
                                               category=obs['category'],
                                               questions=obs.questions, answers=obs.answers)
        input_ids = pipe.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            logits = pipe.model(input_ids).logits
        position = logits[0, -1, yes_no_ids].argmax()
        response = yesno_words[position]
    return response