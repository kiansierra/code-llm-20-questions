from transformers import PreTrainedTokenizer

from ..types import TaskType
from .prompt_templates import prepare_answer_messages, prepare_ask_messages, prepare_guess_messages

__all__ = ["generate_prompt"]


def generate_prompt(tokenizer: PreTrainedTokenizer, task: TaskType, max_options: int = 10):

    def make_question_row(row):
        data = {
            "questions": list(row["questions"]) + [row["question"]],
            "answers": row["answers"],
            "guesses": row["guesses"],
        }
        conversation = prepare_ask_messages(**data)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        return prompt

    def make_answer_row(row):
        data = {
            "questions": row["questions"],
            "answers": list(row["answers"]) + [row["answer"]],
            "keyword": row["keyword"],
            "category": row["category"],
        }
        conversation = prepare_answer_messages(**data)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        return prompt

    def make_guess_row(row):
        data = {
            "questions": row["questions"],
            "answers": row["answers"],
            "guess": row["guess"],
        }
        if "options" in row:
            all_options = row["options"][:max_options] + [row["guess"]]
            data["options"] = all_options
        conversation = prepare_guess_messages(**data)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        return prompt

    match task:
        case "ask":
            return make_question_row
        case "answer":
            return make_answer_row
        case "guess":
            return make_guess_row
        case _:
            raise ValueError(f"Invalid task type: {task}")
