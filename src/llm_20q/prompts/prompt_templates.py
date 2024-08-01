import itertools
from typing import Optional

from ..types import AnswerType, ConversationType

ANSWER_SYSTEM_PROMPT = """You are playing the 20 questions game, you are in charge of responding to the user's questions.
The user will try to guess the object you're thinking of by asking yes/no questions.
You can only respond with 'yes' or 'no'.
The word they are trying to guess is: {keyword}.
The word they are trying to guess is of the following category: {category}."""

QUESTION_SYSTEM_PROMPT = """You are playing the 20 questions game, you're in charge of asking the questions.
Your objective is to guess the keyword, the user can only answer yes/no to the questions.
All keywords belong to one of the following categories: person, place, or thing.
After each question you get to submit one guess, do not submit the guess in the response.
Ask the question directly, here are some examples:
'Is it an animal?'.
'Is the keyword something that can be held in your hand?'.
'Is it a living thing?'.
'Is it something you can eat?'.
'Is it a place?'.
"""

GUESS_SYSTEM_PROMPT = """You are playing the 20 questions game, you're in charge of submitting the guesses.
You're role is to guess the keyword the other player is thinking of, based on the questions they answer.
All keywords belong to one of the following categories: person, place, or thing.
You're guesses should have the format of a single word or concept
Here are some examples:
Question: 'Is it a person?'.
Answer: 'yes'
Guess: 'nelson mandela'
"""

GUESS_PROMPT = """What is your guess?"""

GUESS_PROMPT_OPTIONS = """What is your guess? Chose one of the following options: {options}"""

__all__ = ["prepare_guess_messages", "prepare_answer_messages", "prepare_ask_messages"]


def prepare_guess_messages(
    questions: list[str],
    answers: list[AnswerType],
    guess: Optional[str] = None,
    options: Optional[list[str]] = None,
    guess_system_prompt: str = GUESS_SYSTEM_PROMPT,
    guess_prompt: str = GUESS_PROMPT,
    guess_prompt_options: str = GUESS_PROMPT_OPTIONS,
) -> ConversationType:
    """
    Prepares a conversation with messages for the guessing system.

    Args:
        questions (list[str]): List of questions asked by the user.
        answers (list[str]): List of answers provided by the assistant.
        guesses (list[str]): List of guesses made by the user.
        guess_system_prompt (str, optional): System prompt for the guessing system. Defaults to GUESS_SYSTEM_PROMPT.
        guess_prompt (str, optional): Prompt for the user to make a guess. Defaults to GUESS_PROMPT.

    Returns:
        Conversation: A ConversationType object containing the prepared messages.

    """

    messages = [{"role": "system", "content": guess_system_prompt}]
    assert len(questions) == len(answers), "Questions and answers must have the same length"
    for question, answer in zip(questions, answers):
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    guess_prompt_message = guess_prompt if options is None else guess_prompt_options.format(options=", ".join(options))
    messages.append({"role": "user", "content": guess_prompt_message})
    if guess:
        messages.append({"role": "assistant", "content": guess})
    return messages


def prepare_answer_messages(
    keyword: str,
    category: str,
    questions: list[str],
    answers: list[AnswerType],
    answer_system_prompt: str = ANSWER_SYSTEM_PROMPT,
):
    """
    Prepares a conversation with messages for answering a specific keyword.

    Args:
        keyword (str): The keyword for which the answers are prepared.
        category (str): The category of the keyword.
        questions (list[str]): A list of questions related to the keyword.
        answers (list[str]): A list of answers corresponding to the questions.
        answer_system_prompt (str, optional): The system prompt for the answer. Defaults to ANSWER_SYSTEM_PROMPT.
        uuid (str, optional): The unique identifier for the conversation. Defaults to None.

    Returns:
        Conversation: A ConversationType object containing the prepared messages.
    """

    messages = [{"role": "system", "content": answer_system_prompt.format(keyword=keyword, category=category)}]
    for question, answer in itertools.zip_longest(questions, answers):
        messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages


def prepare_ask_messages(
    questions: list[str],
    answers: list[AnswerType],
    guesses: list[str],
    question_system_prompt: str = QUESTION_SYSTEM_PROMPT,
) -> ConversationType:
    """
    Prepares a conversation object for the "ask" phase of the 20 Questions game.

    Args:
        questions (list[str]): A list of questions asked by the assistant.
        answers (list[str]): A list of answers provided by the user.
        guesses (list[str]): A list of guesses made by the user.
        question_system_prompt (str, optional): The system prompt for asking questions.
            Defaults to QUESTION_SYSTEM_PROMPT.
        guess_prompt (str, optional): The prompt for asking the user to make a guess.
            Defaults to GUESS_PROMPT.

    Returns:
        Conversation: A ConversationType containing the messages exchanged between the assistant and the user.
    """

    messages = [
        {"role": "system", "content": question_system_prompt},
    ]
    messages.append({"role": "user", "content": "You may ask your first question."})
    for question, answer, guess in itertools.zip_longest(questions, answers, guesses):
        messages.append({"role": "assistant", "content": question})
        if answer:
            messages.append({"role": "user", "content": f"{answer}. Take a Guess."})
        if guess:
            messages.append({"role": "assistant", "content": guess})
            messages.append({"role": "user", "content": "Please ask you're next question."})

    return messages
