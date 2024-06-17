import itertools
from typing import Literal, Optional

import torch
import weave
from transformers import Conversation, pipeline

AnswerType = Literal["yes", "no"]

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

GUESS_PROMPT = """What is your guess? """


def prepare_guess_messages(
    questions: list[str],
    answers: list[AnswerType],
    guesses: list[str],
    guess_system_prompt: str = GUESS_SYSTEM_PROMPT,
    guess_prompt: str = GUESS_PROMPT,
    uuid: Optional[str] = None,
):
    """
    Prepares a conversation with messages for the guessing system.

    Args:
        questions (list[str]): List of questions asked by the user.
        answers (list[str]): List of answers provided by the assistant.
        guesses (list[str]): List of guesses made by the user.
        guess_system_prompt (str, optional): System prompt for the guessing system. Defaults to GUESS_SYSTEM_PROMPT.
        guess_prompt (str, optional): Prompt for the user to make a guess. Defaults to GUESS_PROMPT.
        uuid (str, optional): Unique identifier for the conversation. Defaults to None.

    Returns:
        Conversation: A Conversation object containing the prepared messages.

    """

    messages = [{"role": "system", "content": guess_system_prompt}]
    for question, answer, guess in itertools.zip_longest(questions, answers, guesses):
        messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
        else:
            messages.append({"role": "assistant", "content": ""})
        if guess:
            messages.append({"role": "user", "content": guess_prompt})
            messages.append({"role": "assistant", "content": guess})
    return Conversation(messages=messages, conversation_id=uuid)


def prepare_answer_messages(
    keyword: str,
    category: str,
    questions: list[str],
    answers: list[AnswerType],
    answer_system_prompt: str = ANSWER_SYSTEM_PROMPT,
    uuid: Optional[str] = None,
):
    """
    Prepares a conversation with messages for answering a specific keyword.

    Args:
        keyword (str): The keyword for which the answers are prepared.
        questions (list[str]): A list of questions related to the keyword.
        answers (list[str]): A list of answers corresponding to the questions.
        answer_system_prompt (str, optional): The system prompt for the answer. Defaults to ANSWER_SYSTEM_PROMPT.
        uuid (str, optional): The unique identifier for the conversation. Defaults to None.

    Returns:
        Conversation: A Conversation object containing the prepared messages.
    """

    messages = [{"role": "system", "content": answer_system_prompt.format(keyword=keyword, category=category)}]
    for question, answer in itertools.zip_longest(questions, answers):
        messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return Conversation(messages=messages, conversation_id=uuid)


def prepare_ask_messages(
    questions: list[str],
    answers: list[AnswerType],
    guesses: list[str],
    question_system_prompt: str = QUESTION_SYSTEM_PROMPT,
    uuid: Optional[str] = None,
) -> Conversation:
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
        uuid (str, optional): The unique identifier for the conversation. Defaults to None.

    Returns:
        Conversation: A conversation object containing the messages exchanged between the assistant and the user.
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

    return Conversation(messages=messages, conversation_id=uuid)


class LLMQA(weave.Model):
    """
    LLMQA is a conversational question/answering/guess model.

    Attributes:
        model_id (str): The ID of the underlying conversational model.
        answer_system_prompt (str): The system prompt for generating answers.
        question_system_prompt (str): The system prompt for generating questions.
        guess_system_prompt (str): The system prompt for generating guesses.
        guess_prompt (str): The prompt for generating guesses.

    Methods:
        model_post_init(__context): Initializes the LLMQA model.
        ask(questions, answers, guesses=None, uuid=None, **kwargs): Generates a response based on the given questions, answers, and optional guesses.
        answer(keyword, questions, answers, guesses=None, uuid=None, **kwargs): Generates an answer based on the given keyword, questions, and answers.
        guess(questions, answers, guesses, uuid=None, **kwargs): Generates a guess based on the given questions, answers, and previous guesses.
    """

    model_id: str
    answer_system_prompt: str = ANSWER_SYSTEM_PROMPT
    question_system_prompt: str = QUESTION_SYSTEM_PROMPT
    guess_system_prompt: str = GUESS_SYSTEM_PROMPT
    guess_prompt: str = GUESS_PROMPT

    def model_post_init(self, __context):
        """
        Initializes the LLMQA model by setting up the pipeline and tokenizer.

        Args:
            __context: The context object.

        Returns:
            None
        """
        self._pipeline = pipeline(
            "conversational", model=self.model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        self._tokenizer = self._pipeline.tokenizer
        self._terminators = [self._tokenizer.eos_token_id, self._tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self._yes_token_id = self._tokenizer.vocab["yes"]
        self._no_token_id = self._tokenizer.vocab["no"]

    @weave.op()
    def ask(
        self,
        questions: list[str],
        answers: list[str],
        guesses: Optional[list[str]] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generates a response based on the given questions, answers, and optional guesses.

        Args:
            questions (list[str]): A list of questions.
            answers (list[str]): A list of corresponding answers to the questions.
            guesses (list[str], optional): A list of optional guesses. Defaults to None.
            uuid (str, optional): An optional UUID. Defaults to None.
            **kwargs: Additional keyword for generation arguments.

        Returns:
            str: The generated Question.

        """
        conversation = prepare_ask_messages(
            questions,
            answers,
            guesses,
            question_system_prompt=self.question_system_prompt,
            uuid=uuid,
        )
        output = self._pipeline(conversation, eos_token_id=self._terminators, **kwargs)
        return output.messages[-1]["content"]

    @weave.op()
    def answer(
        self,
        keyword: str,
        questions: list[str],
        answers: list[str],
        guesses: Optional[list[str]] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> AnswerType:
        """
        Generates an answer based on the given keyword, questions, and answers.

        Args:
            keyword (str): The keyword to generate an answer for.
            questions (list[str]): A list of questions.
            answers (list[str]): A list of corresponding answers.
            guesses (list[str], optional): A list of guesses. Defaults to None.
            uuid (str, optional): A unique identifier. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            AnswerType: The generated answer.

        Raises:
            None

        Examples:
            >>> model = Model()
            >>> keyword = "cat"
            >>> questions = ["Is it a mammal?", "Does it have fur?"]
            >>> answers = ["yes"]
            >>> model.answer(keyword, questions, answers)
            'yes'
        """
        conversation = prepare_answer_messages(
            keyword, questions, answers, answer_system_prompt=self.answer_system_prompt, uuid=uuid
        )
        input_ids = self._tokenizer.apply_chat_template(conversation.messages, tokenize=True, return_tensors="pt")
        input_ids = input_ids[:, :-1]
        output = self._pipeline.model(input_ids.to(self._pipeline.device))
        answer = output.logits[0, -1, [self._yes_token_id, self._no_token_id]].argmax()
        return ["yes", "no"][answer]

    @weave.op()
    def guess(
        self, questions: list[str], answers: list[str], guesses: list[str], uuid: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generates a guess based on the given questions, answers, and previous guesses.

        Args:
            questions (list[str]): A list of questions asked during the conversation.
            answers (list[str]): A list of answers corresponding to the questions.
            guesses (list[str]): A list of previous guesses made during the conversation.
            uuid (Optional[str], optional): A unique identifier for the conversation. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the underlying pipeline.

        Returns:
            str: The generated guess.

        """
        conversation = prepare_guess_messages(
            questions,
            answers,
            guesses,
            guess_system_prompt=self.guess_system_prompt,
            guess_prompt=self.guess_prompt,
            uuid=uuid,
        )
        output = self._pipeline(conversation, eos_token_id=self._terminators, **kwargs)
        return output.messages[-1]["content"]
