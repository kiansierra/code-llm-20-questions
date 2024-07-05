# flake8: noqa: E501
from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import Choice

from .corpus import ALL_KEYWORDS, CATEGORIES

QUESTION_GENERATOR_PROMPT = f"""You are playing the 20 questions game, you're tasked with asking questions to identify the users keyword.
The keyword will belong to one of the following categories: {', '.join(CATEGORIES)}.
Some of the Keywords available are: {', '.join(ALL_KEYWORDS)}."""

KNOWLEDGE_GENERATOR_PROMPT = (
    "You are a knowledge based generator, you will provide all knowledge you have regarding the provided thing/place."
)

ANSWER_GENERATOR_PROMPT = "You are playing the 20 questions game, you're tasked with answering the questions regarding the keyword: {keyword}. You can only answer yes or no"


def generate_questions(
    client: OpenAI, model: str = "gpt-3.5-turbo", question_generator_prompt: str = QUESTION_GENERATOR_PROMPT, **kwargs
) -> list[Choice]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": question_generator_prompt},
            {"role": "user", "content": "Ask me a question."},
        ],
        **kwargs,
    )
    choices = completion.choices
    return choices


async def generate_questions_async(
    client: AsyncOpenAI,
    model: str = "gpt-3.5-turbo",
    question_generator_prompt: str = QUESTION_GENERATOR_PROMPT,
    **kwargs,
) -> list[Choice]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": question_generator_prompt},
            {"role": "user", "content": "Ask me a question."},
        ],
        **kwargs,
    )
    choices = completion.choices
    return choices


async def generate_answers_async(
    client: AsyncOpenAI,
    keyword: str,
    question: str,
    model: str = "gpt-3.5-turbo",
    answer_generator_prompt: str = ANSWER_GENERATOR_PROMPT,
    **kwargs,
) -> list[Choice]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": answer_generator_prompt.format(keyword=keyword)},
            {"role": "user", "content": question},
        ],
        **kwargs,
    )
    choices = completion.choices
    return choices


def generate_knowledge(
    keyword: str,
    client: OpenAI,
    model: str = "gpt-3.5-turbo",
    knowledge_generation_prompt: str = KNOWLEDGE_GENERATOR_PROMPT,
    **kwargs,
) -> list[Choice]:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": knowledge_generation_prompt}, {"role": "user", "content": keyword}],
        **kwargs
    )
    questions = completion.choices
    return questions

async def generate_knowledge_async(
    keyword: str,
    client: AsyncOpenAI,
    model: str = "gpt-3.5-turbo",
    knowledge_generation_prompt: str = KNOWLEDGE_GENERATOR_PROMPT,
    **kwargs,
) -> list[Choice]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": knowledge_generation_prompt}, {"role": "user", "content": keyword}],
        **kwargs
    )
    questions = completion.choices
    return questions
