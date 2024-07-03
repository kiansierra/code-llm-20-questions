
from typing import Literal
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import Choice
from .corpus import CATEGORIES, ALL_KEYWORDS


QUESTION_GENERATOR_PROMPT = f"""You are playing the 20 questions game, you're tasked with asking questions to identify the users keyword.
The keyword will belong to one of the following categories: {', '.join(CATEGORIES)}.
Some of the Keywords available are: {', '.join(ALL_KEYWORDS)}."""

KNOWLEDGE_GENERATOR_PROMPT = "You are a knowledge based generator, you will provide all knowledge you have regarding the provided thing/place."



def generate_questions(client:OpenAI, model:str="gpt-3.5-turbo", question_generator_prompt:str=QUESTION_GENERATOR_PROMPT, **kwargs) -> list[Choice]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": question_generator_prompt},
            {"role": "user", "content": "Ask me a question."}],
        **kwargs
        )
    choices = completion.choices
    return choices

async def generate_questions_async(client:AsyncOpenAI, model:str="gpt-3.5-turbo",question_generator_prompt:str=QUESTION_GENERATOR_PROMPT, **kwargs) -> list[Choice]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
             {"role": "system", "content": question_generator_prompt},
             {"role": "user", "content": "Ask me a question."}],
        **kwargs

        )
    choices = completion.choices
    return choices

def generate_knowledge(keyword:str, client:OpenAI, model:str="gpt-3.5-turbo", knowledge_generation_prompt:str=KNOWLEDGE_GENERATOR_PROMPT):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": knowledge_generation_prompt},
            {"role": "user", "content": keyword}]
        )
    questions = completion.choices[0].message.content
    return questions
