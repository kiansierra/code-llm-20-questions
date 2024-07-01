
from openai import OpenAI

QUESTION_GENERATOR_PROMPT = """You are a question generator, you will ask questions about the provided thing/place, for which the anser is yes.
Do not include the thing/place in the questions, do not enumerate the questions"""

KNOWLEDGE_GENERATOR_PROMPT = "You are a knowledge based generator, you will provide all knowledge you have regarding the provided thing/place."

def generate_questions(keyword:str, client:OpenAI, model_id:str="gpt-3.5-turbo", question_generator_prompt:str=QUESTION_GENERATOR_PROMPT):
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": question_generator_prompt},
            {"role": "user", "content": keyword}]
        )
    questions = completion.choices[0].message.content
    return questions

def generate_knowledge(keyword:str, client:OpenAI, model_id:str="gpt-3.5-turbo", knowledge_generation_prompt:str=KNOWLEDGE_GENERATOR_PROMPT):
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": knowledge_generation_prompt},
            {"role": "user", "content": keyword}]
        )
    questions = completion.choices[0].message.content
    return questions
