import instructor
from pydantic import BaseModel, field_validator, model_validator
from openai import OpenAI
from typing_extensions import Self
from ..prompts import prepare_ask_messages, prepare_guess_messages, prepare_answer_messages
from ..types import AnswerType

# Define your desired output structure
class Answer(BaseModel):
    answer: AnswerType
    
# Define your desired output structure
class Guess(BaseModel):
    guess: str
    # options: list[str]
    
    # @model_validator(mode='after')
    # def check_guess_in_options(self) -> Self:
    #     if self.guess not in self.options:
    #         raise ValueError(f'{self.guess} is not in the options')
    #     return self
    
# Define your desired output structure
class Question(BaseModel):
    question: str
    
    @classmethod
    @field_validator('question')
    def check_question(cls, v:str) -> str:
        if '?' not in v:
            raise ValueError(f'{v} is not a question')
        return v


class OpenaiPlayer:
    
    def __init__(self, **kwargs) -> None:
        self.client = instructor.from_openai(OpenAI())
        self.kwargs = kwargs
        
        
    def ask(self, obs, cfg) -> str:
        messages = prepare_ask_messages(obs.questions, obs.answers, obs.guesses)
        
        response = self.client.chat.completions.create(
                response_model=Question,
                messages=messages,
                **self.kwargs
            )
        return response.question
    
    def guess(self, obs, cfg) -> str:
        messages = prepare_guess_messages(obs.questions, obs.answers, guess=None, options=None)
        
        response = self.client.chat.completions.create(
                response_model=Guess,
                messages=messages,
                **self.kwargs
            )
        return response.guess
    
    def answer(self, obs, cfg) -> str:
        messages = prepare_answer_messages(
            keyword=obs["keyword"], category=obs["category"], questions=obs.questions, answers=obs.answers
        )
        response = self.client.chat.completions.create(
                response_model=Answer,
                messages=messages,
                **self.kwargs
            )
        return response.answer