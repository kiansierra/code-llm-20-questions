from typing import Literal, Optional,TypeVar, Generic
import instructor
from pydantic import BaseModel, field_validator, model_validator
from openai import OpenAI
from typing_extensions import Self
from ..prompts import prepare_ask_messages, prepare_guess_messages, prepare_answer_messages
from ..types import AnswerType, Observation

T = TypeVar('T')
# Define your desired output structure
class Answer(BaseModel):
    answer: AnswerType
    
# Define your desired output structure
class Guess(BaseModel, Generic[T]):
    guess: T
    
    # This is required to make the tool name short enough
    class Config:
        title = "Guess"


    
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
        
        
    def ask(self, obs:Observation) -> str:
        messages = prepare_ask_messages(obs.questions, obs.answers, obs.guesses)
        
        response = self.client.chat.completions.create(
                response_model=Question,
                messages=messages,
                **self.kwargs
            )
        return response.question
    
    def guess(self, obs:Observation, options:Optional[list[str]]=None) -> str:
        messages = prepare_guess_messages(obs.questions, obs.answers, guess=None, options=options)
        OptionsType = Literal[*options]
        response_model_cls = Guess[str] if not options else Guess[OptionsType]
        response = self.client.chat.completions.create(
                response_model=response_model_cls,
                messages=messages,
                **self.kwargs
            )
        return response.guess
    
    def answer(self, obs:Observation) -> str:
        messages = prepare_answer_messages(
            keyword=obs.keyword, category=obs.category, questions=obs.questions, answers=obs.answers
        )
        response = self.client.chat.completions.create(
                response_model=Answer,
                messages=messages,
                **self.kwargs
            )
        return response.answer