from typing import Dict, List, Literal

from pydantic import BaseModel

TaskType = Literal["ask", "answer", "guess"]
AnswerType = Literal["yes", "no"]
ConversationType = List[Dict[str, str]]


class Observation(BaseModel):
    step: int
    role: Literal["guesser", "answerer"]
    turnType: Literal["ask", "answer", "guess"]
    keyword: str
    category: str
    questions: list[str]
    answers: list[str]
    guesses: list[str]

    @property
    def empty(self) -> bool:
        return all(len(t) == 0 for t in [self.questions, self.answers, self.guesses])
