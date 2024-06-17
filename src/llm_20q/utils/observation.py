import itertools
import typing as t

import rigging as rg  # noqa
from pydantic import BaseModel, StringConstraints, field_validator  # noqa

str_strip = t.Annotated[str, StringConstraints(strip_whitespace=True)]


class Observation(BaseModel):
    step: int
    role: t.Literal["guesser", "answerer"]
    turnType: t.Literal["ask", "answer", "guess"]
    keyword: str
    category: str
    questions: list[str]
    answers: list[str]
    guesses: list[str]
    remainingOverageTime: float

    @property
    def empty(self) -> bool:
        return all(len(t) == 0 for t in [self.questions, self.answers, self.guesses])

    def get_history(self) -> t.Iterator[tuple[str, str, str]]:
        return itertools.zip_longest(self.questions, self.answers, self.guesses, fillvalue="[none]")

    def get_history_as_xml(self, *, skip_guesses: bool = False) -> str:
        return (
            "\n".join(
                f"""\
            <turn-{i}>
            Question: {question}
            Answer: {answer}
            {'Guess: ' + guess if not skip_guesses else ''}
            </turn-{i}>
            """
                for i, (question, answer, guess) in enumerate(self.get_history())
            )
            if not self.empty
            else "none yet."
        )


class Answer(rg.Model):
    content: t.Literal["yes", "no"]

    @field_validator("content", mode="before")
    def validate_content(cls, v: str) -> str:
        for valid in ["yes", "no"]:
            if v.lower().startswith(valid):
                return valid
        raise ValueError("Invalid answer, must be 'yes' or 'no'")

    @classmethod
    def xml_example(cls) -> str:
        return f"{Answer.xml_start_tag()}**yes/no**{Answer.xml_end_tag()}"


class Question(rg.Model):
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return Question(content="**question**").to_pretty_xml()


class Guess(rg.Model):
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return Guess(content="**thing/place/person**").to_pretty_xml()
