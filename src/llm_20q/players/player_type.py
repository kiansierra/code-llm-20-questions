from typing import List, Optional, Protocol

from ..types import AnswerType, Observation


class Player(Protocol):

    def ask(self, obs: Observation) -> str:
        pass

    def guess(self, obs: Observation, options: Optional[List[str]] = None) -> str:
        pass

    def answer(self, obs: Observation) -> AnswerType:
        pass
