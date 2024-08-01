from typing import List, Optional
from ..types import Observation, AnswerType
from ..rag import SentenceTransformerRag

class Player(Protocol):
    
    def ask(self, obs:Observation) -> str:
        pass
    
    def guess(self, obs:Observation, options:Optional[List[str]]=None) -> str:
        pass
    
    def answer(self, obs:Observation) -> AnswerType:
        pass