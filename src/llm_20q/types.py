from typing import Literal, List, Dict

TaskType = Literal["ask", "answer", "guess"]
AnswerType = Literal["yes", "no"]
ConversationType = List[Dict[str, str]]
