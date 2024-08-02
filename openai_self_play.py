from kaggle_environments import make
from loguru import logger

from llm_20q.players.openai import OpenaiPlayer
from llm_20q.agent import LLM20Q, dumb_agent_fn
from llm_20q.rag import SentenceTransformerRag

openaiplayer = OpenaiPlayer(model='gpt-3.5-turbo', max_retries=10)
rag = SentenceTransformerRag.from_folder("../input/rag/nomic-embed-text-v1")
agent = LLM20Q(openaiplayer, rag)
    

if __name__ == "__main__":
    env = make("llm_20_questions", debug=True)
    game = env.run([agent.agent_fn, agent.agent_fn, dumb_agent_fn, dumb_agent_fn])
    print(game)