from kaggle_environments import make
from loguru import logger

from llm_20q.players.openai import OpenaiPlayer

openaiplayer = OpenaiPlayer(model='gpt-3.5-turbo', max_retries=10)

def agent_fn(obs, cfg):
    if obs.turnType == "ask":
        response =  openaiplayer.ask(obs, cfg)
        logger.info(f"{obs.step=} ask: {response}")
    # if agent is guesser and turnType is "guess"
    elif obs.turnType == "guess":
        response = openaiplayer.guess(obs, cfg)
        logger.info(f"{obs.step=} guess: {response}")
    # if agent is the answerer
    elif obs.turnType == "answer":
        # pipe.model.disable_adapters()
        response = openaiplayer.answer(obs, cfg)
        logger.info(f"{obs.step=} answer: {response} -- keyword: {obs.keyword}")
    return response
    
def dumb_agent_fn(obs, cfg):
    return "yes"

if __name__ == "__main__":
    env = make("llm_20_questions", debug=True)
    game = env.run([agent_fn, agent_fn, dumb_agent_fn, dumb_agent_fn])
    print(game)