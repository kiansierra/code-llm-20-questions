from loguru import logger

from .players import Player
from .rag import SentenceTransformerRag
from .types import Observation


class LLM20Q:

    def __init__(self, agent: Player, rag: SentenceTransformerRag):
        self.agent = agent
        self.rag = rag

    def agent_fn(self, obs, _):
        # if agent is guesser and turnType is "ask"
        obs = Observation(**obs.__dict__)
        if obs.turnType == "ask":
            response = self.agent.ask(obs)
            logger.info(f"{obs.step=} ask: {response}")
        # if agent is guesser and turnType is "guess"
        elif obs.turnType == "guess":
            question = obs.questions[-1]
            answer = obs.answers[-1]
            direction = "top" if answer == "yes" else "bottom"
            self.rag.filter(query=question, direction=direction)
            options = self.rag.retrieve_options(10)
            response = self.agent.guess(obs, options=options)
            logger.info(f"{obs.step=} guess: {response}")
            self.rag.remove_guess(response)
        # if agent is the answerer
        elif obs.turnType == "answer":
            # pipe.model.disable_adapters()
            response = self.agent.answer(obs)
            still_available = obs.keyword in self.rag.filter_df.keyword.tolist()
            logger.info(f"{obs.step=} answer: {response} -- keyword: {obs.keyword} {still_available=}")
        return response
