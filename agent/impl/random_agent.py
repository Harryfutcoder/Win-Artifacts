import random

from action.web_action import WebAction
from agent.agent import Agent
from state.web_state import WebState


class RandomAgent(Agent):
    def get_action(self, web_state: WebState, html: str) -> WebAction:
        return random.choice(web_state.get_action_list())
