from abc import ABC, abstractmethod

from action.web_action import WebAction
from state.web_state import WebState


class Agent(ABC):
    @abstractmethod
    def get_action(self, web_state: WebState, html: str) -> WebAction:
        pass
