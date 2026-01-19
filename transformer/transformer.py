from abc import ABC, abstractmethod

from torch import Tensor

from action.web_action import WebAction
from state.web_state import WebState


class Transformer(ABC):
    @abstractmethod
    def action_to_tensor(self, state: WebState, action: WebAction) -> Tensor:
        pass

    @abstractmethod
    def state_to_tensor(self, state: WebState, html: str) -> Tensor:
        pass
