from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from action.web_action import WebAction


class WebState(ABC):
    @abstractmethod
    def get_action_list(self) -> List[WebAction]:
        pass

    @abstractmethod
    def get_action_detailed_data(self) -> Tuple[Dict[WebAction, Any], Any]:
        pass

    @abstractmethod
    def update_action_execution_time(self, action: WebAction) -> None:
        pass

    @abstractmethod
    def update_transition_information(self, action: WebAction, new_state: 'WebState') -> None:
        pass

    @abstractmethod
    def __lt__(self, other: object) -> bool:
        pass

    @abstractmethod
    def similarity(self,other: 'WebState') -> float:
        pass
