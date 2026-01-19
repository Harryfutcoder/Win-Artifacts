from typing import List, Dict, Any, Tuple

from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from state.web_state import WebState


class SameUrlState(WebState):
    def __init__(self, restart_url: str) -> None:
        self.action = RestartAction(restart_url)

    def similarity(self, other: 'WebState') -> float:
        return 0

    def get_action_list(self) -> List[WebAction]:
        return [self.action]

    def get_action_detailed_data(self) -> Tuple[Dict[WebAction, Any], Any]:
        return {self.action: None}, None

    def update_action_execution_time(self, action: WebAction) -> None:
        pass

    def update_transition_information(self, action: WebAction, new_state: 'WebState') -> None:
        pass

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SameUrlState):
            return self.action == other.action
        return False

    def __hash__(self) -> int:
        return hash(self.action)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, SameUrlState):
            return self.action < other.action
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return f'SameUrlState(action={self.action})'
