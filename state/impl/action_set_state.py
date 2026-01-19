from typing import List, Dict, Any, Tuple, Set

from action.web_action import WebAction
from state.web_state import WebState


class ActionSetState(WebState):
    def __init__(self, actions: List[WebAction], url: str) -> None:
        self.action_set: Set[WebAction] = set(actions)
        self.url: str = url

    def similarity(self, other: 'WebState') -> float:
        return 0

    def get_action_list(self) -> List[WebAction]:
        action_list = list(self.action_set)
        action_list.sort()
        return action_list

    def get_action_detailed_data(self) -> Tuple[Dict[WebAction, Any], Any]:
        return {key: None for key in self.action_set}, None

    def update_action_execution_time(self, action: WebAction) -> None:
        pass

    def update_transition_information(self, action: WebAction, new_state: 'WebState') -> None:
        pass

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ActionSetState):
            return (self.action_set == other.action_set) and (self.url == other.url)
        return False

    def __hash__(self) -> int:
        hash_value = 0
        for action in self.action_set:
            hash_value += hash(action)
        hash_value += hash(self.url)
        return hash_value

    def __lt__(self, other: object) -> bool:
        if isinstance(other, ActionSetState):
            return hash(self) < hash(other)
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return f'ActionSetState(action_number={len(self.action_set)}, url={self.url})'
