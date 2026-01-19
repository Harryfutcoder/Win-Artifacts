from typing import Dict

import multi_agent.multi_agent_system
from action.web_action import WebAction
from agent.impl.drl_agent import DRLagent
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name


class IQL(multi_agent.multi_agent_system.MultiAgentSystem):
    def __init__(self, params):
        super(IQL, self).__init__(params)
        self.params = params
        self.agent_dict: Dict[str, DRLagent] = {}

        self.transformer = instantiate_class_by_module_and_class_name(
            params["transformer_module"], params["transformer_class"])
        for i in range(self.params["agent_num"]):
            agent_name = str(i)
            if agent_name not in self.agent_dict:
                agent = DRLagent(params=self.params, transformer=self.transformer)
                self.agent_dict[agent_name] = agent

    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        return self.agent_dict[agent_name].get_action(web_state, html)
