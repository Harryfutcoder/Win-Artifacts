import random
import threading
from typing import Dict
import copy
from typing_extensions import Optional

import multi_agent.multi_agent_system
from action.web_action import WebAction
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.web_state import WebState


class Marg(multi_agent.multi_agent_system.MultiAgentSystem):
    def __init__(self, params):
        super(Marg, self).__init__(params)
        self.params = params
        self.agent_type = params["agent_type"]
        self.epsilon = params["epsilon"]
        self.initial_q_value = params["initial_q_value"]
        self.gamma = params["gamma"]
        self.alpha = params["alpha"]
        self.R_PENALTY = -9999
        self.table_lock = threading.Lock()

        if self.agent_type == 'cql':
            self.q_table: Dict[Optional[WebState], Dict[Optional[WebAction], float]] = {}
        else:  # dql
            self.q_table_agent: Dict[str, Dict[Optional[WebState], Dict[Optional[WebAction], float]]] = {}
            for i in range(self.agent_num):
                q_table: Dict[Optional[WebState], Dict[Optional[WebAction], float]] = {}
                self.q_table_agent[str(i)] = q_table

    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        self.update(web_state, html, agent_name)
        actions = web_state.get_action_list()
        max_action_list = list()
        if self.agent_type == 'cql':
            with self.table_lock:
                q_table = copy.deepcopy(self.q_table)
            max_value = max(q_table[web_state].values())
            for key in q_table[web_state]:
                if q_table[web_state][key] == max_value:
                    max_action_list.append(key)
        else:
            q_table_total = dict()
            with self.table_lock:
                q_table_agent = copy.deepcopy(self.q_table_agent)
            for key in q_table_agent[agent_name][web_state].keys():
                q_table_total[key] = q_table_agent[agent_name][web_state][key]
            for i in range(self.agent_num):
                if q_table_agent[str(i)].__contains__(web_state):
                    for key in q_table_agent[str(i)][web_state].keys():
                        q_table_total[key] += q_table_agent[str(i)][web_state][key]

            # sum to get q_value
            max_value = max(q_table_total.values())
            for key in q_table_total:
                if q_table_total[key] == max_value:
                    max_action_list.append(key)

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        else:
            return random.choice(max_action_list)

    def get_reward(self, action: WebAction, state: WebState) -> float:
        actions = state.get_action_list()
        if len(actions) == 0:
            return self.R_PENALTY
        if isinstance(state, ActionSetWithExecutionTimesState):
            # 注意：action_dict 是 state 的属性，不是 self 的属性
            if action not in state.action_dict:
                return self.R_PENALTY
            execution_time = state.action_dict[action]['execution_time']
            if execution_time == 0:
                return 1.0
            else:
                return 1.0 / float(execution_time)
        return self.R_PENALTY

    def update(self, web_state: WebState, html: str, agent_name: str):
        actions = web_state.get_action_list()

        if self.agent_type == 'cql':
            with self.table_lock:
                if web_state not in self.q_table:
                    self.q_table[web_state] = {}
                for action in actions:
                    if action not in self.q_table[web_state]:
                        self.q_table[web_state][action] = self.initial_q_value
        else:
            with self.table_lock:
                if agent_name not in self.q_table_agent:
                    self.q_table_agent[agent_name] = {}
                if web_state not in self.q_table_agent[agent_name]:
                    flag = False
                    for i in range(self.agent_num):
                        if web_state in self.q_table_agent[str(i)]:
                            flag = True
                            self.q_table_agent[agent_name][web_state] = copy.deepcopy(
                                self.q_table_agent[str(i)][web_state])
                            break
                    if not flag:
                        self.q_table_agent[agent_name][web_state] = {}
                for action in actions:
                    if action not in self.q_table_agent[agent_name][web_state]:
                        self.q_table_agent[agent_name][web_state][action] = self.initial_q_value

        if self.prev_action_dict[agent_name] is None or self.prev_state_dict[agent_name] is None or not isinstance(
                self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState):
            return

        reward = self.get_reward(self.prev_action_dict[agent_name], web_state)
        prev_state = self.prev_state_dict[agent_name]
        prev_action = self.prev_action_dict[agent_name]
        if self.agent_type == 'cql':
            with self.table_lock:
                q_table = copy.deepcopy(self.q_table)
            if not q_table[web_state]:
                max_q_value = self.R_PENALTY
            else:
                max_q_value = max(q_table[web_state].values())
            q_predict = q_table[prev_state][prev_action]
            q_target = reward + self.gamma * max_q_value
            with self.table_lock:
                self.q_table[prev_state][prev_action] = q_predict + self.alpha * (q_target - q_predict)
                print(
                    "Thread " + agent_name + ":  " + f'Transition, cql , update Q value: {q_predict} -> {self.q_table[prev_state][prev_action]}')

        else:
            with self.table_lock:
                q_table_agent = copy.deepcopy(self.q_table_agent)
                q_table = q_table_agent[agent_name]
            values = list()
            for i in range(self.agent_num):
                if str(i) != agent_name and web_state in q_table_agent[str(i)]:
                    values.append(q_table_agent[str(i)][web_state])
            if values.__len__() == 0:
                if not q_table[web_state]:
                    max_q_value = self.R_PENALTY
                else:
                    max_q_value = max(q_table[web_state].values())
                q_predict = q_table[prev_state][prev_action]
                q_target = reward + self.gamma * max_q_value
                with self.table_lock:
                    self.q_table_agent[agent_name][prev_state][prev_action] = q_predict + self.alpha * (
                            q_target - q_predict)
                print(
                    "Thread " + agent_name + ":  " + f'Transition, agent count: {values.__len__()}, update Q value: {q_predict} -> {self.q_table_agent[agent_name][prev_state][prev_action]}')
            else:
                total = 0
                l = values.__len__()
                ps_q_values = q_table[prev_state]
                cs_q_values = q_table[web_state]
                q_predict = ps_q_values[prev_action]
                if not cs_q_values:
                    with self.table_lock:
                        self.q_table_agent[agent_name][prev_state][prev_action] = self.alpha * reward
                else:
                    max_value = max(cs_q_values.values())
                    max_keys = [key for key, value in cs_q_values.items() if value == max_value]
                    ma_idx = random.choice(max_keys)
                    for value in values:
                        total = total + value[ma_idx]
                    with self.table_lock:
                        self.q_table_agent[agent_name][prev_state][prev_action] = q_predict + self.alpha * reward + (
                                self.gamma * total - q_predict * l) / l

                print(
                    "Thread " + agent_name + ":  " + f'Transition, agent count: {values.__len__()}, update Q value: {q_predict} -> {self.q_table_agent[agent_name][prev_state][prev_action]}')
