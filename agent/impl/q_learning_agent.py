import math
import random
from collections import defaultdict

from exceptiongroup import catch

from action.impl.restart_action import RestartAction
from agent.agent import Agent
from exceptions import NoActionsException
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.impl.tag_sequence_state import TagSequenceState


class QLearningAgent(Agent):
    def __init__(self, params):
        self.AGENT_TYPE = params["agent_type"]
        self.ALPHA = params["alpha"]
        self.GAMMA = params["gamma"]
        self.EPSILON = params["epsilon"]
        self.INITIAL_Q_VALUE = params["initial_q_value"]
        self.R_REWARD = params["r_reward"]
        self.R_PENALTY = params["r_penalty"]
        self.MAX_SIM_LINE = params["max_sim_line"]
        self.state_repr_list = list()
        self.action_list = list()
        self.q_table = dict()
        self.action_count = dict()  # key: index(action), value: visited times
        self.url_count = dict()  # key: string(url), value: visited times
        self.state_count = dict()
        self.trans_count = defaultdict(int)
        self.previous_state = None
        self.previous_action = None
        self.stop_update = False

    def state_abstraction(self, state):
        actions = state.get_action_list()
        for a in actions:
            if a not in self.action_list:
                self.action_list.append(a)
                self.action_count[self.action_list.index(a)] = 0

        action_index_set = set()
        for a in actions:
            action_index_set.add(self.action_list.index(a))
        action_index_list = list(action_index_set)
        action_index_list.sort()
        action_index_list_str = [str(x) for x in action_index_list]
        state_representation = ','.join(action_index_list_str)
        return state_representation

    def get_state_index(self, state, html):
        if len(self.state_repr_list) == 0:
            self.state_repr_list.append(OutOfDomainState("111"))
            self.state_repr_list.append(ActionExecuteFailedState("111"))
            self.state_repr_list.append(SameUrlState("111"))
            self.q_table[0] = dict()
            self.q_table[1] = dict()
            self.q_table[2] = dict()
            self.q_table[0][0] = -9999
            self.q_table[1][0] = -99
            self.q_table[2][0] = -99
        if len(self.action_list) == 0:
            self.action_list.append(RestartAction("111"))
            self.action_count[0] = 0
        if isinstance(state, OutOfDomainState):
            return 0
        if isinstance(state, ActionExecuteFailedState):
            return 1
        if isinstance(state, SameUrlState):
            return 2

        if self.AGENT_TYPE == "Q":
            state_instance = self.state_abstraction(state)
            if state_instance not in self.state_repr_list:
                self.state_repr_list.append(state_instance)
                s_idx = self.state_repr_list.index(state_instance)
                action_value = dict()
                actions = state.get_action_list()
                for action in actions:
                    if action not in self.action_list:
                        self.action_list.append(action)
                    a_idx = self.action_list.index(action)
                    action_value[a_idx] = self.INITIAL_Q_VALUE
                self.q_table[s_idx] = action_value
            s_idx = self.state_repr_list.index(state_instance)
            return s_idx
        elif self.AGENT_TYPE == "W":
            new_state = TagSequenceState(html)
            max_sim = -1
            max_state = None
            for temp_state in self.state_repr_list:
                if isinstance(temp_state, OutOfDomainState) or isinstance(temp_state, SameUrlState) or isinstance(
                        temp_state, ActionExecuteFailedState):
                    continue
                new_sim = new_state.similarity(temp_state)
                if new_sim > max_sim:
                    max_sim = new_sim
                    max_state = temp_state
            if max_sim > self.MAX_SIM_LINE:
                s_idx = self.state_repr_list.index(max_state)
                actions = state.get_action_list()
                for action in actions:
                    if action not in self.action_list:
                        self.action_list.append(action)
                        self.action_count[self.action_list.index(action)] = 0
                    a_idx = self.action_list.index(action)
                    if a_idx not in self.q_table[s_idx]:
                        self.q_table[s_idx][a_idx] = self.INITIAL_Q_VALUE
            else:
                self.state_repr_list.append(new_state)
                s_idx = self.state_repr_list.index(new_state)
                action_value = dict()
                actions = state.get_action_list()
                for action in actions:
                    if action not in self.action_list:
                        self.action_list.append(action)
                        self.action_count[self.action_list.index(action)] = 0
                    a_idx = self.action_list.index(action)
                    action_value[a_idx] = self.INITIAL_Q_VALUE
                self.q_table[s_idx] = action_value
            return s_idx

    def get_reward(self, state_index):
        if self.AGENT_TYPE == "W":
            s = "{}-{}-{}".format(self.previous_state, self.previous_action, state_index)
            self.trans_count[s] += 1
            reward = 1 / math.sqrt(self.trans_count[s])
            return reward
        elif self.AGENT_TYPE == "Q":
            action_count = self.action_count[self.previous_action]
            if action_count == 1:
                reward = 500
            else:
                reward = 1 / action_count
            return reward

    def update(self, web_state_index, web_state):
        ps_q_values = self.q_table[self.previous_state]
        cs_q_values = self.q_table[web_state_index]
        reward = self.get_reward(web_state_index)
        q_predict = ps_q_values[self.previous_action]
        if self.AGENT_TYPE == "Q":
            action_len=1
            if isinstance(web_state, ActionExecuteFailedState):
                action_list = web_state.get_action_list()
                action_len = len(action_list)
            gamma = 0.9 * math.exp(-0.1 * (abs(action_len) - 1))
        else:
            gamma = self.GAMMA
        q_target = reward + gamma * max(cs_q_values.values())
        print("Updated Q Value:", self.q_table[self.previous_state][self.previous_action], "->",
              q_predict + self.ALPHA * (q_target - q_predict))
        self.q_table[self.previous_state][self.previous_action] = q_predict + self.ALPHA * (q_target - q_predict)


    def get_action_index(self, action):
        if isinstance(action, RestartAction):
            return 0
        return self.action_list.index(action)


    def get_action(self, web_state, html):
        actions = web_state.get_action_list()
        if len(actions) == 0:
            raise NoActionsException("The state does not have any actions")

        chosen_action = None
        stop_update = False

        state_index = self.get_state_index(web_state, html)
        if random.uniform(0, 1) < self.EPSILON:
            max_val = max(self.q_table[state_index].values())
            chosen_action = random.choice(actions)
        else:
            chosen_action = actions[0]
            max_val = self.q_table[state_index][self.get_action_index(chosen_action)]
            for temp_action in actions:
                if isinstance(temp_action, RestartAction):
                    chosen_action = temp_action
                    stop_update = True
                    break
                if self.q_table[state_index][self.get_action_index(temp_action)] > max_val:
                    max_val = self.q_table[state_index][self.get_action_index(temp_action)]
                    chosen_action = temp_action


        self.action_count[self.get_action_index(chosen_action)] += 1
        if (self.previous_state is not None and self.previous_state > 2 and self.previous_action is not None and
                not self.stop_update):
            self.update(state_index, web_state)
        self.previous_state = state_index
        self.previous_action = self.get_action_index(chosen_action)
        self.stop_update = stop_update
        print("max_q_value: ", max_val, "  chosen_action: ", chosen_action)
        return chosen_action
