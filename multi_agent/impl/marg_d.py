import math
import random
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, List

import torch
from sympy.integrals.intpoly import best_origin
from torch import optim, nn

import multi_agent.multi_agent_system
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from model.dense_net import DenseNet
from model.mixing_network import QMixingNetwork, QTranNetwork
from model.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from transformer.impl.tag_transformer import TagTransformer
from utils import instantiate_class_by_module_and_class_name
from web_test.multi_agent_thread import logger


class MargD(multi_agent.multi_agent_system.MultiAgentSystem):
    def __init__(self, params):
        super(MargD, self).__init__(params)
        self.params = params
        self.algo_type = params["algo_type"]
        self.reward_function = params["reward_function"]
        self.state_list = []
        self.q_eval_agent: Dict = {}
        self.q_target_agent: Dict = {}
        self.transformer = instantiate_class_by_module_and_class_name(
            params["transformer_module"], params["transformer_class"])
        self.max_random = params["max_random"]
        self.min_random = params["min_random"]
        self.start_time = datetime.now()
        self.alive_time = params["alive_time"]
        self.batch_size = params["batch_size"]
        self.mix_batch_size = params["mix_batch_size"]
        self.update_target_interval = params['update_target_interval']
        self.update_network_interval = params['update_network_interval']
        self.update_mixing_network_interval = params['update_mixing_network_interval']
        self.using_mix = params["using_mix"]
        self.algo_type = params["algo_type"]
        self.gamma = params["gamma"]
        self.network_lock = threading.Lock()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # 使用 GPU
            print("GPU is available")
        else:
            self.device = torch.device("cpu")  # 使用 CPU
            print("GPU is not available, using CPU")

        # 额外的记录条目
        self.state_list = []
        self.state_trans_count = defaultdict(int)
        self.action_count = defaultdict(int)
        self.action_list = []
        self.learn_step_count = 0
        self.replay_buffer_mixing = ReplayBuffer(capacity=500)
        self.finish_dict_agent: Dict[str, bool] = {}
        self.replay_buffer_agent: Dict[str, ReplayBuffer] = {}
        self.replay_buffer_total = ReplayBuffer(capacity=1000)
        self.learn_step_count_agent: Dict[str, int] = {}
        self.state_list_agent: Dict[str, List[WebState]] = {}
        self.round = 0

        # 用于记录最近的一次成功执行
        self.prev_state_success_dict: Dict[str, Optional[WebState]] = {}
        self.prev_action_success_dict: Dict[str, Optional[WebAction]] = {}
        self.current_state_success_dict: Dict[str, Optional[WebState]] = {}
        self.prev_html_success_dict: Dict[str, str] = {}

        self.prev_best_action_dict: Dict[str, Optional[WebAction]] = {}  # for q tran
        self.prev_best_action_success_dict: Dict[str, Optional[WebAction]] = {}

        # 用于记录连续url动作，避免卡死
        self.replay_buffer_fail = ReplayBuffer(capacity=500)
        self.fail_tensor_list_agent: Dict = {}
        self.fail_reward_list_agent: Dict = {}
        self.fail_html_list_agent: Dict = {}
        self.fail_web_state_list_agent: Dict = {}
        self.fail_done_list_agent: Dict = {}
        self.same_url_count_agent: Dict = {}
        self.previous_url_agent: Dict = {}

        self.agent_optimizer = {}

        for i in range(self.agent_num):
            q_eval = instantiate_class_by_module_and_class_name(
                params["model_module"], params["model_class"])
            q_target = instantiate_class_by_module_and_class_name(
                params["model_module"], params["model_class"])
            q_eval.to(self.device)
            q_target.to(self.device)
            self.q_eval_agent[str(i)] = q_eval
            self.q_target_agent[str(i)] = q_target
            optimizer = optim.Adam(self.q_eval_agent[str(i)].parameters(), lr=params["learning_rate"])
            self.agent_optimizer[str(i)] = optimizer

            self.prev_state_success_dict[str(i)] = None
            self.prev_action_success_dict[str(i)] = None
            self.current_state_success_dict[str(i)] = None
            self.prev_html_success_dict[str(i)] = ""

            self.fail_tensor_list_agent[str(i)] = []
            self.fail_reward_list_agent[str(i)] = []
            self.fail_html_list_agent[str(i)] = []
            self.fail_web_state_list_agent[str(i)] = []
            self.fail_done_list_agent[str(i)] = []
            self.same_url_count_agent[str(i)] = 0
            self.previous_url_agent[str(i)] = None
            self.finish_dict_agent[str(i)] = False
            self.replay_buffer_agent[str(i)] = ReplayBuffer(capacity=500)
            self.learn_step_count_agent[str(i)] = 0
            self.state_list_agent[str(i)] = []

        # self.optimizer = optim.Adam(self.q_eval_agent['0'].parameters(), lr=params["learning_rate"])
        self.criterion = nn.MSELoss()
        # Mixing network
        if self.algo_type == 'qtran':
            self.mixing_network = QTranNetwork(n_agents=self.agent_num, state_dim=self.agent_num * 52, action_dim=12)
            self.target_mixing_network = QTranNetwork(n_agents=self.agent_num, state_dim=self.agent_num * 52,
                                                      action_dim=12)
        else:
            self.mixing_network = QMixingNetwork(n_agents=self.agent_num, state_dim=self.agent_num * 52)
            self.target_mixing_network = QMixingNetwork(n_agents=self.agent_num, state_dim=self.agent_num * 52)
        self.mixing_network.to(self.device)
        self.target_mixing_network.to(self.device)
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=params["learning_rate"])

        self.R_PENALTY = -99.0
        self.R_SAME_URL_PENALTY = -20.0
        self.R_A_PENALTY = -5.0
        self.R_A_BASE_PENALTY_LITTLE = -1.0
        self.R_A_BASE_HIGH = 50.0
        self.R_A_BASE_MIDDLE = 10.0
        self.R_A_MAX_SIM_LINE = 0.98
        self.R_A_MIDDLE_SIM_LINE = 0.85
        self.R_A_MIN_SIM_LINE = 0.7
        self.R_A_TIME_ROUND = 3600
        self.SIM_FACTOR = 0.5

        self.MAX_SAME_URL_COUNT = 30

    def get_tensor(self, action, html, web_state):
        state_tensor = self.transformer.state_to_tensor(web_state, html)
        execution_time = self.action_dict[action]
        if isinstance(self.transformer, TagTransformer):
            action_tensor = self.transformer.action_to_tensor(web_state, action, execution_time)
        else:
            action_tensor = self.transformer.action_to_tensor(web_state, action)
        tensor = torch.cat((state_tensor, action_tensor))
        tensor = tensor.float()
        return tensor

    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        self.update(web_state, html, agent_name)

        actions = web_state.get_action_list()
        if (len(actions) == 1) and (isinstance(actions[0], RestartAction)):
            chosen_action = actions[0]
        else:
            q_eval = self.q_eval_agent[agent_name]
            q_eval.eval()
            action_tensors = []
            for temp_action in actions:
                action_tensor = self.get_tensor(temp_action, html, web_state)
                action_tensors.append(action_tensor)

            q_values = []
            if isinstance(q_eval, DenseNet):
                output = q_eval(torch.stack(action_tensors).unsqueeze(1).to(self.device))
            else:
                output = q_eval(torch.stack(action_tensors).to(self.device))
            for j in range(output.size(0)):
                value = output[j].item()
                q_values.append(value)
            max_val = q_values[0]
            chosen_action = actions[0]
            for i in range(0, len(actions)):
                temp_action = actions[i]
                q_value = q_values[i]
                # flag = self.action_dict[temp_action] == 0 and self.action_dict[chosen_action] == 0 or self.action_dict[
                #     temp_action] != 0 and self.action_dict[chosen_action] != 0
                # if self.action_dict[temp_action] == 0 and self.action_dict[
                #     chosen_action] != 0 or q_value > max_val and flag:
                if q_value > max_val:
                    max_val = q_value
                    chosen_action = temp_action
            logger.info(
                "Thread " + agent_name + ":  " + f"max val: {max_val}")
            self.prev_best_action_dict[agent_name] = chosen_action

            end_time = datetime.now()
            time_difference = end_time - self.start_time
            seconds_difference = time_difference.total_seconds()
            seconds_difference = min(seconds_difference, 10800)
            all_time = min(self.alive_time, 10800)
            # 前一半时间随时间衰减(1h)，后一半时间保持最低
            random_line = self.max_random - min(float(seconds_difference) / all_time * 2, 1) * (
                    self.max_random - self.min_random)
            if random.uniform(0, 1) < random_line:
                temp_actions = []
                for i in range(0, len(actions)):
                    if self.action_dict[actions[i]] == 0:
                        temp_actions.append(actions[i])
                if len(temp_actions) == 0:
                    chosen_action = random.choice(actions)
                else:
                    chosen_action = random.choice(temp_actions)

            self.action_count[chosen_action] += 1
        return chosen_action

    def get_reward(self, web_state: WebState, agent_name: str):
        if self.reward_function == "A":
            if not isinstance(web_state, ActionSetWithExecutionTimesState):
                return self.R_A_PENALTY
            max_sim = -1
            for temp_state in self.state_list:
                if (isinstance(temp_state, OutOfDomainState) or
                        isinstance(temp_state, ActionExecuteFailedState) or
                        isinstance(temp_state, SameUrlState)): continue
                if web_state == temp_state: continue
                if web_state.similarity(temp_state) > max_sim:
                    max_sim = web_state.similarity(temp_state)

            if max_sim < self.R_A_MIN_SIM_LINE:
                r_state = self.R_A_BASE_HIGH
            elif max_sim < self.R_A_MIDDLE_SIM_LINE:
                r_state = self.R_A_BASE_MIDDLE
            else:
                visited_time = self.state_dict[web_state]
                if visited_time == 0:
                    r_state = 2.0
                else:
                    r_state = 1.0 / visited_time

            if not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState):
                r_action = 0
            else:
                action_index = self.action_list.index(self.prev_action_dict[agent_name])
                execution_time = self.action_count[action_index]
                if execution_time == 0:
                    r_action = 2.0
                else:
                    r_action = 1.0 / float(execution_time)

            # if not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState):
            #     r_trans = 0
            # else:
            #     previous_state_index = self.state_list.index(self.prev_state_dict[agent_name])
            #     current_state_index = self.state_list.index(web_state)
            #     previous_action_index = self.action_list.index(self.prev_action_dict[agent_name])
            #     s = "{}-{}-{}".format(previous_state_index, previous_action_index, current_state_index)
            #     self.state_trans_count[s] += 1
            #     r_trans = (1 - web_state.similarity(self.prev_state_dict[agent_name])) * (
            #             1 / math.sqrt(self.state_trans_count[s]))

            end_time = datetime.now()
            time_difference = end_time - self.start_time
            seconds_difference = time_difference.total_seconds()
            r_time = 1 + float(seconds_difference) / self.alive_time
            # return (r_state + r_action + r_trans) * r_time
            return (r_state + r_action) * r_time
        else:
            return 0

    def get_reward_total(self):
        if self.algo_type == 'qmix_d':
            reward_total = 0.0
            for i in range(self.agent_num):
                for j in range(self.agent_num):
                    if i != j:
                        agent_1 = str(i)
                        agent_2 = str(j)
                        state_1 = self.current_state_success_dict[agent_1]
                        max_sim = -1
                        for temp_state in self.state_list_agent[agent_2]:
                            if (isinstance(temp_state, OutOfDomainState) or
                                    isinstance(temp_state, ActionExecuteFailedState) or
                                    isinstance(temp_state, SameUrlState)): continue
                            if state_1 == temp_state: continue
                            if state_1.similarity(temp_state) > max_sim:
                                max_sim = state_1.similarity(temp_state)
                        reward_total += 1.0 - max_sim

            total_pairs = self.agent_num * (self.agent_num - 1)
            scaled_reward = (reward_total / total_pairs) * 50
            print("scaled_reward_total: ", scaled_reward)
            return scaled_reward
        else:
            reward_total = 0
            for i in range(self.agent_num):
                agent_name = str(i)
                reward_total += self.get_reward(self.current_state_dict[agent_name], agent_name)
            return reward_total

    def update(self, web_state: WebState, html: str, agent_name: str):
        # 更新计数
        if not self.state_list.__contains__(web_state):
            self.state_list.append(web_state)
        if not self.state_list_agent[agent_name].__contains__(web_state):
            self.state_list_agent[agent_name].append(web_state)
        actions = web_state.get_action_list()
        for temp_action in actions:
            if not self.action_list.__contains__(temp_action):
                self.action_list.append(temp_action)

        if self.prev_action_dict[agent_name] is None or self.prev_state_dict[
            agent_name] is None or not isinstance(
            self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState):
            return

        tensor = self.get_tensor(self.prev_action_dict[agent_name], self.prev_html_dict[agent_name],
                                 self.prev_state_dict[agent_name])

        tensor.unsqueeze_(0)

        # 更新agent神经网络
        if self.algo_type != 'qmix' and self.algo_type != 'qtran' and self.algo_type != 'vdn':
            reward = self.get_reward(web_state, agent_name)
            done = False
            self.replay_buffer_agent[agent_name].push(tensor, tensor, reward, web_state, html, done)
            self.replay_buffer_total.push(tensor, tensor, reward, web_state, html, done)
            self.learn_agent(agent_name)

        # 更新mixing神经网络
        with self.lock:
            if not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState) or not isinstance(
                    self.current_state_dict[agent_name], ActionSetWithExecutionTimesState) or self.using_mix == 'F':
                return
            self.finish_dict_agent[agent_name] = True
            self.prev_state_success_dict[agent_name] = self.prev_state_dict[agent_name]
            self.prev_action_success_dict[agent_name] = self.prev_action_dict[agent_name]
            self.prev_best_action_success_dict[agent_name] = self.prev_best_action_dict[agent_name]
            self.current_state_success_dict[agent_name] = self.current_state_dict[agent_name]
            self.prev_html_success_dict[agent_name] = self.prev_html_dict[agent_name]

            for i in range(self.agent_num):
                if not self.finish_dict_agent[str(i)]:
                    return
            for i in range(self.agent_num):
                self.finish_dict_agent[str(i)] = False

            tensors = []
            best_tensors = []
            rewards = self.get_reward_total()
            next_states = []
            htmls = []
            done = False
            for i in range(self.agent_num):
                agent_name = str(i)
                tensor = self.get_tensor(self.prev_action_success_dict[agent_name],
                                         self.prev_html_success_dict[agent_name],
                                         self.prev_state_success_dict[agent_name])

                tensor.unsqueeze_(0)

                best_tensor = self.get_tensor(self.prev_best_action_success_dict[agent_name],
                                              self.prev_html_success_dict[agent_name],
                                              self.prev_state_success_dict[agent_name])

                best_tensor.unsqueeze_(0)

                tensors.append(tensor)
                best_tensors.append(best_tensor)
                next_states.append(self.current_state_success_dict[agent_name])
                htmls.append(self.prev_html_success_dict[agent_name])

            self.replay_buffer_mixing.push(tensors, best_tensors, rewards, next_states, htmls, done)

        self.learn_mixing()

    def learn_agent(self, agent_name: str):
        self.learn_step_count_agent[agent_name] += 1
        if self.learn_step_count_agent[agent_name] % self.update_target_interval == 0:
            self.update_network_parameters_agent(agent_name)
        logger.info(
            "Thread " + agent_name + ":  " + f"learn step count: {self.learn_step_count_agent[agent_name]}")
        if self.learn_step_count_agent[agent_name] % self.update_network_interval != 0:
            return
        replay_buffer_agent = self.replay_buffer_agent[agent_name]
        replay_buffer = self.replay_buffer_total
        # 从经验回放池中采样一批经验
        if self.algo_type == 'share_buffer':
            self.learn_agent_with_buffer(agent_name, replay_buffer)
        else:
            self.learn_agent_with_buffer(agent_name, replay_buffer_agent)

    def learn_agent_with_buffer(self, agent_name, replay_buffer):
        if len(replay_buffer.buffer) < self.batch_size:
            return  # 如果经验池中没有足够的经验，跳过学习
        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            tensors, best_tensors, rewards, next_states, htmls, dones, weights, indices = replay_buffer.sample(
                self.batch_size)
        else:
            tensors, best_tensors, rewards, next_states, htmls, dones = replay_buffer.sample(self.batch_size)
            indices = []
        # 初始化一个空列表，用来保存每个样本的 TD 误差
        td_errors = []
        target_list = []
        q_eval = self.q_eval_agent[agent_name]

        with self.network_lock:
            # 逐个处理批次中的每个样本
            for i in range(self.batch_size):
                tensor = tensors[i]
                reward = rewards[i]
                next_state = next_states[i]
                html = htmls[i]
                done = dones[i]
                if isinstance(q_eval, DenseNet):
                    input_vector = tensor.unsqueeze(0)  # 如果你的 action_tensor 是一个向量，你需要给它添加一个批量维度
                else:
                    input_vector = tensor
                output = q_eval(input_vector.to(self.device))
                current_q = output.item()
                next_q_value = 0

                if isinstance(next_state, ActionSetWithExecutionTimesState):
                    actions = next_state.get_action_list()
                    action_tensors = []
                    for temp_action in actions:
                        action_tensor = self.get_tensor(temp_action, html, next_states[i])
                        action_tensors.append(action_tensor)

                    # 计算当前Q值：将state和action拼接后作为输入传入Q网络
                    q_values = []
                    if isinstance(q_eval, DenseNet):
                        output = q_eval(torch.stack(action_tensors).unsqueeze(1).to(self.device))
                    else:
                        output = q_eval(torch.stack(action_tensors).to(self.device))
                    for j in range(output.size(0)):
                        value = output[j].item()
                        q_values.append(value)
                    max_val = q_values[0]
                    chosen_tensor = action_tensors[0]
                    for j in range(0, len(actions)):
                        temp_tensor = action_tensors[j]
                        q_value = q_values[j]
                        if q_value > max_val:
                            max_val = q_value
                            chosen_tensor = temp_tensor

                    if self.algo_type == 'nndql':
                        other_agents = [agent for agent in self.q_eval_agent.keys() if agent != agent_name]
                        next_q_values = []
                        for agent in other_agents:
                            if self.state_list_agent[agent].__contains__(next_state):
                                q_target = self.q_eval_agent[agent]
                                if isinstance(q_eval, DenseNet):
                                    next_q = q_target(torch.stack([chosen_tensor]).unsqueeze(1).to(self.device)).item()
                                else:
                                    next_q = q_target(torch.stack([chosen_tensor]).to(self.device)).item()
                                next_q_values.append(next_q)
                        if len(next_q_values) > 0:
                            next_q_value = sum(next_q_values) / len(next_q_values)  # Take average
                        else:
                            q_target = self.q_target_agent[agent_name]
                            if isinstance(q_eval, DenseNet):
                                next_q_value = q_target(
                                    torch.stack([chosen_tensor]).unsqueeze(1).to(self.device)).item()
                            else:
                                next_q_value = q_target(torch.stack([chosen_tensor]).to(self.device)).item()
                    else:
                        q_target = self.q_target_agent[agent_name]
                        if isinstance(q_eval, DenseNet):
                            next_q_value = q_target(torch.stack([chosen_tensor]).unsqueeze(1).to(self.device)).item()
                        else:
                            next_q_value = q_target(torch.stack([chosen_tensor]).to(self.device)).item()

                # 计算目标Q值
                target_q = reward + self.gamma * next_q_value * (1 - done)
                target_list.append(target_q)
                td_errors.append(abs(current_q - target_q))  # 计算TD误差

            if isinstance(replay_buffer, PrioritizedReplayBuffer):
                replay_buffer.update_priorities(indices, td_errors)

            q_eval.train()
            if isinstance(q_eval, DenseNet):
                input_tensor = torch.stack(tensors)
            else:
                input_tensor = torch.stack(tensors).squeeze(1)
            q_predicts_tensor = q_eval(input_tensor.to(self.device))
            tensor_list = [torch.tensor([x]) for x in target_list]
            q_target_tensor = torch.stack(tensor_list).to(self.device)
            loss = self.criterion(q_predicts_tensor, q_target_tensor)
            print("Thread " + agent_name + ":  " + "loss:", loss)

            self.agent_optimizer[agent_name].zero_grad()
            loss.backward()
            self.agent_optimizer[agent_name].step()

    def learn_mixing(self):
        if self.using_mix == 'F':
            return
        self.learn_step_count += 1
        logger.info(f"mixing learn step count: {self.learn_step_count}")
        if self.learn_step_count % self.update_mixing_network_interval != 0:
            return
        if len(self.replay_buffer_mixing.buffer) < self.mix_batch_size:
            return  # 如果经验池中没有足够的经验，跳过学习
        if self.learn_step_count % self.update_target_interval == 0:
            self.update_network_parameters()

        # 从经验回放池中采样一批经验
        tensors, best_tensors, rewards, next_states, htmls, dones = self.replay_buffer_mixing.sample(
            self.mix_batch_size)

        # 存储批次的 Q 值
        batch_current_q_values = []  # (batch_size, n_agents)
        batch_current_best_q_values = []  # (batch_size, n_agents)
        batch_next_q_values = []  # (batch_size, n_agents)
        batch_states = []  # (batch_size, state_dim)
        batch_next_states = []  # (batch_size, state_dim)
        batch_actions = []  # (batch_size, n_agents, action_dim)
        batch_best_actions = []  # (batch_size, n_agents, action_dim)
        batch_next_actions = []  # (batch_size, n_agents, action_dim)
        with self.network_lock:
            for i in range(self.mix_batch_size):
                tensor = tensors[i]
                best_tensor = best_tensors[i]
                next_state = next_states[i]
                html = htmls[i]

                current_q_values = []
                current_best_q_values = []
                next_q_values = []
                state_total = None
                next_state_total = None
                actions = []
                best_actions = []
                next_actions = []

                for j in range(self.agent_num):
                    agent_name = str(j)
                    q_eval = self.q_eval_agent[agent_name]
                    q_target = self.q_target_agent[agent_name]
                    if isinstance(q_eval, DenseNet):
                        input_vector = tensor[j].unsqueeze(0)  # 形状 (1, obs_dim)
                        best_vector = best_tensor[j].unsqueeze(0)
                    else:
                        input_vector = tensor[j]
                        best_vector = best_tensor[j]

                    # 拼接状态
                    if state_total is None:
                        state_total = tensor[j].squeeze(0)
                    else:
                        state_total = torch.cat((state_total, tensor[j].squeeze(0)))

                    # 当前 Q 值
                    output = q_eval(input_vector.to(self.device))
                    current_q_values.append(output.squeeze(0))

                    output = q_eval(best_vector.to(self.device))
                    current_best_q_values.append(output.squeeze(0))

                    # 下一状态 Q 值
                    next_q_value = 0
                    best_action_tensor = torch.zeros_like(input_vector.squeeze(0).squeeze(0))
                    if isinstance(next_state[j], ActionSetWithExecutionTimesState):
                        action_list = next_state[j].get_action_list()
                        action_tensors = [self.get_tensor(temp_action, html[j], next_state[j]) for temp_action in
                                          action_list]
                        if isinstance(q_eval, DenseNet):
                            q_values = q_target(torch.stack(action_tensors).unsqueeze(1).to(self.device)).squeeze(1)
                        else:
                            q_values = q_target(torch.stack(action_tensors).to(self.device)).squeeze(1)
                        next_q_value, max_index = q_values.max(0)  # 获取最大值和对应的索引
                        best_action_tensor = action_tensors[max_index.item()]  # 根据索引获取对应的 action_tensor
                    next_q_values.append(next_q_value)
                    if next_state_total is None:
                        next_state_total = best_action_tensor
                    else:
                        next_state_total = torch.cat((next_state_total, best_action_tensor))

                    # 存储动作
                    actions.append(input_vector.squeeze(0).squeeze(0)[-12:])
                    best_actions.append(best_vector.squeeze(0).squeeze(0)[-12:])
                    next_actions.append(best_action_tensor[-12:])

                batch_current_q_values.append(torch.stack(current_q_values))  # 拼接当前批次 Q 值
                batch_current_best_q_values.append(torch.stack(current_best_q_values))
                batch_next_q_values.append(torch.tensor(next_q_values))  # 拼接下一状态 Q 值
                batch_states.append(state_total)  # 拼接当前状态
                batch_next_states.append(next_state_total)  # 拼接下一状态
                batch_actions.append(torch.stack(actions))  # 拼接当前动作
                batch_best_actions.append(torch.stack(best_actions))
                batch_next_actions.append(torch.stack(next_actions))  # 拼接下一动作

            # 将所有批次数据转换为张量
            batch_current_q_values = torch.stack(batch_current_q_values).squeeze(-1)  # (batch_size, n_agents)
            batch_current_best_q_values = torch.stack(batch_current_best_q_values).squeeze(-1)  # (batch_size, n_agents)
            batch_next_q_values = torch.stack(batch_next_q_values).squeeze(-1)  # (batch_size, n_agents)
            batch_states = torch.stack(batch_states)  # (batch_size, state_dim)
            batch_next_states = torch.stack(batch_next_states)  # (batch_size, state_dim)
            batch_actions = torch.stack(batch_actions)  # (batch_size, n_agents, action_dim)
            batch_best_actions = torch.stack(batch_best_actions)  # (batch_size, n_agents, action_dim)
            batch_next_actions = torch.stack(batch_next_actions)  # (batch_size, n_agents, action_dim)
            rewards = torch.tensor(rewards).view(-1, 1).to(self.device)  # (batch_size, 1)

            # QTran 的损失计算
            if self.algo_type == 'qtran':
                # 计算联合 Q 值、局部 Q 值和状态值
                joint_q_value, v_value = self.mixing_network(
                    batch_states.to(self.device), batch_actions.to(self.device))
                best_joint_q_value, best_v_value = self.mixing_network(batch_states.to(self.device),
                                                                       batch_best_actions.to(self.device))
                next_joint_q_value, next_v_value = self.target_mixing_network(
                    batch_next_states.to(self.device), batch_next_actions.to(self.device))

                individual_q_values = batch_current_q_values.to(self.device)
                individual_best_q_values = batch_current_best_q_values.to(self.device)

                # TD loss
                td_target = rewards + self.gamma * next_joint_q_value
                td_loss = self.criterion(joint_q_value, td_target.detach())

                # Optimality loss
                best_sum_q_i = individual_best_q_values.sum(dim=1, keepdim=True)
                opt_loss = self.criterion(best_joint_q_value.detach(), v_value + best_sum_q_i)

                # nopt
                sum_q_i = individual_q_values.sum(dim=1, keepdim=True)
                nopt_residual = sum_q_i - joint_q_value.detach() + v_value
                nopt_loss = torch.mean(torch.relu(-nopt_residual) ** 2)

                loss = td_loss + opt_loss + nopt_loss

            else:
                if self.algo_type == 'vdn':
                    q_tot = batch_current_q_values.sum(dim=1, keepdim=True)  # (batch_size, 1)
                    next_q_tot = batch_next_q_values.sum(dim=1, keepdim=True)  # (batch_size, 1)
                else:
                    q_tot = self.mixing_network(batch_current_q_values.to(self.device),
                                                batch_states.to(self.device))  # (batch_size, 1)
                    next_q_tot = self.target_mixing_network(batch_next_q_values.to(self.device),
                                                            batch_next_states.to(self.device))  # (batch_size, 1)
                target_q_tot = rewards + self.gamma * next_q_tot  # (batch_size, 1)
                loss = self.criterion(q_tot, target_q_tot.detach())  # TD 误差

            # 反向传播和优化
            self.mixing_optimizer.zero_grad()
            for i in range(self.agent_num):
                self.agent_optimizer[str(i)].zero_grad()
            loss.backward()
            self.mixing_optimizer.step()
            for i in range(self.agent_num):
                self.agent_optimizer[str(i)].step()

            logger.info(f"Mixing Loss: {loss.item()}")

    def update_network_parameters_agent(self, agent_name):
        with self.network_lock:
            self.q_target_agent[agent_name].load_state_dict(self.q_eval_agent[agent_name].state_dict())

    def update_network_parameters(self):
        with self.network_lock:
            for i in range(self.agent_num):
                agent_name = str(i)
                self.q_target_agent[agent_name].load_state_dict(self.q_eval_agent[agent_name].state_dict())
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
