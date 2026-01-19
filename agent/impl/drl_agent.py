import math
import os
import random
from collections import defaultdict
from datetime import datetime

import torch
import torch.optim as optim
from torch import nn

from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from agent import agent
from config.settings import settings
from exceptions import NoActionsException
from model.dense_net import DenseNet
from model.replay_buffer import PrioritizedReplayBuffer
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name


class DRLagent(agent.Agent):
    def __init__(self, params, transformer=None):
        super(DRLagent, self).__init__()
        self.q_eval = instantiate_class_by_module_and_class_name(
            params["model_module"], params["model_class"])
        self.q_target = instantiate_class_by_module_and_class_name(
            params["model_module"], params["model_class"])
        if params["model_load_type"] == "load":
            model_path = os.path.join(settings.model_path, params["model_load_name"])
            if os.path.exists(model_path):
                try:
                    self.q_eval.load_state_dict(torch.load(model_path))
                    self.q_target.load_state_dict(torch.load(model_path))
                    print("Model successfully loaded.")
                except Exception as e:
                    print(f"Error loading the model: {e}")
                    exit(0)
            else:
                print("Can find model: ", model_path)
                exit(0)
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = instantiate_class_by_module_and_class_name(
                params["transformer_module"], params["transformer_class"])
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=params["learning_rate"])
        self.replay_buffer = PrioritizedReplayBuffer(capacity=1000)
        self.algo_type = params["algo_type"]
        self.batch_size = params["batch_size"]
        self.criterion = nn.MSELoss()
        self.previous_state = None
        self.previous_html = None
        self.previous_action = None
        self.state_list = []
        self.state_appear_list = []
        self.state_trans_count = defaultdict(int)
        self.action_count = defaultdict(int)
        self.action_list = []
        self.previous_max_value = 0
        self.stop_update = params["stop_update"]
        self.stop_update_round = False
        self.reward_function = params["reward_function"]
        self.gamma = params["gamma"]
        self.tensor_list = []
        self.target_list = []
        self.count = 0
        self.round = 0
        self.fail_tensor_list = []
        self.fail_reward_list = []
        self.fail_html_list = []
        self.fail_web_state_list = []
        self.fail_done_list = []
        self.previous_url = None
        self.same_url_count = 0
        self.total_count = 0
        self.start_time = datetime.now()
        self.alive_time = params["alive_time"]
        self.update_target_interval = params['update_target_interval']
        self.update_network_interval = params['update_network_interval']
        self.learn_step_count = 0
        self.max_random = params["max_random"]
        self.min_random = params["min_random"]
        self.tau = 1.0

        self.R_PENALTY = -99.0
        self.R_SAME_URL_PENALTY = -20.0

        self.R_A_PENALTY = -10.0
        self.R_A_BASE_PENALTY_LITTLE = -1.0
        self.R_A_BASE_HIGH = 50.0
        self.R_A_BASE_MIDDLE = 10.0

        self.R_A_MAX_SIM_LINE = 0.98
        self.R_A_MIDDLE_SIM_LINE = 0.85
        self.R_A_MIN_SIM_LINE = 0.7

        self.R_A_TIME_ROUND = 3600

        self.EPSILON = 0.5
        self.MAX_SAME_URL_COUNT = 30

    def get_reward(self, web_state):
        if self.reward_function == "A":
            if not isinstance(web_state, ActionSetWithExecutionTimesState):
                return self.R_A_PENALTY
            if self.state_appear_list.__contains__(web_state):
                max_sim = 1
            else:
                max_sim = -1
                for temp_state in self.state_list:
                    if (isinstance(temp_state, OutOfDomainState) or
                            isinstance(temp_state, ActionExecuteFailedState) or
                            isinstance(temp_state, SameUrlState)): continue
                    if web_state == temp_state: continue
                    if web_state.similarity(temp_state) > max_sim:
                        max_sim = web_state.similarity(temp_state)
            if max_sim > self.R_A_MAX_SIM_LINE:
                r_state = self.R_A_BASE_PENALTY_LITTLE
            elif max_sim < self.R_A_MIN_SIM_LINE:
                r_state = self.R_A_BASE_HIGH
            elif max_sim < self.R_A_MIDDLE_SIM_LINE:
                r_state = self.R_A_BASE_MIDDLE
            else:
                r_state = 1 - max_sim
            if not self.state_appear_list.__contains__(web_state):
                self.state_appear_list.append(web_state)

            if not isinstance(self.previous_state, ActionSetWithExecutionTimesState):
                r_action = 0
            else:
                action_index = self.action_list.index(self.previous_action)
                execution_time = self.action_count[action_index]
                if execution_time == 0:
                    r_action = 1
                else:
                    r_action = 1 / float(execution_time)

            if not isinstance(self.previous_state, ActionSetWithExecutionTimesState):
                r_trans = 0
            else:
                previous_state_index = self.state_list.index(self.previous_state)
                current_state_index = self.state_list.index(web_state)
                previous_action_index = self.action_list.index(self.previous_action)
                s = "{}-{}-{}".format(previous_state_index, previous_action_index, current_state_index)
                self.state_trans_count[s] += 1
                r_trans = (1 - web_state.similarity(self.previous_state)) * (1 / math.sqrt(self.state_trans_count[s]))

            r_time = 1 + float(self.total_count) / self.R_A_TIME_ROUND
            return (r_state + r_action + r_trans) * r_time
        else:
            return 0

    def get_q_value(self, web_state: WebState, action: WebAction, html: str):
        tensor = self.get_tensor(action, html, web_state)
        return self.q_eval(tensor)

    def get_tensor(self, action, html, web_state):
        state_tensor = self.transformer.state_to_tensor(web_state, html)
        action_tensor = self.transformer.action_to_tensor(web_state, action)
        tensor = torch.cat((state_tensor, action_tensor))
        tensor = tensor.float()
        return tensor

    def get_action(self, web_state: WebState, html: str) -> WebAction:
        self.state_list.append(web_state)
        actions = web_state.get_action_list()
        for temp_action in actions:
            if not self.action_list.__contains__(temp_action):
                self.action_list.append(temp_action)
        if len(actions) == 0:
            raise NoActionsException("The state does not have any actions")

        stop_update_round = False

        if (len(actions) == 1) and (isinstance(actions[0], RestartAction)):
            stop_update_round = True
            max_val = 0
            chosen_action = actions[0]
        else:
            self.q_eval.eval()
            action_tensors = []
            for temp_action in actions:
                action_tensor = self.get_tensor(temp_action, html, web_state)
                action_tensors.append(action_tensor)

            q_values = []
            if isinstance(self.q_eval, DenseNet):
                output = self.q_eval(torch.stack(action_tensors).unsqueeze(1))
            else:
                output = self.q_eval(torch.stack(action_tensors))
            for j in range(output.size(0)):
                value = output[j].item()
                q_values.append(value)
            max_val = q_values[0]
            chosen_action = actions[0]
            for i in range(0, len(actions)):
                temp_action = actions[i]
                q_value = q_values[i]
                if q_value > max_val:
                    max_val = q_value
                    chosen_action = temp_action

            end_time = datetime.now()
            time_difference = end_time - self.start_time
            seconds_difference = time_difference.total_seconds()
            # 前一半时间随时间衰减，后一半时间保持最低
            random_line = self.max_random - min(float(seconds_difference) / self.alive_time * 2, 1) * (
                    self.max_random - self.min_random)
            if random.uniform(0, 1) < random_line:
                stop_update_round = False
                chosen_action = random.choice(actions)

        if self.previous_state is not None and self.previous_action is not None and not self.stop_update and not self.stop_update_round:
            self.update(html, web_state)
        self.stop_update_round = stop_update_round
        self.previous_state = web_state
        self.previous_action = chosen_action
        self.previous_html = html
        action_index = self.action_list.index(self.previous_action)
        self.action_count[action_index] += 1
        print("max_q_value: ", max_val, "  chosen_action: ", chosen_action)

        return chosen_action

    def learn(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return  # 如果经验池中没有足够的经验，跳过学习
        self.learn_step_count += 1
        if self.learn_step_count % self.update_network_interval != 0:
            return

            # 从经验回放池中采样一批经验
        tensors, rewards, next_states, htmls, dones, weights, indices = self.replay_buffer.sample(self.batch_size)

        # 初始化一个空列表，用来保存每个样本的 TD 误差
        td_errors = []
        target_list = []

        # 逐个处理批次中的每个样本
        for i in range(self.batch_size):
            tensor = tensors[i]
            reward = rewards[i]
            next_state = next_states[i]
            html = htmls[i]
            done = dones[i]

            input_vector = tensor.unsqueeze(0)  # 如果你的 action_tensor 是一个向量，你需要给它添加一个批量维度
            output = self.q_eval(input_vector)
            current_q = output.item()
            next_q_value = 0
            if isinstance(next_state, ActionSetWithExecutionTimesState):
                actions = next_state.get_action_list()
                action_tensors = []
                for temp_action in actions:
                    action_tensor = self.get_tensor(temp_action, html, next_states[i])
                    action_tensors.append(action_tensor)

                # 计算当前Q值：将state和action拼接后作为输入传入Q网络
                if self.algo_type == "DDQN":
                    q_values = []
                    if isinstance(self.q_eval, DenseNet):
                        output = self.q_eval(torch.stack(action_tensors).unsqueeze(1))
                    else:
                        output = self.q_eval(torch.stack(action_tensors))
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
                    if isinstance(self.q_eval, DenseNet):
                        next_q_value = self.q_target(torch.stack([chosen_tensor]).unsqueeze(1)).item()
                    else:
                        next_q_value = self.q_target(torch.stack([chosen_tensor])).item()
                else:
                    q_values = []
                    if isinstance(self.q_eval, DenseNet):
                        output = self.q_target(torch.stack(action_tensors).unsqueeze(1))
                    else:
                        output = self.q_target(torch.stack(action_tensors))
                    for j in range(output.size(0)):
                        value = output[j].item()
                        q_values.append(value)
                    next_q_value = max(q_values)

            # 计算目标Q值
            target_q = reward + self.gamma * next_q_value * (1 - done)
            target_list.append(target_q)
            td_errors.append(abs(current_q - target_q))  # 计算TD误差

        self.replay_buffer.update_priorities(indices, td_errors)

        self.q_eval.train()
        input_tensor = torch.stack(tensors)
        q_predicts_tensor = self.q_eval(input_tensor)
        tensor_list = [torch.tensor([x]) for x in target_list]
        q_target_tensor = torch.stack(tensor_list)
        loss = self.criterion(q_predicts_tensor, q_target_tensor)
        print("loss:", loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, html: str, web_state: WebState):
        max_val = 0
        done = False
        if isinstance(web_state, ActionSetWithExecutionTimesState):
            actions = web_state.get_action_list()
            action_tensors = []
            for temp_action in actions:
                action_tensor = self.get_tensor(temp_action, html, web_state)
                action_tensors.append(action_tensor)
            if isinstance(self.q_eval, DenseNet):
                q_values = self.q_target(torch.stack(action_tensors).unsqueeze(1))
            else:
                q_values = self.q_target(torch.stack(action_tensors))
            max_val = max(q_values)
        else:
            done = True

        reward = self.get_reward(web_state)
        tensor = self.get_tensor(self.previous_action, self.previous_html, self.previous_state)
        tensor.unsqueeze_(0)
        self.replay_buffer.push(tensor, reward, web_state, html, done)

        # q_target = reward + self.gamma * max_val
        # print("Count:", self.count, "reward:", reward, "q_target:", q_target)
        # self.count += 1
        # self.total_count += 1
        # self.target_list.append(q_target)
        # self.tensor_list.append(tensor)
        self.update_same_url(tensor, web_state, html)

        self.learn()
        self.round += 1
        if self.round % self.update_target_interval == 0:
            self.update_network_parameters()

    def update_network_parameters(self, ):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def update_same_url(self, tensor, web_state, html):
        if (isinstance(web_state, ActionSetWithExecutionTimesState) and
                self.previous_url is not None and web_state.url == self.previous_url):
            self.same_url_count += 1
            self.fail_tensor_list.append(tensor)
            self.fail_reward_list.append(self.R_SAME_URL_PENALTY)
            self.fail_html_list.append(html)
            self.fail_web_state_list.append(web_state)
            done = True
            if isinstance(web_state, ActionSetWithExecutionTimesState):
                done = False
            self.fail_done_list.append(done)
            if self.same_url_count >= self.MAX_SAME_URL_COUNT:
                for i in range(self.same_url_count):
                    # self.replay_buffer.push(tensor, reward, web_state, html, done)
                    self.replay_buffer.push(self.fail_tensor_list[i], self.fail_reward_list[i],
                                            self.fail_web_state_list[i], self.fail_html_list[i], False)

                # self.q_eval.train()
                # input_tensor = torch.stack(self.fail_tensor_list)
                # q_predicts_tensor = self.q_eval(input_tensor)
                # tensor_list = [torch.tensor([x]) for x in self.fail_reward_list]
                # q_target_tensor = torch.stack(tensor_list)
                # loss = self.criterion(q_predicts_tensor, q_target_tensor)
                # print("same url loss: ", loss)
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

                self.fail_tensor_list.clear()
                self.fail_html_list.clear()
                self.fail_web_state_list.clear()
                self.fail_reward_list.clear()
                self.fail_done_list.clear()
                self.same_url_count = 0
        else:
            self.previous_url = None
            if isinstance(web_state, ActionSetWithExecutionTimesState):
                self.previous_url = web_state.url
            self.same_url_count = 0
            self.fail_tensor_list.clear()
            self.fail_html_list.clear()
            self.fail_web_state_list.clear()
            self.fail_reward_list.clear()
            self.fail_done_list.clear()
