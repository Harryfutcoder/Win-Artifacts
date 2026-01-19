import threading
from collections import deque

import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1e-5):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.pos = 0
        self.alpha = alpha  # 优先级的调整因子
        self.beta = beta  # 重要性采样的调整因子
        self.epsilon = epsilon  # 防止优先级为0
        self.priorities = deque(maxlen=capacity)  # 存储每条经验的优先级
        self.lock = threading.Lock()

    def push(self, tensor, best_tensor, reward, next_state, html, done):
        # 初始优先级是最大TD误差
        with self.lock:
            priority = max(self.priorities, default=1.0)  # 设置为当前最大优先级
            self.buffer.append((tensor, best_tensor, reward, next_state, html, done))
            self.priorities.append(priority)

    def sample(self, batch_size):
        with self.lock:
            # 计算采样的概率分布
            priorities = np.array(self.priorities) ** self.alpha
            prob_distribution = priorities / priorities.sum()

            # 根据概率分布采样
            indices = np.random.choice(len(self.buffer), batch_size, p=prob_distribution)
            batch = [self.buffer[idx] for idx in indices]

            # 计算每条经验的权重，重要性采样的调整
            weights = (len(self.buffer) * prob_distribution[indices]) ** (-self.beta)
            weights /= weights.max()  # 归一化

        # 提取批次中的状态、动作、奖励等信息
        tensors = [b[0] for b in batch]
        best_tensors = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3] for b in batch]
        htmls = [b[4] for b in batch]
        dones = [b[5] for b in batch]

        return tensors, best_tensors, rewards, next_states, htmls, dones, weights, indices

    def update_priorities(self, indices, priorities):
        # 更新优先级
        with self.lock:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority + self.epsilon  # 防止优先级为0


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, tensor, best_tensor, reward, next_state, html, done):
        # 将经验存储到缓冲区
        with self.lock:
            self.buffer.append((tensor, best_tensor, reward, next_state, html, done))

    def sample(self, batch_size):
        with self.lock:
            # 从缓冲区中均匀采样
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[idx] for idx in indices]

        # 提取批次中的状态、动作、奖励等信息
        tensors = [b[0] for b in batch]
        best_tensors = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3] for b in batch]
        htmls = [b[4] for b in batch]
        dones = [b[5] for b in batch]

        return tensors, best_tensors, rewards, next_states, htmls, dones

    def __len__(self):
        # 返回缓冲区中当前的经验数量
        return len(self.buffer)
