"""
JD-IQL: Jump Diffusion Independent Q-Learning (v1.0)

基于 Merton 跳跃扩散过程的多智能体探索激励算法

【算法本质】
IQL (Independent Q-Learning) + Jump Diffusion Exploration Bonus

【适用场景】
弱协作、强探索、稀疏奖励的 Web 测试场景。
相比 SHAQ，它去掉了沉重的 Mixing Network，计算效率更高。

【核心机制】
1. Q-Value SDE 建模: dQ/Q = (μ-λk)dt + σdW + (Y-1)dN
2. 探索红利: R_final = R_base + β * OptionValue(σ, λ)
3. JD-Sortino: 风险调整后的性能评估
"""

import math
import random
import threading
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Set

import torch
import torch.nn as nn
import torch.optim as optim

import multi_agent.multi_agent_system
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from model.replay_buffer import ReplayBuffer
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name
from web_test.multi_agent_thread import logger

from multi_agent.impl.shaq_v2 import DOMStructureEncoder

# ============================================================================
# Component 1: Jump Diffusion Tracker
# ============================================================================

@dataclass
class JumpDiffusionParams:
    mu: float = 0.0
    sigma: float = 0.1
    lambda_jump: float = 0.0
    lambda_pos: float = 0.0
    lambda_neg: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0


class JumpDiffusionTracker:
    """追踪 Agent Q 值的跳跃扩散参数"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
        self.q_history = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        
        self.Q_EMA_ALPHA = 0.3
        self._q_ema = 0.0
        
        self.params = JumpDiffusionParams()
        self.jd_sortino = 0.0
        self.new_state_count = 0
        self.total_steps = 0

    def update(self, q_value: float, reward: float, is_new_state: bool, found_bug: bool):
        self.total_steps += 1
        self.rewards.append(reward)
        
        if self.total_steps == 1:
            self._q_ema = q_value
        else:
            self._q_ema = self.Q_EMA_ALPHA * q_value + (1 - self.Q_EMA_ALPHA) * self._q_ema
        
        if len(self.q_history) > 0:
            prev_q = self.q_history[-1]
            if abs(prev_q) > 1e-6:
                log_return = (self._q_ema - prev_q) / (abs(prev_q) + 1e-6)
                log_return = max(-5.0, min(5.0, log_return))
                self.returns.append(log_return)
        
        self.q_history.append(self._q_ema)
        
        if is_new_state:
            self.new_state_count += 1
            
        if self.total_steps % 10 == 0 and len(self.returns) > 20:
            self._estimate_parameters()

    def _estimate_parameters(self):
        data = np.array(self.returns)
        if len(data) < 5:
            return
            
        mu = np.mean(data)
        std = np.std(data) + 1e-8
        
        threshold = 3 * std
        jumps = data[np.abs(data - mu) > threshold]
        diffusion = data[np.abs(data - mu) <= threshold]
        
        sigma = np.std(diffusion) if len(diffusion) > 0 else std
        
        n_jumps = len(jumps)
        lambda_jump = n_jumps / len(data) if len(data) > 0 else 0
        
        pos_jumps = jumps[jumps > 0]
        neg_jumps = jumps[jumps < 0]
        
        lambda_pos = len(pos_jumps) / len(data) if len(data) > 0 else 0
        lambda_neg = len(neg_jumps) / len(data) if len(data) > 0 else 0
        
        jump_mean = np.mean(jumps) if n_jumps > 0 else 0
        jump_std = np.std(jumps) if n_jumps > 0 else 0
        
        self.params = JumpDiffusionParams(
            mu=mu, sigma=sigma, 
            lambda_jump=lambda_jump,
            lambda_pos=lambda_pos, lambda_neg=lambda_neg,
            jump_mean=jump_mean, jump_std=jump_std
        )
        
        mar = 0.0
        downside_returns = data[data < mar]
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.mean(downside_returns**2))
        else:
            downside_std = 1e-6
            
        self.jd_sortino = (mu - mar) / (downside_std + 1e-6)

    def get_exploration_bonus(self) -> float:
        diff_var = self.params.sigma ** 2
        jump_var = self.params.lambda_pos * (self.params.jump_mean**2 + self.params.jump_std**2)
        implied_vol = math.sqrt(diff_var + jump_var + 1e-8)
        return implied_vol

    @property
    def mean_return(self) -> float:
        return np.mean(self.rewards) if self.rewards else 0.0

    @property
    def new_state_rate(self) -> float:
        return self.new_state_count / max(1, self.total_steps)


# ============================================================================
# Component 2: Exploration Bonus Calculator
# ============================================================================

class ExplorationBonusCalculator:
    # 【v1.1 修复】添加 Bonus 上限，防止 Reward Hacking
    # 如果 Bonus 过大，Agent 可能会倾向于制造 Q 值震荡而不是优化真正的目标
    MAX_BONUS = 5.0  # 最大红利上限
    DECAY_RATE = 0.999  # 衰减率，随时间逐渐降低红利影响
    
    def __init__(self, n_agents: int, beta: float = 1.0):
        self.trackers: Dict[str, JumpDiffusionTracker] = {
            str(i): JumpDiffusionTracker() for i in range(n_agents)
        }
        self.beta = beta
        self.step_count = 0

    def update(self, agent_name: str, q_value: float, reward: float, is_new_state: bool, found_bug: bool):
        if agent_name in self.trackers:
            self.trackers[agent_name].update(q_value, reward, is_new_state, found_bug)
        self.step_count += 1

    def compute_bonus(self, agent_name: str) -> float:
        if agent_name not in self.trackers:
            return 0.0
        
        raw_bonus = self.trackers[agent_name].get_exploration_bonus()
        
        # 1. 使用 tanh 进行初步截断
        scaled_bonus = self.beta * math.tanh(raw_bonus)
        
        # 2. 应用硬上限
        scaled_bonus = min(scaled_bonus, self.MAX_BONUS)
        
        # 3. 应用时间衰减（训练后期降低探索红利，更专注于利用）
        decay = self.DECAY_RATE ** (self.step_count / 1000)
        scaled_bonus *= decay
        
        return scaled_bonus

    def get_diagnostic(self) -> Dict:
        stats = {}
        for name, tracker in self.trackers.items():
            stats[name] = {
                'mu': tracker.params.mu,
                'sigma': tracker.params.sigma,
                'lambda_pos': tracker.params.lambda_pos,
                'lambda_neg': tracker.params.lambda_neg,
                'jd_sortino': tracker.jd_sortino,
                'bonus': tracker.get_exploration_bonus(),
                'mean_return': tracker.mean_return,
                'new_state_rate': tracker.new_state_rate
            }
        return stats


# ============================================================================
# Main Class: JD-IQL
# ============================================================================

class JDIQL(multi_agent.multi_agent_system.MultiAgentSystem):
    """
    JD-IQL: Jump Diffusion Independent Q-Learning
    架构: IQL + JD Exploration Bonus
    """
    
    def __init__(self, params: Dict):
        super().__init__(params)
        self.params = params
        self.agent_num = params.get("agent_num", 5)
        self.beta = params.get("jd_beta", 1.0)
        self.batch_size = params.get("batch_size", 32)
        self.gamma = params.get("gamma", 0.99)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.target_update_freq = params.get("target_update_freq", 100)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"JD-IQL initialized on {self.device}")
        
        # Agent 网络
        self.q_eval_agent = {}
        self.q_target_agent = {}
        self.agent_optimizer = {}
        self.state_list_agent = {}
        
        net_module = params.get("model_module", "model.dense_net")
        net_class = params.get("model_class", "DenseNet")
        
        for i in range(self.agent_num):
            name = str(i)
            q_eval = instantiate_class_by_module_and_class_name(net_module, net_class, params).to(self.device)
            q_target = instantiate_class_by_module_and_class_name(net_module, net_class, params).to(self.device)
            q_target.load_state_dict(q_eval.state_dict())
            
            self.q_eval_agent[name] = q_eval
            self.q_target_agent[name] = q_target
            self.agent_optimizer[name] = optim.Adam(q_eval.parameters(), lr=self.learning_rate)
            self.state_list_agent[name] = []
        
        # JD 组件
        self.bonus_calculator = ExplorationBonusCalculator(n_agents=self.agent_num, beta=self.beta)
        
        # 经验回放
        self.replay_buffer_agent: Dict[str, ReplayBuffer] = {
            str(i): ReplayBuffer(capacity=1000) for i in range(self.agent_num)
        }
        
        # 状态追踪
        self.learn_step_count = 0
        self.state_list = []
        self.action_list = []
        self.prev_state_dict = {}
        self.prev_action_dict = {}
        self.prev_html_dict = {}
        
        self.dom_encoder = DOMStructureEncoder()
        self._seen_dom_signatures = set()
        
        self.criterion = nn.MSELoss()
        self.start_time = datetime.now()
        self.duration = params.get("alive_time", 300)
        
        # Transformer 初始化
        transformer_module = params.get("transformer_module", "transformer.impl.tag_transformer")
        transformer_class = params.get("transformer_class", "TagTransformer")
        self.transformer = instantiate_class_by_module_and_class_name(
            transformer_module, transformer_class, params
        )
        
        # ε-greedy 参数
        self.max_random = params.get("max_random", 0.9)
        self.min_random = params.get("min_random", 0.1)
        
        # 动作字典
        self.action_dict = {}
        
        # 统计
        self.bonus_history = []
        
        logger.info(f"JD-IQL v1.0 initialized: {self.agent_num} agents, β={self.beta}")

    def get_tensor(self, action: WebAction, html: str, web_state: WebState) -> torch.Tensor:
        """将动作转换为张量"""
        return self.transformer.transform(action, html, web_state)

    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        """
        动作选择 - ε-greedy with Q-Network
        """
        self.update_state_records(web_state, html, agent_name)
        
        actions = web_state.get_action_list()
        if len(actions) == 1 and isinstance(actions[0], RestartAction):
            return actions[0]
        
        q_eval = self.q_eval_agent[agent_name]
        q_eval.eval()
        
        # 计算每个动作的 Q 值
        action_tensors = []
        for temp_action in actions:
            action_tensor = self.get_tensor(temp_action, html, web_state)
            action_tensors.append(action_tensor)
        
        q_values = []
        from model.dense_net import DenseNet
        if isinstance(q_eval, DenseNet):
            for action_tensor in action_tensors:
                with torch.no_grad():
                    q = q_eval(action_tensor.unsqueeze(0).unsqueeze(1).to(self.device))
                    q_values.append(q.item())
        else:
            for action_tensor in action_tensors:
                with torch.no_grad():
                    q = q_eval(action_tensor.unsqueeze(0).to(self.device))
                    q_values.append(q.item())
        
        # ε-greedy 策略
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = min(1.0, elapsed / self.duration)
        epsilon = self.max_random - (self.max_random - self.min_random) * progress
        
        if random.random() < epsilon:
            chosen_action = random.choice(actions)
        else:
            best_idx = np.argmax(q_values)
            chosen_action = actions[best_idx]
        
        # 记录动作
        if chosen_action not in self.action_dict:
            self.action_dict[chosen_action] = len(self.action_dict)
        
        return chosen_action

    def update_state_records(self, web_state: WebState, html: str, agent_name: str):
        """更新状态记录并执行学习"""
        # 更新全局状态列表
        if web_state not in self.state_list:
            self.state_list.append(web_state)
        if web_state not in self.state_list_agent[agent_name]:
            self.state_list_agent[agent_name].append(web_state)
        
        for action in web_state.get_action_list():
            if action not in self.action_list:
                self.action_list.append(action)
        
        # 检查前置条件
        if (self.prev_action_dict.get(agent_name) is None or
            self.prev_state_dict.get(agent_name) is None or
            not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState)):
            return
        
        # 计算状态特征
        state_hash = self.dom_encoder.compute_structure_hash(html) if html else ""
        is_new_state = state_hash and hash(state_hash) not in self._seen_dom_signatures
        if is_new_state and state_hash:
            self._seen_dom_signatures.add(hash(state_hash))
        
        found_bug = False
        
        # 计算奖励
        base_reward = self._compute_base_reward(web_state, agent_name, is_new_state, found_bug)
        
        # 获取 Q 值并更新 JD Tracker
        q_value = self._get_current_q_value(web_state, agent_name)
        self.bonus_calculator.update(agent_name, q_value, base_reward, is_new_state, found_bug)
        
        # 计算探索红利
        exploration_bonus = self.bonus_calculator.compute_bonus(agent_name)
        final_reward = base_reward + exploration_bonus
        
        # 记录统计
        if exploration_bonus > 0.01:
            self.bonus_history.append(exploration_bonus)
            if len(self.bonus_history) > 1000:
                self.bonus_history.pop(0)
        
        # 存入 Buffer
        try:
            prev_state_tensor = self.prev_state_dict[agent_name].to_tensor(self.device)
            curr_state_tensor = web_state.to_tensor(self.device)
            
            self.replay_buffer_agent[agent_name].push(
                prev_state_tensor, None, final_reward, curr_state_tensor, False
            )
        except Exception as e:
            logger.debug(f"Buffer push error: {e}")
        
        # 执行学习
        self._learn(agent_name)

    def _compute_base_reward(self, web_state: WebState, agent_name: str, 
                             is_new_state: bool, found_bug: bool) -> float:
        reward = 0.0
        
        # 新状态奖励
        if is_new_state:
            reward += 10.0
        
        # URL 变化奖励
        prev_state = self.prev_state_dict.get(agent_name)
        if hasattr(web_state, 'url') and hasattr(prev_state, 'url'):
            if web_state.url != prev_state.url:
                reward += 5.0
        
        # 动作多样性
        if isinstance(web_state, ActionSetWithExecutionTimesState):
            action_count = len(web_state.get_action_list())
            reward += min(action_count * 0.1, 5.0)
        
        # Bug 发现
        if found_bug:
            reward += 50.0
        
        return reward

    def _get_current_q_value(self, web_state: WebState, agent_name: str) -> float:
        try:
            state_tensor = web_state.to_tensor(self.device)
            with torch.no_grad():
                q_net = self.q_eval_agent[agent_name]
                from model.dense_net import DenseNet
                if isinstance(q_net, DenseNet):
                    q_values = q_net(state_tensor.unsqueeze(0).unsqueeze(1))
                else:
                    q_values = q_net(state_tensor.unsqueeze(0))
                return q_values.max().item()
        except Exception:
            return 0.0

    def _learn(self, agent_name: str):
        """独立 Q-Learning 更新"""
        buffer = self.replay_buffer_agent.get(agent_name)
        if buffer is None or len(buffer) < self.batch_size:
            return
        
        q_eval = self.q_eval_agent.get(agent_name)
        q_target = self.q_target_agent.get(agent_name)
        optimizer = self.agent_optimizer.get(agent_name)
        
        if q_eval is None or q_target is None or optimizer is None:
            return
        
        try:
            batch = buffer.sample(self.batch_size)
            
            states = torch.stack([b[0].squeeze(0) for b in batch]).to(self.device)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
            
            from model.dense_net import DenseNet
            if isinstance(q_eval, DenseNet):
                q_values = q_eval(states.unsqueeze(1))
            else:
                q_values = q_eval(states)
            
            with torch.no_grad():
                if isinstance(q_target, DenseNet):
                    next_q = q_target(states.unsqueeze(1))
                else:
                    next_q = q_target(states)
                target_q = rewards + self.gamma * next_q
            
            loss = self.criterion(q_values, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_eval.parameters(), 10.0)
            optimizer.step()
            
            self.learn_step_count += 1
            
            # 更新目标网络
            if self.learn_step_count % self.target_update_freq == 0:
                self._update_target_networks()
            
        except Exception as e:
            logger.debug(f"Learning error: {e}")

    def _update_target_networks(self):
        for agent_name in self.q_eval_agent:
            self.q_target_agent[agent_name].load_state_dict(
                self.q_eval_agent[agent_name].state_dict()
            )

    def set_prev(self, agent_name: str, state: WebState, action: WebAction, html: str):
        self.prev_state_dict[agent_name] = state
        self.prev_action_dict[agent_name] = action
        self.prev_html_dict[agent_name] = html

    def get_reward(self, web_state: WebState, html: str, agent_name: str) -> float:
        base = self._compute_base_reward(web_state, agent_name, False, False)
        bonus = self.bonus_calculator.compute_bonus(agent_name)
        return base + bonus

    def get_diagnostic_report(self) -> str:
        lines = [
            "=" * 60,
            "JD-IQL Diagnostic Report (v1.0)",
            "=" * 60,
            f"Architecture: IQL + JD Exploration Bonus",
            f"Agents: {self.agent_num}",
            f"Learn Steps: {self.learn_step_count}",
            f"Avg Bonus: {np.mean(self.bonus_history) if self.bonus_history else 0:.4f}",
            "",
            "--- Agent JD Parameters ---"
        ]
        
        diag = self.bonus_calculator.get_diagnostic()
        for name, d in diag.items():
            lines.append(f"Agent {name}:")
            lines.append(f"  μ={d['mu']:.4f}, σ={d['sigma']:.4f}")
            lines.append(f"  λ⁺={d['lambda_pos']:.4f}, λ⁻={d['lambda_neg']:.4f}")
            lines.append(f"  JD-Sortino={d['jd_sortino']:.3f}")
            lines.append(f"  Bonus={d['bonus']:.4f}")
        
        return "\n".join(lines)
