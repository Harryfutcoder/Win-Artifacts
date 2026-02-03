"""
JD-PSHAQ: Jump Diffusion Portfolio-SHAQ (v3.2 - Full Theory)

【算法本质】
SHAQ (Shapley Q-Learning) + Jump Diffusion Exploration Bonus

【适用场景】
强协作、理论验证、高精度信度分配场景。
这是一个"重型"算法，计算开销大，但理论性质完备。

【核心机制】
1. 联合价值函数: Q_tot(s, a) = MixingNetwork(Q_1, ..., Q_n)
2. 真实 Shapley Value: ϕ_i = ∂Q_tot / ∂Q_i (Lovász Extension)
3. 期权探索红利: Bonus_i = OptionValue(σ_i, λ_i)
4. 双重信度分配: Weight_i = Softmax(ϕ_i + Bonus_i)

【与 JD-IQL 的区别】
- JD-IQL: 独立训练，无 Mixing Network，计算高效
- JD-PSHAQ: 联合训练，有 Mixing Network，理论完备
"""

import math
import random
import threading
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import multi_agent.multi_agent_system
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from model.replay_buffer import ReplayBuffer
from model.mixing_network import ShapleyMixingNetwork
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name
from web_test.multi_agent_thread import logger

# 复用 JD-IQL 的数学组件
from multi_agent.impl.jd_iql import JumpDiffusionTracker, ExplorationBonusCalculator
from multi_agent.impl.shaq_v2 import DOMStructureEncoder


class JDPSHAQ(multi_agent.multi_agent_system.MultiAgentSystem):
    """
    JD-PSHAQ: Jump Diffusion Portfolio-SHAQ
    架构: SHAQ (Joint Training) + JD Exploration Bonus
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
        logger.info(f"JD-PSHAQ (Full) initialized on {self.device}")
        
        # ========== SHAQ 核心：Mixing Network ==========
        self.mixing_network = ShapleyMixingNetwork(
            n_agents=self.agent_num, embed_dim=64
        ).to(self.device)
        self.target_mixing_network = ShapleyMixingNetwork(
            n_agents=self.agent_num, embed_dim=64
        ).to(self.device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=self.learning_rate)
        
        # ========== Agent 网络 ==========
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
        
        # ========== JD 组件 ==========
        self.bonus_calculator = ExplorationBonusCalculator(n_agents=self.agent_num, beta=self.beta)
        
        # ========== Shapley 值缓存 ==========
        self.shapley_values_cache = torch.zeros(self.agent_num, device=self.device)
        
        # ========== 经验回放 ==========
        self.replay_buffer_agent: Dict[str, ReplayBuffer] = {
            str(i): ReplayBuffer(capacity=1000) for i in range(self.agent_num)
        }
        
        # ========== 状态追踪 ==========
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
        
        # 统计
        self.bonus_history = []
        self.shapley_history = {str(i): [] for i in range(self.agent_num)}

    def update_state_records(self, web_state: WebState, html: str, agent_name: str):
        """更新状态记录"""
        if web_state not in self.state_list:
            self.state_list.append(web_state)
        if web_state not in self.state_list_agent[agent_name]:
            self.state_list_agent[agent_name].append(web_state)
        
        for action in web_state.get_action_list():
            if action not in self.action_list:
                self.action_list.append(action)
        
        if (self.prev_action_dict.get(agent_name) is None or
            self.prev_state_dict.get(agent_name) is None or
            not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState)):
            return
        
        state_hash = self.dom_encoder.compute_structure_hash(html) if html else ""
        is_new_state = state_hash and hash(state_hash) not in self._seen_dom_signatures
        if is_new_state and state_hash:
            self._seen_dom_signatures.add(hash(state_hash))
        
        found_bug = False
        
        # 计算奖励
        base_reward = self._compute_base_reward(web_state, agent_name, is_new_state, found_bug)
        
        # 更新 JD Tracker
        q_value = self._get_current_q_value(web_state, agent_name)
        self.bonus_calculator.update(agent_name, q_value, base_reward, is_new_state, found_bug)
        
        # 探索红利
        exploration_bonus = self.bonus_calculator.compute_bonus(agent_name)
        
        # 【v3.2 关键修复】应用 Shapley 权重到基础奖励
        # 这是 Portfolio-SHAQ 的核心：贡献大的 Agent 获得更高奖励
        # 之前的代码直接 final_reward = base_reward + bonus，完全忽略了 Shapley 权重！
        shapley_weights = F.softmax(self.shapley_values_cache, dim=0)
        weight = shapley_weights[int(agent_name)].item()
        
        # 最终奖励 = (基础奖励 × 贡献权重 × Agent数量) + 探索红利
        # 乘以 agent_num 是为了保持奖励总量级不变
        final_reward = (base_reward * weight * self.agent_num) + exploration_bonus
        
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
        
        # 联合学习 (只在 Agent 0 触发，确保同步)
        if agent_name == "0":
            self._learn_joint()

    def _compute_base_reward(self, web_state: WebState, agent_name: str, 
                             is_new_state: bool, found_bug: bool) -> float:
        reward = 0.0
        
        if is_new_state:
            reward += 10.0
        
        prev_state = self.prev_state_dict.get(agent_name)
        if hasattr(web_state, 'url') and hasattr(prev_state, 'url'):
            if web_state.url != prev_state.url:
                reward += 5.0
        
        if isinstance(web_state, ActionSetWithExecutionTimesState):
            action_count = len(web_state.get_action_list())
            reward += min(action_count * 0.1, 5.0)
        
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

    def _learn_joint(self):
        """
        联合训练核心逻辑 (SHAQ + JD Bonus)
        使用 Mixing Network 计算 Q_tot 并回传梯度
        """
        # 检查所有 Buffer 是否足够
        for i in range(self.agent_num):
            if len(self.replay_buffer_agent[str(i)]) < self.batch_size:
                return

        try:
            # 1. 采样数据
            agent_batches = {}
            for i in range(self.agent_num):
                agent_batches[str(i)] = self.replay_buffer_agent[str(i)].sample(self.batch_size)

            # 2. 收集所有 Agent 的 Q 值
            all_q_values = []
            all_next_q_values = []
            total_rewards = torch.zeros(self.batch_size, 1, device=self.device)
            
            for i in range(self.agent_num):
                name = str(i)
                batch = agent_batches[name]
                
                states = torch.stack([b[0].squeeze(0) for b in batch]).to(self.device)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
                
                total_rewards += rewards
                
                q_net = self.q_eval_agent[name]
                target_net = self.q_target_agent[name]
                
                from model.dense_net import DenseNet
                if isinstance(q_net, DenseNet):
                    q = q_net(states.unsqueeze(1))
                else:
                    q = q_net(states)
                all_q_values.append(q)
                
                with torch.no_grad():
                    if isinstance(target_net, DenseNet):
                        nq = target_net(states.unsqueeze(1))
                    else:
                        nq = target_net(states)
                    all_next_q_values.append(nq)

            # [batch_size, n_agents]
            q_batch = torch.cat(all_q_values, dim=1)
            next_q_batch = torch.cat(all_next_q_values, dim=1)
            
            # 3. 计算 Shapley 值 (在 w=0.5 处计算梯度)
            w_shapley = torch.full(
                (self.batch_size, self.agent_num), 0.5, 
                device=self.device, requires_grad=True
            )
            q_tot_shapley = self.mixing_network(q_batch.detach(), w_shapley)
            q_tot_shapley.sum().backward()
            
            shapley_values = w_shapley.grad.mean(dim=0)
            self.shapley_values_cache = 0.9 * self.shapley_values_cache + 0.1 * shapley_values.detach()
            
            # 记录 Shapley 历史
            for i in range(self.agent_num):
                self.shapley_history[str(i)].append(self.shapley_values_cache[i].item())
                if len(self.shapley_history[str(i)]) > 100:
                    self.shapley_history[str(i)].pop(0)
            
            # 4. 计算 Q_tot (用于训练)
            participation = torch.ones(self.batch_size, self.agent_num, device=self.device)
            q_tot = self.mixing_network(q_batch, participation)
            
            with torch.no_grad():
                target_q_tot = self.target_mixing_network(next_q_batch, participation)
                y_tot = total_rewards + self.gamma * target_q_tot

            # 5. 计算 Loss 并反向传播
            loss = self.criterion(q_tot, y_tot)
            
            self.mixing_optimizer.zero_grad()
            for i in range(self.agent_num):
                self.agent_optimizer[str(i)].zero_grad()
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 10.0)
            for i in range(self.agent_num):
                torch.nn.utils.clip_grad_norm_(self.q_eval_agent[str(i)].parameters(), 10.0)
            
            self.mixing_optimizer.step()
            for i in range(self.agent_num):
                self.agent_optimizer[str(i)].step()
            
            self.learn_step_count += 1
            
            # 更新目标网络
            if self.learn_step_count % self.target_update_freq == 0:
                self._update_target_networks()

        except Exception as e:
            logger.debug(f"Joint learning error: {e}")
            import traceback
            traceback.print_exc()

    def _update_target_networks(self):
        for agent_name in self.q_eval_agent:
            self.q_target_agent[agent_name].load_state_dict(
                self.q_eval_agent[agent_name].state_dict()
            )
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def set_prev(self, agent_name: str, state: WebState, action: WebAction, html: str):
        self.prev_state_dict[agent_name] = state
        self.prev_action_dict[agent_name] = action
        self.prev_html_dict[agent_name] = html

    def get_reward(self, web_state: WebState, html: str, agent_name: str) -> float:
        base = self._compute_base_reward(web_state, agent_name, False, False)
        bonus = self.bonus_calculator.compute_bonus(agent_name)
        
        # 使用 Shapley 值计算权重分配
        shapley = self.shapley_values_cache[int(agent_name)].item()
        shapley_weights = F.softmax(self.shapley_values_cache, dim=0)
        weight = shapley_weights[int(agent_name)].item()
        
        # 最终奖励 = 权重 * 基础奖励 + 探索红利
        return base * weight * self.agent_num + bonus

    def get_diagnostic_report(self) -> str:
        lines = [
            "=" * 60,
            "JD-PSHAQ Diagnostic Report (v3.2 Full)",
            "=" * 60,
            f"Architecture: SHAQ (Joint Training) + JD Exploration Bonus",
            f"Agents: {self.agent_num}",
            f"Learn Steps: {self.learn_step_count}",
            f"Avg Bonus: {np.mean(self.bonus_history) if self.bonus_history else 0:.4f}",
            "",
            "--- Shapley Values ---"
        ]
        
        shapley_weights = F.softmax(self.shapley_values_cache, dim=0)
        for i in range(self.agent_num):
            raw = self.shapley_values_cache[i].item()
            weight = shapley_weights[i].item()
            lines.append(f"Agent {i}: Raw={raw:.4f}, Weight={weight:.2%}")
        
        lines.append("")
        lines.append("--- JD Parameters ---")
        
        diag = self.bonus_calculator.get_diagnostic()
        for name, d in diag.items():
            lines.append(f"Agent {name}:")
            lines.append(f"  μ={d['mu']:.4f}, σ={d['sigma']:.4f}")
            lines.append(f"  λ⁺={d['lambda_pos']:.4f}, λ⁻={d['lambda_neg']:.4f}")
            lines.append(f"  JD-Sortino={d['jd_sortino']:.3f}, Bonus={d['bonus']:.4f}")
        
        return "\n".join(lines)
