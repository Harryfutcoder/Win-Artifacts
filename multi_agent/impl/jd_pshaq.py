"""
JD-PSHAQ: Jump Diffusion Portfolio-SHAQ

【核心思想】
Agent = 个股，MARL奖励分配 = 持仓管理

通过量化金融的视角来做信度分配：
1. Shapley 值 = 历史贡献 → 基础持仓
2. JD-Sortino = 风险调整收益 → 风险指标（只惩罚下行风险）
3. Option Value = 探索潜力 → 预期指标（高波动=高期权价值）
4. 最终持仓 = Shapley × Softmax(α·Sortino + β·Option + γ·Info)

【Web Testing 特性】
- 奖励稀疏：大部分时间奖励为0
- Jump：发现Bug时奖励猛增
- Bug分布：假设服从泊松过程

【与 SHAQv2 的关系】
继承 SHAQv2，添加"持仓分配"机制：
- SHAQv2 提供 Shapley 值计算
- JD-PSHAQ 添加 JD 参数估计 + 持仓权重计算 + 按持仓分配奖励
"""

import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import torch
import torch.nn.functional as F

from multi_agent.impl.shaq_v2 import SHAQv2
from state.web_state import WebState
from action.web_action import WebAction
from web_test.multi_agent_thread import logger


# ============================================================================
# Component 1: Jump Diffusion Tracker (跳跃扩散参数估计)
# ============================================================================

@dataclass
class JDParams:
    """跳跃扩散参数"""
    mu: float = 0.0           # 漂移率（平均收益）
    sigma: float = 0.1        # 扩散波动率（常规波动）
    lambda_jump: float = 0.0  # 跳跃强度（总体）
    lambda_pos: float = 0.0   # 正向跳跃强度（发现Bug）
    lambda_neg: float = 0.0   # 负向跳跃强度（失败）
    jump_mean: float = 0.0    # 跳跃幅度均值
    jump_std: float = 0.0     # 跳跃幅度标准差


class JumpDiffusionTracker:
    """
    追踪单个 Agent 的 Q 值变化，估计跳跃扩散参数
    
    类比：追踪单只股票的收益率分布
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)  # Q值收益率序列
        self.q_history = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        
        # EMA 平滑
        self.Q_EMA_ALPHA = 0.3
        self._q_ema = 0.0
        
        # 估计的参数
        self.params = JDParams()
        
        # 量化指标
        self.jd_sortino = 0.0      # JD-Sortino 比率
        self.option_value = 0.0    # 期权价值（探索潜力）
        self.info_premium = 0.0    # 信息溢价
        
        # 统计
        self.new_state_count = 0
        self.total_steps = 0
        self.jump_count = 0

    def update(self, q_value: float, reward: float, is_new_state: bool):
        """更新跟踪器"""
        self.total_steps += 1
        self.rewards.append(reward)
        
        # EMA 平滑 Q 值
        if self.total_steps == 1:
            self._q_ema = q_value
        else:
            self._q_ema = self.Q_EMA_ALPHA * q_value + (1 - self.Q_EMA_ALPHA) * self._q_ema
        
        # 计算收益率 r_t = (Q_t - Q_{t-1}) / |Q_{t-1}|
        if len(self.q_history) > 0:
            prev_q = self.q_history[-1]
            if abs(prev_q) > 1e-6:
                ret = (self._q_ema - prev_q) / (abs(prev_q) + 1e-6)
                ret = max(-5.0, min(5.0, ret))  # 裁剪极端值
                self.returns.append(ret)
        
        self.q_history.append(self._q_ema)
        
        if is_new_state:
            self.new_state_count += 1
        
        # 定期重新估计参数
        if self.total_steps % 10 == 0 and len(self.returns) > 20:
            self._estimate_parameters()

    def _estimate_parameters(self):
        """
        矩估计法估计跳跃扩散参数
        
        基于 3σ 规则区分：
        - |r - μ| ≤ 3σ → 扩散（Diffusion）
        - |r - μ| > 3σ → 跳跃（Jump）
        """
        data = np.array(self.returns)
        if len(data) < 5:
            return
        
        # 1. 估计基本统计量
        mu = np.mean(data)
        std = np.std(data) + 1e-8
        
        # 2. 分离跳跃和扩散
        threshold = 3 * std
        jumps = data[np.abs(data - mu) > threshold]
        diffusion = data[np.abs(data - mu) <= threshold]
        
        # 3. 估计参数
        sigma = np.std(diffusion) if len(diffusion) > 0 else std
        
        n_jumps = len(jumps)
        self.jump_count = n_jumps
        lambda_jump = n_jumps / len(data) if len(data) > 0 else 0
        
        # 区分正负跳跃（正=发现Bug，负=失败）
        pos_jumps = jumps[jumps > 0]
        neg_jumps = jumps[jumps < 0]
        
        lambda_pos = len(pos_jumps) / len(data) if len(data) > 0 else 0
        lambda_neg = len(neg_jumps) / len(data) if len(data) > 0 else 0
        
        jump_mean = np.mean(jumps) if n_jumps > 0 else 0
        jump_std = np.std(jumps) if n_jumps > 0 else 0
        
        self.params = JDParams(
            mu=mu, sigma=sigma,
            lambda_jump=lambda_jump,
            lambda_pos=lambda_pos, lambda_neg=lambda_neg,
            jump_mean=jump_mean, jump_std=jump_std
        )
        
        # 4. 计算量化指标
        self._compute_metrics(data, mu)

    def _compute_metrics(self, data: np.ndarray, mu: float):
        """计算量化金融指标"""
        
        # === JD-Sortino 比率 ===
        # 只惩罚下行风险，不惩罚好的波动（发现Bug）
        mar = 0.0  # Minimum Acceptable Return
        downside = data[data < mar]
        if len(downside) > 0:
            downside_std = np.sqrt(np.mean(downside**2))
        else:
            downside_std = 1e-6
        self.jd_sortino = (mu - mar) / (downside_std + 1e-6)
        
        # === Option Value（期权价值 = 探索潜力）===
        # 高波动 + 正向跳跃潜力 = 高期权价值
        # 简化 Merton 公式: Option ≈ λ⁺ × E[J⁺] + 0.5 × σ
        diff_component = 0.5 * self.params.sigma
        jump_component = self.params.lambda_pos * max(0, self.params.jump_mean)
        self.option_value = diff_component + jump_component
        
        # === 信息溢价 ===
        # 新状态发现率
        self.info_premium = self.new_state_count / max(1, self.total_steps)


# ============================================================================
# Component 2: Portfolio Manager (持仓管理器)
# ============================================================================

class PortfolioManager:
    """
    持仓管理器：将 Shapley 值 + JD 指标转换为持仓权重
    
    类比：基金经理根据股票的历史表现和风险指标来调整持仓
    """
    
    def __init__(self, n_agents: int, alpha: float = 0.3, beta: float = 0.3, 
                 gamma: float = 0.2, temperature: float = 1.0):
        """
        Args:
            n_agents: Agent 数量
            alpha: JD-Sortino 权重（风险调整收益）
            beta: Option Value 权重（探索潜力）
            gamma: Info Premium 权重（信息价值）
            temperature: Softmax 温度（τ→∞均匀，τ→0赢家通吃）
        """
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        
        # 每个 Agent 的 JD Tracker
        self.trackers: Dict[str, JumpDiffusionTracker] = {
            str(i): JumpDiffusionTracker() for i in range(n_agents)
        }
        
        # 当前持仓权重
        self.weights: Dict[str, float] = {
            str(i): 1.0 / n_agents for i in range(n_agents)
        }
        
        # 历史记录
        self.weight_history: List[Dict[str, float]] = []

    def update_tracker(self, agent_name: str, q_value: float, reward: float, is_new_state: bool):
        """更新指定 Agent 的 JD Tracker"""
        if agent_name in self.trackers:
            self.trackers[agent_name].update(q_value, reward, is_new_state)

    def compute_weights(self, shapley_values: Dict[str, float]) -> Dict[str, float]:
        """
        计算持仓权重
        
        公式：
        score_i = α × Sortino_i + β × Option_i + γ × Info_i
        w_i = max(0, φ_i + ε) × exp((score_i - max(score)) / τ)
        w_i = w_i / Σw_j  (归一化)
        """
        scores = {}
        for name, tracker in self.trackers.items():
            # 综合评分 = 风险调整收益 + 探索潜力 + 信息价值
            score = (self.alpha * tracker.jd_sortino +
                     self.beta * tracker.option_value +
                     self.gamma * tracker.info_premium)
            scores[name] = score
        
        # Softmax with temperature
        max_score = max(scores.values()) if scores else 0
        
        raw_weights = {}
        for name in self.trackers:
            # 基础持仓 = Shapley 值（历史贡献）
            phi = shapley_values.get(name, 1.0 / self.n_agents)
            phi = max(0, phi + 1e-6)  # 确保非负
            
            # 调整因子 = Softmax(score)
            score = scores.get(name, 0)
            adjustment = math.exp((score - max_score) / (self.temperature + 1e-6))
            
            # 最终权重 = 基础持仓 × 调整因子
            raw_weights[name] = phi * adjustment
        
        # 归一化
        total = sum(raw_weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in raw_weights.items()}
        else:
            self.weights = {str(i): 1.0 / self.n_agents for i in range(self.n_agents)}
        
        # 记录历史
        self.weight_history.append(self.weights.copy())
        if len(self.weight_history) > 1000:
            self.weight_history.pop(0)
        
        return self.weights

    def allocate_reward(self, total_reward: float) -> Dict[str, float]:
        """
        按持仓比例分配总奖励
        
        R_i = w_i × R_total
        """
        return {name: w * total_reward for name, w in self.weights.items()}

    def get_diagnostic(self) -> Dict:
        """获取诊断信息"""
        diag = {}
        for name, tracker in self.trackers.items():
            diag[name] = {
                'weight': self.weights.get(name, 0),
                'mu': tracker.params.mu,
                'sigma': tracker.params.sigma,
                'lambda_pos': tracker.params.lambda_pos,
                'lambda_neg': tracker.params.lambda_neg,
                'jd_sortino': tracker.jd_sortino,
                'option_value': tracker.option_value,
                'info_premium': tracker.info_premium,
                'jump_count': tracker.jump_count
            }
        return diag


# ============================================================================
# Main Class: JD-PSHAQ
# ============================================================================

class JDPSHAQ(SHAQv2):
    """
    JD-PSHAQ: Jump Diffusion Portfolio-SHAQ
    
    核心思想：Agent = 个股，奖励分配 = 持仓管理
    
    1. 继承 SHAQv2 获取 Shapley 值（历史贡献）
    2. 用 JD 模型估计每个 Agent 的风险/收益特征
    3. 计算持仓权重 = Shapley × Softmax(风险指标)
    4. 按持仓比例分配团队奖励
    """
    
    def __init__(self, params: Dict):
        # 调用父类初始化（SHAQv2）
        super().__init__(params)
        
        # JD-PSHAQ 参数
        self.jd_alpha = params.get("alpha", 0.3)        # Sortino 权重
        self.jd_beta = params.get("beta", 0.3)          # Option 权重
        self.jd_gamma = params.get("gamma_info", 0.2)   # Info 权重
        self.jd_temperature = params.get("temperature", 1.0)
        
        # 持仓管理器
        self.portfolio = PortfolioManager(
            n_agents=self.agent_num,
            alpha=self.jd_alpha,
            beta=self.jd_beta,
            gamma=self.jd_gamma,
            temperature=self.jd_temperature
        )
        
        # 权重更新频率
        self.weight_update_interval = params.get("weight_update_interval", 10)
        self.weight_update_counter = 0
        
        logger.info(f"JD-PSHAQ initialized: {self.agent_num} agents, "
                   f"α={self.jd_alpha}, β={self.jd_beta}, γ={self.jd_gamma}, τ={self.jd_temperature}")

    def update_state_records(self, web_state: WebState, html: str, agent_name: str):
        """
        重写状态更新，添加 JD 追踪
        """
        # 1. 调用父类方法（SHAQv2 的学习逻辑）
        super().update_state_records(web_state, html, agent_name)
        
        # 2. 获取当前 Q 值
        try:
            q_value = self._get_current_q_value(web_state, agent_name)
        except Exception:
            q_value = 0.0
        
        # 3. 检测新状态
        state_hash = self.dom_encoder.compute_structure_hash(html) if html else ""
        is_new_state = state_hash and hash(state_hash) not in self._seen_dom_signatures
        
        # 4. 获取基础奖励
        base_reward = super().get_reward(web_state, html, agent_name)
        
        # 5. 更新 JD Tracker
        self.portfolio.update_tracker(agent_name, q_value, base_reward, is_new_state)
        
        # 6. 定期更新持仓权重
        self.weight_update_counter += 1
        if self.weight_update_counter % self.weight_update_interval == 0:
            self.portfolio.compute_weights(self.cached_shapley_values)

    def _get_current_q_value(self, web_state: WebState, agent_name: str) -> float:
        """获取当前状态的 Q 值"""
        try:
            actions = web_state.get_action_list()
            if not actions:
                return 0.0
            
            q_eval = self.q_eval_agent[agent_name]
            q_eval.eval()
            
            action_tensors = []
            for action in actions[:10]:
                action_tensor = self.get_tensor(action, "", web_state)
                action_tensors.append(action_tensor)
            
            with torch.no_grad():
                from model.dense_net import DenseNet
                if isinstance(q_eval, DenseNet):
                    output = q_eval(torch.stack(action_tensors).unsqueeze(1).to(self.device))
                else:
                    output = q_eval(torch.stack(action_tensors).to(self.device))
            
            return output.max().item()
        except Exception:
            return 0.0

    def get_reward(self, web_state: WebState, html: str, agent_name: str) -> float:
        """
        重写奖励函数：按持仓分配
        
        1. 获取团队基础奖励
        2. 按持仓权重分配给当前 Agent
        """
        # 获取基础奖励
        base_reward = super().get_reward(web_state, html, agent_name)
        
        # 获取当前 Agent 的持仓权重
        weight = self.portfolio.weights.get(agent_name, 1.0 / self.agent_num)
        avg_weight = 1.0 / self.agent_num
        
        # 按持仓调整奖励
        # 高持仓 Agent 获得更多奖励，低持仓获得更少
        # 使用相对权重：weight / avg_weight
        weight_factor = weight / avg_weight if avg_weight > 0 else 1.0
        weight_factor = max(0.5, min(2.0, weight_factor))  # 限制范围 [0.5, 2.0]
        
        adjusted_reward = base_reward * weight_factor
        
        return adjusted_reward

    def get_portfolio_summary(self) -> str:
        """获取持仓摘要"""
        lines = [
            "=" * 60,
            "JD-PSHAQ Portfolio Summary",
            "=" * 60,
            f"Agents: {self.agent_num}",
            f"Parameters: α={self.jd_alpha}, β={self.jd_beta}, γ={self.jd_gamma}, τ={self.jd_temperature}",
            "",
            "--- Current Holdings ---"
        ]
        
        diag = self.portfolio.get_diagnostic()
        for name, d in diag.items():
            shapley = self.cached_shapley_values.get(name, 1.0/self.agent_num)
            lines.append(f"Agent {name}:")
            lines.append(f"  Shapley: {shapley:.4f}, Weight: {d['weight']:.4f}")
            lines.append(f"  μ={d['mu']:.4f}, σ={d['sigma']:.4f}")
            lines.append(f"  λ⁺={d['lambda_pos']:.4f}, λ⁻={d['lambda_neg']:.4f}")
            lines.append(f"  Sortino={d['jd_sortino']:.3f}, Option={d['option_value']:.4f}, Info={d['info_premium']:.4f}")
        
        return "\n".join(lines)

    def get_diagnostic_report(self) -> Dict:
        """扩展诊断报告"""
        report = super().get_diagnostic_report()
        
        report['jd_pshaq'] = {
            'portfolio': self.portfolio.get_diagnostic(),
            'weights': self.portfolio.weights.copy(),
            'shapley_values': self.cached_shapley_values.copy()
        }
        
        return report
