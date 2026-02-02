"""
JD-PSHAQ: Jump Diffusion Portfolio-SHAQ

基于 Merton 跳跃扩散模型的多智能体信度分配算法

核心理论：
==========
使用随机微分方程 (SDE) 建模 Agent 的价值演化：

    dS_t = μS_t dt + σS_t dW_t + S_{t-} dJ_t
           ─────────  ──────────  ────────────
           漂移项      扩散项       跳跃项

其中：
- μ: 漂移率（Agent 的基础贡献率，稳定增长）
- σ: 扩散波动率（连续的小幅探索波动）
- W_t: 标准布朗运动
- J_t: 复合泊松过程（稀疏的大幅跳跃，如发现 Bug）

复合泊松过程：
    J_t = Σ_{k=1}^{N_t} Y_k
    
    - N_t ~ Poisson(λt): 跳跃次数
    - Y_k ~ LogNormal(μ_J, σ_J): 跳跃大小

为什么使用跳跃扩散？
==================
1. Web 测试奖励的特性：大部分时间平稳，偶尔大幅跳跃（发现 Bug）
2. 标准 GBM 无法捕获这种"尖峰厚尾"分布
3. 跳跃扩散能区分"日常探索"和"重大发现"

估值公式：
=========
Valuation = Shapley × Adjustment × Time_Factor

Adjustment = 1 + α×(JD_Sortino - 1) + β×Option_Value + γ×Info_Premium

仓位分配（核心修正）：
===================
仓位 w_i = 对 R_total 的分配比例（而非调整探索率！）
R_i = w_i × R_total
约束: Σ w_i = 1

参考文献：
=========
- Merton, R.C. (1976). "Option Pricing When Underlying Stock Returns Are Discontinuous"
- Kou, S.G. (2002). "A Jump-Diffusion Model for Option Pricing"
- Wang et al. "SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning"
"""

import math
import random
import threading
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Set
from scipy import stats
from scipy.special import factorial

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

# 复用 SHAQv2 的组件
from multi_agent.impl.shaq_v2 import (
    DOMStructureEncoder,
    IntrinsicCuriosityModule,
    ShapleyMixingNetwork,
    create_bug_analysis_system,
)


# ============================================================================
# Component 1: Jump Diffusion Process (跳跃扩散过程)
# ============================================================================

@dataclass
class JumpDiffusionParams:
    """跳跃扩散模型参数"""
    mu: float = 0.0          # 漂移率 (drift)
    sigma: float = 0.1       # 扩散波动率 (diffusion volatility)
    lambda_: float = 0.1     # 跳跃强度 (jump intensity, 单位时间平均跳跃次数)
    mu_j: float = 0.5        # 跳跃均值 (jump mean, log-normal)
    sigma_j: float = 0.3     # 跳跃波动率 (jump volatility)
    
    def total_variance(self) -> float:
        """总方差 = 扩散方差 + 跳跃方差"""
        # E[Y^2] for LogNormal = exp(2μ_J + 2σ_J^2)
        jump_second_moment = math.exp(2 * self.mu_j + 2 * self.sigma_j ** 2)
        jump_variance = self.lambda_ * jump_second_moment
        return self.sigma ** 2 + jump_variance


class JumpDiffusionTracker:
    """
    跳跃扩散过程追踪器
    
    追踪每个 Agent 的奖励路径，估计 JD 参数
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self.returns: deque = deque(maxlen=window_size)  # 对数收益率
        
        # 跳跃检测
        self.detected_jumps: List[Tuple[int, float]] = []  # (step, size)
        self.jump_threshold = 2.0  # 超过 2σ 视为跳跃
        
        # 累计统计
        self.total_steps = 0
        self.total_reward = 0.0
        self.new_state_count = 0
        
        # 参数估计
        self.estimated_params = JumpDiffusionParams()
        self._last_estimation_step = 0
        self._estimation_interval = 20
        
    def update(self, reward: float, is_new_state: bool = False, timestamp: float = None):
        """更新追踪器"""
        if timestamp is None:
            timestamp = self.total_steps
            
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
        self.total_steps += 1
        self.total_reward += reward
        
        if is_new_state:
            self.new_state_count += 1
        
        # 计算对数收益率
        if len(self.rewards) >= 2:
            prev = self.rewards[-2]
            curr = self.rewards[-1]
            if prev > 0 and curr > 0:
                log_return = math.log(curr / prev)
            else:
                # 处理非正值
                log_return = curr - prev
            self.returns.append(log_return)
            
            # 跳跃检测
            self._detect_jump(log_return)
        
        # 定期更新参数估计
        if self.total_steps - self._last_estimation_step >= self._estimation_interval:
            self._estimate_parameters()
            self._last_estimation_step = self.total_steps
    
    def _detect_jump(self, log_return: float):
        """
        跳跃检测：超过阈值的收益率视为跳跃
        
        使用 Lee-Mykland 跳跃检测思想的简化版
        """
        if len(self.returns) < 10:
            return
            
        returns_array = np.array(list(self.returns)[:-1])  # 不含当前
        mean_r = np.mean(returns_array)
        std_r = np.std(returns_array) + 1e-8
        
        z_score = abs(log_return - mean_r) / std_r
        
        if z_score > self.jump_threshold:
            self.detected_jumps.append((self.total_steps, log_return))
            # 只保留最近的跳跃
            if len(self.detected_jumps) > 50:
                self.detected_jumps = self.detected_jumps[-50:]
    
    def _estimate_parameters(self):
        """
        参数估计：矩估计 + MLE 混合
        
        基于 Ramezani & Zeng (2007) 的方法简化
        """
        if len(self.returns) < 20:
            return
            
        returns_array = np.array(list(self.returns))
        n = len(returns_array)
        
        # 1. 分离跳跃和扩散
        jump_indices = set()
        for step, _ in self.detected_jumps:
            idx = step - (self.total_steps - len(self.returns))
            if 0 <= idx < len(returns_array):
                jump_indices.add(idx)
        
        # 扩散部分（非跳跃）
        diffusion_returns = [r for i, r in enumerate(returns_array) if i not in jump_indices]
        
        # 跳跃部分
        jump_returns = [r for i, r in enumerate(returns_array) if i in jump_indices]
        
        # 2. 估计扩散参数
        if len(diffusion_returns) > 5:
            self.estimated_params.mu = np.mean(diffusion_returns)
            self.estimated_params.sigma = np.std(diffusion_returns) + 0.01
        
        # 3. 估计跳跃参数
        if len(jump_returns) > 2:
            # 跳跃强度 λ
            self.estimated_params.lambda_ = len(jump_returns) / n
            
            # 跳跃大小参数（假设对数正态）
            jump_sizes = np.abs(jump_returns)
            if np.all(jump_sizes > 0):
                log_sizes = np.log(jump_sizes)
                self.estimated_params.mu_j = np.mean(log_sizes)
                self.estimated_params.sigma_j = np.std(log_sizes) + 0.01
        else:
            # 使用先验
            self.estimated_params.lambda_ = 0.05
    
    @property
    def mean_return(self) -> float:
        """平均收益"""
        if len(self.rewards) == 0:
            return 0.0
        return float(np.mean(list(self.rewards)))
    
    @property
    def volatility(self) -> float:
        """总波动率"""
        if len(self.returns) < 2:
            return 0.1
        return float(np.std(list(self.returns))) + 0.01
    
    @property
    def jump_intensity(self) -> float:
        """跳跃强度"""
        return self.estimated_params.lambda_
    
    @property
    def new_state_rate(self) -> float:
        """新状态发现率"""
        if self.total_steps == 0:
            return 0.0
        return self.new_state_count / self.total_steps


# ============================================================================
# Component 2: JD-Sortino Ratio (跳跃扩散 Sortino 比率)
# ============================================================================

class JDSortinoCalculator:
    """
    跳跃扩散下的 Sortino 比率
    
    传统 Sortino = (R - MAR) / Downside_Std
    
    JD-Sortino 改进：
    1. 分别考虑扩散下行风险和跳跃下行风险
    2. 不惩罚正向跳跃（发现 Bug 是好事！）
    3. 对负向跳跃给予更高权重（灾难性失败）
    """
    
    def __init__(self, mar: float = 0.0, jump_penalty_weight: float = 2.0):
        """
        Args:
            mar: Minimum Acceptable Return (最低可接受回报)
            jump_penalty_weight: 负向跳跃的惩罚权重
        """
        self.mar = mar
        self.jump_penalty_weight = jump_penalty_weight
    
    def compute(self, tracker: JumpDiffusionTracker) -> float:
        """
        计算 JD-Sortino 比率
        
        公式：
            JD_Sortino = (Mean_Return - MAR) / JD_Downside_Risk
            
        其中：
            JD_Downside_Risk = sqrt(Diffusion_Downside² + Jump_Downside²)
        """
        if len(tracker.returns) < 10:
            return 1.0  # 初始值
        
        returns_array = np.array(list(tracker.returns))
        mean_return = np.mean(returns_array)
        
        # 1. 识别下行收益（低于 MAR）
        downside_returns = returns_array[returns_array < self.mar]
        
        if len(downside_returns) < 3:
            # 没有下行风险，给高分
            return 2.0
        
        # 2. 分离扩散下行和跳跃下行
        params = tracker.estimated_params
        jump_threshold = params.sigma * tracker.jump_threshold
        
        diffusion_downside = downside_returns[np.abs(downside_returns) <= jump_threshold]
        jump_downside = downside_returns[np.abs(downside_returns) > jump_threshold]
        
        # 3. 计算下行风险
        diffusion_downside_risk = np.std(diffusion_downside) if len(diffusion_downside) > 1 else 0.01
        
        # 跳跃下行风险（加权）
        if len(jump_downside) > 0:
            jump_downside_risk = np.std(jump_downside) * self.jump_penalty_weight
        else:
            jump_downside_risk = 0.0
        
        # 4. 综合下行风险
        total_downside_risk = math.sqrt(
            diffusion_downside_risk ** 2 + jump_downside_risk ** 2
        ) + 0.01
        
        # 5. 计算 JD-Sortino
        jd_sortino = (mean_return - self.mar) / total_downside_risk
        
        # 归一化到合理范围
        return max(0.0, min(3.0, jd_sortino + 1.0))


# ============================================================================
# Component 3: Merton Option Valuation (Merton 期权估值)
# ============================================================================

class MertonOptionValuator:
    """
    Merton 跳跃扩散期权定价
    
    用于评估 Agent 的"上行潜力"（类似看涨期权的时间价值）
    
    Merton 公式（简化版）：
        C = Σ_{n=0}^{∞} [exp(-λ'τ) × (λ'τ)^n / n!] × BS(S, K, r_n, σ_n, τ)
        
    其中 BS 是 Black-Scholes 公式
    """
    
    def __init__(self, risk_free_rate: float = 0.0, max_terms: int = 10):
        self.r = risk_free_rate
        self.max_terms = max_terms
    
    def compute_option_value(
        self, 
        tracker: JumpDiffusionTracker,
        remaining_time_ratio: float
    ) -> float:
        """
        计算期权价值（简化版 Merton 公式）
        
        Args:
            tracker: JD 追踪器
            remaining_time_ratio: 剩余时间比例 (0-1)
            
        Returns:
            期权价值（归一化到 0-2）
        """
        if remaining_time_ratio <= 0:
            return 0.0
            
        params = tracker.estimated_params
        
        # 当前"价格"
        S = max(tracker.mean_return + 1.0, 0.1)  # 确保正数
        
        # 行权价（设为当前价格，ATM 期权）
        K = S
        
        # 时间（年化，假设总时长为 1）
        tau = remaining_time_ratio
        
        # 跳跃调整后的参数
        lambda_prime = params.lambda_ * (1 + params.mu_j)
        
        # 简化计算：只取前几项
        option_value = 0.0
        
        for n in range(min(self.max_terms, 5)):
            # 泊松权重
            poisson_weight = math.exp(-lambda_prime * tau) * (lambda_prime * tau) ** n / math.factorial(n)
            
            # 调整后的波动率
            sigma_n = math.sqrt(params.sigma ** 2 + n * params.sigma_j ** 2 / tau) if tau > 0 else params.sigma
            
            # 简化的 BS 近似（ATM 期权）
            # ATM 期权价值 ≈ 0.4 × S × σ × √τ
            bs_value = 0.4 * S * sigma_n * math.sqrt(tau)
            
            option_value += poisson_weight * bs_value
        
        # 归一化
        normalized_value = option_value / (S + 0.01)
        return max(0.0, min(2.0, normalized_value))
    
    def compute_greeks(self, tracker: JumpDiffusionTracker, remaining_time_ratio: float) -> Dict[str, float]:
        """
        计算 Greeks（风险敏感度）
        
        简化版：
        - Delta: 对标的价格的敏感度
        - Theta: 时间衰减
        - Vega: 对波动率的敏感度
        """
        params = tracker.estimated_params
        tau = remaining_time_ratio
        sigma = params.sigma
        
        # 简化 Greeks
        delta = 0.5  # ATM 期权的 Delta 约为 0.5
        theta = -0.5 * sigma / math.sqrt(tau + 0.01)  # 时间衰减
        vega = math.sqrt(tau) * 0.4  # Vega
        
        return {
            'delta': delta,
            'theta': theta,
            'vega': vega
        }


# ============================================================================
# Component 4: Portfolio Weight Calculator (仓位计算器)
# ============================================================================

class PortfolioWeightCalculator:
    """
    仓位计算器：基于 JD 模型计算 R_total 的分配比例
    
    核心修正：仓位 w_i 是 R_total 的分配比例
    
    公式：
        w_i^{base} = φ_i (Shapley Value)
        w_i^{adj} = w_i^{base} × (1 + α×(JD_Sortino-1) + β×Option + γ×Info)
        w_i = w_i^{adj} / Σ w_j^{adj}  (归一化)
        
    约束：Σ w_i = 1
    """
    
    def __init__(
        self,
        n_agents: int,
        alpha: float = 0.3,   # JD-Sortino 权重
        beta: float = 0.2,    # 期权价值权重
        gamma: float = 0.2    # 信息溢价权重
    ):
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 追踪器
        self.trackers: Dict[str, JumpDiffusionTracker] = {
            str(i): JumpDiffusionTracker() for i in range(n_agents)
        }
        
        # 计算器
        self.sortino_calculator = JDSortinoCalculator()
        self.option_valuator = MertonOptionValuator()
        
        # 全局统计
        self.total_new_states = 0
        self.agent_new_states: Dict[str, int] = {str(i): 0 for i in range(n_agents)}
        
    def update(self, agent_name: str, reward: float, is_new_state: bool = False):
        """更新 Agent 数据"""
        if agent_name in self.trackers:
            self.trackers[agent_name].update(reward, is_new_state)
            if is_new_state:
                self.total_new_states += 1
                self.agent_new_states[agent_name] = self.agent_new_states.get(agent_name, 0) + 1
    
    def compute_info_premium(self, agent_name: str) -> float:
        """
        信息溢价
        
        注意：这里的"信息"是共享的首发优势，而非私有信息
        """
        if self.total_new_states == 0:
            return 0.0
        
        agent_discoveries = self.agent_new_states.get(agent_name, 0)
        # 首发优势 + 衰减（后续发现价值递减）
        info_share = agent_discoveries / max(self.total_new_states, 1)
        return info_share * 1.5
    
    def compute_weights(
        self,
        shapley_values: Dict[str, float],
        remaining_time_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        计算仓位权重
        
        这是核心方法：计算每个 Agent 应该分得的 R_total 比例
        
        Returns:
            Dict[str, float]: 归一化的权重，满足 Σ w_i = 1
        """
        raw_weights = {}
        
        for agent_name in shapley_values:
            tracker = self.trackers.get(agent_name)
            if not tracker:
                raw_weights[agent_name] = shapley_values[agent_name]
                continue
            
            # 1. 基础仓位 = Shapley Value
            base_weight = shapley_values[agent_name]
            
            # 2. JD-Sortino 调整
            jd_sortino = self.sortino_calculator.compute(tracker)
            sortino_adj = self.alpha * (jd_sortino - 1.0)
            
            # 3. 期权价值（上行潜力）
            option_value = self.option_valuator.compute_option_value(tracker, remaining_time_ratio)
            option_adj = self.beta * option_value
            
            # 4. 信息溢价
            info_premium = self.compute_info_premium(agent_name)
            info_adj = self.gamma * info_premium
            
            # 5. 综合调整
            adjustment = 1.0 + sortino_adj + option_adj + info_adj
            adjustment = max(0.1, adjustment)  # 防止负数
            
            raw_weights[agent_name] = base_weight * adjustment
        
        # 6. 归一化（确保 Σ w_i = 1）
        total = sum(raw_weights.values())
        if total > 0:
            return {k: v / total for k, v in raw_weights.items()}
        else:
            # 平均分配
            return {k: 1.0 / len(raw_weights) for k in raw_weights}
    
    def distribute_reward(
        self,
        r_total: float,
        shapley_values: Dict[str, float],
        remaining_time_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        分配团队总奖励
        
        核心方法：R_i = w_i × R_total
        
        Args:
            r_total: 团队总奖励
            shapley_values: Shapley 值
            remaining_time_ratio: 剩余时间比例
            
        Returns:
            Dict[str, float]: 每个 Agent 的奖励
        """
        weights = self.compute_weights(shapley_values, remaining_time_ratio)
        return {agent: w * r_total for agent, w in weights.items()}
    
    def get_diagnostic(self) -> Dict:
        """诊断报告"""
        report = {
            'total_new_states': self.total_new_states,
            'agents': {}
        }
        
        for agent_name, tracker in self.trackers.items():
            jd_sortino = self.sortino_calculator.compute(tracker)
            report['agents'][agent_name] = {
                'mean_return': tracker.mean_return,
                'volatility': tracker.volatility,
                'jump_intensity': tracker.jump_intensity,
                'detected_jumps': len(tracker.detected_jumps),
                'jd_sortino': jd_sortino,
                'new_state_rate': tracker.new_state_rate,
                'params': {
                    'mu': tracker.estimated_params.mu,
                    'sigma': tracker.estimated_params.sigma,
                    'lambda': tracker.estimated_params.lambda_,
                    'mu_j': tracker.estimated_params.mu_j,
                    'sigma_j': tracker.estimated_params.sigma_j,
                }
            }
        
        return report


# ============================================================================
# Component 5: Team Reward Aggregator (团队奖励聚合器)
# ============================================================================

class TeamRewardAggregator:
    """
    团队奖励聚合器
    
    负责：
    1. 收集所有 Agent 的基础奖励
    2. 计算 R_total
    3. 调用仓位计算器分配奖励
    """
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.current_step_rewards: Dict[str, float] = {}
        self.step_counter = 0
        
    def record_base_reward(self, agent_name: str, reward: float):
        """记录 Agent 的基础奖励"""
        self.current_step_rewards[agent_name] = reward
    
    def compute_r_total(self) -> float:
        """计算团队总奖励"""
        return sum(self.current_step_rewards.values())
    
    def finalize_step(
        self,
        weight_calculator: PortfolioWeightCalculator,
        shapley_values: Dict[str, float],
        remaining_time_ratio: float
    ) -> Dict[str, float]:
        """
        完成一步，分配奖励
        
        Returns:
            每个 Agent 的最终奖励
        """
        r_total = self.compute_r_total()
        
        # 分配奖励
        distributed = weight_calculator.distribute_reward(
            r_total, shapley_values, remaining_time_ratio
        )
        
        # 更新追踪器
        for agent_name, reward in distributed.items():
            is_new = self.current_step_rewards.get(agent_name, 0) > 0
            weight_calculator.update(agent_name, reward, is_new)
        
        # 清空当前步骤
        self.current_step_rewards = {}
        self.step_counter += 1
        
        return distributed


# ============================================================================
# Main Class: JD-PSHAQ
# ============================================================================

class JDPSHAQ(multi_agent.multi_agent_system.MultiAgentSystem):
    """
    JD-PSHAQ: Jump Diffusion Portfolio-SHAQ
    
    核心创新：
    1. 使用 Merton 跳跃扩散 SDE 建模 Agent 价值演化
    2. JD-Sortino Ratio 区分扩散风险和跳跃风险
    3. Merton 期权定价评估上行潜力
    4. 仓位作为 R_total 的分配比例（而非探索率调节）
    
    与 P-SHAQ 的关键区别：
    - P-SHAQ: R_i = R_base × V_i（估值放大）
    - JD-PSHAQ: R_i = w_i × R_total（仓位分配）
    """
    
    def __init__(self, params: Dict):
        super().__init__(params)
        self.params = params
        self.algo_type = params.get("algo_type", "jd_pshaq")
        self.reward_function = params.get("reward_function", "A")
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"JD-PSHAQ: Using {self.device}")
        
        # 超参数
        self.max_random = params.get("max_random", 0.9)
        self.min_random = params.get("min_random", 0.1)
        self.batch_size = params.get("batch_size", 32)
        self.gamma = params.get("gamma", 0.5)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.update_target_interval = params.get("update_target_interval", 20)
        self.update_network_interval = params.get("update_network_interval", 4)
        self.shapley_update_interval = params.get("shapley_update_interval", 10)
        self.alive_time = params.get("alive_time", 10800)
        
        # JD-PSHAQ 特有参数
        self.alpha = params.get("alpha", 0.3)   # JD-Sortino 权重
        self.beta = params.get("beta", 0.2)     # 期权价值权重
        self.gamma_info = params.get("gamma_info", 0.2)  # 信息溢价权重
        
        # Transformer
        self.transformer = instantiate_class_by_module_and_class_name(
            params["transformer_module"], params["transformer_class"]
        )
        
        # 记录
        self.state_list: List[WebState] = []
        self.action_list: List[WebAction] = []
        self.state_list_agent: Dict[str, List[WebState]] = {}
        self.action_count = defaultdict(int)
        self.learn_step_count = 0
        self.start_time = datetime.now()
        
        # 网络锁
        self.network_lock = threading.Lock()
        
        # Q 网络
        self.q_eval_agent: Dict[str, nn.Module] = {}
        self.q_target_agent: Dict[str, nn.Module] = {}
        self.agent_optimizer: Dict[str, optim.Optimizer] = {}
        
        for i in range(self.agent_num):
            agent_name = str(i)
            q_eval = instantiate_class_by_module_and_class_name(
                params["model_module"], params["model_class"]
            )
            q_target = instantiate_class_by_module_and_class_name(
                params["model_module"], params["model_class"]
            )
            q_eval.to(self.device)
            q_target.to(self.device)
            q_target.load_state_dict(q_eval.state_dict())
            
            self.q_eval_agent[agent_name] = q_eval
            self.q_target_agent[agent_name] = q_target
            self.agent_optimizer[agent_name] = optim.Adam(q_eval.parameters(), lr=self.learning_rate)
            self.state_list_agent[agent_name] = []
        
        # 【JD-PSHAQ 核心】仓位计算器
        self.weight_calculator = PortfolioWeightCalculator(
            n_agents=self.agent_num,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma_info
        )
        
        # 团队奖励聚合器
        self.team_aggregator = TeamRewardAggregator(self.agent_num)
        
        # Shapley 混合网络
        self.mixing_network = ShapleyMixingNetwork(n_agents=self.agent_num, embed_dim=64).to(self.device)
        self.target_mixing_network = ShapleyMixingNetwork(n_agents=self.agent_num, embed_dim=64).to(self.device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # DOM 编码器
        self.dom_encoder = DOMStructureEncoder()
        
        # 经验回放
        self.replay_buffer_agent: Dict[str, ReplayBuffer] = {
            str(i): ReplayBuffer(capacity=500) for i in range(self.agent_num)
        }
        
        # Shapley 缓存
        self.cached_shapley_values: Dict[str, float] = {
            str(i): 1.0 / self.agent_num for i in range(self.agent_num)
        }
        self.shapley_update_counter = 0
        
        # 追踪
        self.prev_state_dict: Dict[str, Optional[WebState]] = {str(i): None for i in range(self.agent_num)}
        self.prev_action_dict: Dict[str, Optional[WebAction]] = {str(i): None for i in range(self.agent_num)}
        self.prev_html_dict: Dict[str, str] = {str(i): "" for i in range(self.agent_num)}
        self.action_dict: Dict[WebAction, int] = {}
        
        # Bug 分析
        self.bug_analyzer, self.bug_localizer = create_bug_analysis_system()
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 已见状态签名
        self._seen_dom_signatures: Set[int] = set()
        
        logger.info(f"JD-PSHAQ initialized with {self.agent_num} agents, "
                   f"α={self.alpha}, β={self.beta}, γ={self.gamma_info}")
    
    def _get_remaining_time_ratio(self) -> float:
        """剩余时间比例"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return max(0.0, 1.0 - elapsed / self.alive_time)
    
    def get_tensor(self, action: WebAction, html: str, web_state: WebState) -> torch.Tensor:
        """状态-动作编码"""
        state_tensor = self.transformer.state_to_tensor(web_state, html)
        execution_time = self.action_dict.get(action, 0)
        action_tensor = self.transformer.action_to_tensor(web_state, action, execution_time)
        tensor = torch.cat((state_tensor, action_tensor))
        return tensor.float()
    
    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        """动作选择"""
        self.update_state_records(web_state, html, agent_name)
        
        actions = web_state.get_action_list()
        if len(actions) == 1 and isinstance(actions[0], RestartAction):
            return actions[0]
        
        q_eval = self.q_eval_agent[agent_name]
        q_eval.eval()
        
        action_tensors = []
        for temp_action in actions:
            action_tensor = self.get_tensor(temp_action, html, web_state)
            action_tensors.append(action_tensor)
        
        from model.dense_net import DenseNet
        with torch.no_grad():
            if isinstance(q_eval, DenseNet):
                output = q_eval(torch.stack(action_tensors).unsqueeze(1).to(self.device))
            else:
                output = q_eval(torch.stack(action_tensors).to(self.device))
        
        q_values = output.squeeze(-1).cpu().numpy()
        max_idx = q_values.argmax()
        max_val = q_values[max_idx]
        chosen_action = actions[max_idx]
        
        logger.info(f"[{agent_name}] JD-PSHAQ max Q: {max_val:.4f}")
        
        # ε-greedy（标准方式，不用仓位调整探索率）
        time_diff = (datetime.now() - self.start_time).total_seconds()
        time_diff = min(time_diff, self.alive_time)
        epsilon = self.max_random - min(time_diff / self.alive_time * 2, 1.0) * (
            self.max_random - self.min_random
        )
        
        if random.uniform(0, 1) < epsilon:
            unexplored = [a for a in actions if self.action_dict.get(a, 0) == 0]
            if unexplored:
                chosen_action = random.choice(unexplored)
            else:
                chosen_action = random.choice(actions)
        
        self.action_count[chosen_action] += 1
        self.action_dict[chosen_action] = self.action_dict.get(chosen_action, 0) + 1
        
        return chosen_action
    
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
        
        # 计算基础奖励
        base_reward = self._compute_base_reward(web_state, html, agent_name)
        
        # 记录到团队聚合器
        self.team_aggregator.record_base_reward(agent_name, base_reward)
        
        # 检查是否是新状态
        state_hash = self.dom_encoder.compute_structure_hash(html) if html else ""
        is_new_state = state_hash and hash(state_hash) not in self._seen_dom_signatures
        if is_new_state and state_hash:
            self._seen_dom_signatures.add(hash(state_hash))
        
        # 更新 JD 追踪器
        self.weight_calculator.update(agent_name, base_reward, is_new_state)
        
        # 学习
        self.learn_agent(agent_name, base_reward)
    
    def _compute_base_reward(self, web_state: WebState, html: str, agent_name: str) -> float:
        """计算基础奖励（不含仓位调整）"""
        # 简化的奖励计算
        reward = 0.0
        
        # 新状态奖励
        if web_state not in self.state_list[:-1]:
            reward += 10.0
        
        # URL 变化奖励
        if hasattr(web_state, 'url') and hasattr(self.prev_state_dict.get(agent_name), 'url'):
            if web_state.url != self.prev_state_dict[agent_name].url:
                reward += 5.0
        
        # 动作多样性
        if isinstance(web_state, ActionSetWithExecutionTimesState):
            reward += min(web_state.action_number * 0.1, 5.0)
        
        return reward
    
    def learn_agent(self, agent_name: str, reward: float):
        """Agent 学习"""
        self.learn_step_count += 1
        
        # 存储经验
        tensor = self.get_tensor(
            self.prev_action_dict[agent_name],
            self.prev_html_dict[agent_name],
            self.prev_state_dict[agent_name]
        )
        tensor = tensor.unsqueeze(0)
        
        self.replay_buffer_agent[agent_name].push(
            tensor, tensor, reward, None, None, False
        )
        
        # 更新 Shapley
        self.shapley_update_counter += 1
        if self.shapley_update_counter >= self.shapley_update_interval:
            self._update_shapley_values()
            self.shapley_update_counter = 0
        
        # Q-learning 更新
        if self.learn_step_count % self.update_network_interval == 0:
            self._learn(agent_name)
        
        # 目标网络更新
        if self.learn_step_count % self.update_target_interval == 0:
            self._update_target_networks()
    
    def _update_shapley_values(self):
        """更新 Shapley 值"""
        contributions = {}
        for agent_name in self.cached_shapley_values:
            tracker = self.weight_calculator.trackers.get(agent_name)
            if tracker:
                contrib = tracker.mean_return + 10 * tracker.new_state_rate
                contributions[agent_name] = max(0.01, contrib)
            else:
                contributions[agent_name] = 0.01
        
        total = sum(contributions.values())
        if total > 0:
            for agent_name in self.cached_shapley_values:
                self.cached_shapley_values[agent_name] = contributions[agent_name] / total
    
    def _learn(self, agent_name: str):
        """Q-learning 更新"""
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
            optimizer.step()
            
        except Exception as e:
            logger.debug(f"Learning error: {e}")
    
    def _update_target_networks(self):
        """更新目标网络"""
        for agent_name in self.q_eval_agent:
            self.q_target_agent[agent_name].load_state_dict(
                self.q_eval_agent[agent_name].state_dict()
            )
    
    def set_prev(self, agent_name: str, state: WebState, action: WebAction, html: str):
        """设置前一状态"""
        self.prev_state_dict[agent_name] = state
        self.prev_action_dict[agent_name] = action
        self.prev_html_dict[agent_name] = html
    
    def get_reward(self, web_state: WebState, html: str, agent_name: str) -> float:
        """获取奖励（使用仓位分配）"""
        base_reward = self._compute_base_reward(web_state, html, agent_name)
        
        # 【JD-PSHAQ 核心】使用仓位分配而非估值放大
        weights = self.weight_calculator.compute_weights(
            self.cached_shapley_values,
            self._get_remaining_time_ratio()
        )
        
        # 简化：单个 Agent 的奖励 = 基础奖励 × 相对权重
        # 完整版应该在 team_aggregator 中统一分配 R_total
        weight = weights.get(agent_name, 1.0 / self.agent_num)
        adjusted_reward = base_reward * weight * self.agent_num
        
        return adjusted_reward
    
    def get_diagnostic_report(self) -> str:
        """诊断报告"""
        report_lines = [
            "=" * 60,
            "JD-PSHAQ Diagnostic Report",
            "=" * 60,
            f"Algorithm: Jump Diffusion Portfolio-SHAQ",
            f"Agents: {self.agent_num}",
            f"Learn Steps: {self.learn_step_count}",
            f"Remaining Time: {self._get_remaining_time_ratio():.1%}",
            "",
            "--- JD Parameters ---",
        ]
        
        diag = self.weight_calculator.get_diagnostic()
        report_lines.append(f"Total New States: {diag['total_new_states']}")
        
        for agent_name, metrics in diag['agents'].items():
            report_lines.append(f"\nAgent {agent_name}:")
            report_lines.append(f"  Mean Return: {metrics['mean_return']:.2f}")
            report_lines.append(f"  Volatility: {metrics['volatility']:.3f}")
            report_lines.append(f"  Jump Intensity: {metrics['jump_intensity']:.3f}")
            report_lines.append(f"  Detected Jumps: {metrics['detected_jumps']}")
            report_lines.append(f"  JD-Sortino: {metrics['jd_sortino']:.2f}")
            report_lines.append(f"  New State Rate: {metrics['new_state_rate']:.2%}")
            report_lines.append(f"  JD Params: μ={metrics['params']['mu']:.3f}, "
                              f"σ={metrics['params']['sigma']:.3f}, "
                              f"λ={metrics['params']['lambda']:.3f}")
        
        # 仓位权重
        weights = self.weight_calculator.compute_weights(
            self.cached_shapley_values,
            self._get_remaining_time_ratio()
        )
        report_lines.append("\n--- Portfolio Weights (R_total Distribution) ---")
        for agent_name, weight in sorted(weights.items()):
            report_lines.append(f"  Agent {agent_name}: {weight:.4f} ({weight*100:.1f}%)")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def finish_episode(self, agent_name: str):
        """结束 episode"""
        if all(self.prev_state_dict.values()):
            logger.info(self.get_diagnostic_report())
