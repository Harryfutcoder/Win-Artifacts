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
# Component 1: Jump Diffusion Process (跳跃扩散过程) - CORRECTED VERSION
# ============================================================================
#
# 数学修正说明：
# ==============
# 原版问题：直接对瞬时奖励 R_t 计算对数收益率，导致：
#   1. 量纲混淆：log(R_t/R_{t-1}) 与 R_t - R_{t-1} 不可比
#   2. 语义悖论：稳定高效 Agent [10,10,10] 被判定为"无增长潜力"
#   3. 边界问题：R_t = 0 时无法计算对数
#
# 修正方案：
#   S_t = 累积价值 (EMA 平滑的累积奖励)
#   r_t = (S_t - S_{t-1}) / |S_{t-1}| + ε  (算术收益率，统一量纲)
#
# ============================================================================

@dataclass
class JumpDiffusionParams:
    """跳跃扩散模型参数"""
    mu: float = 0.0          # 漂移率 (drift)
    sigma: float = 0.1       # 扩散波动率 (diffusion volatility)
    lambda_: float = 0.1     # 跳跃强度 (jump intensity, 单位时间平均跳跃次数)
    lambda_pos: float = 0.05 # 正向跳跃强度 (发现 Bug)
    lambda_neg: float = 0.05 # 负向跳跃强度 (卡住/失败)
    mu_j: float = 0.5        # 跳跃均值 (jump mean, log-normal)
    sigma_j: float = 0.3     # 跳跃波动率 (jump volatility)
    
    def total_variance(self) -> float:
        """总方差 = 扩散方差 + 跳跃方差"""
        # E[Y^2] for LogNormal = exp(2μ_J + 2σ_J^2)
        jump_second_moment = math.exp(2 * self.mu_j + 2 * self.sigma_j ** 2)
        jump_variance = self.lambda_ * jump_second_moment
        return self.sigma ** 2 + jump_variance
    
    def downside_variance(self) -> float:
        """下行方差 = 扩散方差 + 负向跳跃方差（只惩罚坏的波动）"""
        # 只考虑负向跳跃的方差
        neg_jump_variance = self.lambda_neg * math.exp(2 * self.mu_j + 2 * self.sigma_j ** 2)
        return self.sigma ** 2 + neg_jump_variance


class JumpDiffusionTracker:
    """
    跳跃扩散过程追踪器 (修正版)
    
    核心修正：
    1. S_t 使用 EMA 平滑的累积价值，而非瞬时奖励
    2. 统一使用算术收益率，避免量纲混淆
    3. 分离正向跳跃和负向跳跃
    """
    
    # 收益率计算的基准值（用于归一化）
    REWARD_SCALE = 10.0  # Web Testing 典型奖励量级
    EMA_ALPHA = 0.1      # EMA 平滑系数
    MIN_VALUE = 1.0      # 最小累积价值（避免除零）
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # 原始奖励序列
        self.raw_rewards: deque = deque(maxlen=window_size)
        
        # 【修正】累积价值序列 S_t = EMA(累积奖励)
        self.cumulative_values: deque = deque(maxlen=window_size)
        self._ema_value: float = self.MIN_VALUE  # EMA 平滑后的价值
        
        # 【修正】算术收益率序列 r_t = (S_t - S_{t-1}) / |S_{t-1}|
        self.returns: deque = deque(maxlen=window_size)
        
        self.timestamps: deque = deque(maxlen=window_size)
        
        # 【修正】分离正向和负向跳跃
        self.positive_jumps: List[Tuple[int, float]] = []  # 好的跳跃（发现Bug）
        self.negative_jumps: List[Tuple[int, float]] = []  # 坏的跳跃（失败/卡住）
        self.jump_threshold = 2.5  # z-score 阈值（约 99% 置信度）
        
        # 累计统计
        self.total_steps = 0
        self.total_reward = 0.0
        self.new_state_count = 0
        self.bug_count = 0  # 发现的 Bug 数量
        
        # 参数估计
        self.estimated_params = JumpDiffusionParams()
        self._last_estimation_step = 0
        self._estimation_interval = 20
        
    def update(self, reward: float, is_new_state: bool = False, 
               found_bug: bool = False, timestamp: float = None):
        """
        更新追踪器
        
        Args:
            reward: 当前步奖励
            is_new_state: 是否发现新状态
            found_bug: 是否发现 Bug（触发正向跳跃）
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = self.total_steps
            
        self.raw_rewards.append(reward)
        self.timestamps.append(timestamp)
        self.total_steps += 1
        self.total_reward += reward
        
        if is_new_state:
            self.new_state_count += 1
        if found_bug:
            self.bug_count += 1
        
        # 【修正】计算 EMA 平滑的累积价值
        # S_t = α * R_t + (1-α) * S_{t-1}
        # 这里 R_t 是归一化后的奖励
        normalized_reward = reward / self.REWARD_SCALE
        self._ema_value = (self.EMA_ALPHA * normalized_reward + 
                          (1 - self.EMA_ALPHA) * self._ema_value)
        # 确保最小值
        self._ema_value = max(self._ema_value, self.MIN_VALUE)
        self.cumulative_values.append(self._ema_value)
        
        # 【修正】计算算术收益率（统一量纲）
        if len(self.cumulative_values) >= 2:
            prev_val = self.cumulative_values[-2]
            curr_val = self.cumulative_values[-1]
            
            # 算术收益率: r = (S_t - S_{t-1}) / |S_{t-1}|
            # 这避免了对数收益率的边界问题
            arithmetic_return = (curr_val - prev_val) / (abs(prev_val) + 1e-8)
            self.returns.append(arithmetic_return)
            
            # 跳跃检测
            self._detect_jump(arithmetic_return, found_bug)
        
        # 定期更新参数估计
        if self.total_steps - self._last_estimation_step >= self._estimation_interval:
            self._estimate_parameters()
            self._last_estimation_step = self.total_steps
    
    def _detect_jump(self, return_value: float, found_bug: bool):
        """
        跳跃检测 (修正版)
        
        核心修正：区分正向跳跃和负向跳跃
        - 正向跳跃：发现 Bug 或新状态带来的大幅收益
        - 负向跳跃：失败或卡住导致的大幅损失
        
        在 JD-Sortino 中，只惩罚负向跳跃
        """
        if len(self.returns) < 10:
            return
            
        returns_array = np.array(list(self.returns)[:-1])
        mean_r = np.mean(returns_array)
        std_r = np.std(returns_array) + 1e-8
        
        z_score = (return_value - mean_r) / std_r
        
        if abs(z_score) > self.jump_threshold:
            if z_score > 0 or found_bug:
                # 正向跳跃（好事！）
                self.positive_jumps.append((self.total_steps, return_value))
                if len(self.positive_jumps) > 50:
                    self.positive_jumps = self.positive_jumps[-50:]
            else:
                # 负向跳跃（风险！）
                self.negative_jumps.append((self.total_steps, return_value))
                if len(self.negative_jumps) > 50:
                    self.negative_jumps = self.negative_jumps[-50:]
    
    def _estimate_parameters(self):
        """
        参数估计 (修正版)
        
        分别估计扩散参数和正/负向跳跃参数
        """
        if len(self.returns) < 20:
            return
            
        returns_array = np.array(list(self.returns))
        n = len(returns_array)
        
        # 识别跳跃索引
        pos_jump_indices = set()
        neg_jump_indices = set()
        
        for step, _ in self.positive_jumps:
            idx = step - (self.total_steps - len(self.returns))
            if 0 <= idx < len(returns_array):
                pos_jump_indices.add(idx)
        
        for step, _ in self.negative_jumps:
            idx = step - (self.total_steps - len(self.returns))
            if 0 <= idx < len(returns_array):
                neg_jump_indices.add(idx)
        
        all_jump_indices = pos_jump_indices | neg_jump_indices
        
        # 扩散部分（非跳跃）
        diffusion_returns = [r for i, r in enumerate(returns_array) 
                           if i not in all_jump_indices]
        
        # 正向跳跃部分
        pos_jump_returns = [r for i, r in enumerate(returns_array) 
                          if i in pos_jump_indices]
        
        # 负向跳跃部分
        neg_jump_returns = [r for i, r in enumerate(returns_array) 
                          if i in neg_jump_indices]
        
        # 估计扩散参数
        if len(diffusion_returns) > 5:
            self.estimated_params.mu = float(np.mean(diffusion_returns))
            self.estimated_params.sigma = float(np.std(diffusion_returns)) + 0.01
        
        # 估计跳跃参数
        self.estimated_params.lambda_pos = len(pos_jump_returns) / n if n > 0 else 0.05
        self.estimated_params.lambda_neg = len(neg_jump_returns) / n if n > 0 else 0.05
        self.estimated_params.lambda_ = self.estimated_params.lambda_pos + self.estimated_params.lambda_neg
        
        # 跳跃大小参数（使用所有跳跃）
        all_jumps = pos_jump_returns + neg_jump_returns
        if len(all_jumps) > 2:
            jump_sizes = np.abs(all_jumps)
            if np.all(jump_sizes > 0):
                log_sizes = np.log(jump_sizes + 1e-8)
                self.estimated_params.mu_j = float(np.mean(log_sizes))
                self.estimated_params.sigma_j = float(np.std(log_sizes)) + 0.01
        else:
            # 使用先验
            self.estimated_params.lambda_ = 0.05
    
    @property
    def mean_return(self) -> float:
        """平均收益（基于原始奖励）"""
        if len(self.raw_rewards) == 0:
            return 0.0
        return float(np.mean(list(self.raw_rewards)))
    
    @property
    def cumulative_value(self) -> float:
        """当前累积价值（EMA 平滑后）"""
        return self._ema_value
    
    @property
    def volatility(self) -> float:
        """总波动率"""
        if len(self.returns) < 2:
            return 0.1
        return float(np.std(list(self.returns))) + 0.01
    
    @property
    def jump_intensity(self) -> float:
        """跳跃强度（总）"""
        return self.estimated_params.lambda_
    
    @property
    def positive_jump_intensity(self) -> float:
        """正向跳跃强度"""
        return self.estimated_params.lambda_pos
    
    @property
    def negative_jump_intensity(self) -> float:
        """负向跳跃强度"""
        return self.estimated_params.lambda_neg
    
    @property
    def new_state_rate(self) -> float:
        """新状态发现率"""
        if self.total_steps == 0:
            return 0.0
        return self.new_state_count / self.total_steps
    
    @property
    def bug_rate(self) -> float:
        """Bug 发现率"""
        if self.total_steps == 0:
            return 0.0
        return self.bug_count / self.total_steps


# ============================================================================
# Component 2: JD-Sortino Ratio (跳跃扩散 Sortino 比率) - CORRECTED VERSION
# ============================================================================
#
# 数学修正说明：
# ==============
# 原版问题：
#   1. 将扩散下行和跳跃下行分开计算后再合成，数学意义不清
#   2. 没有正确区分"正向跳跃"（好事）和"负向跳跃"（坏事）
#
# 修正方案：
#   JD-Sortino = (E[R] - MAR) / sqrt(σ² + λ⁻ × E[(Y⁻)²])
#   
#   其中：
#   - σ: 扩散波动率
#   - λ⁻: 负向跳跃强度（只有负向跳跃才算风险！）
#   - Y⁻: 负向跳跃大小
#   - 正向跳跃不计入分母（不惩罚好的波动）
#
# ============================================================================

class JDSortinoCalculator:
    """
    跳跃扩散下的 Sortino 比率 (修正版)
    
    核心修正：
    1. 只惩罚负向跳跃，不惩罚正向跳跃（发现 Bug 是好事！）
    2. 使用理论上正确的下行风险公式
    3. 更好的边界情况处理
    """
    
    def __init__(self, mar: float = 0.0, neg_jump_penalty_weight: float = 1.5):
        """
        Args:
            mar: Minimum Acceptable Return (最低可接受回报)
            neg_jump_penalty_weight: 负向跳跃的惩罚权重（>1 表示更厌恶负向跳跃）
        """
        self.mar = mar
        self.neg_jump_penalty_weight = neg_jump_penalty_weight
    
    def compute(self, tracker: JumpDiffusionTracker) -> float:
        """
        计算 JD-Sortino 比率 (修正版)
        
        公式：
            JD_Sortino = (Mean_Return - MAR) / JD_Downside_Risk
            
        其中：
            JD_Downside_Risk = sqrt(σ² + λ⁻ × E[(Y⁻)²] × penalty_weight)
            
        关键修正：
        - 正向跳跃不计入下行风险
        - 使用累积价值的收益率而非原始奖励
        """
        if len(tracker.returns) < 10:
            return 1.0  # 初始值
        
        returns_array = np.array(list(tracker.returns))
        mean_return = float(np.mean(returns_array))
        
        params = tracker.estimated_params
        
        # 1. 扩散下行风险 = 扩散波动率²
        diffusion_downside_var = params.sigma ** 2
        
        # 2. 【修正】只计算负向跳跃的下行风险
        # E[(Y⁻)²] for negative jumps
        neg_jump_count = len(tracker.negative_jumps)
        if neg_jump_count > 0:
            # 从负向跳跃中提取大小
            neg_jump_sizes = [abs(j[1]) for j in tracker.negative_jumps]
            neg_jump_second_moment = np.mean([s**2 for s in neg_jump_sizes])
            
            # λ⁻ × E[(Y⁻)²]
            neg_jump_downside_var = (params.lambda_neg * 
                                    neg_jump_second_moment * 
                                    self.neg_jump_penalty_weight)
        else:
            neg_jump_downside_var = 0.0
        
        # 3. 综合下行风险 (不包含正向跳跃！)
        total_downside_var = diffusion_downside_var + neg_jump_downside_var
        total_downside_risk = math.sqrt(total_downside_var) + 0.01
        
        # 4. 计算 JD-Sortino
        jd_sortino = (mean_return - self.mar) / total_downside_risk
        
        # 5. 归一化到合理范围 [0, 3]
        # +1 是为了让基准值为 1.0
        normalized = max(0.0, min(3.0, jd_sortino + 1.0))
        
        return normalized
    
    def compute_detailed(self, tracker: JumpDiffusionTracker) -> Dict[str, float]:
        """
        计算详细的 JD-Sortino 分解（用于诊断）
        
        Returns:
            包含各组件的字典
        """
        returns_array = np.array(list(tracker.returns)) if len(tracker.returns) > 0 else np.array([0.0])
        params = tracker.estimated_params
        
        return {
            'mean_return': float(np.mean(returns_array)),
            'mar': self.mar,
            'diffusion_volatility': params.sigma,
            'positive_jump_intensity': params.lambda_pos,
            'negative_jump_intensity': params.lambda_neg,
            'positive_jumps_count': len(tracker.positive_jumps),
            'negative_jumps_count': len(tracker.negative_jumps),
            'jd_sortino': self.compute(tracker),
        }


# ============================================================================
# Component 3: Merton Option Valuation (Merton 期权估值) - CORRECTED VERSION
# ============================================================================
#
# 数学修正说明：
# ==============
# 原版问题：
#   1. S = mean_return + 1.0 不是累积量，而是瞬时量
#   2. ATM 期权近似 C ≈ 0.4Sσ√τ 在高波动率时高估价值
#
# 修正方案：
#   1. 使用累积价值（EMA 平滑后）作为 S
#   2. 只考虑正向跳跃的上行潜力
#   3. 简化为直接的"上行潜力"指标：
#      Upside_Potential = λ⁺ × E[Y⁺] × τ
#
# ============================================================================

class MertonOptionValuator:
    """
    Merton 期权估值器 (修正版)
    
    核心修正：
    1. 使用累积价值而非瞬时奖励作为 S
    2. 只考虑正向跳跃的上行潜力（发现 Bug）
    3. 提供简化的上行潜力指标用于实际计算
    """
    
    def __init__(self, risk_free_rate: float = 0.0, max_terms: int = 5):
        self.r = risk_free_rate
        self.max_terms = max_terms
    
    def compute_option_value(
        self, 
        tracker: JumpDiffusionTracker,
        remaining_time_ratio: float
    ) -> float:
        """
        计算期权价值 (修正版)
        
        对于 Web Testing，我们关心的是"上行潜力"，即：
        - 发现 Bug 的可能性 (正向跳跃强度 λ⁺)
        - 发现 Bug 的期望价值 (正向跳跃期望 E[Y⁺])
        - 剩余时间 (τ)
        
        简化公式：
            Upside_Potential = λ⁺ × E[Y⁺] × τ + 0.5 × σ × √τ
        
        Returns:
            期权价值（归一化到 [0, 2]）
        """
        if remaining_time_ratio <= 0:
            return 0.0
            
        params = tracker.estimated_params
        tau = remaining_time_ratio
        
        # 1. 正向跳跃的上行潜力
        # E[Y⁺] for positive jumps (简化：使用 exp(μ_j))
        lambda_pos = params.lambda_pos
        if len(tracker.positive_jumps) > 0:
            pos_jump_sizes = [abs(j[1]) for j in tracker.positive_jumps]
            expected_pos_jump = np.mean(pos_jump_sizes)
        else:
            # 使用先验
            expected_pos_jump = math.exp(params.mu_j)
        
        jump_upside = lambda_pos * expected_pos_jump * tau
        
        # 2. 扩散的上行潜力（标准 BS 近似）
        # ATM 期权价值 ≈ 0.4 × σ × √τ（归一化后）
        sigma = params.sigma
        diffusion_upside = 0.4 * sigma * math.sqrt(tau)
        
        # 3. 综合上行潜力
        total_upside = jump_upside + diffusion_upside
        
        # 4. 归一化到 [0, 2]
        # 使用 tanh 进行平滑归一化
        normalized = 2.0 * math.tanh(total_upside / 2.0)
        
        return max(0.0, min(2.0, normalized))
    
    def compute_full_merton(
        self, 
        tracker: JumpDiffusionTracker,
        remaining_time_ratio: float
    ) -> float:
        """
        完整的 Merton 期权定价（用于对比研究）
        
        公式：
            C = Σ_{n=0}^{N} [exp(-λ'τ)(λ'τ)^n / n!] × BS(S, K, r_n, σ_n, τ)
        """
        if remaining_time_ratio <= 0:
            return 0.0
            
        params = tracker.estimated_params
        tau = remaining_time_ratio
        
        # 使用累积价值作为 S
        S = max(tracker.cumulative_value, 0.1)
        K = S  # ATM 期权
        
        # 跳跃调整后的参数
        k = math.exp(params.mu_j + 0.5 * params.sigma_j ** 2) - 1
        lambda_prime = params.lambda_ * (1 + k)
        
        option_value = 0.0
        
        for n in range(self.max_terms):
            # 泊松权重
            try:
                poisson_weight = (math.exp(-lambda_prime * tau) * 
                                 (lambda_prime * tau) ** n / 
                                 math.factorial(n))
            except (OverflowError, ValueError):
                poisson_weight = 0.0
            
            # 调整后的波动率
            if tau > 0.01:
                sigma_n = math.sqrt(params.sigma ** 2 + n * params.sigma_j ** 2 / tau)
            else:
                sigma_n = params.sigma
            sigma_n = min(sigma_n, 10.0)  # 防止过大
            
            # BS 近似
            bs_value = 0.4 * S * sigma_n * math.sqrt(tau)
            option_value += poisson_weight * bs_value
        
        # 归一化
        normalized_value = option_value / (S + 0.01)
        return max(0.0, min(2.0, normalized_value))
    
    def compute_greeks(self, tracker: JumpDiffusionTracker, 
                      remaining_time_ratio: float) -> Dict[str, float]:
        """
        计算 Greeks（风险敏感度）
        
        对于 Web Testing：
        - Delta: 对累积价值的敏感度
        - Theta: 时间衰减（剩余时间减少时期权价值下降）
        - Vega: 对波动率的敏感度
        - Lambda: 对跳跃强度的敏感度（新增）
        """
        params = tracker.estimated_params
        tau = max(remaining_time_ratio, 0.01)
        sigma = params.sigma
        
        delta = 0.5  # ATM 期权
        theta = -0.2 * sigma / math.sqrt(tau)  # 时间衰减
        vega = 0.4 * math.sqrt(tau)  # 对波动率敏感
        
        # Lambda: 对正向跳跃强度的敏感度
        # 正向跳跃强度越高，上行潜力越大
        if len(tracker.positive_jumps) > 0:
            pos_sizes = [abs(j[1]) for j in tracker.positive_jumps]
            lambda_sensitivity = np.mean(pos_sizes) * tau
        else:
            lambda_sensitivity = 0.0
        
        return {
            'delta': delta,
            'theta': theta,
            'vega': vega,
            'lambda_sensitivity': lambda_sensitivity
        }


# ============================================================================
# Component 4: Portfolio Weight Calculator (仓位计算器) - CORRECTED VERSION
# ============================================================================
#
# 数学修正说明：
# ==============
# 原版问题：
#   1. Shapley Value 可能为负，导致 base_weight * adjustment 产生异常权重
#   2. 调整因子使用简单加法叠加，可能导致权重分布极端化
#   3. max(0.1, adjustment) 是临时修补，不是数学上合理的解决方案
#
# 修正方案：使用 Softmax + Temperature 机制
#   w_i^{raw} = max(0, φ_i + ε) × exp((α×S_i + β×O_i + γ×I_i) / T)
#   w_i = w_i^{raw} / Σ_j w_j^{raw}
#
#   其中 T 是温度参数：
#   - T → ∞: 权重趋于均匀（保守分配）
#   - T → 0: 权重趋于 one-hot（赢者通吃）
#
# ============================================================================

class PortfolioWeightCalculator:
    """
    仓位计算器 (修正版) - 使用 Softmax + Temperature 机制
    
    核心修正：
    1. 对 Shapley Value 使用 ReLU + ε 处理负值
    2. 使用 Softmax 而非简单乘法来合成调整因子
    3. 引入温度参数 T 控制权重分布的平滑度
    """
    
    # 温度参数的范围
    TEMP_MIN = 0.1   # 更尖锐的分布
    TEMP_MAX = 10.0  # 更平滑的分布
    TEMP_DEFAULT = 1.0
    
    # Shapley 的最小值（避免负值和零）
    SHAPLEY_EPSILON = 0.01
    
    def __init__(
        self,
        n_agents: int,
        alpha: float = 0.3,   # JD-Sortino 权重
        beta: float = 0.2,    # 期权价值权重（上行潜力）
        gamma: float = 0.2,   # 信息溢价权重
        temperature: float = 1.0  # 【新增】温度参数
    ):
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = max(self.TEMP_MIN, min(self.TEMP_MAX, temperature))
        
        # 追踪器
        self.trackers: Dict[str, JumpDiffusionTracker] = {
            str(i): JumpDiffusionTracker() for i in range(n_agents)
        }
        
        # 计算器
        self.sortino_calculator = JDSortinoCalculator()
        self.option_valuator = MertonOptionValuator()
        
        # 全局统计
        self.total_new_states = 0
        self.total_bugs = 0
        self.agent_new_states: Dict[str, int] = {str(i): 0 for i in range(n_agents)}
        self.agent_bugs: Dict[str, int] = {str(i): 0 for i in range(n_agents)}
        
    def update(self, agent_name: str, reward: float, is_new_state: bool = False, 
               found_bug: bool = False):
        """
        更新 Agent 数据
        
        Args:
            agent_name: Agent 名称
            reward: 当前奖励
            is_new_state: 是否发现新状态
            found_bug: 是否发现 Bug
        """
        if agent_name in self.trackers:
            self.trackers[agent_name].update(reward, is_new_state, found_bug)
            if is_new_state:
                self.total_new_states += 1
                self.agent_new_states[agent_name] = self.agent_new_states.get(agent_name, 0) + 1
            if found_bug:
                self.total_bugs += 1
                self.agent_bugs[agent_name] = self.agent_bugs.get(agent_name, 0) + 1
    
    def compute_info_premium(self, agent_name: str) -> float:
        """
        信息溢价 (修正版)
        
        修正说明：
        - 这里的"信息"是共享的首发优势，而非私有信息
        - 发现新状态 + 发现 Bug 都有价值
        """
        total_discoveries = self.total_new_states + self.total_bugs
        if total_discoveries == 0:
            return 0.0
        
        agent_states = self.agent_new_states.get(agent_name, 0)
        agent_bugs = self.agent_bugs.get(agent_name, 0)
        
        # Bug 发现比普通状态发现更有价值（权重 2.0）
        agent_value = agent_states + 2.0 * agent_bugs
        total_value = self.total_new_states + 2.0 * self.total_bugs
        
        info_share = agent_value / max(total_value, 1)
        return info_share  # 归一化到 [0, 1]
    
    def compute_weights(
        self,
        shapley_values: Dict[str, float],
        remaining_time_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        计算仓位权重 (修正版) - 使用 Softmax + Temperature
        
        公式：
            score_i = α × S_i + β × O_i + γ × I_i
            w_i^{raw} = max(0, φ_i + ε) × exp(score_i / T)
            w_i = w_i^{raw} / Σ_j w_j^{raw}
        
        Returns:
            Dict[str, float]: 归一化的权重，满足 Σ w_i = 1
        """
        scores = {}
        shapley_positive = {}
        
        for agent_name in shapley_values:
            tracker = self.trackers.get(agent_name)
            
            # 【修正1】对 Shapley Value 使用 ReLU + ε
            # 确保基础权重始终为正
            shapley_positive[agent_name] = max(0, shapley_values[agent_name]) + self.SHAPLEY_EPSILON
            
            if not tracker:
                scores[agent_name] = 0.0
                continue
            
            # 计算各组成部分
            # 1. JD-Sortino (已归一化到 [0, 3])
            jd_sortino = self.sortino_calculator.compute(tracker)
            
            # 2. 期权价值（上行潜力，归一化到 [0, 2]）
            option_value = self.option_valuator.compute_option_value(tracker, remaining_time_ratio)
            
            # 3. 信息溢价（归一化到 [0, 1]）
            info_premium = self.compute_info_premium(agent_name)
            
            # 4. 【修正2】使用加权分数（用于 Softmax）
            score = (self.alpha * jd_sortino + 
                    self.beta * option_value + 
                    self.gamma * info_premium)
            scores[agent_name] = score
        
        # 5. 【修正3】使用 Softmax + Temperature 计算权重
        # w_i^{raw} = φ_i^+ × exp(score_i / T)
        raw_weights = {}
        for agent_name in shapley_values:
            shapley_base = shapley_positive[agent_name]
            score = scores.get(agent_name, 0.0)
            
            # Softmax with temperature
            # 数值稳定性：减去最大值
            max_score = max(scores.values()) if scores else 0.0
            exp_score = math.exp((score - max_score) / self.temperature)
            
            raw_weights[agent_name] = shapley_base * exp_score
        
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
    
    def set_temperature(self, temperature: float):
        """
        动态调整温度参数
        
        使用场景：
        - 训练初期：高温度（更均匀的探索）
        - 训练后期：低温度（更集中的利用）
        """
        self.temperature = max(self.TEMP_MIN, min(self.TEMP_MAX, temperature))
    
    def get_diagnostic(self) -> Dict:
        """诊断报告 (增强版)"""
        report = {
            'total_new_states': self.total_new_states,
            'total_bugs': self.total_bugs,
            'temperature': self.temperature,
            'agents': {}
        }
        
        for agent_name, tracker in self.trackers.items():
            jd_sortino = self.sortino_calculator.compute(tracker)
            sortino_detail = self.sortino_calculator.compute_detailed(tracker)
            
            report['agents'][agent_name] = {
                'mean_return': tracker.mean_return,
                'cumulative_value': tracker.cumulative_value,
                'volatility': tracker.volatility,
                'total_jump_intensity': tracker.jump_intensity,
                'positive_jump_intensity': tracker.positive_jump_intensity,
                'negative_jump_intensity': tracker.negative_jump_intensity,
                'positive_jumps_count': len(tracker.positive_jumps),
                'negative_jumps_count': len(tracker.negative_jumps),
                'jd_sortino': jd_sortino,
                'new_state_rate': tracker.new_state_rate,
                'bug_rate': tracker.bug_rate,
                'info_premium': self.compute_info_premium(agent_name),
                'params': {
                    'mu': tracker.estimated_params.mu,
                    'sigma': tracker.estimated_params.sigma,
                    'lambda': tracker.estimated_params.lambda_,
                    'lambda_pos': tracker.estimated_params.lambda_pos,
                    'lambda_neg': tracker.estimated_params.lambda_neg,
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
        self.temperature = params.get("temperature", 1.0)  # 【新增】Softmax 温度参数
        
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
        
        # 【JD-PSHAQ 核心】仓位计算器（使用 Softmax + Temperature）
        self.weight_calculator = PortfolioWeightCalculator(
            n_agents=self.agent_num,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma_info,
            temperature=self.temperature  # 【修正】添加温度参数
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
        
        # 检查是否是新状态
        state_hash = self.dom_encoder.compute_structure_hash(html) if html else ""
        is_new_state = state_hash and hash(state_hash) not in self._seen_dom_signatures
        if is_new_state and state_hash:
            self._seen_dom_signatures.add(hash(state_hash))
        
        # 【修正】检查是否发现 Bug（通过 bug_analyzer 或高奖励判断）
        found_bug = False
        if self.bug_analyzer is not None:
            try:
                # 检查是否有新的 Bug
                prev_bug_count = self.bug_analyzer.get_total_bug_count() if hasattr(self.bug_analyzer, 'get_total_bug_count') else 0
                # Bug 发现逻辑：如果状态是新的且奖励很高，可能发现了 Bug
                found_bug = is_new_state and web_state not in self.state_list[:-5]
            except Exception:
                pass
        
        # 计算基础奖励
        base_reward = self._compute_base_reward(web_state, html, agent_name, found_bug)
        
        # 记录到团队聚合器
        self.team_aggregator.record_base_reward(agent_name, base_reward)
        
        # 【修正】更新 JD 追踪器（传递 found_bug 参数）
        self.weight_calculator.update(agent_name, base_reward, is_new_state, found_bug)
        
        # 【修正】动态调整温度（训练后期降低温度，更集中分配）
        remaining_time = self._get_remaining_time_ratio()
        # 温度从 2.0 线性衰减到 0.5
        dynamic_temp = 0.5 + 1.5 * remaining_time
        self.weight_calculator.set_temperature(dynamic_temp)
        
        # 学习
        self.learn_agent(agent_name, base_reward)
    
    def _compute_base_reward(self, web_state: WebState, html: str, 
                              agent_name: str, found_bug: bool = False) -> float:
        """
        计算基础奖励（不含仓位调整）
        
        奖励组成：
        1. 新状态发现奖励
        2. URL 变化奖励
        3. 动作多样性奖励
        4. 【新增】Bug 发现奖励（正向跳跃）
        """
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
        
        # 【修正】Bug 发现奖励（这会触发 JD 模型中的正向跳跃）
        if found_bug:
            reward += 50.0  # 大幅奖励，对应正向跳跃
        
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
        """诊断报告 (修正版)"""
        report_lines = [
            "=" * 70,
            "JD-PSHAQ Diagnostic Report (Corrected Mathematical Model)",
            "=" * 70,
            f"Algorithm: Jump Diffusion Portfolio-SHAQ v2.0",
            f"Agents: {self.agent_num}",
            f"Learn Steps: {self.learn_step_count}",
            f"Remaining Time: {self._get_remaining_time_ratio():.1%}",
            "",
            "--- Mathematical Model Corrections ---",
            "1. S_t: EMA-smoothed cumulative value (not instant reward)",
            "2. Returns: Arithmetic returns (unified dimensionality)",
            "3. JD-Sortino: Only penalizes negative jumps (not positive!)",
            "4. Weights: Softmax + Temperature mechanism",
            "",
            "--- JD Parameters ---",
        ]
        
        diag = self.weight_calculator.get_diagnostic()
        report_lines.append(f"Total New States: {diag['total_new_states']}")
        report_lines.append(f"Total Bugs Found: {diag.get('total_bugs', 0)}")
        report_lines.append(f"Temperature: {diag.get('temperature', 1.0):.2f}")
        
        for agent_name, metrics in diag['agents'].items():
            report_lines.append(f"\nAgent {agent_name}:")
            report_lines.append(f"  Mean Return: {metrics['mean_return']:.2f}")
            report_lines.append(f"  Cumulative Value: {metrics.get('cumulative_value', 0):.3f}")
            report_lines.append(f"  Volatility: {metrics['volatility']:.3f}")
            report_lines.append(f"  Positive Jumps: {metrics.get('positive_jumps_count', 0)} "
                              f"(λ⁺={metrics['params'].get('lambda_pos', 0):.3f})")
            report_lines.append(f"  Negative Jumps: {metrics.get('negative_jumps_count', 0)} "
                              f"(λ⁻={metrics['params'].get('lambda_neg', 0):.3f})")
            report_lines.append(f"  JD-Sortino: {metrics['jd_sortino']:.2f}")
            report_lines.append(f"  New State Rate: {metrics['new_state_rate']:.2%}")
            report_lines.append(f"  Bug Rate: {metrics.get('bug_rate', 0):.2%}")
            report_lines.append(f"  Info Premium: {metrics.get('info_premium', 0):.3f}")
            report_lines.append(f"  JD Params: μ={metrics['params']['mu']:.3f}, "
                              f"σ={metrics['params']['sigma']:.3f}")
        
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
