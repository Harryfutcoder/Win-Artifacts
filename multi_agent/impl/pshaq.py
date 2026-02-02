"""
P-SHAQ: Portfolio-SHAQ with Financial Pricing for Credit Assignment

核心创新：将金融定价理论引入 MARL 的信度分配

理论基础：
1. Sortino Ratio - 只惩罚下行风险（失败），不惩罚上行波动（探索成功）
2. Information Premium - 发现新状态 = 拥有私有信息，给予"做市商价差"
3. Volatility Tracking - 追踪每个 Agent 的回报波动性

估值公式：
    Agent_Valuation = Shapley_Value + Risk_Premium + Information_Premium

其中：
- Shapley_Value: 传统合作博弈论贡献值
- Risk_Premium: 基于 Sortino Ratio 的风险调整（高波动但上行多的 Agent 获得溢价）
- Information_Premium: 发现新状态/URL 的信息价值

参考文献:
- Sharpe, W. (1966). "Mutual Fund Performance" - Sharpe Ratio
- Sortino, F. (1994). "Performance Measurement in a Downside Risk Framework"
- Black-Scholes (1973). "The Pricing of Options and Corporate Liabilities"
- Wang et al. "SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning"
"""

import math
import random
import threading
import hashlib
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Set
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import multi_agent.multi_agent_system
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from model.replay_buffer import ReplayBuffer
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name
from web_test.multi_agent_thread import logger

# 复用 SHAQv2 的组件
from multi_agent.impl.shaq_v2 import (
    DOMStructureEncoder,
    IntrinsicCuriosityModule,
    MultiObjectiveRewardSystem,
    RoleBasedRewardSystem,
    ShapleyMixingNetwork,
    PseudoELOCTracker,
    create_bug_analysis_system,
)


# ============================================================================
# Component 1: Volatility Tracker (波动率追踪器)
# ============================================================================

class VolatilityTracker:
    """
    波动率追踪器：追踪每个 Agent 的回报波动性
    
    金融启发：
    - 高波动性 Agent 像"妖股"，可能带来巨大收益（发现深层 Bug）
    - 低波动性 Agent 像"蓝筹股"，稳定但无惊喜
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.rewards: deque = deque(maxlen=window_size)
        self.new_state_count = 0
        self.total_steps = 0
        
        # 追踪上行和下行
        self.upside_rewards: List[float] = []
        self.downside_rewards: List[float] = []
        
    def update(self, reward: float, is_new_state: bool = False):
        """更新追踪器"""
        self.rewards.append(reward)
        self.total_steps += 1
        if is_new_state:
            self.new_state_count += 1
        
        # 分类上行/下行
        if reward > 0:
            self.upside_rewards.append(reward)
            if len(self.upside_rewards) > self.window_size:
                self.upside_rewards = self.upside_rewards[-self.window_size:]
        else:
            self.downside_rewards.append(reward)
            if len(self.downside_rewards) > self.window_size:
                self.downside_rewards = self.downside_rewards[-self.window_size:]
    
    @property
    def volatility(self) -> float:
        """计算波动率（标准差）"""
        if len(self.rewards) < 2:
            return 1.0  # 高初始波动率鼓励探索
        return float(np.std(list(self.rewards))) + 0.01
    
    @property
    def mean_return(self) -> float:
        """平均回报"""
        if len(self.rewards) == 0:
            return 0.0
        return float(np.mean(list(self.rewards)))
    
    @property
    def downside_volatility(self) -> float:
        """下行波动率（只看负回报）"""
        if len(self.downside_rewards) < 2:
            return 0.01
        return float(np.std(self.downside_rewards)) + 0.01
    
    @property
    def upside_volatility(self) -> float:
        """上行波动率（只看正回报）"""
        if len(self.upside_rewards) < 2:
            return 0.01
        return float(np.std(self.upside_rewards)) + 0.01
    
    @property
    def new_state_rate(self) -> float:
        """新状态发现率"""
        if self.total_steps == 0:
            return 0.0
        return self.new_state_count / self.total_steps


# ============================================================================
# Component 2: Portfolio Valuator (投资组合估值器) - 修正版
# ============================================================================
#
# 【核心修正】
# 1. 仓位 w_i = R_total 的分配比例（而非探索率调整！）
# 2. 信息价值 = 探索价值 + 验证价值 + 深化价值（而非"私有信息"）
# 3. R_i = w_i × R_total，约束 Σ w_i = 1
# ============================================================================

class InformationValueModel:
    """
    信息价值模型 - 修正版
    
    【原问题】：错误地将"新状态发现"类比为"私有信息"
    【修正】：信息是共享的，区分三种贡献价值：
    
    1. 探索价值 (Exploration Value): 首次发现新状态
    2. 验证价值 (Verification Value): 独立到达已知状态（确认可达）
    3. 深化价值 (Deepening Value): 在已知状态发现新 Bug/功能
    """
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        
        # 状态发现记录：state_hash -> {discoverer, verifiers, deepeners}
        self.state_records: Dict[str, Dict] = {}
        
        # Agent 贡献统计
        self.agent_exploration: Dict[str, int] = {str(i): 0 for i in range(n_agents)}
        self.agent_verification: Dict[str, int] = {str(i): 0 for i in range(n_agents)}
        self.agent_deepening: Dict[str, int] = {str(i): 0 for i in range(n_agents)}
        
        # 价值权重
        self.EXPLORATION_WEIGHT = 1.0   # 首发奖励
        self.VERIFICATION_WEIGHT = 0.3  # 验证也有价值！
        self.DEEPENING_WEIGHT = 1.5     # 深化发现更有价值
        
    def record_visit(
        self, 
        agent_name: str, 
        state_hash: str, 
        found_new_bug: bool = False
    ) -> Tuple[str, float]:
        """
        记录 Agent 访问状态
        
        Returns:
            (contribution_type, value): 贡献类型和价值
        """
        if state_hash not in self.state_records:
            # 首次发现 -> 探索价值
            self.state_records[state_hash] = {
                'discoverer': agent_name,
                'verifiers': set(),
                'deepeners': set(),
                'visit_count': 1
            }
            self.agent_exploration[agent_name] = self.agent_exploration.get(agent_name, 0) + 1
            return ('exploration', self.EXPLORATION_WEIGHT)
        
        record = self.state_records[state_hash]
        record['visit_count'] += 1
        
        if found_new_bug:
            # 在已知状态发现新 Bug -> 深化价值
            if agent_name not in record['deepeners']:
                record['deepeners'].add(agent_name)
                self.agent_deepening[agent_name] = self.agent_deepening.get(agent_name, 0) + 1
                return ('deepening', self.DEEPENING_WEIGHT)
        
        if agent_name != record['discoverer'] and agent_name not in record['verifiers']:
            # 独立到达 -> 验证价值（不是 0！）
            record['verifiers'].add(agent_name)
            self.agent_verification[agent_name] = self.agent_verification.get(agent_name, 0) + 1
            return ('verification', self.VERIFICATION_WEIGHT)
        
        # 重复访问自己发现的状态，价值递减
        return ('revisit', 0.1)
    
    def compute_total_value(self, agent_name: str) -> float:
        """计算 Agent 的总信息价值"""
        exploration = self.agent_exploration.get(agent_name, 0) * self.EXPLORATION_WEIGHT
        verification = self.agent_verification.get(agent_name, 0) * self.VERIFICATION_WEIGHT
        deepening = self.agent_deepening.get(agent_name, 0) * self.DEEPENING_WEIGHT
        return exploration + verification + deepening
    
    def get_info_breakdown(self, agent_name: str) -> Dict[str, float]:
        """获取信息价值分解"""
        return {
            'exploration': self.agent_exploration.get(agent_name, 0),
            'verification': self.agent_verification.get(agent_name, 0),
            'deepening': self.agent_deepening.get(agent_name, 0),
            'total_value': self.compute_total_value(agent_name)
        }


class TeamRewardPool:
    """
    团队奖励池 - 用于正确的仓位分配
    
    【核心修正】
    金融中的仓位 = 资金分配比例
    MARL 中的仓位 = R_total 的分配比例
    
    公式：R_i = w_i × R_total，约束 Σ w_i = 1
    """
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.current_step_base_rewards: Dict[str, float] = {}
        self.step_counter = 0
        
    def record_base_reward(self, agent_name: str, base_reward: float):
        """记录 Agent 的基础奖励（用于计算 R_total）"""
        self.current_step_base_rewards[agent_name] = base_reward
    
    def compute_r_total(self) -> float:
        """计算团队总奖励 R_total"""
        return sum(self.current_step_base_rewards.values())
    
    def distribute_rewards(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        根据仓位分配奖励
        
        R_i = w_i × R_total
        """
        r_total = self.compute_r_total()
        distributed = {}
        
        for agent_name, weight in weights.items():
            distributed[agent_name] = weight * r_total
        
        return distributed
    
    def clear_step(self):
        """清空当前步骤"""
        self.current_step_base_rewards = {}
        self.step_counter += 1


class PortfolioValuator:
    """
    投资组合估值器 - 修正版
    
    【核心修正】
    1. 仓位 w_i 是 R_total 的分配比例（不是探索率调整！）
    2. 使用正确的信息价值模型（探索+验证+深化）
    3. R_i = w_i × R_total，约束 Σ w_i = 1
    
    公式：
        w_i^{base} = φ_i (Shapley Value，已归一化)
        w_i^{adj} = w_i^{base} × (1 + α×(Sortino-1) + β×InfoValue)
        w_i = w_i^{adj} / Σ w_j^{adj}  (归一化确保 Σ=1)
    """
    
    def __init__(
        self, 
        n_agents: int, 
        risk_weight: float = 0.3,
        info_weight: float = 0.4,
        mar: float = 0.0
    ):
        self.n_agents = n_agents
        self.risk_weight = risk_weight
        self.info_weight = info_weight
        self.mar = mar
        
        # 波动率追踪器
        self.trackers: Dict[str, VolatilityTracker] = {
            str(i): VolatilityTracker() for i in range(n_agents)
        }
        
        # 【修正】使用正确的信息价值模型
        self.info_model = InformationValueModel(n_agents)
        
        # 【修正】团队奖励池
        self.reward_pool = TeamRewardPool(n_agents)
        
        # 历史记录
        self.valuation_history: List[Dict] = []
        
        # 统计
        self.total_new_states = 0
        self.agent_new_states: Dict[str, int] = {str(i): 0 for i in range(n_agents)}
        
    def update(
        self, 
        agent_name: str, 
        reward: float, 
        state_hash: str = "",
        found_new_bug: bool = False
    ):
        """更新 Agent 数据"""
        # 更新波动率追踪
        is_new_state = state_hash and state_hash not in [
            h for h in self.info_model.state_records.keys()
        ]
        
        if agent_name in self.trackers:
            self.trackers[agent_name].update(reward, is_new_state)
        
        # 【修正】使用正确的信息价值模型
        if state_hash:
            contrib_type, contrib_value = self.info_model.record_visit(
                agent_name, state_hash, found_new_bug
            )
            if contrib_type == 'exploration':
                self.total_new_states += 1
                self.agent_new_states[agent_name] = self.agent_new_states.get(agent_name, 0) + 1
    
    def compute_sortino_ratio(self, agent_name: str) -> float:
        """计算 Sortino Ratio"""
        tracker = self.trackers.get(agent_name)
        if not tracker or len(tracker.rewards) < 5:
            return 1.0
        
        mean_return = tracker.mean_return
        downside_dev = tracker.downside_volatility
        
        sortino = (mean_return - self.mar) / downside_dev
        return max(0.0, min(2.0, sortino + 1.0))
    
    def compute_information_value(self, agent_name: str) -> float:
        """
        计算信息价值 - 使用修正后的三层模型
        """
        total_value = self.info_model.compute_total_value(agent_name)
        
        # 归一化
        all_values = [self.info_model.compute_total_value(str(i)) for i in range(self.n_agents)]
        max_value = max(all_values) if all_values else 1.0
        
        if max_value > 0:
            return total_value / max_value
        return 0.0
    
    def compute_upside_potential(self, agent_name: str) -> float:
        """计算上行潜力"""
        tracker = self.trackers.get(agent_name)
        if not tracker:
            return 0.0
        
        upside_vol = tracker.upside_volatility
        downside_vol = tracker.downside_volatility
        
        if downside_vol < 0.01:
            return 1.0
        
        ratio = upside_vol / downside_vol
        return min(2.0, ratio)
    
    def compute_weights(
        self, 
        shapley_values: Dict[str, float],
        remaining_steps_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        计算仓位权重
        
        【核心】这是 R_total 的分配比例，不是探索率调整！
        
        公式：
            w_i^{base} = φ_i
            w_i^{adj} = w_i^{base} × (1 + α×(Sortino-1) + β×InfoValue + γ×Upside×TimeRatio)
            w_i = w_i^{adj} / Σ w_j^{adj}
        
        约束：Σ w_i = 1
        """
        raw_weights = {}
        
        for agent_name, shapley in shapley_values.items():
            # 1. 基础仓位 = Shapley Value
            base_weight = shapley
            
            # 2. Sortino 调整
            sortino = self.compute_sortino_ratio(agent_name)
            sortino_adj = self.risk_weight * (sortino - 1.0)
            
            # 3. 信息价值调整（使用修正后的模型）
            info_value = self.compute_information_value(agent_name)
            info_adj = self.info_weight * info_value
            
            # 4. 上行潜力 × 时间因子
            upside = self.compute_upside_potential(agent_name)
            time_adj = 0.1 * upside * remaining_steps_ratio
            
            # 5. 综合调整
            adjustment = 1.0 + sortino_adj + info_adj + time_adj
            adjustment = max(0.1, adjustment)  # 防止负数
            
            raw_weights[agent_name] = base_weight * adjustment
            
            # 记录历史
            self.valuation_history.append({
                'agent': agent_name,
                'base_weight': base_weight,
                'sortino_adj': sortino_adj,
                'info_adj': info_adj,
                'time_adj': time_adj,
                'final_raw_weight': raw_weights[agent_name]
            })
        
        if len(self.valuation_history) > 1000:
            self.valuation_history = self.valuation_history[-1000:]
        
        # 6. 归一化（确保 Σ w_i = 1）
        total = sum(raw_weights.values())
        if total > 0:
            return {k: v / total for k, v in raw_weights.items()}
        else:
            return {k: 1.0 / len(raw_weights) for k in raw_weights}
    
    def distribute_r_total(
        self,
        r_total: float,
        shapley_values: Dict[str, float],
        remaining_steps_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        分配团队总奖励
        
        【核心公式】R_i = w_i × R_total
        """
        weights = self.compute_weights(shapley_values, remaining_steps_ratio)
        return {agent: w * r_total for agent, w in weights.items()}
    
    # 保留旧接口的兼容性
    def get_portfolio_weights(
        self, 
        shapley_values: Dict[str, float],
        remaining_steps_ratio: float = 1.0
    ) -> Dict[str, float]:
        """获取仓位权重（兼容旧接口）"""
        return self.compute_weights(shapley_values, remaining_steps_ratio)
    
    def compute_valuation(
        self, 
        agent_name: str, 
        shapley_value: float,
        remaining_steps_ratio: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """计算估值（兼容旧接口，但内部使用新逻辑）"""
        weights = self.compute_weights({agent_name: shapley_value}, remaining_steps_ratio)
        weight = weights.get(agent_name, shapley_value)
        
        breakdown = {
            'base_value': shapley_value,
            'risk_premium': self.risk_weight * self.compute_sortino_ratio(agent_name),
            'info_premium': self.info_weight * self.compute_information_value(agent_name),
            'time_value': 0.1 * self.compute_upside_potential(agent_name) * remaining_steps_ratio,
            'sortino_ratio': self.compute_sortino_ratio(agent_name),
            'upside_potential': self.compute_upside_potential(agent_name),
            'final_weight': weight,
        }
        
        return weight, breakdown
    
    def get_diagnostic_report(self) -> Dict:
        """获取诊断报告"""
        report = {
            'total_new_states': self.total_new_states,
            'agent_metrics': {}
        }
        
        for agent_name, tracker in self.trackers.items():
            info_breakdown = self.info_model.get_info_breakdown(agent_name)
            report['agent_metrics'][agent_name] = {
                'mean_return': tracker.mean_return,
                'volatility': tracker.volatility,
                'downside_vol': tracker.downside_volatility,
                'upside_vol': tracker.upside_volatility,
                'new_state_rate': tracker.new_state_rate,
                'sortino_ratio': self.compute_sortino_ratio(agent_name),
                'info_value': self.compute_information_value(agent_name),
                'info_breakdown': info_breakdown,  # 探索/验证/深化分解
            }
        
        return report


# ============================================================================
# Main Class: P-SHAQ
# ============================================================================

class PSHAQ(multi_agent.multi_agent_system.MultiAgentSystem):
    """
    P-SHAQ: Portfolio-SHAQ with Financial Pricing
    
    核心创新：
    1. Sortino Ratio 替代简单的 Shapley 值（只惩罚失败，不惩罚探索）
    2. Information Premium 奖励发现新状态的 Agent
    3. Portfolio Optimization 视角的权重分配
    
    相比 SHAQv2 的改进：
    - 更智能的探索-利用平衡（金融定价自动调节）
    - 对高方差但高潜力的 Agent 更友好
    - 信息发现获得额外奖励
    """
    
    def __init__(self, params: Dict):
        super().__init__(params)
        self.params = params
        self.algo_type = params.get("algo_type", "pshaq")
        self.reward_function = params.get("reward_function", "A")
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"P-SHAQ: Using {self.device}")
        
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
        
        # P-SHAQ 特有参数
        self.risk_weight = params.get("risk_weight", 0.3)
        self.info_weight = params.get("info_weight", 0.4)
        
        # ICM 参数
        self.use_icm = params.get("use_icm", True)
        self.icm_weight = params.get("icm_weight", 0.5)
        
        # 角色分工参数
        self.use_role_based = params.get("use_role_based", True)
        
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
        
        # Shapley 混合网络
        self.mixing_network = ShapleyMixingNetwork(n_agents=self.agent_num, embed_dim=64).to(self.device)
        self.target_mixing_network = ShapleyMixingNetwork(n_agents=self.agent_num, embed_dim=64).to(self.device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=self.learning_rate)
        
        # 内在好奇心模块 (ICM)
        if self.use_icm:
            self.icm = IntrinsicCuriosityModule(
                state_dim=64, 
                action_dim=12,
                alive_time=self.alive_time
            ).to(self.device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=self.learning_rate * 0.5)
        else:
            self.icm = None
        
        # 多目标奖励系统
        self.reward_system = MultiObjectiveRewardSystem(
            mode='three_tier',
            alive_time=self.alive_time
        )
        
        # 角色分工系统
        self.role_system = RoleBasedRewardSystem(self.agent_num)
        
        # 【P-SHAQ 核心】投资组合估值器
        self.portfolio_valuator = PortfolioValuator(
            n_agents=self.agent_num,
            risk_weight=self.risk_weight,
            info_weight=self.info_weight
        )
        
        # DOM 编码器
        self.dom_encoder = DOMStructureEncoder()
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.replay_buffer_agent: Dict[str, ReplayBuffer] = {
            str(i): ReplayBuffer(capacity=500) for i in range(self.agent_num)
        }
        
        # 同步
        self.finish_dict_agent: Dict[str, bool] = {str(i): False for i in range(self.agent_num)}
        self.prev_state_success_dict: Dict[str, Optional[WebState]] = {}
        self.prev_action_success_dict: Dict[str, Optional[WebAction]] = {}
        self.current_state_success_dict: Dict[str, Optional[WebState]] = {}
        self.prev_html_success_dict: Dict[str, str] = {}
        
        for i in range(self.agent_num):
            agent_name = str(i)
            self.prev_state_success_dict[agent_name] = None
            self.prev_action_success_dict[agent_name] = None
            self.current_state_success_dict[agent_name] = None
            self.prev_html_success_dict[agent_name] = ""
        
        # Shapley 缓存
        self.cached_shapley_values: Dict[str, float] = {
            str(i): 1.0 / self.agent_num for i in range(self.agent_num)
        }
        self.shapley_update_counter = 0
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 追踪团队访问的 URL
        self.team_visited_urls: Set[str] = set()
        self._seen_dom_signatures: Set[int] = set()
        
        # 浏览器日志存储
        self.agent_browser_logs: Dict[str, List[Dict]] = {}
        self.agent_performance_logs: Dict[str, List[Dict]] = {}
        self.logs_lock = threading.Lock()
        
        # Bug 分析系统
        self.bug_analyzer, self.bug_localizer = create_bug_analysis_system()
        
        # 追踪字典（用于学习）
        self.prev_state_dict: Dict[str, Optional[WebState]] = {str(i): None for i in range(self.agent_num)}
        self.prev_action_dict: Dict[str, Optional[WebAction]] = {str(i): None for i in range(self.agent_num)}
        self.prev_html_dict: Dict[str, str] = {str(i): "" for i in range(self.agent_num)}
        self.action_dict: Dict[WebAction, int] = {}
        
        logger.info(f"P-SHAQ initialized with {self.agent_num} agents, "
                   f"ICM: {self.use_icm}, Role-based: {self.use_role_based}, "
                   f"Risk Weight: {self.risk_weight}, Info Weight: {self.info_weight}")
    
    def set_agent_logs(self, agent_name: str, browser_logs: List[Dict], performance_logs: List[Dict]):
        """存储 Agent 的浏览器日志"""
        with self.logs_lock:
            self.agent_browser_logs[agent_name] = browser_logs
            self.agent_performance_logs[agent_name] = performance_logs
    
    def get_agent_logs(self, agent_name: str) -> Tuple[List[Dict], List[Dict]]:
        """获取 Agent 的浏览器日志"""
        with self.logs_lock:
            browser_logs = self.agent_browser_logs.get(agent_name, [])
            performance_logs = self.agent_performance_logs.get(agent_name, [])
            self.agent_browser_logs[agent_name] = []
            self.agent_performance_logs[agent_name] = []
        return browser_logs, performance_logs
    
    def _get_remaining_steps_ratio(self) -> float:
        """获取剩余步数比例（用于时间价值计算）"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining_ratio = max(0.0, 1.0 - elapsed / self.alive_time)
        return remaining_ratio
    
    def _compute_portfolio_adjusted_reward(
        self, 
        agent_name: str, 
        base_reward: float,
        state_hash: str = "",
        found_new_bug: bool = False
    ) -> float:
        """
        计算投资组合调整后的奖励 - 修正版
        
        【核心修正】
        原来的错误：R_i = R_base × V_i（用估值放大奖励）
        修正后：R_i = w_i × R_total（用仓位分配团队奖励）
        
        由于多 Agent 异步执行，这里使用简化版：
        R_i = base_reward × (w_i × n_agents)
        
        这等价于：如果所有 Agent 贡献相同，按仓位重新分配
        """
        # 1. 更新追踪器（使用修正后的信息价值模型）
        self.portfolio_valuator.update(agent_name, base_reward, state_hash, found_new_bug)
        
        # 2. 记录到团队奖励池
        self.portfolio_valuator.reward_pool.record_base_reward(agent_name, base_reward)
        
        # 3. 获取仓位权重
        remaining_ratio = self._get_remaining_steps_ratio()
        weights = self.portfolio_valuator.compute_weights(
            self.cached_shapley_values, remaining_ratio
        )
        weight = weights.get(agent_name, 1.0 / self.agent_num)
        
        # 4.【修正】使用仓位分配奖励
        # 简化版：R_i = base_reward × (w_i × n_agents)
        # 这保证了如果所有 Agent 贡献相同，总奖励不变
        adjusted_reward = base_reward * weight * self.agent_num
        
        # 日志（每100步记录一次）
        if self.learn_step_count % 100 == 0:
            info_breakdown = self.portfolio_valuator.info_model.get_info_breakdown(agent_name)
            logger.debug(f"[P-SHAQ Reward] Agent {agent_name}: "
                        f"base={base_reward:.2f}, weight={weight:.3f}, "
                        f"adjusted={adjusted_reward:.2f}, "
                        f"info_breakdown={info_breakdown}")
        
        return adjusted_reward
    
    def choose_action(self, agent_name: str, state: WebState, action_list: List[WebAction]) -> WebAction:
        """
        选择动作
        
        【修正】不再用仓位调整探索率！
        仓位是用于 R_total 分配的，不是用于调整 ε 的。
        使用标准的时间衰减 ε-greedy。
        """
        if len(action_list) == 0:
            return RestartAction()
        
        # 【修正】标准 ε-greedy，不用仓位调整
        progress = 1.0 - self._get_remaining_steps_ratio()
        epsilon = self.max_random - (self.max_random - self.min_random) * progress
        epsilon = min(self.max_random, max(self.min_random, epsilon))
        
        if random.random() < epsilon:
            # 探索：随机选择
            return random.choice(action_list)
        else:
            # 利用：选择 Q 值最高的动作
            return self._select_best_action(agent_name, state, action_list)
    
    def _select_best_action(self, agent_name: str, state: WebState, action_list: List[WebAction]) -> WebAction:
        """选择 Q 值最高的动作"""
        best_action = None
        best_q = float('-inf')
        
        q_net = self.q_eval_agent.get(agent_name)
        if q_net is None:
            return random.choice(action_list)
        
        for action in action_list:
            try:
                state_vec = self.transformer.transform(state, self.state_list)
                action_vec = self.transformer.transform_action(action)
                
                if state_vec is None or action_vec is None:
                    continue
                
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
                action_tensor = torch.FloatTensor(action_vec).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_value = q_net(state_tensor, action_tensor).item()
                
                if q_value > best_q:
                    best_q = q_value
                    best_action = action
            except Exception:
                continue
        
        return best_action if best_action else random.choice(action_list)
    
    def update_policy(
        self,
        agent_name: str,
        prev_state: WebState,
        prev_action: WebAction,
        reward: float,
        current_state: WebState,
        done: bool,
        html: str = "",
        prev_html: str = ""
    ):
        """更新策略（核心：使用投资组合调整后的奖励）"""
        self.learn_step_count += 1
        
        # 检查是否是新状态
        state_hash = self.dom_encoder.compute_structure_hash(html) if html else ""
        is_new_state = state_hash and hash(state_hash) not in self._seen_dom_signatures
        if is_new_state and state_hash:
            self._seen_dom_signatures.add(hash(state_hash))
        
        # 【P-SHAQ 核心】计算投资组合调整后的奖励
        adjusted_reward = self._compute_portfolio_adjusted_reward(
            agent_name, reward, is_new_state
        )
        
        # 存储经验
        try:
            state_vec = self.transformer.transform(prev_state, self.state_list)
            action_vec = self.transformer.transform_action(prev_action)
            next_state_vec = self.transformer.transform(current_state, self.state_list)
            
            if state_vec is not None and action_vec is not None and next_state_vec is not None:
                self.replay_buffer_agent[agent_name].push(
                    state_vec, action_vec, adjusted_reward, next_state_vec, done
                )
        except Exception as e:
            logger.debug(f"Error storing experience: {e}")
        
        # 记录状态
        if current_state not in self.state_list:
            self.state_list.append(current_state)
        if agent_name in self.state_list_agent:
            self.state_list_agent[agent_name].append(current_state)
        
        # 更新 Shapley 值
        self.shapley_update_counter += 1
        if self.shapley_update_counter >= self.shapley_update_interval:
            self._update_shapley_values()
            self.shapley_update_counter = 0
        
        # 学习
        if self.learn_step_count % self.update_network_interval == 0:
            self._learn(agent_name)
        
        # 更新目标网络
        if self.learn_step_count % self.update_target_interval == 0:
            self._update_target_networks()
    
    def _update_shapley_values(self):
        """更新 Shapley 值"""
        # 简化版：基于最近贡献计算
        contributions = {}
        for agent_name in self.cached_shapley_values.keys():
            tracker = self.portfolio_valuator.trackers.get(agent_name)
            if tracker:
                # 贡献 = 平均回报 + 新状态发现率
                contrib = tracker.mean_return + 10 * tracker.new_state_rate
                contributions[agent_name] = max(0.01, contrib)
            else:
                contributions[agent_name] = 0.01
        
        # 归一化
        total = sum(contributions.values())
        if total > 0:
            for agent_name in self.cached_shapley_values:
                self.cached_shapley_values[agent_name] = contributions[agent_name] / total
    
    def _learn(self, agent_name: str):
        """学习（Q-learning 更新）"""
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
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            # 当前 Q 值
            q_values = q_eval(states_t, actions_t)
            
            # 目标 Q 值
            with torch.no_grad():
                next_q_values = q_target(next_states_t, actions_t)
                target_q = rewards_t + self.gamma * next_q_values * (1 - dones_t)
            
            # 损失
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
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
    
    def get_diagnostic_report(self) -> str:
        """获取诊断报告"""
        report_lines = [
            "=" * 60,
            "P-SHAQ Diagnostic Report",
            "=" * 60,
            f"Algorithm: Portfolio-SHAQ (Financial Pricing)",
            f"Agents: {self.agent_num}",
            f"Learn Steps: {self.learn_step_count}",
            f"Risk Weight: {self.risk_weight}",
            f"Info Weight: {self.info_weight}",
            "",
            "--- Portfolio Metrics ---",
        ]
        
        # 投资组合报告
        portfolio_report = self.portfolio_valuator.get_diagnostic_report()
        report_lines.append(f"Total New States: {portfolio_report['total_new_states']}")
        
        for agent_name, metrics in portfolio_report['agent_metrics'].items():
            report_lines.append(f"\nAgent {agent_name}:")
            report_lines.append(f"  Mean Return: {metrics['mean_return']:.2f}")
            report_lines.append(f"  Volatility: {metrics['volatility']:.2f}")
            report_lines.append(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            report_lines.append(f"  Info Premium: {metrics['info_premium']:.2f}")
            report_lines.append(f"  New State Rate: {metrics['new_state_rate']:.2%}")
        
        # Shapley 值
        report_lines.append("\n--- Shapley Values ---")
        for agent_name, shapley in self.cached_shapley_values.items():
            report_lines.append(f"  Agent {agent_name}: {shapley:.4f}")
        
        # Portfolio 权重
        weights = self.portfolio_valuator.get_portfolio_weights(
            self.cached_shapley_values,
            self._get_remaining_steps_ratio()
        )
        report_lines.append("\n--- Portfolio Weights ---")
        for agent_name, weight in weights.items():
            report_lines.append(f"  Agent {agent_name}: {weight:.4f}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def get_tensor(self, action: WebAction, html: str, web_state: WebState) -> torch.Tensor:
        """将状态-动作对编码为张量"""
        state_tensor = self.transformer.state_to_tensor(web_state, html)
        execution_time = self.action_dict.get(action, 0)
        action_tensor = self.transformer.action_to_tensor(web_state, action, execution_time)
        tensor = torch.cat((state_tensor, action_tensor))
        return tensor.float()
    
    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        """
        动作选择 - 修正版
        
        【修正】仓位是用于 R_total 分配的，不是用于调整探索率的！
        使用标准的时间衰减 ε-greedy 策略。
        """
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
        
        logger.info(f"[{agent_name}] P-SHAQ max Q: {max_val:.4f}")
        
        # 【修正】标准 ε-greedy，不再用仓位调整探索率
        time_diff = (datetime.now() - self.start_time).total_seconds()
        time_diff = min(time_diff, self.alive_time)
        
        epsilon = self.max_random - min(time_diff / self.alive_time * 2, 1.0) * (
            self.max_random - self.min_random
        )
        epsilon = min(self.max_random, max(self.min_random, epsilon))
        
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
        """更新状态记录并触发学习"""
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
        
        # 计算奖励
        reward = self.get_reward(web_state, html, agent_name)
        
        tensor = self.get_tensor(
            self.prev_action_dict[agent_name],
            self.prev_html_dict[agent_name],
            self.prev_state_dict[agent_name]
        )
        tensor = tensor.unsqueeze(0)
        
        done = not isinstance(web_state, ActionSetWithExecutionTimesState)
        
        self.replay_buffer_agent[agent_name].push(
            tensor, tensor, reward, web_state, html, done
        )
        
        self.learn_agent(agent_name)
    
    def get_reward(self, web_state: WebState, html: str, agent_name: str) -> float:
        """
        计算奖励 - 修正版
        
        【核心修正】使用仓位分配 R_total，而非用估值放大奖励
        公式：R_i = w_i × R_total（简化版：R_i = base × w_i × n_agents）
        """
        # 基础奖励：使用三层奖励系统
        browser_logs, performance_logs = self.get_agent_logs(agent_name)
        
        base_reward, breakdown = self.reward_system.compute_three_tier_reward(
            web_state=web_state,
            action=self.prev_action_dict.get(agent_name),
            browser_logs=browser_logs,
            performance_logs=performance_logs,
            http_status=200,
            html=html,
            agent_name=agent_name
        )
        
        # 检查是否是新状态和是否发现新 Bug
        state_hash = self.dom_encoder.compute_structure_hash(html) if html else ""
        found_new_bug = breakdown.get('target:js_error', 0) > 0 or breakdown.get('target:http_error', 0) > 0
        
        # 【修正】使用正确的仓位分配方法
        adjusted_reward = self._compute_portfolio_adjusted_reward(
            agent_name, base_reward, state_hash, found_new_bug
        )
        
        return adjusted_reward
    
    def learn_agent(self, agent_name: str):
        """单个 Agent 学习"""
        self.learn_step_count += 1
        
        if self.learn_step_count % self.update_network_interval != 0:
            return
        
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
            
            # 处理不同的 buffer 格式
            if len(batch[0]) == 6:
                # (state, action, reward, next_state, html, done)
                states = torch.stack([b[0].squeeze(0) for b in batch]).to(self.device)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
                dones = torch.tensor([float(b[5]) for b in batch]).unsqueeze(1).to(self.device)
            else:
                # 其他格式
                return
            
            # 当前 Q 值
            from model.dense_net import DenseNet
            if isinstance(q_eval, DenseNet):
                q_values = q_eval(states.unsqueeze(1))
            else:
                q_values = q_eval(states)
            
            # 目标 Q 值
            with torch.no_grad():
                if isinstance(q_target, DenseNet):
                    next_q = q_target(states.unsqueeze(1))
                else:
                    next_q = q_target(states)
                target_q = rewards + self.gamma * next_q * (1 - dones)
            
            # 损失
            loss = self.criterion(q_values, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            logger.debug(f"Learning error: {e}")
        
        # 更新 Shapley 值
        self.shapley_update_counter += 1
        if self.shapley_update_counter >= self.shapley_update_interval:
            self._update_shapley_values()
            self.shapley_update_counter = 0
        
        # 更新目标网络
        if self.learn_step_count % self.update_target_interval == 0:
            self._update_target_networks()
    
    def set_prev(self, agent_name: str, state: WebState, action: WebAction, html: str):
        """设置前一个状态/动作（由 MultiAgentThread 调用）"""
        self.prev_state_dict[agent_name] = state
        self.prev_action_dict[agent_name] = action
        self.prev_html_dict[agent_name] = html
    
    def finish_episode(self, agent_name: str):
        """结束 episode"""
        self.finish_dict_agent[agent_name] = True
        
        # 打印诊断报告
        if all(self.finish_dict_agent.values()):
            logger.info(self.get_diagnostic_report())
