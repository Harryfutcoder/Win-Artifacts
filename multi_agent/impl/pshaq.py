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
# Component 2: Portfolio Valuator (投资组合估值器)
# ============================================================================

class PortfolioValuator:
    """
    投资组合估值器：为每个 Agent 计算综合估值
    
    核心公式：
        Valuation = Shapley_Value + α * Risk_Premium + β * Information_Premium
    
    其中：
    - Risk_Premium 基于 Sortino Ratio（只惩罚下行风险）
    - Information_Premium 基于新状态发现（私有信息价值）
    """
    
    def __init__(
        self, 
        n_agents: int, 
        risk_weight: float = 0.3,
        info_weight: float = 0.4,
        mar: float = 0.0  # Minimum Acceptable Return
    ):
        self.n_agents = n_agents
        self.risk_weight = risk_weight
        self.info_weight = info_weight
        self.mar = mar
        
        # 每个 Agent 的追踪器
        self.trackers: Dict[str, VolatilityTracker] = {
            str(i): VolatilityTracker() for i in range(n_agents)
        }
        
        # 全局新状态追踪
        self.total_new_states = 0
        self.agent_new_states: Dict[str, int] = {str(i): 0 for i in range(n_agents)}
        
        # 历史估值（用于诊断）
        self.valuation_history: List[Dict] = []
        
    def update(self, agent_name: str, reward: float, is_new_state: bool = False):
        """更新 Agent 的追踪数据"""
        if agent_name in self.trackers:
            self.trackers[agent_name].update(reward, is_new_state)
            if is_new_state:
                self.total_new_states += 1
                self.agent_new_states[agent_name] = self.agent_new_states.get(agent_name, 0) + 1
    
    def compute_sortino_ratio(self, agent_name: str) -> float:
        """
        计算 Sortino Ratio
        
        Sortino = (Mean Return - MAR) / Downside Deviation
        
        优势：只惩罚下行风险（失败），不惩罚上行波动（探索成功）
        """
        tracker = self.trackers.get(agent_name)
        if not tracker or len(tracker.rewards) < 5:
            return 1.0  # 初期给高值鼓励探索
        
        mean_return = tracker.mean_return
        downside_dev = tracker.downside_volatility
        
        sortino = (mean_return - self.mar) / downside_dev
        
        # 归一化到 0-2 范围
        return max(0.0, min(2.0, sortino + 1.0))
    
    def compute_information_premium(self, agent_name: str) -> float:
        """
        计算信息溢价
        
        金融启发：发现新状态 = 拥有私有信息 = 做市商价差
        """
        if self.total_new_states == 0:
            return 0.0
        
        agent_discoveries = self.agent_new_states.get(agent_name, 0)
        
        # 信息溢价 = 该 Agent 发现的新状态占比
        info_share = agent_discoveries / max(self.total_new_states, 1)
        
        # 奖励探索者：发现越多新状态，溢价越高
        return info_share * 2.0  # 最大 2.0
    
    def compute_upside_potential(self, agent_name: str) -> float:
        """
        计算上行潜力（期权价值的简化版）
        
        金融启发：高上行波动 = 看涨期权价值高
        """
        tracker = self.trackers.get(agent_name)
        if not tracker:
            return 0.0
        
        upside_vol = tracker.upside_volatility
        downside_vol = tracker.downside_volatility
        
        # 上行/下行比率：如果上行波动大于下行，说明有潜力
        if downside_vol < 0.01:
            return 1.0
        
        ratio = upside_vol / downside_vol
        return min(2.0, ratio)  # 最大 2.0
    
    def compute_valuation(
        self, 
        agent_name: str, 
        shapley_value: float,
        remaining_steps_ratio: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算 Agent 的综合估值
        
        Args:
            agent_name: Agent 名称
            shapley_value: 传统 Shapley 值
            remaining_steps_ratio: 剩余步数比例（用于时间价值）
        
        Returns:
            (total_valuation, breakdown_dict)
        """
        # 1. 基础价值：Shapley
        base_value = shapley_value
        
        # 2. 风险溢价：Sortino Ratio
        sortino = self.compute_sortino_ratio(agent_name)
        risk_premium = self.risk_weight * sortino
        
        # 3. 信息溢价：新状态发现
        info_premium = self.info_weight * self.compute_information_premium(agent_name)
        
        # 4. 上行潜力（类似期权的时间价值）
        upside = self.compute_upside_potential(agent_name)
        time_value = 0.1 * upside * remaining_steps_ratio  # 随时间递减
        
        # 综合估值
        total_valuation = base_value + risk_premium + info_premium + time_value
        
        breakdown = {
            'base_value': base_value,
            'risk_premium': risk_premium,
            'info_premium': info_premium,
            'time_value': time_value,
            'sortino_ratio': sortino,
            'upside_potential': upside,
        }
        
        # 记录历史
        self.valuation_history.append({
            'agent': agent_name,
            'valuation': total_valuation,
            **breakdown
        })
        if len(self.valuation_history) > 1000:
            self.valuation_history = self.valuation_history[-1000:]
        
        return total_valuation, breakdown
    
    def get_portfolio_weights(
        self, 
        shapley_values: Dict[str, float],
        remaining_steps_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        获取投资组合权重（归一化的 Agent 估值）
        
        类似于 ETF 中各成分股的权重
        """
        valuations = {}
        for agent_name, shapley in shapley_values.items():
            val, _ = self.compute_valuation(agent_name, shapley, remaining_steps_ratio)
            valuations[agent_name] = max(0.01, val)  # 确保正数
        
        # 归一化
        total = sum(valuations.values())
        if total > 0:
            return {k: v / total for k, v in valuations.items()}
        else:
            return {k: 1.0 / len(valuations) for k in valuations}
    
    def get_diagnostic_report(self) -> Dict:
        """获取诊断报告"""
        report = {
            'total_new_states': self.total_new_states,
            'agent_metrics': {}
        }
        
        for agent_name, tracker in self.trackers.items():
            report['agent_metrics'][agent_name] = {
                'mean_return': tracker.mean_return,
                'volatility': tracker.volatility,
                'downside_vol': tracker.downside_volatility,
                'upside_vol': tracker.upside_volatility,
                'new_state_rate': tracker.new_state_rate,
                'sortino_ratio': self.compute_sortino_ratio(agent_name),
                'info_premium': self.compute_information_premium(agent_name),
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
        is_new_state: bool = False
    ) -> float:
        """
        计算投资组合调整后的奖励
        
        核心创新：用金融定价替代简单的奖励
        """
        # 1. 更新追踪器
        self.portfolio_valuator.update(agent_name, base_reward, is_new_state)
        
        # 2. 获取当前 Shapley 值
        shapley = self.cached_shapley_values.get(agent_name, 1.0 / self.agent_num)
        
        # 3. 计算投资组合估值
        remaining_ratio = self._get_remaining_steps_ratio()
        valuation, breakdown = self.portfolio_valuator.compute_valuation(
            agent_name, shapley, remaining_ratio
        )
        
        # 4. 调整奖励：基础奖励 * 估值倍数
        adjusted_reward = base_reward * valuation
        
        # 日志（每100步记录一次）
        if self.learn_step_count % 100 == 0:
            logger.debug(f"[P-SHAQ Valuation] Agent {agent_name}: "
                        f"base={base_reward:.2f}, valuation={valuation:.3f}, "
                        f"adjusted={adjusted_reward:.2f}, "
                        f"sortino={breakdown['sortino_ratio']:.2f}, "
                        f"info_premium={breakdown['info_premium']:.2f}")
        
        return adjusted_reward
    
    def choose_action(self, agent_name: str, state: WebState, action_list: List[WebAction]) -> WebAction:
        """选择动作（与 SHAQv2 类似，但使用投资组合权重调整探索率）"""
        if len(action_list) == 0:
            return RestartAction()
        
        # 获取投资组合权重
        weights = self.portfolio_valuator.get_portfolio_weights(
            self.cached_shapley_values,
            self._get_remaining_steps_ratio()
        )
        agent_weight = weights.get(agent_name, 1.0 / self.agent_num)
        
        # 探索率：权重高的 Agent 探索更多
        progress = 1.0 - self._get_remaining_steps_ratio()
        base_epsilon = self.max_random - (self.max_random - self.min_random) * progress
        
        # 权重调整：高权重 Agent 更积极探索
        epsilon = base_epsilon * (0.8 + 0.4 * agent_weight * self.agent_num)
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
        """动作选择 - P-SHAQ 核心：使用投资组合权重调整探索率"""
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
        
        # P-SHAQ 核心：使用投资组合权重调整 ε-greedy
        time_diff = (datetime.now() - self.start_time).total_seconds()
        time_diff = min(time_diff, self.alive_time)
        
        base_epsilon = self.max_random - min(time_diff / self.alive_time * 2, 1.0) * (
            self.max_random - self.min_random
        )
        
        # 使用投资组合权重调整探索率
        weights = self.portfolio_valuator.get_portfolio_weights(
            self.cached_shapley_values,
            self._get_remaining_steps_ratio()
        )
        agent_weight = weights.get(agent_name, 1.0 / self.agent_num)
        
        # 高权重 Agent 更积极探索（金融启发：高估值股票更受关注）
        epsilon = base_epsilon * (0.8 + 0.4 * agent_weight * self.agent_num)
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
        """计算奖励 - 使用 P-SHAQ 投资组合调整"""
        # 基础奖励：使用三层奖励系统
        browser_logs, performance_logs = self.get_agent_logs(agent_name)
        
        base_reward, _ = self.reward_system.compute_three_tier_reward(
            web_state=web_state,
            action=self.prev_action_dict.get(agent_name),
            browser_logs=browser_logs,
            performance_logs=performance_logs,
            http_status=200,
            html=html,
            agent_name=agent_name
        )
        
        # 检查是否是新状态
        state_hash = self.dom_encoder.compute_structure_hash(html) if html else ""
        is_new_state = state_hash and hash(state_hash) not in self._seen_dom_signatures
        if is_new_state and state_hash:
            self._seen_dom_signatures.add(hash(state_hash))
        
        # P-SHAQ 核心：投资组合调整
        self.portfolio_valuator.update(agent_name, base_reward, is_new_state)
        
        # 计算调整后的奖励
        shapley = self.cached_shapley_values.get(agent_name, 1.0 / self.agent_num)
        remaining_ratio = self._get_remaining_steps_ratio()
        valuation, _ = self.portfolio_valuator.compute_valuation(agent_name, shapley, remaining_ratio)
        
        adjusted_reward = base_reward * valuation
        
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
