"""
SHAQ (SHapley Q-value) 多智能体强化学习算法实现

使用 Lovasz 扩展作为桥梁来高效计算 Shapley value，避免 Monte Carlo 采样的指数级计算复杂度。

核心思想：
1. Shapley value 可以通过 Lovasz 扩展的梯度来计算
2. 对于集合函数 v(S)，其 Lovasz 扩展 f(x) 在 x = (1/2, ..., 1/2) 处的梯度即为 Shapley value
3. 通过神经网络参数化联合 Q 函数，利用自动微分高效计算梯度

参考文献:
- Wang et al., "SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning"
- Lovasz, L. "Submodular functions and convexity"
"""

import math
import random
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import multi_agent.multi_agent_system
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from model.dense_net import DenseNet
from model.replay_buffer import ReplayBuffer
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name
from web_test.multi_agent_thread import logger


class ShapleyMixingNetwork(nn.Module):
    """
    用于计算联合 Q 值的混合网络，设计为支持 Lovasz 扩展的梯度计算。
    
    该网络接收每个智能体的局部 Q 值和一个"参与度"向量 w ∈ [0,1]^n，
    输出一个标量的联合 Q 值。当 w_i = 1 时表示智能体 i 完全参与，
    w_i = 0 表示不参与。
    
    Lovasz 扩展的关键性质：
    f(w) = E[v(S_w)] 其中 S_w 是根据 w 随机采样的子集
    ∂f/∂w_i |_{w=(1/2,...,1/2)} = φ_i (Shapley value)
    """
    
    def __init__(self, n_agents: int, embed_dim: int = 64):
        super(ShapleyMixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        
        # 超网络：生成混合权重
        self.hyper_w1 = nn.Sequential(
            nn.Linear(n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.hyper_b1 = nn.Sequential(
            nn.Linear(n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, agent_q_values: torch.Tensor, participation_weights: torch.Tensor) -> torch.Tensor:
        """
        计算联合 Q 值。
        
        Args:
            agent_q_values: 各智能体的 Q 值，形状 (batch_size, n_agents)
            participation_weights: 参与度权重，形状 (batch_size, n_agents)，值域 [0, 1]
        
        Returns:
            联合 Q 值，形状 (batch_size, 1)
        """
        batch_size = agent_q_values.shape[0]
        
        # 将参与度权重应用到 Q 值上
        weighted_q = agent_q_values * participation_weights  # (batch_size, n_agents)
        
        # 生成混合网络权重（确保非负以满足单调性约束）
        w1 = torch.abs(self.hyper_w1(participation_weights))  # (batch_size, n_agents * embed_dim)
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        
        b1 = self.hyper_b1(participation_weights)  # (batch_size, embed_dim)
        
        # 第一层：(batch_size, n_agents) @ (batch_size, n_agents, embed_dim) -> (batch_size, embed_dim)
        hidden = torch.bmm(weighted_q.unsqueeze(1), w1).squeeze(1) + b1
        hidden = F.elu(hidden)
        
        # 第二层
        w2 = torch.abs(self.hyper_w2(participation_weights))  # (batch_size, embed_dim)
        b2 = self.hyper_b2(participation_weights)  # (batch_size, 1)
        
        # (batch_size, embed_dim) * (batch_size, embed_dim) -> sum -> (batch_size, 1)
        q_tot = (hidden * w2).sum(dim=1, keepdim=True) + b2
        
        return q_tot


class LovaszShapleyCalculator:
    """
    使用 Lovasz 扩展计算 Shapley value 的工具类。
    
    Lovasz 扩展将离散的集合函数 v: 2^N -> R 扩展到连续空间 [0,1]^n。
    关键定理：对于任意集合函数 v，其 Lovasz 扩展 f 在 x = (1/2, ..., 1/2) 处
    的偏导数等于 Shapley value：
    
    φ_i = ∂f/∂x_i |_{x=(1/2,...,1/2)}
    
    这允许我们使用自动微分来高效计算 Shapley value，而不需要 Monte Carlo 采样。
    
    优化版本：使用批量计算减少循环开销。
    """
    
    def __init__(self, n_agents: int, device: torch.device):
        self.n_agents = n_agents
        self.device = device
        
    def compute_shapley_values(
        self, 
        mixing_network: ShapleyMixingNetwork,
        agent_q_values: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        使用 Lovasz 扩展的梯度计算 Shapley value（优化版本）。
        
        核心方法：在 w = (1/2, ..., 1/2) 处计算混合网络输出关于 w 的梯度。
        
        Args:
            mixing_network: 混合网络
            agent_q_values: 各智能体 Q 值，形状 (batch_size, n_agents)
            num_samples: 用于估计的采样次数
            
        Returns:
            Shapley values，形状 (batch_size, n_agents)
        """
        batch_size = agent_q_values.shape[0]
        agent_q_values = agent_q_values.detach()
        
        # 批量处理多个采样：扩展 batch 维度
        # (num_samples * batch_size, n_agents)
        expanded_q = agent_q_values.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, self.n_agents)
        
        # 创建参与度权重
        w = torch.full(
            (num_samples * batch_size, self.n_agents), 
            0.5, 
            device=self.device, 
            requires_grad=True
        )
        
        # 单次前向传播
        q_tot = mixing_network(expanded_q, w)
        
        # 反向传播计算梯度
        q_tot.sum().backward()
        
        # 重塑并平均
        shapley_values = w.grad.view(num_samples, batch_size, self.n_agents).mean(dim=0)
        
        return shapley_values
    
    def compute_shapley_values_multipoint(
        self,
        mixing_network: ShapleyMixingNetwork,
        agent_q_values: torch.Tensor,
        num_points: int = 5
    ) -> torch.Tensor:
        """
        使用多点采样改进 Shapley value 的估计（优化版本）。
        
        批量计算所有排列的边际贡献，减少循环开销。
        
        Args:
            mixing_network: 混合网络
            agent_q_values: 各智能体 Q 值
            num_points: 采样点数
            
        Returns:
            Shapley values
        """
        batch_size = agent_q_values.shape[0]
        agent_q_values = agent_q_values.detach()
        
        all_marginal_contributions = torch.zeros(
            batch_size, self.n_agents, device=self.device
        )
        
        # 预生成所有排列
        perms = [torch.randperm(self.n_agents, device=self.device) for _ in range(num_points)]
        
        # 预计算空集的 Q 值（所有排列共用）
        w_empty = torch.zeros(batch_size, self.n_agents, device=self.device)
        with torch.no_grad():
            q_empty = mixing_network(agent_q_values, w_empty)
        
        for perm in perms:
            # 批量构建所有中间状态的权重矩阵
            # 对于 n_agents 个智能体，需要 n_agents 个中间状态
            w_states = torch.zeros(self.n_agents, batch_size, self.n_agents, device=self.device)
            
            for idx in range(self.n_agents):
                if idx > 0:
                    w_states[idx] = w_states[idx - 1].clone()
                w_states[idx, :, perm[idx]] = 1.0
            
            # 批量计算所有中间状态的 Q 值
            # 重塑为 (n_agents * batch_size, n_agents)
            w_flat = w_states.view(-1, self.n_agents)
            q_flat = agent_q_values.unsqueeze(0).expand(self.n_agents, -1, -1).reshape(-1, self.n_agents)
            
            with torch.no_grad():
                q_values = mixing_network(q_flat, w_flat)
            
            # 重塑为 (n_agents, batch_size, 1)
            q_values = q_values.view(self.n_agents, batch_size, 1)
            
            # 计算边际贡献
            q_prev = torch.cat([q_empty.unsqueeze(0), q_values[:-1]], dim=0)
            marginals = (q_values - q_prev).squeeze(-1)  # (n_agents, batch_size)
            
            # 按照排列顺序分配边际贡献
            for idx in range(self.n_agents):
                agent_i = perm[idx].item()
                all_marginal_contributions[:, agent_i] += marginals[idx]
        
        # 平均所有排列的边际贡献
        shapley_values = all_marginal_contributions / num_points
        
        return shapley_values
    
    def compute_shapley_values_fast(
        self,
        mixing_network: ShapleyMixingNetwork,
        agent_q_values: torch.Tensor
    ) -> torch.Tensor:
        """
        快速 Shapley value 近似：直接使用梯度计算。
        
        这是最快的方法，只需要一次前向传播和一次反向传播。
        基于 Lovasz 扩展在 w=0.5 处梯度等于 Shapley value 的定理。
        
        Args:
            mixing_network: 混合网络
            agent_q_values: 各智能体 Q 值
            
        Returns:
            Shapley values
        """
        batch_size = agent_q_values.shape[0]
        agent_q_values = agent_q_values.detach()
        
        # 在 w = 0.5 处计算梯度
        w = torch.full(
            (batch_size, self.n_agents), 
            0.5, 
            device=self.device, 
            requires_grad=True
        )
        
        q_tot = mixing_network(agent_q_values, w)
        q_tot.sum().backward()
        
        return w.grad.clone()


class SHAQ(multi_agent.multi_agent_system.MultiAgentSystem):
    """
    SHAQ (SHapley Q-value) 多智能体强化学习算法实现。
    
    使用 Lovasz 扩展高效计算 Shapley value，避免传统 Monte Carlo 方法的
    指数级采样复杂度。
    
    算法特点：
    1. 每个智能体维护独立的 Q 网络
    2. 使用 Shapley 混合网络计算联合 Q 值
    3. 通过 Lovasz 扩展的梯度属性高效分配信用
    4. 支持集中式训练、分散式执行 (CTDE) 范式
    """
    
    def __init__(self, params: Dict):
        super(SHAQ, self).__init__(params)
        self.params = params
        self.algo_type = params.get("algo_type", "shaq")
        self.reward_function = params.get("reward_function", "A")
        
        # 设备配置
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("SHAQ: Using GPU")
        else:
            self.device = torch.device("cpu")
            logger.info("SHAQ: Using CPU")
        
        # 超参数
        self.max_random = params.get("max_random", 0.9)
        self.min_random = params.get("min_random", 0.1)
        self.batch_size = params.get("batch_size", 32)
        self.gamma = params.get("gamma", 0.99)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.update_target_interval = params.get("update_target_interval", 100)
        self.update_network_interval = params.get("update_network_interval", 10)
        self.shapley_samples = params.get("shapley_samples", 10)  # Lovasz 采样次数
        self.shapley_method = params.get("shapley_method", "fast")  # fast/multipoint/gradient
        self.shapley_update_interval = params.get("shapley_update_interval", 5)  # 每N步计算一次Shapley
        self.use_shapley_exploration = params.get("use_shapley_exploration", True)  # 使用Shapley引导探索
        self.shapley_exploration_bonus = params.get("shapley_exploration_bonus", 1.5)  # 增强Shapley探索奖励系数 (原0.5)
        self.exploration_decay_rate = params.get("exploration_decay_rate", 0.995)  # 探索衰减率
        self.async_joint_learning = params.get("async_joint_learning", True)  # 异步联合学习
        self.alive_time = params.get("alive_time", 10800)

        # Transformer 用于状态/动作编码
        self.transformer = instantiate_class_by_module_and_class_name(
            params["transformer_module"], params["transformer_class"]
        )

        # 记录数据
        self.state_list: List[WebState] = []
        self.action_list: List[WebAction] = []
        self.state_list_agent: Dict[str, List[WebState]] = {}
        self.action_count = defaultdict(int)
        self.learn_step_count = 0
        self.start_time = datetime.now()
        
        # 各智能体的 Q 网络
        self.q_eval_agent: Dict[str, nn.Module] = {}
        self.q_target_agent: Dict[str, nn.Module] = {}
        self.agent_optimizer: Dict[str, optim.Optimizer] = {}
        
        # 初始化各智能体网络
        for i in range(self.agent_num):
            agent_name = str(i)
            
            # 创建评估网络和目标网络
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
            self.agent_optimizer[agent_name] = optim.Adam(
                q_eval.parameters(), lr=self.learning_rate
            )
            
            # 初始化智能体状态列表
            self.state_list_agent[agent_name] = []
        
        # Shapley 混合网络
        self.mixing_network = ShapleyMixingNetwork(
            n_agents=self.agent_num, embed_dim=64
        ).to(self.device)
        self.target_mixing_network = ShapleyMixingNetwork(
            n_agents=self.agent_num, embed_dim=64
            ).to(self.device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        self.mixing_optimizer = optim.Adam(
            self.mixing_network.parameters(), lr=self.learning_rate
        )
        
        # Lovasz Shapley 计算器
        self.shapley_calculator = LovaszShapleyCalculator(
            n_agents=self.agent_num, device=self.device
        )
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.replay_buffer_agent: Dict[str, ReplayBuffer] = {
            str(i): ReplayBuffer(capacity=500) for i in range(self.agent_num)
        }
        
        # 用于同步多智能体更新
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
        
        # 线程锁
        self.network_lock = threading.Lock()
        
        # 损失函数
        self.criterion = nn.MSELoss()

        # 奖励常量 - 优化版本：增强探索激励
        self.R_PENALTY = -99.0
        self.R_A_PENALTY = -5.0
        self.R_A_BASE_HIGH = 80.0      # 提高新状态奖励 (原 50.0)
        self.R_A_BASE_MIDDLE = 30.0    # 提高中等新颖奖励 (原 10.0)
        self.R_A_MIN_SIM_LINE = 0.6    # 降低阈值，更容易获得高奖励 (原 0.7)
        self.R_A_MIDDLE_SIM_LINE = 0.8 # 调整中间阈值 (原 0.85)
        
        # 新增：重复状态惩罚系数
        self.R_REPEAT_PENALTY = 0.3    # 重复访问的惩罚因子
        self.R_URL_DIVERSITY_BONUS = 20.0  # 新 URL 奖励
        
        # Shapley 相关的缓存和状态
        self.cached_shapley_values: Dict[str, float] = {str(i): 1.0 / self.agent_num for i in range(self.agent_num)}
        self.agent_contribution_history: Dict[str, List[float]] = {str(i): [] for i in range(self.agent_num)}
        self.shapley_update_counter = 0
        self.joint_learn_counter = 0
        
        # 探索奖励追踪
        self.agent_exploration_scores: Dict[str, float] = {str(i): 0.0 for i in range(self.agent_num)}
        self.new_states_by_agent: Dict[str, int] = {str(i): 0 for i in range(self.agent_num)}
        
        # 新增：URL 多样性追踪
        self.visited_urls: set = set()
        self.visited_url_paths: set = set()  # 只记录路径部分
        
        logger.info(f"SHAQ initialized with {self.agent_num} agents, "
                   f"Shapley method: {self.shapley_method}, "
                   f"async_learning: {self.async_joint_learning}, "
                   f"shapley_exploration: {self.use_shapley_exploration}")

    def get_tensor(self, action: WebAction, html: str, web_state: WebState) -> torch.Tensor:
        """将状态-动作对编码为张量。"""
        state_tensor = self.transformer.state_to_tensor(web_state, html)
        execution_time = self.action_dict.get(action, 0)
        action_tensor = self.transformer.action_to_tensor(web_state, action, execution_time)
        tensor = torch.cat((state_tensor, action_tensor))
        return tensor.float()
    
    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        """
        根据当前状态选择动作。
        
        使用 ε-greedy 策略，其中 ε 随时间衰减。
        """
        # 更新状态记录
        self.update_state_records(web_state, html, agent_name)

        actions = web_state.get_action_list()
        
        # 如果只有重启动作，直接返回
        if len(actions) == 1 and isinstance(actions[0], RestartAction):
            return actions[0]

        # 使用 Q 网络选择最优动作
        q_eval = self.q_eval_agent[agent_name]
        q_eval.eval()
        
        action_tensors = []
        for temp_action in actions:
            action_tensor = self.get_tensor(temp_action, html, web_state)
            action_tensors.append(action_tensor)
        
        with torch.no_grad():
            if isinstance(q_eval, DenseNet):
                output = q_eval(torch.stack(action_tensors).unsqueeze(1).to(self.device))
            else:
                output = q_eval(torch.stack(action_tensors).to(self.device))
        
        q_values = output.squeeze(-1).cpu().numpy()
        max_idx = q_values.argmax()
        max_val = q_values[max_idx]
        chosen_action = actions[max_idx]
        
        logger.info(f"[{agent_name}] SHAQ max Q-value: {max_val:.4f}")
        
        # Shapley 引导的 ε-greedy 探索
        end_time = datetime.now()
        time_diff = (end_time - self.start_time).total_seconds()
        time_diff = min(time_diff, self.alive_time)
        
        # 基础 ε 线性衰减
        base_epsilon = self.max_random - min(time_diff / self.alive_time * 2, 1.0) * (
            self.max_random - self.min_random
        )

        # Shapley 引导的探索调整 - 优化版：
        # 1. 低 Shapley value 的智能体应更激进探索
        # 2. 前期所有智能体都应更多探索
        # 3. 添加探索分数奖励
        if self.use_shapley_exploration:
            shapley_val = self.cached_shapley_values.get(agent_name, 1.0 / self.agent_num)
            avg_shapley = 1.0 / self.agent_num
            
            # 计算相对 Shapley 贡献
            relative_shapley = (shapley_val - avg_shapley) / (avg_shapley + 1e-6)
            
            # 如果该智能体的 Shapley value 低于平均，大幅增加探索
            # 使用更激进的调整因子
            if relative_shapley < 0:
                # 低贡献者：大幅增加探索
                shapley_factor = 1.0 + self.shapley_exploration_bonus * abs(relative_shapley) * 2.0
            else:
                # 高贡献者：适度减少探索，但保持最低探索率
                shapley_factor = max(0.5, 1.0 - self.shapley_exploration_bonus * relative_shapley * 0.5)
            
            # 结合探索分数：探索得分高的 agent 应继续探索
            exploration_score = self.agent_exploration_scores.get(agent_name, 0)
            if exploration_score > 0.3:  # 高探索得分
                shapley_factor *= 1.2
            
            epsilon = min(self.max_random, max(self.min_random, base_epsilon * shapley_factor))
        else:
            epsilon = base_epsilon
        
        if random.uniform(0, 1) < epsilon:
            # 探索：优先选择未执行过或执行次数少的动作
            unexplored = [a for a in actions if self.action_dict.get(a, 0) == 0]
            rarely_used = [a for a in actions if 0 < self.action_dict.get(a, 0) <= 2]
            
            if unexplored:
                chosen_action = random.choice(unexplored)
                # 记录探索新动作
                self.new_states_by_agent[agent_name] = self.new_states_by_agent.get(agent_name, 0) + 1
            elif rarely_used:
                # 次优选择：较少使用的动作
                chosen_action = random.choice(rarely_used)
            else:
                chosen_action = random.choice(actions)
        
        self.action_count[chosen_action] += 1
        return chosen_action

    def update_state_records(self, web_state: WebState, html: str, agent_name: str):
        """更新状态和动作记录，并触发学习。"""
        # 更新全局和局部状态列表
        if web_state not in self.state_list:
            self.state_list.append(web_state)
        if web_state not in self.state_list_agent[agent_name]:
            self.state_list_agent[agent_name].append(web_state)
        
        # 更新动作列表
        for action in web_state.get_action_list():
            if action not in self.action_list:
                self.action_list.append(action)
        
        # 检查是否有前一步的状态-动作记录
        if (self.prev_action_dict.get(agent_name) is None or 
            self.prev_state_dict.get(agent_name) is None or
            not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState)):
            return
        
        # 计算奖励并存储经验
        reward = self.get_reward(web_state, agent_name)
        tensor = self.get_tensor(
            self.prev_action_dict[agent_name],
            self.prev_html_dict[agent_name],
            self.prev_state_dict[agent_name]
        )
        tensor = tensor.unsqueeze(0)
        
        done = not isinstance(web_state, ActionSetWithExecutionTimesState)
        
        # 存储到智能体自己的回放缓冲区
        self.replay_buffer_agent[agent_name].push(
            tensor, tensor, reward, web_state, html, done
        )
        
        # 智能体级别的学习
        self.learn_agent(agent_name)
        
        # 检查是否所有智能体都完成了一步，进行联合学习
        self.try_joint_learning(web_state, html, agent_name)
    
    def get_reward(self, web_state: WebState, agent_name: str) -> float:
        """
        计算奖励 - 保持原始逻辑，性能优化版本。
        """
        if self.reward_function != "A":
            return 0.0
        
        if not isinstance(web_state, ActionSetWithExecutionTimesState):
            return self.R_A_PENALTY
        
        # 计算与已知状态的最大相似度
        max_sim = -1.0
        for temp_state in self.state_list[-200:]:
            if isinstance(temp_state, (OutOfDomainState, ActionExecuteFailedState, SameUrlState)):
                continue
            if web_state == temp_state:
                continue
            sim = web_state.similarity(temp_state)
            if sim > max_sim:
                max_sim = sim
        
        # 基于新颖度的奖励
        if max_sim < self.R_A_MIN_SIM_LINE:
            r_state = self.R_A_BASE_HIGH
        elif max_sim < self.R_A_MIDDLE_SIM_LINE:
            r_state = self.R_A_BASE_MIDDLE
        else:
            visited_time = self.state_dict.get(web_state, 0)
            if visited_time == 0:
                r_state = 5.0
            else:
                r_state = 5.0 * (self.R_REPEAT_PENALTY ** visited_time)
                r_state = max(0.1, r_state)
        
        # URL 多样性奖励
        r_url = 0.0
        current_url = getattr(web_state, 'url', None)
        if current_url:
            from urllib.parse import urlparse
            parsed = urlparse(current_url)
            url_path = parsed.path
            
            if url_path and url_path not in self.visited_url_paths:
                r_url = self.R_URL_DIVERSITY_BONUS
                self.visited_url_paths.add(url_path)
            
            if current_url not in self.visited_urls:
                self.visited_urls.add(current_url)
        
        # 动作执行频率奖励
        prev_state = self.prev_state_dict.get(agent_name)
        prev_action = self.prev_action_dict.get(agent_name)
        
        if not isinstance(prev_state, ActionSetWithExecutionTimesState) or prev_action is None:
            r_action = 0.0
        else:
            exec_count = self.action_count.get(prev_action, 0)
            if exec_count == 0:
                r_action = 5.0
            else:
                r_action = 5.0 * (0.5 ** exec_count)
                r_action = max(0.1, r_action)
        
        # 时间探索因子
        time_diff = (datetime.now() - self.start_time).total_seconds()
        progress = time_diff / self.alive_time
        exploration_factor = max(1.0, 1.5 - 0.5 * progress)
        
        # 最终奖励
        base_reward = r_state + r_action + r_url
        return base_reward * exploration_factor
    
    def try_joint_learning(self, web_state: WebState, html: str, agent_name: str):
        """
        尝试进行联合学习。
        
        支持两种模式：
        1. 同步模式：等待所有智能体都完成一步
        2. 异步模式：收集到足够经验后直接学习
        """
        with self.lock:
            if (not isinstance(self.prev_state_dict.get(agent_name), ActionSetWithExecutionTimesState) or
                not isinstance(self.current_state_dict.get(agent_name), ActionSetWithExecutionTimesState)):
                return
            
            # 标记当前智能体已完成
            self.finish_dict_agent[agent_name] = True
            self.prev_state_success_dict[agent_name] = self.prev_state_dict[agent_name]
            self.prev_action_success_dict[agent_name] = self.prev_action_dict[agent_name]
            self.current_state_success_dict[agent_name] = self.current_state_dict[agent_name]
            self.prev_html_success_dict[agent_name] = self.prev_html_dict[agent_name]
            
            # 计算该智能体的探索贡献
            state_novelty = self._compute_state_novelty(web_state)
            self.agent_exploration_scores[agent_name] = (
                0.9 * self.agent_exploration_scores.get(agent_name, 0) + 0.1 * state_novelty
            )
            
            if self.async_joint_learning:
                # 异步模式：立即存储该智能体的经验
                self._store_agent_experience(agent_name)
                self.joint_learn_counter += 1
                
                # 每隔一定步数进行联合学习
                if self.joint_learn_counter % self.agent_num == 0:
                    self.learn_joint_async()
                return
            
            # 同步模式：检查是否所有智能体都完成
            if not all(self.finish_dict_agent.values()):
                return
            
            # 重置完成标记
            for i in range(self.agent_num):
                self.finish_dict_agent[str(i)] = False
            
            # 收集所有智能体的数据
            self._store_joint_experience()
        
        # 联合学习（使用 Shapley 分配）
        self.learn_joint()
    
    def _compute_state_novelty(self, web_state: WebState) -> float:
        """计算状态新颖度"""
        if not isinstance(web_state, ActionSetWithExecutionTimesState):
            return 0.0
        
        max_sim = 0.0
        for temp_state in self.state_list[-100:]:  # 只看最近100个状态
            if isinstance(temp_state, (OutOfDomainState, ActionExecuteFailedState, SameUrlState)):
                continue
            if web_state == temp_state:
                continue
            sim = web_state.similarity(temp_state)
            if sim > max_sim:
                max_sim = sim
        
        return 1.0 - max_sim
    
    def _store_agent_experience(self, agent_name: str):
        """存储单个智能体的经验"""
        tensor = self.get_tensor(
            self.prev_action_success_dict[agent_name],
            self.prev_html_success_dict[agent_name],
            self.prev_state_success_dict[agent_name]
        )
        tensor = tensor.unsqueeze(0)
        
        # 计算该智能体的奖励（包含探索奖励）
        base_reward = self.get_reward(self.current_state_success_dict[agent_name], agent_name)
        exploration_bonus = self.agent_exploration_scores[agent_name] * self.shapley_exploration_bonus * 10
        reward = base_reward + exploration_bonus
        
        # 存储到共享的回放缓冲区
        self.replay_buffer_agent[agent_name].push(
            tensor, tensor, reward, 
            self.current_state_success_dict[agent_name],
            self.prev_html_success_dict[agent_name], False
        )
    
    def _store_joint_experience(self):
        """存储联合经验（同步模式）"""
        tensors = []
        next_states = []
        htmls = []
        
        total_reward = self.get_total_reward()
        
        for i in range(self.agent_num):
            an = str(i)
            tensor = self.get_tensor(
                self.prev_action_success_dict[an],
                self.prev_html_success_dict[an],
                self.prev_state_success_dict[an]
            )
            tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
            next_states.append(self.current_state_success_dict[an])
            htmls.append(self.prev_html_success_dict[an])
        
        self.replay_buffer.push(tensors, tensors, total_reward, next_states, htmls, False)
    
    def get_total_reward(self) -> float:
        """计算团队总奖励。"""
        total = 0.0
        for i in range(self.agent_num):
            agent_name = str(i)
            state = self.current_state_success_dict.get(agent_name)
            if state is not None:
                total += self.get_reward(state, agent_name)
        return total
    
    def learn_agent(self, agent_name: str):
        """单智能体学习。"""
        replay_buffer = self.replay_buffer_agent[agent_name]
        
        if len(replay_buffer.buffer) < self.batch_size:
            return
        
        self.learn_step_count += 1
        
        if self.learn_step_count % self.update_target_interval == 0:
            self.update_target_networks()
        
        if self.learn_step_count % self.update_network_interval != 0:
            return
        
        # 采样经验
        tensors, _, rewards, next_states, htmls, dones = replay_buffer.sample(self.batch_size)
        
        q_eval = self.q_eval_agent[agent_name]
        q_target = self.q_target_agent[agent_name]
        
        with self.network_lock:
            target_list = []
            
            for i in range(self.batch_size):
                tensor = tensors[i]
                reward = rewards[i]
                next_state = next_states[i]
                html = htmls[i]
                done = dones[i]
                
                # 计算下一状态的最大 Q 值
                next_q_value = 0.0
                if isinstance(next_state, ActionSetWithExecutionTimesState) and not done:
                    next_actions = next_state.get_action_list()
                    if next_actions:
                        next_tensors = [
                            self.get_tensor(a, html, next_state) for a in next_actions
                        ]
                        with torch.no_grad():
                            if isinstance(q_target, DenseNet):
                                next_q = q_target(
                                    torch.stack(next_tensors).unsqueeze(1).to(self.device)
                                )
                            else:
                                next_q = q_target(
                                    torch.stack(next_tensors).to(self.device)
                                )
                        next_q_value = next_q.max().item()
                
                # 计算目标 Q 值
                target_q = reward + self.gamma * next_q_value * (1 - int(done))
                target_list.append(target_q)
            
            # 训练
            q_eval.train()
            if isinstance(q_eval, DenseNet):
                input_tensor = torch.stack(tensors).to(self.device)
            else:
                input_tensor = torch.stack(tensors).squeeze(1).to(self.device)
            
            q_predicts = q_eval(input_tensor)
            q_targets = torch.tensor(target_list, device=self.device).unsqueeze(-1)
            
            loss = self.criterion(q_predicts, q_targets)
            
            self.agent_optimizer[agent_name].zero_grad()
            loss.backward()
            self.agent_optimizer[agent_name].step()
            
            logger.debug(f"[{agent_name}] SHAQ agent loss: {loss.item():.4f}")
    
    def learn_joint_async(self):
        """
        异步联合学习：从各智能体的缓冲区采样并使用 Shapley 信用分配。
        
        这是优化版本，不需要等待所有智能体同步。
        """
        self.shapley_update_counter += 1
        
        # 检查是否有足够的经验
        min_buffer_size = min(len(self.replay_buffer_agent[str(i)].buffer) for i in range(self.agent_num))
        if min_buffer_size < self.batch_size // self.agent_num:
            return
        
        with self.network_lock:
            # 从各智能体的缓冲区采样
            sample_size = self.batch_size // self.agent_num
            agent_q_values_list = []
            agent_rewards_list = []
            
            for i in range(self.agent_num):
                agent_name = str(i)
                buffer = self.replay_buffer_agent[agent_name]
                
                if len(buffer.buffer) < sample_size:
                    continue
                    
                tensors, _, rewards, _, _, _ = buffer.sample(sample_size)
                
                q_eval = self.q_eval_agent[agent_name]
                q_eval.eval()
                
                with torch.no_grad():
                    if isinstance(q_eval, DenseNet):
                        q_vals = q_eval(torch.stack(tensors).to(self.device))
                    else:
                        q_vals = q_eval(torch.stack(tensors).squeeze(1).to(self.device))
                
                agent_q_values_list.append(q_vals.squeeze(-1))
                agent_rewards_list.append(torch.tensor(rewards, device=self.device))
            
            if len(agent_q_values_list) < self.agent_num:
                return
            
            # 构建 Q 值矩阵 (sample_size, n_agents)
            agent_q_values = torch.stack(agent_q_values_list, dim=1)
            agent_rewards = torch.stack(agent_rewards_list, dim=1)
            
            # 只在间隔步数时计算 Shapley（减少开销）
            if self.shapley_update_counter % self.shapley_update_interval == 0:
                if self.shapley_method == "fast":
                    shapley_values = self.shapley_calculator.compute_shapley_values_fast(
                        self.mixing_network,
                        agent_q_values.detach()
                    )
                else:
                    shapley_values = self.shapley_calculator.compute_shapley_values(
                        self.mixing_network,
                        agent_q_values.detach(),
                        num_samples=max(1, self.shapley_samples // 2)
                    )
                
                # 更新缓存的 Shapley values
                mean_shapley = shapley_values.mean(dim=0)
                for i in range(self.agent_num):
                    self.cached_shapley_values[str(i)] = mean_shapley[i].item()
                
                # 记录贡献历史
                for i in range(self.agent_num):
                    self.agent_contribution_history[str(i)].append(mean_shapley[i].item())
                    if len(self.agent_contribution_history[str(i)]) > 100:
                        self.agent_contribution_history[str(i)] = self.agent_contribution_history[str(i)][-100:]
            
            # 使用缓存的 Shapley values 计算加权奖励
            shapley_weights = torch.tensor(
                [self.cached_shapley_values[str(i)] for i in range(self.agent_num)],
                device=self.device
            )
            shapley_weights = F.softmax(shapley_weights, dim=0)  # 归一化
            
            # 根据 Shapley 分配团队奖励
            total_reward = agent_rewards.sum(dim=1, keepdim=True)
            shapley_rewards = total_reward * shapley_weights.unsqueeze(0)
            
            # 计算每个智能体的目标 Q 值
            target_q_values = shapley_rewards + self.gamma * agent_q_values.detach()
            
            # 更新各智能体的 Q 网络
            total_loss = 0.0
            for i in range(self.agent_num):
                agent_name = str(i)
                q_eval = self.q_eval_agent[agent_name]
                q_eval.train()
                
                # 重新计算 Q 值（需要梯度）
                buffer = self.replay_buffer_agent[agent_name]
                tensors, _, _, _, _, _ = buffer.sample(sample_size)
                
                if isinstance(q_eval, DenseNet):
                    q_pred = q_eval(torch.stack(tensors).to(self.device))
                else:
                    q_pred = q_eval(torch.stack(tensors).squeeze(1).to(self.device))
                
                loss = self.criterion(q_pred.squeeze(-1), target_q_values[:, i].detach())
                
                self.agent_optimizer[agent_name].zero_grad()
                loss.backward()
                self.agent_optimizer[agent_name].step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / self.agent_num
            
            if self.shapley_update_counter % 10 == 0:
                shapley_str = ", ".join([f"A{i}:{self.cached_shapley_values[str(i)]:.3f}" 
                                        for i in range(self.agent_num)])
                logger.info(f"SHAQ async loss: {avg_loss:.4f}, Shapley: [{shapley_str}]")
    
    def learn_joint(self):
        """
        同步联合学习：使用 Lovasz 扩展计算 Shapley value 进行信用分配。
        
        这是 SHAQ 的核心：通过 Lovasz 扩展的梯度属性高效计算每个智能体
        对团队收益的边际贡献（即 Shapley value）。
        """
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        self.shapley_update_counter += 1
        
        # 采样联合经验
        tensors_batch, _, rewards_batch, next_states_batch, htmls_batch, dones_batch = \
            self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer.buffer)))
        
        actual_batch_size = len(tensors_batch)
        
        with self.network_lock:
            # 批量收集所有智能体的 Q 值
            agent_q_values = []
            next_agent_q_values = []
            
            for batch_idx in range(actual_batch_size):
                tensors = tensors_batch[batch_idx]
                next_states = next_states_batch[batch_idx]
                htmls = htmls_batch[batch_idx]
                
                current_qs = []
                next_qs = []
                
                for agent_idx in range(self.agent_num):
                    agent_name = str(agent_idx)
                    q_eval = self.q_eval_agent[agent_name]
                    q_target = self.q_target_agent[agent_name]
                    
                    tensor = tensors[agent_idx]
                    if isinstance(q_eval, DenseNet):
                        q_val = q_eval(tensor.unsqueeze(0).to(self.device))
                    else:
                        q_val = q_eval(tensor.to(self.device))
                    current_qs.append(q_val.squeeze())
                    
                    next_state = next_states[agent_idx]
                    html = htmls[agent_idx]
                    next_q_val = torch.tensor(0.0, device=self.device)
                    
                    if isinstance(next_state, ActionSetWithExecutionTimesState):
                        next_actions = next_state.get_action_list()
                        if next_actions:
                            next_tensors = [self.get_tensor(a, html, next_state) for a in next_actions[:10]]
                            with torch.no_grad():
                                if isinstance(q_target, DenseNet):
                                    next_q = q_target(torch.stack(next_tensors).unsqueeze(1).to(self.device))
                                else:
                                    next_q = q_target(torch.stack(next_tensors).to(self.device))
                            next_q_val = next_q.max()
                    
                    next_qs.append(next_q_val)
                
                agent_q_values.append(torch.stack(current_qs))
                next_agent_q_values.append(torch.stack(next_qs))
            
            agent_q_values = torch.stack(agent_q_values)
            next_agent_q_values = torch.stack(next_agent_q_values).detach()
            
            # 只在间隔时计算 Shapley values（减少开销）
            if self.shapley_update_counter % self.shapley_update_interval == 0:
                if self.shapley_method == "fast":
                    shapley_values = self.shapley_calculator.compute_shapley_values_fast(
                        self.mixing_network, agent_q_values.detach()
                    )
                elif self.shapley_method == "gradient":
                    shapley_values = self.shapley_calculator.compute_shapley_values(
                        self.mixing_network, agent_q_values.detach(), num_samples=self.shapley_samples
                    )
                else:
                    shapley_values = self.shapley_calculator.compute_shapley_values_multipoint(
                        self.mixing_network, agent_q_values.detach(), num_points=self.shapley_samples
                    )
                
                # 更新缓存
                mean_shapley = shapley_values.mean(dim=0)
                for i in range(self.agent_num):
                    self.cached_shapley_values[str(i)] = mean_shapley[i].item()
            
            # 计算联合 Q 值
            full_participation = torch.ones(actual_batch_size, self.agent_num, device=self.device)
            q_tot = self.mixing_network(agent_q_values, full_participation)
            
            with torch.no_grad():
                next_q_tot = self.target_mixing_network(next_agent_q_values, full_participation)
            
            rewards = torch.tensor(rewards_batch, device=self.device).unsqueeze(-1)
            dones = torch.tensor([float(d) for d in dones_batch], device=self.device).unsqueeze(-1)
            
            target_q_tot = rewards + self.gamma * next_q_tot * (1 - dones)
            
            # TD 损失
            td_loss = self.criterion(q_tot, target_q_tot.detach())
            
            # 更新网络
            self.mixing_optimizer.zero_grad()
            for agent_name in self.agent_optimizer:
                self.agent_optimizer[agent_name].zero_grad()
            
            td_loss.backward()
            
            self.mixing_optimizer.step()
            for agent_name in self.agent_optimizer:
                self.agent_optimizer[agent_name].step()
            
            if self.shapley_update_counter % 10 == 0:
                shapley_str = ", ".join([f"A{i}:{self.cached_shapley_values[str(i)]:.3f}" 
                                        for i in range(self.agent_num)])
                logger.info(f"SHAQ joint loss: {td_loss.item():.4f}, Shapley: [{shapley_str}]")
    
    def update_target_networks(self):
        """更新所有目标网络。"""
        with self.network_lock:
            for i in range(self.agent_num):
                agent_name = str(i)
                self.q_target_agent[agent_name].load_state_dict(
                    self.q_eval_agent[agent_name].state_dict()
                )
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        logger.info("SHAQ target networks updated")
