# -*- coding: utf-8 -*-
"""
Multi-Agent Reinforcement Learning Algorithm Benchmark Tool

Supported algorithms:
- SHAQ (Shapley Q-value with Lovasz Extension)
- Marg (Marginal Contribution Q-learning)
- MargD (Deep Marginal Contribution)
- IQL (Independent Q-Learning)
- QMIX, VDN, QTRAN (via MargD config)

Metrics:
1. State Coverage
2. Action Coverage
3. Time per Step
4. Cumulative Reward
5. State Novelty
6. URL Coverage
7. Memory Usage

Usage:
    python benchmark.py --profile github-marl-3h-shaq-5agent --duration 300
    python benchmark.py --compare shaq,marg,qmix --duration 600
"""

import argparse
import io
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import psutil

# ============== Fix Windows encoding issues ==============
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import yaml


def set_random_seed(seed: int):
    """
    设置所有随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 尝试设置 PyTorch 种子（如果可用）
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    logger.info(f"Random seed set: {seed}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    # 基本信息
    algorithm: str = ""
    profile: str = ""
    agent_num: int = 0
    duration_seconds: float = 0.0
    effective_duration_seconds: float = 0.0  # 有效测试时间（排除异常卡顿）
    
    # 覆盖率指标
    total_states_discovered: int = 0
    total_actions_discovered: int = 0
    total_urls_visited: int = 0
    unique_states: int = 0
    unique_actions: int = 0
    unique_urls: int = 0
    
    # 时间性能
    total_steps: int = 0
    avg_decision_time_ms: float = 0.0       # 平均算法决策时间（神经网络推理）
    avg_step_duration_ms: float = 0.0       # 平均每步端到端时间（总时间/步数）
    effective_step_duration_ms: float = 0.0  # 有效每步时间（排除异常后）
    max_decision_time_ms: float = 0.0
    min_decision_time_ms: float = 0.0
    steps_per_second: float = 0.0
    effective_steps_per_second: float = 0.0  # 有效吞吐量（排除异常后）
    anomaly_time_seconds: float = 0.0        # 异常卡顿总时间
    anomaly_count: int = 0                   # 异常卡顿次数
    
    # 学习性能
    # 【修复】区分原始奖励和标准化指标
    # - total_reward: Agent 内部优化的原始奖励（不同算法可能不可比）
    # - standardized_novelty_reward: 基于状态新颖度的标准化奖励（用于横向对比）
    total_reward: float = 0.0              # 原始奖励（Agent 训练目标）
    total_standardized_reward: float = 0.0 # 标准化奖励（基于新颖度，用于对比）
    avg_reward_per_step: float = 0.0
    avg_standardized_reward_per_step: float = 0.0
    learning_updates: int = 0
    avg_loss: float = 0.0
    
    # 原始奖励曲线（用于绘图）
    reward_curve: List[float] = field(default_factory=list)
    standardized_reward_curve: List[float] = field(default_factory=list)
    
    # 状态新颖度（越高越好）
    avg_state_novelty: float = 0.0
    novelty_over_time: List[float] = field(default_factory=list)
    state_discovery_curve: List[int] = field(default_factory=list)  # 状态发现曲线
    
    # 资源使用
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # 时间序列数据（用于绘图）
    states_over_time: List[int] = field(default_factory=list)
    actions_over_time: List[int] = field(default_factory=list)
    rewards_over_time: List[float] = field(default_factory=list)
    step_times_over_time: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    # === 新增：实际网页测试指标 ===
    # URL 路径覆盖
    unique_url_paths: int = 0           # 唯一 URL 路径数 (不含参数)
    url_depth_max: int = 0              # 最大 URL 深度
    url_depth_avg: float = 0.0          # 平均 URL 深度
    
    # 动作类型分布
    click_actions: int = 0              # 点击动作数
    input_actions: int = 0              # 输入动作数
    select_actions: int = 0             # 选择动作数
    
    # 错误和异常发现
    js_errors_found: int = 0            # JavaScript 错误数
    page_errors_found: int = 0          # 页面错误数 (404, 500等)
    action_failures: int = 0            # 动作执行失败数
    out_of_domain_count: int = 0        # 跳出域名次数
    same_url_stuck_count: int = 0       # URL 卡住次数
    
    # 探索效率
    new_state_rate: float = 0.0         # 新状态发现率 (新状态数/总步数)
    exploration_efficiency: float = 0.0 # 探索效率 (唯一URL/总步数)
    action_diversity: float = 0.0       # 动作多样性 (唯一动作/总动作)
    
    # 【新增】效率比指标（ASE 2024 + ARES 思路）
    # 公式：Efficiency = Unique_API_Endpoints / Total_Steps
    # 意义：证明 SHAQ 是靠"精准打击"赢的，而不是"暴力破解（随机乱点）"
    api_efficiency: float = 0.0         # API 效率 = API端点数 / 步数
    state_efficiency: float = 0.0       # 状态效率 = 状态数 / 步数  
    url_efficiency: float = 0.0         # URL 效率 = URL数 / 步数
    reward_efficiency: float = 0.0      # 奖励效率 = 总奖励 / 步数（真正的效率指标）
    
    # 【ASE 论文核心】Bug 分析指标
    total_bugs_detected: int = 0        # 检测到的 Bug 总数
    combination_bugs_count: int = 0     # 组合 Bug 数量
    combination_bug_ratio: float = 0.0  # 组合 Bug 占比
    avg_interaction_index: float = 0.0  # 平均交互指数
    high_confidence_localizations: int = 0  # 高置信度定位数
    localization_confidence: float = 0.0    # 平均定位置信度
    bug_analysis_raw: Dict = field(default_factory=dict)  # 原始 Bug 分析数据
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def summary(self) -> str:
        """生成性能摘要"""
        # 异常信息
        anomaly_info = ""
        if self.anomaly_count > 0:
            anomaly_info = f"\n║   ⚠ 检测到 {self.anomaly_count} 次异常卡顿，共 {self.anomaly_time_seconds:.1f} 秒"
        
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    性能测试报告 - {self.algorithm}
╠══════════════════════════════════════════════════════════════════╣
║ 配置: {self.profile}
║ 智能体数量: {self.agent_num}
║ 测试时长: {self.duration_seconds:.1f} 秒 (有效: {self.effective_duration_seconds:.1f} 秒){anomaly_info}
╠══════════════════════════════════════════════════════════════════╣
║ 【覆盖率指标】
║   - 发现状态数: {self.unique_states} (总访问: {self.total_states_discovered})
║   - 发现动作数: {self.unique_actions} (总执行: {self.total_actions_discovered})
║   - 访问 URL 数: {self.unique_urls} (总访问: {self.total_urls_visited})
║   - URL 路径数: {self.unique_url_paths} (最大深度: {self.url_depth_max})
╠══════════════════════════════════════════════════════════════════╣
║ 【时间性能】
║   - 总步数: {self.total_steps}
║   - 平均每步时间: {self.avg_step_duration_ms:.0f} ms (有效: {self.effective_step_duration_ms:.0f} ms)
║   - 算法决策时间: {self.avg_decision_time_ms:.2f} ms (最大: {self.max_decision_time_ms:.2f} ms)
║   - 每秒步数: {self.steps_per_second:.3f} (有效: {self.effective_steps_per_second:.3f})
╠══════════════════════════════════════════════════════════════════╣
║ 【学习性能】
║   - 累计奖励: {self.total_reward:.2f}
║   - 平均奖励/步: {self.avg_reward_per_step:.4f}
║   - 学习更新次数: {self.learning_updates}
║   - 平均损失: {self.avg_loss:.4f}
╠══════════════════════════════════════════════════════════════════╣
║ 【探索效率】
║   - 平均状态新颖度: {self.avg_state_novelty:.4f}
║   - 新状态发现率: {self.new_state_rate:.4f}
║   - 探索效率: {self.exploration_efficiency:.4f}
║   - 动作多样性: {self.action_diversity:.4f}
║   - 状态发现速率: {self.unique_states / max(self.effective_duration_seconds, 1) * 60:.2f} 个/分钟
╠══════════════════════════════════════════════════════════════════╣
║ 【效率比指标】（ASE 2024 / ARES 核心指标）
║   - 状态效率: {self.state_efficiency:.4f} (状态数/步数)
║   - URL 效率: {self.url_efficiency:.4f} (URL数/步数)
║   - 奖励效率: {self.reward_efficiency:.4f} (奖励/步数)
║   说明: 值越高说明算法越"精准打击"，而非"暴力破解"
╠══════════════════════════════════════════════════════════════════╣
║ 【动作类型分布】
║   - 点击: {self.click_actions}  输入: {self.input_actions}  选择: {self.select_actions}
╠══════════════════════════════════════════════════════════════════╣
║ 【Bug 发现与定位】
║   - 检测 Bug 总数: {self.total_bugs_detected}
║   - 组合 Bug 数量: {self.combination_bugs_count} (占比: {self.combination_bug_ratio:.1%})
║   - 平均交互指数: {self.avg_interaction_index:.3f}
║   - 高置信定位数: {self.high_confidence_localizations}
║   - 平均定位置信度: {self.localization_confidence:.1%}
╠══════════════════════════════════════════════════════════════════╣
║ 【错误与异常】
║   - 动作失败: {self.action_failures}  跳出域名: {self.out_of_domain_count}
║   - URL 卡住: {self.same_url_stuck_count}  异常卡顿: {self.anomaly_count} 次
╠══════════════════════════════════════════════════════════════════╣
║ 【资源使用】
║   - 峰值内存: {self.peak_memory_mb:.1f} MB
║   - 平均内存: {self.avg_memory_mb:.1f} MB
║   - CPU 使用率: {self.cpu_usage_percent:.1f}%
╚══════════════════════════════════════════════════════════════════╝
"""


@dataclass
class AggregatedMetrics:
    """
    多次运行的聚合指标（mean ± std）
    用于论文报告中的统计显著性
    """
    algorithm: str = ""
    profile: str = ""
    num_runs: int = 0
    seeds: List[int] = field(default_factory=list)
    
    # 聚合指标（每个都是 (mean, std) 元组）
    unique_states: Tuple[float, float] = (0.0, 0.0)
    unique_actions: Tuple[float, float] = (0.0, 0.0)
    unique_urls: Tuple[float, float] = (0.0, 0.0)
    total_steps: Tuple[float, float] = (0.0, 0.0)
    effective_steps_per_second: Tuple[float, float] = (0.0, 0.0)
    avg_decision_time_ms: Tuple[float, float] = (0.0, 0.0)
    total_reward: Tuple[float, float] = (0.0, 0.0)
    avg_state_novelty: Tuple[float, float] = (0.0, 0.0)
    new_state_rate: Tuple[float, float] = (0.0, 0.0)
    anomaly_count: Tuple[float, float] = (0.0, 0.0)
    peak_memory_mb: Tuple[float, float] = (0.0, 0.0)
    
    # 原始数据（用于绘制置信区间）
    raw_metrics: List[PerformanceMetrics] = field(default_factory=list)
    
    @staticmethod
    def from_runs(runs: List[PerformanceMetrics], seeds: List[int]) -> 'AggregatedMetrics':
        """从多次运行结果创建聚合指标"""
        if not runs:
            return AggregatedMetrics()
        
        def calc_mean_std(values: List[float]) -> Tuple[float, float]:
            if not values:
                return (0.0, 0.0)
            mean = np.mean(values)
            std = np.std(values) if len(values) > 1 else 0.0
            return (float(mean), float(std))
        
        agg = AggregatedMetrics(
            algorithm=runs[0].algorithm,
            profile=runs[0].profile,
            num_runs=len(runs),
            seeds=seeds,
            raw_metrics=runs,
        )
        
        # 计算各指标的 mean ± std
        agg.unique_states = calc_mean_std([r.unique_states for r in runs])
        agg.unique_actions = calc_mean_std([r.unique_actions for r in runs])
        agg.unique_urls = calc_mean_std([r.unique_urls for r in runs])
        agg.total_steps = calc_mean_std([r.total_steps for r in runs])
        agg.effective_steps_per_second = calc_mean_std([r.effective_steps_per_second for r in runs])
        agg.avg_decision_time_ms = calc_mean_std([r.avg_decision_time_ms for r in runs])
        agg.total_reward = calc_mean_std([r.total_reward for r in runs])
        agg.avg_state_novelty = calc_mean_std([r.avg_state_novelty for r in runs])
        agg.new_state_rate = calc_mean_std([r.new_state_rate for r in runs])
        agg.anomaly_count = calc_mean_std([r.anomaly_count for r in runs])
        agg.peak_memory_mb = calc_mean_std([r.peak_memory_mb for r in runs])
        
        return agg
    
    def summary(self) -> str:
        """生成聚合报告（论文格式）"""
        def fmt(metric: Tuple[float, float], precision: int = 2) -> str:
            mean, std = metric
            if std == 0:
                return f"{mean:.{precision}f}"
            return f"{mean:.{precision}f} ± {std:.{precision}f}"
        
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║           聚合性能报告 - {self.algorithm} ({self.num_runs} 次运行)
╠══════════════════════════════════════════════════════════════════╣
║ 配置: {self.profile}
║ 随机种子: {self.seeds}
╠══════════════════════════════════════════════════════════════════╣
║ 【覆盖率指标】 (mean ± std)
║   - 状态覆盖: {fmt(self.unique_states, 1)} 个
║   - 动作覆盖: {fmt(self.unique_actions, 1)} 个
║   - URL 覆盖: {fmt(self.unique_urls, 1)} 个
╠══════════════════════════════════════════════════════════════════╣
║ 【时间性能】
║   - 总步数: {fmt(self.total_steps, 0)} 步
║   - 有效吞吐: {fmt(self.effective_steps_per_second, 3)} 步/秒
║   - 决策时间: {fmt(self.avg_decision_time_ms)} ms
╠══════════════════════════════════════════════════════════════════╣
║ 【学习性能】
║   - 累计奖励: {fmt(self.total_reward)}
║   - 平均新颖度: {fmt(self.avg_state_novelty, 4)}
║   - 新状态率: {fmt(self.new_state_rate, 4)}
╠══════════════════════════════════════════════════════════════════╣
║ 【稳定性】
║   - 异常卡顿: {fmt(self.anomaly_count, 1)} 次
║   - 峰值内存: {fmt(self.peak_memory_mb, 1)} MB
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def to_dict(self) -> Dict:
        """转为字典格式（用于 JSON 保存）"""
        return {
            'algorithm': self.algorithm,
            'profile': self.profile,
            'num_runs': self.num_runs,
            'seeds': self.seeds,
            'metrics': {
                'unique_states': {'mean': self.unique_states[0], 'std': self.unique_states[1]},
                'unique_actions': {'mean': self.unique_actions[0], 'std': self.unique_actions[1]},
                'unique_urls': {'mean': self.unique_urls[0], 'std': self.unique_urls[1]},
                'total_steps': {'mean': self.total_steps[0], 'std': self.total_steps[1]},
                'effective_steps_per_second': {'mean': self.effective_steps_per_second[0], 'std': self.effective_steps_per_second[1]},
                'avg_decision_time_ms': {'mean': self.avg_decision_time_ms[0], 'std': self.avg_decision_time_ms[1]},
                'total_reward': {'mean': self.total_reward[0], 'std': self.total_reward[1]},
                'avg_state_novelty': {'mean': self.avg_state_novelty[0], 'std': self.avg_state_novelty[1]},
                'new_state_rate': {'mean': self.new_state_rate[0], 'std': self.new_state_rate[1]},
                'anomaly_count': {'mean': self.anomaly_count[0], 'std': self.anomaly_count[1]},
                'peak_memory_mb': {'mean': self.peak_memory_mb[0], 'std': self.peak_memory_mb[1]},
            },
            'raw_runs': [m.to_dict() for m in self.raw_metrics]
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, algorithm: str, profile: str, agent_num: int):
        self.metrics = PerformanceMetrics(
            algorithm=algorithm,
            profile=profile,
            agent_num=agent_num
        )
        self.start_time = None
        self.step_times: List[float] = []
        self.rewards: List[float] = []
        self.losses: List[float] = []
        self.states_set = set()
        self.actions_set = set()
        self.urls_set = set()
        self.novelty_scores: List[float] = []
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.lock = threading.Lock()
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        self.record_interval = 10  # 每10秒记录一次时间序列数据
        self._last_record_time = 0
        
        # 新增：收集实际测试指标
        self.url_paths_set = set()        # URL 路径集合 (不含参数)
        self.url_depths: List[int] = []   # URL 深度列表
        self.action_types: Dict[str, int] = {'click': 0, 'input': 0, 'select': 0, 'other': 0}
        self.error_counts: Dict[str, int] = {'js': 0, 'page': 0, 'action_fail': 0, 'out_of_domain': 0, 'same_url': 0}
        
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._resource_monitor, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Performance monitor started: {self.metrics.algorithm}")
        
    def stop(self):
        """停止监控"""
        self._stop_monitor.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.metrics.duration_seconds = time.time() - self.start_time
        self._finalize_metrics()
        logger.info(f"Performance monitor stopped: {self.metrics.algorithm}")
        
    def _resource_monitor(self):
        """后台资源监控线程"""
        process = psutil.Process()
        while not self._stop_monitor.is_set():
            try:
                mem_info = process.memory_info()
                self.memory_samples.append(mem_info.rss / (1024 * 1024))
                self.cpu_samples.append(process.cpu_percent(interval=0.1))
            except Exception:
                pass
            time.sleep(1)
    
    def record_step(self, step_time: float, reward: float, state_hash: int, 
                    action_hash: int, url: str, novelty: float = 0.0,
                    action_type: str = 'other',
                    standardized_reward: float = None):
        """
        记录单步执行数据
        
        Args:
            step_time: 步骤执行时间
            reward: 原始奖励（Agent 内部优化目标，不同算法可能不可比）
            state_hash: 状态哈希
            action_hash: 动作哈希
            url: 当前 URL
            novelty: 状态新颖度
            action_type: 动作类型
            standardized_reward: 标准化奖励（基于新颖度，用于横向对比）
        """
        from urllib.parse import urlparse
        
        # 如果没有提供标准化奖励，使用基于新颖度的计算
        if standardized_reward is None:
            standardized_reward = novelty * 50.0  # 基于新颖度的简单标准化
        
        with self.lock:
            self.step_times.append(step_time * 1000)  # 转换为毫秒
            self.rewards.append(reward)
            self.states_set.add(state_hash)
            self.actions_set.add(action_hash)
            if url:
                self.urls_set.add(url)
                # 解析 URL 路径和深度
                try:
                    parsed = urlparse(url)
                    path = parsed.path.rstrip('/')
                    self.url_paths_set.add(f"{parsed.netloc}{path}")
                    depth = len([p for p in path.split('/') if p])
                    self.url_depths.append(depth)
                except:
                    pass
            self.novelty_scores.append(novelty)
            
            # 记录动作类型
            if action_type in self.action_types:
                self.action_types[action_type] += 1
            else:
                self.action_types['other'] += 1
            
            self.metrics.total_states_discovered += 1
            self.metrics.total_actions_discovered += 1
            self.metrics.total_urls_visited += 1
            
            # 【新增】记录原始奖励和标准化奖励曲线
            self.metrics.reward_curve.append(reward)
            self.metrics.standardized_reward_curve.append(standardized_reward)
            self.metrics.state_discovery_curve.append(len(self.states_set))
            
            # 定期记录时间序列数据
            current_time = time.time()
            if current_time - self._last_record_time >= self.record_interval:
                elapsed = current_time - self.start_time
                self.metrics.timestamps.append(elapsed)
                self.metrics.states_over_time.append(len(self.states_set))
                self.metrics.actions_over_time.append(len(self.actions_set))
                self.metrics.rewards_over_time.append(sum(self.rewards))
                if self.step_times:
                    self.metrics.step_times_over_time.append(
                        sum(self.step_times[-100:]) / min(len(self.step_times), 100)
                    )
                self._last_record_time = current_time
    
    def record_error(self, error_type: str):
        """记录错误/异常事件"""
        with self.lock:
            if error_type in self.error_counts:
                self.error_counts[error_type] += 1
    
    def record_learning_update(self, loss: float):
        """记录学习更新"""
        with self.lock:
            self.losses.append(loss)
            self.metrics.learning_updates += 1
    
    def _detect_anomalies(self) -> tuple:
        """
        检测时间序列中的异常卡顿
        
        通过分析 timestamps 数组中相邻时间点的间隔来识别异常。
        如果某个间隔超过中位数间隔的10倍，则认为是异常卡顿。
        
        Returns:
            (anomaly_total_time, anomaly_count): 异常总时间和异常次数
        """
        timestamps = self.metrics.timestamps
        if len(timestamps) < 3:
            return 0.0, 0
        
        # 计算所有时间间隔
        intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            intervals.append(interval)
        
        if not intervals:
            return 0.0, 0
        
        # 使用中位数作为正常间隔的参考（比平均值更稳健）
        sorted_intervals = sorted(intervals)
        median_interval = sorted_intervals[len(sorted_intervals) // 2]
        
        # 异常阈值：中位数的10倍，但至少60秒
        # （避免正常的页面加载延迟被误判为异常）
        anomaly_threshold = max(60.0, median_interval * 10)
        
        # 检测异常
        anomaly_total = 0.0
        anomaly_count = 0
        
        for interval in intervals:
            if interval > anomaly_threshold:
                # 异常时间 = 实际间隔 - 正常间隔（用中位数估计）
                anomaly_time = interval - median_interval
                anomaly_total += anomaly_time
                anomaly_count += 1
                logger.warning(f"Detected anomaly stall: {interval:.1f}s (normal ~{median_interval:.1f}s)")
        
        if anomaly_count > 0:
            logger.info(f"Detected {anomaly_count} anomaly stalls, total {anomaly_total:.1f}s")
        
        return anomaly_total, anomaly_count
    
    def _finalize_metrics(self):
        """计算最终指标"""
        with self.lock:
            # 覆盖率
            self.metrics.unique_states = len(self.states_set)
            self.metrics.unique_actions = len(self.actions_set)
            self.metrics.unique_urls = len(self.urls_set)
            
            # 检测异常卡顿（时间间隔超过正常值的10倍视为异常）
            anomaly_time, anomaly_count = self._detect_anomalies()
            self.metrics.anomaly_time_seconds = anomaly_time
            self.metrics.anomaly_count = anomaly_count
            self.metrics.effective_duration_seconds = max(1, self.metrics.duration_seconds - anomaly_time)
            
            # 时间性能
            self.metrics.total_steps = len(self.step_times)
            if self.step_times:
                # 决策时间（算法推理时间）
                self.metrics.avg_decision_time_ms = sum(self.step_times) / len(self.step_times)
                self.metrics.max_decision_time_ms = max(self.step_times)
                self.metrics.min_decision_time_ms = min(self.step_times)
            
            # 端到端时间（总时间/步数）
            if self.metrics.total_steps > 0:
                self.metrics.avg_step_duration_ms = (self.metrics.duration_seconds * 1000) / self.metrics.total_steps
                self.metrics.effective_step_duration_ms = (self.metrics.effective_duration_seconds * 1000) / self.metrics.total_steps
            
            # 吞吐量
            if self.metrics.duration_seconds > 0:
                self.metrics.steps_per_second = self.metrics.total_steps / self.metrics.duration_seconds
            if self.metrics.effective_duration_seconds > 0:
                self.metrics.effective_steps_per_second = self.metrics.total_steps / self.metrics.effective_duration_seconds
            
            # 学习性能
            # 【修复】分别计算原始奖励和标准化奖励
            self.metrics.total_reward = sum(self.rewards)
            if self.rewards:
                self.metrics.avg_reward_per_step = self.metrics.total_reward / len(self.rewards)
            
            # 计算标准化奖励（从 reward_curve 中提取 standardized 部分）
            if self.metrics.standardized_reward_curve:
                self.metrics.total_standardized_reward = sum(self.metrics.standardized_reward_curve)
                self.metrics.avg_standardized_reward_per_step = (
                    self.metrics.total_standardized_reward / len(self.metrics.standardized_reward_curve)
                )
            
            if self.losses:
                self.metrics.avg_loss = sum(self.losses) / len(self.losses)
            
            # 新颖度
            if self.novelty_scores:
                self.metrics.avg_state_novelty = sum(self.novelty_scores) / len(self.novelty_scores)
                # 按时间窗口计算新颖度变化
                window_size = max(1, len(self.novelty_scores) // 10)
                for i in range(0, len(self.novelty_scores), window_size):
                    window = self.novelty_scores[i:i+window_size]
                    self.metrics.novelty_over_time.append(sum(window) / len(window))
            
            # 资源使用
            if self.memory_samples:
                self.metrics.peak_memory_mb = max(self.memory_samples)
                self.metrics.avg_memory_mb = sum(self.memory_samples) / len(self.memory_samples)
            if self.cpu_samples:
                self.metrics.cpu_usage_percent = sum(self.cpu_samples) / len(self.cpu_samples)
            
            # === 新增：实际测试指标 ===
            # URL 路径覆盖
            self.metrics.unique_url_paths = len(self.url_paths_set)
            if self.url_depths:
                self.metrics.url_depth_max = max(self.url_depths)
                self.metrics.url_depth_avg = sum(self.url_depths) / len(self.url_depths)
            
            # 动作类型分布
            self.metrics.click_actions = self.action_types.get('click', 0)
            self.metrics.input_actions = self.action_types.get('input', 0)
            self.metrics.select_actions = self.action_types.get('select', 0)
            
            # 错误和异常
            self.metrics.js_errors_found = self.error_counts.get('js', 0)
            self.metrics.page_errors_found = self.error_counts.get('page', 0)
            self.metrics.action_failures = self.error_counts.get('action_fail', 0)
            self.metrics.out_of_domain_count = self.error_counts.get('out_of_domain', 0)
            self.metrics.same_url_stuck_count = self.error_counts.get('same_url', 0)
            
            # 探索效率计算
            if self.metrics.total_steps > 0:
                self.metrics.new_state_rate = self.metrics.unique_states / self.metrics.total_steps
                self.metrics.exploration_efficiency = self.metrics.unique_urls / self.metrics.total_steps
                self.metrics.action_diversity = self.metrics.unique_actions / self.metrics.total_steps
            
            # 【新增】效率比指标计算（ASE 2024 / ARES 核心指标）
            # 公式：Efficiency = Metric / Total_Steps
            # 意义：证明算法是靠"精准打击"赢的，而非"暴力破解"
            if self.metrics.total_steps > 0:
                self.metrics.state_efficiency = self.metrics.unique_states / self.metrics.total_steps
                self.metrics.url_efficiency = self.metrics.unique_urls / self.metrics.total_steps
                self.metrics.reward_efficiency = self.metrics.total_reward / self.metrics.total_steps


class BenchmarkRunner:
    """性能测试运行器"""
    
    # 配置名称别名映射（支持简写）
    PROFILE_ALIASES = {
        # 原有配置
        'shaq': 'github-marl-3h-shaq-5agent',
        'shaq-5': 'github-marl-3h-shaq-5agent',
        'shaq-quick': 'quick-test-shaq',
        'marg': 'github-marl-3h-marg-dql-5agent',
        'marg-dql': 'github-marl-3h-marg-dql-5agent',
        'marg-quick': 'quick-test-mac',
        'qtran': 'github-marl-3h-qtran-5agent',
        'qmix': 'github-marl-3h-qtran-5agent',
        'nndql': 'github-marl-3h-nndql-5agent',
        'nn': 'github-marl-3h-nn-5agent',
        'hybrid': 'github-marl-3h-hybrid-5agent',
        'shaq-qtran': 'github-marl-3h-hybrid-5agent',
        'shaqv2': 'github-marl-3h-shaqv2-5agent',
        'shaq-v2': 'github-marl-3h-shaqv2-5agent',
        'v2': 'github-marl-3h-shaqv2-5agent',
        # P-SHAQ: Portfolio-SHAQ with Financial Pricing
        'pshaq': 'github-marl-3h-pshaq-5agent',
        'p-shaq': 'github-marl-3h-pshaq-5agent',
        'portfolio-shaq': 'github-marl-3h-pshaq-5agent',
        # ASE 论文实验配置
        'iql': 'github-marl-3h-iql-5agent',
        'vdn': 'github-marl-3h-vdn-5agent',
        # 消融实验
        'shaqv2-no-icm': 'github-marl-3h-shaqv2-no-icm-5agent',
        'shaqv2-noicm': 'github-marl-3h-shaqv2-no-icm-5agent',
        'no-icm': 'github-marl-3h-shaqv2-no-icm-5agent',
        'shaqv2-no-role': 'github-marl-3h-shaqv2-no-role-5agent',
        'shaqv2-norole': 'github-marl-3h-shaqv2-no-role-5agent',
        'no-role': 'github-marl-3h-shaqv2-no-role-5agent',
        'shaqv2-base': 'github-marl-3h-shaqv2-base-5agent',
        'base': 'github-marl-3h-shaqv2-base-5agent',
        
        # 多网站泛化实验 (RQ5)
        'wikipedia': 'wikipedia-shaqv2-5agent',
        'wikipedia-shaqv2': 'wikipedia-shaqv2-5agent',
        'stackoverflow': 'stackoverflow-shaqv2-5agent',
        'stackoverflow-shaqv2': 'stackoverflow-shaqv2-5agent',
        'bing': 'bing-shaqv2-5agent',
        'bing-shaqv2': 'bing-shaqv2-5agent',
    }
    
    def __init__(self, config_path: str = "settings.yaml"):
        self.config_path = config_path
        self.results: Dict[str, PerformanceMetrics] = {}
        self.aggregated_results: Dict[str, AggregatedMetrics] = {}
        self._temp_dirs: List[str] = []  # 临时目录列表，用于清理
        
    def resolve_profile_name(self, profile: str) -> str:
        """解析配置名称（支持别名）"""
        # 先检查是否是别名
        if profile.lower() in self.PROFILE_ALIASES:
            resolved = self.PROFILE_ALIASES[profile.lower()]
            logger.info(f"Profile alias: {profile} -> {resolved}")
            return resolved
        return profile
        
    def load_config(self, profile: str) -> Dict:
        """加载配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 解析别名
        resolved_profile = self.resolve_profile_name(profile)
        
        if resolved_profile not in config.get('profiles', {}):
            available = list(config.get('profiles', {}).keys())
            aliases = list(self.PROFILE_ALIASES.keys())
            raise ValueError(
                f"Profile '{profile}' not found.\n"
                f"可用配置: {', '.join(available)}\n"
                f"可用别名: {', '.join(aliases)}"
            )
        
        return config['profiles'][resolved_profile]
    
    def get_algorithm_name(self, profile_config: Dict) -> str:
        """从配置中提取算法名称"""
        agent_module = profile_config.get('agent', {}).get('module', '')
        agent_class = profile_config.get('agent', {}).get('class', '')
        algo_type = profile_config.get('agent', {}).get('params', {}).get('algo_type', '')
        
        if 'pshaq' in agent_module.lower() or agent_class.lower() == 'pshaq':
            return 'P-SHAQ'
        elif 'shaq_v2' in agent_module.lower() or agent_class.lower() == 'shaqv2':
            return 'SHAQv2'
        elif 'hybrid' in agent_module.lower() or agent_class.lower() == 'shaqqqtranhybrid':
            return 'HYBRID'
        elif 'shaq' in agent_module.lower() or agent_class.lower() == 'shaq':
            return 'SHAQ'
        elif 'marg_d' in agent_module.lower() or agent_class.lower() == 'margd':
            if algo_type == 'qtran':
                return 'QTRAN'
            elif algo_type == 'qmix' or algo_type == 'qmix_d':
                return 'QMIX'
            elif algo_type == 'vdn':
                return 'VDN'
            elif algo_type == 'nndql':
                return 'NNDQL'
            else:
                return 'MargD'
        elif 'marg' in agent_module.lower() or agent_class.lower() == 'marg':
            return 'Marg-DQL' if 'dql' in str(algo_type).lower() else 'Marg-CQL'
        elif 'iql' in agent_module.lower() or agent_class.lower() == 'iql':
            return 'IQL'
        else:
            return agent_class or 'Unknown'
    
    def run_benchmark(self, profile: str, duration: int = 300, 
                      dry_run: bool = False, seed: Optional[int] = None,
                      use_incognito: bool = True) -> PerformanceMetrics:
        """
        运行单个配置的性能测试
        
        Args:
            profile: 配置名称（支持别名）
            duration: 测试时长（秒）
            dry_run: 是否干跑（不实际启动浏览器）
            seed: 随机种子（None 表示使用当前时间）
            use_incognito: 是否使用隐身模式（确保浏览器环境干净）
        """
        # 设置随机种子
        if seed is None:
            seed = int(time.time()) % 100000
        set_random_seed(seed)
        
        # 解析别名
        resolved_profile = self.resolve_profile_name(profile)
        logger.info(f"Starting benchmark: {resolved_profile}, duration: {duration}s, seed: {seed}, incognito: {use_incognito}")
        
        config = self.load_config(profile)
        algorithm = self.get_algorithm_name(config)
        agent_num = config.get('agent_num', 1)
        
        monitor = PerformanceMonitor(algorithm, resolved_profile, agent_num)
        
        if dry_run:
            # 模拟运行
            monitor.start()
            self._simulate_run(monitor, duration)
            monitor.stop()
        else:
            # 实际运行
            monitor.start()
            try:
                self._actual_run(resolved_profile, config, monitor, duration, use_incognito)
            except Exception as e:
                logger.error(f"Error during test: {e}")
                traceback.print_exc()
            finally:
                monitor.stop()
        
        self.results[resolved_profile] = monitor.metrics
        return monitor.metrics
    
    def run_multiple(self, profile: str, duration: int = 300, 
                     num_runs: int = 3, seeds: Optional[List[int]] = None,
                     use_incognito: bool = True, dry_run: bool = False) -> AggregatedMetrics:
        """
        多次运行同一配置的性能测试，并计算统计聚合结果
        
        用于论文报告：确保结果的统计显著性
        
        Args:
            profile: 配置名称（支持别名）
            duration: 每次测试时长（秒）
            num_runs: 运行次数（建议 3-5 次）
            seeds: 随机种子列表（None 表示自动生成）
            use_incognito: 是否使用隐身模式
            dry_run: 是否干跑
            
        Returns:
            AggregatedMetrics: 包含 mean ± std 的聚合结果
        """
        resolved_profile = self.resolve_profile_name(profile)
        
        # 生成随机种子
        if seeds is None:
            base_seed = int(time.time()) % 10000
            seeds = [base_seed + i * 1000 for i in range(num_runs)]
        elif len(seeds) < num_runs:
            # 补充种子
            seeds = seeds + [seeds[-1] + i * 1000 for i in range(1, num_runs - len(seeds) + 1)]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting multi-run test: {resolved_profile}")
        logger.info(f"Runs: {num_runs}, duration each: {duration}s")
        logger.info(f"Seeds: {seeds[:num_runs]}")
        logger.info(f"{'='*60}")
        
        runs: List[PerformanceMetrics] = []
        
        for i, seed in enumerate(seeds[:num_runs]):
            logger.info(f"\n--- Run {i+1}/{num_runs} (seed={seed}) ---")
            
            # 每次运行使用不同的结果键，避免覆盖
            run_key = f"{resolved_profile}_run{i+1}"
            
            try:
                metrics = self.run_benchmark(
                    profile=profile,
                    duration=duration,
                    dry_run=dry_run,
                    seed=seed,
                    use_incognito=use_incognito
                )
                runs.append(metrics)
                logger.info(f"Run {i+1} completed: steps={metrics.total_steps}, states={metrics.unique_states}")
                
                # 运行之间暂停，让系统资源恢复
                if i < num_runs - 1:
                    logger.info("Waiting 10s before next run...")
                    time.sleep(10)
                    
            except Exception as e:
                logger.error(f"Run {i+1} failed: {e}")
                traceback.print_exc()
        
        # 计算聚合统计
        if runs:
            aggregated = AggregatedMetrics.from_runs(runs, seeds[:len(runs)])
            self.aggregated_results[resolved_profile] = aggregated
            logger.info(f"\nMulti-run test completed, success {len(runs)}/{num_runs}")
            return aggregated
        else:
            logger.error("All runs failed")
            return AggregatedMetrics()
    
    def _simulate_run(self, monitor: PerformanceMonitor, duration: int):
        """模拟运行（用于测试监控系统）"""
        import random
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < duration:
            step_start = time.time()
            
            # 模拟处理时间
            time.sleep(random.uniform(0.05, 0.2))
            
            step_time = time.time() - step_start
            reward = random.uniform(-1, 10)
            state_hash = hash(f"state_{step % 100}")
            action_hash = hash(f"action_{step % 50}")
            url = f"https://example.com/page{step % 20}"
            novelty = max(0, 1 - (step % 100) / 100)
            
            monitor.record_step(step_time, reward, state_hash, action_hash, url, novelty)
            
            if step % 10 == 0:
                loss = random.uniform(0.01, 1.0)
                monitor.record_learning_update(loss)
            
            step += 1
        
        logger.info(f"Simulated run completed: {step} steps")
    
    def _actual_run(self, profile: str, config: Dict, monitor: PerformanceMonitor, 
                    duration: int, use_incognito: bool = True):
        """
        实际运行测试
        
        Args:
            profile: 配置名称
            config: 配置字典
            monitor: 性能监控器
            duration: 测试时长
            use_incognito: 是否使用隐身模式（确保浏览器环境干净）
        """
        import sys
        import importlib
        from selenium.webdriver.chrome.options import Options
        
        # 临时修改 sys.argv 以避免与 cli_options 的参数解析冲突
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], '--profile', profile]
        
        # 创建临时数据目录（确保每次测试都是干净的环境）
        temp_data_dir = None
        if use_incognito:
            temp_data_dir = tempfile.mkdtemp(prefix='benchmark_chrome_')
            self._temp_dirs.append(temp_data_dir)
            logger.info(f"Created temp browser data dir: {temp_data_dir}")
        
        try:
            # 完全清除相关模块缓存，然后重新导入
            # 这比 reload() 更可靠，确保所有模块都使用新的配置
            modules_to_remove = [
                'web_test.multi_agent_thread',
                'web_test.webtest_multi_agent',
                'multi_agent.multi_agent_system',
                'config',
                'config.settings',
                'config.cli_options',
            ]
            # 还需要删除 agent 实现模块的缓存
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith('multi_agent.impl.'):
                    modules_to_remove.append(mod_name)
            
            for mod_name in modules_to_remove:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
            
            # 重新导入模块（会使用新的 sys.argv）
            from config import settings
            from web_test.webtest_multi_agent import WebtestMultiAgent
            
            # settings 已经通过 cli_options 自动加载了正确的 profile
            
            # 动态修改 settings 对象的 alive_time 以匹配我们的 duration
            # 这确保测试在指定时间后停止
            original_alive_time = settings.alive_time
            settings.alive_time = duration
            if settings.agent and 'params' in settings.agent:
                settings.agent['params']['alive_time'] = duration
            logger.info(f"Overrode alive_time: {original_alive_time} -> {duration}s")
            
            # 创建 Chrome 选项（使用 settings 对象而不是 config 字典）
            chrome_options = Options()
            for arg in settings.browser_arguments:
                chrome_options.add_argument(arg)
            chrome_options.binary_location = settings.browser_path
            
            # 添加隐身模式和干净环境的参数
            if use_incognito:
                chrome_options.add_argument('--incognito')
                chrome_options.add_argument(f'--user-data-dir={temp_data_dir}')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-plugins')
                chrome_options.add_argument('--disable-application-cache')
                chrome_options.add_argument('--disable-cache')
                logger.info("Enabled incognito mode and clean browser env")
            
            # 创建测试实例
            webtest = WebtestMultiAgent(chrome_options)
            
            # 导入状态类型用于错误监控
            from state.impl.out_of_domain_state import OutOfDomainState
            from state.impl.same_url_state import SameUrlState
            from state.impl.action_execute_failed_state import ActionExecuteFailedState
            
            # 注入监控钩子
            original_get_action = webtest.multi_agent_system.get_action
            
            def monitored_get_action(web_state, html, agent_name, url, check_result):
                step_start = time.time()
                action = None
                action_success = False
                
                # 记录状态类型（用于统计错误）
                if isinstance(web_state, OutOfDomainState):
                    monitor.record_error('out_of_domain')
                elif isinstance(web_state, SameUrlState):
                    monitor.record_error('same_url')
                elif isinstance(web_state, ActionExecuteFailedState):
                    monitor.record_error('action_fail')
                
                try:
                    action = original_get_action(web_state, html, agent_name, url, check_result)
                    action_success = True
                except Exception as e:
                    logger.error(f"original_get_action error: {e}")
                    # 记录失败的步骤（使用默认值）
                    step_time = time.time() - step_start
                    monitor.record_step(
                        step_time=step_time,
                        reward=0.0,
                        state_hash=hash(str(web_state)),
                        action_hash=0,
                        url=url,
                        novelty=0.0,
                        action_type='error',
                        standardized_reward=0.0
                    )
                    raise
                
                step_time = time.time() - step_start
                
                # 使用默认值初始化，然后尝试计算更精确的值
                novelty = 0.0
                standardized_reward = 0.0
                reward = 0.0
                action_type = 'other'
                
                try:
                    # 计算新颖度（用于标准化奖励）
                    state_dict = webtest.multi_agent_system.state_dict
                    if hasattr(web_state, 'similarity') and len(state_dict) > 1:
                        max_sim = max(
                            (web_state.similarity(s) for s in state_dict if s != web_state),
                            default=0
                        )
                        novelty = 1 - max_sim
                    
                    # 计算标准化奖励（用于横向对比）
                    from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
                    R_A_BASE_HIGH = 50.0
                    R_A_BASE_MIDDLE = 10.0
                    R_A_MIN_SIM_LINE = 0.7
                    R_A_MIDDLE_SIM_LINE = 0.85
                    
                    if isinstance(web_state, ActionSetWithExecutionTimesState):
                        max_sim = -1.0
                        for temp_state in state_dict.keys():
                            if web_state == temp_state:
                                continue
                            if hasattr(web_state, 'similarity'):
                                sim = web_state.similarity(temp_state)
                                if sim > max_sim:
                                    max_sim = sim
                        
                        if max_sim < R_A_MIN_SIM_LINE:
                            standardized_reward = R_A_BASE_HIGH
                        elif max_sim < R_A_MIDDLE_SIM_LINE:
                            standardized_reward = R_A_BASE_MIDDLE
                        else:
                            visited_time = state_dict.get(web_state, 0)
                            standardized_reward = 2.0 / max(1, visited_time)
                except Exception as e:
                    logger.debug(f"Error calculating novelty/standardized reward: {e}")
                
                try:
                    # 获取原始奖励（Agent 训练目标）
                    if hasattr(webtest.multi_agent_system, 'get_reward'):
                        import inspect
                        sig = inspect.signature(webtest.multi_agent_system.get_reward)
                        param_count = len(sig.parameters)
                        if param_count >= 3:
                            reward = webtest.multi_agent_system.get_reward(web_state, html, agent_name)
                        else:
                            reward = webtest.multi_agent_system.get_reward(web_state, agent_name)
                    else:
                        reward = standardized_reward
                except Exception as e:
                    reward = standardized_reward
                    logger.debug(f"Error getting raw reward: {e}")
                
                try:
                    # 识别动作类型
                    action_class = action.__class__.__name__ if action else 'None'
                    if 'Click' in action_class:
                        action_type = 'click'
                    elif 'Input' in action_class:
                        action_type = 'input'
                    elif 'Select' in action_class:
                        action_type = 'select'
                except Exception:
                    pass
                
                # 始终记录步骤（即使某些计算失败）
                monitor.record_step(
                    step_time=step_time,
                    reward=reward,
                    state_hash=hash(str(web_state)),
                    action_hash=hash(str(action)) if action else 0,
                    url=url,
                    novelty=novelty,
                    action_type=action_type,
                    standardized_reward=standardized_reward
                )
                
                return action
            
            webtest.multi_agent_system.get_action = monitored_get_action
            
            # 启动测试
            webtest.start()
            
            # 等待指定时间（强制停止，不受alive_time影响）
            start_time = time.time()
            elapsed = 0
            while elapsed < duration:
                remaining = duration - elapsed
                sleep_time = min(remaining, 60)  # 每60秒检查一次
                time.sleep(sleep_time)
                elapsed = time.time() - start_time
                
                # 检查是否超时
                if elapsed >= duration:
                    logger.info(f"Reached duration limit ({duration}s), forcing stop")
                    break
            
            # 强制停止测试
            logger.info(f"Stopping test, actual runtime: {elapsed:.1f}s")
            webtest.stop()
            webtest.join(timeout=30)
            
            # 如果线程仍在运行，强制终止所有浏览器进程
            if webtest.is_alive():
                logger.warning("Test did not stop normally, force killing browser...")
                import subprocess
                import platform
                try:
                    if platform.system() == 'Windows':
                        # Windows: 使用 taskkill 强制终止 Chrome 和 chromedriver
                        subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe'], 
                                       capture_output=True, timeout=10)
                        subprocess.run(['taskkill', '/F', '/IM', 'chromedriver.exe'], 
                                       capture_output=True, timeout=10)
                    else:
                        # Linux/Mac: 使用 pkill
                        subprocess.run(['pkill', '-9', '-f', 'Chromium'], 
                                       capture_output=True, timeout=10)
                        subprocess.run(['pkill', '-9', '-f', 'chromedriver'], 
                                       capture_output=True, timeout=10)
                except Exception as e:
                    logger.warning(f"Force kill browser failed: {e}")
            
            # 【2026-01-26 新增】每次测试后强制清理内存
            import gc
            gc.collect()
            logger.info("Memory garbage collection done")
            
            # 【ASE 论文核心】收集 Bug 分析数据
            try:
                mas = webtest.multi_agent_system
                # 检查是否是 SHAQv2（有 bug_analyzer 属性）
                if hasattr(mas, 'bug_analyzer') and mas.bug_analyzer is not None:
                    bug_stats = mas.bug_analyzer.get_combination_bug_statistics()
                    monitor.metrics.total_bugs_detected = bug_stats.get('total_bugs_detected', 0)
                    monitor.metrics.combination_bugs_count = bug_stats.get('combination_bugs_count', 0)
                    monitor.metrics.combination_bug_ratio = bug_stats.get('combination_bug_ratio', 0.0)
                    monitor.metrics.avg_interaction_index = bug_stats.get('avg_interaction_index', 0.0)
                    
                    # 获取定位统计
                    if hasattr(mas, 'bug_localizer') and mas.bug_localizer is not None:
                        loc_stats = mas.bug_localizer.get_localization_statistics()
                        monitor.metrics.high_confidence_localizations = loc_stats.get('high_confidence_count', 0)
                        monitor.metrics.localization_confidence = loc_stats.get('avg_confidence', 0.0)
                    
                    # 保存原始数据（用于详细分析）
                    monitor.metrics.bug_analysis_raw = {
                        'combination_bugs': bug_stats,
                        'localization': loc_stats if hasattr(mas, 'bug_localizer') else {},
                    }
                    
                    logger.info(f"[Bug Analysis] Total: {monitor.metrics.total_bugs_detected}, "
                               f"Combination: {monitor.metrics.combination_bugs_count}, "
                               f"High-confidence: {monitor.metrics.high_confidence_localizations}")
            except Exception as e:
                logger.warning(f"Error collecting bug analysis data: {e}")
            
            logger.info(f"Profile {profile} test completed")
        
        finally:
            # 恢复原始 argv
            sys.argv = original_argv
            
            # 清理临时目录
            if temp_data_dir and os.path.exists(temp_data_dir):
                try:
                    shutil.rmtree(temp_data_dir)
                    self._temp_dirs.remove(temp_data_dir)
                    logger.info(f"Cleaned up temp dir: {temp_data_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean temp dir: {e}")
    
    def cleanup_temp_dirs(self):
        """清理所有临时目录"""
        for temp_dir in self._temp_dirs[:]:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self._temp_dirs.remove(temp_dir)
                    logger.info(f"Cleaned up temp dir: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean temp dir: {e}")
    
    def compare_profiles(self, profiles: List[str], duration: int = 300,
                        dry_run: bool = False, use_incognito: bool = True,
                        seed: Optional[int] = None) -> Dict[str, PerformanceMetrics]:
        """
        对比多个配置的性能
        
        Args:
            profiles: 配置名称列表
            duration: 每个配置的测试时长
            dry_run: 是否干跑
            use_incognito: 是否使用隐身模式
            seed: 随机种子（所有配置使用相同种子以确保公平）
        """
        # 预加载 Word2Vec 模型，避免在第一个算法测试时花费 45 秒加载
        # 这样可以确保所有算法的测试时间都是公平的
        if not dry_run:
            try:
                logger.info("Preloading Word2Vec model...")
                from transformer.utils.word2vec_cache import preload_word2vec_model, get_cache_info
                preload_word2vec_model()
                cache_info = get_cache_info()
                logger.info(f"Word2Vec model cached: load_time={cache_info['load_time']:.1f}s, "
                           f"vocab_size={cache_info['vocab_size']}")
            except Exception as e:
                logger.warning(f"Failed to preload Word2Vec (non-fatal): {e}")
        
        for profile in profiles:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing profile: {profile}")
            logger.info(f"{'='*60}")
            
            try:
                self.run_benchmark(profile, duration, dry_run, seed=seed, use_incognito=use_incognito)
            except Exception as e:
                logger.error(f"Profile {profile} test failed: {e}")
                traceback.print_exc()
        
        return self.results
    
    def compare_profiles_multi_run(self, profiles: List[str], duration: int = 300,
                                   num_runs: int = 3, use_incognito: bool = True,
                                   seeds: Optional[List[int]] = None,
                                   dry_run: bool = False) -> Dict[str, AggregatedMetrics]:
        """
        对比多个配置的性能（多次运行版本，用于论文）
        
        Args:
            profiles: 配置名称列表
            duration: 每次测试时长
            num_runs: 每个配置运行次数
            use_incognito: 是否使用隐身模式
            seeds: 随机种子列表（所有配置使用相同种子序列）
            dry_run: 是否干跑
        """
        # 预加载 Word2Vec 模型
        if not dry_run:
            try:
                logger.info("Preloading Word2Vec model...")
                from transformer.utils.word2vec_cache import preload_word2vec_model, get_cache_info
                preload_word2vec_model()
                cache_info = get_cache_info()
                logger.info(f"Word2Vec model cached: load_time={cache_info['load_time']:.1f}s, "
                           f"vocab_size={cache_info['vocab_size']}")
            except Exception as e:
                logger.warning(f"Failed to preload Word2Vec (non-fatal): {e}")
        
        for profile in profiles:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing profile: {profile} ({num_runs} runs)")
            logger.info(f"{'='*60}")
            
            try:
                self.run_multiple(
                    profile=profile,
                    duration=duration,
                    num_runs=num_runs,
                    seeds=seeds,
                    use_incognito=use_incognito,
                    dry_run=dry_run
                )
            except Exception as e:
                logger.error(f"Profile {profile} test failed: {e}")
                traceback.print_exc()
        
        return self.aggregated_results
    
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        if not self.results:
            return "没有测试结果"
        
        report = []
        report.append("\n" + "=" * 80)
        report.append(" " * 25 + "Multi-Agent Algorithm Performance Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Metrics to compare
        metrics_to_compare = [
            # Coverage metrics
            ("States", "unique_states", "", True),
            ("Actions", "unique_actions", "", True),
            ("URLs", "unique_urls", "", True),
            ("URL Paths", "unique_url_paths", "", True),
            # Time performance
            ("Total Steps", "total_steps", "", True),
            ("Duration", "effective_duration_seconds", "s", False),
            ("Step Time", "effective_step_duration_ms", "ms", False),
            ("Decision Time", "avg_decision_time_ms", "ms", False),
            ("Throughput", "effective_steps_per_second", "/s", True),
            ("Anomalies", "anomaly_count", "", False),
            # Reward metrics
            ("Total Reward", "total_reward", "", True),
            ("Avg Reward", "avg_reward_per_step", "", True),
            ("Novelty", "avg_state_novelty", "", True),
            # Exploration efficiency
            ("New State Rate", "new_state_rate", "", True),
            ("Explore Eff", "exploration_efficiency", "", True),
            ("Action Div", "action_diversity", "", True),
            # Efficiency ratios
            ("State Eff", "state_efficiency", "", True),
            ("URL Eff", "url_efficiency", "", True),
            ("Reward Eff", "reward_efficiency", "", True),
            # Action types
            ("Clicks", "click_actions", "", True),
            ("Inputs", "input_actions", "", True),
            # Errors
            ("Act Fails", "action_failures", "", False),
            ("Out Domain", "out_of_domain_count", "", False),
            # Bug detection
            ("Bugs Total", "total_bugs_detected", "", True),
            ("Combo Bugs", "combination_bugs_count", "", True),
            ("Combo Ratio", "combination_bug_ratio", "", True),
            ("Interact Idx", "avg_interaction_index", "", True),
            ("Hi-Conf Loc", "high_confidence_localizations", "", True),
            ("Loc Conf", "localization_confidence", "", True),
            # Learning and resources
            ("Learn Updates", "learning_updates", "", True),
            ("Avg Loss", "avg_loss", "", False),
            ("Peak Mem", "peak_memory_mb", "MB", False),
        ]
        
        # Header
        profiles = list(self.results.keys())
        header = f"{'Metric':<15} | " + " | ".join(f"{p[:15]:<15}" for p in profiles)
        report.append(header)
        report.append("-" * len(header))
        
        # 找出每个指标的最佳值
        for name, attr, unit, higher_better in metrics_to_compare:
            values = []
            for p in profiles:
                v = getattr(self.results[p], attr, 0)
                values.append(v)
            
            # 找最佳
            if higher_better:
                best_idx = values.index(max(values)) if values else -1
            else:
                non_zero = [v for v in values if v > 0]
                best_idx = values.index(min(non_zero)) if non_zero else -1
            
            # 格式化行
            row = f"{name:<15} | "
            for i, v in enumerate(values):
                if isinstance(v, float):
                    val_str = f"{v:.2f}{unit}"
                else:
                    val_str = f"{v}{unit}"
                
                if i == best_idx:
                    val_str = f"*{val_str}*"  # 标记最佳
                
                row += f"{val_str:<15} | "
            
            report.append(row)
        
        report.append("")
        report.append("Note: * indicates best value for this metric")
        report.append("")
        
        # 综合评分
        report.append("\n[Overall Score] (max 100)")
        report.append("-" * 40)
        
        for profile, metrics in self.results.items():
            score = self._calculate_score(metrics)
            # 使用 ASCII 兼容字符，避免 Windows GBK 编码问题
            filled = int(score / 5)
            bar = "#" * filled + "-" * (20 - filled)
            report.append(f"{metrics.algorithm:<10} [{bar}] {score:.1f}")
        
        report.append("")
        return "\n".join(report)
    
    def _calculate_score(self, metrics: PerformanceMetrics) -> float:
        """
        计算综合评分
        
        评分维度：
        - 状态覆盖 (30分): 发现的唯一状态数越多越好
        - 吞吐量 (25分): 有效每秒步数越高越好
        - 探索效率 (20分): 新状态发现率 + 动作多样性
        - 稳定性 (15分): 异常卡顿越少越好
        - 资源效率 (10分): 内存使用越低越好
        """
        score = 0.0
        
        # 状态覆盖 (30分) - 100个状态得满分
        score += min(30, metrics.unique_states / 100 * 30)
        
        # 吞吐量 (25分) - 使用有效每秒步数，0.5步/秒得满分
        if metrics.effective_steps_per_second > 0:
            throughput_score = min(25, metrics.effective_steps_per_second / 0.5 * 25)
            score += throughput_score
        
        # 探索效率 (20分) - 新状态率和动作多样性各占一半
        exploration_score = (metrics.new_state_rate + metrics.action_diversity) / 2 * 20
        score += min(20, exploration_score)
        
        # 稳定性 (15分) - 无异常得满分，每次异常扣3分
        stability_score = max(0, 15 - metrics.anomaly_count * 3)
        score += stability_score
        
        # 资源效率 (10分) - 500MB以下得满分
        if metrics.peak_memory_mb > 0:
            mem_score = max(0, 10 - max(0, metrics.peak_memory_mb - 500) / 100)
            score += mem_score
        else:
            score += 10  # 无数据时给满分
        
        return min(100, score)
    
    def save_results(self, output_path: str):
        """保存结果到 JSON 文件"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": {
                profile: metrics.to_dict() 
                for profile, metrics in self.results.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")


class BenchmarkVisualizer:
    """
    Benchmark 结果可视化工具
    
    生成论文级图表：
    1. Cumulative Reward 曲线（带置信区间阴影）
    2. Unique States 发现曲线
    3. Step Times 计算开销对比
    4. 综合雷达图
    """
    
    # 论文风格配色（区分度高，打印友好）
    # 注意：匹配时按 key 长度降序，避免 'SHAQ' 错误匹配 'SHAQv2' 或 'P-SHAQ'
    COLORS = {
        'P-SHAQ': '#17becf',    # 青色 (新算法，突出显示)
        'SHAQv2': '#2ca02c',    # 绿色
        'SHAQ': '#1f77b4',      # 蓝色
        'QTRAN': '#ff7f0e',     # 橙色
        'QMIX': '#d62728',      # 红色
        'Marg-DQL': '#9467bd',  # 紫色
        'NNDQL': '#bcbd22',     # 黄绿色
        'IQL': '#8c564b',       # 棕色
        'VDN': '#e377c2',       # 粉色
        'default': '#7f7f7f',   # 灰色
    }
    
    def __init__(self, output_dir: str = './benchmark_figures'):
        """
        初始化可视化器
        
        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 尝试导入 matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端，适合服务器
            import matplotlib.pyplot as plt
            self.plt = plt
            self.matplotlib = matplotlib
            
            # 设置中文字体支持
            import platform
            system = platform.system()
            if system == 'Windows':
                # Windows 系统使用微软雅黑
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
            elif system == 'Darwin':
                # macOS 系统
                plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'DejaVu Sans']
            else:
                # Linux 系统
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 设置论文风格
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'legend.fontsize': 11,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'figure.figsize': (10, 6),
                'figure.dpi': 150,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
            })
            self.available = True
            logger.info("Matplotlib visualization module loaded")
        except ImportError as e:
            logger.warning(f"Matplotlib not available, skipping visualization: {e}")
            self.available = False
    
    def get_color(self, algorithm: str) -> str:
        """获取算法对应的颜色（按 key 长度降序匹配，避免短 key 错误匹配长名称）"""
        algo_lower = algorithm.lower()
        # 按 key 长度降序排序，确保 'P-SHAQ' 和 'SHAQv2' 优先于 'SHAQ' 匹配
        sorted_colors = sorted(self.COLORS.items(), key=lambda x: len(x[0]), reverse=True)
        for key, color in sorted_colors:
            if key.lower() in algo_lower:
                return color
        return self.COLORS['default']
    
    def load_results(self, json_path: str) -> Dict:
        """加载 benchmark 结果 JSON 文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_comparison(self, json_path: str, show: bool = False):
        """
        生成完整的对比图表集
        
        Args:
            json_path: benchmark 结果 JSON 文件路径
            show: 是否显示图表（交互模式）
        """
        if not self.available:
            logger.warning("Matplotlib not available, skipping plots")
            return
        
        data = self.load_results(json_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 检查是否有聚合结果（多次运行）
        has_aggregated = bool(data.get('aggregated_results'))
        has_single = bool(data.get('single_run_results') or data.get('results'))
        
        single_results = data.get('single_run_results') or data.get('results', {})
        agg_results = data.get('aggregated_results', {})
        
        if has_single and single_results:
            # 1. 累积奖励曲线（需要时间序列数据，使用 single_run）
            self._plot_cumulative_reward(single_results, timestamp)
            
            # 2. 状态发现曲线（使用 single_run，但标注聚合均值）
            self._plot_state_discovery(single_results, timestamp, agg_results if has_aggregated else None)
        
        # 3. 计算时间对比 - 使用聚合数据 + single_run 的 step duration
        if has_aggregated and agg_results:
            self._plot_step_times_aggregated(agg_results, single_results, timestamp)
        elif has_single and single_results:
            self._plot_step_times(single_results, timestamp)
        
        # 4. 综合对比柱状图 - 优先使用聚合数据
        if has_aggregated and agg_results:
            self._plot_summary_bars_aggregated(agg_results, timestamp)
        elif has_single and single_results:
            self._plot_summary_bars(single_results, timestamp)
        
        if has_aggregated and agg_results:
            # 5. 多次运行的置信区间图
            self._plot_aggregated_comparison(agg_results, timestamp)
            
            # 6. 稳定性对比图（均值 vs 标准差）
            self._plot_stability_comparison(agg_results, timestamp)
        
        # 6. 生成综合面板图 - 优先使用聚合数据
        if has_aggregated and agg_results:
            self._plot_combined_panel_aggregated(agg_results, single_results, timestamp)
        elif has_single and single_results:
            self._plot_combined_panel(single_results, timestamp)
        
        logger.info(f"Charts saved to: {self.output_dir}")
        
        if show and self.available:
            self.plt.show()
    
    def _plot_cumulative_reward(self, results: Dict, timestamp: str):
        """绘制累积奖励曲线"""
        plt = self.plt
        fig, ax = plt.subplots(figsize=(10, 6))
        
        has_data = False
        for profile, metrics in results.items():
            algo = metrics.get('algorithm', profile)
            color = self.get_color(algo)
            
            # 从奖励曲线数据绘制
            reward_curve = metrics.get('reward_curve', [])
            
            if reward_curve and len(reward_curve) > 0:
                # 计算累积奖励，X轴使用步数
                cumulative = np.cumsum(reward_curve)
                steps = list(range(len(reward_curve)))
                ax.plot(steps, cumulative, label=algo, color=color, linewidth=2)
                has_data = True
            else:
                # 如果没有曲线数据，用总奖励画一个点
                total_reward = metrics.get('total_reward', 0)
                total_steps = metrics.get('total_steps', 1)
                if total_reward != 0:
                    ax.scatter([total_steps], [total_reward], label=f"{algo} (总计)", 
                              color=color, s=100, marker='o')
                    has_data = True
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward Comparison')
        if has_data:
            ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        fig.savefig(f'{self.output_dir}/cumulative_reward_{timestamp}.png')
        plt.close(fig)
        logger.info(f"Generated: cumulative_reward_{timestamp}.png")
    
    def _plot_state_discovery(self, results: Dict, timestamp: str, agg_results: Dict = None):
        """绘制状态发现曲线，可选地标注聚合均值"""
        plt = self.plt
        fig, ax = plt.subplots(figsize=(12, 7))
        
        max_steps = 0
        for profile, metrics in results.items():
            algo = metrics.get('algorithm', profile)
            color = self.get_color(algo)
            
            state_curve = metrics.get('state_discovery_curve', [])
            
            if state_curve and len(state_curve) > 0:
                steps = list(range(len(state_curve)))
                max_steps = max(max_steps, len(state_curve))
                ax.plot(steps, state_curve, label=algo, color=color, linewidth=2)
                
                # 如果有聚合数据，在曲线末端标注均值
                if agg_results and profile in agg_results:
                    agg_m = agg_results[profile].get('metrics', {})
                    agg_states = agg_m.get('unique_states', {})
                    mean_val = agg_states.get('mean', 0) if isinstance(agg_states, dict) else agg_states
                    final_val = state_curve[-1]
                    
                    # 用虚线标注均值位置
                    ax.hlines(mean_val, 0, len(state_curve), colors=color, 
                             linestyles='dashed', alpha=0.5, linewidth=1)
                    ax.annotate(f'mean={mean_val:.0f}', 
                               xy=(len(state_curve)*0.02, mean_val),
                               fontsize=8, color=color, alpha=0.8)
            else:
                unique_states = metrics.get('unique_states', 0)
                total_steps = metrics.get('total_steps', 1)
                if unique_states > 0:
                    ax.scatter([total_steps], [unique_states], label=f"{algo} (final)", 
                              color=color, s=100, marker='s')
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Unique States Discovered')
        title = 'State Discovery Efficiency\n(Solid line: single run, Dashed line: 3-run mean)'
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        fig.savefig(f'{self.output_dir}/state_discovery_{timestamp}.png', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Generated: state_discovery_{timestamp}.png")
    
    def _plot_step_times(self, results: Dict, timestamp: str):
        """绘制计算时间对比（关键：评估 Lovasz Extension 的计算代价）"""
        plt = self.plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        algorithms = []
        decision_times = []
        step_durations = []
        colors = []
        
        for profile, metrics in results.items():
            algo = metrics.get('algorithm', profile)
            algorithms.append(algo)
            decision_times.append(metrics.get('avg_decision_time_ms', 0))
            step_durations.append(metrics.get('effective_step_duration_ms', 0))
            colors.append(self.get_color(algo))
        
        # Left: Algorithm decision time (key metric)
        bars1 = ax1.bar(algorithms, decision_times, color=colors, alpha=0.8)
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Avg Decision Time (ms)')
        ax1.set_title('Decision Time Comparison\n(NN Inference + Shapley/Mixing)')
        ax1.tick_params(axis='x', rotation=15)
        
        # Add value labels
        for bar, val in zip(bars1, decision_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Right: End-to-end step time
        bars2 = ax2.bar(algorithms, step_durations, color=colors, alpha=0.8)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Avg Step Time (ms)')
        ax2.set_title('End-to-End Step Time\n(incl. Browser/Network)')
        ax2.tick_params(axis='x', rotation=15)
        
        for bar, val in zip(bars2, step_durations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/step_times_{timestamp}.png')
        plt.close(fig)
        logger.info(f"Generated: step_times_{timestamp}.png")
    
    def _plot_step_times_aggregated(self, agg_results: Dict, single_results: Dict, timestamp: str):
        """绘制计算时间对比（决策时间用聚合数据，端到端用 single_run）"""
        plt = self.plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        algorithms = list(agg_results.keys())
        algo_names = [agg_results[p].get('algorithm', p) for p in algorithms]
        colors = [self.get_color(name) for name in algo_names]
        
        # 从聚合数据提取 decision time 均值和标准差
        decision_times_mean = []
        decision_times_std = []
        # 从 single_run 提取 step duration（因为聚合数据中丢失了）
        step_durations = []
        
        for p in algorithms:
            # Decision time from aggregated
            data = agg_results[p]
            metrics = data.get('metrics', {})
            dt = metrics.get('avg_decision_time_ms', {})
            decision_times_mean.append(dt.get('mean', 0) if isinstance(dt, dict) else dt)
            decision_times_std.append(dt.get('std', 0) if isinstance(dt, dict) else 0)
            
            # Step duration from single_run (fallback)
            if single_results and p in single_results:
                sd = single_results[p].get('effective_step_duration_ms', 0)
            else:
                sd = metrics.get('effective_step_duration_ms', {})
                sd = sd.get('mean', 0) if isinstance(sd, dict) else sd
            step_durations.append(sd)
        
        x_pos = np.arange(len(algorithms))
        
        # Left: Decision time with error bars
        bars1 = ax1.bar(x_pos, decision_times_mean, yerr=decision_times_std, 
                       capsize=5, color=colors, alpha=0.8, ecolor='black')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Avg Decision Time (ms)')
        ax1.set_title('Decision Time Comparison (mean ± std)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(algo_names, rotation=25, ha='right')
        
        max_dt = max(decision_times_mean) if decision_times_mean else 1
        for bar, val, std in zip(bars1, decision_times_mean, decision_times_std):
            label = f'{val:.1f}' if std == 0 else f'{val:.1f}±{std:.1f}'
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + max_dt*0.02,
                    label, ha='center', va='bottom', fontsize=8)
        
        # Right: Step duration (from single run data)
        bars2 = ax2.bar(x_pos, step_durations, color=colors, alpha=0.8)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Avg Step Time (ms)')
        ax2.set_title('End-to-End Step Time\n(incl. Browser/Network, from representative run)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(algo_names, rotation=25, ha='right')
        
        max_sd = max(step_durations) if step_durations else 1
        for bar, val in zip(bars2, step_durations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_sd*0.02,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        fig.tight_layout()
        fig.savefig(f'{self.output_dir}/step_times_{timestamp}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info(f"Generated: step_times_{timestamp}.png (aggregated + single_run)")
    
    def _plot_summary_bars(self, results: Dict, timestamp: str):
        """绘制综合对比柱状图"""
        plt = self.plt
        
        metrics_to_plot = [
            ('unique_states', 'State Coverage', 'count'),
            ('unique_urls', 'URL Coverage', 'count'),
            ('total_steps', 'Total Steps', 'steps'),
            ('effective_steps_per_second', 'Throughput', 'steps/s'),
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        algorithms = list(results.keys())
        algo_names = [results[p].get('algorithm', p) for p in algorithms]
        colors = [self.get_color(name) for name in algo_names]
        
        for idx, (metric_key, title, unit) in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [results[p].get(metric_key, 0) for p in algorithms]
            
            bars = ax.bar(algo_names, values, color=colors, alpha=0.8)
            ax.set_title(title)
            ax.set_ylabel(f'{title} ({unit})')
            ax.tick_params(axis='x', rotation=15)
            
            # 标记最佳值
            if values:
                max_idx = values.index(max(values))
                bars[max_idx].set_edgecolor('gold')
                bars[max_idx].set_linewidth(3)
        
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/summary_comparison_{timestamp}.png')
        plt.close(fig)
        logger.info(f"Generated: summary_comparison_{timestamp}.png")
    
    def _plot_summary_bars_aggregated(self, agg_results: Dict, timestamp: str):
        """绘制综合对比柱状图（使用聚合数据的均值）"""
        plt = self.plt
        
        metrics_to_plot = [
            ('unique_states', 'State Coverage', 'count'),
            ('unique_urls', 'URL Coverage', 'count'),
            ('total_steps', 'Total Steps', 'steps'),
            ('effective_steps_per_second', 'Throughput', 'steps/s'),
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        algorithms = list(agg_results.keys())
        algo_names = [agg_results[p].get('algorithm', p) for p in algorithms]
        colors = [self.get_color(name) for name in algo_names]
        
        for idx, (metric_key, title, unit) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # 从 metrics 嵌套结构中获取均值
            values = []
            for p in algorithms:
                data = agg_results[p]
                if 'metrics' in data and metric_key in data['metrics']:
                    values.append(data['metrics'][metric_key].get('mean', 0))
                else:
                    values.append(data.get(f'{metric_key}_mean', 0))
            
            bars = ax.bar(algo_names, values, color=colors, alpha=0.8)
            ax.set_title(title)
            ax.set_ylabel(f'{title} ({unit})')
            ax.tick_params(axis='x', rotation=15)
            
            # 标记最佳值
            if values:
                max_idx = values.index(max(values))
                bars[max_idx].set_edgecolor('gold')
                bars[max_idx].set_linewidth(3)
        
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/summary_comparison_{timestamp}.png')
        plt.close(fig)
        logger.info(f"Generated: summary_comparison_{timestamp}.png (aggregated)")
    
    def _plot_aggregated_comparison(self, agg_results: Dict, timestamp: str):
        """绘制多次运行的聚合对比图（带置信区间）"""
        plt = self.plt
        
        metrics_to_plot = [
            ('unique_states', 'State Coverage'),
            ('total_steps', 'Total Steps'),
            ('avg_decision_time_ms', 'Decision Time (ms)'),
            ('total_reward', 'Cumulative Reward'),
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        algorithms = list(agg_results.keys())
        algo_names = [agg_results[p].get('algorithm', p) for p in algorithms]
        colors = [self.get_color(name) for name in algo_names]
        x_pos = np.arange(len(algorithms))
        
        for idx, (metric_key, title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            means = []
            stds = []
            for p in algorithms:
                metric_data = agg_results[p].get('metrics', {}).get(metric_key, {})
                means.append(metric_data.get('mean', 0))
                stds.append(metric_data.get('std', 0))
            
            # 绘制带误差棒的柱状图
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                         color=colors, alpha=0.8, ecolor='black')
            
            ax.set_title(f'{title}\n(mean ± std, n={agg_results[algorithms[0]].get("num_runs", "?")})')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(algo_names, rotation=15)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                label = f'{mean:.1f}±{std:.1f}' if std > 0 else f'{mean:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                       label, ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/aggregated_comparison_{timestamp}.png')
        plt.close(fig)
        logger.info(f"Generated: aggregated_comparison_{timestamp}.png (with confidence intervals)")
    
    def _plot_stability_comparison(self, agg_results: Dict, timestamp: str):
        """绘制稳定性对比图（均值 vs 变异系数）"""
        plt = self.plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        algorithms = list(agg_results.keys())
        algo_names = [agg_results[p].get('algorithm', p) for p in algorithms]
        colors = [self.get_color(name) for name in algo_names]
        
        metrics_to_plot = [
            ('unique_states', 'State Coverage'),
            ('unique_urls', 'URL Coverage'),
            ('total_reward', 'Total Reward'),
            ('total_steps', 'Total Steps'),
        ]
        
        for idx, (metric_key, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            means = []
            stds = []
            cvs = []  # 变异系数 (Coefficient of Variation)
            
            for p in algorithms:
                metrics = agg_results[p].get('metrics', {})
                m = metrics.get(metric_key, {})
                mean = m.get('mean', 0) if isinstance(m, dict) else m
                std = m.get('std', 0) if isinstance(m, dict) else 0
                cv = (std / mean * 100) if mean > 0 else 0
                means.append(mean)
                stds.append(std)
                cvs.append(cv)
            
            x_pos = np.arange(len(algorithms))
            width = 0.35
            
            # 左Y轴：均值（柱状图）
            bars = ax.bar(x_pos - width/2, means, width, color=colors, alpha=0.8, label='Mean')
            ax.set_ylabel(f'{title} (Mean)', color='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(algo_names, rotation=25, ha='right')
            
            # 添加误差棒
            ax.errorbar(x_pos - width/2, means, yerr=stds, fmt='none', 
                       ecolor='black', capsize=3, capthick=1.5)
            
            # 右Y轴：变异系数（线图）
            ax2 = ax.twinx()
            line = ax2.plot(x_pos, cvs, 'ro-', linewidth=2, markersize=8, label='CV%')
            ax2.set_ylabel('Coefficient of Variation (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, max(cvs) * 1.3 if cvs else 50)
            
            # 标注CV值
            for i, cv in enumerate(cvs):
                ax2.annotate(f'{cv:.1f}%', (x_pos[i], cv), 
                           textcoords="offset points", xytext=(0, 8),
                           ha='center', fontsize=9, color='red')
            
            ax.set_title(f'{title}\n(Higher bar = better, Lower CV% = more stable)')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 添加图例说明
        fig.suptitle('Algorithm Stability Comparison\n(Mean Performance vs Consistency)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/stability_comparison_{timestamp}.png', 
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info(f"Generated: stability_comparison_{timestamp}.png")
    
    def _plot_combined_panel(self, results: Dict, timestamp: str):
        """生成综合面板图（适合论文单图展示）"""
        plt = self.plt
        
        fig = plt.figure(figsize=(16, 12))
        
        # 创建 2x2 的子图布局
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        algorithms = list(results.keys())
        algo_names = [results[p].get('algorithm', p) for p in algorithms]
        colors = [self.get_color(name) for name in algo_names]
        
        # 1. 左上：累积奖励曲线
        ax1 = fig.add_subplot(gs[0, 0])
        for i, (profile, metrics) in enumerate(results.items()):
            algo = metrics.get('algorithm', profile)
            reward_curve = metrics.get('reward_curve', [])
            if reward_curve:
                cumulative = np.cumsum(reward_curve)
                ax1.plot(cumulative, label=algo, color=colors[i], linewidth=2)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('(a) Cumulative Reward')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 右上：状态发现曲线
        ax2 = fig.add_subplot(gs[0, 1])
        for i, (profile, metrics) in enumerate(results.items()):
            algo = metrics.get('algorithm', profile)
            state_curve = metrics.get('state_discovery_curve', [])
            if state_curve:
                ax2.plot(state_curve, label=algo, color=colors[i], linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Unique States')
        ax2.set_title('(b) State Discovery')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. 左下：决策时间对比
        ax3 = fig.add_subplot(gs[1, 0])
        decision_times = [results[p].get('avg_decision_time_ms', 0) for p in algorithms]
        bars = ax3.bar(algo_names, decision_times, color=colors, alpha=0.8)
        ax3.set_ylabel('Decision Time (ms)')
        ax3.set_title('(c) Computational Cost')
        ax3.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, decision_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 右下：综合性能雷达图
        ax4 = fig.add_subplot(gs[1, 1], projection='polar')
        
        # 雷达图的维度
        categories = ['States', 'URLs', 'Steps', 'Throughput', 'Stability']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合
        
        for i, (profile, metrics) in enumerate(results.items()):
            algo = metrics.get('algorithm', profile)
            
            # 归一化各指标到 0-1
            values = [
                min(1, metrics.get('unique_states', 0) / 100),
                min(1, metrics.get('unique_urls', 0) / 50),
                min(1, metrics.get('total_steps', 0) / 1000),
                min(1, metrics.get('effective_steps_per_second', 0) / 0.5),
                max(0, 1 - metrics.get('anomaly_count', 0) / 5),
            ]
            values += values[:1]  # 闭合
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors[i])
            ax4.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_title('(d) Overall Performance')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.suptitle('Multi-Agent Algorithm Performance Comparison', fontsize=16, y=1.02)
        fig.savefig(f'{self.output_dir}/combined_panel_{timestamp}.png', 
                   bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        logger.info(f"Generated: combined_panel_{timestamp}.png (combined panel)")
    
    def _plot_combined_panel_aggregated(self, agg_results: Dict, single_results: Dict, timestamp: str):
        """生成综合面板图（使用聚合数据，曲线用单次运行数据）"""
        plt = self.plt
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        algorithms = list(agg_results.keys())
        algo_names = [agg_results[p].get('algorithm', p) for p in algorithms]
        colors = [self.get_color(name) for name in algo_names]
        
        # 1. 左上：累积奖励曲线（用 single_results 的时间序列）
        ax1 = fig.add_subplot(gs[0, 0])
        for i, profile in enumerate(algorithms):
            algo = agg_results[profile].get('algorithm', profile)
            # 尝试从 single_results 获取曲线数据
            if single_results and profile in single_results:
                reward_curve = single_results[profile].get('reward_curve', [])
                if reward_curve:
                    cumulative = np.cumsum(reward_curve)
                    ax1.plot(cumulative, label=algo, color=colors[i], linewidth=2)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('(a) Cumulative Reward')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 右上：状态发现曲线（用 single_results）
        ax2 = fig.add_subplot(gs[0, 1])
        for i, profile in enumerate(algorithms):
            algo = agg_results[profile].get('algorithm', profile)
            if single_results and profile in single_results:
                state_curve = single_results[profile].get('state_discovery_curve', [])
                if state_curve:
                    ax2.plot(state_curve, label=algo, color=colors[i], linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Unique States')
        ax2.set_title('(b) State Discovery')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. 左下：决策时间对比（用聚合均值）
        ax3 = fig.add_subplot(gs[1, 0])
        decision_times = []
        for p in algorithms:
            metrics = agg_results[p].get('metrics', {})
            dt = metrics.get('avg_decision_time_ms', {})
            decision_times.append(dt.get('mean', 0) if isinstance(dt, dict) else dt)
        
        bars = ax3.bar(algo_names, decision_times, color=colors, alpha=0.8)
        ax3.set_ylabel('Decision Time (ms)')
        ax3.set_title('(c) Computational Cost (mean)')
        ax3.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, decision_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 右下：综合性能雷达图（用聚合均值）
        ax4 = fig.add_subplot(gs[1, 1], projection='polar')
        
        categories = ['States', 'URLs', 'Steps', 'Throughput', 'Stability']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        for i, profile in enumerate(algorithms):
            algo = agg_results[profile].get('algorithm', profile)
            metrics = agg_results[profile].get('metrics', {})
            
            # 从聚合数据获取均值
            def get_mean(key, default=0):
                val = metrics.get(key, {})
                return val.get('mean', default) if isinstance(val, dict) else val
            
            values = [
                min(1, get_mean('unique_states', 0) / 500),  # 调整归一化范围
                min(1, get_mean('unique_urls', 0) / 250),
                min(1, get_mean('total_steps', 0) / 1500),
                min(1, get_mean('effective_steps_per_second', 0) / 0.5),
                max(0, 1 - get_mean('anomaly_count', 0) / 5),
            ]
            values += values[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors[i])
            ax4.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_title('(d) Overall Performance')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.suptitle('Multi-Agent Algorithm Performance Comparison (Aggregated)', fontsize=16, y=1.02)
        fig.savefig(f'{self.output_dir}/combined_panel_{timestamp}.png', 
                   bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        logger.info(f"Generated: combined_panel_{timestamp}.png (aggregated)")
    
    def plot_reward_with_confidence(self, agg_results: Dict, timestamp: str = None):
        """
        绘制带置信区间阴影的奖励曲线（论文核心图）
        
        这是验证"信度分配"效果的关键图表：
        - SHAQ 的曲线上升趋势应该更陡峭
        - 阴影（std）越窄，说明 Lovasz Extension 让协同更稳定
        """
        if not self.available:
            return
        
        plt = self.plt
        timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for profile, agg in agg_results.items():
            algo = agg.get('algorithm', profile)
            color = self.get_color(algo)
            
            raw_runs = agg.get('raw_runs', [])
            if not raw_runs:
                continue
            
            # 收集所有运行的奖励曲线
            all_reward_curves = []
            max_len = 0
            for run in raw_runs:
                curve = run.get('reward_curve', [])
                if curve:
                    all_reward_curves.append(np.cumsum(curve))
                    max_len = max(max_len, len(curve))
            
            if not all_reward_curves:
                continue
            
            # 对齐长度（用最后一个值填充）
            aligned_curves = []
            for curve in all_reward_curves:
                if len(curve) < max_len:
                    padding = np.full(max_len - len(curve), curve[-1])
                    curve = np.concatenate([curve, padding])
                aligned_curves.append(curve[:max_len])
            
            # 计算均值和标准差
            curves_array = np.array(aligned_curves)
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            
            x = np.arange(len(mean_curve))
            
            # 绘制均值曲线
            ax.plot(x, mean_curve, label=algo, color=color, linewidth=2)
            
            # 绘制置信区间阴影
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                           color=color, alpha=0.2)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward with Confidence Interval\n(Shaded area = ±1 std)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        fig.savefig(f'{self.output_dir}/reward_confidence_{timestamp}.png')
        plt.close(fig)
        logger.info(f"Generated: reward_confidence_{timestamp}.png (with confidence interval)")


def main():
    parser = argparse.ArgumentParser(
        description="多智能体强化学习算法性能对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试单个配置
  python benchmark.py --profile shaq --duration 300
  
  # 对比多个配置（单次运行）
  python benchmark.py --compare shaq,qtran --duration 1800
  
  # 论文级对比：多次运行 + 统计汇总
  python benchmark.py --compare shaq,qtran --duration 1800 --num-runs 3
  
  # 指定随机种子（可复现）
  python benchmark.py --profile shaq --duration 300 --seed 42
  
  # 干跑测试（不启动浏览器）
  python benchmark.py --profile shaq --duration 60 --dry-run
  
  # 禁用隐身模式（不推荐，可能导致缓存影响）
  python benchmark.py --profile shaq --no-incognito
  
  # 列出所有可用配置
  python benchmark.py --list-profiles
        """
    )
    
    parser.add_argument('--profile', type=str, help='要测试的配置名称')
    parser.add_argument('--compare', type=str, help='要对比的配置列表（逗号分隔）')
    parser.add_argument('--duration', type=int, default=300, help='测试时长（秒），默认300')
    parser.add_argument('--dry-run', action='store_true', help='干跑模式（不启动浏览器）')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='输出文件路径')
    parser.add_argument('--list-profiles', action='store_true', help='列出所有可用配置')
    parser.add_argument('--config', type=str, default='settings.yaml', help='配置文件路径')
    
    # 新增参数：随机种子和多次运行
    parser.add_argument('--seed', type=int, default=None, 
                        help='随机种子（用于可复现性），默认使用当前时间')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='每个配置运行次数（论文建议 3-5 次），默认1')
    parser.add_argument('--seeds', type=str, default=None,
                        help='多次运行的种子列表（逗号分隔），如 "42,123,456"')
    
    # 新增参数：浏览器环境隔离
    parser.add_argument('--no-incognito', action='store_true',
                        help='禁用隐身模式（不推荐，可能导致缓存影响测试公平性）')
    
    # 新增参数：可视化
    parser.add_argument('--figure-dir', type=str, default='./benchmark_figures',
                        help='图表输出目录，默认 ./benchmark_figures')
    parser.add_argument('--no-plot', action='store_true',
                        help='禁用自动生成图表')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.config)
    
    if args.list_profiles:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\nAvailable profiles:")
        print("-" * 70)
        for name, profile in config.get('profiles', {}).items():
            algo = runner.get_algorithm_name(profile)
            agent_num = profile.get('agent_num', '?')
            # 查找该配置的别名
            aliases = [k for k, v in runner.PROFILE_ALIASES.items() if v == name]
            alias_str = f" (别名: {', '.join(aliases)})" if aliases else ""
            print(f"  {name:<40} [{algo}, {agent_num} agents]{alias_str}")
        
        print("\nAvailable aliases:")
        print("-" * 70)
        for alias, full_name in sorted(runner.PROFILE_ALIASES.items()):
            print(f"  {alias:<15} -> {full_name}")
        print()
        return
    
    # 解析参数
    use_incognito = not args.no_incognito
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    elif args.seed:
        seeds = [args.seed]
    
    # 运行测试
    try:
        if args.compare:
            profiles = [p.strip() for p in args.compare.split(',')]
            
            if args.num_runs > 1:
                # 多次运行模式（论文级）
                logger.info(f"Paper-level test mode: {args.num_runs} runs per profile")
                runner.compare_profiles_multi_run(
                    profiles=profiles,
                    duration=args.duration,
                    num_runs=args.num_runs,
                    use_incognito=use_incognito,
                    seeds=seeds,
                    dry_run=args.dry_run
                )
                
                # 输出聚合报告
                print("\n" + "=" * 80)
                print(" " * 20 + "Performance Comparison Report (Multi-Run)")
                print("=" * 80)
                for profile, agg in runner.aggregated_results.items():
                    print(agg.summary())
            else:
                # 单次运行模式
                runner.compare_profiles(
                    profiles=profiles,
                    duration=args.duration,
                    dry_run=args.dry_run,
                    use_incognito=use_incognito,
                    seed=seeds[0] if seeds else None
                )
                
        elif args.profile:
            if args.num_runs > 1:
                # 多次运行模式
                runner.run_multiple(
                    profile=args.profile,
                    duration=args.duration,
                    num_runs=args.num_runs,
                    seeds=seeds,
                    use_incognito=use_incognito,
                    dry_run=args.dry_run
                )
                
                # 输出聚合报告
                for profile, agg in runner.aggregated_results.items():
                    print(agg.summary())
            else:
                # 单次运行
                runner.run_benchmark(
                    profile=args.profile,
                    duration=args.duration,
                    dry_run=args.dry_run,
                    seed=seeds[0] if seeds else None,
                    use_incognito=use_incognito
                )
        else:
            parser.print_help()
            return
        
        # 先保存结果（确保即使打印失败也能保存）
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "duration": args.duration,
                "num_runs": args.num_runs,
                "use_incognito": use_incognito,
                "seeds": seeds,
            },
            "single_run_results": {
                profile: metrics.to_dict() 
                for profile, metrics in runner.results.items()
            },
            "aggregated_results": {
                profile: agg.to_dict()
                for profile, agg in runner.aggregated_results.items()
            } if runner.aggregated_results else {}
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {args.output}")
        
        # Output report (use try-except to avoid encoding issues crashing the program)
        if runner.results:
            try:
                print(runner.generate_comparison_report())
                for profile, metrics in runner.results.items():
                    print(metrics.summary())
            except UnicodeEncodeError as e:
                logger.warning(f"Encoding error when printing report (results saved to file): {e}")
                # Try writing to file
                report_path = args.output.replace('.json', '_report.txt')
                try:
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(runner.generate_comparison_report())
                        for profile, metrics in runner.results.items():
                            f.write(metrics.summary())
                    logger.info(f"Report saved to: {report_path}")
                except Exception as e2:
                    logger.error(f"Failed to save report: {e2}")
        
        # === 生成可视化图表 ===
        if not args.no_plot:
            try:
                visualizer = BenchmarkVisualizer(output_dir=args.figure_dir)
                if visualizer.available:
                    visualizer.plot_comparison(args.output)
                    
                    # 如果有多次运行的聚合结果，额外生成置信区间图
                    if runner.aggregated_results:
                        visualizer.plot_reward_with_confidence(
                            {p: agg.to_dict() for p, agg in runner.aggregated_results.items()}
                        )
                        logger.info("Paper-level charts generated (with confidence intervals)")
                else:
                    logger.warning("Skipping charts (matplotlib not available)")
            except Exception as e:
                logger.error(f"Error generating charts: {e}")
                traceback.print_exc()
        
    finally:
        # 清理临时目录
        runner.cleanup_temp_dirs()


if __name__ == "__main__":
    main()
