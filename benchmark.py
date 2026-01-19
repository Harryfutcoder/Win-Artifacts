"""
多智能体强化学习算法性能对比测试工具

支持的算法:
- SHAQ (Shapley Q-value with Lovasz Extension)
- Marg (Marginal Contribution Q-learning)
- MargD (Deep Marginal Contribution)
- IQL (Independent Q-Learning)
- QMIX, VDN, QTRAN (通过 MargD 配置)

测试指标:
1. 状态覆盖率 (State Coverage)
2. 动作覆盖率 (Action Coverage)
3. 每步计算时间 (Time per Step)
4. 奖励曲线 (Cumulative Reward)
5. 状态新颖度 (State Novelty)
6. URL 覆盖率 (URL Coverage)
7. 内存使用 (Memory Usage)

使用方法:
    python benchmark.py --profile github-marl-3h-shaq-5agent --duration 300
    python benchmark.py --compare shaq,marg,qmix --duration 600
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import psutil

import yaml

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
    
    # 覆盖率指标
    total_states_discovered: int = 0
    total_actions_discovered: int = 0
    total_urls_visited: int = 0
    unique_states: int = 0
    unique_actions: int = 0
    unique_urls: int = 0
    
    # 时间性能
    total_steps: int = 0
    avg_step_time_ms: float = 0.0
    max_step_time_ms: float = 0.0
    min_step_time_ms: float = 0.0
    steps_per_second: float = 0.0
    
    # 学习性能
    total_reward: float = 0.0
    avg_reward_per_step: float = 0.0
    learning_updates: int = 0
    avg_loss: float = 0.0
    
    # 状态新颖度（越高越好）
    avg_state_novelty: float = 0.0
    novelty_over_time: List[float] = field(default_factory=list)
    
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
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def summary(self) -> str:
        """生成性能摘要"""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    性能测试报告 - {self.algorithm}
╠══════════════════════════════════════════════════════════════════╣
║ 配置: {self.profile}
║ 智能体数量: {self.agent_num}
║ 测试时长: {self.duration_seconds:.1f} 秒
╠══════════════════════════════════════════════════════════════════╣
║ 【覆盖率指标】
║   - 发现状态数: {self.unique_states} (总访问: {self.total_states_discovered})
║   - 发现动作数: {self.unique_actions} (总执行: {self.total_actions_discovered})
║   - 访问 URL 数: {self.unique_urls} (总访问: {self.total_urls_visited})
╠══════════════════════════════════════════════════════════════════╣
║ 【时间性能】
║   - 总步数: {self.total_steps}
║   - 平均每步时间: {self.avg_step_time_ms:.2f} ms
║   - 最大步时间: {self.max_step_time_ms:.2f} ms
║   - 每秒步数: {self.steps_per_second:.2f}
╠══════════════════════════════════════════════════════════════════╣
║ 【学习性能】
║   - 累计奖励: {self.total_reward:.2f}
║   - 平均奖励/步: {self.avg_reward_per_step:.4f}
║   - 学习更新次数: {self.learning_updates}
║   - 平均损失: {self.avg_loss:.4f}
╠══════════════════════════════════════════════════════════════════╣
║ 【探索效率】
║   - 平均状态新颖度: {self.avg_state_novelty:.4f}
║   - 状态发现速率: {self.unique_states / max(self.duration_seconds, 1) * 60:.2f} 个/分钟
║   - URL 发现速率: {self.unique_urls / max(self.duration_seconds, 1) * 60:.2f} 个/分钟
╠══════════════════════════════════════════════════════════════════╣
║ 【资源使用】
║   - 峰值内存: {self.peak_memory_mb:.1f} MB
║   - 平均内存: {self.avg_memory_mb:.1f} MB
║   - CPU 使用率: {self.cpu_usage_percent:.1f}%
╚══════════════════════════════════════════════════════════════════╝
"""


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
        
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._resource_monitor, daemon=True)
        self._monitor_thread.start()
        logger.info(f"性能监控已启动: {self.metrics.algorithm}")
        
    def stop(self):
        """停止监控"""
        self._stop_monitor.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.metrics.duration_seconds = time.time() - self.start_time
        self._finalize_metrics()
        logger.info(f"性能监控已停止: {self.metrics.algorithm}")
        
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
                    action_hash: int, url: str, novelty: float = 0.0):
        """记录单步执行数据"""
        with self.lock:
            self.step_times.append(step_time * 1000)  # 转换为毫秒
            self.rewards.append(reward)
            self.states_set.add(state_hash)
            self.actions_set.add(action_hash)
            if url:
                self.urls_set.add(url)
            self.novelty_scores.append(novelty)
            
            self.metrics.total_states_discovered += 1
            self.metrics.total_actions_discovered += 1
            self.metrics.total_urls_visited += 1
            
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
    
    def record_learning_update(self, loss: float):
        """记录学习更新"""
        with self.lock:
            self.losses.append(loss)
            self.metrics.learning_updates += 1
    
    def _finalize_metrics(self):
        """计算最终指标"""
        with self.lock:
            # 覆盖率
            self.metrics.unique_states = len(self.states_set)
            self.metrics.unique_actions = len(self.actions_set)
            self.metrics.unique_urls = len(self.urls_set)
            
            # 时间性能
            self.metrics.total_steps = len(self.step_times)
            if self.step_times:
                self.metrics.avg_step_time_ms = sum(self.step_times) / len(self.step_times)
                self.metrics.max_step_time_ms = max(self.step_times)
                self.metrics.min_step_time_ms = min(self.step_times)
            if self.metrics.duration_seconds > 0:
                self.metrics.steps_per_second = self.metrics.total_steps / self.metrics.duration_seconds
            
            # 学习性能
            self.metrics.total_reward = sum(self.rewards)
            if self.rewards:
                self.metrics.avg_reward_per_step = self.metrics.total_reward / len(self.rewards)
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


class BenchmarkRunner:
    """性能测试运行器"""
    
    # 配置名称别名映射（支持简写）
    PROFILE_ALIASES = {
        'shaq': 'github-marl-3h-shaq-5agent',
        'shaq-5': 'github-marl-3h-shaq-5agent',
        'shaq-quick': 'quick-test-shaq',
        'marg': 'github-marl-3h-marg-dql-5agent',
        'marg-dql': 'github-marl-3h-marg-dql-5agent',
        'marg-quick': 'quick-test-mac',
        'qtran': 'github-marl-3h-qtran-5agent',
        'qmix': 'github-marl-3h-qtran-5agent',  # qtran 配置可用于 qmix 对比
        'nndql': 'github-marl-3h-nndql-5agent',
        'nn': 'github-marl-3h-nn-5agent',
    }
    
    def __init__(self, config_path: str = "settings.yaml"):
        self.config_path = config_path
        self.results: Dict[str, PerformanceMetrics] = {}
        
    def resolve_profile_name(self, profile: str) -> str:
        """解析配置名称（支持别名）"""
        # 先检查是否是别名
        if profile.lower() in self.PROFILE_ALIASES:
            resolved = self.PROFILE_ALIASES[profile.lower()]
            logger.info(f"配置别名: {profile} -> {resolved}")
            return resolved
        return profile
        
    def load_config(self, profile: str) -> Dict:
        """加载配置"""
        with open(self.config_path, 'r') as f:
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
        
        if 'shaq' in agent_module.lower() or agent_class.lower() == 'shaq':
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
                      dry_run: bool = False) -> PerformanceMetrics:
        """
        运行单个配置的性能测试
        
        Args:
            profile: 配置名称（支持别名）
            duration: 测试时长（秒）
            dry_run: 是否干跑（不实际启动浏览器）
        """
        # 解析别名
        resolved_profile = self.resolve_profile_name(profile)
        logger.info(f"开始性能测试: {resolved_profile}, 时长: {duration}秒")
        
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
                self._actual_run(resolved_profile, config, monitor, duration)
            except Exception as e:
                logger.error(f"测试过程中出错: {e}")
                traceback.print_exc()
            finally:
                monitor.stop()
        
        self.results[resolved_profile] = monitor.metrics
        return monitor.metrics
    
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
        
        logger.info(f"模拟运行完成: {step} 步")
    
    def _actual_run(self, profile: str, config: Dict, monitor: PerformanceMonitor, 
                    duration: int):
        """实际运行测试"""
        import sys
        import importlib
        from selenium.webdriver.chrome.options import Options
        
        # 临时修改 sys.argv 以避免与 cli_options 的参数解析冲突
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], '--profile', profile]
        
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
            logger.info(f"已覆盖 alive_time: {original_alive_time} -> {duration}秒")
            
            # 创建 Chrome 选项（使用 settings 对象而不是 config 字典）
            chrome_options = Options()
            for arg in settings.browser_arguments:
                chrome_options.add_argument(arg)
            chrome_options.binary_location = settings.browser_path
            
            # 创建测试实例
            webtest = WebtestMultiAgent(chrome_options)
            
            # 注入监控钩子
            original_get_action = webtest.multi_agent_system.get_action
            
            def monitored_get_action(web_state, html, agent_name, url, check_result):
                step_start = time.time()
                try:
                    action = original_get_action(web_state, html, agent_name, url, check_result)
                    step_time = time.time() - step_start
                    
                    # 计算新颖度
                    novelty = 0.0
                    state_dict = webtest.multi_agent_system.state_dict
                    if hasattr(web_state, 'similarity') and len(state_dict) > 1:
                        max_sim = max(
                            (web_state.similarity(s) for s in state_dict if s != web_state),
                            default=0
                        )
                        novelty = 1 - max_sim
                    
                    # 获取奖励 - 使用统一的基于状态新颖度的奖励计算
                    # 这样可以公平比较不同算法的探索效率
                    reward = 0.0
                    algo_class = webtest.multi_agent_system.__class__.__name__
                    try:
                        if algo_class == 'Marg':
                            # Marg 原生使用基于动作执行次数的奖励，为了公平比较
                            # 这里使用与 SHAQ/MargD 相同的基于状态新颖度的奖励计算
                            from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
                            R_A_BASE_HIGH = 50.0
                            R_A_BASE_MIDDLE = 10.0
                            R_A_MIN_SIM_LINE = 0.7
                            R_A_MIDDLE_SIM_LINE = 0.85
                            
                            if isinstance(web_state, ActionSetWithExecutionTimesState):
                                # 计算与已知状态的最大相似度
                                max_sim = -1.0
                                state_dict = webtest.multi_agent_system.state_dict
                                for temp_state in state_dict.keys():
                                    if web_state == temp_state:
                                        continue
                                    if hasattr(web_state, 'similarity'):
                                        sim = web_state.similarity(temp_state)
                                        if sim > max_sim:
                                            max_sim = sim
                                
                                # 基于新颖度的奖励（与 SHAQ/MargD 相同）
                                if max_sim < R_A_MIN_SIM_LINE:
                                    reward = R_A_BASE_HIGH  # 50 分
                                elif max_sim < R_A_MIDDLE_SIM_LINE:
                                    reward = R_A_BASE_MIDDLE  # 10 分
                                else:
                                    # 重复访问的状态，奖励递减
                                    visited_time = state_dict.get(web_state, 0)
                                    if visited_time == 0:
                                        reward = 2.0
                                    else:
                                        reward = 2.0 / float(visited_time)
                        elif hasattr(webtest.multi_agent_system, 'get_reward'):
                            # SHAQ/MargD 风格的调用 (web_state, agent_name)
                            reward = webtest.multi_agent_system.get_reward(web_state, agent_name)
                    except Exception as e:
                        # 如果有任何错误，记录但不中断
                        pass
                    
                    monitor.record_step(
                        step_time=step_time,
                        reward=reward,
                        state_hash=hash(str(web_state)),
                        action_hash=hash(str(action)),
                        url=url,
                        novelty=novelty
                    )
                    
                    return action
                except Exception as e:
                    logger.error(f"监控 get_action 时出错: {e}")
                    raise
            
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
                    logger.info(f"达到测试时长限制 ({duration}秒)，强制停止测试")
                    break
            
            # 强制停止测试
            logger.info(f"停止测试，实际运行时间: {elapsed:.1f}秒")
            webtest.stop()
            webtest.join(timeout=30)
            
            # 如果线程仍在运行，强制终止所有浏览器进程
            if webtest.is_alive():
                logger.warning("测试未能正常停止，强制终止浏览器进程...")
                import subprocess
                try:
                    subprocess.run(['pkill', '-9', '-f', 'Chromium'], 
                                   capture_output=True, timeout=10)
                except Exception as e:
                    logger.warning(f"强制终止浏览器失败: {e}")
            
            logger.info(f"配置 {profile} 测试完成")
        
        finally:
            # 恢复原始 argv
            sys.argv = original_argv
    
    def compare_profiles(self, profiles: List[str], duration: int = 300,
                        dry_run: bool = False) -> Dict[str, PerformanceMetrics]:
        """
        对比多个配置的性能
        
        Args:
            profiles: 配置名称列表
            duration: 每个配置的测试时长
            dry_run: 是否干跑
        """
        # 预加载 Word2Vec 模型，避免在第一个算法测试时花费 45 秒加载
        # 这样可以确保所有算法的测试时间都是公平的
        if not dry_run:
            try:
                logger.info("预加载 Word2Vec 模型（避免影响测试时间）...")
                from transformer.utils.word2vec_cache import preload_word2vec_model, get_cache_info
                preload_word2vec_model()
                cache_info = get_cache_info()
                logger.info(f"Word2Vec 模型已缓存: 加载时间 {cache_info['load_time']:.1f}s, "
                           f"词汇量 {cache_info['vocab_size']}")
            except Exception as e:
                logger.warning(f"预加载 Word2Vec 模型失败（非致命错误）: {e}")
        
        for profile in profiles:
            logger.info(f"\n{'='*60}")
            logger.info(f"测试配置: {profile}")
            logger.info(f"{'='*60}")
            
            try:
                self.run_benchmark(profile, duration, dry_run)
            except Exception as e:
                logger.error(f"配置 {profile} 测试失败: {e}")
                traceback.print_exc()
        
        return self.results
    
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        if not self.results:
            return "没有测试结果"
        
        report = []
        report.append("\n" + "=" * 80)
        report.append(" " * 25 + "多智能体算法性能对比报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 按指标对比
        metrics_to_compare = [
            ("状态覆盖", "unique_states", "个", True),
            ("动作覆盖", "unique_actions", "个", True),
            ("URL 覆盖", "unique_urls", "个", True),
            ("总步数", "total_steps", "步", True),
            ("平均步时间", "avg_step_time_ms", "ms", False),
            ("每秒步数", "steps_per_second", "步/秒", True),
            ("累计奖励", "total_reward", "", True),
            ("平均奖励", "avg_reward_per_step", "", True),
            ("状态新颖度", "avg_state_novelty", "", True),
            ("学习更新", "learning_updates", "次", True),
            ("平均损失", "avg_loss", "", False),
            ("峰值内存", "peak_memory_mb", "MB", False),
        ]
        
        # 表头
        profiles = list(self.results.keys())
        header = f"{'指标':<15} | " + " | ".join(f"{p[:15]:<15}" for p in profiles)
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
        report.append("注: * 标记表示该指标最佳")
        report.append("")
        
        # 综合评分
        report.append("\n【综合评分】(满分100)")
        report.append("-" * 40)
        
        for profile, metrics in self.results.items():
            score = self._calculate_score(metrics)
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            report.append(f"{metrics.algorithm:<10} [{bar}] {score:.1f}")
        
        report.append("")
        return "\n".join(report)
    
    def _calculate_score(self, metrics: PerformanceMetrics) -> float:
        """计算综合评分"""
        score = 0.0
        
        # 状态覆盖 (30分)
        score += min(30, metrics.unique_states / 10 * 30)
        
        # 速度 (20分)
        if metrics.avg_step_time_ms > 0:
            speed_score = min(20, 200 / metrics.avg_step_time_ms)
            score += speed_score
        
        # 奖励 (25分)
        if metrics.avg_reward_per_step > 0:
            score += min(25, metrics.avg_reward_per_step * 5)
        
        # 新颖度 (15分)
        score += metrics.avg_state_novelty * 15
        
        # 资源效率 (10分)
        if metrics.peak_memory_mb > 0:
            mem_score = max(0, 10 - metrics.peak_memory_mb / 100)
            score += mem_score
        
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
        
        logger.info(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="多智能体强化学习算法性能对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试单个配置
  python benchmark.py --profile github-marl-3h-shaq-5agent --duration 300
  
  # 对比多个配置
  python benchmark.py --compare github-marl-3h-shaq-5agent,github-marl-3h-marg-dql-5agent
  
  # 干跑测试（不启动浏览器）
  python benchmark.py --profile quick-test-shaq --duration 60 --dry-run
  
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
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.config)
    
    if args.list_profiles:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\n可用配置:")
        print("-" * 70)
        for name, profile in config.get('profiles', {}).items():
            algo = runner.get_algorithm_name(profile)
            agent_num = profile.get('agent_num', '?')
            # 查找该配置的别名
            aliases = [k for k, v in runner.PROFILE_ALIASES.items() if v == name]
            alias_str = f" (别名: {', '.join(aliases)})" if aliases else ""
            print(f"  {name:<40} [{algo}, {agent_num} agents]{alias_str}")
        
        print("\n可用别名:")
        print("-" * 70)
        for alias, full_name in sorted(runner.PROFILE_ALIASES.items()):
            print(f"  {alias:<15} -> {full_name}")
        print()
        return
    
    if args.compare:
        profiles = [p.strip() for p in args.compare.split(',')]
        runner.compare_profiles(profiles, args.duration, args.dry_run)
    elif args.profile:
        runner.run_benchmark(args.profile, args.duration, args.dry_run)
    else:
        parser.print_help()
        return
    
    # 输出报告
    print(runner.generate_comparison_report())
    
    # 输出每个配置的详细报告
    for profile, metrics in runner.results.items():
        print(metrics.summary())
    
    # 保存结果
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
