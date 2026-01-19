import logging
import threading
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict

from action.web_action import WebAction
from config.log_config import LogConfig
from exceptions import NoActionsException
from state.web_state import WebState

logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())


class MultiAgentSystem(ABC):
    """
    抽象基类：用于实现多智能体协作系统。
    支持多线程状态管理、动作生成和异常处理。
    """

    def __init__(self, params) -> None:
        super(MultiAgentSystem, self).__init__()
        # 状态字典
        self.prev_state_dict: Dict[str, Optional[WebState]] = {}
        self.prev_action_dict: Dict[str, Optional[WebAction]] = {}
        self.current_state_dict: Dict[str, Optional[WebState]] = {}
        self.current_html_dict: Dict[str, str] = {}
        self.prev_html_dict: Dict[str, str] = {}
        self.action_dict: Dict[WebAction, int] = {}
        self.state_dict: Dict[WebState, int] = {}  # 记录每个状态的访问次数
        self.url_count_dict: Dict[str, int] = {}  # 记录每个URL的访问次数
        self.transition_record_list: List[Tuple[Optional[WebState], WebAction, WebState]] = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # 配置参数
        self.agent_num = params["agent_num"]
        self.url_count_dict[params["entry_url"]] = 9999  # 避免没有 URL 重启

    @abstractmethod
    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        """
        子类必须实现的抽象方法，用于定义动作生成策略。
        """
        pass

    def get_restart_url(self, agent_name: str) -> str:
        """
        获取重启时优先访问的 URL。
        """
        return min(self.url_count_dict, key=self.url_count_dict.get)

    def get_action(self, web_state: WebState, html: str, agent_name: str, url: str, check_result: bool) -> WebAction:
        """
        核心逻辑：生成当前智能体应该执行的动作。

        参数：
        - web_state: 当前的网页状态。
        - html: 当前网页的 HTML。
        - agent_name: 智能体名称。
        - url: 当前页面 URL。
        - check_result: 是否需要检查结果。

        返回：
        - 一个 WebAction 动作实例。
        """
        # 初始化智能体状态信息
        if agent_name not in self.current_state_dict:
            self._initialize_agent_state(agent_name)

        # 获取当前状态的动作列表
        actions = web_state.get_action_list()
        if not actions:
            logger.warning(f"[{agent_name}] No actions available in the current state.")
            self.prev_action_dict[agent_name] = None
            raise NoActionsException("No actions available for the given state.")

        # 状态转移逻辑
        self.transit(web_state, agent_name, url, check_result, html)

        # 通过智能体算法生成动作
        try:
            chosen_action = self.get_action_algorithm(web_state, html, agent_name)
            self.prev_action_dict[agent_name] = chosen_action
            with self.lock:
                self.action_dict[chosen_action] = self.action_dict.get(chosen_action, 0) + 1
            logger.info(f"[{agent_name}] Chosen action: {chosen_action}")
            return chosen_action

        except Exception as e:
            # 发生异常时记录日志并返回空动作防止程序中断
            logger.error(f"[{agent_name}] Error in generating action: {e}")
            raise e  # 根据需要，可选择抛异常或返回默认动作

    def transit(self, new_state: WebState, agent_name: str, url: str, check_result: bool, html: str) -> None:
        """
        状态转移逻辑，用于更新当前状态字典和状态统计。
        """
        actions = new_state.get_action_list()

        with self.lock:
            # 更新动作统计信息
            for action in actions:
                self.action_dict[action] = self.action_dict.get(action, 0)

            # 更新状态统计信息
            if new_state in self.state_dict:
                self.state_dict[new_state] += 1
            else:
                self.state_dict[new_state] = 1

            # 更新智能体的状态和 HTML
            self.prev_state_dict[agent_name] = self.current_state_dict.get(agent_name)
            self.current_state_dict[agent_name] = new_state
            self.prev_html_dict[agent_name] = self.current_html_dict.get(agent_name, "")
            self.current_html_dict[agent_name] = html

            # 更新 URL 访问统计
            if check_result:
                self.url_count_dict[url] = self.url_count_dict.get(url, 0) + 1

            # 记录状态转移记录
            if (self.prev_action_dict.get(agent_name) and
                self.prev_state_dict.get(agent_name) and
                self.current_state_dict.get(agent_name)):
                self.transition_record_list.append((
                    self.prev_state_dict[agent_name],
                    self.prev_action_dict[agent_name],
                    self.current_state_dict[agent_name]
                ))
                logger.debug(f"[{agent_name}] Transition recorded.")

    def restart_fail(self, agent_name: str, restart_url: str) -> None:
        """
        处理重启失败的情况。
        """
        with self.lock:
            self.url_count_dict[restart_url] += 1
            logger.warning(f"[{agent_name}] Restart failed, URL: {restart_url}")

    def deal_exception(self, agent_name: str) -> None:
        """
        处理智能体异常，清理状态。
        """
        logger.error(f"[{agent_name}] Handling exception, resetting state.")
        self._initialize_agent_state(agent_name)

    def get_state(self, web_state: WebState) -> WebState:
        """
        获取当前 WebState 的状态对象。
        """
        with self.lock:
            if web_state not in self.state_dict:
                self.state_dict[web_state] = 0
                return web_state

        # 如果状态已存在，返回匹配的状态
        for state in self.state_dict.keys():
            if state == web_state:
                return state

    def _initialize_agent_state(self, agent_name: str) -> None:
        """
        初始化智能体的状态，防止因未初始化导致 KeyError。
        """
        self.current_state_dict[agent_name] = None
        self.prev_state_dict[agent_name] = None
        self.prev_action_dict[agent_name] = None
        self.current_html_dict[agent_name] = ''
        self.prev_html_dict[agent_name] = ''
        logger.info(f"[{agent_name}] State initialized.")