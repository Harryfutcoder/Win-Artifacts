import logging
import threading
import time

from selenium.webdriver.chrome.options import Options

from config import settings
from config.log_config import LogConfig
from multi_agent.multi_agent_system import MultiAgentSystem
from utils import instantiate_class_by_module_and_class_name_and_params
from web_test.multi_agent_thread import MultiAgentThread

logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())


class WebtestMultiAgent(threading.Thread):
    def __init__(self, chrome_options: Options) -> None:
        super().__init__()
        self.agent_num = settings.agent_num
        self.multi_agent_system: MultiAgentSystem = instantiate_class_by_module_and_class_name_and_params(
            settings.agent["module"],
            settings.agent["class"],
            settings.agent["params"])
        self.agent_threads = []
        self.chrome_options = chrome_options
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_agent_threads, daemon=True)

    def _create_agent_thread(self, agent_name: str) -> MultiAgentThread:
        """创建并返回一个新的 MultiAgentThread 实例"""
        return MultiAgentThread(
            chrome_options=self.chrome_options,
            multi_agent_system=self.multi_agent_system,
            agent_name=agent_name
        )

    def _monitor_agent_threads(self):
        """监控线程状态，并在必要时重启线程"""
        while not self.stop_event.is_set():
            for i, agent_thread in enumerate(self.agent_threads):
                if not agent_thread.is_alive():
                    logger.warning(f"Agent thread {i} is not alive. Restarting...")
                    self.agent_threads[i] = self._create_agent_thread(str(i))
                    self.agent_threads[i].start()
            time.sleep(300)  # 每 5 分钟检查一次

    def run(self):
        """启动所有 Agent 线程和监控线程"""
        for i in range(self.agent_num):
            agent_thread = self._create_agent_thread(str(i))
            self.agent_threads.append(agent_thread)
            agent_thread.start()
        self.monitor_thread.start()  # 启动监控线程

    def stop(self):
        """停止所有 Agent 线程和监控线程"""
        self.stop_event.set()  # 设置停止标志
        for agent_thread in self.agent_threads:
            agent_thread.stop()
        self.monitor_thread.join()  # 等待监控线程结束