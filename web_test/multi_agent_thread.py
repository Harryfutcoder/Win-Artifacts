import logging
import os.path
import platform
import threading
import time
from datetime import datetime
from typing import Tuple, Optional, List, Dict
from urllib.parse import urlparse

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from urllib3.exceptions import ReadTimeoutError

from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from action.web_action_detector import WebActionDetector
from config.log_config import LogConfig
from config.settings import settings
from exceptions import NoActionsException
from multi_agent.multi_agent_system import MultiAgentSystem
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name, get_class_by_module_and_class_name

logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())

# Windows 窗口隐藏功能
_hide_windows_stop = False
_hide_thread = None

def _hide_chrome_windows():
    """持续隐藏 Chrome 相关窗口（Windows only）"""
    global _hide_windows_stop
    if platform.system() != 'Windows':
        return
    
    try:
        import win32gui
        import win32con
    except ImportError:
        return
    
    while not _hide_windows_stop:
        def callback(hwnd, _):
            try:
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    class_name = win32gui.GetClassName(hwnd)
                    # 隐藏 Chrome 控制台窗口和浏览器窗口，但不隐藏 Cursor
                    if ('chrome' in title.lower() or class_name == 'Chrome_WidgetWin_1') and 'cursor' not in title.lower():
                        win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
            except:
                pass
            return True
        try:
            win32gui.EnumWindows(callback, None)
        except:
            pass
        time.sleep(0.05)

def start_window_hiding():
    """启动窗口隐藏线程"""
    global _hide_windows_stop, _hide_thread
    if platform.system() != 'Windows':
        return
    _hide_windows_stop = False
    _hide_thread = threading.Thread(target=_hide_chrome_windows, daemon=True)
    _hide_thread.start()
    time.sleep(0.1)  # 等待线程启动

def stop_window_hiding():
    """停止窗口隐藏线程"""
    global _hide_windows_stop
    _hide_windows_stop = True


def enable_performance_logging(options: Options) -> Options:
    """
    启用 Chrome Performance Logging（用于三层奖励系统的伪 ELOC 指标）
    
    这允许获取网络请求、JS 执行等信息，作为代码覆盖率的代理指标。
    基于 ASE 2024 论文：ELOC 比 Activity Coverage 更能预测 Bug。
    """
    # 启用 performance logging
    options.set_capability('goog:loggingPrefs', {
        'browser': 'ALL',
        'performance': 'ALL'
    })
    return options


class MultiAgentThread(threading.Thread):
    def __init__(self, chrome_options: Options, agent_name: str, multi_agent_system: MultiAgentSystem) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.multi_agent_system = multi_agent_system
        
        # 为每个 agent 创建独立的 chrome_options（避免共享 user-data-dir 冲突）
        self.chrome_options = self._create_agent_specific_options(chrome_options)
        
        # 启用 performance logging（用于三层奖励系统）
        self.chrome_options = enable_performance_logging(self.chrome_options)
        
        # 启动窗口隐藏线程（Windows only）
        start_window_hiding()
        
        # 在 Windows 上隐藏 chromedriver 控制台窗口
        if platform.system() == 'Windows':
            from subprocess import CREATE_NO_WINDOW
            service = Service(executable_path=settings.driver_path, creationflags=CREATE_NO_WINDOW)
        else:
            service = Service(executable_path=settings.driver_path)
        
        self.driver = webdriver.Chrome(
            service=service,
            options=self.chrome_options
        )
        logger.info(f"Thread {self.agent_name}: Webdriver created successfully with performance logging (hidden window)")
        
        # 初始化其他属性
        self._init_after_driver()
    
    def _create_agent_specific_options(self, base_options: Options) -> Options:
        """为每个 agent 创建独立的 chrome_options，确保使用独立的 user-data-dir"""
        import random
        
        # 创建新的 Options 对象
        new_options = Options()
        new_options.binary_location = base_options.binary_location
        
        # 添加稳定性参数（防止 Chrome 崩溃）+ headless 模式
        stability_args = [
            '--headless',  # 确保 headless 模式
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-extensions',
            '--disable-software-rasterizer',
            '--window-position=-2400,-2400',  # 窗口移到屏幕外
        ]
        for arg in stability_args:
            new_options.add_argument(arg)
        logger.info(f"Thread {self.agent_name}: Added stability args including --headless")
        
        has_user_data_dir = False
        has_debugging_port = False
        
        # 复制所有参数，但替换 user-data-dir 和 remote-debugging-port
        for arg in base_options.arguments:
            if arg.startswith('--user-data-dir='):
                has_user_data_dir = True
                # 为每个 agent 创建独立的用户数据目录
                agent_data_dir = os.path.join(
                    os.path.abspath(settings.browser_data_path),
                    f"agent_{self.agent_name}"
                )
                os.makedirs(agent_data_dir, exist_ok=True)
                new_options.add_argument(f'--user-data-dir={agent_data_dir}')
                logger.info(f"Thread {self.agent_name}: Using user-data-dir: {agent_data_dir}")
            elif arg.startswith('--remote-debugging-port='):
                has_debugging_port = True
                # 为每个 agent 分配独立的调试端口
                port = 9300 + int(self.agent_name) * 10 + random.randint(0, 9)
                new_options.add_argument(f'--remote-debugging-port={port}')
                logger.info(f"Thread {self.agent_name}: Using debugging port: {port}")
            else:
                new_options.add_argument(arg)
        
        # 如果原配置没有 user-data-dir，也要为每个 agent 添加独立的目录
        if not has_user_data_dir:
            agent_data_dir = os.path.join(
                os.path.abspath(settings.browser_data_path),
                f"agent_{self.agent_name}"
            )
            os.makedirs(agent_data_dir, exist_ok=True)
            new_options.add_argument(f'--user-data-dir={agent_data_dir}')
            logger.info(f"Thread {self.agent_name}: Using user-data-dir: {agent_data_dir}")
        
        # 如果原配置没有调试端口，也分配一个
        if not has_debugging_port:
            port = 9300 + int(self.agent_name) * 10 + random.randint(0, 9)
            new_options.add_argument(f'--remote-debugging-port={port}')
            logger.info(f"Thread {self.agent_name}: Using debugging port: {port}")
        
        # 复制实验性选项（使用正确的方法）
        if hasattr(base_options, '_experimental_options') and base_options._experimental_options:
            for key, value in base_options._experimental_options.items():
                new_options.add_experimental_option(key, value)
        
        return new_options

    def _init_after_driver(self):
        """在 driver 创建后初始化其他属性"""
        self.action_detector_class: type = get_class_by_module_and_class_name(settings.action_detector["module"],
                                                                              settings.action_detector["class"])
        if settings.action_detector["class"] == "CombinationDetector":
            detectors: List[WebActionDetector] = []
            for detector_name in settings.action_detector["detectors"]:
                detectors.append(
                    instantiate_class_by_module_and_class_name(detector_name["module"], detector_name["class"]))
            self.action_detector: WebActionDetector = self.action_detector_class(detectors)
        else:
            self.action_detector: WebActionDetector = self.action_detector_class()
        self.state_class: type = get_class_by_module_and_class_name(settings.state["module"], settings.state["class"])

        self.prev_state: Optional[WebState] = None
        self.current_state: Optional[WebState] = None
        self.action_dict: Dict[WebAction, int] = {}
        self.state_dict: Dict[WebState, int] = {}
        self.transition_record_list: List[Tuple[Optional[WebState], WebAction, WebState]] = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.screen_shot_count = 0
        self.screen_shot_record_round = 50
        self.same_url_count = 0
        self.current_url = "111"
        self.check_result = False
        self.restart_interval = settings.restart_interval

    def run(self):
        logger.info(f"Thread {self.agent_name}: Execution start")
        self.driver.get(settings.entry_url)
        self.stop_event.wait(10)
        html = self.init_state()
        self.driver.set_page_load_timeout(settings.page_load_timeout)
        continuous_restart_count = 0
        while not self.stop_event.is_set():
            try:
                wait = WebDriverWait(self.driver, 10)
                accept_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]")))
                accept_button.click()
                print('accepted cookies')
            except Exception as e:
                print('no cookie button!')
            chosen_action = None
            try:
                chosen_action = self.multi_agent_system.get_action(self.current_state, html, self.agent_name,
                                                                   self.current_url, self.check_result)
                logger.info(f"Thread {self.agent_name}: Chosen action: {chosen_action}")
                if not isinstance(chosen_action, RestartAction):
                    self.action_dict[chosen_action] += 1
                    continuous_restart_count = 0
                else:
                    continuous_restart_count += 1
                if continuous_restart_count >= settings.continuous_restart_threshold:
                    self.restart_webdriver()
                    continuous_restart_count = 0
                chosen_action.execute(self.driver)
                WebDriverWait(self.driver, settings.page_load_timeout).until(
                    lambda x: x.execute_script('return document.readyState') == 'complete'
                )
                self.stop_event.wait(2)
                check_result, domain = self.check_domain()
                self.check_result = check_result
                if check_result and self.same_url_count < self.restart_interval:
                    self.trace_error()
                    action_list = self.action_detector.get_actions(self.driver)
                    with self.lock:
                        for action in action_list:
                            self.action_dict.setdefault(action, 0)
                    if self.driver.current_url == self.current_url:
                        self.same_url_count += 1
                    else:
                        self.same_url_count = 0
                        self.current_url = self.driver.current_url
                    new_state = self.multi_agent_system.get_state(
                        self.state_class(action_list, self.driver.current_url))
                    html = self.driver.page_source
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.info(f"Thread {self.agent_name}: Execute action success, New state: {new_state}")
                elif not check_result:
                    restart_url = self.multi_agent_system.get_restart_url(self.agent_name)
                    new_state = OutOfDomainState(restart_url)
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.warning(f"Thread {self.agent_name}: Out of domain, New state: {new_state}")
                else:
                    self.same_url_count = 0
                    restart_url = self.multi_agent_system.get_restart_url(self.agent_name)
                    new_state = SameUrlState(restart_url)
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.warning(f"Thread {self.agent_name}: Same url too many times, New state: {new_state}")
            except (NoActionsException, WebDriverException, ReadTimeoutError, TimeoutError) as e:
                restart_url = self.multi_agent_system.get_restart_url(self.agent_name)
                new_state = ActionExecuteFailedState(restart_url)
                self.transit(chosen_action, new_state)
                if chosen_action is not None and isinstance(chosen_action, RestartAction):
                    self.multi_agent_system.restart_fail(agent_name=self.agent_name, restart_url=restart_url)
                if isinstance(e, NoActionsException):
                    logger.warning(f"Thread {self.agent_name}: Choose action failed, no actions, New state: {new_state}")
                else:
                    logger.warning(f"Thread {self.agent_name}: Execute action failed ({type(e).__name__}), New state: {new_state}")
            except (Exception, KeyboardInterrupt) as e:
                logger.exception(f"Thread {self.agent_name}: Error happened")
                try:
                    self.restart_webdriver()
                    continuous_restart_count = 0
                    self.driver.get(settings.entry_url)
                    self.stop_event.wait(2)
                    html = self.init_state()
                    self.multi_agent_system.deal_exception(self.agent_name)
                except Exception as restart_error:
                    logger.exception(f"Thread {self.agent_name}: Failed to restart webdriver: {restart_error}")
                    self.stop_event.wait(5)
                    try:
                        self.restart_webdriver()
                        self.driver.get(settings.entry_url)
                        self.stop_event.wait(2)
                        html = self.init_state()
                    except Exception as final_error:
                        logger.exception(f"Thread {self.agent_name}: Final restart attempt failed, exiting: {final_error}")
                        break
            try:
                if len(self.driver.window_handles) > 1:
                    self.driver.switch_to.window(self.driver.window_handles[1])
                    self.close_other_windows()
            except Exception as window_error:
                logger.warning(f"Thread {self.agent_name}: Failed to handle windows: {window_error}")
        try:
            self.driver.quit()
        except Exception as quit_error:
            logger.warning(f"Thread {self.agent_name}: Failed to quit driver: {quit_error}")

    def init_state(self):
        action_list = self.action_detector.get_actions(self.driver)
        self.current_state = self.multi_agent_system.get_state(
            self.state_class(action_list, self.driver.current_url))
        with self.lock:
            for action in action_list:
                self.action_dict.setdefault(action, 0)
            self.state_dict[self.current_state] = 1
        html: str = self.driver.page_source
        browser_logs, performance_logs = self.trace_error()
        logger.info(f"Thread {self.agent_name}: Initial state: {self.current_state}")
        return html

    def trace_error(self) -> Tuple[List[Dict], List[Dict]]:
        """
        获取浏览器日志和性能日志
        
        Returns:
            (browser_logs, performance_logs)
        """
        browser_logs = []
        performance_logs = []
        
        with self.lock:
            # 获取浏览器日志（错误检测）
            try:
                browser_logs = self.driver.get_log("browser")
                with open(os.path.join(settings.output_path, "bug.log"), "a", encoding="utf-8") as f:
                    for log in browser_logs:
                        if (log["level"] == "WARNING") or (log["level"] == "SEVERE"):
                            logger.info(f"Thread {self.agent_name}: Detect browser error: {log}")
                            f.write(str(log) + "\n")
            except Exception as e:
                logger.debug(f"Thread {self.agent_name}: Failed to get browser logs: {e}")
            
            # 获取性能日志（伪 ELOC 指标）
            try:
                performance_logs = self.driver.get_log("performance")
            except Exception as e:
                logger.debug(f"Thread {self.agent_name}: Failed to get performance logs: {e}")
        
        # 存储到 multi_agent_system（供 get_reward 使用）
        if hasattr(self.multi_agent_system, 'set_agent_logs'):
            self.multi_agent_system.set_agent_logs(
                self.agent_name, 
                browser_logs, 
                performance_logs
            )
        
        return browser_logs, performance_logs

    def transit(self, chosen_action: WebAction, new_state: WebState) -> None:
        if chosen_action is not None:
            self.current_state.update_action_execution_time(chosen_action)
            self.current_state.update_transition_information(chosen_action, new_state)
        self.prev_state = self.current_state
        self.current_state = new_state

    def check_domain(self) -> Tuple[bool, str]:
        current_url = self.driver.current_url
        domain = urlparse(current_url).netloc.lower()
        if domain in settings.domains:
            return True, domain
        else:
            return False, domain

    def close_other_windows(self) -> None:
        current_window = self.driver.current_window_handle
        for handle in self.driver.window_handles:
            if handle != current_window:
                self.driver.switch_to.window(handle)
                self.driver.close()
        self.driver.switch_to.window(current_window)

    def restart_webdriver(self) -> None:
        self.driver.quit()
        
        # 启动窗口隐藏线程（Windows only）
        start_window_hiding()
        
        # 在 Windows 上隐藏 chromedriver 控制台窗口
        if platform.system() == 'Windows':
            from subprocess import CREATE_NO_WINDOW
            service = Service(executable_path=settings.driver_path, creationflags=CREATE_NO_WINDOW)
        else:
            service = Service(executable_path=settings.driver_path)
        
        self.driver = webdriver.Chrome(
            service=service,
            options=self.chrome_options
        )
        self.driver.set_page_load_timeout(settings.page_load_timeout)
        logger.info(f"Thread {self.agent_name}: Webdriver restart successfully (hidden window)")

    def stop(self):
        self.stop_event.set()
        try:
            self.driver.quit()
        except Exception as e:
            logger.warning(f"Thread {self.agent_name}: Error closing driver: {e}")

    def save_screen_shot(self):
        if self.screen_shot_count % self.screen_shot_record_round == 0:
            original_window_size = self.driver.get_window_size()
            js = "return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);"
            scroll_height = self.driver.execute_script(js)
            self.driver.set_window_size(original_window_size['width'], scroll_height)
            screenshot = self.driver.get_screenshot_as_png()
            self.driver.set_window_size(original_window_size['width'], original_window_size['height'])
            folder_path = os.path.join(settings.output_path, "ScreenShots/agent-" + self.agent_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path,
                                     "screenShot-" + f"{self.screen_shot_count}" + "---" + datetime.now().strftime(
                                         "%Y-%m-%d_%H_%M_%S") + ".png")
            with open(file_path, "wb") as file:
                file.write(screenshot)
        self.screen_shot_count += 1
