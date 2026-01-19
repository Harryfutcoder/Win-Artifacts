import os.path
import random
import sys
import time
import yaml
from config.cli_options import cli_options

class Settings:
    def __init__(self) -> None:
        self.settings_path = cli_options.settings  # YAML 文件路径
        self.output_path = cli_options.output
        self.model_path = cli_options.model_path
        self.restart_interval = cli_options.restart_interval
        self.continuous_restart_threshold = cli_options.continuous_restart_threshold
        self.enable_screen_shot = cli_options.enable_screen_shot
        self.profile = cli_options.profile
        self.session = cli_options.session
        self.agent_num = cli_options.agent_num
        self.record_interval = None
        self.alive_time = None
        self.page_load_timeout = None
        self.browser_path = None
        self.browser_data_path = None
        self.driver_path = None
        self.resources_path = None
        self.entry_url = None
        self.domains = None
        self.browser_arguments = None
        self.action_detector = None
        self.state = None
        self.agent = None

    def load_settings(self) -> None:
        # 检查 YAML 路径是否正确
        print(f"[DEBUG] 配置文件路径: {self.settings_path}")
        if not os.path.exists(self.settings_path):
            raise FileNotFoundError(f"[ERROR] 配置文件未找到: {self.settings_path}")

        try:
            # 加载 YAML 文件
            with open(self.settings_path, 'r') as f:
                settings_data = yaml.safe_load(f)
                print("[DEBUG] 配置文件成功加载:")
                print(settings_data)

                # 加载默认值
                self.output_path = self.output_path or settings_data['default_output_path']
                self.profile = self.profile or settings_data['default_profile']
                self.session = self.session or settings_data['default_session']
                self.model_path = self.model_path or settings_data['default_model_path']
                self.restart_interval = self.restart_interval or settings_data['default_restart_interval']
                self.continuous_restart_threshold = (
                    self.continuous_restart_threshold or settings_data['default_continuous_restart_threshold']
                )
                self.enable_screen_shot = self.enable_screen_shot or settings_data['default_enable_screen_shot']

                print(f"[DEBUG] 使用的 Profile: {self.profile}, Session: {self.session}")

                # 检查是否存在指定的 Profile
                if self.profile not in settings_data['profiles']:
                    print(f"[ERROR] Profile \"{self.profile}\" 不存在", file=sys.stderr)
                    sys.exit(1)

                # 生成输出路径
                self.generate_output_path(settings_data)

                # 加载 Profile 相关配置
                profile_data = settings_data['profiles'][self.profile]
                self.agent_num = profile_data['agent_num']
                self.record_interval = profile_data['record_interval']
                self.alive_time = profile_data['alive_time']
                self.page_load_timeout = profile_data['page_load_timeout']
                self.browser_path = profile_data['browser_path']
                self.browser_data_path = profile_data['browser_data_path']
                self.driver_path = profile_data['driver_path']
                self.resources_path = profile_data['resources_path']
                self.entry_url = profile_data['entry_url']
                self.domains = profile_data['domains']
                self.browser_arguments = profile_data['browser_arguments']
                self.action_detector = profile_data['action_detector']
                self.state = profile_data['state']
                self.agent = profile_data['agent']

                # 根据不同的 Agent 类型加载额外的 CLI Options
                if self.agent['module'] == "agent.impl.drl_agent" and self.agent['class'] == "DRLagent":
                    self.load_drl_agent_cli_options()
                elif self.agent['module'] == "agent.impl.q_learning_agent" and self.agent['class'] == "QLearningAgent":
                    self.load_q_learning_agent_cli_options()
                else:
                    self.load_multi_agent_cli_options()

        except Exception as e:
            print(f"[ERROR] 加载配置文件失败: {e}")
            raise e

    def generate_output_path(self, settings_data) -> None:
        """
        生成唯一的输出路径并确保不会冲突
        """
        MAX_RETRIES = 10
        attempts = 0

        if self.session == settings_data['default_session']:
            folder_name = self.profile + "-" + self.session + "-" + time.strftime("%Y%m%d-%H%M%S")
        else:
            folder_name = self.profile + "-" + self.session

        output_path = os.path.join(self.output_path, folder_name)

        while attempts < MAX_RETRIES:
            if os.path.exists(output_path):
                print(f"[WARNING] 输出路径已存在: {output_path}")
                # 如果路径已存在，生成新路径
                folder_name = self.profile + "-" + self.session + "-" + time.strftime("%Y%m%d-%H%M%S")
                output_path = os.path.join(self.output_path, folder_name)
                print(f"[DEBUG] 尝试生成新路径: {output_path}")
            else:
                print(f"[DEBUG] 创建新路径: {output_path}")
                os.makedirs(output_path, exist_ok=True)  # 安全创建路径
                break

            attempts += 1

        if attempts == MAX_RETRIES:
            raise RuntimeError(f"[ERROR] 输出路径解析失败，重试次数超过限制：{output_path}")

        print(f"[DEBUG] 最终输出路径: {output_path}")
        self.output_path = output_path

    def load_multi_agent_cli_options(self) -> None:
        # 加载多 Agent 的额外 CLI 选项
        print("[DEBUG] 加载 Multi-Agent CLI Options")
        self.agent["params"]["alive_time"] = self.alive_time
        self.agent["params"]["agent_num"] = self.agent_num
        self.agent["params"]["entry_url"] = self.entry_url
        # 加载可选参数
        if cli_options.model_module is not None:
            self.agent["params"]["model_module"] = cli_options.model_module
        if cli_options.model_class is not None:
            self.agent["params"]["model_class"] = cli_options.model_class
        if cli_options.model_load_type is not None:
            self.agent["params"]["model_load_type"] = cli_options.model_load_type
        if cli_options.model_load_name is not None:
            self.agent["params"]["model_load_name"] = cli_options.model_load_name
        if cli_options.transformer_module is not None:
            self.agent["params"]["transformer_module"] = cli_options.transformer_module
        if cli_options.transformer_class is not None:
            self.agent["params"]["transformer_class"] = cli_options.transformer_class
        if cli_options.reward_function is not None:
            self.agent["params"]["reward_function"] = cli_options.reward_function

settings = Settings()
settings.load_settings()