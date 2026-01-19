from config.log_config import LogConfig
from config.settings import settings

LogConfig.init_log_config(settings.output_path)
