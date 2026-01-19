import random
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)
from config.log_config import LogConfig
logger.addHandler(LogConfig.get_file_handler())

import numpy as np
import torch
from torch import Tensor

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from transformer.utils.generator import embedding
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.web_state import WebState
from transformer.utils.state_analysis import get_state_embedding
from transformer.transformer import Transformer


def load_embedding_model():
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


class TagTransformer(Transformer):
    def __init__(self):
        self.wv_from_bin = load_embedding_model()
        self.state_tensor_table = defaultdict(Tensor)
        self.action_tensor_table = defaultdict(Tensor)

    def action_to_tensor(self, state: WebState, action: WebAction, execution_time=-1):
        action_data, execution_histogram = state.get_action_detailed_data()

        # 检查 action 是否在 action_data 中存在
        if action not in action_data:
            logger.warning(f"Action {action} is not found in action_data. Using default values.")
            details = {"execution_time": 0, "child_state": None}  # 默认执行时间和子状态
        else:
            details = action_data[action]

        if execution_time == -1:  # 如果执行时间未提供，使用 details 中的值
            execution_time = details['execution_time']
        child_state = details['child_state']

        # 子状态的执行时间直方图
        child_array = [0] * 10  # 默认全零
        if isinstance(child_state, ActionSetWithExecutionTimesState):
            child_array = child_state.action_execution_time_histogram

        # 根据动作类型生成张量
        if (isinstance(action, ClickAction) or isinstance(action, RandomSelectAction) or
                isinstance(action, RandomInputAction)):
            text = action.text
            embedding_result = embedding(text, execution_time, child_array, self.wv_from_bin)
        else:  # 对未知动作类型的默认处理
            text_similar = random.uniform(0, 0.5)
            embedding_result = np.concatenate(
                (np.array([text_similar]), np.array([execution_time]), np.array(child_array))
            )

        # 转换成 PyTorch 张量
        tensor = torch.tensor(embedding_result)
        return tensor

    def state_to_tensor(self, state: WebState, html: str):
        if state not in self.state_tensor_table:
            embedding_result = get_state_embedding(html)
            tensor = torch.tensor(embedding_result)
            self.state_tensor_table[state] = tensor
        else:
            tensor = self.state_tensor_table[state]
        return tensor
