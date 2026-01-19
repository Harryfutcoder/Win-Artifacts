import json
import logging
import os
import threading
from datetime import datetime

from action.impl.restart_action import RestartAction
from config import LogConfig
from config.settings import settings
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from web_test.webtest_single_agent import Webtest

logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())


class DataCollector(threading.Thread):
    def __init__(self, webtest: Webtest):
        super().__init__()
        self.output_path = settings.output_path
        self.model_path = settings.model_path
        self.record_interval = settings.record_interval
        self.profile = settings.profile
        self.webtest = webtest
        self.stop_event = threading.Event()
        os.makedirs(os.path.join(self.output_path, "output_data"), exist_ok=True)

    def run(self):
        while not self.stop_event.is_set():
            self.stop_event.wait(self.record_interval)
            if not self.stop_event.is_set():
                self.save_data()

        if self.stop_event.is_set():
            self.webtest.join()
            self.save_data(finish=True)

    def save_data(self, finish=False):
        with self.webtest.lock:
            action_list = sorted(self.webtest.action_dict.keys())
            action_list_with_execution_time = [(str(key), self.webtest.action_dict[key]) for key in action_list]
            state_list = sorted(self.webtest.state_dict.keys())
            state_dict_list = []
            for state in state_list:
                if (not isinstance(state, ActionExecuteFailedState)) and (not isinstance(state, OutOfDomainState)) and (
                        not isinstance(state, SameUrlState)):
                    action_list_in_state = state.get_action_list()
                    action_index_list = []
                    for action in action_list_in_state:
                        action_index_list.append(action_list.index(action))
                    state_dict = {"info": str(state), "action_list": action_index_list,
                                  "visited_time": self.webtest.state_dict[state]}
                    if isinstance(state, ActionSetWithExecutionTimesState):
                        detailed_data_index_dict = {}
                        detailed_data_dict = state.get_action_detailed_data()[0]
                        for action, data_dict in detailed_data_dict.items():
                            action_index = action_list.index(action) if not isinstance(action, RestartAction) else str(
                                action)
                            data_index_dict = {"execution_time": data_dict["execution_time"],
                                               "child_state": state_list.index(data_dict["child_state"]) if
                                               data_dict["child_state"] is not None else None}
                            detailed_data_index_dict[action_index] = data_index_dict
                        state_dict["detailed_data"] = detailed_data_index_dict
                    state_dict_list.append(state_dict)
                else:
                    state_dict_list.append({"info": str(state), "action_list": list(map(str, state.get_action_list())),
                                            "visited_time": self.webtest.state_dict[state]})
            url_count_dict = self.webtest.url_count_dict.copy()
            transition_tuple_list = []
            for transition_record in self.webtest.transition_record_list:
                if transition_record[1] is None:
                    action_index_transition_record = None
                elif isinstance(transition_record[1], RestartAction):
                    action_index_transition_record = str(transition_record[1])
                else:
                    action_index_transition_record = action_list.index(transition_record[1])
                transition_tuple_list.append((state_list.index(transition_record[0]) if transition_record[
                                                                                            0] is not None else None,
                                              action_index_transition_record,
                                              state_list.index(transition_record[2])))

        with open(os.path.join(self.output_path, "output_data",
                               datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ("-finish" if finish else "") + ".json"),
                  "w") as f:
            json.dump(
                {"action_list": action_list_with_execution_time, "state_list": state_dict_list,
                 "url_count": url_count_dict,
                 "transition_list": transition_tuple_list}, f, indent=4, sort_keys=True)

        with open(os.path.join(self.output_path, "output_data", "newest" + ".json"), "w") as f:
            json.dump(
                {"action_list": action_list_with_execution_time, "state_list": state_dict_list,
                 "url_count": url_count_dict,
                 "transition_list": transition_tuple_list}, f, indent=4, sort_keys=True)

        # 暂时不存模型
        # if finish:
        #     if isinstance(self.webtest.agent, DRLagent):
        #         torch.save(self.webtest.agent.q_eval.state_dict(), self.model_path + "/" + self.profile)
        #         torch.save(self.webtest.agent.q_eval.state_dict(), self.output_path + "/" + self.profile)

        logger.info("Data saved successfully")

    def stop(self):
        self.stop_event.set()
