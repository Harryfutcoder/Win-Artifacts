import os
from typing import List

from selenium.webdriver.chrome.webdriver import WebDriver

from action.element_locator import ElementLocator
from action.element_text_detect_mode import ElementTextDetectMode
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from action.web_action_detector import WebActionDetector
from config.settings import settings


class RandomSelectActionDetector(WebActionDetector):
    def __init__(self):
        self.js_file_path = os.path.join(settings.resources_path, "js", "action_detector.js")
        self.selectors = ["select"]

    def get_actions(self, driver: WebDriver) -> List[WebAction]:
        web_action_list = []
        with open(self.js_file_path, 'r') as js_file:
            js_code = js_file.read()
        result = driver.execute_script(js_code, self.selectors, ElementTextDetectMode.LABEL.value)
        for web_action_info in result:
            if web_action_info["visible"]:
                web_action_list.append(RandomSelectAction(ElementLocator.XPATH, web_action_info["xpath"], web_action_info["text"]))
        return web_action_list
