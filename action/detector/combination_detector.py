from typing import List

from selenium.webdriver.chrome.webdriver import WebDriver

from action.web_action import WebAction
from action.web_action_detector import WebActionDetector


class CombinationDetector(WebActionDetector):

    def __init__(self, detectors: List[WebActionDetector]):
        self.detectors = detectors

    def get_actions(self, driver: WebDriver) -> List[WebAction]:
        actions = []
        for detector in self.detectors:
            actions.extend(detector.get_actions(driver))
        return actions
