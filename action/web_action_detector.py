from abc import ABC, abstractmethod
from typing import List

from selenium.webdriver.chrome.webdriver import WebDriver

from action.web_action import WebAction


class WebActionDetector(ABC):
    @abstractmethod
    def get_actions(self, driver: WebDriver) -> List[WebAction]:
        pass
