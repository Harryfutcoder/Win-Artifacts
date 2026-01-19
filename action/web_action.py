from abc import ABC, abstractmethod
from typing import Optional

from selenium.webdriver.chrome.webdriver import WebDriver

from action.element_locator import ElementLocator


class WebAction(ABC):
    def __init__(self) -> None:
        self.locator: Optional[ElementLocator] = None
        self.location: Optional[str] = None

    @abstractmethod
    def execute(self, driver: WebDriver) -> None:
        pass

    @abstractmethod
    def __lt__(self, other: object) -> bool:
        pass
