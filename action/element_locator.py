from enum import Enum
from typing import Optional

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement


class ElementLocator(Enum):
    ID = "id"
    XPATH = "xpath"
    CSS_SELECTOR = "css"
    NAME = "name"
    TAG_NAME = "tagName"
    CLASS_NAME = "className"
    LINK_TEXT = "linkText"
    PARTIAL_LINK_TEXT = "partialLinkText"

    def locate(self, driver: WebDriver, location: str) -> Optional[WebElement]:
        if self == ElementLocator.ID:
            return driver.find_element(By.ID, location)
        elif self == ElementLocator.XPATH:
            return driver.find_element(By.XPATH, location)
        elif self == ElementLocator.CSS_SELECTOR:
            return driver.find_element(By.CSS_SELECTOR, location)
        elif self == ElementLocator.NAME:
            return driver.find_element(By.NAME, location)
        elif self == ElementLocator.TAG_NAME:
            return driver.find_element(By.TAG_NAME, location)
        elif self == ElementLocator.CLASS_NAME:
            return driver.find_element(By.CLASS_NAME, location)
        elif self == ElementLocator.LINK_TEXT:
            return driver.find_element(By.LINK_TEXT, location)
        elif self == ElementLocator.PARTIAL_LINK_TEXT:
            return driver.find_element(By.PARTIAL_LINK_TEXT, location)
        else:
            return None
