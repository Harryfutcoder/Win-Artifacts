from selenium.webdriver.chrome.webdriver import WebDriver

from action.element_locator import ElementLocator
from action.web_action import WebAction


class ClickAction(WebAction):
    def __init__(self, locator: ElementLocator, location: str, text: str, action_type: str, addition_info: str ) -> None:
        super().__init__()
        self.locator = locator
        self.location = location
        self.text = text
        self.action_type = action_type
        self.addition_info = addition_info

    def execute(self, driver: WebDriver) -> None:
        web_element = self.locator.locate(driver, self.location)
        web_element.click()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ClickAction):
            return (self.locator == other.locator) and (self.location == other.location) and (self.text == other.text)
        return False

    def __hash__(self) -> int:
        return hash((self.locator, self.location, self.text))

    def __lt__(self, other: object) -> bool:
        if isinstance(other, ClickAction):
            return (self.locator.value + self.location + self.text) < (other.locator.value + other.location + other.text)
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return f'ClickAction(locator={self.locator}, location={self.location}, text={self.text})'
