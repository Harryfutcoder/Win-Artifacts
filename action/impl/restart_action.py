from selenium.webdriver.chrome.webdriver import WebDriver

from action.web_action import WebAction


class RestartAction(WebAction):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url

    def execute(self, driver: WebDriver) -> None:
        driver.get(self.url)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RestartAction):
            return self.url == other.url
        return False

    def __hash__(self) -> int:
        return hash(self.url)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, RestartAction):
            return self.url < other.url
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return f'RestartAction(restart_url={self.url})'
