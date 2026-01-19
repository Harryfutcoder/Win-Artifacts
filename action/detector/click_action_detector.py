import os
from typing import List

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By

from action.element_locator import ElementLocator
from action.element_text_detect_mode import ElementTextDetectMode
from action.impl.click_action import ClickAction
from action.web_action import WebAction
from action.web_action_detector import WebActionDetector
from config.settings import settings


class ClickActionDetector(WebActionDetector):
    def __init__(self):
        self.js_file_path = os.path.join(settings.resources_path, "js", "action_detector.js")
        self.selectors = ["a",
                          "button",
                          "input[type=\"button\"]",
                          "input[type=\"submit\"]",
                          "input[type=\"checkbox\"]",
                          "input[type=\"radio\"]",
                          "input[type=\"image\"]",
                          "summary"]

    def get_actions(self, driver: WebDriver) -> List[WebAction]:
        web_action_list = []
        with open(self.js_file_path, 'r') as js_file:
            js_code = js_file.read()
        result = driver.execute_script(js_code, self.selectors, ElementTextDetectMode.INNER_TEXT.value)
        for web_action_info in result:
            if web_action_info["visible"]:
                web_element = driver.find_element(By.XPATH, web_action_info["xpath"])
                # 检查 href 属性
                href_value = web_element.get_attribute("href")
                # 检查 表单属性
                element_type = web_element.get_attribute("type")

                click_type = "default"
                additionInfo = ""
                # 判断是否是点击跳转
                if href_value:
                    click_type = "redirect"
                    additionInfo = href_value
                # 判断是否是表单
                elif element_type and element_type.lower() == "submit":
                    click_type = "submit"
                    parent_element = web_element
                    while parent_element is not None:
                        parent_element = parent_element.find_element(By.XPATH, '..')
                        if parent_element.tag_name.lower() == 'form':
                            break
                        if parent_element.tag_name.lower() == 'body':
                            parent_element = None
                            break
                    # 统计 <form> 的子元素
                    if parent_element:
                        children_count = len(parent_element.find_elements(By.XPATH, './*'))
                        additionInfo = children_count  # 统计子元素数量
                    else:
                        additionInfo = 0
                else:
                    tag_name = web_element.tag_name.lower()
                    additionInfo = tag_name

                web_action_list.append(ClickAction(ElementLocator.XPATH, web_action_info["xpath"], web_action_info["text"], click_type, additionInfo))
        return web_action_list
