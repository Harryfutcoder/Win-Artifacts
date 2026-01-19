print("1. 开始导入...")
import logging
import os
import time

print("2. 导入 selenium...")
from selenium.webdriver.chrome.options import Options

print("3. 导入 settings...")
from config.settings import settings

print("4. Settings 加载成功!")
print(f"   browser_path: {settings.browser_path}")
print(f"   driver_path: {settings.driver_path}")
print(f"   agent_num: {settings.agent_num}")
print(f"   browser_arguments: {settings.browser_arguments}")

print("5. 导入 webtest single...")
from web_test.webtest_single_agent import Webtest

print("6. 配置 chrome options...")
chrome_options = Options()
chrome_options.binary_location = settings.browser_path
for argument in settings.browser_arguments:
    chrome_options.add_argument(argument)

print("7. 创建 Webtest 单 agent...")
webtest = Webtest(chrome_options)

print("8. 启动 webtest...")
webtest.start()

print("9. 等待 10 秒...")
time.sleep(10)

print("10. 停止 ...")
webtest.stop()
print("11. 运行完毕！")
