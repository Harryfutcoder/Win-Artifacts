from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# 自定义 Chromium 路径和 Chromedriver 路径
chrome_path = "./chrome-mac/Chromium.app/Contents/MacOS/Chromium"
driver_path = "./chromedriver-3"

options = Options()
options.binary_location = chrome_path  # 设置 Chromium 二进制文件路径
options.add_argument("--window-size=1280,1000")  # 设置窗口大小
options.add_argument("--headless")  # 设置无头模式，可注释掉以显示页面

service = Service(driver_path)

try:
    print("尝试启动 WebDriver...")
    driver = webdriver.Chrome(service=service, options=options)
    print("WebDriver 已启动")

    print("尝试访问 https://github.com...")
    driver.get("https://github.com")
    title = driver.title
    print(f"页面已加载，标题为：{title}")

    driver.quit()
    print("WebDriver 测试成功，浏览器已关闭")
except Exception as e:
    print(f"WebDriver 测试失败: {type(e)} - {str(e)}")