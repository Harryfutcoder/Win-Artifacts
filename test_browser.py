from selenium import webdriver
from selenium.webdriver. chrome.service import Service
from selenium.webdriver.chrome.options import Options

options = Options()
options.binary_location = "./chrome-mac/Chromium.app/Contents/MacOS/Chromium"
options.add_argument("--window-size=1280,1000")

service = Service(executable_path="./chromedriver-3")

print("正在启动浏览器...")
driver = webdriver.Chrome(service=service, options=options)
print("浏览器已启动！")

driver.get("https://github.com")
print("已打开 GitHub")
print("页面标题:", driver.title)

input("按回车关闭浏览器...")
driver.quit()
