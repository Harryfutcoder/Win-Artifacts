from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

chrome_options = Options()
chrome_options.binary_location = "./chrome-mac/Chromium.app/Contents/MacOS/Chromium"
chrome_options.add_argument("--window-size=1280,1000")

driver1 = webdriver.Chrome(
    service=Service(executable_path="./chromedriver-3"),
    options=chrome_options
)
driver2 = webdriver.Chrome(
    service=Service(executable_path="./chromedriver-3"),
    options=chrome_options
)
driver1.get("https://github.com")
driver2.get("https://github.com")
input("按回车退出...")
driver1.quit()
driver2.quit()
