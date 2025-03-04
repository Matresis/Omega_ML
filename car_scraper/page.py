from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode (no UI)

# Fix: Proper way to initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://newyork.craigslist.org/search/cta")
page_source = driver.page_source
driver.quit()

# Save HTML to inspect
with open("html/craigslist_test.html", "w", encoding="utf-8") as file:
    file.write(page_source)

print("Saved Craigslist HTML as craigslist_test.html - Open and check elements!")