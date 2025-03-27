import time
import pandas as pd
import requests
import configparser
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# Load API configurations from config.ini
config = configparser.ConfigParser()
config.read("C:\SCHOOL\C4c\_Projects\Omega_ML\config.ini")

API_KEY = config.get("API", "API_KEY")
API_URL = config.get("API", "API_URL")

# Configure WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

AUCTION_KEYWORDS = {"auction", "public auction", "auto auction", "dealer auction", "wholesale"}
brand_cache = {}


def get_brand_and_model(full_make_model):
    words = full_make_model.split()
    for i in range(len(words), 0, -1):
        possible_brand = " ".join(words[:i]).lower()
        if possible_brand in brand_cache:
            return brand_cache[possible_brand]
        if check_car_make_exists(possible_brand):
            brand_model = possible_brand.title(), " ".join(words[i:]) if len(words) > i else "Unknown"
            brand_cache[possible_brand] = brand_model
            return brand_model
    brand_model = words[0].title(), " ".join(words[1:]) if len(words) > 1 else "Unknown"
    brand_cache[words[0].lower()] = brand_model
    return brand_model


def check_car_make_exists(brand):
    try:
        response = requests.get(API_URL, headers={"X-Api-Key": API_KEY}, params={"make": brand})
        return response.status_code == 200 and bool(response.json())
    except Exception as e:
        print(f"⚠️ Error checking brand: {e}")
    return False


def scrape_page():
    """Scrape the currently loaded Craigslist page."""
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "cl-search-result")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return soup.find_all("div", class_="cl-search-result cl-search-view-mode-gallery")
    except Exception as e:
        print(f"⚠️ Error scraping page: {e}")
        return []

def scrape_craigslist(city="columbus", max_records=2000):
    base_url = f"https://{city}.craigslist.org/search/cta?purveyor=owner"
    scroll_positions = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000]
    cars = []
    visited_links = set()
    visited_vins = set()

    print("🚀 Starting Craigslist Scraper...")

    for pos in scroll_positions:
        driver.get(f"{base_url}#search=2~gallery~{pos}")
        time.sleep(5)  # Wait for new results
        listings = scrape_page()

        if not listings:
            print(f"⚠️ No listings found at scroll position {pos}.")
            continue

        for listing_index, listing in enumerate(listings):
            if len(cars) >= max_records:
                break

            try:
                title_tag = listing.find("a", class_="cl-app-anchor text-only posting-title")
                title = title_tag.text.strip().lower() if title_tag else "Unknown"
                if any(keyword in title for keyword in AUCTION_KEYWORDS):
                    print(f"⏩ Skipping auction listing: {title}")
                    continue

                price_element = listing.find("span", class_="priceinfo")
                price = price_element.text.strip().replace("$", "").replace(",", "") if price_element else "Unknown"

                link_tag = listing.find("a", href=True)
                link = link_tag["href"] if link_tag else None
                if not link or link in visited_links:
                    continue

                visited_links.add(link)
                print(f"🚗 Scraping car #{len(cars) + 1}: {title} ({link})")

                driver.get(link)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "attrgroup")))
                detail_soup = BeautifulSoup(driver.page_source, "html.parser")

                attributes = {}
                attr_groups = detail_soup.find_all("div", class_="attrgroup")
                brand, model, year = "Unknown", "Unknown", "Unknown"

                for group in attr_groups:
                    year_tag = group.find("span", class_="valu year")
                    if year_tag:
                        year = year_tag.text.strip()

                    make_model_tag = group.find("span", class_="valu makemodel")
                    if make_model_tag:
                        make_model_link = make_model_tag.find("a")
                        if make_model_link:
                            full_make_model = make_model_link.text.strip()
                            brand, model = get_brand_and_model(full_make_model)

                for group in attr_groups:
                    for attr in group.find_all("div", class_="attr"):
                        label = attr.find("span", class_="labl")
                        value = attr.find("span", class_="valu")
                        if label and value:
                            attributes[label.text.strip().replace(":", "").lower()] = value.text.strip()

                mileage = attributes.get("odometer", "Unknown").replace(",", "")
                condition = attributes.get("condition", "Unknown")
                cylinders = attributes.get("cylinders", "Unknown").split()[0]
                cylinders = int(cylinders) if cylinders.isdigit() else "Unknown"
                fuel_type = attributes.get("fuel", "Unknown")
                title_status = attributes.get("title status", "Unknown")
                transmission = attributes.get("transmission", "Unknown")
                vin = attributes.get("vin", "Unknown")
                body_type = attributes.get("type", "Unknown")

                if vin != "Unknown" and vin in visited_vins:
                    print(f"⚠️ Skipping duplicate VIN: {vin}")
                    continue
                visited_vins.add(vin)

                car_data = {
                    "Brand": brand,
                    "Model": model,
                    "Price": price,
                    "Year": year,
                    "Mileage": mileage,
                    "Transmission": transmission,
                    "Body Type": body_type,
                    "Condition": condition,
                    "Cylinders": cylinders,
                    "Fuel Type": fuel_type,
                    "VIN": vin,
                    "Title Status": title_status,
                    "Link": link
                }

                cars.append(car_data)

                # Save data gradually
                df = pd.DataFrame(cars)
                df.to_csv("data/craigslist_cars_col.csv", index=False)
                print(f"📝 Car data saved for {title}")

            except Exception as e:
                print(f"⚠️ Skipping a listing due to error: {e}")

    print(f"🏁 Scraping completed! Total cars scraped: {len(cars)}")
    return cars


# Run scraper
cars_data = scrape_craigslist()

# Close driver
driver.quit()

print("✅ Scraping complete! Data saved as craigslist_cars_la.csv")