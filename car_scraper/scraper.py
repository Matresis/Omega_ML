import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# API Key (Replace with your actual API key)
API_KEY = "S711EBOUek2pf145pTwPug==MbebzFBDWwPqNkZK"
API_URL = "https://api.api-ninjas.com/v1/cars"

# Configure WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Keywords to filter out auction listings
AUCTION_KEYWORDS = {"auction", "public auction", "auto auction", "dealer auction", "wholesale"}


def get_brand_and_model(full_make_model):
    """Extract the correct brand and model from a Craigslist listing."""
    words = full_make_model.split()

    # Check from longest to shortest if it's a valid brand
    for i in range(len(words), 0, -1):
        possible_brand = " ".join(words[:i]).lower()

        # Check if the brand exists using the API
        if check_car_make_exists(possible_brand):
            return possible_brand.title(), " ".join(words[i:]) if len(words) > i else "Unknown"

    # Default to first word as brand if not found
    return words[0].title(), " ".join(words[1:]) if len(words) > 1 else "Unknown"


def check_car_make_exists(brand):
    """Check if the car make exists in the API."""
    try:
        response = requests.get(API_URL, headers={"X-Api-Key": API_KEY}, params={"make": brand})
        if response.status_code == 200:
            data = response.json()
            return bool(data)  # Returns True if the brand exists and data is returned
    except Exception as e:
        print(f"⚠️ Error checking brand: {e}")
    return False


def fetch_car_details(brand, model):
    """Fetch additional car details from the API."""
    try:
        response = requests.get(
            API_URL,
            headers={"X-Api-Key": API_KEY},
            params={"make": brand, "model": model}
        )
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]  # Return the first matching car
    except Exception as e:
        print(f"⚠️ API request failed: {e}")
    return {}


def scrape_craigslist(city="chicago", max_pages=1, max_records=20):
    base_url = f"https://{city}.craigslist.org/search/cta"
    cars = []
    visited_links = set()
    visited_vins = set()  # Track unique VINs

    print("🚀 Starting Craigslist Scraper...")

    for page in range(0, max_pages * 120, 120):
        if len(cars) >= max_records:
            break

        url = f"{base_url}?s={page}"
        print(f"📄 Scraping page: {url}")

        driver.get(url)
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        listings = soup.find_all("div", class_="cl-search-result cl-search-view-mode-gallery")

        for listing in listings:
            if len(cars) >= max_records:
                break

            try:
                # Extract title
                title_tag = listing.find("a", class_="cl-app-anchor text-only posting-title")
                title = title_tag.text.strip().lower() if title_tag else "Unknown"

                # Skip auction listings
                if any(keyword in title for keyword in AUCTION_KEYWORDS):
                    print(f"⏩ Skipping auction listing: {title}")
                    continue

                # Extract price
                price_element = listing.find("span", class_="priceinfo")
                price = price_element.text.strip().replace("$", "").replace(",", "") if price_element else "Unknown"

                # Extract link
                link_tag = listing.find("a", href=True)
                link = link_tag["href"] if link_tag else None

                if not link or link in visited_links:
                    continue

                visited_links.add(link)
                print(f"🚗 Scraping car: {title} ({link})")

                # Save current page before navigating
                current_page = driver.current_url

                # Open details page
                driver.get(link)
                time.sleep(2)
                detail_soup = BeautifulSoup(driver.page_source, "html.parser")

                # Extract attributes
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

                # Skip duplicate VINs
                if vin != "Unknown" and vin in visited_vins:
                    print(f"⚠️ Skipping duplicate VIN: {vin}")
                    continue
                visited_vins.add(vin)

                # Store data
                cars.append({
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
                })

                #print(f"✅ {len(cars)} cars scraped so far...")

                # Return to main page
                driver.get(current_page)
                time.sleep(1)

            except Exception as e:
                print(f"⚠️ Skipping a listing due to error: {e}")

    print(f"🏁 Scraping completed! Total cars scraped: {len(cars)}")
    return cars


# Run scraper
cars_data = scrape_craigslist()

# Save to CSV
df = pd.DataFrame(cars_data)
df.to_csv("data/craigslist_cars_la.csv", index=False)

# Close driver
driver.quit()

print("✅ Scraping complete! Data saved as craigslist_cars_la.csv")