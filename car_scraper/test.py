import time
import pandas as pd
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# API Configuration
API_KEY = "S711EBOUek2pf145pTwPug==MbebzFBDWwPqNkZK"
API_URL = "https://api.api-ninjas.com/v1/cars"

# Configure Selenium WebDriver (Headless)
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Keywords to filter out auction listings
AUCTION_KEYWORDS = {"auction", "public auction", "auto auction", "dealer auction", "wholesale"}

async def fetch_car_details(session, brand, model):
    """Fetch car details asynchronously from API."""
    params = {"make": brand, "model": model}
    headers = {"X-Api-Key": API_KEY}

    try:
        async with session.get(API_URL, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data[0] if data else {}
    except Exception as e:
        print(f"⚠️ API request failed: {e}")
    return {}

def get_brand_and_model(full_make_model):
    """Extract brand and model from Craigslist listing."""
    words = full_make_model.split()
    return (words[0].title(), " ".join(words[1:])) if len(words) > 1 else (words[0].title(), "Unknown")

def scrape_listing_details(url):
    """Scrape detailed car listing from individual Craigslist pages."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract attributes
        attributes = {}
        attr_groups = soup.find_all("div", class_="attrgroup")
        brand, model, year = "Unknown", "Unknown", "Unknown"

        for group in attr_groups:
            year_tag = group.find("span", class_="valu year")
            if year_tag:
                year = year_tag.text.strip()

            make_model_tag = group.find("span", class_="valu makemodel")
            if make_model_tag:
                brand, model = get_brand_and_model(make_model_tag.text.strip())

        for group in attr_groups:
            for attr in group.find_all("div", class_="attr"):
                label = attr.find("span", class_="labl")
                value = attr.find("span", class_="valu")
                if label and value:
                    attributes[label.text.strip().replace(":", "").lower()] = value.text.strip()

        return {
            "Brand": brand,
            "Model": model,
            "Year": year,
            "Mileage": attributes.get("odometer", "Unknown").replace(",", ""),
            "Condition": attributes.get("condition", "Unknown"),
            "Cylinders": attributes.get("cylinders", "Unknown"),
            "Fuel Type": attributes.get("fuel", "Unknown"),
            "Transmission": attributes.get("transmission", "Unknown"),
            "VIN": attributes.get("vin", "Unknown"),
            "Title Status": attributes.get("title status", "Unknown"),
            "Body Type": attributes.get("type", "Unknown"),
            "Link": url,
        }
    except Exception as e:
        print(f"⚠️ Error scraping details: {e}")
        return {}

def scrape_craigslist(city="losangeles", max_pages=1, max_records=2000):
    """Scrape car listings from Craigslist."""
    base_url = f"https://{city}.craigslist.org/search/cta"
    listings_data = []
    visited_links = set()

    print("🚀 Starting Craigslist Scraper...")

    for page in range(0, max_pages * 120, 120):
        if len(listings_data) >= max_records:
            break

        url = f"{base_url}?s={page}"
        print(f"📄 Scraping page: {url}")

        driver.get(url)
        time.sleep(2)  # Allow page to load

        soup = BeautifulSoup(driver.page_source, "html.parser")
        listings = soup.find_all("div", class_="cl-search-result cl-search-view-mode-gallery")

        for listing in listings:
            if len(listings_data) >= max_records:
                break

            try:
                title_tag = listing.find("a", class_="cl-app-anchor text-only posting-title")
                title = title_tag.text.strip().lower() if title_tag else "Unknown"

                # Skip auction listings
                if any(keyword in title for keyword in AUCTION_KEYWORDS):
                    continue

                price_element = listing.find("span", class_="priceinfo")
                price = price_element.text.strip().replace("$", "").replace(",", "") if price_element else "Unknown"

                link_tag = listing.find("a", href=True)
                link = link_tag["href"] if link_tag else None

                if not link or link in visited_links:
                    continue

                visited_links.add(link)
                listings_data.append({"Title": title, "Price": price, "Link": link})

            except Exception as e:
                print(f"⚠️ Error scraping listing: {e}")

    print(f"✅ Scraped {len(listings_data)} listings. Fetching details in parallel...")

    # Process listings in parallel using multithreading
    with ThreadPoolExecutor(max_workers=10) as executor:
        detailed_data = list(executor.map(scrape_listing_details, [item["Link"] for item in listings_data]))

    # Merge data
    for i, details in enumerate(detailed_data):
        listings_data[i].update(details)

    return listings_data

async def fetch_car_data(cars):
    """Fetch additional car details asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_car_details(session, car["Brand"], car["Model"]) for car in cars]
        details = await asyncio.gather(*tasks)

    for i, detail in enumerate(details):
        if detail:
            cars[i].update(detail)

# Run scraper
car_data = scrape_craigslist()

# Fetch API details asynchronously
asyncio.run(fetch_car_data(car_data))

# Filter out unnecessary columns
desired_columns = [
    "Brand", "Model", "Year", "Mileage", "Condition",
    "Cylinders", "Fuel Type", "Transmission", "VIN",
    "Title Status", "Body Type"
]

df = pd.DataFrame(car_data)[desired_columns]

# Save to CSV
df.to_csv("data/craigslist_cars.csv", index=False)

# Close driver
driver.quit()

print("✅ Scraping complete! Data saved as craigslist_cars.csv")