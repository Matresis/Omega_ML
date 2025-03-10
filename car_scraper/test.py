import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# Configure WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Keywords to filter out auction listings
AUCTION_KEYWORDS = {"auction", "public auction", "auto auction", "dealer auction", "wholesale"}

def scrape_craigslist(city="losangeles", max_pages=1, max_records=2000):
    base_url = f"https://{city}.craigslist.org/search/cta"
    cars = []
    visited_links = set()

    print("🚀 Starting Craigslist Scraper...")

    for page in range(0, max_pages * 120, 120):
        if len(cars) >= max_records:
            break

        url = f"{base_url}?s={page}"
        print(f"📄 Scraping page: {url}")

        driver.get(url)
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        listings = soup.find_all("div", class_="cl-search-result cl-search-view-mode-gallery")

        for listing in listings:
            if len(cars) >= max_records:
                break

            try:
                title_tag = listing.find("a", class_="cl-app-anchor text-only posting-title")
                title = title_tag.text.strip().lower() if title_tag else "Unknown"

                # Skip auction listings
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

                print(f"🚗 Scraping car: {title} ({link})")

                # Save current page before navigating
                current_page = driver.current_url

                # Open details page
                driver.get(link)
                time.sleep(1)
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
                            words = full_make_model.split()
                            brand, model = words[0].title(), " ".join(words[1:]) if len(words) > 1 else "Unknown"

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

                print(f"✅ {len(cars)} cars scraped so far...")

                # Return to main page
                driver.get(current_page)
                time.sleep(1)

            except Exception as e:
                print(f"⚠️ Skipping a listing due to error: {e}")

    print(f"🏁 Scraping completed! Total cars scraped: {len(cars)}")
    return cars


# Run scraper
cars_data = scrape_craigslist()

# Save raw data
df = pd.DataFrame(cars_data)
df.to_csv("data/raw_craigslist_cars.csv", index=False)

# Close driver
driver.quit()

print("✅ Scraping complete! Data saved as raw_craigslist_cars.csv")