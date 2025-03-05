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

def scrape_craigslist(city="losangeles", max_pages=2):
    base_url = f"https://{city}.craigslist.org/search/cta"
    cars = []
    visited_links = set()

    for page in range(0, max_pages * 120, 120):
        url = f"{base_url}?s={page}"
        driver.get(url)
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        listings = soup.find_all("div", class_="cl-search-results")

        for listing in listings:
            try:
                # Extract title
                title_tag = listing.find("a", class_="cl-app-anchor text-only posting-title")
                title = title_tag.text.strip() if title_tag else "Unknown"

                # Extract price
                price_element = listing.find("span", class_="priceinfo")
                price = price_element.text.strip().replace("$", "").replace(",", "") if price_element else "Unknown"

                # Extract link
                link_tag = listing.find("a", class_="cl-app-anchor text-only posting-title")
                link = link_tag["href"] if link_tag else None

                if not link or link in visited_links:
                    continue  # Skip duplicate listings

                visited_links.add(link)  # Mark as visited

                # Open details page
                driver.get(link)
                time.sleep(2)
                detail_soup = BeautifulSoup(driver.page_source, "html.parser")

                # Extract attributes from "attrgroup"
                attributes = {}
                attr_groups = detail_soup.find_all("div", class_="attrgroup")

                brand = "Unknown"
                model = "Unknown"
                year = "Unknown"

                for group in attr_groups:
                    # Extract year correctly
                    year_tag = group.find("span", class_="valu year")
                    if year_tag:
                        year = year_tag.text.strip()

                    make_model_tag = group.find("span", class_="valu makemodel")
                    if make_model_tag:
                        make_model_link = make_model_tag.find("a")
                        if make_model_link:
                            full_make_model = make_model_link.text.strip()
                            split_model = full_make_model.split(" ", 1)
                            brand = split_model[0] if len(split_model) > 0 else "Unknown"
                            model = split_model[1] if len(split_model) > 1 else "Unknown"

                # Extract other details
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

            except Exception as e:
                print(f"Skipping a listing due to error: {e}")

    return cars

# Run scraper
cars_data = scrape_craigslist()

# Save to CSV
df = pd.DataFrame(cars_data)
df.to_csv("data/craigslist_cars.csv", index=False)

# Close driver
driver.quit()

print("✅ Scraping complete! Data saved as craigslist_cars.csv")