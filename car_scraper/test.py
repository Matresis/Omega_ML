import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
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

def scrape_craigslist(city="newyork", max_pages=2):
    base_url = f"https://{city}.craigslist.org/search/cta"
    cars = []

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
                link = link_tag["href"] if link_tag else "Unknown"

                # Open details page
                driver.get(link)
                time.sleep(2)
                detail_soup = BeautifulSoup(driver.page_source, "html.parser")

                # Extract attributes from "attrgroup"
                attributes = {}
                attr_groups = detail_soup.find_all("div", class_="attrgroup")

                for group in attr_groups:
                    for attr in group.find_all("div", class_="attr"):
                        label = attr.find("span", class_="labl")
                        value = attr.find("span", class_="valu")
                        if label and value:
                            attributes[label.text.strip().replace(":", "").lower()] = value.text.strip()
                        elif label and attr.find("a"):  # Handle cases where value is inside <a>
                            attributes[label.text.strip().replace(":", "").lower()] = attr.find("a").text.strip()

                # Extract car details with default values if not found
                year = attributes.get("year", title.split()[0])  # If not found, use the title
                brand_model = attributes.get("makemodel", "Unknown")
                mileage = attributes.get("odometer", "Unknown").replace(",", "")
                condition = attributes.get("condition", "Unknown")
                cylinders = attributes.get("cylinders", "Unknown")
                fuel_type = attributes.get("fuel", "Unknown")
                title_status = attributes.get("title status", "Unknown")
                transmission = attributes.get("transmission", "Unknown")
                body_type = attributes.get("type", "Unknown")

                # Extract image URL
                img_element = detail_soup.find("img", src=True)
                img_url = img_element["src"] if img_element else "No image"

                # Store data
                cars.append({
                    "Title": title,
                    "Price": price,
                    "Year": year,
                    "Make & Model": brand_model,
                    "Mileage": mileage,
                    "Condition": condition,
                    "Cylinders": cylinders,
                    "Fuel Type": fuel_type,
                    "Title Status": title_status,
                    "Transmission": transmission,
                    "Body Type": body_type,
                    "Image URL": img_url,
                    "Link": link
                })

            except Exception as e:
                print("Skipping a listing due to error:", e)

    return cars

# Run scraper
cars_data = scrape_craigslist()

# Save to CSV
df = pd.DataFrame(cars_data)
df.to_csv("craigslist_cars.csv", index=False)

# Close driver
driver.quit()

print("✅ Scraping complete! Data saved as craigslist_cars.csv")
