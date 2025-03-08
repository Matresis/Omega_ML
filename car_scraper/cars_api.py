# API Key (Replace with your actual API key)
import requests
#
# API_KEY = "S711EBOUek2pf145pTwPug==MbebzFBDWwPqNkZK"
# API_URL = "https://api.api-ninjas.com/v1/cars"
#
#
# def get_all_brands():
#     """Fetch a list of known car brands from the API."""
#     try:
#         response = requests.get(API_URL, headers={"X-Api-Key": API_KEY})
#         if response.status_code == 200:
#             return {car["make"].lower() for car in response.json()}
#     except Exception as e:
#         print(f"⚠️ Failed to fetch car brands from API: {e}")
#     return set()
#
# KNOWN_BRANDS = get_all_brands()
#
# for car in KNOWN_BRANDS:
#     print(car)

import requests

model = 'mini'
api_url = 'https://api.api-ninjas.com/v1/cars?make={}'.format(model)
response = requests.get(api_url, headers={'X-Api-Key': 'S711EBOUek2pf145pTwPug==MbebzFBDWwPqNkZK'})
if response.status_code == requests.codes.ok:
    print(response.text)
else:
    print("Error:", response.status_code, response.text)