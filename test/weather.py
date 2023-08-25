# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:43:47 2023

@author: MaxGr
"""

from bs4 import BeautifulSoup
import requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
 
 
def weather(city):
    city = city.replace(" ", "+")
    res = requests.get(
        f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8', headers=headers)
    print("Searching...\n")
    soup = BeautifulSoup(res.text, 'html.parser')
    location = soup.select('#wob_loc')[0].getText().strip()
    time = soup.select('#wob_dts')[0].getText().strip()
    info = soup.select('#wob_dc')[0].getText().strip()
    weather = soup.select('#wob_tm')[0].getText().strip()
    print(location)
    print(time)
    print(info)
    print(weather+"°C")
 
 
# city = input("Enter the Name of City ->  ")
city = 'Atlanta'
city = city+" weather"
weather(city)
print("Have a Nice Day:)")








import requests

BASE_URL = 'https://www.metaweather.com/api/location/search/'

# Parameters for the API request
city_name = 'Clemson'

# Construct the API URL
url = f'{BASE_URL}?query={city_name}'

try:
    response = requests.get(url)
    data = response.json()
    
    if data:
        woeid = data[0]['woeid']  # Get the WOEID (Where on Earth ID)
        
        # Construct the weather forecast API URL using the obtained WOEID
        forecast_url = f'https://www.metaweather.com/api/location/{woeid}/'
        
        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()
        
        # Print the weather forecast data
        print(forecast_data)
        
        # Process the data as needed
        # ...
    else:
        print("City not found.")
    
except requests.exceptions.RequestException as e:
    print("Error fetching weather data:", e)



















import requests

BASE_URL = 'https://api.weather.gov/'

# Parameters for the API request
location = 'Clemson,SC'  # Replace with the city and state

# Construct the API URL
url = f'{BASE_URL}points/{location}'

try:
    response = requests.get(url)
    data = response.json()
    
    if 'properties' in data:
        forecast_url = data['properties']['forecast']
        
        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()
        
        # Print the weather forecast data
        print(forecast_data)
        
        # Process the data as needed
        # ...
    else:
        print("Location not found.")
    
except requests.exceptions.RequestException as e:
    print("Error fetching weather data:", e)













