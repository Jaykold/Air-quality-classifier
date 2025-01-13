import requests

url = 'http://localhost:8000/predict'

data = {
    "temperature": 30.4,
    "humidity": 98.0,
    "pm2.5": 24.9,
    "no2": 23.5,
    "so2": 21.2,
    "co": 1.55,
    "proximity_to_industrial_areas": 6.7,
    "population_density": 594.0
}

try:
    response = requests.post(url, json=data)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")