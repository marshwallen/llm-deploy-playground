import requests
import json

url = "http://127.0.0.1:8000/predict"
data = {
    "system_prompt": "You are a helpful assistant.",
    "user_prompt": "Help me with my homework."
    }
response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
print(response.text)