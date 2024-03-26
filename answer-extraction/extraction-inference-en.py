import requests
import dotenv
import os

dotenv.load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/mt5-small"
headers = {"Authorization": f"Bearer {token}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query("hello")

print(output)
