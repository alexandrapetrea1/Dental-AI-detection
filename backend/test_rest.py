import requests
import json
import os
from dotenv import load_dotenv

load_dotenv("/Users/alexandrapetrea/Desktop/licenta/backend/.env")
api_key = os.environ.get("GEMINI_API_KEY", "").strip()

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
payload = {
    "contents": [{"parts": [{"text": "Hello"}]}]
}
try:
    response = requests.post(url, json=payload, timeout=10)
    print("STATUS:", response.status_code)
    print("BODY:", response.text)
except Exception as e:
    print("REST ERROR:", str(e))
