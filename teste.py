import requests

CHROMA_API_URL = "https://chroma-production-6065.up.railway.app/api/v1/collections/40535baa-0a68-4862-9e4c-1963f4981795"

response = requests.get(CHROMA_API_URL)
print(response.json())
