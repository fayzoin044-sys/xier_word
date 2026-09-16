import os

MODEL_NAME = "deepseek-v4-flash"
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_BASE_URL = "https://api.deepseek.com"

WEATHER_AGENT_URL = "http://localhost:8001"
RAG_AGENT_URL = "http://localhost:8002"
