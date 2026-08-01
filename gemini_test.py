#
# Код позволяет узнать какие модели gemini доступны для использования
#

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Ошибка: Нет GOOGLE_API_KEY в .env")
    exit()

client = genai.Client(api_key=api_key)

print("Доступные модели:")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(f"- {m.name}")
client.close()
