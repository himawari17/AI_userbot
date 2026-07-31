#
# Код позволяет узнать какие модели gemini доступны для использования
#

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Ошибка: Нет GOOGLE_API_KEY в .env")
    exit()

genai.configure(api_key=api_key)

print("Доступные модели:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
