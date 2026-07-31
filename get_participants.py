import asyncio
import os
import csv
from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv()

API_ID_RAW = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
TARGET_CHAT = os.getenv("CHAT_ID")
OUTPUT_FILE = "data/participants.csv"

if not API_ID_RAW:
    raise RuntimeError("Missing API_ID in .env")
if not API_HASH:
    raise RuntimeError("Missing API_HASH in .env")
if not TARGET_CHAT:
    raise RuntimeError("Missing CHAT_ID in .env")

API_ID = int(API_ID_RAW)

def parse_target_chat(value: str):
    value = value.strip()
    if value.startswith("@"):
        return value
    try:
        return int(value)
    except ValueError:
        return value

async def main():
    target = parse_target_chat(TARGET_CHAT)

    async with TelegramClient("anon", API_ID, API_HASH) as client:
        entity = await client.get_entity(target)

        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "first_name", "last_name", "full_name", "username"])

            async for user in client.iter_participants(entity):
                first_name = user.first_name or ""
                last_name = user.last_name or ""
                full_name = f"{first_name} {last_name}".strip()
                username = user.username or ""

                writer.writerow([
                    user.id,
                    first_name,
                    last_name,
                    full_name,
                    username
                ])

        print(f"Done. Saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())

