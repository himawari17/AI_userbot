import asyncio
import os
import csv
from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv()

API_ID = int(os.getenv("API_ID", "0"))
if not API_ID:
    raise RuntimeError("Missing API_ID in .env")
API_HASH = os.getenv("API_HASH", "")
if not API_HASH:
    raise RuntimeError("Missing API_HASH in .env")

OUTPUT_FILE = "data/chats.csv"


async def main():
    async with TelegramClient("anon", API_ID, API_HASH) as client:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "id", "is_group", "is_channel", "is_user"])

            async for dialog in client.iter_dialogs():
                writer.writerow([
                    dialog.name,
                    dialog.id,
                    dialog.is_group,
                    dialog.is_channel,
                    dialog.is_user,
                ])

        print(f"Done. Saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
