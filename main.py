import os
import re
import random
import asyncio
import logging

import qrcode
from dotenv import load_dotenv
import google.generativeai as genai 
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import MessageEntityMentionName
from pathlib import Path
from memory import (
    append_chat_history,
    build_author_label,
    build_model_prompt,
    format_chat_context,
    load_participant_memory,
    load_recent_chat_history,
    remember_participant,
    save_participant_memory,
)
load_dotenv()

# =========================
# CONFIG
# =========================
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
BOT_USERNAME = os.getenv("BOT_USERNAME", "").lstrip("@").lower()  #
RANDOM_REPLY_CHANCE = float(os.getenv("RANDOM_REPLY_CHANCE", "0.3"))
PARTICIPANT_MEMORY_FILE = Path(os.getenv("PARTICIPANT_MEMORY_FILE", "data/participant_memory.json"))
CHAT_HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", "data/chat_history"))
CHAT_CONTEXT_MESSAGES = 20
HISTORY_SNAPSHOT_FILE = Path("history.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

GEMENI_MODEL = "models/gemini-3.5-flash-lite"

if not API_ID or not API_HASH or not GOOGLE_API_KEY:
    raise RuntimeError("Отсутствуют API_ID, API_HASH или GOOGLE_API_KEY в .env")

def load_system_prompt_from_env() -> str:
    prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "prompt.txt").strip()
    path = Path(prompt_file)
    if not path.exists():
        raise RuntimeError(f"SYSTEM_PROMPT_FILE указывает на несуществующий файл: {prompt_file}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError(f"Файл промпта пуст: {prompt_file}")
    return prompt
SYSTEM_PROMPT = load_system_prompt_from_env()

genai.configure(api_key=GOOGLE_API_KEY)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

model = genai.GenerativeModel(
    model_name=GEMENI_MODEL,
    system_instruction=SYSTEM_PROMPT,
    safety_settings=safety_settings
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

my_message_ids = set()
my_user_id = None
participant_memory = load_participant_memory(PARTICIPANT_MEMORY_FILE)


def is_command(text: str) -> bool:
    return bool(text and text.strip().startswith("/"))


def is_mention(event, text: str) -> bool:
    if not text and not getattr(event, "message", None):
        return False

    if text and BOT_USERNAME and f"@{BOT_USERNAME}" in text.lower():
        return True

    message = getattr(event, "message", None)
    if not message:
        return False

    if getattr(message, "mentioned", False):
        return True

    entities = getattr(message, "entities", None) or []
    for entity in entities:
        if isinstance(entity, MessageEntityMentionName) and getattr(entity, "user_id", None) == my_user_id:
            return True

    return False

async def build_user_text_with_reply_context(event, current_text: str) -> str:
    user_text = strip_bot_mention(current_text).strip()

    if event.is_reply:
        try:
            replied = await event.get_reply_message()
            if replied and (replied.raw_text or "").strip():
                replied_text = replied.raw_text.strip()
                user_text = (
                    f"Сообщение, на которое отвечают:\n{replied_text}\n\n"
                    f"Вопрос пользователя:\n{user_text}"
                )
        except Exception as e:
            log.warning("Не удалось получить reply-контекст: %s", e)

    return user_text


async def is_reply_to_me(event) -> bool:
    """
    True, если сообщение является reply на сообщение текущего клиента.
    """
    if not event.is_reply:
        return False

    try:
        replied = await event.get_reply_message()
        if not replied:
            return False
        return replied.out is True or (replied.sender_id == my_user_id) or (replied.id in my_message_ids)
    except Exception:
        return False


def should_random_interject(text: str) -> bool:
    if not text or is_command(text):
        return False
    return random.random() < RANDOM_REPLY_CHANCE

def strip_bot_mention(text: str) -> str:
    if not text:
        return ""
    return re.sub(rf"@{re.escape(BOT_USERNAME)}\b", "", text, flags=re.IGNORECASE).strip()


def _shorten_for_log(text: str, max_len: int = 160) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

async def generate_reply(chat_id: int, user_text: str) -> str:
    """Ask Gemini with the last 20 visible messages of this exact chat."""
    recent_history = load_recent_chat_history(
        CHAT_HISTORY_DIR, chat_id, limit=CHAT_CONTEXT_MESSAGES
    )
    HISTORY_SNAPSHOT_FILE.write_text(format_chat_context(recent_history), encoding="utf-8")
    prompt = build_model_prompt(recent_history, user_text)
    try:
        response = await model.generate_content_async(prompt)
        answer = response.text.strip()
    except Exception:
        log.exception("Ошибка Gemini для chat_id=%s", chat_id)
        return "🎁 Ежедневная награда"
    return answer

async def cmd_help(event):
    text = (
        "Доступные команды:\n"
        "/help — список команд\n"
        "/ping — проверка, жив ли бот\n"
        "/stats — заглушка под статистику"
    )
    msg = await event.reply(text)
    my_message_ids.add(msg.id)
    append_chat_history(CHAT_HISTORY_DIR, event.chat_id, text, "Бот", message_id=msg.id, is_bot=True)
    log.info("Команда /help: ответ отправлен chat_id=%s msg_id=%s", event.chat_id, msg.id)


async def cmd_ping(event):
    msg = await event.reply("/pidor🏓")
    my_message_ids.add(msg.id)
    append_chat_history(CHAT_HISTORY_DIR, event.chat_id, "/pidor🏓", "Бот", message_id=msg.id, is_bot=True)
    log.info("Команда /ping: ответ отправлен chat_id=%s msg_id=%s", event.chat_id, msg.id)

async def cmd_stats(event):
    # TODO: реализовать сбор статистики чата
    msg = await event.reply("pultim")
    my_message_ids.add(msg.id)
    append_chat_history(CHAT_HISTORY_DIR, event.chat_id, "pultim", "Бот", message_id=msg.id, is_bot=True)
    log.info("Команда /stats: ответ отправлен chat_id=%s msg_id=%s", event.chat_id, msg.id)

"""
Обработчик команд
"""
async def handle_command(event):
    text = (event.raw_text or "").strip()
    cmd = text.split()[0].lower()
    log.info(
        "Получена команда chat_id=%s sender_id=%s msg_id=%s cmd=%s",
        event.chat_id,
        event.sender_id,
        event.id,
        cmd,
    )

    if cmd.startswith("/help"):
        await cmd_help(event)
        return True
    if cmd.startswith("/ping"):
        await cmd_ping(event)
        return True
    if cmd.startswith("/stats"):
        await cmd_stats(event)
        return True

    return False

async def main():
    global my_user_id, BOT_USERNAME

    client = TelegramClient("anon", API_ID, API_HASH)
    await client.connect()

    if not await client.is_user_authorized():
        qr = await client.qr_login()
        print("\nСканируй через Telegram: Settings → Devices → Link Desktop Device\n")

        qr_code = qrcode.QRCode(border=2)
        qr_code.add_data(qr.url)
        qr_code.make(fit=True)
        qr_code.print_ascii(invert=True)

        try:
            await qr.wait(timeout=120)
        except asyncio.TimeoutError:
            print("QR истёк, обновляю...\n")
            await qr.recreate()
            qr_code = qrcode.QRCode(border=2)
            qr_code.add_data(qr.url)
            qr_code.make(fit=True)
            qr_code.print_ascii(invert=True)
            await qr.wait(timeout=120)
        except SessionPasswordNeededError:
            password = input("Пароль 2FA: ")
            await client.sign_in(password=password)

    me = await client.get_me()
    my_user_id = me.id
    if not BOT_USERNAME and getattr(me, "username", None):
        BOT_USERNAME = me.username.lower()
    log.info("Успешный вход: %s (%s)", me.username, me.id)
    log.info("BOT_USERNAME=%s", BOT_USERNAME or "<empty>")

    @client.on(events.NewMessage(incoming=True))
    async def on_message(event):

        text = (event.raw_text or "").strip()
        if not text:
            return

        log.info(
            "Входящее сообщение chat_id=%s sender_id=%s msg_id=%s text='%s'",
            event.chat_id,
            event.sender_id,
            event.id,
            _shorten_for_log(text),
        )

        sender = await event.get_sender()
        username = getattr(sender, "username", None)
        first_name = getattr(sender, "first_name", None)
        author = username or first_name or f"id:{event.sender_id}"
        memory_entry = remember_participant(participant_memory, event.chat_id, sender)
        save_participant_memory(PARTICIPANT_MEMORY_FILE, participant_memory)
        author_label = build_author_label(memory_entry, author)
        append_chat_history(
            CHAT_HISTORY_DIR,
            event.chat_id,
            text,
            author_label,
            sender_id=event.sender_id,
            message_id=event.id,
        )

        # 1) команды
        if text.startswith("/"):
            handled = await handle_command(event)
            if handled:
                return

        # 2) триггеры ответа
        mention = is_mention(event, text)
        reply_to_me = await is_reply_to_me(event)
        random_hit = should_random_interject(text)

        if not (mention or reply_to_me or random_hit):
            log.info(
                "Сообщение пропущено chat_id=%s msg_id=%s (mention=%s reply_to_me=%s random_hit=%s)",
                event.chat_id,
                event.id,
                mention,
                reply_to_me,
                random_hit,
            )
            return

        log.info(
            "Триггер ответа chat_id=%s msg_id=%s (mention=%s reply_to_me=%s random_hit=%s)",
            event.chat_id,
            event.id,
            mention,
            reply_to_me,
            random_hit,
        )
        
        if mention:
            user_text = await build_user_text_with_reply_context(event, text)
        else:
            user_text=text
        user_text = f"{author_label}: {user_text}"
        log.info(
            "Генерация ответа chat_id=%s msg_id=%s prompt='%s'",
            event.chat_id,
            event.id,
            _shorten_for_log(user_text),
        )

        try:
            answer = await generate_reply(event.chat_id, user_text)
            sent = await event.reply(answer)
            my_message_ids.add(sent.id)
            append_chat_history(
                CHAT_HISTORY_DIR,
                event.chat_id,
                answer,
                "Бот",
                sender_id=my_user_id,
                message_id=sent.id,
                is_bot=True,
            )
            log.info(
                "Ответ отправлен chat_id=%s in_reply_to=%s sent_msg_id=%s text='%s'",
                event.chat_id,
                event.id,
                sent.id,
                _shorten_for_log(answer),
            )
        except Exception:
            log.exception(
                "Ошибка отправки ответа chat_id=%s in_reply_to=%s",
                event.chat_id,
                event.id,
            )

    log.info("Бот запущен. Слушаю сообщения...")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
