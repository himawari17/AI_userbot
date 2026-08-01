import os
import re
import random
import asyncio
import logging

import qrcode
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import MessageEntityMentionName
from pathlib import Path
from memory import (
    apply_participant_summary,
    append_chat_history,
    build_author_label,
    build_model_prompt,
    build_participant_summary_prompt,
    collect_participant_message,
    continues_bot_dialogue,
    format_chat_context,
    load_participant_memory,
    load_recent_chat_history,
    next_participant_batch,
    parse_bot_reply,
    remember_participant,
    save_participant_memory,
)

load_dotenv()

API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
BOT_USERNAME = os.getenv("BOT_USERNAME", "").lstrip("@").lower()  #
RANDOM_REPLY_CHANCE = float(os.getenv("RANDOM_REPLY_CHANCE", "0.3"))
REPLY_DEBOUNCE_SECONDS = float(os.getenv("REPLY_DEBOUNCE_SECONDS", "5"))
PARTICIPANT_MEMORY_FILE = Path(os.getenv("PARTICIPANT_MEMORY_FILE", "data/participant_memory.json"))
CHAT_HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", "data/chat_history"))
CHAT_CONTEXT_MESSAGES = 10
PARTICIPANT_BATCH_SIZE = 10
HISTORY_SNAPSHOT_FILE = Path("history.txt")
MODEL_LOG_FILE = Path(os.getenv("MODEL_LOG_FILE", "data/model.log"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SYSTEM_PROMPT_FILEPATH = os.getenv("SYSTEM_PROMPT", "").strip()

GEMENI_MODEL = "models/gemini-3.5-flash-lite"
MEMORY_MODEL = os.getenv("MEMORY_MODEL", "models/gemini-2.5-flash-lite")

if not API_ID or not API_HASH or not GOOGLE_API_KEY:
    raise RuntimeError("Отсутствуют API_ID, API_HASH или GOOGLE_API_KEY в .env")

def load_system_prompt_from_env() -> str:
    prompt_file = SYSTEM_PROMPT_FILEPATH
    path = Path(prompt_file)
    #print(path)
    if not path.exists():
        raise RuntimeError(f"SYSTEM_PROMPT_FILE указывает на несуществующий файл: {prompt_file}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError(f"Файл промпта пуст: {prompt_file}")
    return prompt
SYSTEM_PROMPT = load_system_prompt_from_env()
#print(SYSTEM_PROMPT)

genai_client = genai.Client(api_key=GOOGLE_API_KEY)

safety_settings = [
    types.SafetySetting(category=category, threshold="BLOCK_NONE")
    for category in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)
log.info("Файл системного промпта: %s", Path(SYSTEM_PROMPT_FILEPATH).name)

MODEL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
model_log = logging.getLogger(f"{__name__}.model")
model_log.setLevel(logging.INFO)
model_log.propagate = False
model_log_handler = logging.FileHandler(MODEL_LOG_FILE, encoding="utf-8")
model_log_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
model_log.addHandler(model_log_handler)

MODEL_LOG_BORDER = "=" * 80
MODEL_LOG_DIVIDER = "-" * 80

my_message_ids = set()
my_user_id = None
participant_memory = load_participant_memory(PARTICIPANT_MEMORY_FILE)
summary_tasks: dict[tuple[int, int], asyncio.Task] = {}
pending_reply_tasks: dict[tuple[int, int], asyncio.Task] = {}


GENERATED_REPLY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "summary": {"type": "STRING"},
    },
    "required": ["answer", "summary"],
}


def log_model_payload(title: str, model_name: str, context: str, payload: str) -> None:
    model_log.info(
        "%s\n%s\nМодель: %s\nКонтекст: %s\n%s\n%s\n%s",
        MODEL_LOG_BORDER,
        title,
        model_name,
        context,
        MODEL_LOG_DIVIDER,
        payload,
        MODEL_LOG_BORDER,
    )


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

async def update_participant_summary(memory_entry: dict) -> None:
    participant_key = (memory_entry.get("chat_id"), memory_entry.get("user_id"))
    try:
        while batch := next_participant_batch(memory_entry, PARTICIPANT_BATCH_SIZE):
            prompt = build_participant_summary_prompt(memory_entry, batch)
            context = (
                f"сводка участника | chat_id={memory_entry.get('chat_id')} | "
                f"user_id={memory_entry.get('user_id')}"
            )
            log_model_payload("ОТПРАВКА МОДЕЛИ", MEMORY_MODEL, context, prompt)
            response = await genai_client.aio.models.generate_content(
                model=MEMORY_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=safety_settings,
                    temperature=0.2,
                    max_output_tokens=500,
                ),
            )
            log_model_payload("ОТВЕТ МОДЕЛИ", MEMORY_MODEL, context, response.text)
            if not apply_participant_summary(memory_entry, batch, response.text):
                return
            save_participant_memory(PARTICIPANT_MEMORY_FILE, participant_memory)
            log.info(
                "Сводка участника обновлена chat_id=%s user_id=%s",
                memory_entry.get("chat_id"),
                memory_entry.get("user_id"),
            )
    except Exception:
        log.exception("Не удалось обновить сводку участника user_id=%s", memory_entry.get("user_id"))
    finally:
        if summary_tasks.get(participant_key) is asyncio.current_task():
            summary_tasks.pop(participant_key, None)


def schedule_participant_summary(memory_entry: dict) -> None:
    if not next_participant_batch(memory_entry, PARTICIPANT_BATCH_SIZE):
        return
    participant_key = (memory_entry.get("chat_id"), memory_entry.get("user_id"))
    task = summary_tasks.get(participant_key)
    if not task or task.done():
        summary_tasks[participant_key] = asyncio.create_task(
            update_participant_summary(memory_entry)
        )


async def generate_reply(chat_id: int, user_text: str, memory_entry: dict) -> tuple[str, str]:
    """Generate the visible answer and its history summary in one request."""
    recent_history = load_recent_chat_history(
        CHAT_HISTORY_DIR, chat_id, limit=CHAT_CONTEXT_MESSAGES
    )
    HISTORY_SNAPSHOT_FILE.write_text(format_chat_context(recent_history), encoding="utf-8")
    prompt = build_model_prompt(recent_history, user_text, memory_entry)
    context = f"ответ в чат | chat_id={chat_id}"
    try:
        log_model_payload(
            "ОТПРАВКА МОДЕЛИ",
            GEMENI_MODEL,
            context,
            prompt,
        )
        response = await genai_client.aio.models.generate_content(
            model=GEMENI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                safety_settings=safety_settings,
                response_mime_type="application/json",
                response_schema=GENERATED_REPLY_SCHEMA,
            ),
        )
        log_model_payload("ОТВЕТ МОДЕЛИ", GEMENI_MODEL, context, response.text)
        return parse_bot_reply(response.text)
    except Exception:
        log.exception("Ошибка Gemini для chat_id=%s", chat_id)
        fallback = "🎁 Ежедневная награда"
        return fallback, fallback


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

    for chat_memory in participant_memory.values():
        for memory_entry in chat_memory.values():
            schedule_participant_summary(memory_entry)

    async def reply_after_pause(event, author_label: str, memory_entry: dict) -> None:
        reply_key = (event.chat_id, event.sender_id)
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(REPLY_DEBOUNCE_SECONDS)
            text = (event.raw_text or "").strip()
            user_text = await build_user_text_with_reply_context(event, text)
            user_text = f"{author_label}: {user_text}"
            log.info(
                "Генерация ответа chat_id=%s msg_id=%s prompt='%s'",
                event.chat_id,
                event.id,
                _shorten_for_log(user_text),
            )
            answer, answer_summary = await generate_reply(
                event.chat_id, user_text, memory_entry
            )
            sent = await event.reply(answer)
            my_message_ids.add(sent.id)
            append_chat_history(
                CHAT_HISTORY_DIR,
                event.chat_id,
                answer_summary,
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
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception(
                "Ошибка отправки ответа chat_id=%s in_reply_to=%s",
                event.chat_id,
                event.id,
            )
        finally:
            if pending_reply_tasks.get(reply_key) is current_task:
                pending_reply_tasks.pop(reply_key, None)

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
        reply_key = (event.chat_id, event.sender_id)
        dialogue_active = continues_bot_dialogue(
            load_recent_chat_history(CHAT_HISTORY_DIR, event.chat_id, limit=2),
            event.sender_id,
        )
        memory_entry = remember_participant(participant_memory, event.chat_id, sender)
        collect_participant_message(memory_entry, text, PARTICIPANT_BATCH_SIZE)
        save_participant_memory(PARTICIPANT_MEMORY_FILE, participant_memory)
        schedule_participant_summary(memory_entry)
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
            pending_task = pending_reply_tasks.pop(reply_key, None)
            if pending_task:
                pending_task.cancel()
            handled = await handle_command(event)
            if handled:
                return

        # 2) триггеры ответа
        mention = is_mention(event, text)
        reply_to_me = await is_reply_to_me(event)
        random_hit = should_random_interject(text)
        pending_task = pending_reply_tasks.get(reply_key)
        dialogue_continuation = event.is_private or dialogue_active or bool(
            pending_task and not pending_task.done()
        )

        if not (mention or reply_to_me or random_hit or dialogue_continuation):
            log.info(
                "Сообщение пропущено chat_id=%s msg_id=%s (mention=%s reply_to_me=%s random_hit=%s dialogue=%s)",
                event.chat_id,
                event.id,
                mention,
                reply_to_me,
                random_hit,
                dialogue_continuation,
            )
            return

        log.info(
            "Триггер ответа chat_id=%s msg_id=%s (mention=%s reply_to_me=%s random_hit=%s dialogue=%s)",
            event.chat_id,
            event.id,
            mention,
            reply_to_me,
            random_hit,
            dialogue_continuation,
        )
        if pending_task and not pending_task.done():
            pending_task.cancel()
        pending_reply_tasks[reply_key] = asyncio.create_task(
            reply_after_pause(event, author_label, memory_entry)
        )

    log.info("Бот запущен. Слушаю сообщения...")
    try:
        await client.run_until_disconnected()
    finally:
        await genai_client.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())
