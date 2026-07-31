
from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


log = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def chat_history_file(history_dir: Path, chat_id: int) -> Path:
    return history_dir / f"{chat_id}.jsonl"


def load_recent_chat_history(history_dir: Path, chat_id: int, limit: int) -> list[dict[str, Any]]:
    """Return at most *limit* valid messages, in chronological order."""
    if limit <= 0:
        return []

    path = chat_history_file(history_dir, chat_id)
    if not path.exists():
        return []

    messages: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as history_file:
            for line_number, line in enumerate(history_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Пропущена повреждённая строка %s в %s", line_number, path)
                    continue
                if isinstance(entry, dict) and isinstance(entry.get("content"), str):
                    messages.append(entry)
    except OSError as error:
        log.warning("Не удалось прочитать историю чата %s: %s", path, error)
        return []

    return list(messages)


def append_chat_history(
    history_dir: Path,
    chat_id: int,
    content: str,
    author: str,
    *,
    sender_id: Optional[int] = None,
    message_id: Optional[int] = None,
    is_bot: bool = False,
) -> None:
    """Append one visible chat message. The journal is intentionally append-only."""
    content = content.strip()
    if not content:
        return

    path = chat_history_file(history_dir, chat_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "content": content,
        "author": author or "Неизвестный участник",
        "sender_id": sender_id,
        "message_id": message_id,
        "is_bot": is_bot,
        "created_at": utc_now_iso(),
    }
    with path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def format_chat_context(messages: list[dict[str, Any]]) -> str:
    """Produce an unambiguous, chronological transcript for the model."""
    lines = []
    for message in messages:
        author = message.get("author")
        if not isinstance(author, str) or not author.strip():
            # Compatibility with the old role/content history format.
            author = "Бот" if message.get("role") == "assistant" else "Участник"
        content = message.get("content", "").strip()
        if content:
            lines.append(f"{author}: {content}")
    return "\n".join(lines)


def build_model_prompt(messages: list[dict[str, Any]], current_request: str) -> str:
    """Wrap the transcript so it is context, not an instruction from a participant."""
    return (
        "Последние 20 сообщений чата (в хронологическом порядке):\n"
        "---\n"
        f"{format_chat_context(messages)}\n"
        "---\n"
        "Ответь на последнее сообщение с учётом контекста. "
        "Если сообщение является ответом на другое, используй уточнение ниже.\n\n"
        f"Текущее обращение:\n{current_request}"
    )


def load_participant_memory(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        memory = json.loads(path.read_text(encoding="utf-8"))
        return memory if isinstance(memory, dict) else {}
    except (OSError, json.JSONDecodeError) as error:
        log.warning("Не удалось прочитать память участников %s: %s", path, error)
        return {}


def save_participant_memory(path: Path, memory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def remember_participant(memory: dict[str, Any], chat_id: int, sender: Any) -> dict[str, Any]:
    """Update a participant profile without losing previously observed identity data."""
    user_id = getattr(sender, "id", None)
    if user_id is None:
        return {}

    chat_memory = memory.setdefault(str(chat_id), {})
    entry = chat_memory.setdefault(
        str(user_id),
        {
            "user_id": user_id,
            "chat_id": chat_id,
            "username": None,
            "first_name": None,
            "last_name": None,
            "display_name": None,
            "aliases": [],
            "message_count": 0,
            "first_seen_at": utc_now_iso(),
            "last_seen_at": None,
        },
    )

    username = getattr(sender, "username", None)
    first_name = getattr(sender, "first_name", None)
    last_name = getattr(sender, "last_name", None)
    display_name = " ".join(part for part in (first_name, last_name) if part).strip() or None

    # Telegram may omit fields in an update. Preserve the last known value then.
    for field, value in (("username", username), ("first_name", first_name), ("last_name", last_name)):
        if value:
            entry[field] = value
    if display_name:
        entry["display_name"] = display_name

    aliases = set(entry.get("aliases", []))
    for value in (username, first_name, last_name, display_name):
        if value:
            aliases.add(value)
    if username:
        aliases.add(f"@{username}")

    entry["aliases"] = sorted(aliases, key=str.lower)
    entry["message_count"] = int(entry.get("message_count", 0)) + 1
    entry["last_seen_at"] = utc_now_iso()
    return entry


def build_author_label(memory_entry: dict[str, Any], fallback_author: str) -> str:
    if not memory_entry:
        return fallback_author
    return (
        memory_entry.get("display_name")
        or memory_entry.get("username")
        or memory_entry.get("first_name")
        or fallback_author
    )
