"""
Хранилище истории диалогов — SQLite-файл вместо словаря в памяти процесса.

Раньше (SESSIONS: Dict[str, List[Messages]] = {} в app.py) история жила
только внутри запущенного Python-процесса:
  - перезапуск backend (деплой, падение, `--reload`) → все истории пропадали;
  - несколько воркеров (uvicorn --workers N) → у каждого воркера была бы
    своя копия SESSIONS, и один и тот же посетитель видел бы разную историю
    в зависимости от того, на какой воркер попал следующий запрос.

SQLite-файл переживает перезапуск процесса и не требует поднимать отдельный
сервис (Redis и т.п.) ради портфолио-проекта на одном процессе. Если проект
вырастет до нескольких воркеров/серверов одновременно — стоит перейти на
Redis; интерфейс load_history/save_history можно оставить таким же,
поменяется только реализация внутри этого файла.

sqlite3 из стандартной библиотеки — синхронный. Для объёма данных чата
(несколько сообщений на сессию) это доли миллисекунды и не создаёт заметной
задержки в async-обработчике FastAPI, поэтому отдельная async-обвязка
(aiosqlite и т.п.) сюда осознанно не добавлялась — лишняя зависимость ради
небольшого выигрыша.

SESSION_TTL_DAYS — та же идея, что была у MAX_SESSIONS в старом app.py
(ограничить рост числа хранимых сессий), только не "не больше N сессий
одновременно в памяти", а "не храним то, чем не пользовались дольше N дней".
Чистка запускается не при каждом запросе (лишний DELETE на каждый чат ни к
чему), а раз в CLEANUP_INTERVAL_SECONDS.
"""

import json
import os
import sqlite3
import time
from typing import Dict, List

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "sessions.db")

SESSION_TTL_DAYS = 30
CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60  # раз в 6 часов, не на каждый запрос

_connection: sqlite3.Connection | None = None
_last_cleanup: float = 0.0


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is not None:
        return _connection

    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            history_json TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    _connection = conn
    return conn


def _maybe_cleanup(conn: sqlite3.Connection) -> None:
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < CLEANUP_INTERVAL_SECONDS:
        return
    cutoff = now - SESSION_TTL_DAYS * 24 * 60 * 60
    conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
    conn.commit()
    _last_cleanup = now


def load_history(session_id: str) -> List[Dict[str, str]]:
    """Список {"role": ..., "content": ...} для сессии (пустой список, если её ещё нет)."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT history_json FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not row:
        return []
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        # повреждённая запись — не роняем чат, начинаем историю заново
        return []


def save_history(session_id: str, history: List[Dict[str, str]]) -> None:
    """Полностью перезаписывает историю сессии (она и так обрезается по MAX_HISTORY_MESSAGES)."""
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO sessions (session_id, history_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            history_json = excluded.history_json,
            updated_at = excluded.updated_at
        """,
        (session_id, json.dumps(history, ensure_ascii=False), time.time()),
    )
    conn.commit()
    _maybe_cleanup(conn)
