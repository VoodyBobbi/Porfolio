import logging
import os
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pydantic import BaseModel, Field

from .history_store import load_history, save_history
from .rag_index import CHROMA_DIR, COLLECTION_NAME, TOKEN_RE, create_embedding_function, load_index, search_similar


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("faq_assistant")

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
# По документации GigaChat API (developers.sber.ru/docs/ru/gigachat/guides/
# selecting-a-model): в поле model допустимые значения — "GigaChat-2",
# "GigaChat-2-Pro", "GigaChat-2-Max", "GigaChat-3-Ultra". Самый быстрый и
# дешёвый — именно "GigaChat-2" (без суффикса) — это то же самое, что в
# прайс-листе называется "GigaChat 2 Lite" (это маркетинговое название, не
# значение для API). "GigaChat-2-Lite" как строка для API не существует.
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2")
# Не берём из .env: адрес уже есть в подвале сайта (frontend/index.html) и
# тут, в коде. Если понадобится сменить — правьте здесь, руками, осознанно.
IVAN_TELEGRAM = "https://t.me/Ivan_Paro"

giga = None
if GIGACHAT_CREDENTIALS:
    giga = GigaChat(credentials=GIGACHAT_CREDENTIALS, model=GIGACHAT_MODEL, verify_ssl_certs=False)
else:
    logger.warning("GIGACHAT_CREDENTIALS не задан — бот будет отвечать только по FAQ, без LLM.")
embedding_function = create_embedding_function()

app = FastAPI(title="FAQ RAG Assistant")

# CORS: явный список разрешённых доменов из .env, а не "*". "*" разрешает
# дёргать /chat с любого чужого сайта; виджет не использует cookies, поэтому
# allow_credentials=False (сочетание "*" + allow_credentials=True вообще
# запрещено спецификацией CORS и браузер бы такой ответ всё равно отклонил).
# Локально (открывая frontend/index.html через простой http-сервер, см.
# README) страница обычно живёт на http://localhost:5500 или похожем адресе —
# он в списке по умолчанию. В проде впишите в .env реальный домен сайта:
# ALLOWED_ORIGINS=https://вашдомен.ru,https://www.вашдомен.ru
_DEFAULT_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    top_k: int = 3
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    session_id: str


def _load_or_build_index():
    """
    Обычно индекс уже есть и просто открывается — быстро. Но если его нет
    (первый запуск после скачивания проекта) или он устарел (поменялась
    версия embedding-алгоритма — load_index() как раз это и проверяет),
    собираем его прямо тут, автоматически. Это и есть та самая "одна
    команда": `python -m backend.build_index` больше не нужно помнить и
    запускать отдельно — `uvicorn ...` теперь делает всё сам.
    """
    try:
        return load_index(CHROMA_DIR, COLLECTION_NAME, embedding_function)
    except RuntimeError:
        logger.info("Индекс отсутствует или устарел — собираю автоматически...")
        from . import build_index as _build_index

        _build_index.main()
        return load_index(CHROMA_DIR, COLLECTION_NAME, embedding_function)


collection = _load_or_build_index()

# Единая формулировка личности ассистента — используется только тут (раньше
# в разных местах проекта встречались разные формулировки: "ассистент
# Ивана", "FAQ-ассистент компании" и т.п.). Можно переопределить без
# изменения кода, задав SYSTEM_PROMPT в .env.
DEFAULT_SYSTEM_PROMPT = (
    "Ты — личный AI-ассистент Ивана Паро, AI/Backend/Telegram-ботов разработчика "
    "(Python, FastAPI, AI-агенты, RAG, Telegram-боты). Отвечаешь от его лица на "
    "сайте-портфолио и консультируешь по любым вопросам о нём: стек технологий, опыт, "
    "процесс работы, цены, сроки, проекты. Отвечай кратко, по делу, на русском языке, "
    "уверенно и дружелюбно — как человек, который хорошо знает Ивана и его работу.\n\n"
    "Опирайся на контекст с фактами, который передаётся вместе с вопросом. Никогда не "
    "говори фразы вроде «я не знаю», «в контексте нет информации», «не могу ответить» — "
    "это запрещено. Если в переданном контексте нет точного ответа на вопрос, не "
    "признавайся в этом явно: коротко ответь тем, что по теме есть (в общих словах о "
    "подходе, стеке или процессе работы Ивана), и естественно предложи уточнить детали "
    f"напрямую у Ивана в Telegram: {IVAN_TELEGRAM} — он отвечает оперативно. Не выдумывай "
    "цифры, сроки, ссылки или факты, которых нет в контексте: при нехватке конкретики "
    "веди разговор к Telegram, а не сочиняй ответ."
)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# --------------------------------------------------------------------------
# История переписки — в SQLite-файле (data/sessions.db), а не в памяти
# процесса: переживает перезапуск backend и (если когда-нибудь появятся
# несколько uvicorn-воркеров) читается/пишется из общего файла, а не
# расходится по разным процессам. Реализация и когда переходить на Redis —
# см. backend/history_store.py. Старые сессии сами вычищаются по TTL
# (см. SESSION_TTL_DAYS там же) — это замена прежнему MAX_SESSIONS/LRU.
# --------------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 12  # сколько последних сообщений (user+bot) держим на сессию


def _history_to_messages(raw_history: List[Dict[str, str]]) -> List[Messages]:
    return [Messages(role=MessagesRole(item["role"]), content=item["content"]) for item in raw_history]


def _messages_to_history(messages: List[Messages]) -> List[Dict[str, str]]:
    # ВАЖНО: gigachat/pydantic хранит Messages.role уже как обычную строку
    # ("user"/"assistant"), а не как объект MessagesRole — несмотря на то,
    # что конструируем мы его через MessagesRole.USER. У обычной строки нет
    # .value, поэтому берём m.role напрямую (проверено: Messages(role=
    # MessagesRole.USER, ...).role is str, не MessagesRole).
    return [{"role": m.role, "content": m.content} for m in messages]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")

    session_id = req.session_id or str(uuid.uuid4())
    history = _history_to_messages(load_history(session_id))

    try:
        similar_items = search_similar(collection, req.message, k=req.top_k)
    except Exception:
        # search_similar сама поднимает RuntimeError с понятным текстом при
        # рассинхронизации embedding-провайдера/размерности (см. rag_index.py).
        # load_index() ловит это при старте для типового случая (сменили
        # EMBEDDING_PROVIDER в .env, не пересобрав индекс) — но если
        # рассинхронизация возникнет уже во время работы (например, код
        # LocalEmbeddingFunction поменяли, а embedding_provider в метаданных
        # остался тем же "local"), это всплывёт только здесь, при живом
        # запросе. Раньше исключение отсюда ничем не ловилось и утекало
        # наружу как HTTP 500 — посетитель просто не получал ответ. Теперь
        # логируем для Ивана и отвечаем без RAG-контекста, а не падаем.
        logger.exception("Поиск по индексу не сработал для запроса %r (session=%s)", req.message, session_id)
        similar_items = []

    context_text = "\n\n".join(
        [f"Q: {item['question']}\nA: {item['answer']}" for item in similar_items]
    )

    # В саму историю кладём чистый вопрос без FAQ-контекста (см. ниже), а вот
    # модели за этот конкретный запрос показываем вопрос + свежий контекст —
    # так на каждом шаге модель видит актуальные факты, а история не раздувается
    # повторным контекстом из прошлых сообщений.
    current_user_message = Messages(
        role=MessagesRole.USER,
        content=f"Вопрос пользователя: {req.message}\n\nКонтекст FAQ:\n{context_text}",
    )

    payload = Chat(
        messages=[Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT)]
        + history
        + [current_user_message],
        temperature=0.2,
        max_tokens=500,  # FAQ-ответы короткие; без лимита модель может расписаться на выход
    )

    if giga is None:
        answer = build_fallback_answer(req.message, similar_items)
    else:
        try:
            completion = giga.chat(payload)
            answer = completion.choices[0].message.content
        except Exception:
            # Не прячем сбой молча: он попадёт в лог процесса (stdout /
            # journalctl / Sentry и т.п.), чтобы Иван реально увидел, что
            # GigaChat недоступен — закончился лимит, неверный ключ или
            # сервис лёг. Посетителю сайта при этом всё равно приходит
            # спокойный, полезный ответ, а не техническая ошибка.
            logger.exception("GigaChat недоступен, отвечаю фолбэком по FAQ (session=%s)", session_id)
            answer = build_fallback_answer(req.message, similar_items)

    history.append(Messages(role=MessagesRole.USER, content=req.message))
    history.append(Messages(role=MessagesRole.ASSISTANT, content=answer))
    history = history[-MAX_HISTORY_MESSAGES:]
    save_history(session_id, _messages_to_history(history))

    return ChatResponse(answer=answer, context=similar_items, session_id=session_id)


@app.get("/health")
async def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Отдаём сайт (frontend/) и картинки (images/) этим же процессом — один
# backend, один порт, один процесс вместо двух (`uvicorn` + отдельный
# `python -m http.server`). Раньше frontend/index.html грузил картинки по
# пути "../images/..." — это ломалось при запуске `python -m http.server`
# из папки frontend (сервер не видит файлы выше своей папки). Здесь монтируем
# images/ отдельно, и относительный "../images/..." браузер сам сводит к
# "/images/..." — уже работает.
#
# ВАЖНО: порядок mount() имеет значение. "/" — это "поймай всё, что не
# подошло выше" (html=True отдаёт index.html на "/"), поэтому и "/images",
# и все @app.* роуты (/chat, /health) обязаны быть объявлены раньше него —
# иначе "/" перехватит их первым.
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/images", StaticFiles(directory=os.path.join(PROJECT_ROOT, "images")), name="images")
app.mount("/", StaticFiles(directory=os.path.join(PROJECT_ROOT, "frontend"), html=True), name="frontend")


def build_fallback_answer(query: str, similar_items: List[Dict[str, Any]]) -> str:
    """
    Срабатывает, если GigaChat недоступен/не настроен. Не должен звучать как
    «я не знаю»: либо отдаёт наиболее похожий на вопрос готовый ответ из FAQ,
    либо аккуратно ведёт в Telegram к Ивану.

    Без LLM некому оценить смысл, поэтому среди top_k кандидатов от ChromaDB
    здесь дополнительно выбирается тот, чей вопрос сильнее всего пересекается
    по словам с вопросом пользователя (пересечение множеств токенов). Это не
    снимает ограничения локального поиска (см. LocalEmbeddingFunction в
    rag_index.py), но подстраховывает от случаев, когда top-1 от ChromaDB
    лексически «зацепился» не за ту тему — на практике это встречается
    (например, на вопрос о цене может выйти на первое место приветствие,
    если в нём случайно больше общих слов с вопросом).
    """
    if not similar_items:
        return (
            "Хороший вопрос — по нему лучше уточнить детали напрямую. Расскажите, что вас "
            f"интересует, Ивану в Telegram: {IVAN_TELEGRAM} — он оперативно ответит."
        )

    query_tokens = set(TOKEN_RE.findall(query.lower()))
    if not query_tokens:
        return similar_items[0]["answer"]

    def overlap(item: Dict[str, Any]) -> int:
        item_tokens = set(TOKEN_RE.findall(item.get("question", "").lower()))
        return len(query_tokens & item_tokens)

    best = max(similar_items, key=overlap)
    return best["answer"]
