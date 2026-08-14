# Портфолио Ивана Паро — сайт + FAQ RAG-ассистент

Статический лендинг + чат-виджет. Отвечает на вопросы про стек, опыт, процесс и стоимость по локальной базе знаний `data/qa_database.txt` и `data/tech_stack.json`. GigaChat используется только для генерации финального ответа, поиск по базе — полностью локальный.

## Архитектура

frontend/index.html, style.css, script.js
  -> POST /chat (API_BASE = localhost:8000 локально / /api в проде)
  -> backend/app.py (FastAPI, /chat, /health)
  -> backend/rag_index.py (LocalEmbeddingFunction + ChromaDB data/chroma_db/)
  -> backend/app.py: giga.chat() или build_fallback_answer()
  -> backend/history_store.py (data/sessions.db, SQLite WAL, TTL 30 дней)

Поток в app.py:
1. Валидация ChatRequest.message (Pydantic, max 2000)
2. load_history(session_id) — последние 12 сообщений
3. search_similar(collection, query, k=top_k) — ChromaDB query -> LocalEmbeddingFunction.__call__
4. Сборка context_text из найденных FAQ + системный промпт
5. Chat(messages=[SYSTEM,...history, user+context]) -> giga.chat(chat) или фолбэк
6. save_history(session_id, user+assistant)

## Поиск и эмбеддинги

Только локальные, внешних запросов на этапе поиска нет.

backend/rag_index.py: LocalEmbeddingFunction:
- hashing trick: bag-of-words + char 4-grams
- Токенизация: TOKEN_RE = [0-9A-Za-zА-Яа-яЁё]+, lowercased
- Хеш: blake2b(key, digest_size=8), индекс % 1536, знак по 5-му байту
- Веса: слово w:token = 1.0, 4-грамма n:xxxx = 0.4 для слов >4 символов
- L2 нормализация, размерность LOCAL_EMBEDDING_DIMENSIONS = 1536
- Версия LOCAL_EMBEDDING_VERSION = "2-ngram-1536d" пишется в collection.metadata

ChromaDB не делает эмбеддинги сама — get_or_create_collection(embedding_function=...) и collection.query() делегируют в LocalEmbeddingFunction.__call__.

Индекс: data/chroma_db/, коллекция faqs. Проверка версии при load_index() — если версия не совпала -> RuntimeError с требованием python -m backend.build_index.

## Структура проекта

- backend/app.py — FastAPI, CORS по ALLOWED_ORIGINS (не *, allow_credentials=False), /chat, /health, build_fallback_answer()
- backend/rag_index.py — LocalEmbeddingFunction, create_embedding_function(), load_index(), search_similar(), load_tech_stack()
- backend/build_index.py — парсер TXT (Вопрос:/Ответ:, пропуск БЛОК, ===, ---), загрузка faqs.json + все data/*.txt + tech_stack.json, delete_collection() + add()
- backend/history_store.py — init_db(), WAL, sqlite3.connect(..., check_same_thread=False), INSERT OR REPLACE, cleanup_old_sessions(TTL_DAYS=30) раз в CLEANUP_INTERVAL=21600s
- frontend/index.html, style.css, script.js — лендинг, модалка 32 технологии, тогглы скиллов (12 видимых + Показать ещё), scroll-reveal IntersectionObserver(threshold=0), сессия localStorage faq_chat_session_id
- data/qa_database.txt, tech_stack.json, chroma_db/, sessions.db
- requirements.txt — fastapi, uvicorn[standard], chromadb==0.5.5, gigachat==0.1.40, python-dotenv==1.0.1

## Установка

py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

.env в корне:
GIGACHAT_CREDENTIALS=ваш_ключ
GIGACHAT_MODEL=GigaChat-2
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,https://ваш-домен
IVAN_TELEGRAM=https://t.me/Ivan_Paro
SYSTEM_PROMPT=опционально

## Индексация

python -m backend.build_index

## Запуск

uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
cd frontend && python -m http.server 5500

GET /health -> {"status":"ok"}

POST /chat:
{"message":"Чем занимается Иван?","top_k":3,"session_id":"опционально"}

build_fallback_answer() считает пересечение токенов запроса и каждого документа по TOKEN_RE, берет max score, если score=0 -> ведет в Telegram IVAN_TELEGRAM

## Деплой

location /api/ {
  proxy_pass http://127.0.0.1:8000/;
}
location / {
  root /path/to/frontend;
}

frontend/script.js API_BASE: localhost/127.0.0.1 -> http://127.0.0.1:8000, иначе /api. Добавьте домен в ALLOWED_ORIGINS.

## Ограничения

- gigaChat verify_ssl_certs=False — включите в проде
- history_store check_same_thread=False — не для workers>1, для масштаба Redis
- Rate-limit на /chat добавьте на Nginx
- Локальный эмбеддинг не ловит синонимы без общих букв (цена/стоимость)