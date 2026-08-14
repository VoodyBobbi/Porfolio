Портфолио Ивана Паро — сайт + FAQ RAG-ассистент
Сайт-визитка с AI-ассистентом. Отвечает про стек, опыт, процесс и цены по локальной базе знаний. GigaChat используется только для генерации ответов, не для эмбеддингов.

Как работает
frontend/index.html → виджет чата → POST /chat на FastAPI → локальный поиск по ChromaDB → промпт с FAQ → GigaChat (или фолбэк без LLM) → ответ + сохранение истории в SQLite.

Поиск (эмбеддинги)
Только локальный, без внешних API:

LocalEmbeddingFunction в backend/rag_index.py — hashing trick: хеш слов + 4-граммы символов
Размерность 1536, нормализация L2, версия 2-ngram-1536d
Ловит точные и однокоренные формы (важно для русского), не ловит синонимы без общих букв
ChromaDB вызывает эту функцию сама, своих эмбеддингов не делает: collection.query() → LocalEmbeddingFunction.__call__() → вектор
GigaChat для эмбеддингов не используется. Требуется только для giga.chat() в app.py.

Состав
backend/app.py — FastAPI /chat, /health, CORS по ALLOWED_ORIGINS, история 12 сообщений
backend/rag_index.py — LocalEmbeddingFunction, load_index(), search_similar()
backend/build_index.py — парсер Вопрос:/Ответ: из data/*.txt + tech_stack.json → data/chroma_db/
backend/history_store.py — SQLite sessions.db, TTL 30 дней, WAL
frontend/ — index.html, style.css, script.js (тогглы скиллов, scroll-reveal, API_BASE localhost:8000 vs /api)
data/qa_database.txt — FAQ, tech_stack.json — 32 технологии с keywords
requirements.txt
Установка
bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
.env (пример в .env.example):

GIGACHAT_CREDENTIALS=...
GIGACHAT_MODEL=GigaChat-2
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500
IVAN_TELEGRAM=https://t.me/Ivan_Paro
SYSTEM_PROMPT=опционально
Индекс
bash
python -m backend.build_index
Пересобирать при изменении qa_database.txt или tech_stack.json или версии LOCAL_EMBEDDING_VERSION.

Запуск
bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
# frontend
cd frontend && python -m http.server 5500
GET /health → {"status":"ok"}

POST /chat:

json
{"message":"Чем занимается Иван?","top_k":3,"session_id":"опционально"}
Логика: load_history → search_similar → context_text → Chat(messages=[SYSTEM, ...history, user+context]) → giga.chat() или build_fallback_answer() (выбирает из top_k по пересечению токенов TOKEN_RE) → save_history.

Деплой
Nginx проксирует /api/ на backend:

location /api/ { proxy_pass http://127.0.0.1:8000/; }
Добавьте прод-домен в ALLOWED_ORIGINS.

Замечания
verify_ssl_certs=False в GigaChat — включите проверку в проде
history_store с check_same_thread=False — не для workers>1, для масштаба — Redis
Rate-limit на /chat добавляйте на Nginx
