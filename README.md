# Портфолио + FAQ-ассистент

Сайт-портфолио с чат-виджетом. Чат отвечает на вопросы про Ивана: стек, проекты, процесс работы, цены, сроки. Ответы ищутся в локальной базе (`data/`), GigaChat только красиво их формулирует.

## Структура

- `frontend/` — сайт (HTML/CSS/JS).
- `backend/` — API на FastAPI.
- `data/qa_database.txt`, `data/tech_stack.json` — база знаний, редактируете сами.
- `data/chroma_db/`, `data/sessions.db` — создаются автоматически при запуске. В Git не нужны.

## Запуск (Windows, Python 3.11)

1. `py -3.11 -m venv venv`
2. `venv\Scripts\activate`
3. `$env:PYTHONUTF8 = "1"` — нужно, если в пути к проекту есть русские буквы.
4. `pip install -r requirements.txt`
5. Создать `.env` в корне проекта:
   ```env
   GIGACHAT_CREDENTIALS=ваш_ключ
   ```
   Без ключа бот тоже работает — просто отвечает готовыми фразами из базы, без GigaChat.
6. Собрать базу знаний: `python -m backend.build_index`
7. Запустить backend: `uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000`
8. Во втором терминале запустить frontend:
   ```powershell
   cd frontend
   python -m http.server 5500
   ```

Сайт: http://127.0.0.1:5500
Проверка backend: http://127.0.0.1:8000/health

## Обновить базу знаний

1. Отредактировать `data/qa_database.txt` или `data/tech_stack.json`.
2. `python -m backend.build_index`
3. Перезапустить backend.

## Как это работает

Вопрос на сайте → backend ищет похожее в базе → если есть ключ GigaChat, формулирует ответ через него → если ключа нет или GigaChat недоступен, отвечает готовым ответом из базы.

## Доп. настройки в .env (не обязательны)

| Переменная | Зачем | Если не задано |
|---|---|---|
| `GIGACHAT_MODEL` | какую модель GigaChat использовать | `GigaChat-2` — самая дешёвая и быстрая |
| `ALLOWED_ORIGINS` | с каких доменов разрешены запросы к backend | только localhost |
| `IVAN_TELEGRAM` | куда бот направляет, если не может ответить | `https://t.me/Ivan_Paro` |
| `SYSTEM_PROMPT` | полностью своя инструкция для ассистента | встроенная в код |

## Деплой

- Домен сайта нужно добавить в `ALLOWED_ORIGINS`.
- Frontend ждёт backend по адресу `/api` — настраивается через Nginx:
  ```nginx
  location /api/ {
      proxy_pass http://127.0.0.1:8000/;
  }
  ```
