import hashlib
import json
import math
import os
import re
from typing import Any, List

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from gigachat import GigaChat


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "faqs.json")
TECH_JSON_PATH = os.path.join(DATA_DIR, "tech_stack.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "faqs"
LOCAL_EMBEDDING_DIMENSIONS = 1536  # было 512 — при 512 короткие запросы (1 слово) иногда
# проигрывали по ранжированию не связанным длинным документам из-за коллизий хэшей;
# смотрите заметку про "питон" в ревью ниже.

# Версия алгоритма LocalEmbeddingFunction. load_index() сравнивает её с тем,
# что сохранено в метаданных индекса — так индекс, построенный старым
# алгоритмом (или с другой LOCAL_EMBEDDING_DIMENSIONS), явно попросит
# пересборку вместо тихой рассинхронизации размерности во время запроса.
# Это отдельная метка ОТ embedding_provider: смена local<->gigachat и смена
# версии самого локального алгоритма — разные события, и метка провайдера
# сама по себе не ловит второе (см. воспроизведённый краш ниже).
LOCAL_EMBEDDING_VERSION = "2-ngram-1536d"
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


class LocalEmbeddingFunction(EmbeddingFunction):
    """
    Локальная hash-based embedding-функция для FAQ-поиска без внешнего API.

    ВАЖНО (честно о лимитах): это НЕ семантический поиск в полном смысле —
    это hashing trick (bag-of-words + char n-grams) поверх текста. Он ловит:
      - точные и частично совпадающие слова;
      - однокоренные слова и словоформы (через 4-граммы символов — важно для
        русского языка с его богатой морфологией: "стоимость"/"стоимости",
        "работает"/"работал" дадут пересекающиеся n-граммы и попадут рядом).
    Он НЕ ловит смысловые синонимы без общих букв (например "цена" и
    "стоимость" почти не пересекаются по n-граммам). Для по-настоящему
    смыслового поиска нужны обученные embeddings — используйте
    EMBEDDING_PROVIDER=gigachat (see README), там эмбеддинги считает
    реальная модель через GigaChat Embeddings API.
    """

    def __init__(self, dimensions: int = LOCAL_EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        lowered = text.lower()
        tokens = TOKEN_RE.findall(lowered)

        for token in tokens:
            # Целое слово — как и раньше, ловит точные совпадения.
            self._hash_into(vector, "w:" + token, weight=1.0)

            # Символьные 4-граммы слова — ловят совпадение корня/основы даже
            # при разных окончаниях/падежах ("технологии" vs "технологиями").
            # Короткие слова (<=4 символов) не дробим — n-грамма из них самого
            # по себе уже покрывает всё слово.
            if len(token) > 4:
                for i in range(len(token) - 3):
                    ngram = token[i : i + 4]
                    self._hash_into(vector, "n:" + ngram, weight=0.4)

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def _hash_into(self, vector: List[float], key: str, weight: float) -> None:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * weight


class GigaChatEmbeddingFunction(EmbeddingFunction):
    """Embedding-функция ChromaDB поверх GigaChat Embeddings API."""

    def __init__(self, client: GigaChat, model: str = "Embeddings"):
        self.client = client
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        result = self.client.embeddings(list(input), model=self.model)
        return [item.embedding for item in result.data]


def create_embedding_function(provider: str, giga: GigaChat | None = None) -> EmbeddingFunction:
    provider = provider.lower().strip()
    if provider == "local":
        return LocalEmbeddingFunction()
    if provider == "gigachat":
        if giga is None:
            raise RuntimeError("GigaChat client is required for EMBEDDING_PROVIDER=gigachat.")
        return GigaChatEmbeddingFunction(giga)
    raise RuntimeError("Unsupported EMBEDDING_PROVIDER. Use `local` or `gigachat`.")


def load_index(
    chroma_dir: str,
    collection_name: str,
    embedding_function: EmbeddingFunction,
    provider: str = "",
):
    """
    Открывает существующий ChromaDB-индекс.

    Проверяет ДВА независимых источника рассинхронизации:
    1) EMBEDDING_PROVIDER поменяли (local <-> gigachat) без пересборки —
       ловится по collection.metadata["embedding_provider"];
    2) сам алгоритм local-провайдера поменялся (например, поменяли
       LOCAL_EMBEDDING_DIMENSIONS в коде), а EMBEDDING_PROVIDER остался
       "local" и до и после — (1) это НЕ ловит, т.к. имя провайдера не
       изменилось. Для этого отдельно сверяется LOCAL_EMBEDDING_VERSION.
    В обоих случаях падаем сразу при старте backend'а с понятной ошибкой —
    вместо того чтобы упасть позже, на первом вопросе живого посетителя сайта
    (или, что ещё хуже в случае (2) без этой проверки, вообще не упасть, а
    вернуть мусорные результаты поиска).
    """
    if not os.path.isdir(chroma_dir):
        raise RuntimeError(
            "ChromaDB директория не найдена. "
            "Сначала запустите `python -m backend.build_index`, чтобы построить индекс."
        )
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

    meta = collection.metadata or {}
    built_with = meta.get("embedding_provider")
    provider = provider.lower().strip()
    if built_with and provider and built_with != provider:
        raise RuntimeError(
            f"Индекс был построен с EMBEDDING_PROVIDER={built_with!r}, а сейчас в .env "
            f"указан EMBEDDING_PROVIDER={provider!r}. local и gigachat дают векторы разной "
            "природы и размерности, поэтому поиск сломается. Исправьте .env на "
            f"{built_with!r} либо пересоберите индекс: `python -m backend.build_index`."
        )

    if provider == "local":
        built_version = meta.get("local_embedding_version")
        # Строго (не "if built_version and ..."): индекс вовсе без метки
        # версии тоже считается устаревшим — иначе он тихо продолжит
        # работать со старым алгоритмом вместо явной подсказки пересобрать.
        if built_version != LOCAL_EMBEDDING_VERSION:
            raise RuntimeError(
                "Индекс построен другой версией локальной embedding-функции "
                f"({built_version!r}, сейчас {LOCAL_EMBEDDING_VERSION!r}) — либо старым "
                "кодом, либо после обновления rag_index.py без пересборки. Пересоберите "
                "индекс: `python -m backend.build_index`."
            )

    return collection


def search_similar(collection, query_text: str, k: int = 3) -> List[Any]:
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=k,
            include=["metadatas"],
        )
    except Exception as exc:
        raise RuntimeError(
            "Не удалось выполнить поиск по индексу. Похоже, текущий EMBEDDING_PROVIDER "
            "не совпадает с тем, каким индекс был построен (local и gigachat дают векторы "
            "разной размерности). Пересоберите индекс: `python -m backend.build_index`."
        ) from exc
    metadatas = results.get("metadatas") or [[]]
    return metadatas[0]


def load_faq_data(path: str):
    """Загружает FAQ данные из JSON файла."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten(value: Any) -> str:
    """how_it_works/benefits в tech_stack.json бывают строкой или списком строк."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def load_tech_stack(path: str):
    """
    Загружает data/tech_stack.json и превращает каждую технологию в
    отдельный документ {question, answer, source} — в том же формате,
    что и остальные источники, чтобы попасть в общий индекс.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for tech in data.get("technologies", []):
        answer = (
            f"{tech.get('description', '')}\n\n"
            f"Как это работает: {_flatten(tech.get('how_it_works', ''))}\n\n"
            f"Преимущества: {_flatten(tech.get('benefits', ''))}\n\n"
            f"Применение: {_flatten(tech.get('use_cases', ''))}"
        )
        # keywords в tech_stack.json как раз для альтернативных формулировок
        # ("питон" для Python и т.п.), но раньше это поле не попадало в
        # индексируемый текст — записывалось в JSON, но не влияло на поиск.
        keywords = tech.get("keywords") or []
        if keywords:
            answer += "\n\nКлючевые слова: " + ", ".join(str(k) for k in keywords)

        docs.append(
            {
                "question": tech.get("name", tech.get("id", "")),
                "answer": answer,
                "source": "tech_stack.json",
            }
        )
    return docs
