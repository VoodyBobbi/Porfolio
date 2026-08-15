import os

import chromadb
from dotenv import load_dotenv

from .rag_index import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    LOCAL_EMBEDDING_VERSION,
    TECH_JSON_PATH,
    create_embedding_function,
    load_tech_stack,
)


load_dotenv()

# Индекс строится только локальной embedding-функцией — без внешнего API,
# без GigaChat, без сетевых вызовов и без токенов. GigaChat в этом файле
# больше не участвует вообще (раньше был опциональный путь через
# EMBEDDING_PROVIDER=gigachat — его убрали как ненужный для этого проекта).
embedding_function = create_embedding_function()


def _is_text_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"=", "-"}


def parse_qa_pairs(content: str, source: str):
    docs = []
    current_question = None
    answer_lines = []

    def flush_current():
        if not current_question:
            return
        answer = "\n".join(answer_lines).strip()
        if answer:
            docs.append(
                {
                    "question": current_question,
                    "answer": answer,
                    "source": source,
                }
            )

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if line.startswith("Вопрос:"):
            flush_current()
            current_question = line.removeprefix("Вопрос:").strip()
            answer_lines = []
            continue

        if line.startswith("Ответ:") and current_question:
            answer_lines.append(line.removeprefix("Ответ:").strip())
            continue

        if current_question and answer_lines:
            if line.startswith("БЛОК ") or _is_text_separator(line):
                continue
            answer_lines.append(raw_line.rstrip())

    flush_current()
    return docs


def load_txt_documents(directory: str):
    """
    Загружает все .txt-файлы из папки.
    Если файл содержит блоки "Вопрос:" / "Ответ:", каждый блок становится
    отдельным документом для поиска.
    """
    docs = []
    if not os.path.isdir(directory):
        return docs

    for name in os.listdir(directory):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError:
            continue

        if not content:
            continue

        qa_docs = parse_qa_pairs(content, name)
        if qa_docs:
            docs.extend(qa_docs)
            continue

        lines = content.splitlines()
        title = next((line.strip() for line in lines if line.strip()), name)
        body_lines = lines[1:] if len(lines) > 1 else []
        body = "\n".join(body_lines).strip() or content

        docs.append(
            {
                "question": title,
                "answer": body,
                "source": name,
            }
        )

    return docs


def main():
    items = []

    txt_docs = load_txt_documents(DATA_DIR)
    items.extend(txt_docs)
    print(f"Loaded {len(txt_docs)} TXT documents from {DATA_DIR}")

    if os.path.exists(TECH_JSON_PATH):
        tech_items = load_tech_stack(TECH_JSON_PATH)
        items.extend(tech_items)
        print(f"Loaded {len(tech_items)} technology items from tech_stack.json")

    if not items:
        raise RuntimeError("No data found to build index (no txt files).")

    documents = [f"{item['question']}\n{item['answer']}" for item in items]
    metadatas = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "source": item.get("source", "unknown"),
        }
        for item in items
    ]
    ids = [str(i) for i in range(len(items))]

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # пересобираем коллекцию с нуля, если она уже была
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"local_embedding_version": LOCAL_EMBEDDING_VERSION},
    )

    print(f"Embedding {len(items)} items locally and adding to ChromaDB...")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Index built and saved to {CHROMA_DIR} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    main()
