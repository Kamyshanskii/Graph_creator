import os
import re
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch


DOCS_DIR = Path("docs")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "index.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_markdown_files() -> List[Dict]:
    docs = []
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Папка {DOCS_DIR} не найдена. Создай её и положи туда .md файлы.")

    for path in DOCS_DIR.glob("*.md"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"path": path, "text": text})
    if not docs:
        raise RuntimeError(f"В {DOCS_DIR} нет .md файлов.")
    return docs


def split_into_chunks(text: str, max_chars: int = 1000) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    for p in paragraphs:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf += "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    return chunks


def build_index():
    print("Чтение документов из docs/ ...")
    docs = read_markdown_files()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Загрузка модели эмбеддингов: {EMBED_MODEL_NAME} (device={device})")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    all_texts: List[str] = []
    all_embeddings = []
    all_metadatas: List[Dict] = []

    for doc in docs:
        path = doc["path"]
        text = doc["text"]
        chunks = split_into_chunks(text, max_chars=1000)

        print(f"{path.name}: {len(chunks)} чанков")
        embeddings = model.encode(
            chunks,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for chunk, emb in zip(chunks, embeddings):
            all_texts.append(chunk)
            all_embeddings.append(emb.astype("float32"))
            all_metadatas.append({"source": path.name})

    embeddings_matrix = np.stack(all_embeddings, axis=0)

    index = {
        "embeddings": embeddings_matrix,
        "texts": all_texts,
        "metadatas": all_metadatas,
        "model_name": EMBED_MODEL_NAME,
    }

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"Сохранён индекс: {INDEX_PATH} (chunks={embeddings_matrix.shape[0]})")


if __name__ == "__main__":
    build_index()
