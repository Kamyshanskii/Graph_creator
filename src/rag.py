# src/rag.py

import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


INDEX_PATH = Path("data/index.pkl")


class RAGRetriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        if not index_path.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {index_path}. "
                f"Сначала запусти: python -m src.build_index"
            )

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.embeddings: np.ndarray = data["embeddings"]
        self.texts: List[str] = data["texts"]
        self.metadatas: List[Dict[str, Any]] = data["metadatas"]
        self.model_name: str = data.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.model = SentenceTransformer(self.model_name, device="cpu")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Возвращает top_k самых похожих чанков по косинусному сходству.
        """
        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = np.dot(self.embeddings, q_vec)  # cos similarity, т.к. нормализовано
        top_idx = np.argsort(-scores)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            results.append(
                {
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(scores[idx]),
                }
            )
        return results

    def make_context(self, query: str, top_k: int = 3) -> str:
        """
        Склеивает топ-чанки в текст для подсказки LLM.
        """
        docs = self.retrieve(query, top_k=top_k)
        parts = []
        for i, d in enumerate(docs, start=1):
            src = d["metadata"].get("source", "unknown")
            parts.append(
                f"[Документ {i} | {src} | score={d['score']:.3f}]\n{d['text']}"
            )
        return "\n\n".join(parts)
