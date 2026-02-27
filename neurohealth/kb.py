from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import KnowledgeChunk, RetrievedChunk


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match for cosine similarity.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def load_knowledge_chunks(path: str | Path) -> list[KnowledgeChunk]:
    kb_path = Path(path)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    raw_data = json.loads(kb_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("Knowledge base JSON must contain a list of chunk objects.")

    chunks: list[KnowledgeChunk] = []
    for index, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Knowledge chunk at index {index} is not an object.")

        chunk_id = str(item.get("id", "")).strip()
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        source = str(item.get("source", "")).strip()
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError(f"Knowledge chunk {chunk_id or index} has invalid tags.")
        normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        if not chunk_id or not title or not content or not source:
            raise ValueError(f"Knowledge chunk {index} is missing required fields.")

        chunks.append(
            KnowledgeChunk(
                id=chunk_id,
                title=title,
                content=content,
                source=source,
                tags=normalized_tags,
            )
        )

    if not chunks:
        raise ValueError("Knowledge base must contain at least one chunk.")
    return chunks


@dataclass
class KnowledgeIndex:
    chunks: list[KnowledgeChunk]
    vectors: list[list[float]]

    @classmethod
    def build(cls, path: str | Path, embedding_client: EmbeddingClient) -> "KnowledgeIndex":
        chunks = load_knowledge_chunks(path)
        texts = [f"{chunk.title}\n{chunk.content}" for chunk in chunks]
        vectors = embedding_client.embed_texts(texts)
        if len(vectors) != len(chunks):
            raise ValueError("Knowledge vectors count does not match chunk count.")
        return cls(chunks=chunks, vectors=vectors)

    def retrieve(
        self,
        query: str,
        embedding_client: EmbeddingClient,
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if not query.strip():
            raise ValueError("query must not be empty.")

        query_vector = embedding_client.embed_query(query)
        scored: list[RetrievedChunk] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = _cosine_similarity(query_vector, vector)
            scored.append(
                RetrievedChunk(
                    id=chunk.id,
                    title=chunk.title,
                    content=chunk.content,
                    source=chunk.source,
                    tags=chunk.tags,
                    score=score,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
