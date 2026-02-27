from __future__ import annotations

from pathlib import Path

from .config import Settings
from .embeddings import GitHubModelsEmbeddingClient
from .engine import NeuroHealthEngine
from .feedback import FeedbackStore
from .kb import KnowledgeIndex
from .llm import GeminiClient


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def parse_biometrics(items: list[str]) -> dict[str, float | int | str]:
    biometrics: dict[str, float | int | str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid biometric format '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        metric = key.strip()
        value = raw_value.strip()
        if not metric:
            raise ValueError(f"Invalid biometric key in '{item}'.")
        if not value:
            raise ValueError(f"Invalid biometric value in '{item}'.")

        if "." in value:
            try:
                biometrics[metric] = float(value)
                continue
            except ValueError:
                pass
        try:
            biometrics[metric] = int(value)
            continue
        except ValueError:
            biometrics[metric] = value
    return biometrics


def build_engine(settings: Settings) -> NeuroHealthEngine:
    embedding_client = GitHubModelsEmbeddingClient(
        api_key=settings.github_token,
        model=settings.embedding_model,
        endpoint=settings.embedding_endpoint,
    )
    knowledge_index = KnowledgeIndex.build(Path(settings.knowledge_base_path), embedding_client)
    llm_client = GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
    feedback_store = FeedbackStore(Path(settings.feedback_store_path))
    return NeuroHealthEngine(
        knowledge_index=knowledge_index,
        embedding_client=embedding_client,
        llm_client=llm_client,
        feedback_store=feedback_store,
    )
