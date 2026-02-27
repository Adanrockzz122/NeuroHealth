from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv_defaults(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        env_key = key.strip()
        if not env_key:
            continue
        env_value = _strip_optional_quotes(raw_value.strip())
        os.environ.setdefault(env_key, env_value)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    github_token: str
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_endpoint: str = "https://models.github.ai/inference/embeddings"
    knowledge_base_path: str = "data/knowledge_base.json"
    feedback_store_path: str = "data/feedback.jsonl"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv_defaults(".env")

        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        github_token = os.getenv(
            "GITHUB_TOKEN",
            os.getenv("GITHUB_MODELS_TOKEN", os.getenv("OPENAI_API_KEY", "")),
        ).strip()
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required (set env var or .env entry).")
        if not github_token:
            raise ValueError(
                "GITHUB_TOKEN is required for GitHub Models embeddings (set env var or .env entry)."
            )

        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        embedding_model = os.getenv(
            "GITHUB_EMBEDDING_MODEL",
            os.getenv("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        ).strip()
        embedding_endpoint = os.getenv(
            "GITHUB_MODELS_EMBEDDING_ENDPOINT",
            "https://models.github.ai/inference/embeddings",
        ).strip()
        kb_path = os.getenv("NEUROHEALTH_KB_PATH", "data/knowledge_base.json").strip()
        feedback_path = os.getenv("NEUROHEALTH_FEEDBACK_PATH", "data/feedback.jsonl").strip()

        if not gemini_model:
            raise ValueError("GEMINI_MODEL must not be empty.")
        if not embedding_model:
            raise ValueError("GITHUB_EMBEDDING_MODEL must not be empty.")
        if not embedding_endpoint:
            raise ValueError("GITHUB_MODELS_EMBEDDING_ENDPOINT must not be empty.")
        if not kb_path:
            raise ValueError("NEUROHEALTH_KB_PATH must not be empty.")
        if not feedback_path:
            raise ValueError("NEUROHEALTH_FEEDBACK_PATH must not be empty.")

        return cls(
            gemini_api_key=gemini_api_key,
            github_token=github_token,
            gemini_model=gemini_model,
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
            knowledge_base_path=kb_path,
            feedback_store_path=feedback_path,
        )
