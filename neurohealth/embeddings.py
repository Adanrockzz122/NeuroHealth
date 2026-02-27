from __future__ import annotations

import json
from collections.abc import Callable
from urllib import error, request

JsonPostCallable = Callable[[str, dict[str, str], dict[str, object]], dict[str, object]]


class OpenAIEmbeddingError(RuntimeError):
    """Raised when embedding requests fail."""


def _default_post_json(url: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **headers}
    req = request.Request(url=url, data=body, headers=request_headers, method="POST")
    try:
        with request.urlopen(req, timeout=45) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise OpenAIEmbeddingError(f"Embedding request HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise OpenAIEmbeddingError(f"Embedding request network error: {exc.reason}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpenAIEmbeddingError("Embedding request returned invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise OpenAIEmbeddingError("Embedding request returned an unexpected payload type.")
    return parsed


class GitHubModelsEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/text-embedding-3-small",
        endpoint: str = "https://models.github.ai/inference/embeddings",
        post_json: JsonPostCallable | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Embedding API key/token must not be empty.")
        if not model.strip():
            raise ValueError("Embedding model must not be empty.")
        if not endpoint.strip():
            raise ValueError("Embedding endpoint must not be empty.")

        normalized_endpoint = endpoint.rstrip("/")
        if not normalized_endpoint.endswith("/embeddings"):
            normalized_endpoint = normalized_endpoint + "/embeddings"
        self._api_key = api_key
        self._model = model
        self._endpoint = normalized_endpoint
        self._post_json = post_json or _default_post_json

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty.")
        if any(not text.strip() for text in texts):
            raise ValueError("every text must be non-empty.")

        payload: dict[str, object] = {"model": self._model, "input": texts}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        response = self._post_json(self._endpoint, headers, payload)
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise OpenAIEmbeddingError("Embedding payload is missing expected vectors.")

        indexed_items = sorted(data, key=lambda item: int(item.get("index", -1)))
        vectors: list[list[float]] = []
        for item in indexed_items:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise OpenAIEmbeddingError("Embedding payload includes an invalid vector.")
            vectors.append([float(value) for value in embedding])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


# Backward-compatible alias.
OpenAIEmbeddingClient = GitHubModelsEmbeddingClient
