from __future__ import annotations

import json
from collections.abc import Callable
from urllib import error, request

JsonPostCallable = Callable[[str, dict[str, str], dict[str, object]], dict[str, object]]


class GeminiError(RuntimeError):
    """Raised when the Gemini API request fails."""


def _default_post_json(url: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise GeminiError(f"Gemini HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise GeminiError(f"Gemini network error: {exc.reason}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiError("Gemini returned invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise GeminiError("Gemini returned an unexpected payload type.")
    return parsed


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        endpoint_template: str = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        post_json: JsonPostCallable | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key must not be empty.")
        if not model.strip():
            raise ValueError("Gemini model must not be empty.")
        if not endpoint_template.strip():
            raise ValueError("Gemini endpoint template must not be empty.")

        self._api_key = api_key
        self._model = model
        self._endpoint_template = endpoint_template
        self._post_json = post_json or _default_post_json

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        endpoint = self._endpoint_template.format(model=self._model, key=self._api_key)
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "topP": 0.9},
        }
        if system_instruction:
            payload["system_instruction"] = {"parts": [{"text": system_instruction}]}

        response = self._post_json(endpoint, {}, payload)
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise GeminiError("Gemini response did not include candidates.")

        first_candidate = candidates[0]
        content = first_candidate.get("content")
        if not isinstance(content, dict):
            raise GeminiError("Gemini response included an invalid content payload.")

        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise GeminiError("Gemini response did not include text parts.")

        text_parts: list[str] = []
        for part in parts:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

        if not text_parts:
            raise GeminiError("Gemini response did not include textual output.")
        return "\n".join(text_parts)
