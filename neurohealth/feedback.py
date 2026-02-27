from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import RecommendationFeedback


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, feedback: RecommendationFeedback) -> None:
        payload = json.dumps(asdict(feedback), ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    def read_all(self) -> list[RecommendationFeedback]:
        if not self.path.exists():
            return []

        records: list[RecommendationFeedback] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid feedback JSON at line {line_number}.") from exc
                records.append(RecommendationFeedback(**payload))
        return records
