from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neurohealth.kb import KnowledgeIndex


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._to_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._to_vector(text)

    @staticmethod
    def _to_vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "fever" in lowered else 0.0,
            1.0 if "cough" in lowered else 0.0,
            1.0,
        ]


class KnowledgeIndexTests(unittest.TestCase):
    def test_retrieval_prefers_relevant_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / "kb.json"
            kb = [
                {
                    "id": "f1",
                    "title": "Fever guidance",
                    "content": "High fever should be monitored carefully.",
                    "source": "Source A",
                    "tags": ["fever"],
                },
                {
                    "id": "c1",
                    "title": "Cough guidance",
                    "content": "Mild cough can improve with rest.",
                    "source": "Source B",
                    "tags": ["cough"],
                },
            ]
            kb_path.write_text(json.dumps(kb), encoding="utf-8")

            embedder = FakeEmbeddingClient()
            index = KnowledgeIndex.build(kb_path, embedder)
            retrieved = index.retrieve("I have high fever", embedder, top_k=1)
            self.assertEqual(retrieved[0].id, "f1")


if __name__ == "__main__":
    unittest.main()
