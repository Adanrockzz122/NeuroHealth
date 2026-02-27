from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neurohealth.engine import NeuroHealthEngine
from neurohealth.feedback import FeedbackStore
from neurohealth.kb import KnowledgeIndex
from neurohealth.models import NeuroHealthRequest, RecommendationFeedback, SymptomReport, UserProfile


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
            1.0 if "nutrition" in lowered or "vegetarian" in lowered else 0.0,
            1.0,
        ]


class FakeGeminiClient:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.call_count += 1
        if not prompt:
            raise ValueError("prompt cannot be empty")
        return "Hydrate, monitor symptoms, and book follow-up care based on urgency."


class NeuroHealthEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)

        kb_path = base / "kb.json"
        kb_payload = [
            {
                "id": "fever",
                "title": "Fever basics",
                "content": "Hydration and rest are recommended for mild fever.",
                "source": "MedlinePlus Fever Guidance",
                "tags": ["fever"],
            },
            {
                "id": "nutrition",
                "title": "Vegetarian planning",
                "content": "Vegetarian protein planning can support healthy weight goals.",
                "source": "WHO Healthy Diet Summary",
                "tags": ["nutrition"],
            },
        ]
        kb_path.write_text(json.dumps(kb_payload), encoding="utf-8")

        self.embedding_client = FakeEmbeddingClient()
        self.knowledge_index = KnowledgeIndex.build(kb_path, self.embedding_client)
        self.llm_client = FakeGeminiClient()
        self.feedback_store = FeedbackStore(base / "feedback.jsonl")
        self.engine = NeuroHealthEngine(
            knowledge_index=self.knowledge_index,
            embedding_client=self.embedding_client,
            llm_client=self.llm_client,
            feedback_store=self.feedback_store,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_emergency_path_bypasses_llm(self) -> None:
        request = NeuroHealthRequest(
            user_input="I have chest pain and shortness of breath",
            user_profile=UserProfile(age=40),
            symptom_report=SymptomReport(symptoms=["chest pain", "shortness of breath"], pain_level=9),
        )
        recommendation = self.engine.generate(request)
        self.assertTrue(recommendation.needs_emergency)
        self.assertEqual(recommendation.urgency, "emergency")
        self.assertEqual(self.llm_client.call_count, 0)

    def test_non_emergency_path_uses_llm(self) -> None:
        request = NeuroHealthRequest(
            user_input="I am vegetarian and have mild fever",
            user_profile=UserProfile(age=32, preferences=["vegetarian"]),
            symptom_report=SymptomReport(symptoms=["mild fever"], pain_level=2),
        )
        recommendation = self.engine.generate(request)
        self.assertFalse(recommendation.needs_emergency)
        self.assertEqual(self.llm_client.call_count, 1)
        self.assertIn("Hydrate", recommendation.assistant_message)
        self.assertTrue(recommendation.sources)

    def test_feedback_persistence(self) -> None:
        feedback = RecommendationFeedback(
            conversation_id="test-convo",
            rating=5,
            comment="Helpful and clear guidance.",
        )
        self.engine.record_feedback(feedback)
        records = self.feedback_store.read_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].rating, 5)


if __name__ == "__main__":
    unittest.main()
