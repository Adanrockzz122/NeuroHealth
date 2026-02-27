from __future__ import annotations

from typing import Protocol

from .feedback import FeedbackStore
from .kb import EmbeddingClient, KnowledgeIndex
from .models import NeuroHealthRequest, Recommendation, RecommendationFeedback
from .prompts import build_reasoning_prompt, build_system_instruction
from .safety import (
    appointment_for_urgency,
    assess_urgency,
    build_clarifying_questions,
    build_safety_instructions,
)


class LLMClient(Protocol):
    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        ...


class NeuroHealthEngine:
    def __init__(
        self,
        knowledge_index: KnowledgeIndex,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        feedback_store: FeedbackStore,
    ) -> None:
        self._knowledge_index = knowledge_index
        self._embedding_client = embedding_client
        self._llm_client = llm_client
        self._feedback_store = feedback_store

    def generate(self, request: NeuroHealthRequest) -> Recommendation:
        urgency, triggers = assess_urgency(request.symptom_report, request.user_input)
        appointment = appointment_for_urgency(urgency)
        safety = build_safety_instructions(urgency)
        clarifying_questions = build_clarifying_questions(request.symptom_report, request.user_input)
        reasoning_notes = [f"urgency:{urgency}", *[f"trigger:{trigger}" for trigger in triggers]]

        if urgency == "emergency":
            emergency_message = (
                "Your symptoms may indicate a medical emergency. "
                "Please seek immediate emergency care now."
            )
            return Recommendation(
                assistant_message=emergency_message,
                urgency=urgency,
                appointment_recommendation=appointment,
                safety_instructions=safety,
                sources=[],
                reasoning_notes=reasoning_notes,
                clarifying_questions=[],
                needs_emergency=True,
            )

        retrieved = self._knowledge_index.retrieve(
            query=request.user_input,
            embedding_client=self._embedding_client,
            top_k=4,
        )
        prompt = build_reasoning_prompt(
            request=request,
            retrieved_chunks=retrieved,
            urgency=urgency,
            appointment_recommendation=appointment,
            safety_instructions=safety,
            clarifying_questions=clarifying_questions,
        )
        generated = self._llm_client.generate(
            prompt=prompt,
            system_instruction=build_system_instruction(),
        )

        unique_sources = sorted({chunk.source for chunk in retrieved})
        return Recommendation(
            assistant_message=generated,
            urgency=urgency,
            appointment_recommendation=appointment,
            safety_instructions=safety,
            sources=unique_sources,
            reasoning_notes=reasoning_notes,
            clarifying_questions=clarifying_questions,
            needs_emergency=False,
        )

    def record_feedback(self, feedback: RecommendationFeedback) -> None:
        self._feedback_store.record(feedback)
