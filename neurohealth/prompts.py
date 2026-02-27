from __future__ import annotations

import json
from dataclasses import asdict

from .models import NeuroHealthRequest, RetrievedChunk, UrgencyLevel


def build_system_instruction() -> str:
    return (
        "You are NeuroHealth, a clinically cautious AI health assistant. "
        "Ground recommendations in supplied medical snippets, adapt language to user literacy, "
        "include nutrition and planning advice only when contextually relevant, and avoid definitive diagnoses. "
        "Always prioritize patient safety and include explicit escalation cues."
    )


def build_reasoning_prompt(
    request: NeuroHealthRequest,
    retrieved_chunks: list[RetrievedChunk],
    urgency: UrgencyLevel,
    appointment_recommendation: str,
    safety_instructions: list[str],
    clarifying_questions: list[str],
) -> str:
    profile_json = json.dumps(asdict(request.user_profile), indent=2)
    symptom_json = json.dumps(asdict(request.symptom_report), indent=2)
    history_json = json.dumps([asdict(turn) for turn in request.history], indent=2)
    knowledge_context = "\n".join(
        [
            f"- [{chunk.source}] {chunk.title}: {chunk.content}"
            for chunk in retrieved_chunks
        ]
    )
    clarifications = "\n".join([f"- {question}" for question in clarifying_questions]) or "- None"
    safety_text = "\n".join([f"- {instruction}" for instruction in safety_instructions])

    return f"""
User profile:
{profile_json}

Symptom report:
{symptom_json}

Conversation history:
{history_json}

Latest user message:
{request.user_input}

Retrieved validated health knowledge:
{knowledge_context}

Current urgency label: {urgency}
Proposed appointment recommendation: {appointment_recommendation}

Safety instructions to include:
{safety_text}

Clarifying questions to ask if needed:
{clarifications}

Produce a concise, user-friendly response with this structure:
1) Personalized guidance summary
2) Nutrition/planning advice relevant to profile + symptoms
3) Appointment recommendation with urgency rationale
4) Safety instructions and escalation cues
5) Optional clarifying questions (only if truly needed)
""".strip()
