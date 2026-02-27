from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

UrgencyLevel = Literal["self_care", "routine", "urgent", "emergency"]


@dataclass
class UserProfile:
    age: int | None = None
    preferences: list[str] = field(default_factory=list)
    medical_constraints: list[str] = field(default_factory=list)
    chronic_conditions: list[str] = field(default_factory=list)
    health_literacy: Literal["basic", "intermediate", "advanced"] = "intermediate"

    def __post_init__(self) -> None:
        if self.age is not None and self.age <= 0:
            raise ValueError("age must be a positive integer when provided.")


@dataclass
class SymptomReport:
    symptoms: list[str] = field(default_factory=list)
    biometrics: dict[str, float | int | str] = field(default_factory=dict)
    duration_hours: int | None = None
    pain_level: int | None = None

    def __post_init__(self) -> None:
        if self.duration_hours is not None and self.duration_hours < 0:
            raise ValueError("duration_hours must be >= 0 when provided.")
        if self.pain_level is not None and not (0 <= self.pain_level <= 10):
            raise ValueError("pain_level must be in the [0, 10] range when provided.")


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError("role must be either 'user' or 'assistant'.")
        if not self.content.strip():
            raise ValueError("content must not be empty.")


@dataclass
class NeuroHealthRequest:
    user_input: str
    user_profile: UserProfile = field(default_factory=UserProfile)
    symptom_report: SymptomReport = field(default_factory=SymptomReport)
    history: list[ConversationTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.user_input.strip():
            raise ValueError("user_input must not be empty.")


@dataclass
class KnowledgeChunk:
    id: str
    title: str
    content: str
    source: str
    tags: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk(KnowledgeChunk):
    score: float = 0.0


@dataclass
class Recommendation:
    assistant_message: str
    urgency: UrgencyLevel
    appointment_recommendation: str
    safety_instructions: list[str]
    sources: list[str]
    reasoning_notes: list[str] = field(default_factory=list)
    clarifying_questions: list[str] = field(default_factory=list)
    needs_emergency: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RecommendationFeedback:
    conversation_id: str
    rating: int
    comment: str = ""

    def __post_init__(self) -> None:
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be empty.")
        if not (1 <= self.rating <= 5):
            raise ValueError("rating must be in the [1, 5] range.")
