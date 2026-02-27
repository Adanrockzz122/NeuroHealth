from __future__ import annotations

from .models import SymptomReport, UrgencyLevel

EMERGENCY_KEYWORDS = {
    "chest pain",
    "shortness of breath",
    "severe bleeding",
    "stroke",
    "one-sided weakness",
    "slurred speech",
    "loss of consciousness",
    "suicidal",
    "anaphylaxis",
    "seizure",
}

URGENT_KEYWORDS = {
    "high fever",
    "persistent vomiting",
    "dehydration",
    "wheezing",
    "rapid heartbeat",
    "severe headache",
    "infection",
    "painful urination",
}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _extract_numeric(biometrics: dict[str, float | int | str], key: str) -> float | None:
    value = biometrics.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return float(normalized)
    return None


def assess_urgency(report: SymptomReport, user_input: str) -> tuple[UrgencyLevel, list[str]]:
    combined = _normalize(" ".join([user_input, *report.symptoms]))
    triggers: list[str] = []

    emergency_hits = [keyword for keyword in EMERGENCY_KEYWORDS if keyword in combined]
    if emergency_hits:
        triggers.extend([f"keyword:{keyword}" for keyword in emergency_hits])
        return "emergency", triggers

    oxygen_saturation = _extract_numeric(report.biometrics, "oxygen_saturation")
    if oxygen_saturation is not None and oxygen_saturation < 92.0:
        triggers.append("biometric:low_oxygen_saturation")
        return "emergency", triggers

    temperature_c = _extract_numeric(report.biometrics, "temperature_c")
    if temperature_c is not None and temperature_c >= 39.0:
        triggers.append("biometric:high_fever")
        return "urgent", triggers

    if report.pain_level is not None and report.pain_level >= 7:
        triggers.append("pain_level:>=7")
        return "urgent", triggers

    urgent_hits = [keyword for keyword in URGENT_KEYWORDS if keyword in combined]
    if urgent_hits:
        triggers.extend([f"keyword:{keyword}" for keyword in urgent_hits])
        return "urgent", triggers

    if report.symptoms or user_input.strip():
        triggers.append("symptoms_present")
        return "routine", triggers

    return "self_care", ["no_risk_signal"]


def appointment_for_urgency(urgency: UrgencyLevel) -> str:
    if urgency == "emergency":
        return "Go to the nearest emergency department now or call local emergency services."
    if urgency == "urgent":
        return "Book a same-day urgent care or telemedicine appointment."
    if urgency == "routine":
        return "Schedule a primary care appointment within 2-7 days."
    return "Self-care is reasonable; monitor symptoms and schedule routine care if symptoms persist."


def build_safety_instructions(urgency: UrgencyLevel) -> list[str]:
    base = [
        "This assistant provides educational guidance and does not replace professional medical diagnosis.",
    ]
    if urgency == "emergency":
        return base + [
            "Seek emergency care immediately.",
            "Do not delay care while waiting for additional online advice.",
        ]
    if urgency == "urgent":
        return base + [
            "Seek same-day clinical evaluation.",
            "Escalate to emergency care if breathing, consciousness, or severe pain worsens.",
        ]
    if urgency == "routine":
        return base + [
            "Track symptom progression and schedule a clinician follow-up.",
            "Escalate care if red-flag symptoms emerge.",
        ]
    return base + [
        "Continue monitoring symptoms and hydration/rest routines.",
        "Seek medical care if symptoms worsen or fail to improve.",
    ]


def build_clarifying_questions(report: SymptomReport, user_input: str) -> list[str]:
    questions: list[str] = []
    normalized_input = _normalize(user_input)

    if report.duration_hours is None:
        questions.append("How long have these symptoms been present?")
    if report.pain_level is None and "pain" in normalized_input:
        questions.append("On a 0-10 scale, what is your pain level right now?")
    if "fever" in normalized_input and "temperature_c" not in report.biometrics:
        questions.append("Do you have a measured temperature in Celsius?")
    if "cough" in normalized_input and "shortness of breath" not in normalized_input:
        questions.append("Is the cough dry or productive, and is breathing comfortable at rest?")

    return questions[:3]
