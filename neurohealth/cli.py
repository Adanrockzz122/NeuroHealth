from __future__ import annotations

import argparse
import json
from uuid import uuid4

from .config import Settings
from .models import ConversationTurn, NeuroHealthRequest, RecommendationFeedback, SymptomReport, UserProfile
from .runtime import build_engine, parse_biometrics, parse_csv, parse_optional_int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroHealth prototype CLI")
    parser.add_argument("--query", help="Latest user health question/message.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode.")
    parser.add_argument("--age", type=int, default=None, help="User age.")
    parser.add_argument("--preferences", default="", help="Comma-separated user preferences.")
    parser.add_argument(
        "--medical-constraints",
        default="",
        help="Comma-separated medical constraints (e.g., knee pain, hypertension).",
    )
    parser.add_argument(
        "--chronic-conditions",
        default="",
        help="Comma-separated chronic conditions.",
    )
    parser.add_argument(
        "--health-literacy",
        choices=["basic", "intermediate", "advanced"],
        default="intermediate",
        help="Health literacy level for response adaptation.",
    )
    parser.add_argument("--symptoms", default="", help="Comma-separated symptoms.")
    parser.add_argument("--duration-hours", type=int, default=None, help="Symptom duration in hours.")
    parser.add_argument("--pain-level", type=int, default=None, help="Pain level (0-10).")
    parser.add_argument(
        "--biometric",
        action="append",
        default=[],
        help="Repeatable key=value biometric input (e.g., temperature_c=38.4).",
    )
    parser.add_argument("--feedback-rating", type=int, default=None, help="Optional feedback rating (1-5).")
    parser.add_argument("--feedback-comment", default="", help="Optional feedback comment.")
    return parser


def _build_profile(args: argparse.Namespace) -> UserProfile:
    return UserProfile(
        age=args.age,
        preferences=parse_csv(args.preferences),
        medical_constraints=parse_csv(args.medical_constraints),
        chronic_conditions=parse_csv(args.chronic_conditions),
        health_literacy=args.health_literacy,
    )


def _build_request(args: argparse.Namespace, profile: UserProfile, history: list[ConversationTurn]) -> NeuroHealthRequest:
    if not args.query:
        raise ValueError("--query is required unless running --interactive.")
    symptom_report = SymptomReport(
        symptoms=parse_csv(args.symptoms),
        biometrics=parse_biometrics(args.biometric),
        duration_hours=args.duration_hours,
        pain_level=args.pain_level,
    )
    return NeuroHealthRequest(
        user_input=args.query,
        user_profile=profile,
        symptom_report=symptom_report,
        history=history,
    )


def _run_interactive(engine: NeuroHealthEngine, profile: UserProfile) -> None:
    print("NeuroHealth interactive mode. Type 'exit' to quit.")
    history: list[ConversationTurn] = []
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            return
        if not query:
            print("Please enter a message.")
            continue

        symptoms = parse_csv(input("Symptoms (comma-separated, optional): ").strip())
        duration_hours = parse_optional_int(input("Duration hours (optional): "))
        pain_level = parse_optional_int(input("Pain level 0-10 (optional): "))
        biometrics_text = input("Biometrics key=value (comma-separated, optional): ").strip()
        biometrics_items = [item.strip() for item in biometrics_text.split(",") if item.strip()]
        symptom_report = SymptomReport(
            symptoms=symptoms,
            duration_hours=duration_hours,
            pain_level=pain_level,
            biometrics=parse_biometrics(biometrics_items),
        )
        request = NeuroHealthRequest(
            user_input=query,
            user_profile=profile,
            symptom_report=symptom_report,
            history=history,
        )
        recommendation = engine.generate(request)
        print("\nAssistant:")
        print(recommendation.assistant_message)
        print(f"\nUrgency: {recommendation.urgency}")
        print(f"Appointment: {recommendation.appointment_recommendation}")
        print("Safety instructions:")
        for instruction in recommendation.safety_instructions:
            print(f"- {instruction}")
        if recommendation.clarifying_questions:
            print("Clarifying questions:")
            for question in recommendation.clarifying_questions:
                print(f"- {question}")
        print("")

        history.append(ConversationTurn(role="user", content=query))
        history.append(ConversationTurn(role="assistant", content=recommendation.assistant_message))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()
    engine = build_engine(settings)
    profile = _build_profile(args)

    if args.interactive:
        _run_interactive(engine, profile)
        return

    request = _build_request(args, profile, history=[])
    recommendation = engine.generate(request)
    print(json.dumps(recommendation.to_dict(), indent=2))

    if args.feedback_rating is not None:
        feedback = RecommendationFeedback(
            conversation_id=str(uuid4()),
            rating=args.feedback_rating,
            comment=args.feedback_comment,
        )
        engine.record_feedback(feedback)


if __name__ == "__main__":
    main()
