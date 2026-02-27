from __future__ import annotations

import unittest

from neurohealth.models import SymptomReport
from neurohealth.safety import (
    appointment_for_urgency,
    assess_urgency,
    build_clarifying_questions,
)


class SafetyTests(unittest.TestCase):
    def test_emergency_for_chest_pain(self) -> None:
        report = SymptomReport(symptoms=["chest pain", "shortness of breath"], pain_level=8)
        urgency, triggers = assess_urgency(report, "I have chest pain and feel breathless")
        self.assertEqual(urgency, "emergency")
        self.assertTrue(any("keyword:chest pain" == trigger for trigger in triggers))

    def test_urgent_for_high_fever(self) -> None:
        report = SymptomReport(symptoms=["fever"], biometrics={"temperature_c": 39.2}, pain_level=4)
        urgency, _ = assess_urgency(report, "My fever is high")
        self.assertEqual(urgency, "urgent")

    def test_routine_when_symptoms_present_without_red_flags(self) -> None:
        report = SymptomReport(symptoms=["mild cough"], pain_level=2)
        urgency, _ = assess_urgency(report, "I have mild cough for one day")
        self.assertEqual(urgency, "routine")

    def test_appointment_mapping(self) -> None:
        self.assertIn("emergency", appointment_for_urgency("emergency").lower())
        self.assertIn("same-day", appointment_for_urgency("urgent").lower())
        self.assertIn("primary care", appointment_for_urgency("routine").lower())

    def test_clarifying_questions(self) -> None:
        report = SymptomReport(symptoms=["fever"])
        questions = build_clarifying_questions(report, "I have fever and pain")
        self.assertGreaterEqual(len(questions), 2)


if __name__ == "__main__":
    unittest.main()
