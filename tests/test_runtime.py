from __future__ import annotations

import unittest

from neurohealth.runtime import parse_biometrics, parse_csv, parse_optional_int


class RuntimeParsingTests(unittest.TestCase):
    def test_parse_csv(self) -> None:
        self.assertEqual(parse_csv("a, b,,c "), ["a", "b", "c"])
        self.assertEqual(parse_csv(""), [])
        self.assertEqual(parse_csv(None), [])

    def test_parse_optional_int(self) -> None:
        self.assertEqual(parse_optional_int(" 42 "), 42)
        self.assertIsNone(parse_optional_int(""))
        self.assertIsNone(parse_optional_int(None))

    def test_parse_biometrics(self) -> None:
        parsed = parse_biometrics(["temperature_c=37.8", "heart_rate=89", "note=slight cough"])
        self.assertEqual(parsed["temperature_c"], 37.8)
        self.assertEqual(parsed["heart_rate"], 89)
        self.assertEqual(parsed["note"], "slight cough")

    def test_parse_biometrics_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_biometrics(["invalid"])


if __name__ == "__main__":
    unittest.main()
