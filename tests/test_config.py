from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from neurohealth.config import Settings


class SettingsTests(unittest.TestCase):
    def test_from_env_loads_dotenv_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            Path(temp_dir, ".env").write_text(
                "\n".join(
                    [
                        "GEMINI_API_KEY=test-gemini",
                        "GITHUB_TOKEN=test-token",
                        "GEMINI_MODEL=gemini-2.0-flash",
                        "GITHUB_EMBEDDING_MODEL=openai/text-embedding-3-small",
                    ]
                ),
                encoding="utf-8",
            )
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                settings = Settings.from_env()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(settings.gemini_api_key, "test-gemini")
            self.assertEqual(settings.github_token, "test-token")

    def test_from_env_falls_back_to_openai_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            Path(temp_dir, ".env").write_text(
                "\n".join(
                    [
                        "GEMINI_API_KEY=test-gemini",
                        "OPENAI_API_KEY=legacy-token",
                    ]
                ),
                encoding="utf-8",
            )
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                settings = Settings.from_env()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(settings.github_token, "legacy-token")


if __name__ == "__main__":
    unittest.main()
