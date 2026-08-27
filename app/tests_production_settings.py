import os
from pathlib import Path
import subprocess
import sys

from django.test import SimpleTestCase


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MISSING = object()


class ProductionAdminPasskeySettingsTests(SimpleTestCase):
    @staticmethod
    def _import_production_settings(passkey=_MISSING):
        env = os.environ.copy()
        env["DJANGO_SECRET_KEY"] = "test-only-production-secret-key"
        env.pop("ADMIN_PASSKEY", None)
        if passkey is not _MISSING:
            env["ADMIN_PASSKEY"] = passkey

        # Disable automatic .env discovery so each subprocess sees only the
        # configuration supplied by this test, independent of a developer's
        # local untracked .env file.
        command = (
            "import dotenv; "
            "dotenv.load_dotenv = lambda *args, **kwargs: False; "
            "import inventory.settings_production"
        )
        return subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_production_rejects_unsafe_admin_passkeys(self):
        unsafe_values = (
            ("missing", _MISSING),
            ("empty", ""),
            ("whitespace-only", "   "),
            ("leading whitespace", " test-only-private-admin-passkey"),
            ("trailing whitespace", "test-only-private-admin-passkey "),
            ("shorter than minimum", "short-key"),
            ("source default", "pharmacy-admin"),
            ("example placeholder", "replace-with-a-private-admin-passkey"),
        )

        for label, value in unsafe_values:
            with self.subTest(label=label):
                result = self._import_production_settings(value)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "Production requires a private ADMIN_PASSKEY of at least 12 characters",
                    result.stderr,
                )

    def test_production_accepts_a_private_admin_passkey(self):
        result = self._import_production_settings(
            "test-only-private-admin-passkey"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
