import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

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

        with TemporaryDirectory(
            prefix="production-settings-test-",
        ) as temporary_directory:
            role_root = Path(temporary_directory).resolve()
            runtime = role_root / ".runtime"
            runtime.mkdir()
            (role_root / ".env").write_text("", encoding="utf-8")
            (runtime / "production-role.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "role": "production",
                    "worktree": str(role_root),
                    "branch": "main",
                    "remote": "origin",
                    "created_at": "2026-08-28T00:00:00+00:00",
                }),
                encoding="utf-8",
            )
            env["PHARMACY_PRODUCTION_ROLE_ROOT"] = str(role_root)
            env["PHARMACY_PRODUCTION_ENV_FILE"] = str(role_root / ".env")

            # Disable automatic .env discovery and isolate the git branch
            # probe so this passkey test reaches only the settings contract it
            # owns, independent of the developer's checkout and private files.
            command = (
                "from types import SimpleNamespace; "
                "import dotenv; "
                "dotenv.load_dotenv = lambda *args, **kwargs: False; "
                "import inventory.production_guard as guard; "
                "guard.subprocess.run = lambda *args, **kwargs: "
                "SimpleNamespace(returncode=0, "
                "stdout=('main\\n' if 'symbolic-ref' in args[0] else ''), "
                "stderr=''); "
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
