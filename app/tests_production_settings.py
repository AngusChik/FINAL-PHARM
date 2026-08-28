import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from inventory.production_guard import validate_production_role


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


class ProductionRoleGitStatusTests(SimpleTestCase):
    @staticmethod
    def _role_root(temporary_directory):
        role_root = Path(temporary_directory).resolve()
        runtime = role_root / ".runtime"
        runtime.mkdir()
        (role_root / ".env").write_text("", encoding="utf-8")
        (runtime / "production-role.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "role": "production",
                    "worktree": str(role_root),
                    "branch": "main",
                    "remote": "origin",
                    "created_at": "2026-08-28T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        return role_root

    @staticmethod
    def _result(stdout="", returncode=0, stderr=""):
        return SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def test_clean_status_succeeds_without_sleeping(self):
        with TemporaryDirectory(prefix="production-role-test-") as temporary:
            role_root = self._role_root(temporary)
            results = [self._result("main\n"), self._result()]
            with (
                patch(
                    "inventory.production_guard.subprocess.run",
                    side_effect=results,
                ) as run,
                patch("inventory.production_guard.time.sleep") as sleep,
            ):
                self.assertEqual(validate_production_role(role_root), role_root)

        self.assertEqual(run.call_count, 2)
        sleep.assert_not_called()
        self.assertIn("--no-optional-locks", run.call_args_list[-1].args[0])

    def test_transient_git_failure_is_retried_and_must_finish_clean(self):
        with TemporaryDirectory(prefix="production-role-test-") as temporary:
            role_root = self._role_root(temporary)
            results = [
                self._result("main\n"),
                self._result(returncode=128, stderr="transient git failure"),
                self._result(),
            ]
            with (
                patch("inventory.production_guard.subprocess.run", side_effect=results) as run,
                patch("inventory.production_guard.time.sleep") as sleep,
            ):
                self.assertEqual(validate_production_role(role_root), role_root)

        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 1)
        status_command = run.call_args_list[-1].args[0]
        self.assertIn("--no-optional-locks", status_command)

    def test_dirty_status_fails_immediately_without_retry(self):
        with TemporaryDirectory(prefix="production-role-test-") as temporary:
            role_root = self._role_root(temporary)
            dirty = self._result("?? unexpected.py\n")
            results = [self._result("main\n"), dirty]
            with (
                patch(
                    "inventory.production_guard.subprocess.run",
                    side_effect=results,
                ) as run,
                patch("inventory.production_guard.time.sleep") as sleep,
            ):
                with self.assertRaisesMessage(
                    ImproperlyConfigured,
                    "Production settings require a clean authorized main worktree.",
                ):
                    validate_production_role(role_root)

        self.assertEqual(run.call_count, 2)
        sleep.assert_not_called()

    def test_persistent_git_failure_still_fails_closed(self):
        with TemporaryDirectory(prefix="production-role-test-") as temporary:
            role_root = self._role_root(temporary)
            failed = self._result(returncode=128, stderr="git failure")
            results = [self._result("main\n"), failed, failed, failed]
            with (
                patch(
                    "inventory.production_guard.subprocess.run",
                    side_effect=results,
                ),
                patch("inventory.production_guard.time.sleep") as sleep,
            ):
                with self.assertRaisesMessage(
                    ImproperlyConfigured,
                    "Git could not verify production worktree cleanliness "
                    "after 3 attempts (exit 128): git failure",
                ):
                    validate_production_role(role_root)

        self.assertEqual(sleep.call_count, 2)

    def test_nonzero_then_dirty_still_fails_on_dirty_result(self):
        with TemporaryDirectory(prefix="production-role-test-") as temporary:
            role_root = self._role_root(temporary)
            results = [
                self._result("main\n"),
                self._result(returncode=128, stderr="index locked"),
                self._result(" M inventory/settings.py\n"),
            ]
            with (
                patch(
                    "inventory.production_guard.subprocess.run",
                    side_effect=results,
                ) as run,
                patch("inventory.production_guard.time.sleep") as sleep,
            ):
                with self.assertRaisesMessage(
                    ImproperlyConfigured,
                    "Production settings require a clean authorized main worktree.",
                ):
                    validate_production_role(role_root)

        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 1)

    def test_git_timeout_fails_immediately(self):
        with TemporaryDirectory(prefix="production-role-test-") as temporary:
            role_root = self._role_root(temporary)
            results = [
                self._result("main\n"),
                subprocess.TimeoutExpired(["git", "status"], 15),
            ]
            with (
                patch(
                    "inventory.production_guard.subprocess.run",
                    side_effect=results,
                ) as run,
                patch("inventory.production_guard.time.sleep") as sleep,
            ):
                with self.assertRaisesMessage(
                    ImproperlyConfigured,
                    "Git timed out while verifying production worktree cleanliness.",
                ):
                    validate_production_role(role_root)

        self.assertEqual(run.call_count, 2)
        sleep.assert_not_called()
