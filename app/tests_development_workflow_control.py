from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class DevelopmentWorkflowControlSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "scripts" / "development.ps1"
        ).read_text(encoding="utf-8")

    def test_start_validates_database_isolation_before_migrating(self):
        assertion_start = self.source.index(
            "function Assert-DevelopmentConfiguration"
        )
        assertion_end = self.source.index(
            "function Invoke-ControllerScript", assertion_start
        )
        assertion = self.source[assertion_start:assertion_end]

        self.assertIn(".env.development is missing", assertion)
        self.assertIn("-Action status", assertion)
        self.assertIn("Development database isolation validation failed", assertion)

        start = self.source.index("function Start-Development")
        migrate = self.source.index("manage.py migrate --noinput", start)
        assert_call = self.source.index("Assert-DevelopmentConfiguration", start)
        self.assertLess(assert_call, migrate)

    def test_menu_exposes_development_and_release_operations(self):
        for label in (
            "Set up isolated development database",
            "Refresh development data from production snapshot",
            "Run release checks",
            "Publish tested release (production, then GitHub)",
            "Open production",
            "Open production logs",
        ):
            with self.subTest(label=label):
                self.assertIn(label, self.source)

    def test_development_is_localhost_only(self):
        self.assertIn('$bindHost = "127.0.0.1"', self.source)
        self.assertIn("Development is localhost-only", self.source)
        self.assertIn("if ($PortNumber -ne 8001)", self.source)
        self.assertIn("Development is fixed to localhost port 8001", self.source)
        self.assertNotIn('$bindHost = if ($AllowLan)', self.source)

    def test_development_state_identifies_the_exact_managed_process(self):
        for field in (
            '"pid"',
            '"port"',
            '"project_root"',
            '"python_path"',
            '"process_start_utc"',
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.source)
        self.assertIn("$process.StartTime.ToUniversalTime()", self.source)
        self.assertIn("[IO.Path]::GetFullPath($process.Path)", self.source)
        self.assertIn("Port 8001 is active but its process identity", self.source)
        self.assertIn("Wait-TcpPortClosed 8001", self.source)
        self.assertIn("Invoke-ElevatedDevelopmentStop", self.source)
        self.assertIn('@("/PID", "$ProcessId", "/T", "/F")', self.source)
        self.assertNotIn("ElevatedRetry", self.source)

    def test_runserver_uses_a_stable_process_tree_on_windows(self):
        start = self.source.index("function Start-Development")
        stop = self.source.index("function Stop-Development", start)
        start_source = self.source[start:stop]

        self.assertIn('"manage.py", "runserver"', start_source)
        self.assertIn('"--noreload"', start_source)
        self.assertNotIn("Auto-reload: enabled", self.source)
        self.assertIn(
            "use Restart development after code changes",
            start_source,
        )

    def test_production_commands_resolve_the_isolated_worktree(self):
        self.assertIn(
            'Join-Path $runtimeDir "development-workflow.json"',
            self.source,
        )
        self.assertIn("production_worktree", self.source)
        self.assertIn(
            'Join-Path $productionRoot "scripts\\production.ps1"',
            self.source,
        )

    def test_release_commands_use_the_dedicated_controller(self):
        self.assertIn(
            'Join-Path $PSScriptRoot "publish-release.ps1"',
            self.source,
        )
        self.assertIn('Invoke-ReleaseController "check"', self.source)
        self.assertIn('Invoke-ReleaseController "publish"', self.source)
