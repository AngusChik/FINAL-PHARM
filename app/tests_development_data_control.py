from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class DevelopmentDataControlSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "scripts" / "development-data.ps1"
        ).read_text(encoding="utf-8")

    def test_refresh_refuses_the_production_database(self):
        self.assertIn(
            "if ($developmentName -ieq $productionName)",
            self.source,
        )
        self.assertIn(
            "Development database '$developmentName' matches the production database",
            self.source,
        )
        self.assertIn(
            '$developmentName -iin @("postgres", "template0", "template1")',
            self.source,
        )

    def test_refresh_requires_explicit_development_environment(self):
        self.assertIn(
            '$environmentName -cne "development"',
            self.source,
        )

    def test_setup_generates_only_a_gitignored_development_configuration(self):
        self.assertIn("function Ensure-SecureDevelopmentEnvironment", self.source)
        self.assertIn('"PHARMACY_ENVIRONMENT=development"', self.source)
        self.assertIn('"DB_NAME=$requiredDevelopmentDatabase"', self.source)
        self.assertIn('"PRODUCTION_DB_NAME=$(ConvertTo-DotEnvValue $productionDatabase)"', self.source)
        self.assertIn('"DB_USER=$requiredDevelopmentRole"', self.source)
        self.assertIn('"DEVELOPMENT_TEST_DB_NAME=$requiredTestDatabase"', self.source)
        self.assertNotIn("EMAIL_HOST_PASSWORD=", self.source)
        self.assertNotIn("GSHEET_SPREADSHEET_ID=", self.source)
        self.assertIn(
            "PHARMACY_ENVIRONMENT=development",
            self.source,
        )

    def test_refresh_stops_at_a_running_development_server(self):
        self.assertIn("function Assert-DevelopmentStopped", self.source)
        self.assertIn(
            "Development is running. Stop it before replacing its database.",
            self.source,
        )
        self.assertIn("Development runtime state is unreadable", self.source)
        self.assertIn(
            '$client.BeginConnect("127.0.0.1", 8001',
            self.source,
        )
        self.assertIn(
            "Port 8001 is active without a safe stopped state",
            self.source,
        )
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

    def test_refresh_uses_verified_snapshot_and_post_restore_cleanup(self):
        self.assertIn('-Reason "development-refresh"', self.source)
        self.assertIn('Invoke-Native $pgRestore @("--list"', self.source)
        self.assertIn("manage.py prepare_development_snapshot", self.source)
        self.assertLess(
            self.source.index('Invoke-Native $pgRestore @("--list"'),
            self.source.index('Invoke-Native $dropdb @('),
        )
        self.assertIn("development-refresh-incomplete.json", self.source)
        self.assertLess(
            self.source.index("Set-Content -LiteralPath $refreshIncompleteMarker"),
            self.source.index('Invoke-Native $dropdb @('),
        )
        self.assertIn(
            "Rerun Refresh Development Data before starting development",
            self.source,
        )

    def test_refresh_only_removes_its_validated_runtime_child(self):
        self.assertIn(
            "[IO.Path]::GetDirectoryName($resolvedOperation) -cne $resolvedRoot",
            self.source,
        )
        self.assertIn(
            "Remove-Item -LiteralPath $resolvedOperation -Recurse -Force",
            self.source,
        )
