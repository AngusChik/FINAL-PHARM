from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class WorkstationSetupSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        scripts = Path(settings.BASE_DIR) / "scripts"
        cls.setup_source = (scripts / "setup-workstation.ps1").read_text(
            encoding="utf-8"
        )
        cls.kit_source = (scripts / "build-workstation-kit.ps1").read_text(
            encoding="utf-8"
        )

    def test_workstation_accepts_only_an_https_origin(self):
        self.assertIn('$uri.Scheme -ne "https"', self.setup_source)
        self.assertIn("$uri.UserInfo", self.setup_source)
        self.assertIn("$uri.AbsolutePath -ne \"/\"", self.setup_source)

    def test_workstation_installs_only_certificate_and_url_shortcut(self):
        self.assertIn("certutil.exe -user -addstore Root", self.setup_source)
        self.assertIn('"[InternetShortcut]"', self.setup_source)
        self.assertIn('"URL=$ServerUrl"', self.setup_source)
        for forbidden in ("git clone", "python -m", "postgresql", "caddy run"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.setup_source.lower())

    def test_kit_contains_only_public_workstation_material(self):
        for expected in (
            "Pharmacy-Root-Certificate.crt",
            "setup-workstation.bat",
            "setup-workstation.ps1",
            "server-url.txt",
            "README.txt",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.kit_source)
        self.assertNotIn("Copy-Item -LiteralPath $envFile", self.kit_source)
        for forbidden in ("DB_PASSWORD", "google_credentials.json"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.kit_source)
