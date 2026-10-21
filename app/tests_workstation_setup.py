from pathlib import Path
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory
from unittest import skipUnless

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


@skipUnless(os.name == 'nt', 'Windows workstation kit builder.')
class WorkstationKitTests(SimpleTestCase):
    def test_kit_uses_active_root_and_new_desktop_url_without_private_files(self):
        with TemporaryDirectory(prefix='pharmacy-kit-test-') as temporary:
            root = Path(temporary)
            scripts = root / 'scripts'
            scripts.mkdir()
            for name in ('build-workstation-kit.ps1', 'setup-workstation.ps1'):
                shutil.copyfile(Path(settings.BASE_DIR) / 'scripts' / name, scripts / name)
            shutil.copyfile(Path(settings.BASE_DIR) / 'setup-workstation.bat', root / 'setup-workstation.bat')
            (root / '.env').write_text('PHARMACY_HOST=192.168.0.22\nDB_PASSWORD=test-secret-not-for-kit\n', encoding='utf-8')
            (root / 'Pharmacy-Root-Certificate.crt').write_text('stale exported certificate', encoding='utf-8')
            authority = root / 'caddy_data' / 'caddy' / 'pki' / 'authorities' / 'local'
            authority.mkdir(parents=True)
            (authority / 'root.crt').write_text('current server public root', encoding='utf-8')
            (authority / 'root.key').write_text('private key must stay on server', encoding='utf-8')

            result = subprocess.run(
                ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                 '-File', str(scripts / 'build-workstation-kit.ps1')],
                capture_output=True, text=True, timeout=20, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            kit = root / 'workstation-kit'
            self.assertEqual((kit / 'server-url.txt').read_text().strip(), 'https://192.168.0.22')
            self.assertEqual((kit / 'Pharmacy-Root-Certificate.crt').read_text(), 'current server public root')
            self.assertIn('https://192.168.0.22/connect-phone/', (kit / 'README.txt').read_text(encoding='utf-8-sig'))
            self.assertSetEqual({path.relative_to(kit).as_posix() for path in kit.rglob('*') if path.is_file()}, {
                'Pharmacy-Root-Certificate.crt', 'setup-workstation.bat',
                'scripts/setup-workstation.ps1', 'server-url.txt', 'README.txt',
            })
