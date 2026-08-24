from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class ProductionControlSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "scripts" / "production.ps1"
        ).read_text(encoding="utf-8")

    def test_taskkill_stderr_cannot_abort_restart_before_verification(self):
        stop_start = self.source.index("function Stop-TrackedProcessTree")
        stop_end = self.source.index("function Wait-TcpPortClosed", stop_start)
        stop_source = self.source[stop_start:stop_end]

        self.assertIn('$ErrorActionPreference = "Continue"', stop_source)
        self.assertIn("& taskkill.exe /PID $ProcessId /T /F 2>&1", stop_source)
        self.assertIn("Stop-Process -Id $treeProcessId", stop_source)
        self.assertIn("Get-Process -Id $ProcessId", stop_source)
        self.assertIn("Could not stop tracked production process", stop_source)

    def test_stop_verifies_ports_before_discarding_process_state(self):
        stop_start = self.source.index("function Stop-Production")
        stop_end = self.source.index("function Show-Status", stop_start)
        stop_source = self.source[stop_start:stop_end]

        port_8000 = stop_source.index("Wait-TcpPortClosed 8000")
        port_443 = stop_source.index("Wait-TcpPortClosed 443")
        remove_state = stop_source.index("Remove-Item -LiteralPath $pidFile")
        self.assertLess(port_8000, remove_state)
        self.assertLess(port_443, remove_state)

    def test_pid_reuse_is_fenced_by_identity_and_start_time(self):
        self.assertIn('$process.ProcessName -notin $allowedNames', self.source)
        self.assertIn('waitress_started_at = $waitressProcess.StartTime', self.source)
        self.assertIn('caddy_started_at = $caddyProcess.StartTime', self.source)
        self.assertIn('[Math]::Abs(($actualStart - $expectedStart).TotalSeconds)', self.source)

    def test_access_denied_shutdown_retries_once_with_uac(self):
        self.assertIn("[switch]$ElevatedRetry", self.source)
        self.assertIn('function Invoke-ElevatedProductionStop', self.source)
        self.assertIn('-Verb RunAs', self.source)
        self.assertIn('$detail -match "Access is denied"', self.source)
        self.assertIn('-not $ElevatedRetry', self.source)
        self.assertIn('-Action stop -NoBrowser -ElevatedRetry', self.source)
