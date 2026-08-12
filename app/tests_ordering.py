import subprocess
from pathlib import Path
from unittest.mock import mock_open, patch

from django.test import SimpleTestCase

from .views import (
    _launch_order_process,
    _order_process_creationflags,
)


class SupplierOrderingProcessTests(SimpleTestCase):
    @patch('app.views.os.name', 'nt')
    @patch('app.views._windows_process_in_job', return_value=True)
    def test_windows_job_process_uses_breakaway_flag(self, _mock_in_job):
        flags = _order_process_creationflags()

        self.assertTrue(flags & getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))
        self.assertTrue(flags & getattr(subprocess, 'CREATE_BREAKAWAY_FROM_JOB', 0x01000000))

    @patch('app.views.os.name', 'nt')
    @patch('app.views._windows_process_in_job', return_value=False)
    def test_regular_windows_process_does_not_request_breakaway(self, _mock_in_job):
        flags = _order_process_creationflags()

        self.assertTrue(flags & getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))
        self.assertFalse(flags & getattr(subprocess, 'CREATE_BREAKAWAY_FROM_JOB', 0x01000000))

    @patch('app.views.subprocess.Popen')
    def test_launcher_isolates_server_standard_input(self, mock_popen):
        opened = mock_open()

        with patch('builtins.open', opened):
            _launch_order_process(
                ['python.exe', 'mckesson_order.py'],
                Path(r'C:\pharmacy'),
                Path(r'C:\pharmacy\logs\mckesson_order.log'),
            )

        kwargs = mock_popen.call_args.kwargs
        self.assertIs(kwargs['stdin'], subprocess.DEVNULL)
        self.assertIs(kwargs['stderr'], subprocess.STDOUT)
        self.assertTrue(kwargs['close_fds'])
        self.assertEqual(kwargs['cwd'], r'C:\pharmacy')
