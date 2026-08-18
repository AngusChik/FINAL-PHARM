import json
import shutil
import subprocess
from io import StringIO
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, mock_open, patch
from uuid import uuid4

from django.conf import settings
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from .models import SupplierOrderRun
from .supplier_orders import (
    SUPPLIER_ORDER_TASK_NAME,
    dispatch_scheduled_supplier_launches,
    queue_scheduled_supplier_launch,
)
from .views import (
    _launch_or_schedule_order_process,
    _launch_order_process,
    _order_process_creationflags,
)


def _test_runtime_directory():
    runtime_dir = Path(settings.BASE_DIR) / '.runtime'
    runtime_dir.mkdir(exist_ok=True)
    path = runtime_dir / f'supplier-order-test-{uuid4().hex}'
    path.mkdir()
    return path


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

    @patch('app.views._launch_order_process')
    @patch('app.views._windows_process_in_job', return_value=True)
    @patch('app.supplier_orders.queue_scheduled_supplier_launch')
    def test_job_constrained_server_uses_scheduled_broker(
            self, mock_queue, _mock_in_job, mock_direct):
        run = SimpleNamespace(
            pk=17,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )
        base = Path(r'C:\pharmacy')

        process = _launch_or_schedule_order_process(
            run,
            ['python.exe', 'mckesson_order.py'],
            base,
            base / 'logs' / 'mckesson_order.log',
        )

        self.assertIsNone(process)
        mock_queue.assert_called_once_with(run, base)
        mock_direct.assert_not_called()

    @patch('app.supplier_orders.subprocess.run')
    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_scheduled_request_contains_only_validated_run_metadata(
            self, _mock_windows, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, '', '')
        run = SimpleNamespace(
            pk=23,
            vendor=SupplierOrderRun.VENDOR_KOHLFRISCH,
        )

        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)
        marker = queue_scheduled_supplier_launch(run, base)
        payload = json.loads(marker.read_text(encoding='utf-8'))

        self.assertEqual(payload['run_id'], 23)
        self.assertEqual(payload['vendor'], SupplierOrderRun.VENDOR_KOHLFRISCH)
        command = mock_run.call_args.args[0]
        self.assertEqual(
            command,
            ['schtasks.exe', '/Run', '/TN', SUPPLIER_ORDER_TASK_NAME],
        )


class SupplierOrderingScheduledDispatchTests(TestCase):
    @patch('app.supplier_orders.subprocess.run')
    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_scheduled_dispatch_launches_exact_database_run(
            self, _mock_windows, mock_task_run):
        mock_task_run.return_value = subprocess.CompletedProcess([], 0, '', '')
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_STARTING,
            message='Starting...',
        )
        process = Mock(pid=4321)
        popen = Mock(return_value=process)

        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)
        (base / 'env' / 'Scripts').mkdir(parents=True)
        (base / 'env' / 'Scripts' / 'python.exe').touch()
        (base / 'mckesson_order.py').touch()
        marker = queue_scheduled_supplier_launch(run, base)

        results = dispatch_scheduled_supplier_launches(
            base_dir=base,
            popen=popen,
        )

        self.assertFalse(marker.exists())

        run.refresh_from_db()
        self.assertEqual(run.process_id, 4321)
        self.assertEqual(results, [{'run_id': run.pk, 'pid': 4321, 'error': ''}])
        command = popen.call_args.args[0]
        self.assertEqual(command[-4:], [
            str(base / 'mckesson_order.py'),
            '--no-input',
            '--run-id',
            str(run.pk),
        ])


class SupplierOrderingBrokerSmokeTests(SimpleTestCase):
    @patch(
        'app.management.commands.launch_supplier_orders._run_browser_smoke'
    )
    @patch(
        'app.management.commands.launch_supplier_orders.'
        'dispatch_scheduled_supplier_launches',
        return_value=[],
    )
    def test_idle_broker_uses_local_browser_smoke(
            self, mock_dispatch, mock_browser_smoke):
        output = StringIO()

        call_command(
            'launch_supplier_orders',
            '--browser-smoke-if-idle',
            stdout=output,
        )

        mock_dispatch.assert_called_once_with(wait_for_workers=True)
        mock_browser_smoke.assert_called_once_with()
        self.assertIn('browser smoke passed', output.getvalue())
