import json
import shutil
import subprocess
from io import StringIO
from types import SimpleNamespace
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, mock_open, patch
from uuid import uuid4

from django.conf import settings
from django.core.management import call_command
from django.db import DatabaseError
from django.test import SimpleTestCase, TestCase

from .models import SupplierOrderRun
from .supplier_orders import (
    SCHEDULED_LAUNCH_START_TIMEOUT,
    SUPPLIER_ORDER_TASK_NAME,
    _launch_supplier_worker,
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
    def test_windows_direct_helper_never_relies_on_breakaway_flag(self):
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
    @patch('app.views.os.name', 'nt')
    @patch('app.supplier_orders.queue_scheduled_supplier_launch')
    def test_every_windows_web_start_uses_scheduled_broker(
            self, mock_queue, mock_direct):
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

    @patch('app.views._launch_order_process')
    @patch('app.views.os.name', 'posix')
    def test_non_windows_development_can_launch_directly(self, mock_direct):
        expected = object()
        mock_direct.return_value = expected
        run = SimpleNamespace(pk=18, vendor=SupplierOrderRun.VENDOR_MCKESSON)
        base = PurePosixPath('/pharmacy')

        process = _launch_or_schedule_order_process(
            run,
            ['python', 'mckesson_order.py'],
            base,
            base / 'logs' / 'mckesson_order.log',
        )

        self.assertIs(process, expected)
        mock_direct.assert_called_once()
    @patch('app.supplier_orders.subprocess.run')
    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_scheduled_request_contains_only_validated_run_metadata(
            self, _mock_windows, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, '', '')
        run = SimpleNamespace(
            pk=23,
            vendor=SupplierOrderRun.VENDOR_KOHLFRISCH,
            attempt=1,
        )

        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)
        marker = queue_scheduled_supplier_launch(run, base)
        payload = json.loads(marker.read_text(encoding='utf-8'))

        self.assertEqual(payload['run_id'], 23)
        self.assertEqual(payload['vendor'], SupplierOrderRun.VENDOR_KOHLFRISCH)
        self.assertEqual(payload['attempt'], 1)
        command = mock_run.call_args.args[0]
        self.assertEqual(
            command,
            ['schtasks.exe', '/Run', '/TN', SUPPLIER_ORDER_TASK_NAME],
        )

    @patch('app.supplier_orders.subprocess.run')
    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_task_scheduler_rejection_is_immediate_and_actionable(
            self, _mock_windows, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            [], 1, '', 'ERROR: Access is denied.',
        )
        run = SimpleNamespace(
            pk=24,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            attempt=1,
        )
        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)

        with self.assertRaisesRegex(
                OSError, r'signed in.*setup-main-computer\.bat'):
            queue_scheduled_supplier_launch(run, base)

        self.assertFalse((base / '.runtime' / 'supplier-order-24.launch').exists())

    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_cleanup_denial_does_not_mask_actionable_marker_error(self, _mock_windows):
        run = SimpleNamespace(
            pk=25,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            attempt=1,
        )
        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)

        with (
            patch.object(Path, 'write_text', side_effect=PermissionError(5, 'write denied')),
            patch.object(Path, 'unlink', side_effect=PermissionError(5, 'cleanup denied')),
        ):
            with self.assertRaisesRegex(
                    OSError,
                    r'create the supplier-launch request.*setup-main-computer\.bat.*write denied',
            ):
                queue_scheduled_supplier_launch(run, base)


class SupplierScheduledLaunchPersistenceTests(TestCase):
    def make_base(self):
        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)
        scripts = base / 'env' / 'Scripts'
        scripts.mkdir(parents=True)
        (scripts / 'python.exe').touch()
        (base / 'mckesson_order.py').touch()
        (base / 'kohlfrisch_order.py').touch()
        return base

    def test_mckesson_worker_is_stopped_if_pid_lease_cannot_be_saved(self):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_STARTING,
        )
        process = Mock(pid=48192)

        with patch.object(
            SupplierOrderRun,
            'save',
            side_effect=DatabaseError('save failed'),
        ):
            with self.assertRaises(DatabaseError):
                _launch_supplier_worker(
                    run.pk,
                    self.make_base(),
                    popen=Mock(return_value=process),
                    attempt=run.attempt,
                )

        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    def test_kohlfrisch_pid_save_failure_keeps_legacy_cleanup_behavior(self):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_KOHLFRISCH,
            state=SupplierOrderRun.STATE_STARTING,
        )
        process = Mock(pid=48193)

        with patch.object(
            SupplierOrderRun,
            'save',
            side_effect=DatabaseError('save failed'),
        ):
            with self.assertRaises(DatabaseError):
                _launch_supplier_worker(
                    run.pk,
                    self.make_base(),
                    popen=Mock(return_value=process),
                    attempt=run.attempt,
                )

        process.kill.assert_not_called()
        process.wait.assert_not_called()


class SupplierOrderingScheduledDispatchTests(TestCase):
    @patch('app.supplier_orders._launch_supplier_worker')
    @patch('app.supplier_orders.subprocess.run')
    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_mckesson_pid_save_failure_is_terminalized_by_dispatcher(
            self, _mock_windows, mock_task_run, mock_launch):
        mock_task_run.return_value = subprocess.CompletedProcess([], 0, '', '')
        mock_launch.side_effect = DatabaseError('lease save failed')
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_STARTING,
            message='Starting...',
        )
        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)
        marker = queue_scheduled_supplier_launch(run, base)

        results = dispatch_scheduled_supplier_launches(base_dir=base)

        self.assertFalse(marker.exists())
        run.refresh_from_db()
        self.assertEqual(run.state, SupplierOrderRun.STATE_ERROR)
        self.assertIsNone(run.process_id)
        self.assertIn('Could not persist scheduled McKesson launch', run.message)
        self.assertEqual(results[0]['run_id'], run.pk)
        self.assertIsNone(results[0]['pid'])

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
        self.assertEqual(command[-6:], [
            str(base / 'mckesson_order.py'),
            '--no-input',
            '--run-id',
            str(run.pk),
            '--attempt',
            str(run.attempt),
        ])

    @patch('app.supplier_orders.subprocess.run')
    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_old_broker_exit_cannot_overwrite_replacement_attempt(
            self, _mock_windows, mock_task_run):
        mock_task_run.return_value = subprocess.CompletedProcess([], 0, '', '')
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_STARTING,
            message='First attempt',
        )
        process = Mock(pid=4322)

        def finish_old_process():
            SupplierOrderRun.objects.filter(pk=run.pk).update(
                attempt=2,
                state=SupplierOrderRun.STATE_STARTING,
                process_id=None,
                message='Replacement attempt',
            )
            return 1

        process.wait.side_effect = finish_old_process
        popen = Mock(return_value=process)
        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)
        (base / 'env' / 'Scripts').mkdir(parents=True)
        (base / 'env' / 'Scripts' / 'python.exe').touch()
        (base / 'mckesson_order.py').touch()
        queue_scheduled_supplier_launch(run, base)

        dispatch_scheduled_supplier_launches(
            base_dir=base,
            popen=popen,
            wait_for_workers=True,
        )

        run.refresh_from_db()
        self.assertEqual(run.attempt, 2)
        self.assertEqual(run.state, SupplierOrderRun.STATE_STARTING)
        self.assertEqual(run.message, 'Replacement attempt')

    @patch('app.supplier_orders.subprocess.run')
    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_terminal_mckesson_worker_exit_clears_inner_worker_pid(
            self, _mock_windows, mock_task_run):
        mock_task_run.return_value = subprocess.CompletedProcess([], 0, '', '')
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_STARTING,
            message='Starting...',
        )
        process = Mock(pid=4323)

        def finish_handled_error():
            SupplierOrderRun.objects.filter(pk=run.pk).update(
                state=SupplierOrderRun.STATE_ERROR,
                process_id=9876,
                message='Portal failure retained',
            )
            return 1

        process.wait.side_effect = finish_handled_error
        popen = Mock(return_value=process)
        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)
        (base / 'env' / 'Scripts').mkdir(parents=True)
        (base / 'env' / 'Scripts' / 'python.exe').touch()
        (base / 'mckesson_order.py').touch()
        queue_scheduled_supplier_launch(run, base)

        dispatch_scheduled_supplier_launches(
            base_dir=base,
            popen=popen,
            wait_for_workers=True,
        )

        run.refresh_from_db()
        self.assertEqual(run.state, SupplierOrderRun.STATE_ERROR)
        self.assertEqual(run.message, 'Portal failure retained')
        self.assertIsNone(run.process_id)

    @patch('app.supplier_orders.subprocess.run')
    @patch('app.supplier_orders._is_windows', return_value=True)
    def test_terminal_kohlfrisch_worker_retains_existing_pid_behavior(
            self, _mock_windows, mock_task_run):
        mock_task_run.return_value = subprocess.CompletedProcess([], 0, '', '')
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_KOHLFRISCH,
            state=SupplierOrderRun.STATE_STARTING,
            message='Starting...',
        )
        process = Mock(pid=4324)

        def finish_handled_error():
            SupplierOrderRun.objects.filter(pk=run.pk).update(
                state=SupplierOrderRun.STATE_ERROR,
                process_id=9877,
                message='K&F failure retained',
            )
            return 1

        process.wait.side_effect = finish_handled_error
        popen = Mock(return_value=process)
        base = _test_runtime_directory()
        self.addCleanup(shutil.rmtree, base, True)
        (base / 'env' / 'Scripts').mkdir(parents=True)
        (base / 'env' / 'Scripts' / 'python.exe').touch()
        (base / 'kohlfrisch_order.py').touch()
        queue_scheduled_supplier_launch(run, base)

        dispatch_scheduled_supplier_launches(
            base_dir=base,
            popen=popen,
            wait_for_workers=True,
        )

        run.refresh_from_db()
        self.assertEqual(run.state, SupplierOrderRun.STATE_ERROR)
        self.assertEqual(run.message, 'K&F failure retained')
        self.assertEqual(run.process_id, 9877)

    @patch('app.views.os.name', 'nt')
    def test_unacknowledged_windows_broker_request_gets_repair_instructions(self):
        from django.utils import timezone
        from app.views import _supplier_run_status

        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_STARTING,
            message='Waiting for the Windows supplier launcher...',
        )
        SupplierOrderRun.objects.filter(pk=run.pk).update(
            updated_at=timezone.now() - SCHEDULED_LAUNCH_START_TIMEOUT,
        )

        payload = _supplier_run_status(SupplierOrderRun.VENDOR_MCKESSON)

        self.assertEqual(payload['state'], SupplierOrderRun.STATE_ERROR)
        self.assertIn('did not acknowledge', payload['message'])
        self.assertIn('setup-main-computer.bat', payload['message'])


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
