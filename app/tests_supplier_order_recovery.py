import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import RequestFactory, SimpleTestCase, TestCase
from django.utils import timezone

from .models import SupplierOrderPlan, SupplierOrderRun, SupplierOrderRunItem
from .supplier_orders import DatabaseRunStatus
from .views import (
    MCKESSON_ACTIVE_STATES,
    OrderControlView,
    _start_supplier_run,
    _supplier_run_status,
    _windows_supplier_process_tree,
)

import kohlfrisch_order
import mckesson_order


class SupplierOrderFrozenRunRecoveryTests(TestCase):
    """Regression coverage for browser workers that remain alive but stop reporting."""

    HEARTBEAT_FIELD_CANDIDATES = (
        'heartbeat_at',
        'last_heartbeat_at',
        'worker_heartbeat_at',
        'last_heartbeat',
        'worker_heartbeat',
    )

    def set_heartbeat_age(self, run, age):
        """Age either the current timestamp or a dedicated future heartbeat field."""
        heartbeat = timezone.now() - age
        field_names = {
            field.name for field in SupplierOrderRun._meta.get_fields()
        }
        updates = {'updated_at': heartbeat}
        for candidate in self.HEARTBEAT_FIELD_CANDIDATES:
            if candidate in field_names:
                updates[candidate] = heartbeat
        SupplierOrderRun.objects.filter(pk=run.pk).update(**updates)
        run.refresh_from_db()

    def create_run(self, *, state, age=timedelta(0), process_id=48192):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=state,
            process_id=process_id,
            message='Last worker update',
        )
        self.set_heartbeat_age(run, age)
        return run

    @patch('app.views._pid_alive', return_value=True)
    def test_old_running_heartbeat_is_reclaimed_even_when_pid_still_exists(
            self, _mock_pid_alive):
        run = self.create_run(
            state=SupplierOrderRun.STATE_RUNNING,
            age=timedelta(hours=1),
        )

        payload = _supplier_run_status(SupplierOrderRun.VENDOR_MCKESSON)

        run.refresh_from_db()
        self.assertNotIn(payload['state'], MCKESSON_ACTIVE_STATES)
        self.assertNotIn(run.state, MCKESSON_ACTIVE_STATES)
        self.assertIsNotNone(run.completed_at)
        self.assertNotEqual(run.message, 'Last worker update')

    @patch('app.views._pid_alive', return_value=True)
    def test_fresh_worker_heartbeat_is_not_reclaimed(self, mock_pid_alive):
        run = self.create_run(state=SupplierOrderRun.STATE_RUNNING)

        payload = _supplier_run_status(SupplierOrderRun.VENDOR_MCKESSON)

        run.refresh_from_db()
        self.assertEqual(payload['state'], SupplierOrderRun.STATE_RUNNING)
        self.assertEqual(run.state, SupplierOrderRun.STATE_RUNNING)
        self.assertIsNone(run.completed_at)
        mock_pid_alive.assert_called_once_with(48192)

    def test_late_old_worker_update_cannot_resurrect_a_terminal_run(self):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_RUNNING,
            message='First attempt',
        )
        old_worker = DatabaseRunStatus(
            SupplierOrderRun.VENDOR_MCKESSON,
            run.pk,
            attempt=run.attempt,
        )
        SupplierOrderRun.objects.filter(pk=run.pk).update(
            attempt=run.attempt + 1,
            state=SupplierOrderRun.STATE_STARTING,
            process_id=None,
            message='Replacement attempt',
        )

        updated = old_worker.update(
            state=SupplierOrderRun.STATE_RUNNING,
            message='Late update from the abandoned worker',
        )

        run.refresh_from_db()
        self.assertFalse(updated)
        self.assertEqual(run.attempt, 2)
        self.assertEqual(run.state, SupplierOrderRun.STATE_STARTING)
        self.assertEqual(run.message, 'Replacement attempt')

    @patch('app.views._launch_or_schedule_order_process')
    @patch('app.views._pid_alive', return_value=False)
    def test_new_run_can_start_after_closed_review_worker_has_exited(
            self, _mock_pid_alive, mock_launch):
        stale = self.create_run(
            state=SupplierOrderRun.STATE_REVIEW,
            age=timedelta(hours=1),
        )
        user = get_user_model().objects.create_user(
            username='supplier-recovery-admin',
            password='not-used',
        )
        request = RequestFactory().post(
            '/low-stock/mckesson-order/start/',
            data=json.dumps({
                'items': [{
                    'product_id': None,
                    'name': 'Recovery test item',
                    'barcode': '64642000001',
                    'quantity': 1,
                }],
            }),
            content_type='application/json',
        )
        request.user = user
        mock_launch.return_value = SimpleNamespace(pid=59304)

        response = _start_supplier_run(
            request,
            SupplierOrderRun.VENDOR_MCKESSON,
            'mckesson_order.py',
        )

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.content)
        self.assertTrue(body['ok'])

        stale.refresh_from_db()
        self.assertNotIn(stale.state, MCKESSON_ACTIVE_STATES)
        self.assertIsNotNone(stale.completed_at)

        replacement = SupplierOrderRun.objects.get(pk=body['run_id'])
        self.assertNotEqual(replacement.pk, stale.pk)
        self.assertEqual(replacement.state, SupplierOrderRun.STATE_STARTING)
        self.assertEqual(replacement.process_id, 59304)
        mock_launch.assert_called_once()


class SupplierOrderExplicitRetryTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='supplier-retry-admin',
            password='not-used',
        )
        self.plan = SupplierOrderPlan.objects.create(
            created_by=self.user,
            vendor_sequence=[SupplierOrderRun.VENDOR_MCKESSON],
            status=SupplierOrderPlan.STATUS_RUNNING,
            started_at=timezone.now() - timedelta(minutes=10),
        )
        self.run = SupplierOrderRun.objects.create(
            plan=self.plan,
            created_by=self.user,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_ERROR,
            message='The browser stopped responding.',
            current=2,
            total=3,
            process_id=48192,
            pause_requested=True,
            cancel_requested=True,
            started_at=timezone.now() - timedelta(minutes=9),
            completed_at=timezone.now() - timedelta(minutes=1),
        )
        SupplierOrderRunItem.objects.bulk_create([
            SupplierOrderRunItem(
                run=self.run,
                product_name='Already added',
                barcode='64642000001',
                quantity_requested=1,
                position=0,
                outcome=SupplierOrderRunItem.OUTCOME_ADDED,
                processed_at=timezone.now() - timedelta(minutes=5),
            ),
            SupplierOrderRunItem(
                run=self.run,
                product_name='Still pending',
                barcode='64642000002',
                quantity_requested=2,
                position=1,
                outcome=SupplierOrderRunItem.OUTCOME_PENDING,
            ),
            SupplierOrderRunItem(
                run=self.run,
                product_name='Already skipped',
                barcode='64642000003',
                quantity_requested=1,
                position=2,
                outcome=SupplierOrderRunItem.OUTCOME_SKIPPED,
                reason='Unavailable',
                processed_at=timezone.now() - timedelta(minutes=4),
            ),
        ])

    def post_retry(self):
        request = RequestFactory().post(
            '/low-stock/order-control/',
            data=json.dumps({
                'action': 'retry',
                'vendor': SupplierOrderRun.VENDOR_MCKESSON,
                'plan_id': self.plan.pk,
                'run_id': self.run.pk,
            }),
            content_type='application/json',
        )
        request.user = self.user
        return OrderControlView().post(request)

    def normal_start_request(self):
        request = RequestFactory().post(
            '/low-stock/mckesson-order/start/',
            data=json.dumps({
                'plan_id': self.plan.pk,
                'items': [{
                    'product_id': None,
                    'name': 'Still pending',
                    'barcode': '64642000002',
                    'quantity': 2,
                }],
            }),
            content_type='application/json',
        )
        request.user = self.user
        return request

    @patch('app.views._launch_or_schedule_order_process')
    def test_explicit_retry_reuses_terminal_run_and_launches_pending_once(
            self, mock_launch):
        launch_snapshot = {}

        def launch(reused_run, command, _base, _log_path):
            reused_run.refresh_from_db()
            launch_snapshot.update({
                'run_id': reused_run.pk,
                'state': reused_run.state,
                'current': reused_run.current,
                'total': reused_run.total,
                'process_id': reused_run.process_id,
                'pause_requested': reused_run.pause_requested,
                'cancel_requested': reused_run.cancel_requested,
                'started_at': reused_run.started_at,
                'completed_at': reused_run.completed_at,
                'attempt': reused_run.attempt,
                'command': command,
            })
            return SimpleNamespace(pid=59305)

        mock_launch.side_effect = launch

        first = self.post_retry()
        second = self.post_retry()

        self.assertEqual(first.status_code, 200)
        self.assertEqual(json.loads(first.content)['run_id'], self.run.pk)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(mock_launch.call_count, 1)
        self.assertEqual(SupplierOrderRun.objects.count(), 1)

        self.assertEqual(launch_snapshot['run_id'], self.run.pk)
        self.assertEqual(launch_snapshot['state'], SupplierOrderRun.STATE_STARTING)
        self.assertEqual(launch_snapshot['current'], 0)
        self.assertEqual(launch_snapshot['total'], 1)
        self.assertIsNone(launch_snapshot['process_id'])
        self.assertFalse(launch_snapshot['pause_requested'])
        self.assertFalse(launch_snapshot['cancel_requested'])
        self.assertIsNone(launch_snapshot['started_at'])
        self.assertIsNone(launch_snapshot['completed_at'])
        self.assertEqual(launch_snapshot['attempt'], 2)
        self.assertEqual(
            launch_snapshot['command'][-4:],
            ['--run-id', str(self.run.pk), '--attempt', '2'],
        )

        self.run.refresh_from_db()
        self.assertEqual(self.run.process_id, 59305)
        outcomes = list(
            self.run.items.order_by('position').values_list(
                'product_name', 'outcome',
            )
        )
        self.assertEqual(outcomes, [
            ('Already added', SupplierOrderRunItem.OUTCOME_ADDED),
            ('Still pending', SupplierOrderRunItem.OUTCOME_PENDING),
            ('Already skipped', SupplierOrderRunItem.OUTCOME_SKIPPED),
        ])

        worker = object.__new__(DatabaseRunStatus)
        worker.run = self.run
        worker.attempt = self.run.attempt
        pending = worker.pending_items()
        self.assertEqual(
            [(item['name'], item['quantity']) for item in pending],
            [('Still pending', 2)],
        )

    @patch('app.views._launch_or_schedule_order_process')
    @patch('app.views._terminate_supplier_process_tree', return_value=True)
    @patch('app.views._pid_alive', return_value=True)
    def test_retry_stops_verified_frozen_worker_before_new_attempt(
            self, _mock_alive, mock_terminate, mock_launch):
        mock_launch.return_value = SimpleNamespace(pid=59306)

        response = self.post_retry()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_terminate.call_count, 1)
        self.assertEqual(mock_terminate.call_args.args[0].pk, self.run.pk)
        self.run.refresh_from_db()
        self.assertEqual(self.run.attempt, 2)
        self.assertEqual(self.run.process_id, 59306)

    @patch('app.views._launch_or_schedule_order_process')
    @patch('app.views._terminate_supplier_process_tree', return_value=False)
    @patch('app.views._pid_alive', return_value=True)
    def test_retry_refuses_replacement_when_frozen_tree_cannot_be_stopped(
            self, _mock_alive, mock_terminate, mock_launch):
        response = self.post_retry()

        self.assertEqual(response.status_code, 409)
        self.assertIn('No replacement browser', json.loads(response.content)['error'])
        mock_terminate.assert_called_once()
        mock_launch.assert_not_called()
        self.run.refresh_from_db()
        self.assertEqual(self.run.attempt, 1)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)

    @patch('app.views._launch_or_schedule_order_process')
    def test_same_plan_terminal_run_still_conflicts_without_explicit_retry(
            self, mock_launch):
        response = _start_supplier_run(
            self.normal_start_request(),
            SupplierOrderRun.VENDOR_MCKESSON,
            'mckesson_order.py',
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(json.loads(response.content)['run_id'], self.run.pk)
        mock_launch.assert_not_called()
        self.run.refresh_from_db()
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)


class SupplierWorkerIdentityTests(TestCase):
    @patch('app.views.os.name', 'nt')
    @patch('app.views.subprocess.run')
    def test_windows_tree_requires_exact_run_and_attempt_command(
            self, mock_run):
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )
        mock_run.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps([
                {
                    'ProcessId': 48192,
                    'ParentProcessId': 100,
                    'Name': 'python.exe',
                    'CommandLine': (
                        'python.exe mckesson_order.py --no-input '
                        '--run-id 42 --attempt 3'
                    ),
                },
                {
                    'ProcessId': 48193,
                    'ParentProcessId': 48192,
                    'Name': 'chrome.exe',
                    'CommandLine': 'chrome.exe --type=browser',
                },
            ]),
        )

        tree = _windows_supplier_process_tree(run)

        self.assertEqual(tree, [48192, 48193])

    @patch('app.views.os.name', 'nt')
    @patch('app.views.subprocess.run')
    def test_windows_tree_rejects_recycled_pid_for_another_attempt(
            self, mock_run):
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )
        mock_run.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                'ProcessId': 48192,
                'ParentProcessId': 100,
                'Name': 'python.exe',
                'CommandLine': (
                    'python.exe mckesson_order.py --run-id 42 --attempt 2'
                ),
            }),
        )

        self.assertIsNone(_windows_supplier_process_tree(run))


class SupplierWorkerHeartbeatTests(SimpleTestCase):
    def test_paused_workers_continue_heartbeating(self):
        for module in (mckesson_order, kohlfrisch_order):
            with self.subTest(module=module.__name__):
                status = Mock()
                status.control.side_effect = [
                    {'pause_requested': True},
                    {'pause_requested': True},
                    {'pause_requested': False},
                ]
                status.update.return_value = True
                with patch.object(module.time, 'sleep'):
                    result = module.control_gate(status, None, 4)

                self.assertEqual(result, 'continue')
                self.assertEqual(status.update.call_count, 2)


class SupplierControlManagerSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(__file__).resolve().parent / 'templates' / 'low_stock.html'
        ).read_text(encoding='utf-8')

    def test_cancelled_worker_aborts_remaining_supplier_sequence(self):
        self.assertIn("if (s === 'cancelled') seqAborted = true;", self.source)
        self.assertIn('finishPlan(seqAborted)', self.source)

    def test_start_failures_hold_for_explicit_retry(self):
        self.assertIn(
            'holdForRetry(key, runners[key].data, finishOne, doStart)',
            self.source,
        )
        self.assertIn("action: 'retry'", self.source)
        self.assertIn('Stop frozen worker & retry', self.source)
