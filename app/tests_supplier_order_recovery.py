import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.test import RequestFactory, SimpleTestCase, TestCase
from django.utils import timezone

from .models import (
    SupplierOrderPlan,
    SupplierOrderPlanItem,
    SupplierOrderRun,
    SupplierOrderRunItem,
    UserAction,
)
from .supplier_orders import DatabaseRunStatus
from .views import (
    MCKESSON_PROCESS_MATCHED,
    MCKESSON_PROCESS_GONE,
    MCKESSON_PROCESS_OTHER_WORKER,
    MCKESSON_PROCESS_UNRELATED,
    MCKESSON_PROCESS_UNKNOWN,
    MCKESSON_ACTIVE_STATES,
    OrderControlView,
    SupplierOrderPlanView,
    _clear_stale_mckesson_process_id,
    _inspect_mckesson_worker_process,
    _mckesson_snapshot_process_state,
    _start_supplier_run,
    _supplier_run_status,
    _terminate_mckesson_process_tree,
    _windows_mckesson_process_snapshot,
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

    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_MATCHED,
    )
    @patch('app.views._mckesson_windows_pid_liveness', return_value='alive')
    def test_old_running_heartbeat_is_reclaimed_even_when_pid_still_exists(
            self, _mock_liveness, _mock_identity):
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

    @patch('app.views._mckesson_windows_pid_liveness', return_value='alive')
    def test_fresh_worker_heartbeat_is_not_reclaimed(self, mock_liveness):
        run = self.create_run(state=SupplierOrderRun.STATE_RUNNING)

        payload = _supplier_run_status(SupplierOrderRun.VENDOR_MCKESSON)

        run.refresh_from_db()
        self.assertEqual(payload['state'], SupplierOrderRun.STATE_RUNNING)
        self.assertEqual(run.state, SupplierOrderRun.STATE_RUNNING)
        self.assertIsNone(run.completed_at)
        mock_liveness.assert_called_once_with(48192)

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
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_GONE,
    )
    @patch(
        'app.views._mckesson_windows_pid_liveness',
        return_value=MCKESSON_PROCESS_GONE,
    )
    def test_new_run_can_start_after_closed_review_worker_has_exited(
            self, _mock_liveness, _mock_identity, mock_launch):
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

    def post_retry(self, user=None):
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
        request.user = user or self.user
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
        self.assertEqual(
            SupplierOrderRun.objects.filter(plan=self.plan).count(),
            1,
        )

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
    @patch('app.views._terminate_mckesson_process_tree', return_value=True)
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_MATCHED,
    )
    def test_retry_stops_verified_frozen_worker_before_new_attempt(
            self, _mock_identity, mock_terminate, mock_launch):
        mock_launch.return_value = SimpleNamespace(pid=59306)

        response = self.post_retry()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_terminate.call_count, 1)
        self.assertEqual(mock_terminate.call_args.args[0].pk, self.run.pk)
        self.run.refresh_from_db()
        self.assertEqual(self.run.attempt, 2)
        self.assertEqual(self.run.process_id, 59306)

    @patch('app.views._launch_or_schedule_order_process')
    @patch('app.views._terminate_mckesson_process_tree', return_value=False)
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_MATCHED,
    )
    def test_retry_refuses_replacement_when_frozen_tree_cannot_be_stopped(
            self, _mock_identity, mock_terminate, mock_launch):
        response = self.post_retry()

        self.assertEqual(response.status_code, 409)
        self.assertIn('No replacement browser', json.loads(response.content)['error'])
        mock_terminate.assert_called_once()
        mock_launch.assert_not_called()
        self.run.refresh_from_db()
        self.assertEqual(self.run.attempt, 1)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)

    @patch('app.views._launch_or_schedule_order_process')
    @patch('app.views._terminate_mckesson_process_tree')
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_UNRELATED,
    )
    def test_retry_ignores_recycled_non_supplier_pid_without_killing_it(
            self, _mock_identity, mock_terminate, mock_launch):
        mock_launch.return_value = SimpleNamespace(pid=59307)

        response = self.post_retry()

        self.assertEqual(response.status_code, 200)
        mock_terminate.assert_not_called()
        mock_launch.assert_called_once()
        self.run.refresh_from_db()
        self.assertEqual(self.run.attempt, 2)
        self.assertEqual(self.run.process_id, 59307)

    @patch('app.views._launch_or_schedule_order_process')
    @patch('app.views._terminate_mckesson_process_tree')
    @patch('app.views._clear_stale_mckesson_process_id', return_value=False)
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_UNRELATED,
    )
    def test_retry_fails_closed_when_stale_pid_clear_loses_race(
            self, _mock_identity, mock_clear, mock_terminate, mock_launch):
        response = self.post_retry()

        self.assertEqual(response.status_code, 409)
        self.assertIn('worker changed', json.loads(response.content)['error'])
        mock_clear.assert_called_once()
        mock_terminate.assert_not_called()
        mock_launch.assert_not_called()
        self.run.refresh_from_db()
        self.assertEqual(self.run.attempt, 1)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)

    @patch('app.views._launch_or_schedule_order_process')
    @patch('app.views._terminate_mckesson_process_tree')
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_UNKNOWN,
    )
    def test_retry_with_unknown_process_identity_fails_closed(
            self, _mock_identity, mock_terminate, mock_launch):
        response = self.post_retry()

        self.assertEqual(response.status_code, 409)
        self.assertIn('could not safely verify', json.loads(response.content)['error'])
        mock_terminate.assert_not_called()
        mock_launch.assert_not_called()
        self.run.refresh_from_db()
        self.assertEqual(self.run.attempt, 1)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)
        self.assertEqual(self.run.process_id, 48192)

    @patch('app.views._launch_or_schedule_order_process')
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_GONE,
    )
    def test_other_admin_can_take_over_pending_mckesson_retry(
            self, _mock_identity, mock_launch):
        other_admin = get_user_model().objects.create_user(
            username='mck-recovery-admin',
            password='not-used',
            is_staff=True,
        )
        mock_launch.return_value = SimpleNamespace(pid=59308)

        response = self.post_retry(other_admin)

        self.assertEqual(response.status_code, 200)
        self.run.refresh_from_db()
        self.plan.refresh_from_db()
        self.assertEqual(self.run.created_by, self.user)
        self.assertEqual(self.plan.created_by, self.user)
        self.assertEqual(self.plan.mckesson_recovery_claimed_by, other_admin)
        self.assertIsNotNone(self.plan.mckesson_recovery_claimed_at)
        self.assertEqual(self.run.attempt, 2)
        self.assertEqual(self.run.process_id, 59308)
        self.assertTrue(UserAction.objects.filter(
            user=other_admin,
            action='supplier_order_update',
            target=f'McKesson automation plan #{self.plan.pk}',
        ).exists())

        self.run.state = SupplierOrderRun.STATE_REVIEW
        self.run.save(update_fields=['state', 'updated_at'])
        get_request = RequestFactory().get('/low-stock/supplier-plan/')
        get_request.user = other_admin
        get_response = SupplierOrderPlanView().get(get_request)
        self.assertEqual(
            json.loads(get_response.content)['plan']['id'],
            self.plan.pk,
        )

        finish_request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'finish',
                'plan_id': self.plan.pk,
                'cancelled': False,
            }),
            content_type='application/json',
        )
        finish_request.user = other_admin
        finish_response = SupplierOrderPlanView().post(finish_request)
        self.assertEqual(finish_response.status_code, 200)
        self.plan.refresh_from_db()
        self.assertEqual(self.plan.status, SupplierOrderPlan.STATUS_COMPLETED)

    @patch('app.views._launch_or_schedule_order_process')
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_GONE,
    )
    def test_long_mckesson_retry_launch_error_is_truncated(
            self, _mock_identity, mock_launch):
        mock_launch.side_effect = OSError('x' * 800)

        response = self.post_retry()

        self.assertEqual(response.status_code, 500)
        self.run.refresh_from_db()
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)
        self.assertEqual(len(self.run.message), 500)

    @patch('app.views._launch_or_schedule_order_process')
    def test_finished_plan_cannot_be_resurrected_by_stale_retry(self, mock_launch):
        self.plan.status = SupplierOrderPlan.STATUS_CANCELLED
        self.plan.completed_at = timezone.now()
        self.plan.save(update_fields=['status', 'completed_at'])

        response = self.post_retry()

        self.assertEqual(response.status_code, 409)
        mock_launch.assert_not_called()
        self.run.refresh_from_db()
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)
        self.assertEqual(self.run.attempt, 1)

    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_UNRELATED,
    )
    def test_status_clears_recycled_pid_but_preserves_pending_review_barrier(
            self, _mock_identity):
        payload = _supplier_run_status(
            SupplierOrderRun.VENDOR_MCKESSON,
            plan_id=self.plan.pk,
        )

        self.run.refresh_from_db()
        self.assertIsNone(self.run.process_id)
        self.assertFalse(payload['worker_alive'])
        self.assertFalse(payload['worker_identity_uncertain'])
        self.assertTrue(payload['can_retry'])
        self.assertTrue(payload['requires_resolution'])
        self.assertEqual(payload['pending_count'], 1)

    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_UNRELATED,
    )
    @patch('app.views._clear_stale_mckesson_process_id')
    def test_status_fails_closed_when_stale_pid_clear_loses_race(
            self, mock_clear, _mock_identity):
        def replacement_won(run):
            SupplierOrderRun.objects.filter(pk=run.pk).update(process_id=59312)
            run.refresh_from_db()
            return False

        mock_clear.side_effect = replacement_won

        payload = _supplier_run_status(
            SupplierOrderRun.VENDOR_MCKESSON,
            plan_id=self.plan.pk,
        )

        self.assertEqual(payload['pid'], 59312)
        self.assertFalse(payload['worker_alive'])
        self.assertTrue(payload['worker_identity_uncertain'])
        self.assertFalse(payload['can_retry'])

    @patch('app.views._launch_or_schedule_order_process')
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_MATCHED,
    )
    def test_verified_terminal_worker_blocks_normal_start(
            self, _mock_identity, mock_launch):
        response = _start_supplier_run(
            self.normal_start_request(),
            SupplierOrderRun.VENDOR_MCKESSON,
            'mckesson_order.py',
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(json.loads(response.content)['run_id'], self.run.pk)
        mock_launch.assert_not_called()

    @patch('app.views._launch_or_schedule_order_process')
    @patch(
        'app.views._inspect_mckesson_worker_process',
        return_value=MCKESSON_PROCESS_UNKNOWN,
    )
    def test_unknown_terminal_worker_blocks_normal_start(
            self, _mock_identity, mock_launch):
        response = _start_supplier_run(
            self.normal_start_request(),
            SupplierOrderRun.VENDOR_MCKESSON,
            'mckesson_order.py',
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn('could not safely verify', json.loads(response.content)['error'])
        mock_launch.assert_not_called()

    def test_new_mckesson_plan_is_blocked_until_failed_run_is_resolved(self):
        request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'create',
                'seq': [SupplierOrderRun.VENDOR_MCKESSON],
                'items': [{
                    'product_id': None,
                    'name': 'New plan item',
                    'barcode': '64642000009',
                    'quantity': 1,
                }],
            }),
            content_type='application/json',
        )
        request.user = self.user

        response = SupplierOrderPlanView().post(request)

        self.assertEqual(response.status_code, 409)
        body = json.loads(response.content)
        self.assertTrue(body['requires_resolution'])
        self.assertEqual(body['run_id'], self.run.pk)
        self.assertEqual(SupplierOrderPlan.objects.count(), 1)

    def test_cancelled_failed_plan_releases_new_plan_barrier(self):
        self.plan.status = SupplierOrderPlan.STATUS_CANCELLED
        self.plan.completed_at = timezone.now()
        self.plan.save(update_fields=['status', 'completed_at'])
        request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'create',
                'seq': [SupplierOrderRun.VENDOR_MCKESSON],
                'items': [{
                    'product_id': None,
                    'name': 'Acknowledged replacement item',
                    'barcode': '64642000010',
                    'quantity': 1,
                }],
            }),
            content_type='application/json',
        )
        request.user = self.user

        response = SupplierOrderPlanView().post(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(SupplierOrderPlan.objects.count(), 2)

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


class McKessonManagedRecoveryTests(TestCase):
    def setUp(self):
        self.owner = get_user_model().objects.create_user(
            username='mck-plan-owner', password='not-used', is_staff=True,
        )
        self.recovery_admin = get_user_model().objects.create_user(
            username='mck-plan-recovery', password='not-used', is_staff=True,
        )
        self.plan = SupplierOrderPlan.objects.create(
            created_by=self.owner,
            vendor_sequence=[SupplierOrderRun.VENDOR_MCKESSON],
            status=SupplierOrderPlan.STATUS_RUNNING,
            started_at=timezone.now() - timedelta(minutes=10),
        )
        SupplierOrderPlanItem.objects.create(
            plan=self.plan,
            product_name='Saved pending item',
            barcode='64642000111',
            quantity=2,
            position=0,
        )
        self.run = SupplierOrderRun.objects.create(
            plan=self.plan,
            created_by=self.owner,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_ERROR,
            message='Portal stopped after a click.',
            total=1,
        )
        SupplierOrderRunItem.objects.create(
            run=self.run,
            product_name='Saved pending item',
            barcode='64642000111',
            quantity_requested=2,
            position=0,
            outcome=SupplierOrderRunItem.OUTCOME_PENDING,
        )

    def test_other_admin_get_sees_unresolved_mckesson_plan(self):
        request = RequestFactory().get('/low-stock/supplier-plan/')
        request.user = self.recovery_admin

        response = SupplierOrderPlanView().get(request)

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.content)
        self.assertEqual(body['plan']['id'], self.plan.pk)
        self.assertEqual(body['plan']['items'][0]['name'], 'Saved pending item')

    @patch('app.views._launch_or_schedule_order_process')
    def test_cross_admin_mixed_plan_recovery_never_claims_kohlfrisch(
            self, mock_launch):
        self.plan.vendor_sequence = [
            SupplierOrderRun.VENDOR_MCKESSON,
            SupplierOrderRun.VENDOR_KOHLFRISCH,
        ]
        self.plan.save(update_fields=['vendor_sequence'])
        get_request = RequestFactory().get('/low-stock/supplier-plan/')
        get_request.user = self.recovery_admin

        serialized = json.loads(
            SupplierOrderPlanView().get(get_request).content
        )['plan']

        self.assertEqual(serialized['seq'], [SupplierOrderRun.VENDOR_MCKESSON])
        self.assertTrue(serialized['mckesson_recovery_only'])

        retry_request = RequestFactory().post(
            '/low-stock/order-control/',
            data=json.dumps({
                'action': 'retry',
                'vendor': SupplierOrderRun.VENDOR_MCKESSON,
                'plan_id': self.plan.pk,
                'run_id': self.run.pk,
            }),
            content_type='application/json',
        )
        retry_request.user = self.recovery_admin
        mock_launch.return_value = SimpleNamespace(pid=59313)
        response = OrderControlView().post(retry_request)

        self.assertEqual(response.status_code, 200)
        self.plan.refresh_from_db()
        self.assertEqual(
            self.plan.vendor_sequence,
            [SupplierOrderRun.VENDOR_MCKESSON],
        )
        self.assertEqual(
            self.plan.mckesson_recovery_claimed_by,
            self.recovery_admin,
        )

    def test_other_admin_can_end_unresolved_mckesson_plan(self):
        request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'finish',
                'plan_id': self.plan.pk,
                'run_id': self.run.pk,
                'cancelled': True,
            }),
            content_type='application/json',
        )
        request.user = self.recovery_admin

        response = SupplierOrderPlanView().post(request)

        self.assertEqual(response.status_code, 200)
        self.plan.refresh_from_db()
        self.run.refresh_from_db()
        self.assertEqual(self.plan.status, SupplierOrderPlan.STATUS_CANCELLED)
        self.assertIsNotNone(self.plan.completed_at)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_CANCELLED)
        self.assertTrue(UserAction.objects.filter(
            user=self.recovery_admin,
            action='supplier_order_update',
            target=f'McKesson automation plan #{self.plan.pk}',
        ).exists())

        replacement_request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'create',
                'seq': [SupplierOrderRun.VENDOR_MCKESSON],
                'items': [{
                    'product_id': None,
                    'name': 'Replacement item',
                    'barcode': '64642000112',
                    'quantity': 1,
                }],
            }),
            content_type='application/json',
        )
        replacement_request.user = self.recovery_admin
        replacement = SupplierOrderPlanView().post(replacement_request)
        self.assertEqual(replacement.status_code, 200)

    @patch('app.views._launch_or_schedule_order_process')
    def test_end_then_retry_has_one_terminal_winner(self, mock_launch):
        finish_request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'finish',
                'plan_id': self.plan.pk,
                'run_id': self.run.pk,
                'cancelled': True,
            }),
            content_type='application/json',
        )
        finish_request.user = self.recovery_admin
        retry_request = RequestFactory().post(
            '/low-stock/order-control/',
            data=json.dumps({
                'action': 'retry',
                'vendor': SupplierOrderRun.VENDOR_MCKESSON,
                'plan_id': self.plan.pk,
                'run_id': self.run.pk,
            }),
            content_type='application/json',
        )
        retry_request.user = self.recovery_admin

        finish_response = SupplierOrderPlanView().post(finish_request)
        retry_response = OrderControlView().post(retry_request)

        self.assertEqual(finish_response.status_code, 200)
        self.assertEqual(retry_response.status_code, 409)
        mock_launch.assert_not_called()
        self.plan.refresh_from_db()
        self.run.refresh_from_db()
        self.assertEqual(self.plan.status, SupplierOrderPlan.STATUS_CANCELLED)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_CANCELLED)

    @patch('app.views._launch_or_schedule_order_process')
    def test_retry_then_end_has_one_active_winner(self, mock_launch):
        mock_launch.return_value = SimpleNamespace(pid=59314)
        retry_request = RequestFactory().post(
            '/low-stock/order-control/',
            data=json.dumps({
                'action': 'retry',
                'vendor': SupplierOrderRun.VENDOR_MCKESSON,
                'plan_id': self.plan.pk,
                'run_id': self.run.pk,
            }),
            content_type='application/json',
        )
        retry_request.user = self.recovery_admin
        finish_request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'finish',
                'plan_id': self.plan.pk,
                'run_id': self.run.pk,
                'cancelled': True,
            }),
            content_type='application/json',
        )
        finish_request.user = self.recovery_admin

        retry_response = OrderControlView().post(retry_request)
        finish_response = SupplierOrderPlanView().post(finish_request)

        self.assertEqual(retry_response.status_code, 200)
        self.assertEqual(finish_response.status_code, 409)
        mock_launch.assert_called_once()
        self.plan.refresh_from_db()
        self.run.refresh_from_db()
        self.assertEqual(self.plan.status, SupplierOrderPlan.STATUS_RUNNING)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_STARTING)
        self.assertEqual(self.run.process_id, 59314)

    def test_planless_mckesson_error_does_not_block_managed_plan_creation(self):
        self.plan.status = SupplierOrderPlan.STATUS_CANCELLED
        self.plan.completed_at = timezone.now()
        self.plan.save(update_fields=['status', 'completed_at'])
        planless = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            source=SupplierOrderRun.SOURCE_CLI,
            state=SupplierOrderRun.STATE_ERROR,
            message='Direct CLI run failed',
        )
        SupplierOrderRunItem.objects.create(
            run=planless,
            product_name='CLI pending item',
            barcode='64642000113',
            quantity_requested=1,
            outcome=SupplierOrderRunItem.OUTCOME_PENDING,
        )
        request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'create',
                'seq': [SupplierOrderRun.VENDOR_MCKESSON],
                'items': [{
                    'product_id': None,
                    'name': 'Managed item',
                    'barcode': '64642000114',
                    'quantity': 1,
                }],
            }),
            content_type='application/json',
        )
        request.user = self.recovery_admin

        response = SupplierOrderPlanView().post(request)
        status = _supplier_run_status(
            SupplierOrderRun.VENDOR_MCKESSON,
            plan_id=None,
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(status['requires_resolution'])
        planless.refresh_from_db()
        self.assertEqual(planless.state, SupplierOrderRun.STATE_ERROR)
        self.assertEqual(
            planless.items.get().outcome,
            SupplierOrderRunItem.OUTCOME_PENDING,
        )

    @patch('app.views._launch_or_schedule_order_process')
    def test_newer_planless_error_cannot_hide_managed_mckesson_barrier(
            self, mock_launch):
        planless = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            source=SupplierOrderRun.SOURCE_CLI,
            state=SupplierOrderRun.STATE_ERROR,
            message='Newer direct run failure',
        )
        SupplierOrderRunItem.objects.create(
            run=planless,
            product_name='CLI pending item',
            barcode='64642000117',
            quantity_requested=1,
            outcome=SupplierOrderRunItem.OUTCOME_PENDING,
        )
        request = RequestFactory().post(
            '/low-stock/mckesson-order/start/',
            data=json.dumps({
                'items': [{
                    'product_id': None,
                    'name': 'Attempted new item',
                    'barcode': '64642000118',
                    'quantity': 1,
                }],
            }),
            content_type='application/json',
        )
        request.user = self.recovery_admin

        response = _start_supplier_run(
            request,
            SupplierOrderRun.VENDOR_MCKESSON,
            'mckesson_order.py',
        )

        self.assertEqual(response.status_code, 409)
        body = json.loads(response.content)
        self.assertEqual(body['run_id'], self.run.pk)
        self.assertTrue(body['requires_resolution'])
        mock_launch.assert_not_called()

    @patch('app.views._terminate_mckesson_process_tree')
    @patch('app.views._launch_or_schedule_order_process')
    @patch('app.views._windows_mckesson_process_snapshot')
    @patch(
        'app.views._mckesson_windows_process_image_name',
        return_value='lghub_system_tray.exe',
    )
    @patch('app.views._mckesson_windows_pid_liveness', return_value='alive')
    @patch('app.views.os.name', 'nt')
    def test_cancelled_old_plan_with_logitech_pid_starts_once_without_kill(
            self, _mock_liveness, _mock_image, mock_snapshot, mock_launch,
            mock_terminate):
        self.plan.status = SupplierOrderPlan.STATUS_CANCELLED
        self.plan.completed_at = timezone.now()
        self.plan.save(update_fields=['status', 'completed_at'])
        self.run.process_id = 16696
        self.run.save(update_fields=['process_id', 'updated_at'])
        replacement_plan = SupplierOrderPlan.objects.create(
            created_by=self.recovery_admin,
            vendor_sequence=[SupplierOrderRun.VENDOR_MCKESSON],
        )
        mock_snapshot.return_value = (MCKESSON_PROCESS_UNKNOWN, None)
        mock_launch.return_value = SimpleNamespace(pid=59309)
        request = RequestFactory().post(
            '/low-stock/mckesson-order/start/',
            data=json.dumps({
                'plan_id': replacement_plan.pk,
                'items': [{
                    'product_id': None,
                    'name': 'New managed item',
                    'barcode': '64642000115',
                    'quantity': 1,
                }],
            }),
            content_type='application/json',
        )
        request.user = self.recovery_admin

        response = _start_supplier_run(
            request,
            SupplierOrderRun.VENDOR_MCKESSON,
            'mckesson_order.py',
        )

        self.assertEqual(response.status_code, 200)
        self.run.refresh_from_db()
        self.assertIsNone(self.run.process_id)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)
        self.assertEqual(
            self.run.items.get().outcome,
            SupplierOrderRunItem.OUTCOME_PENDING,
        )
        replacement = SupplierOrderRun.objects.get(plan=replacement_plan)
        self.assertEqual(replacement.process_id, 59309)
        mock_launch.assert_called_once()
        mock_terminate.assert_not_called()
        self.assertEqual(
            SupplierOrderRun.objects.filter(
                vendor=SupplierOrderRun.VENDOR_MCKESSON,
                state__in=MCKESSON_ACTIVE_STATES,
            ).count(),
            1,
        )


class McKessonLaunchFailureTests(TestCase):
    @patch('app.views._launch_or_schedule_order_process')
    def test_long_mckesson_start_error_is_truncated(self, mock_launch):
        user = get_user_model().objects.create_user(
            username='mck-launch-error-admin', password='not-used', is_staff=True,
        )
        request = RequestFactory().post(
            '/low-stock/mckesson-order/start/',
            data=json.dumps({
                'items': [{
                    'product_id': None,
                    'name': 'Launch failure item',
                    'barcode': '64642000116',
                    'quantity': 1,
                }],
            }),
            content_type='application/json',
        )
        request.user = user
        mock_launch.side_effect = OSError('x' * 800)

        response = _start_supplier_run(
            request,
            SupplierOrderRun.VENDOR_MCKESSON,
            'mckesson_order.py',
        )

        self.assertEqual(response.status_code, 500)
        run = SupplierOrderRun.objects.get()
        self.assertEqual(run.state, SupplierOrderRun.STATE_ERROR)
        self.assertEqual(len(run.message), 500)


class SupplierWorkerIdentityTests(TestCase):
    @patch('app.views.os.name', 'nt')
    @patch('app.views.subprocess.run')
    def test_mckesson_snapshot_distinguishes_gone_from_probe_failure(
            self, mock_run):
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )
        mock_run.return_value = SimpleNamespace(returncode=3, stdout='', stderr='')
        self.assertEqual(
            _windows_mckesson_process_snapshot(run),
            (MCKESSON_PROCESS_UNKNOWN, None),
        )
        mock_run.return_value = SimpleNamespace(
            returncode=1,
            stdout='',
            stderr='Access denied',
        )
        self.assertEqual(
            _windows_mckesson_process_snapshot(run),
            (MCKESSON_PROCESS_UNKNOWN, None),
        )

    @patch('app.views.os.name', 'nt')
    @patch('app.views.subprocess.run')
    def test_malformed_mckesson_snapshot_fails_closed(self, mock_run):
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )
        mock_run.return_value = SimpleNamespace(
            returncode=0,
            stdout='not-json',
            stderr='',
        )

        self.assertEqual(
            _windows_mckesson_process_snapshot(run),
            (MCKESSON_PROCESS_UNKNOWN, None),
        )

    @patch('app.views.os.name', 'nt')
    @patch(
        'app.views._mckesson_windows_process_image_name',
        return_value='lghub_system_tray.exe',
    )
    @patch(
        'app.views._windows_mckesson_process_snapshot',
        return_value=(MCKESSON_PROCESS_UNKNOWN, None),
    )
    @patch('app.views._mckesson_windows_pid_liveness', return_value='alive')
    def test_mckesson_probe_recognizes_recycled_non_python_pid(
            self, _mock_liveness, _mock_snapshot, _mock_image):
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )

        self.assertEqual(
            _inspect_mckesson_worker_process(run),
            MCKESSON_PROCESS_UNRELATED,
        )

    @patch('app.views.os.name', 'nt')
    @patch(
        'app.views._mckesson_windows_process_image_name',
        return_value='python.exe',
    )
    @patch(
        'app.views._windows_mckesson_process_snapshot',
        return_value=(MCKESSON_PROCESS_UNKNOWN, None),
    )
    @patch('app.views._mckesson_windows_pid_liveness', return_value='alive')
    def test_mckesson_probe_fails_closed_for_unverified_python_process(
            self, _mock_liveness, _mock_snapshot, _mock_image):
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )

        self.assertEqual(
            _inspect_mckesson_worker_process(run),
            MCKESSON_PROCESS_UNKNOWN,
        )

    @patch('app.views.os.name', 'nt')
    @patch('app.views._mckesson_windows_process_image_name', return_value=None)
    @patch(
        'app.views._windows_mckesson_process_snapshot',
        return_value=(MCKESSON_PROCESS_UNKNOWN, None),
    )
    @patch(
        'app.views._mckesson_windows_pid_liveness',
        return_value=MCKESSON_PROCESS_UNKNOWN,
    )
    def test_mckesson_probe_fails_closed_when_windows_denies_inspection(
            self, _mock_liveness, _mock_snapshot, _mock_image):
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )

        self.assertEqual(
            _inspect_mckesson_worker_process(run),
            MCKESSON_PROCESS_UNKNOWN,
        )

    @patch('app.views.os.name', 'nt')
    @patch(
        'app.views._mckesson_snapshot_process_state',
        return_value=MCKESSON_PROCESS_MATCHED,
    )
    @patch(
        'app.views._windows_mckesson_process_snapshot',
        return_value=('ready', [{'ProcessId': 48192}]),
    )
    @patch('app.views._mckesson_windows_pid_liveness', return_value='alive')
    def test_mckesson_probe_accepts_only_exact_verified_worker(
            self, _mock_liveness, _mock_snapshot, _mock_classifier):
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
        )

        self.assertEqual(
            _inspect_mckesson_worker_process(run),
            MCKESSON_PROCESS_MATCHED,
        )

    @patch('app.views._windows_command_line_args')
    def test_mckesson_snapshot_matches_exact_absolute_worker_command(
            self, mock_args):
        base = Path(__file__).resolve().parent.parent
        run = SimpleNamespace(
            pk=42,
            attempt=3,
            process_id=48192,
        )
        mock_args.return_value = [
            str(base / 'env' / 'Scripts' / 'python.exe'),
            str(base / 'mckesson_order.py'),
            '--no-input',
            '--run-id',
            '42',
            '--attempt=3',
        ]

        state = _mckesson_snapshot_process_state(
            run,
            [{
                'ProcessId': 48192,
                'ExecutablePath': r'C:\Python\python3.13.exe',
                'CommandLine': 'mocked',
            }],
        )

        self.assertEqual(state, MCKESSON_PROCESS_MATCHED)

    @patch('app.views._windows_command_line_args')
    def test_mckesson_script_as_later_data_argument_cannot_match(self, mock_args):
        base = Path(__file__).resolve().parent.parent
        run = SimpleNamespace(pk=42, attempt=3, process_id=48192)
        mock_args.return_value = [
            str(base / 'env' / 'Scripts' / 'python.exe'),
            str(base / 'maintenance_script.py'),
            str(base / 'mckesson_order.py'),
            '--no-input',
            '--run-id',
            '42',
            '--attempt',
            '3',
        ]

        state = _mckesson_snapshot_process_state(
            run,
            [{
                'ProcessId': 48192,
                'ExecutablePath': r'C:\Windows\System32\cmd.exe',
                'CommandLine': 'mocked',
            }],
        )

        self.assertEqual(state, MCKESSON_PROCESS_OTHER_WORKER)

    @patch('app.views._windows_command_line_args')
    def test_relative_worker_paths_cannot_authorize_taskkill(self, mock_args):
        run = SimpleNamespace(pk=42, attempt=3, process_id=48192)
        mock_args.return_value = [
            r'env\Scripts\python.exe',
            'mckesson_order.py',
            '--no-input',
            '--run-id',
            '42',
            '--attempt',
            '3',
        ]

        state = _mckesson_snapshot_process_state(
            run,
            [{
                'ProcessId': 48192,
                'ExecutablePath': r'C:\Python\python3.13.exe',
                'CommandLine': 'mocked',
            }],
        )

        self.assertEqual(state, MCKESSON_PROCESS_OTHER_WORKER)

    @patch('app.views._windows_command_line_args')
    def test_readable_unrelated_python_command_is_not_a_worker(self, mock_args):
        base = Path(__file__).resolve().parent.parent
        run = SimpleNamespace(pk=42, attempt=3, process_id=48192)
        mock_args.return_value = [
            str(base / 'env' / 'Scripts' / 'python.exe'),
            str(base / 'maintenance_script.py'),
        ]

        state = _mckesson_snapshot_process_state(
            run,
            [{'ProcessId': 48192, 'CommandLine': 'mocked'}],
        )

        self.assertEqual(state, MCKESSON_PROCESS_UNRELATED)

    @patch('app.views._windows_command_line_args')
    def test_wrong_mckesson_attempt_is_blocked_as_another_worker(self, mock_args):
        base = Path(__file__).resolve().parent.parent
        run = SimpleNamespace(pk=42, attempt=3, process_id=48192)
        mock_args.return_value = [
            str(base / 'env' / 'Scripts' / 'python.exe'),
            str(base / 'mckesson_order.py'),
            '--no-input',
            '--run-id',
            '42',
            '--attempt',
            '2',
        ]

        state = _mckesson_snapshot_process_state(
            run,
            [{'ProcessId': 48192, 'CommandLine': 'mocked'}],
        )

        self.assertEqual(state, MCKESSON_PROCESS_OTHER_WORKER)

    @patch('app.views._windows_command_line_args')
    def test_script_name_suffix_cannot_impersonate_mckesson_worker(self, mock_args):
        base = Path(__file__).resolve().parent.parent
        run = SimpleNamespace(pk=42, attempt=3, process_id=48192)
        mock_args.return_value = [
            str(base / 'env' / 'Scripts' / 'python.exe'),
            str(base / 'mckesson_order.py.bak'),
            '--no-input',
            '--run-id',
            '42',
            '--attempt',
            '3',
        ]

        state = _mckesson_snapshot_process_state(
            run,
            [{'ProcessId': 48192, 'CommandLine': 'mocked'}],
        )

        self.assertEqual(state, MCKESSON_PROCESS_UNRELATED)

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


class McKessonTerminationSafetyTests(TestCase):
    def make_run(self):
        return SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_ERROR,
            process_id=48192,
        )

    @patch('app.views.subprocess.run')
    @patch('app.views._inspect_mckesson_worker_process', return_value=MCKESSON_PROCESS_UNKNOWN)
    @patch('app.views._open_mckesson_termination_handle', return_value=None)
    def test_unknown_identity_is_never_taskkilled(
            self, _mock_handle, _mock_inspect, mock_run):
        self.assertFalse(_terminate_mckesson_process_tree(self.make_run()))
        mock_run.assert_not_called()

    @patch('app.views.subprocess.run')
    @patch('app.views._inspect_mckesson_worker_process', return_value=MCKESSON_PROCESS_UNRELATED)
    @patch('app.views._open_mckesson_termination_handle', return_value=None)
    def test_unrelated_identity_is_never_taskkilled(
            self, _mock_handle, _mock_inspect, mock_run):
        self.assertTrue(_terminate_mckesson_process_tree(self.make_run()))
        mock_run.assert_not_called()

    @patch('app.views._inspect_mckesson_worker_process', return_value=MCKESSON_PROCESS_MATCHED)
    @patch('app.views._open_mckesson_termination_handle')
    @patch('app.views.subprocess.run')
    def test_exact_worker_handle_is_retained_through_kill_and_wait(
            self, mock_run, mock_open, _mock_inspect):
        handle = object()
        wait = Mock(side_effect=[258, 0])
        close = Mock()
        mock_open.return_value = (handle, wait, close)
        mock_run.return_value = SimpleNamespace(returncode=0)
        run = self.make_run()

        result = _terminate_mckesson_process_tree(run, timeout_seconds=2)

        self.assertTrue(result)
        mock_run.assert_called_once()
        self.assertEqual(
            mock_run.call_args.args[0],
            ['taskkill.exe', '/PID', '48192', '/T', '/F'],
        )
        self.assertEqual(wait.call_args_list[0].args, (handle, 0))
        self.assertEqual(wait.call_args_list[1].args, (handle, 2000))
        close.assert_called_once_with(handle)

    @patch('app.views._inspect_mckesson_worker_process', return_value=MCKESSON_PROCESS_MATCHED)
    @patch('app.views._open_mckesson_termination_handle')
    @patch('app.views.subprocess.run')
    def test_taskkill_or_wait_failure_never_authorizes_replacement(
            self, mock_run, mock_open, _mock_inspect):
        handle = object()
        wait = Mock(side_effect=[258, 258])
        close = Mock()
        mock_open.return_value = (handle, wait, close)
        mock_run.return_value = SimpleNamespace(returncode=0)

        self.assertFalse(
            _terminate_mckesson_process_tree(self.make_run(), timeout_seconds=0)
        )
        close.assert_called_once_with(handle)

    @patch('app.views._inspect_mckesson_worker_process', return_value=MCKESSON_PROCESS_MATCHED)
    @patch('app.views._open_mckesson_termination_handle')
    @patch('app.views.subprocess.run')
    def test_nonzero_taskkill_result_never_authorizes_replacement(
            self, mock_run, mock_open, _mock_inspect):
        handle = object()
        wait = Mock(return_value=258)
        close = Mock()
        mock_open.return_value = (handle, wait, close)
        mock_run.return_value = SimpleNamespace(returncode=1)

        self.assertFalse(_terminate_mckesson_process_tree(self.make_run()))
        self.assertEqual(wait.call_count, 1)
        close.assert_called_once_with(handle)


class McKessonStalePidCasTests(TestCase):
    def test_exact_attempt_and_pid_are_cleared(self):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_ERROR,
            process_id=48192,
        )

        self.assertTrue(_clear_stale_mckesson_process_id(run))
        run.refresh_from_db()
        self.assertIsNone(run.process_id)

    def test_stale_attempt_cannot_clear_replacement_pid(self):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_ERROR,
            process_id=48192,
        )
        SupplierOrderRun.objects.filter(pk=run.pk).update(
            attempt=run.attempt + 1,
            process_id=59310,
        )

        self.assertFalse(_clear_stale_mckesson_process_id(run))
        run.refresh_from_db()
        self.assertEqual(run.process_id, 59310)

    def test_stale_pid_cannot_clear_same_attempt_replacement(self):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_ERROR,
            process_id=48192,
        )
        SupplierOrderRun.objects.filter(pk=run.pk).update(process_id=59311)

        self.assertFalse(_clear_stale_mckesson_process_id(run))
        run.refresh_from_db()
        self.assertEqual(run.process_id, 59311)


class KohlFrischProcessIdentityBoundaryTests(TestCase):
    @patch(
        'app.views._inspect_mckesson_worker_process',
        side_effect=AssertionError('McKesson identity probe must not run for K&F'),
    )
    @patch('app.views._pid_alive', return_value=True)
    def test_kohlfrisch_terminal_status_retains_legacy_pid_behavior(
            self, _mock_pid_alive, _mock_mckesson_probe):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_KOHLFRISCH,
            state=SupplierOrderRun.STATE_ERROR,
            process_id=48192,
            message='K&F terminal test',
        )

        payload = _supplier_run_status(SupplierOrderRun.VENDOR_KOHLFRISCH)

        run.refresh_from_db()
        self.assertTrue(payload['worker_alive'])
        self.assertEqual(run.process_id, 48192)
        self.assertNotIn('worker_identity_uncertain', payload)
        self.assertNotIn('requires_resolution', payload)

    def test_other_admin_cannot_take_over_kohlfrisch_plan(self):
        owner = get_user_model().objects.create_user(
            username='kf-owner', password='not-used', is_staff=True,
        )
        other_admin = get_user_model().objects.create_user(
            username='kf-other-admin', password='not-used', is_staff=True,
        )
        plan = SupplierOrderPlan.objects.create(
            created_by=owner,
            vendor_sequence=[SupplierOrderRun.VENDOR_KOHLFRISCH],
            status=SupplierOrderPlan.STATUS_ERROR,
        )
        run = SupplierOrderRun.objects.create(
            plan=plan,
            created_by=owner,
            vendor=SupplierOrderRun.VENDOR_KOHLFRISCH,
            state=SupplierOrderRun.STATE_ERROR,
        )
        SupplierOrderRunItem.objects.create(
            run=run,
            product_name='K&F pending item',
            quantity_requested=1,
            outcome=SupplierOrderRunItem.OUTCOME_PENDING,
        )
        get_request = RequestFactory().get('/low-stock/supplier-plan/')
        get_request.user = other_admin
        finish_request = RequestFactory().post(
            '/low-stock/supplier-plan/',
            data=json.dumps({
                'action': 'finish',
                'plan_id': plan.pk,
                'cancelled': True,
            }),
            content_type='application/json',
        )
        finish_request.user = other_admin
        retry_request = RequestFactory().post(
            '/low-stock/order-control/',
            data=json.dumps({
                'action': 'retry',
                'vendor': SupplierOrderRun.VENDOR_KOHLFRISCH,
                'plan_id': plan.pk,
                'run_id': run.pk,
            }),
            content_type='application/json',
        )
        retry_request.user = other_admin

        get_response = SupplierOrderPlanView().get(get_request)
        finish_response = SupplierOrderPlanView().post(finish_request)
        retry_response = OrderControlView().post(retry_request)

        self.assertIsNone(json.loads(get_response.content)['plan'])
        self.assertEqual(finish_response.status_code, 200)
        self.assertEqual(retry_response.status_code, 409)
        plan.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(plan.status, SupplierOrderPlan.STATUS_ERROR)
        self.assertEqual(run.attempt, 1)


class McKessonActiveRunLeaseTests(TestCase):
    def test_database_allows_only_one_active_mckesson_run(self):
        first = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_RUNNING,
        )

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SupplierOrderRun.objects.create(
                    vendor=SupplierOrderRun.VENDOR_MCKESSON,
                    state=SupplierOrderRun.STATE_STARTING,
                )

        first.state = SupplierOrderRun.STATE_ERROR
        first.save(update_fields=['state', 'updated_at'])
        replacement = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_STARTING,
        )
        self.assertIsNotNone(replacement.pk)

    def test_mckesson_lease_does_not_limit_kohlfrisch_runs(self):
        SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_RUNNING,
        )
        SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_KOHLFRISCH,
            state=SupplierOrderRun.STATE_RUNNING,
        )
        second = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_KOHLFRISCH,
            state=SupplierOrderRun.STATE_STARTING,
        )

        self.assertIsNotNone(second.pk)

    def test_mckesson_status_messages_are_truncated_at_database_boundary(self):
        run = SupplierOrderRun.objects.create(
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_RUNNING,
        )
        worker = object.__new__(DatabaseRunStatus)
        worker.run = run
        worker.attempt = run.attempt

        self.assertTrue(worker.update(message='x' * 800))

        run.refresh_from_db()
        self.assertEqual(len(run.message), 500)
        self.assertEqual(run.message, 'x' * 500)


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

    def test_failed_mckesson_plan_requires_cart_review_before_ending(self):
        self.assertIn("title: 'End failed McKesson run'", self.source)
        self.assertIn('Ending this run does not undo any portal click', self.source)
        self.assertIn('await finishPlan(true, retry.data.run_id)', self.source)
        self.assertIn("endErrorBtn.textContent = 'Ending…'", self.source)
        self.assertIn("endErrorBtn.textContent = 'End plan'", self.source)
        self.assertIn('retryBtn.disabled = true;', self.source)

    def test_plan_conflict_attaches_to_recoverable_mckesson_plan(self):
        self.assertIn(
            'r.status === 409 && d.requires_resolution && d.plan',
            self.source,
        )
        self.assertIn('runSequence(runSeq, recovering)', self.source)
