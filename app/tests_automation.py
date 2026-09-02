import json
import subprocess
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from app import scheduled_jobs
from app.inventory_audit import run_inventory_audit, serialize_audit_run
from app.management.commands.run_scheduled_jobs import (
    Command as RunScheduledJobsCommand,
)
from app.models import (
    DailyReportArchive,
    InventoryAuditIssue,
    InventoryAuditRun,
    Product,
    ProductExpiryDate,
    ProductLot,
    ScheduledJobRun,
    StoreHours,
)
from app.reporting import prune_daily_report_archives
from app.scheduled_jobs import (
    ensure_store_hours,
    gsheet_schedule_for,
    next_gsheet_pull,
    run_due_jobs,
    run_google_sheet_sync,
    store_hours_payload,
)


class BackupProcessSafetyTests(SimpleTestCase):
    @patch('app.scheduled_jobs.os.name', 'nt')
    @patch('app.scheduled_jobs.subprocess.run')
    def test_windows_timeout_uses_recursive_forced_taskkill(self, run_process):
        process = Mock(pid=2460)
        process.poll.return_value = None

        scheduled_jobs._terminate_backup_process_tree(process)

        command = run_process.call_args.args[0]
        self.assertEqual(command[:3], ['taskkill.exe', '/PID', '2460'])
        self.assertIn('/T', command)
        self.assertIn('/F', command)
        process.wait.assert_called_once_with(
            timeout=scheduled_jobs.BACKUP_PROCESS_STOP_TIMEOUT_SECONDS,
        )

    @patch('app.scheduled_jobs._terminate_backup_process_tree')
    @patch('app.scheduled_jobs.subprocess.Popen')
    def test_outer_timeout_terminates_backup_process_tree(
        self, popen, terminate_tree,
    ):
        process = Mock()
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(['powershell.exe'], 900),
            ('', ''),
        ]
        popen.return_value = process

        with self.assertRaises(subprocess.TimeoutExpired):
            scheduled_jobs._run_backup_process(['powershell.exe'])

        terminate_tree.assert_called_once_with(process)


class DailyReportRetentionTests(TestCase):
    def test_cleanup_removes_only_expired_pdf_snapshots(self):
        DailyReportArchive.objects.create(
            report_date=date(2026, 7, 16), pdf=b'old', summary='old',
        )
        DailyReportArchive.objects.create(
            report_date=date(2026, 7, 17), pdf=b'cutoff', summary='keep',
        )
        DailyReportArchive.objects.create(
            report_date=date(2026, 8, 16), pdf=b'new', summary='new',
        )

        deleted = prune_daily_report_archives(reference_date=date(2026, 8, 16))

        self.assertEqual(deleted, 1)
        self.assertQuerySetEqual(
            DailyReportArchive.objects.order_by('report_date').values_list(
                'report_date', flat=True,
            ),
            [date(2026, 7, 17), date(2026, 8, 16)],
            transform=lambda value: value,
        )


class InventoryAuditTests(TestCase):
    def setUp(self):
        self.product = Product.objects.create(
            name='Audit Product',
            barcode='0012345',
            price=Decimal('7.99'),
            price_per_unit=Decimal('3.00'),
            quantity_in_stock=5,
        )

    def test_audit_retains_structured_lot_mismatch(self):
        run = run_inventory_audit()

        self.assertEqual(run.status, InventoryAuditRun.STATUS_ISSUES)
        self.assertEqual(run.issue_count, 1)
        issue = run.issues.get()
        self.assertEqual(issue.code, 'lot_total_mismatch')
        self.assertTrue(issue.repairable)
        self.assertFalse(issue.repaired)

    def test_repair_assigns_missing_stock_without_changing_total(self):
        run = run_inventory_audit(repair_unassigned=True)

        self.product.refresh_from_db()
        self.assertEqual(run.status, InventoryAuditRun.STATUS_REPAIRED)
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertEqual(
            ProductLot.objects.get(
                product=self.product,
                lot_number=ProductLot.UNASSIGNED,
            ).quantity_on_hand,
            5,
        )
        self.assertTrue(run.issues.get().repaired)
        self.assertEqual(run_inventory_audit().status, InventoryAuditRun.STATUS_PASSED)

    def test_audit_detects_current_expiry_on_zero_stock_product(self):
        expiry = date(2026, 10, 31)
        product = Product.objects.create(
            name='Zero stock expiry', price=Decimal('5.00'),
            quantity_in_stock=0, expiry_date=expiry,
        )
        ProductExpiryDate.objects.create(product=product, expiry_date=expiry)

        run = run_inventory_audit()
        issue = run.issues.get(
            product=product, code='zero_stock_current_expiry',
        )
        payload = serialize_audit_run(run)

        self.assertTrue(issue.repairable)
        self.assertFalse(issue.repaired)
        self.assertEqual(issue.metadata['action'], 'clear_zero_stock_expiry')
        self.assertEqual(issue.metadata['snapshot']['quantity_in_stock'], 0)
        self.assertEqual(payload['clearable_expiry_count'], 1)
        self.assertEqual(payload['repairable_count'], 1)  # setUp product only

    def test_historical_audit_text_keeps_unassigned_without_rewriting_history(self):
        run = InventoryAuditRun.objects.create(
            status=InventoryAuditRun.STATUS_ISSUES,
            issue_count=1,
            summary='Review UNASSIGNED lot history',
        )
        issue = InventoryAuditIssue.objects.create(
            run=run,
            code='legacy_lot_text',
            title='Legacy UNASSIGNED finding',
            detail='Stock remains in UNASSIGNED after repair.',
            expected_value='No UNASSIGNED balance',
            actual_value='UNASSIGNED: 2',
        )

        encoded = json.dumps(serialize_audit_run(run))

        self.assertIn(ProductLot.UNASSIGNED, encoded)
        self.assertNotIn('MAIN', encoded)
        issue.refresh_from_db()
        run.refresh_from_db()
        self.assertIn(ProductLot.UNASSIGNED, issue.detail)
        self.assertIn(ProductLot.UNASSIGNED, run.summary)

    def test_audit_detects_either_zero_stock_current_expiry_source(self):
        direct = Product.objects.create(
            name='Direct current expiry', price=Decimal('5.00'),
            quantity_in_stock=0, expiry_date=date(2026, 10, 1),
        )
        mirrored = Product.objects.create(
            name='Mirrored current expiry', price=Decimal('5.00'),
            quantity_in_stock=0,
        )
        ProductExpiryDate.objects.create(
            product=mirrored, expiry_date=date(2026, 11, 1),
        )

        run = run_inventory_audit()

        self.assertTrue(run.issues.filter(
            product=direct, code='zero_stock_current_expiry', repairable=True,
        ).exists())
        self.assertTrue(run.issues.filter(
            product=mirrored, code='zero_stock_current_expiry', repairable=True,
        ).exists())

    def test_depleted_dated_lot_is_history_not_a_current_expiry_issue(self):
        product = Product.objects.create(
            name='Depleted dated history', price=Decimal('5.00'),
            quantity_in_stock=0,
        )
        ProductLot.objects.create(
            product=product, lot_number='EMPTY-HISTORY',
            expiry_date=date(2026, 1, 1), quantity_on_hand=0,
        )

        run = run_inventory_audit()

        self.assertFalse(run.issues.filter(
            product=product, code='zero_stock_current_expiry',
        ).exists())

    def test_zero_stock_expiry_with_positive_lot_mismatch_stays_visible_but_blocked(self):
        expiry = date(2026, 10, 31)
        product = Product.objects.create(
            name='Unsafe zero stock expiry', price=Decimal('5.00'),
            quantity_in_stock=0, expiry_date=expiry,
        )
        ProductExpiryDate.objects.create(product=product, expiry_date=expiry)
        ProductLot.objects.create(
            product=product, lot_number='MISMATCH', expiry_date=expiry,
            quantity_on_hand=2,
        )

        run = run_inventory_audit()
        issue = run.issues.get(
            product=product, code='zero_stock_current_expiry',
        )

        self.assertFalse(issue.repairable)
        self.assertNotIn('action', issue.metadata)
        self.assertEqual(issue.metadata['result'], 'blocked_by_lot_mismatch')
        self.assertIn('active lots still total 2', issue.detail)

    def test_expiry_clear_rolls_back_if_audit_results_cannot_be_saved(self):
        expiry = date(2026, 12, 31)
        product = Product.objects.create(
            name='Atomic expiry clear', price=Decimal('5.00'),
            quantity_in_stock=0, expiry_date=expiry,
        )
        ProductExpiryDate.objects.create(product=product, expiry_date=expiry)
        source_run = run_inventory_audit()
        source_issue = source_run.issues.get(
            product=product, code='zero_stock_current_expiry',
        )

        with patch(
            'app.inventory_audit.InventoryAuditIssue.objects.bulk_create',
            side_effect=RuntimeError('forced result write failure'),
        ):
            failed_run = run_inventory_audit(
                clear_zero_stock_expiry_issue_ids=[source_issue.pk],
                clear_zero_stock_expiry_run_id=source_run.pk,
            )

        product.refresh_from_db()
        self.assertEqual(failed_run.status, InventoryAuditRun.STATUS_ERROR)
        self.assertEqual(failed_run.issue_count, 0)
        self.assertEqual(failed_run.issues.count(), 0)
        self.assertIn('forced result write failure', failed_run.error)
        self.assertEqual(product.expiry_date, expiry)
        self.assertTrue(ProductExpiryDate.objects.filter(
            product=product, expiry_date=expiry,
        ).exists())


class InventoryAuditAPITests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.staff = User.objects.create_user('audit-admin', password='test-pass', is_staff=True)
        self.user = User.objects.create_user('audit-user', password='test-pass')

    def test_signed_in_user_can_view_run_and_repair_inventory_audit(self):
        product = Product.objects.create(
            name='Repairable audit product', price=Decimal('5.00'),
            quantity_in_stock=5,
        )
        self.client.force_login(self.user)
        response = self.client.get(reverse('inventory_integrity_api'))
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['run'])
        self.assertTrue(response.json()['can_repair'])

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['run']['status'], 'issues')
        self.assertEqual(InventoryAuditRun.objects.latest('pk').created_by, self.user)

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'repair'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['run']['status'], 'repaired')
        product.refresh_from_db()
        self.assertEqual(product.quantity_in_stock, 5)
        self.assertEqual(
            ProductLot.objects.get(
                product=product, lot_number=ProductLot.UNASSIGNED,
            ).quantity_on_hand,
            5,
        )
        self.assertEqual(InventoryAuditRun.objects.latest('pk').created_by, self.user)

    def test_staff_can_run_audit_without_inventory_page_reload(self):
        self.client.force_login(self.staff)
        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['run']['status'], 'passed')
        self.assertEqual(InventoryAuditRun.objects.count(), 1)

    @staticmethod
    def _zero_stock_expiry_product(name, expiry):
        product = Product.objects.create(
            name=name, price=Decimal('5.00'), quantity_in_stock=0,
            expiry_date=expiry,
        )
        ProductExpiryDate.objects.create(product=product, expiry_date=expiry)
        lot = ProductLot.objects.create(
            product=product, lot_number=f'{name[:10]}-EMPTY',
            expiry_date=expiry, quantity_on_hand=0,
        )
        return product, lot

    def test_user_can_selectively_clear_reviewed_zero_stock_expiry(self):
        product, depleted_lot = self._zero_stock_expiry_product(
            'Reviewed expiry', date(2026, 11, 30),
        )
        self.client.force_login(self.user)
        audit = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        ).json()['run']
        issue = next(
            item for item in audit['issues']
            if item['product_id'] == product.pk
            and item['code'] == 'zero_stock_current_expiry'
        )

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({
                'action': 'clear_zero_stock_expiry',
                'run_id': audit['id'],
                'issue_ids': [issue['id']],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200, response.content)
        result = response.json()['run']
        self.assertEqual(result['status'], 'repaired')
        self.assertEqual(result['clearable_expiry_count'], 0)
        product.refresh_from_db()
        depleted_lot.refresh_from_db()
        self.assertIsNone(product.expiry_date)
        self.assertFalse(ProductExpiryDate.objects.filter(product=product).exists())
        self.assertEqual(depleted_lot.quantity_on_hand, 0)
        self.assertEqual(depleted_lot.expiry_date, date(2026, 11, 30))
        repair_run = InventoryAuditRun.objects.latest('pk')
        self.assertEqual(repair_run.created_by, self.user)
        repaired_issue = repair_run.issues.get(
            product=product, code='zero_stock_current_expiry', repaired=True,
        )
        self.assertEqual(repaired_issue.metadata['source_issue_id'], issue['id'])
        self.assertEqual(
            repaired_issue.metadata['resolved_by_run_id'], repair_run.pk,
        )
        source_issue = InventoryAuditIssue.objects.get(pk=issue['id'])
        self.assertTrue(source_issue.repaired)
        self.assertEqual(source_issue.metadata['result'], 'cleared')
        self.assertEqual(
            source_issue.metadata['resolved_by_run_id'], repair_run.pk,
        )
        self.assertEqual(
            source_issue.metadata['resolved_by_user_id'], self.user.pk,
        )
        reviewed_again = self.client.get(
            reverse('inventory_integrity_api'), {'run_id': audit['id']},
        ).json()['run']
        reviewed_issue = next(
            item for item in reviewed_again['issues'] if item['id'] == issue['id']
        )
        self.assertEqual(reviewed_again['clearable_expiry_count'], 0)
        self.assertTrue(reviewed_issue['repaired'])
        self.assertEqual(reviewed_issue['action_result'], 'cleared')

        run_count = InventoryAuditRun.objects.count()
        replay = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({
                'action': 'clear_zero_stock_expiry',
                'run_id': audit['id'],
                'issue_ids': [issue['id']],
            }),
            content_type='application/json',
        )
        self.assertEqual(replay.status_code, 409)
        self.assertEqual(InventoryAuditRun.objects.count(), run_count)

    def test_expiry_clear_is_all_or_nothing_when_one_selection_is_stale(self):
        first, _ = self._zero_stock_expiry_product(
            'Unchanged expiry', date(2026, 10, 1),
        )
        changed, changed_lot = self._zero_stock_expiry_product(
            'Changed expiry', date(2026, 12, 1),
        )
        self.client.force_login(self.user)
        audit = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        ).json()['run']
        issue_ids = [
            item['id'] for item in audit['issues']
            if item['code'] == 'zero_stock_current_expiry'
        ]
        Product.objects.filter(pk=changed.pk).update(quantity_in_stock=1)
        ProductLot.objects.filter(pk=changed_lot.pk).update(quantity_on_hand=1)

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({
                'action': 'clear_zero_stock_expiry',
                'run_id': audit['id'],
                'issue_ids': issue_ids,
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(response.json()['run']['status'], 'issues')
        first.refresh_from_db()
        changed.refresh_from_db()
        self.assertEqual(first.expiry_date, date(2026, 10, 1))
        self.assertEqual(changed.expiry_date, date(2026, 12, 1))
        self.assertTrue(ProductExpiryDate.objects.filter(product=first).exists())
        self.assertTrue(ProductExpiryDate.objects.filter(product=changed).exists())
        action_results = {
            issue['action_result'] for issue in response.json()['run']['issues']
            if issue['action'] == 'clear_zero_stock_expiry'
        }
        self.assertTrue(
            {'changed_since_review', 'not_cleared'} & action_results,
        )

    def test_expiry_clear_rejects_an_invalid_selection(self):
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({
                'action': 'clear_zero_stock_expiry', 'issue_ids': [],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn('Select at least one', response.json()['error'])

        oversized = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({
                'action': 'clear_zero_stock_expiry',
                'run_id': 1,
                'issue_ids': list(range(1, 102)),
            }),
            content_type='application/json',
        )
        self.assertEqual(oversized.status_code, 400)
        self.assertIn('no more than 100', oversized.json()['error'])

    def test_expiry_clear_requires_the_reviewed_audit_run(self):
        product, _ = self._zero_stock_expiry_product(
            'Missing run expiry', date(2026, 9, 30),
        )
        self.client.force_login(self.user)
        audit = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        ).json()['run']
        issue = next(
            item for item in audit['issues']
            if item['product_id'] == product.pk
            and item['code'] == 'zero_stock_current_expiry'
        )

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({
                'action': 'clear_zero_stock_expiry',
                'issue_ids': [issue['id']],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn('audit run is missing', response.json()['error'])
        product.refresh_from_db()
        self.assertEqual(product.expiry_date, date(2026, 9, 30))

    def test_expiry_clear_rejects_findings_from_another_audit_run(self):
        product, _ = self._zero_stock_expiry_product(
            'Run-bound expiry', date(2026, 9, 30),
        )
        self.client.force_login(self.user)
        first = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        ).json()['run']
        second = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        ).json()['run']
        first_issue = next(
            item for item in first['issues']
            if item['product_id'] == product.pk
        )

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({
                'action': 'clear_zero_stock_expiry',
                'run_id': second['id'],
                'issue_ids': [first_issue['id']],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 409)
        product.refresh_from_db()
        self.assertEqual(product.expiry_date, date(2026, 9, 30))

    def test_expiry_clear_rejects_zero_stock_finding_with_positive_lot_stock(self):
        expiry = date(2026, 9, 30)
        product = Product.objects.create(
            name='Blocked lot mismatch expiry', price=Decimal('5.00'),
            quantity_in_stock=0, expiry_date=expiry,
        )
        ProductExpiryDate.objects.create(product=product, expiry_date=expiry)
        ProductLot.objects.create(
            product=product, lot_number='POSITIVE-LOT', expiry_date=expiry,
            quantity_on_hand=1,
        )
        self.client.force_login(self.user)
        audit = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        ).json()['run']
        issue = next(
            item for item in audit['issues']
            if item['product_id'] == product.pk
            and item['code'] == 'zero_stock_current_expiry'
        )
        self.assertFalse(issue['repairable'])

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({
                'action': 'clear_zero_stock_expiry',
                'run_id': audit['id'],
                'issue_ids': [issue['id']],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 409)
        product.refresh_from_db()
        self.assertEqual(product.expiry_date, expiry)
        self.assertTrue(ProductExpiryDate.objects.filter(product=product).exists())


class ScheduledAutomationTests(TestCase):
    def setUp(self):
        StoreHours.objects.update_or_create(
            weekday=StoreHours.MONDAY,
            defaults={
                'is_closed': False,
                'opens_at': time(9, 30),
                'closes_at': time(18, 0),
            },
        )
        StoreHours.objects.update_or_create(
            weekday=StoreHours.SUNDAY,
            defaults={'is_closed': True, 'opens_at': None, 'closes_at': None},
        )

    @staticmethod
    def _local(year, month, day, hour, minute):
        return timezone.make_aware(
            datetime(year, month, day, hour, minute),
            timezone.get_current_timezone(),
        )

    def _mark_preclose_complete(self, business_date):
        scheduled_for = gsheet_schedule_for(business_date)
        for job_key in (
            ScheduledJobRun.JOB_GSHEET_PRECLOSE,
            ScheduledJobRun.JOB_DATABASE_BACKUP,
        ):
            ScheduledJobRun.objects.create(
                job_key=job_key,
                trigger=ScheduledJobRun.TRIGGER_SCHEDULED,
                business_date=business_date,
                scheduled_for=scheduled_for,
                status=ScheduledJobRun.STATUS_SUCCESS,
                completed_at=scheduled_for,
                summary='Already completed.',
            )

    def test_dashboard_hours_and_next_pull_share_database_schedule(self):
        payload = store_hours_payload()
        self.assertEqual(payload['1'], [9, 30, 18, 0])
        self.assertIsNone(payload['0'])

        next_pull = next_gsheet_pull(at=self._local(2026, 8, 17, 16, 30))
        self.assertEqual(
            timezone.localtime(next_pull['scheduled_for']).time(),
            time(17, 0),
        )
        self.assertFalse(next_pull['due'])

    def test_7pm_closing_schedules_google_sheet_pull_at_6pm(self):
        StoreHours.objects.update_or_create(
            weekday=StoreHours.TUESDAY,
            defaults={
                'is_closed': False,
                'opens_at': time(9, 30),
                'closes_at': time(19, 0),
            },
        )

        scheduled_for = gsheet_schedule_for(date(2026, 8, 18))

        self.assertEqual(timezone.localtime(scheduled_for).time(), time(18, 0))

    @patch('app.scheduled_jobs._run_backup_process')
    @patch('app.gsheet_sync.is_configured', return_value=True)
    @patch('app.gsheet_sync.sync_all', return_value={
        'last_sync': 1,
        'imported': 2,
        'tabs': [{'title': 'Orders', 'imported': 2}],
        'errors': [],
    })
    def test_preclosing_jobs_run_once_at_5pm_on_monday(
        self, _sync, _configured, backup_process,
    ):
        backup_process.return_value = SimpleNamespace(
            returncode=0,
            stdout='Database backup verified: test.dump\ntest.dump\n',
            stderr='',
        )
        before = self._local(2026, 8, 17, 16, 59)
        due = self._local(2026, 8, 17, 17, 0)
        self._mark_preclose_complete(date(2026, 8, 15))

        early_runs = run_due_jobs(at=before)
        self.assertFalse(any(
            run.job_key == ScheduledJobRun.JOB_GSHEET_PRECLOSE
            for run in early_runs
        ))

        due_runs = run_due_jobs(at=due)
        sheet_runs = [
            run for run in due_runs
            if run.job_key == ScheduledJobRun.JOB_GSHEET_PRECLOSE
        ]
        self.assertEqual(len(sheet_runs), 1)
        self.assertEqual(sheet_runs[0].status, ScheduledJobRun.STATUS_SUCCESS)
        self.assertEqual(sheet_runs[0].result['imported'], 2)
        backup_runs = [
            run for run in due_runs
            if run.job_key == ScheduledJobRun.JOB_DATABASE_BACKUP
        ]
        self.assertEqual(len(backup_runs), 1)
        self.assertEqual(backup_runs[0].status, ScheduledJobRun.STATUS_SUCCESS)
        self.assertEqual(backup_runs[0].result['artifact'], 'test.dump')

        run_due_jobs(at=self._local(2026, 8, 17, 17, 5))
        self.assertEqual(
            ScheduledJobRun.objects.filter(
                job_key=ScheduledJobRun.JOB_GSHEET_PRECLOSE,
                business_date=date(2026, 8, 17),
            ).count(),
            1,
        )
        self.assertEqual(
            ScheduledJobRun.objects.filter(
                job_key=ScheduledJobRun.JOB_DATABASE_BACKUP,
                business_date=date(2026, 8, 17),
            ).count(),
            1,
        )
        self.assertEqual(_sync.call_count, 1)
        self.assertEqual(backup_process.call_count, 1)
        backup_command = backup_process.call_args.args[0]
        business_date_index = backup_command.index('-BusinessDate')
        self.assertEqual(backup_command[business_date_index + 1], '2026-08-17')
        not_before_index = backup_command.index('-NotBefore')
        self.assertIn('2026-08-17T17:00:00', backup_command[not_before_index + 1])
        self.assertNotIn('-ForceNew', backup_command)
        backup_process.assert_called_once_with(backup_command)

    @patch('app.scheduled_jobs._run_backup_process')
    @patch('app.gsheet_sync.is_configured', return_value=False)
    def test_sunday_catches_only_saturdays_missed_backup_without_cleanup(
        self, _configured, backup_process,
    ):
        backup_process.return_value = SimpleNamespace(
            returncode=0, stdout='saturday.dump\n', stderr='',
        )

        runs = run_due_jobs(at=self._local(2026, 8, 16, 23, 30))

        self.assertEqual(
            [run.job_key for run in runs],
            [
                ScheduledJobRun.JOB_GSHEET_PRECLOSE,
                ScheduledJobRun.JOB_DATABASE_BACKUP,
            ],
        )
        self.assertTrue(all(
            run.business_date == date(2026, 8, 15) for run in runs
        ))
        self.assertFalse(ScheduledJobRun.objects.filter(
            job_key=ScheduledJobRun.JOB_REPORT_CLEANUP,
        ).exists())
        self.assertTrue(ScheduledJobRun.objects.filter(
            job_key=ScheduledJobRun.JOB_DATABASE_BACKUP,
            business_date=date(2026, 8, 15),
        ).exists())
        self.assertFalse(ScheduledJobRun.objects.filter(
            business_date=date(2026, 8, 14),
        ).exists())

        # Once the newest due day is complete, the scheduler deliberately does
        # not walk backward and create misleading late dumps for older dates.
        self.assertEqual(
            run_due_jobs(at=self._local(2026, 8, 17, 9, 30)),
            [],
        )
        self.assertFalse(ScheduledJobRun.objects.filter(
            business_date=date(2026, 8, 14),
        ).exists())
        backup_process.assert_called_once()

    def test_cleanup_is_not_a_force_option_but_historical_rows_still_display(self):
        parser = RunScheduledJobsCommand().create_parser(
            'manage.py', 'run_scheduled_jobs',
        )
        force_action = next(
            action for action in parser._actions if action.dest == 'force'
        )
        self.assertNotIn(
            ScheduledJobRun.JOB_REPORT_CLEANUP,
            force_action.choices,
        )
        self.assertIn(
            ScheduledJobRun.JOB_DATABASE_BACKUP,
            force_action.choices,
        )

        historical = ScheduledJobRun.objects.create(
            job_key=ScheduledJobRun.JOB_REPORT_CLEANUP,
            trigger=ScheduledJobRun.TRIGGER_SCHEDULED,
            business_date=date(2026, 8, 15),
            status=ScheduledJobRun.STATUS_SUCCESS,
            summary='Historical cleanup result.',
        )
        self.assertEqual(
            historical.get_job_key_display(),
            'Daily report archive cleanup',
        )

    @override_settings(SCHEDULED_JOBS_ENABLED=True)
    @patch('app.management.commands.run_scheduled_jobs.run_due_jobs')
    def test_scheduler_self_test_does_not_run_due_jobs(self, due_jobs):
        ensure_store_hours()
        output = StringIO()

        call_command('run_scheduled_jobs', '--self-test', stdout=output)

        self.assertIn('scheduled jobs self-test passed', output.getvalue())
        self.assertIn('store_hours=7', output.getvalue())
        due_jobs.assert_not_called()

    @override_settings(SCHEDULED_JOBS_ENABLED=True)
    @patch('app.management.commands.run_scheduled_jobs.run_due_jobs')
    def test_scheduler_self_test_reports_missing_hours_without_repairing_them(
        self, due_jobs,
    ):
        ensure_store_hours()
        StoreHours.objects.filter(weekday=StoreHours.TUESDAY).delete()

        with self.assertRaisesMessage(
            CommandError,
            'requires all seven StoreHours rows',
        ):
            call_command('run_scheduled_jobs', '--self-test')

        self.assertEqual(StoreHours.objects.count(), 6)
        due_jobs.assert_not_called()

    def test_store_hours_rejects_closing_time_off_the_hour(self):
        ensure_store_hours()
        monday = StoreHours.objects.get(weekday=StoreHours.MONDAY)
        monday.closes_at = time(18, 15)

        with self.assertRaisesMessage(
            ValidationError,
            'Closing time must be on the hour',
        ):
            monday.full_clean()

    @override_settings(SCHEDULED_JOBS_ENABLED=True)
    @patch('app.management.commands.run_scheduled_jobs.run_due_jobs')
    def test_scheduler_self_test_rejects_misaligned_closing_time(
        self, due_jobs,
    ):
        ensure_store_hours()
        StoreHours.objects.filter(weekday=StoreHours.MONDAY).update(
            closes_at=time(18, 15),
        )

        with self.assertRaisesMessage(
            CommandError,
            'requires whole-hour closing times for: Monday',
        ):
            call_command('run_scheduled_jobs', '--self-test')

        due_jobs.assert_not_called()

    @patch('app.scheduled_jobs._run_backup_process')
    @patch('app.gsheet_sync.is_configured', return_value=True)
    @patch('app.gsheet_sync.sync_all', return_value={
        'last_sync': 1,
        'imported': 0,
        'tabs': [],
        'errors': ['Sheet unavailable'],
    })
    def test_backup_still_runs_when_preclosing_sheet_pull_fails(
        self, _sync, _configured, backup_process,
    ):
        backup_process.return_value = SimpleNamespace(
            returncode=0,
            stdout='test.dump\n',
            stderr='',
        )

        runs = run_due_jobs(at=self._local(2026, 8, 17, 17, 0))

        self.assertEqual(
            [run.job_key for run in runs],
            [
                ScheduledJobRun.JOB_GSHEET_PRECLOSE,
                ScheduledJobRun.JOB_DATABASE_BACKUP,
            ],
        )
        self.assertEqual(runs[0].status, ScheduledJobRun.STATUS_ERROR)
        self.assertEqual(runs[1].status, ScheduledJobRun.STATUS_SUCCESS)
        backup_process.assert_called_once()

    @patch('app.scheduled_jobs._run_backup_process')
    @patch('app.gsheet_sync.is_configured', return_value=False)
    def test_failed_preclosing_backup_retries_without_duplicate_row(
        self, _configured, backup_process,
    ):
        backup_process.side_effect = [
            SimpleNamespace(
                returncode=1,
                stdout='',
                stderr='pg_dump failed',
            ),
            SimpleNamespace(
                returncode=0,
                stdout='retry.dump\n',
                stderr='',
            ),
        ]

        run_due_jobs(at=self._local(2026, 8, 17, 17, 0))
        backup_run = ScheduledJobRun.objects.get(
            job_key=ScheduledJobRun.JOB_DATABASE_BACKUP,
            business_date=date(2026, 8, 17),
        )
        self.assertEqual(backup_run.status, ScheduledJobRun.STATUS_ERROR)
        self.assertIn('pg_dump failed', backup_run.error)
        ScheduledJobRun.objects.filter(pk=backup_run.pk).update(
            completed_at=timezone.now() - timedelta(hours=1),
        )

        run_due_jobs(at=self._local(2026, 8, 17, 18, 0))
        backup_run.refresh_from_db()
        self.assertEqual(backup_run.status, ScheduledJobRun.STATUS_SUCCESS)
        self.assertEqual(backup_run.attempt_count, 2)
        self.assertEqual(backup_run.result['artifact'], 'retry.dump')
        self.assertEqual(backup_process.call_count, 2)
        self.assertEqual(
            ScheduledJobRun.objects.filter(
                job_key=ScheduledJobRun.JOB_DATABASE_BACKUP,
                business_date=date(2026, 8, 17),
            ).count(),
            1,
        )

    @patch('app.scheduled_jobs._run_backup_process')
    @patch('app.gsheet_sync.is_configured', return_value=False)
    def test_timed_out_backup_is_terminalized_for_safe_retry(
        self, _configured, backup_process,
    ):
        backup_process.side_effect = subprocess.TimeoutExpired(
            ['powershell.exe', 'database-backup.ps1'], 900,
        )

        run_due_jobs(at=self._local(2026, 8, 17, 17, 0))

        backup_run = ScheduledJobRun.objects.get(
            job_key=ScheduledJobRun.JOB_DATABASE_BACKUP,
            business_date=date(2026, 8, 17),
        )
        self.assertEqual(backup_run.status, ScheduledJobRun.STATUS_ERROR)
        self.assertIn('timed out', backup_run.summary.lower())
        self.assertIn('process tree was terminated', backup_run.error)

    @patch('app.scheduled_jobs._run_backup_process')
    @patch('app.gsheet_sync.is_configured', return_value=True)
    @patch('app.gsheet_sync.sync_all')
    def test_successful_sheet_retry_refreshes_that_days_backup(
        self, sync_all, _configured, backup_process,
    ):
        sync_all.side_effect = [
            {
                'last_sync': 1,
                'imported': 0,
                'tabs': [],
                'errors': ['Sheet unavailable'],
            },
            {
                'last_sync': 2,
                'imported': 3,
                'tabs': [{'title': 'Orders', 'imported': 3}],
                'errors': [],
            },
        ]
        backup_process.side_effect = [
            SimpleNamespace(returncode=0, stdout='initial.dump\n', stderr=''),
            SimpleNamespace(returncode=0, stdout='refreshed.dump\n', stderr=''),
        ]

        run_due_jobs(at=self._local(2026, 8, 17, 17, 0))
        sheet_run = ScheduledJobRun.objects.get(
            job_key=ScheduledJobRun.JOB_GSHEET_PRECLOSE,
            business_date=date(2026, 8, 17),
        )
        self.assertEqual(sheet_run.status, ScheduledJobRun.STATUS_ERROR)
        ScheduledJobRun.objects.filter(pk=sheet_run.pk).update(
            completed_at=timezone.now() - timedelta(hours=1),
        )

        run_due_jobs(at=self._local(2026, 8, 17, 18, 0))

        sheet_run.refresh_from_db()
        backup_run = ScheduledJobRun.objects.get(
            job_key=ScheduledJobRun.JOB_DATABASE_BACKUP,
            business_date=date(2026, 8, 17),
        )
        self.assertEqual(sheet_run.status, ScheduledJobRun.STATUS_SUCCESS)
        self.assertEqual(sheet_run.attempt_count, 2)
        self.assertEqual(backup_run.status, ScheduledJobRun.STATUS_SUCCESS)
        self.assertEqual(backup_run.attempt_count, 2)
        self.assertEqual(backup_run.result['artifact'], 'refreshed.dump')
        self.assertEqual(backup_process.call_count, 2)
        self.assertIn('-ForceNew', backup_process.call_args_list[1].args[0])

    @patch('app.scheduled_jobs._run_backup_process')
    def test_closed_day_backup_runs_only_when_explicitly_forced(
        self, backup_process,
    ):
        backup_process.return_value = SimpleNamespace(
            returncode=0, stdout='forced.dump\n', stderr='',
        )

        runs = run_due_jobs(
            at=self._local(2026, 8, 16, 12, 0),
            force_job=ScheduledJobRun.JOB_DATABASE_BACKUP,
        )

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].job_key, ScheduledJobRun.JOB_DATABASE_BACKUP)
        self.assertEqual(runs[0].business_date, date(2026, 8, 16))
        backup_process.assert_called_once()

    @patch('app.scheduled_jobs._run_backup_process')
    @patch('app.gsheet_sync.is_configured', return_value=False)
    def test_backup_can_recover_after_three_failed_attempts(
        self, _configured, backup_process,
    ):
        due = self._local(2026, 8, 17, 17, 0)
        backup_run = ScheduledJobRun.objects.create(
            job_key=ScheduledJobRun.JOB_DATABASE_BACKUP,
            trigger=ScheduledJobRun.TRIGGER_SCHEDULED,
            business_date=date(2026, 8, 17),
            scheduled_for=due,
            status=ScheduledJobRun.STATUS_ERROR,
            attempt_count=3,
            completed_at=timezone.now() - timedelta(hours=1),
            error='Three prior failures',
        )
        backup_process.return_value = SimpleNamespace(
            returncode=0, stdout='recovered.dump\n', stderr='',
        )

        run_due_jobs(at=due)

        backup_run.refresh_from_db()
        self.assertEqual(backup_run.status, ScheduledJobRun.STATUS_SUCCESS)
        self.assertEqual(backup_run.attempt_count, 4)
        self.assertEqual(backup_run.result['artifact'], 'recovered.dump')

    @patch('app.gsheet_sync.is_configured', return_value=True)
    @patch('app.gsheet_sync.sync_all', return_value={
        'last_sync': 1, 'imported': 0, 'tabs': [], 'errors': [],
    })
    def test_manual_pull_is_saved_in_database(self, _sync, _configured):
        run, _result = run_google_sheet_sync()
        self.assertEqual(run.trigger, ScheduledJobRun.TRIGGER_MANUAL)
        self.assertEqual(run.status, ScheduledJobRun.STATUS_SUCCESS)
        self.assertIsNone(run.business_date)
