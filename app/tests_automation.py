import json
from datetime import date, datetime, time
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from app.inventory_audit import run_inventory_audit
from app.models import (
    DailyReportArchive,
    InventoryAuditRun,
    Product,
    ProductLot,
    ScheduledJobRun,
    StoreHours,
)
from app.reporting import prune_daily_report_archives
from app.scheduled_jobs import (
    next_gsheet_pull,
    run_due_jobs,
    run_google_sheet_sync,
    store_hours_payload,
)


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

    def test_protected_repair_assigns_missing_stock_without_changing_total(self):
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


class InventoryAuditAPITests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.staff = User.objects.create_user('audit-admin', password='test-pass', is_staff=True)
        self.user = User.objects.create_user('audit-user', password='test-pass')

    def test_signed_in_user_can_view_latest_but_cannot_run(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse('inventory_integrity_api'))
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['run'])

        response = self.client.post(
            reverse('inventory_integrity_api'),
            data=json.dumps({'action': 'run'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 403)
        self.assertTrue(response.json()['requires_admin'])

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

    def test_dashboard_hours_and_next_pull_share_database_schedule(self):
        payload = store_hours_payload()
        self.assertEqual(payload['1'], [9, 30, 18, 0])
        self.assertIsNone(payload['0'])

        next_pull = next_gsheet_pull(at=self._local(2026, 8, 17, 17, 0))
        self.assertEqual(
            timezone.localtime(next_pull['scheduled_for']).time(),
            time(17, 30),
        )
        self.assertFalse(next_pull['due'])

    @patch('app.gsheet_sync.is_configured', return_value=True)
    @patch('app.gsheet_sync.sync_all', return_value={
        'last_sync': 1,
        'imported': 2,
        'tabs': [{'title': 'Orders', 'imported': 2}],
        'errors': [],
    })
    def test_preclosing_pull_runs_once_at_530_on_monday(self, _sync, _configured):
        before = self._local(2026, 8, 17, 17, 29)
        due = self._local(2026, 8, 17, 17, 30)

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

        run_due_jobs(at=self._local(2026, 8, 17, 17, 35))
        self.assertEqual(
            ScheduledJobRun.objects.filter(
                job_key=ScheduledJobRun.JOB_GSHEET_PRECLOSE,
                business_date=date(2026, 8, 17),
            ).count(),
            1,
        )
        self.assertEqual(_sync.call_count, 1)

    @patch('app.gsheet_sync.is_configured', return_value=True)
    @patch('app.gsheet_sync.sync_all', return_value={
        'last_sync': 1, 'imported': 0, 'tabs': [], 'errors': [],
    })
    def test_manual_pull_is_saved_in_database(self, _sync, _configured):
        run, _result = run_google_sheet_sync()
        self.assertEqual(run.trigger, ScheduledJobRun.TRIGGER_MANUAL)
        self.assertEqual(run.status, ScheduledJobRun.STATUS_SUCCESS)
        self.assertIsNone(run.business_date)
