"""Contracts for the compact sheet, targeted deletion and honest pull times."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from app.models import OrderingSheetEntry, ScheduledJobRun, UserAction


@override_settings(AXES_ENABLED=False, TIME_ZONE='America/Toronto')
class OrderingRedesignTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.staff = get_user_model().objects.create_user('sheet-staff', is_staff=True)
        cls.pu = get_user_model().objects.create_user('sheet-pu')

    def setUp(self):
        self.url = reverse('ordering_sheet')
        self.client.force_login(self.staff)
        for name, value in (
            ('app.gsheet_sync.is_configured', True),
            ('app.gsheet_sync.load_state', None),
            ('app.scheduled_jobs.next_gsheet_pull', None),
        ):
            mock = patch(name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def entry(self, **kwargs):
        return OrderingSheetEntry.objects.create(name='Shared drug', initials='AB', created_by=self.staff, **kwargs)

    def delete(self, ids, *, bulk=False, suffix=''):
        payload = {'action': 'delete_selected', 'entry_ids': ','.join(map(str, ids))} if bulk else {
            'action': 'delete', 'entry_id': ids[0],
        }
        return self.client.post(self.url + suffix, payload, HTTP_X_ORDERING_DELETE='1')

    def test_json_deletion_for_both_roles_and_views_is_recoverable_and_idempotent(self):
        for user in (self.staff, self.pu):
            self.client.force_login(user)
            for suffix in ('', '?embed=1'):
                for bulk in (False, True):
                    with self.subTest(user=user.username, suffix=suffix, bulk=bulk):
                        entries = [self.entry(order_note='Keep audit context') for _ in range(2 if bulk else 1)]
                        untouched = self.entry()
                        ids = [entry.pk for entry in entries]
                        before = UserAction.objects.filter(action='ordering_delete').count()
                        response = self.delete(ids, bulk=bulk, suffix=suffix)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json()['deleted_ids'], ids)
                        self.assertTrue(response.json()['ok'])
                        for entry in entries:
                            entry.refresh_from_db()
                            self.assertTrue(entry.is_deleted)
                            self.assertEqual(entry.deleted_by_id, user.pk)
                            self.assertIsNotNone(entry.deleted_at)
                            self.assertEqual(entry.order_note, 'Keep audit context')
                        self.assertFalse(OrderingSheetEntry.objects.get(pk=untouched.pk).is_deleted)
                        self.assertEqual(UserAction.objects.filter(action='ordering_delete').count(), before + len(ids))
                        snapshot = list(OrderingSheetEntry.objects.filter(pk__in=ids).values())
                        retry = self.delete(ids, bulk=bulk, suffix=suffix)
                        self.assertTrue(retry.json()['ok'])
                        self.assertEqual(list(OrderingSheetEntry.objects.filter(pk__in=ids).values()), snapshot)
                        self.assertEqual(UserAction.objects.filter(action='ordering_delete').count(), before + len(ids))

    def test_delete_rejects_invalid_missing_and_unsigned_requests(self):
        entry = self.entry()
        for invalid in ('', 'nope', '1.5', '-1', '9' * 100, f'{entry.pk},99'):
            response = self.delete([invalid])
            self.assertEqual(response.status_code, 400)
            self.assertFalse(response.json()['ok'])
        self.assertEqual(self.delete([99999999]).status_code, 404)
        strict = Client(enforce_csrf_checks=True)
        strict.cookies[settings.SESSION_COOKIE_NAME] = self.client.cookies[settings.SESSION_COOKIE_NAME].value
        self.assertEqual(strict.post(self.url, {'action': 'delete', 'entry_id': entry.pk}, HTTP_X_ORDERING_DELETE='1', HTTP_X_REQUESTED_WITH='XMLHttpRequest').status_code, 403)
        self.client.logout()
        self.assertEqual(self.delete([entry.pk]).status_code, 302)
        entry.refresh_from_db()
        self.assertFalse(entry.is_deleted)
        self.assertFalse(UserAction.objects.filter(action='ordering_delete').exists())

    def test_input_and_table_orders_and_roles_match_in_both_views(self):
        self.entry()
        for user in (self.staff, self.pu):
            self.client.force_login(user)
            for suffix in ('', '?embed=1'):
                response = self.client.get(self.url + suffix)
                html = response.content.decode()
                form = html[html.index('id="os-form"'):html.index('id="os-form-otc"')]
                fields = ['patient_name', 'reasoning', 'urgency', 'name', 'quantity_needed', 'quantity_remaining', 'initials']
                positions = [form.index(f'name="drug-{field}"') for field in fields]
                self.assertEqual(positions, sorted(positions))
                head = html[html.index('<thead>'):html.index('</thead>')]
                headings = ['os-check-col', 'os-patient-col', 'os-reason-col', 'os-name-col', 'os-qty-needed-col', 'os-qty-remaining-col', 'os-initial-date-col', 'os-actions-col']
                positions = [head.index(heading) for heading in headings]
                self.assertEqual(positions, sorted(positions))
                self.assertNotIn('>#<', head)
                self.assertIn('data-column-key="reasoning"', head)
                self.assertIn('data-column-key="drug-name"', head)
                self.assertContains(response, 'Staff account' if user.is_staff else 'PU account')
                self.assertContains(response, 'os-delete-form')
                self.assertEqual(response.context['can_administer'], user.is_staff)
                self.assertTrue(response.context['can_delete_entries'])

    def make_run(self, status, completed):
        run = ScheduledJobRun.objects.create(
            job_key=ScheduledJobRun.JOB_GSHEET_PRECLOSE,
            trigger=ScheduledJobRun.TRIGGER_MANUAL,
            status=status, completed_at=completed,
        )
        ScheduledJobRun.objects.filter(pk=run.pk).update(started_at=completed - timedelta(seconds=10))
        return run

    def test_last_success_stays_visible_despite_later_failed_skipped_or_running_pulls(self):
        completed = datetime(2026, 10, 18, 13, 42, tzinfo=timezone.utc)
        self.make_run(ScheduledJobRun.STATUS_SUCCESS, completed)
        for i, status in enumerate((ScheduledJobRun.STATUS_ERROR, ScheduledJobRun.STATUS_SKIPPED, ScheduledJobRun.STATUS_RUNNING), 1):
            self.make_run(status, completed + timedelta(hours=i))
            with patch('app.scheduled_jobs.next_gsheet_pull', return_value={'scheduled_for': completed + timedelta(days=1), 'due': False}):
                for suffix in ('', '?embed=1'):
                    response = self.client.get(self.url + suffix)
                    self.assertEqual(response.context['gsheet_last_sync'], '18/10/2026, 09:42')
                    self.assertContains(response, 'Last pulled:')
                    self.assertContains(response, '18/10/2026, 09:42')
                    self.assertContains(response, 'Last attempt failed' if i == 1 else 'Last attempt skipped')

    def test_missing_and_failed_legacy_history_is_never_presented_as_a_success(self):
        for state in (None, {'last_sync': 12345, 'errors': ['failed']}, {'last_sync': 'bad', 'errors': []}):
            with patch('app.gsheet_sync.load_state', return_value=state):
                self.assertContains(self.client.get(self.url), 'Never pulled')
        completed = datetime(2026, 10, 18, 13, 42, tzinfo=timezone.utc)
        with patch('app.gsheet_sync.load_state', return_value={'last_sync': completed.timestamp(), 'errors': []}):
            self.assertEqual(self.client.get(self.url).context['gsheet_last_sync'], '18/10/2026, 09:42')

    def test_pu_cannot_pull_or_update_status_but_can_delete(self):
        entry = self.entry()
        self.client.force_login(self.pu)
        response = self.client.get(self.url)
        self.assertContains(response, 'disabled title="Staff or passkey access is required')
        with patch('app.scheduled_jobs.run_google_sheet_sync') as sync:
            self.client.post(self.url, {'action': 'sync_gsheet'})
            sync.assert_not_called()
        self.client.post(self.url, {'action': 'update_status', 'entry_id': entry.pk, 'status': 'ordered', 'status_only': '1'})
        entry.refresh_from_db()
        self.assertEqual(entry.status, 'pending')
        self.assertTrue(self.delete([entry.pk]).json()['ok'])
