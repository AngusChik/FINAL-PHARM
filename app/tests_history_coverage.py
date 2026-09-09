from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.paginator import Paginator
from django.db import connection
from django.test import TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from .models import (
    CheckinSession, LabelSession, LabelSessionItem, LoginAudit, StockChange, UserAction,
)
from .views import ActivityLogView


@override_settings(AXES_ENABLED=False)
class ActivityHistoryCoverageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(username='history-admin', is_staff=True)
        cls.stamp = timezone.now() - timedelta(days=60)
        LoginAudit.objects.bulk_create([
            LoginAudit(user=cls.user, username=f'login-{i}', success=True)
            for i in range(505)
        ])
        StockChange.objects.bulk_create([
            StockChange(user=cls.user, product_name=f'stock-{i}', change_type='return', quantity=i + 1)
            for i in range(505)
        ])
        UserAction.objects.bulk_create([
            UserAction(user=cls.user, action='transaction_correction', target=f'action-{i}', detail=f'Original reason {i}')
            for i in range(505)
        ])
        for model in (LoginAudit, StockChange, UserAction):
            model.objects.all().update(timestamp=cls.stamp)

    def events(self, event_type='', user='', date_from=None, date_to=None):
        return ActivityLogView()._build_events(event_type, user, date_from, date_to)

    def test_all_sources_remain_accessible_beyond_old_caps_with_stable_ties(self):
        pages = Paginator(self.events(), 100)
        self.assertEqual(pages.count, 1515)
        self.assertEqual(pages.num_pages, 16)
        details = [event['detail'] for number in pages.page_range for event in pages.page(number)]
        self.assertEqual(len(details), 1515)
        self.assertEqual(len(set(details[:505])), 505)
        self.assertIn('action-0', details[504])
        self.assertIn('stock-0', details[-1])
        self.assertEqual(details[:100], [event['detail'] for event in pages.page(1)])

    def test_html_page_fetches_only_one_slice_of_history(self):
        with CaptureQueriesContext(connection) as queries:
            page = Paginator(self.events(), 50).page(11)
            events = list(page)
        self.assertEqual(len(events), 50)
        self.assertLessEqual(len(queries), 6)
        self.assertTrue(any('LIMIT 50 OFFSET 500' in query['sql'] for query in queries))

    def test_each_source_supports_full_counts_and_direct_new_filters(self):
        for event_type in ('all_logins', 'stock:return', 'action:transaction_correction'):
            with self.subTest(event_type=event_type):
                self.assertEqual(len(self.events(event_type)), 505)
        self.assertIn('Original reason', self.events('action:transaction_correction')[0]['detail'])

    def test_filters_apply_to_complete_history_before_pagination(self):
        self.assertEqual(len(self.events('all_logins', user='login-504')), 1)
        day = timezone.localtime(self.stamp).date()
        self.assertEqual(len(self.events(date_from=day, date_to=day)), 1515)
        self.assertEqual(len(self.events(date_from=day + timedelta(days=1))), 0)

    def test_pdf_receives_every_matching_record_including_oldest(self):
        self.client.force_login(self.user)
        with patch.object(ActivityLogView, '_render_pdf') as render_pdf:
            from django.http import HttpResponse
            render_pdf.return_value = HttpResponse(b'PDF', content_type='application/pdf')
            response = self.client.get(reverse('activity_log'), {'export': 'pdf'})
        self.assertEqual(response.status_code, 200)
        events = list(render_pdf.call_args.args[0])
        self.assertEqual(len(events), 1515)
        self.assertIn('stock-0', events[-1]['detail'])

    def test_page_exposes_all_stock_and_action_filter_choices(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse('activity_log'), {'type': 'stock:return'})
        self.assertEqual(response.context['page_obj'].paginator.count, 505)
        for key, _ in StockChange.CHANGE_TYPE_CHOICES:
            self.assertContains(response, f'value="stock:{key}"')
        for key, _ in UserAction.ACTION_CHOICES:
            self.assertContains(response, f'value="action:{key}"')


@override_settings(AXES_ENABLED=False)
class SavedLabelHistoryTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='label-retention', is_staff=True)
        self.other = get_user_model().objects.create_user(username='other-label-retention', is_staff=True)
        self.client.force_login(self.user)
        self.saved = LabelSession.objects.create(user=self.user, label_count=2)
        self.item = LabelSessionItem.objects.create(
            session=self.saved, product_name='Original custom label', product_price='2.50',
            qty=2, is_custom=True, custom_lines=[{'text': 'Original text', 'price': 2.5}],
        )

    def test_remove_and_clear_keep_snapshots_and_are_user_scoped(self):
        other = LabelSession.objects.create(user=self.other, label_count=1)
        removed = self.client.post(reverse('label_session_delete', args=[self.saved.pk]))
        self.assertEqual(removed.status_code, 200)
        self.saved.refresh_from_db()
        self.item.refresh_from_db()
        self.assertIsNotNone(self.saved.archived_at)
        self.assertEqual(self.item.custom_lines[0]['text'], 'Original text')
        self.assertEqual(self.client.get(reverse('label_sessions')).json()['sessions'], [])
        second = LabelSession.objects.create(user=self.user, label_count=1)
        cleared = self.client.post(reverse('label_sessions_clear'))
        self.assertEqual(cleared.json()['deleted'], 1)
        self.assertEqual(LabelSession.objects.filter(user=self.user).count(), 2)
        second.refresh_from_db()
        other.refresh_from_db()
        self.assertIsNotNone(second.archived_at)
        self.assertIsNone(other.archived_at)
        self.assertEqual(self.client.post(reverse('label_session_delete', args=[other.pk])).status_code, 404)

    def test_saved_hidden_snapshot_remains_readable_to_its_owner(self):
        self.client.post(reverse('label_session_delete', args=[self.saved.pk]))
        response = self.client.get(reverse('label_session_detail', args=[self.saved.pk]))
        self.assertEqual(response.json()['items'][0]['product_name'], 'Original custom label')


@override_settings(AXES_ENABLED=False)
class RecentScanHistoryTests(TestCase):
    def test_dashboard_and_session_show_latest_100_scans(self):
        user = get_user_model().objects.create_user(username='scans-100', is_staff=True)
        self.client.force_login(user)
        session = CheckinSession.objects.create(user=user)
        StockChange.objects.bulk_create([
            StockChange(user=user, session=session, product_name=f'Scan {i}', change_type='checkin', quantity=1)
            for i in range(105)
        ])
        for url in (reverse('checkin_dashboard'), reverse('checkin_session', args=[session.pk])):
            response = self.client.get(url, {'format': 'recent_scans'}, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
            self.assertEqual(response.status_code, 200)
            entries = response.json()['entries']
            self.assertEqual(len(entries), 100)
            self.assertEqual(entries[0]['name'], 'Scan 104')
            self.assertEqual(entries[-1]['name'], 'Scan 5')
