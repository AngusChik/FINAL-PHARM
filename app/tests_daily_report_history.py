"""Preservation, versioning, and navigation contracts for saved daily reports."""

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import DailyReportArchive
from .reporting import archive_daily_report, prune_daily_report_archives


class ReportHistoryBoundaryLinks(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.links = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        label = attrs.get('aria-label')
        if tag == 'a' and label in ('First page', 'Last page'):
            self.links[label] = attrs['href']


class DailyReportSnapshotTests(TestCase):
    def setUp(self):
        self.day = date.today()
        self.digest = {
            'day': self.day, 'inventory_day': self.day,
            'sales': {'revenue_today': Decimal('12.30'), 'orders_today': 1, 'units_sold': 2},
        }

    @patch('app.reporting.build_daily_report_pdf')
    def test_download_can_save_its_already_rendered_pdf_without_rendering_again(self, pdf):
        result = archive_daily_report(digest=self.digest, pdf=b'%PDF-downloaded-copy')

        self.assertEqual(bytes(result.pdf), b'%PDF-downloaded-copy')
        pdf.assert_not_called()

    @patch('app.reporting.build_daily_report_pdf', return_value=b'%PDF-version-one')
    def test_unchanged_content_skips_rendering_and_preserves_saved_time(self, pdf):
        first = archive_daily_report(digest=self.digest)
        repeated = deepcopy(self.digest)
        repeated['generated_at'] = timezone.now()

        second = archive_daily_report(digest=repeated)

        self.assertEqual(second.pk, first.pk)
        self.assertEqual(second.updated_at, first.updated_at)
        self.assertEqual(DailyReportArchive.objects.count(), 1)
        pdf.assert_called_once_with(self.digest)
        self.assertEqual(second.snapshot_data['sales']['revenue_today'], '12.30')
        self.assertEqual(second.snapshot_data['day'], self.day.isoformat())

    @patch('app.reporting.build_daily_report_pdf', side_effect=[b'first', b'changed', b'reverted'])
    def test_changes_and_return_to_original_values_keep_all_versions(self, pdf):
        first = archive_daily_report(digest=self.digest)
        changed = deepcopy(self.digest)
        changed['sales']['revenue_today'] = Decimal('24.60')
        second = archive_daily_report(digest=changed)
        third = archive_daily_report(digest=self.digest)

        self.assertEqual(len({first.pk, second.pk, third.pk}), 3)
        self.assertEqual(first.content_sha256, third.content_sha256)
        self.assertEqual(list(DailyReportArchive.objects.values_list('pk', flat=True)),
                         [third.pk, second.pk, first.pk])
        first.refresh_from_db()
        self.assertEqual(bytes(first.pdf), b'first')
        self.assertEqual(first.snapshot_data['sales']['revenue_today'], '12.30')

    @patch('app.reporting.build_daily_report_pdf', return_value=b'new')
    def test_existing_legacy_pdfs_and_old_reports_survive_new_snapshot(self, pdf):
        old = DailyReportArchive.objects.create(report_date=self.day - timedelta(days=365), pdf=b'old')
        legacy = DailyReportArchive.objects.create(report_date=self.day, pdf=b'legacy', summary='Original')
        saved_at = legacy.updated_at

        current = archive_daily_report(digest=self.digest)
        self.assertEqual(prune_daily_report_archives(reference_date=self.day), 0)

        self.assertEqual(DailyReportArchive.objects.count(), 3)
        legacy.refresh_from_db()
        self.assertEqual((bytes(legacy.pdf), legacy.summary, legacy.updated_at), (b'legacy', 'Original', saved_at))
        self.assertNotEqual(current.pk, legacy.pk)
        self.assertTrue(DailyReportArchive.objects.filter(pk=old.pk).exists())

    @patch('app.reporting.build_daily_report_pdf', return_value=b'original')
    def test_hiding_latest_snapshot_does_not_duplicate_or_unhide_it_on_refresh(self, pdf):
        original = archive_daily_report(digest=self.digest)
        hidden_at = timezone.now()
        DailyReportArchive.objects.filter(pk=original.pk).update(archived_at=hidden_at)

        result = archive_daily_report(digest=self.digest)

        self.assertEqual(result.pk, original.pk)
        self.assertEqual(result.archived_at, hidden_at)
        self.assertEqual(pdf.call_count, 1)

    @patch('app.reporting.build_daily_report_pdf', side_effect=RuntimeError('PDF unavailable'))
    def test_failed_pdf_generation_never_replaces_existing_snapshot(self, pdf):
        original = DailyReportArchive.objects.create(report_date=self.day, pdf=b'existing')

        with self.assertRaises(RuntimeError):
            archive_daily_report(digest=self.digest)

        self.assertEqual(DailyReportArchive.objects.count(), 1)
        original.refresh_from_db()
        self.assertEqual(bytes(original.pdf), b'existing')


@override_settings(AXES_ENABLED=False)
class DailyReportHistoryTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='history-admin', is_staff=True)
        self.client.force_login(self.user)
        self.day = date.today()

    def snapshot(self, day=None, **values):
        return DailyReportArchive.objects.create(
            report_date=day or self.day, pdf=b'%PDF-retained-report', **values,
        )

    def test_all_saved_versions_have_paginated_navigation_with_filters(self):
        for number in range(52):
            self.snapshot(summary=f'Matching report {number}')
        self.snapshot(summary='Unrelated report')
        self.snapshot(self.day - timedelta(days=1), summary='Matching older report')
        self.snapshot(summary='Matching hidden report', archived_at=timezone.now())
        origin = reverse('daily_report') + '?date=2026-01-02&ignore_snacks=1#report-stock'

        response = self.client.get(reverse('daily_report_history'), {
            'q': 'Matching', 'date_from': self.day.isoformat(), 'date_to': self.day.isoformat(),
            'page': 2, 'return_to': origin,
        })

        self.assertEqual(response.status_code, 200)
        page = response.context['page_obj']
        self.assertEqual((page.paginator.count, page.number, page.paginator.num_pages), (52, 2, 3))
        self.assertEqual(len(page.object_list), 25)
        self.assertEqual(response.context['report_return'], origin)
        for label in ('Previous', 'Next'):
            self.assertContains(response, f'>{label}</a>')
        query = parse_qs(response.context['query_string'])
        boundary_links = ReportHistoryBoundaryLinks(response.content.decode()).links
        for label, target in (('First page', 1), ('Last page', 3)):
            self.assertIn(label, boundary_links)
            self.assertEqual(parse_qs(urlsplit(boundary_links[label]).query), {
                **query, 'page': [str(target)],
            })
        self.assertEqual(query['q'], ['Matching'])
        self.assertEqual(query['date_from'], [self.day.isoformat()])
        self.assertEqual(query['return_to'], [origin])
        self.assertNotIn('page', query)
        self.assertTrue(all('pdf' in report.get_deferred_fields() for report in page.object_list))
        last = self.client.get(reverse('daily_report_history'), {
            'q': 'Matching', 'date_from': self.day.isoformat(), 'page': 3,
        })
        self.assertEqual(len(last.context['page_obj'].object_list), 2)

    def test_date_search_invalid_range_and_return_url_are_safe(self):
        report = self.snapshot()
        response = self.client.get(reverse('daily_report_history'), {
            'q': self.day.isoformat(), 'return_to': 'https://elsewhere.example/',
        })
        self.assertEqual([item.pk for item in response.context['page_obj']], [report.pk])
        self.assertEqual(response.context['report_return'], reverse('daily_report'))
        for filters in ({'date_from': '2026-02-30'}, {'date_to': 'not-a-date'},
                        {'date_from': '2026-09-08', 'date_to': '2026-01-01'}):
            with self.subTest(filters=filters):
                response = self.client.get(reverse('daily_report_history'), filters)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.context['filter_errors'])
                self.assertEqual(response.context['page_obj'].paginator.count, 0)

    def test_hide_and_restore_preserve_pdf_data_and_original_save_time(self):
        report = self.snapshot(summary='Original summary', snapshot_data={'preserved': True})
        saved_at = report.updated_at
        history = reverse('daily_report_history') + '?visibility=all&page=2'

        response = self.client.post(reverse('daily_report_archive_delete', args=[report.pk]), {'return_to': history})

        self.assertRedirects(response, history, fetch_redirect_response=False)
        report.refresh_from_db()
        self.assertIsNotNone(report.archived_at)
        self.assertEqual(report.archived_by, self.user)
        self.assertEqual((bytes(report.pdf), report.snapshot_data, report.updated_at),
                         (b'%PDF-retained-report', {'preserved': True}, saved_at))
        self.assertEqual(self.client.get(reverse('daily_report_history')).context['page_obj'].paginator.count, 0)
        hidden = self.client.get(reverse('daily_report_history'), {'visibility': 'hidden'})
        self.assertContains(hidden, 'Restore')
        self.assertEqual(hidden.context['page_obj'].paginator.count, 1)
        pdf = self.client.get(reverse('daily_report_archive_pdf', args=[report.pk]), {'download': '1'})
        self.assertEqual(pdf.content, b'%PDF-retained-report')
        self.assertIn(f'snapshot_{report.pk}.pdf', pdf['Content-Disposition'])

        self.client.post(reverse('daily_report_archive_restore', args=[report.pk]), {'return_to': history})

        report.refresh_from_db()
        self.assertIsNone(report.archived_at)
        self.assertIsNone(report.archived_by)
        self.assertEqual(report.updated_at, saved_at)
        self.assertEqual(bytes(report.pdf), b'%PDF-retained-report')

    def test_locked_accounts_cannot_browse_download_hide_or_restore_history(self):
        report = self.snapshot()
        locked = User.objects.create_user(username='history-locked')
        self.client.force_login(locked)
        endpoints = (
            ('get', 'daily_report_history', []),
            ('get', 'daily_report_archive_pdf', [report.pk]),
            ('post', 'daily_report_archive_delete', [report.pk]),
            ('post', 'daily_report_archive_restore', [report.pk]),
        )
        for method, route, args in endpoints:
            with self.subTest(route=route):
                response = getattr(self.client, method)(reverse(route, args=args))
                self.assertEqual(response.status_code, 302)
                self.assertEqual(urlsplit(response.url).path, reverse('passkey_unlock'))
        report.refresh_from_db()
        self.assertIsNone(report.archived_at)
