import time
from datetime import date, datetime, time as datetime_time, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from unittest.mock import Mock, call, patch
from urllib.parse import parse_qs, urlencode, urlsplit

from django.contrib.auth.models import AnonymousUser, User
from django.contrib.messages import get_messages
from django.contrib.messages.storage.fallback import FallbackStorage
from django.http import HttpResponse
from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .mixins import PASSKEY_SESSION_KEY
from .models import Category, Order, OrderDetail, Product, ProductLot, StockChange
from .navigation import product_return_label, safe_product_details_return_url
from .views import DailyReportArchiveDeleteView, DailyReportHistoryView, DailyReportPDFView, DailyReportView, _daily_report_date


class DailyReportViewTests(SimpleTestCase):
    """Exercise date/auth/export/archive boundaries without writing archives."""

    def setUp(self):
        self.factory = RequestFactory()
        self.today = date.today()
        self.staff = User(username='report-admin', is_staff=True)

    def request(self, route='daily_report', query=None, user=None, unlocked=False):
        request = self.factory.get(reverse(route), query or {})
        request.user = self.staff if user is None else user
        request.session = {PASSKEY_SESSION_KEY: time.time()} if unlocked else {}
        request._messages = FallbackStorage(request)
        return request

    @staticmethod
    def digest(day, exclude_snacks=False):
        return {'day': day, 'exclude_snacks': exclude_snacks}

    def test_date_validator_accepts_default_and_lower_bound(self):
        self.assertEqual(_daily_report_date(self.request()), self.today)
        self.assertEqual(_daily_report_date(self.request(query={'date': ''})), self.today)
        self.assertEqual(_daily_report_date(self.request(query={'date': '1900-01-01'})), date(1900, 1, 1))

    def test_date_validator_rejects_invalid_future_and_pre_1900_dates(self):
        for raw in ('not-a-date', '2026-02-30', '0001-01-01', '1899-12-31',
                    (self.today + timedelta(days=1)).isoformat()):
            with self.subTest(date=raw), self.assertRaises(ValueError):
                _daily_report_date(self.request(query={'date': raw}))

    @patch('app.reporting.archive_daily_report')
    @patch('app.daily_reporting.build_daily_report')
    def test_page_and_pdf_require_admin_access_before_building_report(self, build, archive):
        for route, view in (
            ('daily_report', DailyReportView), ('daily_report_pdf', DailyReportPDFView),
            ('daily_report_history', DailyReportHistoryView),
        ):
            with self.subTest(route=route, role='anonymous'):
                response = view.as_view()(self.request(route, user=AnonymousUser()))
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.url, reverse('login'))
            with self.subTest(route=route, role='locked staff account'):
                request = self.request(route, query={'date': '2000-01-01', 'ignore_snacks': '1'},
                                       user=User(username='PU', is_staff=False))
                response = view.as_view()(request)
                destination = urlsplit(response.url)
                self.assertEqual(response.status_code, 302)
                self.assertEqual(destination.path, reverse('passkey_unlock'))
                self.assertEqual(parse_qs(destination.query)['next'], [request.get_full_path()])
        build.assert_not_called()
        archive.assert_not_called()

    @patch('app.views.render', return_value=HttpResponse('report'))
    @patch('app.reporting.archive_daily_report')
    @patch('app.daily_reporting.build_daily_report')
    def test_today_full_report_reuses_the_visible_digest_for_archive(self, build, archive, render):
        digest = self.digest(self.today)
        build.return_value = digest

        response = DailyReportView.as_view()(self.request())

        self.assertEqual(response.status_code, 200)
        build.assert_called_once_with(self.today, exclude_snacks=False)
        archive.assert_called_once_with(digest=digest)
        context = render.call_args.args[2]
        self.assertIs(context['digest'], digest)
        self.assertTrue(context['is_today'])
        self.assertFalse(context['archive_error'])
        self.assertIsNone(context['next_day'])
        self.assertEqual(context['previous_day'], self.today - timedelta(days=1))
        self.assertEqual(parse_qs(context['report_query']), {'date': [self.today.isoformat()]})

    @patch('app.views.render', return_value=HttpResponse('report'))
    @patch('app.reporting.archive_daily_report')
    @patch('app.daily_reporting.build_daily_report')
    def test_filtered_or_historical_view_archives_selected_days_full_report(self, build, archive, render):
        selected = self.today - timedelta(days=12)
        for day, ignore_snacks in ((self.today, True), (selected, False), (selected, True)):
            with self.subTest(day=day, ignore_snacks=ignore_snacks):
                build.reset_mock()
                archive.reset_mock()
                visible = self.digest(day, exclude_snacks=ignore_snacks)
                canonical = self.digest(day)
                build.side_effect = [visible, canonical] if ignore_snacks else [visible]
                query = {'date': day.isoformat()}
                if ignore_snacks:
                    query['ignore_snacks'] = '1'

                response = DailyReportView.as_view()(self.request(query=query))

                self.assertEqual(response.status_code, 200)
                expected_builds = [call(day, exclude_snacks=ignore_snacks)]
                if ignore_snacks:
                    expected_builds.append(call(day))
                self.assertEqual(build.call_args_list, expected_builds)
                archive.assert_called_once_with(digest=canonical if ignore_snacks else visible)
                context = render.call_args.args[2]
                self.assertIs(context['digest'], visible)
                self.assertEqual(context['current_day'], self.today)
                self.assertEqual(parse_qs(context['report_query']), {key: [value] for key, value in query.items()})
                self.assertEqual(context['report_return'], reverse('daily_report') + '?' + context['report_query'])

    @patch('app.views.render', return_value=HttpResponse('report'))
    @patch('app.reporting.archive_daily_report')
    @patch('app.daily_reporting.build_daily_report')
    def test_earliest_date_has_no_previous_navigation_and_unlocked_pu_can_view(self, build, archive, render):
        day = date(1900, 1, 1)
        build.return_value = self.digest(day)
        request = self.request(query={'date': day.isoformat()}, user=User(username='PU'), unlocked=True)

        response = DailyReportView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        context = render.call_args.args[2]
        self.assertIsNone(context['previous_day'])
        self.assertEqual(context['next_day'], date(1900, 1, 2))
        self.assertFalse(context['is_today'])

    @patch('app.views.render', return_value=HttpResponse('report'))
    @patch('app.reporting.archive_daily_report')
    @patch('app.daily_reporting.build_daily_report')
    def test_invalid_page_date_falls_back_with_warning_and_normalized_query(self, build, archive, render):
        build.return_value = self.digest(self.today)
        for raw in ('bad', '1899-12-31', (self.today + timedelta(days=1)).isoformat()):
            with self.subTest(date=raw):
                build.reset_mock()
                request = self.request(query={'date': raw})

                response = DailyReportView.as_view()(request)

                self.assertEqual(response.status_code, 200)
                build.assert_called_once_with(self.today, exclude_snacks=False)
                self.assertEqual(render.call_args.args[2]['today'], self.today)
                self.assertEqual(parse_qs(render.call_args.args[2]['report_query']), {'date': [self.today.isoformat()]})
                self.assertTrue(any('Showing today instead' in str(message) for message in get_messages(request)))

    @patch('app.views.render', return_value=HttpResponse('report'))
    @patch('app.reporting.archive_daily_report', side_effect=RuntimeError('Archive unavailable'))
    @patch('app.daily_reporting.build_daily_report')
    def test_archive_failure_keeps_report_available_and_sets_notice(self, build, archive, render):
        build.return_value = self.digest(self.today)

        response = DailyReportView.as_view()(self.request())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(render.call_args.args[2]['archive_error'])
        self.assertEqual(render.call_args.args[2]['digest']['day'], self.today)

    @patch('app.views.render', return_value=HttpResponse('report'))
    @patch('app.reporting.archive_daily_report')
    @patch('app.daily_reporting.build_daily_report')
    def test_canonical_report_failure_keeps_filtered_historical_report_available(self, build, archive, render):
        day = self.today - timedelta(days=2)
        digest = self.digest(day)
        build.side_effect = [digest, RuntimeError('Canonical data unavailable')]

        response = DailyReportView.as_view()(self.request(query={'date': day.isoformat(), 'ignore_snacks': '1'}))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(render.call_args.args[2]['archive_error'])
        self.assertIs(render.call_args.args[2]['digest'], digest)
        archive.assert_not_called()

    @patch('app.reporting.archive_daily_report')
    @patch('app.reporting.build_daily_report_pdf', return_value=b'%PDF-daily-report')
    @patch('app.daily_reporting.build_daily_report')
    def test_filtered_pdf_saves_full_selected_report_and_downloads_requested_filter(self, build, pdf, archive):
        day = self.today - timedelta(days=12)
        digest = self.digest(day, exclude_snacks=True)
        canonical = self.digest(day)
        build.side_effect = [digest, canonical]
        request = self.request('daily_report_pdf', {'date': day.isoformat(), 'ignore_snacks': '1'})

        response = DailyReportPDFView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertEqual(response['Content-Disposition'], f'attachment; filename="daily_report_{day:%Y%m%d}.pdf"')
        self.assertEqual(response.content, b'%PDF-daily-report')
        self.assertEqual(build.call_args_list, [call(day, exclude_snacks=True), call(day)])
        pdf.assert_called_once_with(digest)
        archive.assert_called_once_with(digest=canonical)
        self.assertEqual(response['X-Report-History'], 'saved')

    @patch('app.reporting.archive_daily_report')
    @patch('app.reporting.build_daily_report_pdf', return_value=b'%PDF-daily-report')
    @patch('app.daily_reporting.build_daily_report')
    def test_full_pdf_reuses_rendered_bytes_for_selected_day_snapshot(self, build, pdf, archive):
        day = self.today - timedelta(days=12)
        digest = self.digest(day)
        build.return_value = digest

        response = DailyReportPDFView.as_view()(self.request('daily_report_pdf', {'date': day.isoformat()}))

        self.assertEqual(response.status_code, 200)
        build.assert_called_once_with(day, exclude_snacks=False)
        archive.assert_called_once_with(digest=digest, pdf=b'%PDF-daily-report')

    @patch('app.reporting.archive_daily_report', side_effect=RuntimeError('Archive unavailable'))
    @patch('app.reporting.build_daily_report_pdf', return_value=b'%PDF-daily-report')
    @patch('app.daily_reporting.build_daily_report')
    def test_archive_failure_keeps_pdf_download_available_with_notice(self, build, pdf, archive):
        build.return_value = self.digest(self.today)
        request = self.request('daily_report_pdf')

        response = DailyReportPDFView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'%PDF-daily-report')
        self.assertEqual(response['X-Report-History'], 'unavailable')
        self.assertTrue(any('could not be saved' in str(message) for message in get_messages(request)))

    @patch('app.reporting.build_daily_report_pdf')
    @patch('app.daily_reporting.build_daily_report')
    def test_invalid_pdf_dates_fail_before_report_work(self, build, pdf):
        for raw in ('bad', '0001-01-01', '1899-12-31', (self.today + timedelta(days=1)).isoformat()):
            with self.subTest(date=raw):
                response = DailyReportPDFView.as_view()(self.request('daily_report_pdf', {'date': raw}))
                self.assertEqual(response.status_code, 400)
        build.assert_not_called()
        pdf.assert_not_called()

    def test_product_details_preserves_exact_safe_report_return_and_label(self):
        origin = reverse('inventory_display') + '?q=vitamin%20C&sort=name&page=3#inventory-row-7'
        report = reverse('daily_report') + '?' + urlencode({
            'date': '2026-01-02', 'ignore_snacks': '1', 'return_to': origin,
        }) + '#drLowStock'
        request = self.factory.get(reverse('product_details', args=[7]), {'return_to': report})

        self.assertEqual(safe_product_details_return_url(request, report), report)
        self.assertEqual(product_return_label(report), 'Back to Daily Report')
        for unsafe in ('https://elsewhere.example/reports/daily/?date=2026-01-02',
                       '//elsewhere.example/reports/daily/', reverse('daily_report_pdf')):
            with self.subTest(return_url=unsafe):
                self.assertEqual(safe_product_details_return_url(request, unsafe), reverse('inventory_display'))

    @patch('app.views.render', return_value=HttpResponse('report'))
    @patch('app.reporting.archive_daily_report')
    @patch('app.daily_reporting.build_daily_report')
    def test_origin_query_keeps_exact_explicit_return_for_date_controls(self, build, archive, render):
        day = self.today - timedelta(days=2)
        origin = reverse('inventory_display') + '?q=vitamin%20C&sort=-quantity&page=4#product-23'
        build.side_effect = [self.digest(day, exclude_snacks=True), self.digest(day)]

        response = DailyReportView.as_view()(self.request(query={
            'date': day.isoformat(), 'ignore_snacks': '1', 'return_to': origin,
        }))

        self.assertEqual(response.status_code, 200)
        context = render.call_args.args[2]
        self.assertEqual(context['origin_query'], '&' + urlencode({'return_to': origin}))
        self.assertEqual(parse_qs(urlsplit(context['report_return']).query), {
            'date': [day.isoformat()], 'ignore_snacks': ['1'], 'return_to': [origin],
        })

    @patch('app.views.DailyReportArchive.objects.filter')
    def test_archive_hide_retains_data_and_returns_only_to_report_pages(self, archives):
        archive = Mock(report_date=self.today)
        archives.return_value.first.return_value = archive
        report = reverse('daily_report') + '?date=2026-01-02&ignore_snacks=1#drLowStock'
        for candidate, expected in (
            (report, report),
            (reverse('daily_report_history') + '?visibility=hidden&page=2',
             reverse('daily_report_history') + '?visibility=hidden&page=2'),
            ('https://elsewhere.example/reports/daily/', reverse('daily_report')),
            ('//elsewhere.example/reports/daily/', reverse('daily_report')),
            (reverse('inventory_display'), reverse('daily_report')),
            (reverse('daily_report_pdf'), reverse('daily_report')),
            ('', reverse('daily_report')),
        ):
            with self.subTest(return_url=candidate):
                request = self.factory.post(reverse('daily_report_archive_delete', args=[41]), {'return_to': candidate})
                request.user = self.staff
                request.session = {}
                request._messages = FallbackStorage(request)

                response = DailyReportArchiveDeleteView.as_view()(request, pk=41)

                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.url, expected)
        self.assertEqual(archive.save.call_count, 7)
        archive.save.assert_called_with(update_fields=['archived_at', 'archived_by'])
        archive.delete.assert_not_called()
        self.assertIs(archive.archived_by, self.staff)


class _ReportHTML(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.anchors = []
        self.inputs = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.anchors.append(dict(attrs))
        elif tag == 'input':
            self.inputs.append(dict(attrs))


@override_settings(AXES_ENABLED=False)
class DailyReportRenderedTests(TestCase):
    def setUp(self):
        self.today = date.today()
        self.user = User.objects.create_user(username='report-render-admin', is_staff=True)
        self.client.force_login(self.user)

    @patch('app.reporting.archive_daily_report')
    def test_empty_report_renders_real_data_and_unavailable_margin(self, archive):
        day = self.today - timedelta(days=2)

        response = self.client.get(reverse('daily_report'), {'date': day.isoformat()})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'No sales for this day')
        self.assertContains(response, 'No sales margin available')
        self.assertContains(response, 'No stock activity recorded')
        self.assertContains(response, 'Stock sections show current inventory')
        self.assertEqual(response.context['digest']['inventory_day'], self.today)
        self.assertEqual(len(response.context['digest']['trend']), 7)
        self.assertEqual(archive.call_args.kwargs['digest']['day'], day)

    @patch('app.reporting.archive_daily_report')
    def test_populated_report_links_products_and_date_controls_with_exact_origin(self, archive):
        category = Category.objects.create(name='Daily render category')
        product = Product.objects.create(
            name='Daily render product', barcode='DAILY-RENDER-1', price=Decimal('12.00'),
            price_per_unit=Decimal('4.00'), quantity_in_stock=2, category=category,
        )
        ProductLot.objects.create(
            product=product, lot_number='DAILY-LOT', quantity_on_hand=2,
            expiry_date=self.today + timedelta(days=2),
        )
        selected = self.today - timedelta(days=2)
        timestamp = timezone.make_aware(datetime.combine(selected, datetime_time(12)))
        order = Order.objects.create(submitted=True)
        Order.objects.filter(pk=order.pk).update(order_date=timestamp)
        OrderDetail.objects.create(
            order=order, product=product, product_name=product.name, product_barcode=product.barcode,
            quantity=2, price=Decimal('12.00'), cost_per_unit_at_sale=Decimal('4.00'), taxable_at_sale=False,
        )
        change = StockChange.objects.create(product=product, change_type='checkin', quantity=3, user=self.user)
        StockChange.objects.filter(pk=change.pk).update(timestamp=timestamp)
        origin = reverse('inventory_display') + '?q=vitamin%20C&sort=name&page=3#product-7'

        response = self.client.get(reverse('daily_report'), {
            'date': selected.isoformat(), 'ignore_snacks': '1', 'return_to': origin,
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Product performance')
        self.assertContains(response, 'DAILY-LOT')
        self.assertContains(response, '$24.00')
        self.assertEqual(response.context['digest']['sales']['profit'], Decimal('16.00'))
        self.assertEqual(response.context['digest']['activity']['checkin_units'], 3)
        parsed = _ReportHTML(response.content.decode())
        product_links = [anchor['href'] for anchor in parsed.anchors if 'dr-product-link' in anchor.get('class', '')]
        self.assertTrue(product_links)
        for link in product_links:
            self.assertEqual(urlsplit(link).path, reverse('product_details', args=[product.pk]))
            self.assertEqual(parse_qs(urlsplit(link).query)['return_to'], [response.context['report_return']])
        day_links = [anchor['href'] for anchor in parsed.anchors if anchor.get('href', '').startswith('?date=')]
        self.assertEqual(len(day_links), 3)
        for link in day_links:
            self.assertEqual(parse_qs(urlsplit(link).query)['return_to'], [origin])
            self.assertEqual(parse_qs(urlsplit(link).query)['ignore_snacks'], ['1'])
        self.assertTrue(any(field.get('name') == 'return_to' and field.get('value') == origin for field in parsed.inputs))
        pdf_links = [anchor['href'] for anchor in parsed.anchors if urlsplit(anchor.get('href', '')).path == reverse('daily_report_pdf')]
        self.assertEqual(parse_qs(urlsplit(pdf_links[0]).query), {'date': [selected.isoformat()], 'ignore_snacks': ['1']})

        details = self.client.get(product_links[0])
        self.assertEqual(details.status_code, 200)
        self.assertEqual(details.context['return_to'], response.context['report_return'])
        self.assertContains(details, 'Back to Daily Report')
