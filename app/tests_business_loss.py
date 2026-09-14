import io
import json
import re
from datetime import date, datetime, timezone as datetime_timezone
from decimal import Decimal
from html import unescape
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import Category, Product, ProductLotMovement, StockChange


PHARMACY_IDENTITY = (
    'MPCP',
    'Meadowvale Professional Center Pharmacy',
    '6855 Meadowvale Town Centre Cir, Mississauga, ON L5N 2Y1',
    '(905) 821-9992',
)


@override_settings(AXES_ENABLED=False, TIME_ZONE='America/Toronto')
class BusinessLossTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='business-loss-staff', password='test-pass',
        )
        self.client.force_login(self.user)
        self.url = reverse('business_loss')
        self.product = Product.objects.create(
            name='Current product name', barcode='LOSS-CURRENT', price='19.99',
            price_per_unit='2.35', quantity_in_stock=12,
            category=Category.objects.create(name='Business loss'),
        )
        self.log = self._log(
            product_name='Recorded product name', quantity=-3,
            timestamp=datetime(2025, 12, 31, 16, 30, tzinfo=datetime_timezone.utc),
        )
        self.movement = ProductLotMovement.objects.create(
            stock_change=self.log, lot_number='LOSS-LOT', expiry_date=date(2024, 8, 31),
            quantity=3, direction=ProductLotMovement.DIRECTION_OUT,
        )

    def _log(self, *, timestamp=None, **overrides):
        values = {
            'product': self.product, 'product_name': 'Recorded product name',
            'product_barcode': 'LOSS-SNAPSHOT', 'quantity': 1,
            'change_type': 'expired', 'user': self.user,
        }
        values.update(overrides)
        log = StockChange.objects.create(**values)
        if timestamp:
            StockChange.objects.filter(pk=log.pk).update(timestamp=timestamp)
        return log

    def _row(self, **overrides):
        row = {
            'product': 'Edited product', 'quantity': '3',
            'cost_per_unit': '5.55', 'total_cost': '16.17', 'year': '2023',
        }
        row.update(overrides)
        return row

    def _post(self, rows=None, **overrides):
        data = {
            'report_date': '2026-09-10',
            'return_to': reverse('expired_log'),
            'rows': rows if isinstance(rows, str) else json.dumps(
                rows if rows is not None else [self._row()]
            ),
        }
        data.update(overrides)
        return self.client.post(self.url, data)

    def test_login_is_required_for_form_and_pdf(self):
        self.client.logout()
        for method in (self.client.get, self.client.post):
            with self.subTest(method=method.__name__):
                response = method(self.url)
                self.assertRedirects(response, reverse('login') + '?next=' + self.url)

    def test_form_uses_recorded_product_removed_units_and_database_cost(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'business_loss.html')
        row = response.context['rows'][0]
        self.assertEqual(row['product'], 'Recorded product name')
        self.assertEqual(row['quantity'], '3')
        self.assertEqual(row['cost_per_unit'], '2.35')
        self.assertEqual(row['total_cost'], '7.05')
        self.assertEqual(row['year'], '2025')
        self.assertEqual(row['added_date'], date(2025, 12, 31))
        self.assertContains(response, '<time datetime="2025-12-31">Dec 31, 2025</time>', html=True)

    @patch('app.business_loss.timezone.localdate', return_value=date(2026, 9, 10))
    def test_report_date_defaults_to_local_today(self, _localdate):
        response = self.client.get(self.url)
        self.assertEqual(response.context['report_date'], '2026-09-10')

    def test_form_includes_every_expired_entry_beyond_log_page(self):
        StockChange.objects.bulk_create([
            StockChange(
                product=self.product, product_name=f'Expired row {index}',
                quantity=1, change_type='expired',
            ) for index in range(51)
        ])
        self._log(product_name='Ordinary checkin', change_type='checkin', quantity=90)
        self._log(product_name='Ordinary sale', change_type='checkout', quantity=60)
        response = self.client.get(self.url, {'page': '2'})
        rows = response.context['rows']
        self.assertEqual(len(rows), 52)
        self.assertEqual(sum(int(row['quantity']) for row in rows), 54)
        names = {row['product'] for row in rows}
        self.assertIn('Recorded product name', names)
        self.assertNotIn('Ordinary checkin', names)
        self.assertNotIn('Ordinary sale', names)

    def test_archived_and_legacy_products_keep_database_cost(self):
        self.product.archived_at = timezone.now()
        self.product.save(update_fields=['archived_at'])
        self.log.product_name = ''
        self.log.save(update_fields=['product_name'])
        row = self.client.get(self.url).context['rows'][0]
        self.assertEqual(row['product'], 'Current product name')
        self.assertEqual(row['cost_per_unit'], '2.35')
        self.assertEqual(row['total_cost'], '7.05')

    def test_deleted_product_keeps_snapshot_and_requires_cost_entry(self):
        self.product.delete()
        row = self.client.get(self.url).context['rows'][0]
        self.assertEqual(row['product'], 'Recorded product name')
        self.assertEqual(row['quantity'], '3')
        self.assertEqual(row['cost_per_unit'], '')
        self.assertEqual(row['total_cost'], '')

    def test_missing_cost_is_blank_but_explicit_zero_is_preserved(self):
        for cost, expected in ((None, ''), (Decimal('0.00'), '0.00')):
            with self.subTest(cost=cost):
                self.product.price_per_unit = cost
                self.product.save(update_fields=['price_per_unit'])
                row = self.client.get(self.url).context['rows'][0]
                self.assertEqual(row['cost_per_unit'], expected)
                self.assertEqual(row['total_cost'], expected)

    def test_year_uses_local_logged_date_not_utc_or_expiry(self):
        StockChange.objects.filter(pk=self.log.pk).update(
            timestamp=datetime(2026, 1, 1, 2, 0, tzinfo=datetime_timezone.utc),
        )
        with timezone.override('America/Toronto'):
            response = self.client.get(self.url, {'from': '2025-12-31', 'to': '2025-12-31'})
            self.assertEqual(response.context['rows'][0]['year'], '2025')
            self.assertEqual(response.context['rows'][0]['added_date'], date(2025, 12, 31))
            excluded = self.client.get(self.url, {'from': '2026-01-01'})
            self.assertEqual(excluded.context['rows'], [])

    def test_search_and_inclusive_date_filters_match_expired_log(self):
        self._log(
            product_name='Other period',
            timestamp=datetime(2026, 1, 2, 15, 0, tzinfo=datetime_timezone.utc),
        )
        for query in ('Recorded product', 'LOSS-SNAPSHOT', 'Current product',
                      'LOSS-CURRENT', 'LOSS-LOT', self.user.username):
            with self.subTest(query=query):
                response = self.client.get(self.url, {
                    'q': query, 'from': '2025-12-31', 'to': '2025-12-31',
                })
                self.assertEqual(len(response.context['rows']), 1)
                self.assertEqual(response.context['rows'][0]['product'], 'Recorded product name')
        response = self.client.get(self.url, {'q': 'Nothing matches'})
        self.assertEqual(response.context['rows'], [])

    def test_multiple_matching_lots_do_not_duplicate_loss(self):
        ProductLotMovement.objects.create(
            stock_change=self.log, lot_number='LOSS-LOT-2', quantity=1,
            direction=ProductLotMovement.DIRECTION_OUT,
        )
        response = self.client.get(self.url, {'q': 'LOSS-LOT'})
        self.assertEqual(len(response.context['rows']), 1)
        self.assertEqual(response.context['rows'][0]['total_cost'], '7.05')

    def test_invalid_filters_show_feedback_without_exporting_all_logs(self):
        for filters in ({'from': '2026-02-31'}, {'to': 'bad-date'},
                        {'from': '2026-01-02', 'to': '2026-01-01'}):
            with self.subTest(filters=filters):
                response = self.client.get(self.url, filters)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.context['filter_error'])
                self.assertEqual(response.context['rows'], [])

    def test_expired_log_button_preserves_filters_and_return_location(self):
        response = self.client.get(reverse('expired_log'), {
            'q': 'Recorded product', 'from': '2025-12-31', 'to': '2025-12-31',
            'return_to': reverse('expired_products') + '?date_filter=custom', 'page': '2',
        })
        self.assertContains(response, 'Generate Business loss')
        link = re.search(
            r'href="([^"]+)"[^>]*>Generate Business loss</a>',
            response.content.decode(),
        )
        self.assertIsNotNone(link)
        parsed = urlsplit(unescape(link.group(1)))
        self.assertEqual(parsed.path, self.url)
        self.assertEqual(parse_qs(parsed.query), {
            'q': ['Recorded product'], 'from': ['2025-12-31'], 'to': ['2025-12-31'],
            'return_to': [response.wsgi_request.get_full_path()],
        })

    @patch('app.business_loss.build_business_loss_pdf', return_value=b'%PDF-1.4\n%%EOF')
    def test_every_edited_field_including_manual_total_reaches_pdf(self, build_pdf):
        response = self._post([
            self._row(added_date='2025-12-31'),
            self._row(product='Second edited product', quantity='2',
                      cost_per_unit='3.00', total_cost='6.00', year='2024'),
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        rows, report_date = build_pdf.call_args.args
        self.assertEqual(report_date, date(2026, 9, 10))
        self.assertEqual(rows[0], {
            'product': 'Edited product', 'quantity': 3,
            'cost_per_unit': Decimal('5.55'), 'total_cost': Decimal('16.17'), 'year': 2023,
        })
        self.assertEqual(rows[1]['product'], 'Second edited product')
        self.assertEqual(rows[1]['year'], 2024)
        self.assertEqual(sum(row['quantity'] for row in rows), 5)
        self.assertEqual(sum(row['total_cost'] for row in rows), Decimal('22.17'))

    @patch('app.business_loss.build_business_loss_pdf', return_value=b'%PDF-1.4\n%%EOF')
    def test_money_accepts_ordinary_leading_and_trailing_decimal_input(self, build_pdf):
        for unit_cost, total_cost in (('.50', '1.'), ('1.', '.50')):
            with self.subTest(unit_cost=unit_cost, total_cost=total_cost):
                response = self._post([
                    self._row(cost_per_unit=unit_cost, total_cost=total_cost),
                ])
                self.assertEqual(response.status_code, 200)
                rows, _report_date = build_pdf.call_args.args
                self.assertEqual(rows[0]['cost_per_unit'], Decimal(unit_cost))
                self.assertEqual(rows[0]['total_cost'], Decimal(total_cost))

    def test_multiline_product_name_cannot_create_a_taller_than_page_pdf_row(self):
        product = 'First line' + ('\n' * 175) + 'Last line'
        response = self._post([self._row(product=product)])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b'%PDF'))
        try:
            import pdfplumber
        except ImportError:
            return
        with pdfplumber.open(io.BytesIO(response.content)) as document:
            self.assertEqual(len(document.pages), 1)
            text = document.pages[0].extract_text() or ''
        self.assertIn('First line Last line', ' '.join(text.split()))

    def test_real_pdf_contains_header_edited_rows_date_and_exact_totals(self):
        response = self._post([
            self._row(),
            self._row(product='Second edited product', quantity='2',
                      cost_per_unit='3.00', total_cost='6.00', year='2024'),
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertTrue(response.content.startswith(b'%PDF'))
        self.assertIn('.pdf', response['Content-Disposition'])
        try:
            import pdfplumber
        except ImportError:
            return  # PDF parsing is optional; ReportLab generation is still checked.
        with pdfplumber.open(io.BytesIO(response.content)) as document:
            text = '\n'.join(page.extract_text() or '' for page in document.pages)
        for expected in (*PHARMACY_IDENTITY, 'Business Loss', 'Edited product',
                         'Second edited product', '2023', '2024', '5.55', '16.17',
                         '22.17', '2026'):
            self.assertIn(expected, text)
        self.assertIn('NUMBER OF UNITS', text)
        self.assertRegex(text, r'Total units:\s+5\s+\$22\.17')
        self.assertNotIn('16.65', text)

    def test_pharmacy_header_repeats_above_content_on_every_pdf_page(self):
        response = self._post([
            self._row(product=f'Expired product {index:03d}') for index in range(70)
        ])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b'%PDF'))
        try:
            import pdfplumber
        except ImportError:
            return
        with pdfplumber.open(io.BytesIO(response.content)) as document:
            self.assertGreater(len(document.pages), 1)
            for page_number, page in enumerate(document.pages, start=1):
                with self.subTest(page=page_number):
                    header_boxes = []
                    for identity_line in PHARMACY_IDENTITY:
                        matches = page.search(identity_line, regex=False, case=True)
                        self.assertTrue(matches, f'Missing header text: {identity_line}')
                        header_boxes.append(min(matches, key=lambda box: box['top']))
                    content_boxes = [
                        box
                        for label in ('Business Loss', 'Product', 'Qty')
                        for box in page.search(label, regex=False, case=True)
                    ]
                    self.assertTrue(content_boxes, 'Report content is missing from the page')
                    self.assertLess(
                        max(box['bottom'] for box in header_boxes),
                        min(box['top'] for box in content_boxes),
                        'The pharmacy header overlaps the report title or table',
                    )

    @patch('app.business_loss.build_business_loss_pdf')
    def test_invalid_numbers_keep_other_edits_and_never_generate_pdf(self, build_pdf):
        invalid = {
            'quantity': ('', '-1', '1.5', 'NaN', '2147483648'),
            'cost_per_unit': ('', '-0.01', 'NaN', 'Infinity', '0.001', '100000000.00'),
            'total_cost': ('', '-0.01', 'NaN', 'Infinity', '2.345', '1000000000000000000.00'),
            'year': ('', '-1', '0', '10000', '2025.5', 'invalid'),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    row = self._row(**{field: value})
                    response = self._post([row])
                    self.assertEqual(response.status_code, 400)
                    self.assertTemplateUsed(response, 'business_loss.html')
                    actual = response.context['rows'][0]
                    for key, entered in row.items():
                        self.assertEqual(actual[key], entered)
                    self.assertTrue(actual['errors'])
        build_pdf.assert_not_called()

    @patch('app.business_loss.build_business_loss_pdf')
    def test_blank_product_and_invalid_report_date_retain_edits(self, build_pdf):
        response = self._post([self._row(product='', added_date='2025-12-31')])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.context['rows'][0]['product'], '')
        self.assertEqual(response.context['rows'][0]['added_date'], date(2025, 12, 31))
        self.assertContains(response, '<time datetime="2025-12-31">Dec 31, 2025</time>', html=True, status_code=400)
        self.assertTrue(response.context['rows'][0]['errors'])
        for report_date in ('', '2026-02-31', 'not-a-date'):
            with self.subTest(report_date=report_date):
                response = self._post(report_date=report_date)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.context['report_date'], report_date)
                self.assertEqual(response.context['rows'][0]['product'], 'Edited product')
        build_pdf.assert_not_called()

    @patch('app.business_loss.build_business_loss_pdf')
    def test_malformed_or_empty_rows_cannot_generate_pdf(self, build_pdf):
        for raw_rows in ('', '[', '{}', 'null', '[]', '[null]', '["text"]'):
            with self.subTest(raw_rows=raw_rows):
                response = self._post(rows=raw_rows)
                self.assertEqual(response.status_code, 400)
                self.assertTemplateUsed(response, 'business_loss.html')
        missing_rows = self.client.post(self.url, {'report_date': '2026-09-10'})
        self.assertEqual(missing_rows.status_code, 400)
        build_pdf.assert_not_called()

    def test_saved_and_edited_product_text_is_escaped_in_form(self):
        dangerous = '<script>alert("loss")</script>'
        self.log.product_name = dangerous
        self.log.save(update_fields=['product_name'])
        response = self.client.get(self.url)
        self.assertNotContains(response, dangerous)
        self.assertContains(response, '&lt;script&gt;')
        invalid_response = self._post([self._row(product=dangerous, quantity='bad')])
        self.assertEqual(invalid_response.status_code, 400)
        self.assertNotContains(invalid_response, dangerous, status_code=400)
        self.assertContains(invalid_response, '&lt;script&gt;', status_code=400)

    @patch('app.business_loss.build_business_loss_pdf', return_value=b'%PDF-1.4\n%%EOF')
    def test_pdf_post_enforces_csrf(self, build_pdf):
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        data = {'report_date': '2026-09-10', 'rows': json.dumps([self._row()])}
        self.assertEqual(client.post(self.url, data).status_code, 403)
        build_pdf.assert_not_called()
        client.get(self.url)
        token = client.cookies['csrftoken'].value
        response = client.post(self.url, {**data, 'csrfmiddlewaretoken': token})
        self.assertEqual(response.status_code, 200)
        build_pdf.assert_called_once()

    def test_form_and_pdf_leave_inventory_and_expired_audit_unchanged(self):
        products_before = list(Product.all_objects.order_by('pk').values())
        logs_before = list(StockChange.objects.order_by('pk').values())
        movements_before = list(ProductLotMovement.objects.order_by('pk').values())
        self.assertEqual(self.client.get(self.url).status_code, 200)
        self.assertEqual(self._post().status_code, 200)
        self.assertEqual(list(Product.all_objects.order_by('pk').values()), products_before)
        self.assertEqual(list(StockChange.objects.order_by('pk').values()), logs_before)
        self.assertEqual(list(ProductLotMovement.objects.order_by('pk').values()), movements_before)
