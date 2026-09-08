import csv
import io
from datetime import date, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from django.contrib.auth.models import User
from django.http import HttpResponse
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils.timezone import now

from .models import (
    Category,
    CheckoutOrder,
    CheckoutOrderItem,
    Order,
    OrderDetail,
    Product,
    StockChange,
    TransactionCorrection,
    TransactionCorrectionLine,
    TransactionCorrectionUndo,
)
from .views import build_order_transaction_context


class RecordingCanvas:
    """Minimal ReportLab stand-in that records text drawn by the PDF view."""

    strings = []

    def __init__(self, buffer, pagesize=None):
        self.buffer = buffer
        type(self).strings = []

    def _record(self, value):
        type(self).strings.append(str(value))

    def drawString(self, _x, _y, value):
        self._record(value)

    def drawRightString(self, _x, _y, value):
        self._record(value)

    def drawCentredString(self, _x, _y, value):
        self._record(value)

    def save(self):
        self.buffer.write(b'%PDF-1.4\n% transaction redesign test\n%%EOF')

    def setStrokeColor(self, *_args):
        pass

    def setLineWidth(self, *_args):
        pass

    def line(self, *_args):
        pass

    def setFont(self, *_args):
        pass

    def setFillColor(self, *_args):
        pass

    def rect(self, *_args, **_kwargs):
        pass

    def roundRect(self, *_args, **_kwargs):
        pass

    def showPage(self):
        pass


@override_settings(AXES_ENABLED=False)
class TransactionsRedesignBackendTests(TestCase):
    def setUp(self):
        self.gina = User.objects.create_user(
            username='gina-redesign', password='pass1234', is_staff=True,
        )
        self.pu = User.objects.create_user(
            username='pu-redesign', password='pass1234',
        )
        self.category = Category.objects.create(name='Transaction redesign')
        self.product = self._product(
            'Aspirin Search Product', 'REDESIGN-POS-001', Decimal('10.00'),
            taxable=True,
        )
        self.pos_order, self.pos_line = self._order_with_line(
            product=self.product,
            user=self.gina,
            quantity=3,
            subtotal=Decimal('30.00'),
            tax=Decimal('3.90'),
            total=Decimal('33.90'),
        )

        giveaway_product = self._product(
            'No-sale Search Product', 'REDESIGN-NS-001', Decimal('5.00'),
            taxable=True,
        )
        self.checkout = CheckoutOrder.objects.create(
            user=self.pu,
            status=CheckoutOrder.STATUS_SUBMITTED,
            subtotal=Decimal('5.00'),
            tax=Decimal('0.65'),
            total_price=Decimal('5.65'),
            submitted_at=now(),
        )
        self.checkout_item = CheckoutOrderItem.objects.create(
            checkout=self.checkout,
            product=giveaway_product,
            product_name=giveaway_product.name,
            product_barcode=giveaway_product.barcode,
            price=giveaway_product.price,
            taxable=True,
            quantity=1,
        )

        self.client = Client()
        self.client.force_login(self.gina)

    def _product(self, name, barcode, price, *, taxable=False):
        return Product.objects.create(
            name=name,
            barcode=barcode,
            price=price,
            price_per_unit=Decimal('3.00'),
            quantity_in_stock=20,
            category=self.category,
            taxable=taxable,
        )

    def _order_with_line(
            self, *, product, user, quantity=1, subtotal=None,
            tax=Decimal('0.00'), total=None, submitted=True,
            is_deleted=False, expiry_at_sale=None):
        subtotal = subtotal if subtotal is not None else product.price * quantity
        total = total if total is not None else subtotal + tax
        order = Order.objects.create(
            user=user,
            submitted=submitted,
            subtotal=subtotal,
            discount_amount=Decimal('0.00'),
            tax=tax,
            tax_rate=Decimal('0.1300'),
            total_price=total,
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
            is_deleted=is_deleted,
        )
        line = OrderDetail.objects.create(
            order=order,
            product=product,
            product_name=product.name,
            product_barcode=product.barcode or '',
            quantity=quantity,
            price=product.price,
            taxable_at_sale=product.taxable,
            cost_per_unit_at_sale=product.price_per_unit,
            expiry_at_sale=expiry_at_sale,
        )
        return order, line

    def _correction(
            self, *, order, line, quantity, amount,
            correction_type=TransactionCorrection.TYPE_RETURN):
        correction = TransactionCorrection.objects.create(
            correction_type=correction_type,
            order=order,
            reason='Transaction redesign regression test',
            note='Keep the immutable original snapshot.',
            adjustment_amount=amount,
            created_by=self.gina,
        )
        TransactionCorrectionLine.objects.create(
            correction=correction,
            order_detail=line,
            product=line.product,
            product_name=line.product_name,
            product_barcode=line.product_barcode,
            quantity=quantity,
            unit_price=line.price,
            disposition=TransactionCorrectionLine.DISPOSITION_RESTOCK,
        )
        return correction

    def _transaction_rows(self, params=None):
        response = self.client.get(reverse('order_view'), params or {})
        self.assertEqual(response.status_code, 200)
        return response, list(response.context['page_obj'].object_list)

    @staticmethod
    def _row(rows, source, transaction_id):
        return next(
            row for row in rows
            if row['source'] == source and row['id'] == transaction_id
        )

    def test_list_rows_separate_original_adjustment_current_and_account(self):
        self._correction(
            order=self.pos_order,
            line=self.pos_line,
            quantity=2,
            amount=Decimal('22.60'),
        )

        _response, rows = self._transaction_rows()
        pos = self._row(rows, 'pos', self.pos_order.pk)
        no_sale = self._row(rows, 'giveaway', self.checkout.pk)

        self.assertEqual(pos['account_name'], self.gina.username)
        self.assertEqual(pos['original_total'], Decimal('33.90'))
        self.assertEqual(pos['adjustment_total'], Decimal('22.60'))
        self.assertEqual(pos['current_total'], Decimal('11.30'))
        self.assertTrue(pos['has_active_corrections'])

        self.assertEqual(no_sale['account_name'], self.pu.username)
        self.assertEqual(no_sale['current_total'], Decimal('0.00'))
        self.assertEqual(no_sale['total'], Decimal('5.65'))

        self.pos_order.refresh_from_db()
        self.assertEqual(self.pos_order.total_price, Decimal('33.90'))

    def test_search_covers_hash_id_product_barcode_and_account(self):
        cases = (
            ({'source': 'pos', 'q': f'#{self.pos_order.pk}'}, 'pos', self.pos_order.pk),
            ({'source': 'pos', 'q': 'Aspirin Search'}, 'pos', self.pos_order.pk),
            ({'source': 'giveaway', 'q': 'REDESIGN-NS-001'}, 'giveaway', self.checkout.pk),
            ({'source': 'giveaway', 'q': self.pu.username}, 'giveaway', self.checkout.pk),
        )
        for params, source, transaction_id in cases:
            with self.subTest(params=params):
                _response, rows = self._transaction_rows(params)
                self.assertEqual(
                    [(row['source'], row['id']) for row in rows],
                    [(source, transaction_id)],
                )

        duplicate_order = Order.objects.create(
            user=self.gina,
            submitted=True,
            subtotal=Decimal('4.00'),
            total_price=Decimal('4.00'),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        for suffix in ('A', 'B'):
            product = self._product(
                f'Duplicate Match {suffix}', f'DUPLICATE-{suffix}',
                Decimal('2.00'),
            )
            OrderDetail.objects.create(
                order=duplicate_order,
                product=product,
                product_name=product.name,
                product_barcode=product.barcode,
                quantity=1,
                price=product.price,
                taxable_at_sale=False,
            )

        _response, duplicate_rows = self._transaction_rows({
            'source': 'pos', 'q': 'Duplicate Match',
        })
        self.assertEqual(
            [row['id'] for row in duplicate_rows], [duplicate_order.pk],
        )

    def test_corrected_filter_excludes_an_undone_void(self):
        self._correction(
            order=self.pos_order,
            line=self.pos_line,
            quantity=1,
            amount=Decimal('11.30'),
        )
        undone_product = self._product(
            'Undone correction product', 'UNDONE-001', Decimal('4.00'),
        )
        undone_order, undone_line = self._order_with_line(
            product=undone_product, user=self.gina,
        )
        undone = self._correction(
            order=undone_order,
            line=undone_line,
            quantity=1,
            amount=Decimal('4.00'),
            correction_type=TransactionCorrection.TYPE_VOID,
        )
        TransactionCorrectionUndo.objects.create(
            correction=undone,
            created_by=self.gina,
        )

        _response, rows = self._transaction_rows({
            'source': 'pos', 'status': 'corrected',
        })

        self.assertEqual([row['id'] for row in rows], [self.pos_order.pk])
        self.assertTrue(rows[0]['has_active_corrections'])

    def test_review_filter_and_row_state_follow_expired_at_sale_notice(self):
        review_product = self._product(
            'Expired-at-sale review', 'REVIEW-001', Decimal('7.00'),
        )
        review_order, _line = self._order_with_line(
            product=review_product,
            user=self.gina,
            expiry_at_sale=date.today() - timedelta(days=1),
        )

        _response, rows = self._transaction_rows({
            'source': 'pos', 'status': 'review',
        })

        self.assertEqual([row['id'] for row in rows], [review_order.pk])
        self.assertTrue(rows[0]['requires_notice'])
        self.assertEqual(rows[0]['state_label'], 'Review')

    def test_export_and_detail_urls_preserve_the_server_filters(self):
        unrelated = self._product(
            'Unrelated export product', 'UNRELATED-001', Decimal('3.00'),
        )
        self._order_with_line(product=unrelated, user=self.gina)
        today = now().date().isoformat()
        params = {
            'q': 'Aspirin Search',
            'source': 'pos',
            'status': 'completed',
            'date_from': today,
            'date_to': today,
            'page': '1',
        }

        response, rows = self._transaction_rows(params)
        self.assertEqual([row['id'] for row in rows], [self.pos_order.pk])

        export_query = parse_qs(response.context['transaction_export_query'])
        self.assertEqual(export_query['q'], ['Aspirin Search'])
        self.assertEqual(export_query['source'], ['pos'])
        self.assertEqual(export_query['status'], ['completed'])
        self.assertNotIn('page', export_query)

        detail_query = parse_qs(urlsplit(rows[0]['detail_url']).query)
        for key, expected in params.items():
            self.assertEqual(detail_query[key], [expected])

        csv_response = self.client.get(
            reverse('export_transactions_csv'), params,
        )
        self.assertEqual(csv_response.status_code, 200)
        csv_rows = list(csv.DictReader(io.StringIO(
            csv_response.content.decode('utf-8'),
        )))
        self.assertEqual(
            {row['Order ID'] for row in csv_rows},
            {str(self.pos_order.pk)},
        )

    def test_detail_context_reports_requested_supplied_missed_returned_and_net(self):
        product = self._product(
            'Partially supplied item', 'FULFILMENT-001', Decimal('10.00'),
            taxable=True,
        )
        order, line = self._order_with_line(
            product=product,
            user=self.gina,
            quantity=2,
            subtotal=Decimal('20.00'),
            tax=Decimal('2.60'),
            total=Decimal('22.60'),
        )
        StockChange.objects.create(
            product=product,
            product_name=product.name,
            product_barcode=product.barcode,
            user=self.gina,
            order_detail=line,
            change_type='checkout',
            quantity=2,
        )
        StockChange.objects.create(
            product=product,
            product_name=product.name,
            product_barcode=product.barcode,
            user=self.gina,
            order_detail=line,
            change_type='checkout_unfulfilled',
            quantity=3,
        )
        self._correction(
            order=order,
            line=line,
            quantity=1,
            amount=Decimal('11.30'),
        )

        context = build_order_transaction_context(order)
        row = context['order_details_with_total'][0]

        self.assertEqual(row['requested_qty'], 5)
        self.assertEqual(row['fulfilled_qty'], 2)
        self.assertEqual(row['unfulfilled_qty'], 3)
        self.assertEqual(row['returned_qty'], 1)
        self.assertEqual(row['net_qty'], 1)
        self.assertEqual(row['current_line_total'], Decimal('11.30'))
        self.assertEqual(context['total_requested_units'], 5)
        self.assertEqual(context['total_units'], 2)
        self.assertEqual(context['total_unfulfilled_units'], 3)
        self.assertEqual(context['total_returned_units'], 1)
        self.assertEqual(context['net_units'], 1)
        self.assertEqual(context['current_subtotal'], Decimal('10.00'))
        self.assertEqual(context['current_tax'], Decimal('1.30'))
        self.assertEqual(context['current_total'], Decimal('11.30'))
        self.assertEqual(context['correction_total'], Decimal('11.30'))

        order.refresh_from_db()
        self.assertEqual(order.total_price, Decimal('22.60'))

    def test_detail_navigation_skips_orders_outside_the_list_search_scope(self):
        older_product = self._product(
            'Navigation Match Older', 'NAV-OLDER', Decimal('2.00'),
        )
        older, _line = self._order_with_line(
            product=older_product, user=self.gina,
        )
        gap_product = self._product(
            'Navigation Gap', 'NAV-GAP', Decimal('2.00'),
        )
        self._order_with_line(product=gap_product, user=self.gina)
        current_product = self._product(
            'Navigation Match Current', 'NAV-CURRENT', Decimal('2.00'),
        )
        current, _line = self._order_with_line(
            product=current_product, user=self.gina,
        )
        newer_product = self._product(
            'Navigation Match Newer', 'NAV-NEWER', Decimal('2.00'),
        )
        newer, _line = self._order_with_line(
            product=newer_product, user=self.gina,
        )

        def rendered(_request, _template, context):
            response = HttpResponse('ok')
            response.render_context = context
            return response

        params = {
            'q': 'Navigation Match',
            'source': 'pos',
            'status': 'completed',
            'page': '2',
        }
        with patch('app.views.render', side_effect=rendered):
            response = self.client.get(
                reverse('order_detail', args=[current.pk]), params,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.render_context['prev_order'], older.pk)
        self.assertEqual(response.render_context['next_order'], newer.pk)

    def test_detail_back_always_returns_to_transactions(self):
        detail_url = reverse('order_detail', args=[self.pos_order.pk])
        transactions_url = reverse('order_view')
        cases = (
            ({}, {}),
            ({}, {'HTTP_REFERER': 'http://testserver' + reverse('dashboard')}),
            ({}, {'HTTP_REFERER': 'http://testserver' + reverse('order_detail', args=[999])}),
            ({'return_to': reverse('inventory_display')}, {}),
        )
        for params, headers in cases:
            with self.subTest(params=params, headers=headers):
                response = self.client.get(detail_url, params, **headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.context['page_return'], {
                    'url': transactions_url,
                    'destination': 'Transactions',
                    'label': 'Back to Transactions',
                    'source': 'explicit',
                })
                self.assertContains(response, f'''
                    <a class="ui-page-return" href="{transactions_url}"
                       aria-label="Back to Transactions" title="Back to Transactions"
                       data-page-return data-page-return-source="explicit">
                      <span class="ui-page-return-arrow" aria-hidden="true">&larr;</span>
                      <span data-page-return-destination>Transactions</span>
                    </a>
                ''', html=True, count=1)

    def test_detail_back_preserves_transaction_filters(self):
        params = {'q': 'Aspirin', 'source': 'pos', 'status': 'completed', 'page': '2'}
        response = self.client.get(
            reverse('order_detail', args=[self.pos_order.pk]), params,
            HTTP_REFERER='http://testserver' + reverse('dashboard'),
        )
        self.assertEqual(response.status_code, 200)
        target = urlsplit(response.context['page_return']['url'])
        self.assertEqual(target.path, reverse('order_view'))
        self.assertEqual(parse_qs(target.query), {key: [value] for key, value in params.items()})
        self.assertContains(response, 'data-page-return-source="explicit"', count=1)

    def test_order_pdf_supports_original_adjusted_and_audit_variants(self):
        self._correction(
            order=self.pos_order,
            line=self.pos_line,
            quantity=2,
            amount=Decimal('22.60'),
        )
        url = reverse('order_pdf', args=[self.pos_order.pk])

        default_response = self.client.get(url)
        self.assertEqual(default_response.status_code, 200)
        self.assertIn(
            f'MPCP-Order-{self.pos_order.pk}.pdf',
            default_response['Content-Disposition'],
        )

        rendered = {}
        titles = {
            'original': 'ORIGINAL RECEIPT',
            'adjusted': 'ADJUSTED STATEMENT',
            'audit': 'AUDIT REPORT',
        }
        for variant in ('original', 'adjusted', 'audit'):
            with self.subTest(variant=variant):
                with patch('app.views.canvas.Canvas', RecordingCanvas):
                    response = self.client.get(url, {'variant': variant})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response['Content-Type'], 'application/pdf')
                self.assertIn(
                    f'MPCP-Order-{self.pos_order.pk}',
                    response['Content-Disposition'],
                )
                rendered[variant] = '\n'.join(RecordingCanvas.strings)
                self.assertIn(titles[variant], rendered[variant])

        self.assertIn('$33.90', rendered['original'])
        self.assertIn('$11.30', rendered['adjusted'])
        self.assertIn('Recorded adjustments', rendered['audit'])
        self.assertIn('-$22.60', rendered['audit'])

        invalid = self.client.get(url, {'variant': 'unknown-document'})
        self.assertEqual(invalid.status_code, 400)

    def test_audit_pdf_requires_admin_or_passkey_access(self):
        self.client.force_login(self.pu)

        response = self.client.get(
            reverse('order_pdf', args=[self.pos_order.pk]),
            {'variant': 'audit'},
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith(reverse('passkey_unlock')))
        self.assertIn('next=', response.url)
