import csv
import io
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
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
    TransactionCorrection,
    TransactionCorrectionLine,
    TransactionCorrectionUndo,
)


class RecordingCanvas:
    """Small ReportLab stand-in that records text drawn by the PDF view."""

    strings = []

    def __init__(self, buffer, pagesize=None):
        self.buffer = buffer
        type(self).strings = []

    def _record(self, text):
        type(self).strings.append(str(text))

    def drawString(self, _x, _y, text):
        self._record(text)

    def drawRightString(self, _x, _y, text):
        self._record(text)

    def drawCentredString(self, _x, _y, text):
        self._record(text)

    def save(self):
        self.buffer.write(b'%PDF-1.4\n% transaction export test\n%%EOF')

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
class TransactionExportSourceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='transaction-export-user', password='pass1234',
        )
        self.client = Client()
        self.client.force_login(self.user)
        category = Category.objects.create(name='Export category')
        pos_product = Product.objects.create(
            name='POS export product', barcode='900000000001',
            price=Decimal('8.00'), quantity_in_stock=10, category=category,
        )
        giveaway_product = Product.objects.create(
            name='PU no-sale export product', barcode='900000000002',
            price=Decimal('5.00'), quantity_in_stock=10, category=category,
        )

        self.pos_order = Order.objects.create(
            user=self.user,
            submitted=True,
            subtotal=Decimal('24.00'),
            discount_amount=Decimal('0.00'),
            tax=Decimal('3.12'),
            total_price=Decimal('27.12'),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        self.pos_line = OrderDetail.objects.create(
            order=self.pos_order,
            product=pos_product,
            product_name=pos_product.name,
            product_barcode=pos_product.barcode,
            quantity=3,
            price=Decimal('8.00'),
            taxable_at_sale=True,
        )

        self.checkout = CheckoutOrder.objects.create(
            user=self.user,
            status=CheckoutOrder.STATUS_SUBMITTED,
            subtotal=Decimal('5.00'),
            tax=Decimal('0.65'),
            total_price=Decimal('5.65'),
            submitted_at=now(),
        )
        CheckoutOrderItem.objects.create(
            checkout=self.checkout,
            product=giveaway_product,
            product_name=giveaway_product.name,
            product_barcode=giveaway_product.barcode,
            price=Decimal('5.00'),
            taxable=True,
            quantity=1,
        )

    def _record_pos_correction(self, quantity, correction_type):
        correction = TransactionCorrection.objects.create(
            correction_type=correction_type,
            order=self.pos_order,
            reason='Transaction export regression test',
            adjustment_amount=Decimal('9.04') * quantity,
            created_by=self.user,
        )
        TransactionCorrectionLine.objects.create(
            correction=correction,
            order_detail=self.pos_line,
            product=self.pos_line.product,
            product_name=self.pos_line.product_name,
            product_barcode=self.pos_line.product_barcode,
            quantity=quantity,
            unit_price=self.pos_line.price,
            disposition=TransactionCorrectionLine.DISPOSITION_RESTOCK,
        )
        return correction

    def _pos_csv_row(self):
        response = self.client.get(
            reverse('export_transactions_csv'), {'source': 'pos'},
        )
        self.assertEqual(response.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(response.content.decode('utf-8'))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['Source'], 'POS')
        return rows[0]

    def test_giveaway_csv_includes_pu_session_and_excludes_pos_order(self):
        response = self.client.get(
            reverse('export_transactions_csv'), {'source': 'giveaway'},
        )

        self.assertEqual(response.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(response.content.decode('utf-8'))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['Source'], 'PU No-Sale')
        self.assertEqual(rows[0]['Product Name at Sale'], 'PU no-sale export product')
        self.assertNotIn('POS export product', response.content.decode('utf-8'))

    def test_giveaway_pdf_includes_pu_session_and_excludes_pos_order(self):
        with patch('app.views.canvas.Canvas', RecordingCanvas):
            response = self.client.get(
                reverse('export_transactions_pdf'), {'source': 'giveaway'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        rendered_text = '\n'.join(RecordingCanvas.strings)
        self.assertIn('PU no-sale export product', rendered_text)
        self.assertIn(f'No-sale #{self.checkout.pk}', rendered_text)
        self.assertNotIn('POS export product', rendered_text)

    def test_giveaway_pdf_real_renderer_returns_pdf(self):
        response = self.client.get(
            reverse('export_transactions_pdf'), {'source': 'giveaway'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertTrue(response.content.startswith(b'%PDF'))

    def test_full_void_exports_zero_realized_pos_values(self):
        self._record_pos_correction(3, TransactionCorrection.TYPE_VOID)

        row = self._pos_csv_row()

        self.assertEqual(row['Quantity'], '0')
        self.assertEqual(row['Line Total'], '0.00')
        self.assertEqual(row['Order Subtotal'], '0.00')
        self.assertEqual(row['Order Tax'], '0.00')
        self.assertEqual(row['Order Total'], '0.00')
        self.pos_order.refresh_from_db()
        self.assertEqual(self.pos_order.total_price, Decimal('27.12'))

    def test_partial_return_exports_remaining_pos_values(self):
        self._record_pos_correction(2, TransactionCorrection.TYPE_RETURN)

        row = self._pos_csv_row()

        self.assertEqual(row['Quantity'], '1')
        self.assertEqual(row['Line Total'], '8.00')
        self.assertEqual(row['Order Subtotal'], '8.00')
        self.assertEqual(row['Order Tax'], '1.04')
        self.assertEqual(row['Order Total'], '9.04')

    def test_void_undo_restores_export_values_without_rewriting_snapshot(self):
        correction = self._record_pos_correction(
            3, TransactionCorrection.TYPE_VOID,
        )
        TransactionCorrectionUndo.objects.create(
            correction=correction,
            created_by=self.user,
        )

        row = self._pos_csv_row()

        self.assertEqual(row['Quantity'], '3')
        self.assertEqual(row['Line Total'], '24.00')
        self.assertEqual(row['Order Subtotal'], '24.00')
        self.assertEqual(row['Order Tax'], '3.12')
        self.assertEqual(row['Order Total'], '27.12')
        self.pos_order.refresh_from_db()
        self.assertEqual(self.pos_order.total_price, Decimal('27.12'))

    def test_partial_return_pdf_uses_realized_cover_and_order_total(self):
        self._record_pos_correction(2, TransactionCorrection.TYPE_RETURN)

        with patch('app.views.canvas.Canvas', RecordingCanvas):
            response = self.client.get(
                reverse('export_transactions_pdf'), {'source': 'pos'},
            )

        self.assertEqual(response.status_code, 200)
        rendered_text = '\n'.join(RecordingCanvas.strings)
        self.assertIn('$9.04', rendered_text)
        self.assertNotIn('$27.12', rendered_text)

    def test_export_links_preserve_combined_list_filters(self):
        today = now().date().isoformat()
        response = self.client.get(reverse('order_view'), {
            'date_from': today,
            'date_to': today,
            'status': 'completed',
            'source': 'giveaway',
        })

        query = response.context['transaction_export_query']
        escaped_query = query.replace('&', '&amp;')
        content = response.content.decode('utf-8')
        self.assertIn(f"{reverse('export_transactions_csv')}?{escaped_query}", content)
        self.assertIn(f"{reverse('export_transactions_pdf')}?{escaped_query}", content)
