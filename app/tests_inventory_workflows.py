import time
import warnings
from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from . import reporting
from .inventory_services import remove_stock_from_lots
from .mixins import PASSKEY_SESSION_KEY
from .models import (
    Category,
    CheckinSession,
    CheckinReceivingDraft,
    CheckoutOrder,
    CheckoutOrderItem,
    Item,
    Order,
    OrderDetail,
    OrderingSheetEntry,
    OrderingSheetStatusEvent,
    Product,
    ProductLot,
    ProductLotMovement,
    StockChange,
    SupplierOrderPlan,
    SupplierOrderPlanItem,
    SupplierPurchaseOrder,
    TransactionCorrection,
    TransactionCorrectionLine,
    TransactionCorrectionUndo,
    UserAction,
)
from .views import record_stock_change


@override_settings(AXES_ENABLED=False)
class MultiLotInventoryTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='lot-admin', password='pass1234', is_staff=True,
        )
        self.category = Category.objects.create(name='Lot Test')
        self.product = Product.objects.create(
            name='Multi Lot Product', barcode='LOT1001', price=Decimal('10.00'),
            quantity_in_stock=5, category=self.category,
        )
        self.early = ProductLot.objects.create(
            product=self.product, lot_number='EARLY',
            expiry_date=date.today() + timedelta(days=30), quantity_on_hand=2,
        )
        self.late = ProductLot.objects.create(
            product=self.product, lot_number='LATE',
            expiry_date=date.today() + timedelta(days=90), quantity_on_hand=3,
        )

    def test_stock_removal_uses_fefo_and_records_exact_lot_movements(self):
        self.product.quantity_in_stock = 1
        self.product.save(update_fields=['quantity_in_stock'])
        change = record_stock_change(
            self.product, 4, 'checkout', user=self.user, note='FEFO test',
        )

        remove_stock_from_lots(self.product, 4, change)

        self.early.refresh_from_db()
        self.late.refresh_from_db()
        self.assertEqual(self.early.quantity_on_hand, 0)
        self.assertEqual(self.late.quantity_on_hand, 1)
        self.assertEqual(
            list(change.lot_movements.values_list('lot_number', 'quantity')),
            [('EARLY', 2), ('LATE', 2)],
        )
        self.product.refresh_from_db()
        self.assertEqual(self.product.expiry_date, self.late.expiry_date)
        self.assertEqual(
            list(self.product.expiry_dates.values_list('expiry_date', flat=True)),
            [self.late.expiry_date],
        )

    def test_purchase_submission_uses_nearest_valid_expiry_before_other_lots(self):
        expired = ProductLot.objects.create(
            product=self.product, lot_number='EXPIRED',
            expiry_date=date.today() - timedelta(days=1), quantity_on_hand=2,
        )
        undated = ProductLot.objects.create(
            product=self.product, lot_number='UNDATED',
            expiry_date=None, quantity_on_hand=1,
        )
        self.product.quantity_in_stock = 8
        self.product.save(update_fields=['quantity_in_stock'])
        order = Order.objects.create(
            user=self.user,
            draft_cart={
                str(self.product.pk): {
                    'name': self.product.name,
                    'price': str(self.product.price),
                    'quantity': 3,
                },
            },
        )
        client = Client()
        client.force_login(self.user)
        session = client.session
        session['order_id'] = order.pk
        session.save()

        response = client.post(reverse('submit_order'))

        self.assertEqual(response.status_code, 302)
        self.early.refresh_from_db()
        self.late.refresh_from_db()
        expired.refresh_from_db()
        undated.refresh_from_db()
        self.assertEqual(self.early.quantity_on_hand, 0)
        self.assertEqual(self.late.quantity_on_hand, 2)
        self.assertEqual(expired.quantity_on_hand, 2)
        self.assertEqual(undated.quantity_on_hand, 1)
        movement_lots = list(
            ProductLotMovement.objects.filter(
                stock_change__order_detail__order=order,
                direction=ProductLotMovement.DIRECTION_OUT,
            ).values_list('lot_number', 'quantity')
        )
        self.assertEqual(movement_lots, [('EARLY', 2), ('LATE', 1)])

    def test_empty_expired_lot_does_not_flag_or_retire_future_stock(self):
        expired_date = date.today() - timedelta(days=5)
        future_date = date.today() + timedelta(days=60)
        self.early.expiry_date = expired_date
        self.early.quantity_on_hand = 0
        self.early.save(update_fields=['expiry_date', 'quantity_on_hand'])
        self.late.expiry_date = future_date
        self.late.quantity_on_hand = 5
        self.late.save(update_fields=['expiry_date', 'quantity_on_hand'])
        self.product.expiry_date = expired_date
        self.product.save(update_fields=['expiry_date'])

        client = Client()
        client.force_login(self.user)
        expired_list = client.get(reverse('expired_products'))
        self.assertNotIn(
            self.product.pk,
            [product.pk for product in expired_list.context['products']],
        )

        detail = client.get(
            reverse('expired_products'),
            {'mode': 'log', 'pid': self.product.pk},
        )
        self.assertEqual(detail.context['product_extra']['status'], 'ok')
        self.assertEqual(detail.context['product_extra']['expired_quantity'], 0)
        self.assertEqual(detail.context['product_extra']['value'], Decimal('0.00'))

        response = client.post(reverse('expired_products'), {
            'mode': 'log',
            'barcode': self.product.barcode,
            'retire_expired': '1',
            'retire_quantity': '1',
        }, follow=True)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertContains(
            response,
            'No quantity-bearing lot is expired or within one month of its expiry date.',
        )

    def test_value_at_risk_sums_expired_and_soon_lot_quantities(self):
        self.early.expiry_date = date.today() - timedelta(days=5)
        self.early.save(update_fields=['expiry_date'])
        self.late.expiry_date = date.today() + timedelta(days=20)
        self.late.save(update_fields=['expiry_date'])

        client = Client()
        client.force_login(self.user)
        detail = client.get(
            reverse('expired_products'),
            {'mode': 'log', 'pid': self.product.pk},
        )

        summary = detail.context['product_extra']
        self.assertEqual(summary['expired_quantity'], 2)
        self.assertEqual(summary['at_risk_quantity'], 5)
        self.assertEqual(summary['value'], Decimal('50.00'))

    def test_checkin_named_lot_adds_stock_and_shows_named_toast(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('add_quantity', args=[session.pk, self.product.pk]),
            {
                'amount': '3',
                'lot_number': 'new-77',
                'lot_expiry': '31-12-2030',
            },
            follow=True,
        )

        self.product.refresh_from_db()
        lot = ProductLot.objects.get(
            product=self.product, lot_number='NEW-77', expiry_date='2030-12-31',
        )
        self.assertEqual(self.product.quantity_in_stock, 8)
        self.assertEqual(lot.quantity_on_hand, 3)
        self.assertContains(response, 'lot NEW-77')

    def test_checkin_inline_edit_derives_stock_and_expiry_from_all_lot_rows(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('checkin_edit_product', args=[session.pk, self.product.pk]),
            {
                'name': self.product.name,
                'brand': '',
                'item_number': '',
                'price': str(self.product.price),
                'barcode': self.product.barcode,
                'quantity_in_stock': '999',
                'category': str(self.category.pk),
                'unit_size': '',
                'description': '',
                'expiry_date': '31-12-2099',
                'extra_expiry_dates': ['30-11-2099'],
                'price_per_unit': '',
                'status': 'on',
                'lot_number': ['EARLY', 'LATE'],
                'lot_expiry': [
                    self.early.expiry_date.strftime('%d-%m-%Y'),
                    self.late.expiry_date.strftime('%d-%m-%Y'),
                ],
                'lot_quantity': ['6', '1'],
            },
        )

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.early.refresh_from_db()
        self.late.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 7)
        self.assertEqual(self.early.quantity_on_hand, 6)
        self.assertEqual(self.late.quantity_on_hand, 1)
        self.assertEqual(self.product.expiry_date, self.early.expiry_date)
        self.assertEqual(
            list(self.product.expiry_dates.values_list('expiry_date', flat=True)),
            [self.early.expiry_date, self.late.expiry_date],
        )
        change = StockChange.objects.get(
            product=self.product,
            session=session,
            change_type='error_add',
        )
        self.assertEqual(change.quantity, 2)
        self.assertIn('Lot-derived inline edit', change.note)

    def test_checkin_inline_lot_rename_retargets_the_receiving_draft(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        draft = CheckinReceivingDraft.objects.create(
            session=session,
            product=self.product,
            existing_lot=self.early,
            lot_number=self.early.lot_number,
            lot_expiry=self.early.expiry_date,
            revision=1,
        )
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('checkin_edit_product', args=[session.pk, self.product.pk]),
            {
                'name': self.product.name,
                'brand': '',
                'item_number': '',
                'price': str(self.product.price),
                'barcode': self.product.barcode,
                'quantity_in_stock': str(self.product.quantity_in_stock),
                'inline_stock_baseline': str(self.product.quantity_in_stock),
                'category': str(self.category.pk),
                'unit_size': '',
                'description': '',
                'expiry_date': '',
                'price_per_unit': '',
                'status': 'on',
                'lot_number': ['RENAMED-EARLY', self.late.lot_number],
                'lot_expiry': [
                    self.early.expiry_date.strftime('%d-%m-%Y'),
                    self.late.expiry_date.strftime('%d-%m-%Y'),
                ],
                'lot_quantity': [
                    str(self.early.quantity_on_hand),
                    str(self.late.quantity_on_hand),
                ],
            },
        )

        self.assertEqual(response.status_code, 302)
        draft.refresh_from_db()
        self.assertIsNotNone(draft.existing_lot)
        self.assertEqual(draft.existing_lot.lot_number, 'RENAMED-EARLY')
        self.assertEqual(draft.lot_number, 'RENAMED-EARLY')
        self.assertEqual(draft.revision, 2)
        self.early.refresh_from_db()
        self.assertIsNotNone(self.early.archived_at)

    def test_checkin_inline_edit_clearing_all_lots_sets_stock_to_zero(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('checkin_edit_product', args=[session.pk, self.product.pk]),
            {
                'name': self.product.name,
                'brand': '',
                'item_number': '',
                'price': str(self.product.price),
                'barcode': self.product.barcode,
                'quantity_in_stock': '999',
                'category': str(self.category.pk),
                'unit_size': '',
                'description': '',
                'expiry_date': '31-12-2099',
                'price_per_unit': '',
                'status': 'on',
                'lot_number': [''],
                'lot_expiry': [''],
                'lot_quantity': [''],
            },
        )

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 0)
        self.assertIsNone(self.product.expiry_date)
        self.assertFalse(
            self.product.lots.filter(archived_at__isnull=True).exists()
        )
        change = StockChange.objects.get(
            product=self.product,
            session=session,
            change_type='error_subtract',
        )
        self.assertEqual(change.quantity, 5)

    def test_checkin_inline_edit_rejects_stale_lots_after_direct_stock_change(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        # A direct stock adjustment happened after inline Edit was opened. Its
        # lot allocation is already authoritative in the database, while the
        # inline form still contains the old 2 + 3 lot quantities.
        self.product.quantity_in_stock = 8
        self.product.save(update_fields=['quantity_in_stock'])
        self.early.quantity_on_hand = 5
        self.early.save(update_fields=['quantity_on_hand'])

        payload = {
            'name': self.product.name,
            'brand': '',
            'item_number': '',
            'price': str(self.product.price),
            'barcode': self.product.barcode,
            'quantity_in_stock': '5',
            'inline_stock_baseline': '5',
            'category': str(self.category.pk),
            'unit_size': '',
            'description': '',
            'expiry_date': '',
            'price_per_unit': '',
            'status': 'on',
            'lot_number': ['EARLY', 'LATE'],
            'lot_expiry': [
                self.early.expiry_date.strftime('%d-%m-%Y'),
                self.late.expiry_date.strftime('%d-%m-%Y'),
            ],
            'lot_quantity': ['2', '3'],
        }

        response = client.post(
            reverse('checkin_edit_product', args=[session.pk, self.product.pk]),
            payload,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Stock changed to 8 while you were editing')
        self.assertContains(response, 'the lot quantities total 5')
        self.assertContains(response, 'data-lot-sync-error')
        self.product.refresh_from_db()
        self.early.refresh_from_db()
        self.late.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 8)
        self.assertEqual(self.early.quantity_on_hand, 5)
        self.assertEqual(self.late.quantity_on_hand, 3)

        # Once the user accounts for the direct adjustment in the lot rows,
        # the same stale baseline is accepted without changing stock again.
        payload['lot_quantity'] = ['5', '3']
        response = client.post(
            reverse('checkin_edit_product', args=[session.pk, self.product.pk]),
            payload,
        )
        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 8)

    def test_invalid_lot_expiry_does_not_change_stock(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('add_quantity', args=[session.pk, self.product.pk]),
            {'amount': '3', 'lot_number': 'BAD', 'lot_expiry': 'not-a-date'},
            follow=True,
        )

        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertFalse(ProductLot.objects.filter(lot_number='BAD').exists())
        self.assertContains(response, 'Enter the lot expiry as DD-MM-YYYY')

    def test_checkin_can_reuse_saved_lot_without_retyping_details(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('add_quantity', args=[session.pk, self.product.pk]),
            {
                'amount': '3',
                'existing_lot_id': str(self.early.pk),
                # Saved-lot selection is authoritative even if stale browser
                # fields are also posted.
                'lot_number': 'WRONG-LOT',
                'lot_expiry': '31-12-2099',
            },
            follow=True,
        )

        self.product.refresh_from_db()
        self.early.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 8)
        self.assertEqual(self.early.quantity_on_hand, 5)
        self.assertFalse(
            ProductLot.objects.filter(product=self.product, lot_number='WRONG-LOT').exists()
        )
        self.assertContains(response, 'saved lot EARLY')
        self.assertContains(response, f'value="{self.early.pk}"')

    def test_checkin_rejects_saved_lot_from_another_product(self):
        other_product = Product.objects.create(
            name='Other Product', barcode='LOT2002', price=Decimal('5.00'),
            quantity_in_stock=0, category=self.category,
        )
        other_lot = ProductLot.objects.create(
            product=other_product, lot_number='OTHER-LOT',
            expiry_date=date.today() + timedelta(days=120), quantity_on_hand=0,
        )
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('add_quantity', args=[session.pk, self.product.pk]),
            {'amount': '2', 'existing_lot_id': str(other_lot.pk)},
            follow=True,
        )

        self.product.refresh_from_db()
        other_lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertEqual(other_lot.quantity_on_hand, 0)
        self.assertContains(response, 'saved lot is no longer available for this product')

    def test_repeated_barcode_scan_uses_selected_saved_lot(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('checkin_session', args=[session.pk]),
            {
                'barcode': self.product.barcode,
                'current_barcode': self.product.barcode,
                'existing_lot_id': str(self.late.pk),
            },
            follow=True,
        )

        self.product.refresh_from_db()
        self.late.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 6)
        self.assertEqual(self.late.quantity_on_hand, 4)
        self.assertContains(response, 'tracked in saved lot LATE')

    def test_exact_quantity_increase_uses_selected_saved_lot(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse('set_quantity', args=[session.pk, self.product.pk]),
            {'quantity': '10', 'existing_lot_id': str(self.late.pk)},
            follow=True,
        )

        self.product.refresh_from_db()
        self.late.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)
        self.assertEqual(self.late.quantity_on_hand, 8)
        self.assertContains(response, 'stock set to 10 (+5)')

    def test_checkin_page_lists_saved_lot_and_expiry_pairs(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by='AB')
        client = Client()
        client.force_login(self.user)

        response = client.get(
            reverse('checkin_session', args=[session.pk]),
            {'product_id': self.product.pk, 'receiving_lot_id': self.early.pk},
        )

        self.assertContains(response, 'id="receivingLotSelect"')
        self.assertContains(response, 'Receive into')
        self.assertContains(response, self.early.expiry_date.strftime('%d-%m-%Y'))
        self.assertEqual(response.context['selected_receiving_lot_id'], self.early.pk)


@override_settings(AXES_ENABLED=False)
class TransactionCorrectionWorkflowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='correction-admin', password='pass1234', is_staff=True,
        )
        category = Category.objects.create(name='Corrections')
        self.product = Product.objects.create(
            name='Returnable Product', barcode='RET1001', price=Decimal('10.00'),
            quantity_in_stock=10, category=category, taxable=True,
        )
        self.lot = ProductLot.objects.create(
            product=self.product, lot_number='SOURCE-LOT',
            expiry_date=date.today() + timedelta(days=60), quantity_on_hand=10,
        )
        self.order = Order.objects.create(
            user=self.user, submitted=True, subtotal=Decimal('30.00'),
            tax=Decimal('3.90'), total_price=Decimal('33.90'),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        self.line = OrderDetail.objects.create(
            order=self.order, product=self.product,
            product_name=self.product.name, product_barcode=self.product.barcode,
            quantity=3, price=Decimal('10.00'), taxable_at_sale=True,
        )
        self.product.quantity_in_stock = 7
        self.product.save(update_fields=['quantity_in_stock'])
        sold = record_stock_change(
            self.product, 3, 'checkout', user=self.user,
            order_detail=self.line, note='Original sale',
        )
        remove_stock_from_lots(self.product, 3, sold)
        self.client = Client()
        self.client.force_login(self.user)

    def _record_void(self, disposition='restock'):
        response = self.client.post(
            reverse('order_correction', args=[self.order.pk]),
            {
                'correction_type': 'void',
                'reason': 'Entered by mistake',
                f'disposition_{self.line.pk}': disposition,
            },
        )
        self.assertRedirects(
            response, reverse('order_detail', args=[self.order.pk]),
        )
        return TransactionCorrection.objects.get(correction_type='void')

    def test_partial_return_restocks_original_lot_and_preserves_sale(self):
        response = self.client.post(
            reverse('order_correction', args=[self.order.pk]),
            {
                'correction_type': 'return',
                'reason': 'Customer return',
                f'qty_{self.line.pk}': '2',
                f'disposition_{self.line.pk}': 'restock',
            },
        )

        self.assertRedirects(
            response, reverse('order_detail', args=[self.order.pk]),
        )
        correction = TransactionCorrection.objects.get()
        correction_line = correction.lines.get()
        self.assertEqual(correction.adjustment_amount, Decimal('22.60'))
        self.assertEqual(correction_line.quantity, 2)
        self.product.refresh_from_db()
        self.lot.refresh_from_db()
        self.line.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 9)
        self.assertEqual(self.product.stock_sold, 1)
        self.assertEqual(self.lot.quantity_on_hand, 9)
        self.assertEqual(self.line.quantity, 3)
        movement = ProductLotMovement.objects.get(
            stock_change__correction_line=correction_line,
        )
        self.assertEqual((movement.lot_number, movement.quantity, movement.direction),
                         ('SOURCE-LOT', 2, ProductLotMovement.DIRECTION_IN))
        correction_report = reporting.stock_corrections(date.today())
        self.assertEqual(correction_report['correction_count'], 1)
        self.assertEqual(correction_report['corrections'][0]['action'],
                         'Transaction Return — Restocked')

        # Only one supplied unit remains correctable.
        self.client.post(
            reverse('order_correction', args=[self.order.pk]),
            {
                'correction_type': 'return', 'reason': 'Too many',
                f'qty_{self.line.pk}': '2',
                f'disposition_{self.line.pk}': 'restock',
            },
        )
        self.assertEqual(TransactionCorrection.objects.count(), 1)

    def test_unfulfilled_units_are_never_returned_to_inventory(self):
        second = OrderDetail.objects.create(
            order=self.order, product=self.product,
            product_name=self.product.name, product_barcode=self.product.barcode,
            quantity=5, price=Decimal('10.00'), taxable_at_sale=False,
        )
        record_stock_change(
            self.product, 2, 'checkout', user=self.user,
            order_detail=second, note='Partial fulfillment',
        )
        record_stock_change(
            self.product, 3, 'checkout_unfulfilled', user=self.user,
            order_detail=second, note='Stockout',
        )

        page = self.client.get(reverse('order_correction', args=[self.order.pk]))
        self.assertEqual(page.content.count(b'<main '), 1)
        row = next(r for r in page.context['line_rows'] if r['line'].pk == second.pk)
        self.assertEqual(row['original_qty'], 5)
        self.assertEqual(row['fulfilled_qty'], 2)
        self.assertEqual(row['remaining_qty'], 2)

    def test_void_can_be_undone_without_deleting_its_audit_history(self):
        correction = self._record_void()
        self.product.refresh_from_db()
        self.lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)
        self.assertEqual(self.product.stock_sold, 0)
        self.assertEqual(self.lot.quantity_on_hand, 10)

        detail = self.client.get(reverse('order_detail', args=[self.order.pk]))
        self.assertContains(detail, 'Undo void')
        self.assertContains(
            detail, reverse('transaction_correction_undo', args=[correction.pk]),
        )

        response = self.client.post(
            reverse('transaction_correction_undo', args=[correction.pk]),
            follow=True,
        )

        self.product.refresh_from_db()
        self.lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 7)
        self.assertEqual(self.product.stock_sold, 3)
        self.assertEqual(self.lot.quantity_on_hand, 7)
        self.assertTrue(
            TransactionCorrection.objects.filter(pk=correction.pk).exists(),
        )
        undo = TransactionCorrectionUndo.objects.get(correction=correction)
        self.assertEqual(undo.created_by, self.user)
        undo_change = StockChange.objects.get(change_type='correction_undo')
        self.assertEqual(undo_change.quantity, -3)
        movement = ProductLotMovement.objects.get(stock_change=undo_change)
        self.assertEqual(
            (movement.lot_number, movement.quantity, movement.direction),
            ('SOURCE-LOT', 3, ProductLotMovement.DIRECTION_OUT),
        )
        row = response.context['order_details_with_total'][0]
        self.assertEqual(row['corrected_qty'], 0)
        self.assertEqual(
            response.context['net_total_after_corrections'], Decimal('33.90'),
        )
        self.assertContains(response, 'Void undone')
        self.assertContains(response, 'Undone')

        # Repeated requests are idempotent and cannot change stock twice.
        self.client.post(
            reverse('transaction_correction_undo', args=[correction.pk]),
        )
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 7)
        self.assertEqual(self.product.stock_sold, 3)
        self.assertEqual(TransactionCorrectionUndo.objects.count(), 1)
        self.assertEqual(
            StockChange.objects.filter(change_type='correction_undo').count(), 1,
        )

        # The units become correctable again, and a later return still uses the
        # original sale lot rather than treating the undone void as active.
        self.client.post(
            reverse('order_correction', args=[self.order.pk]),
            {
                'correction_type': 'return',
                'reason': 'Actual return',
                f'qty_{self.line.pk}': '1',
                f'disposition_{self.line.pk}': 'restock',
            },
        )
        self.lot.refresh_from_db()
        self.assertEqual(self.lot.quantity_on_hand, 8)
        return_movement = ProductLotMovement.objects.get(
            stock_change__change_type='return',
        )
        self.assertEqual(return_movement.lot_number, 'SOURCE-LOT')

    def test_void_undo_without_restock_only_restores_transaction_counter(self):
        correction = self._record_void(
            TransactionCorrectionLine.DISPOSITION_DAMAGED,
        )
        self.product.refresh_from_db()
        self.lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 7)
        self.assertEqual(self.product.stock_sold, 0)
        self.assertEqual(self.lot.quantity_on_hand, 7)

        self.client.post(
            reverse('transaction_correction_undo', args=[correction.pk]),
        )

        self.product.refresh_from_db()
        self.lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 7)
        self.assertEqual(self.product.stock_sold, 3)
        self.assertEqual(self.lot.quantity_on_hand, 7)
        undo_change = StockChange.objects.get(change_type='correction_undo')
        self.assertFalse(undo_change.lot_movements.exists())

    def test_void_undo_fails_safely_when_returned_stock_is_unavailable(self):
        correction = self._record_void()
        self.product.quantity_in_stock = 2
        self.product.save(update_fields=['quantity_in_stock'])
        self.lot.quantity_on_hand = 2
        self.lot.save(update_fields=['quantity_on_hand'])

        response = self.client.post(
            reverse('transaction_correction_undo', args=[correction.pk]),
            follow=True,
        )

        self.product.refresh_from_db()
        self.lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 2)
        self.assertEqual(self.product.stock_sold, 0)
        self.assertEqual(self.lot.quantity_on_hand, 2)
        self.assertFalse(TransactionCorrectionUndo.objects.exists())
        self.assertFalse(
            StockChange.objects.filter(change_type='correction_undo').exists(),
        )
        self.assertContains(response, 'Undo unavailable')

    def test_no_sale_void_undo_restores_giveaway_counter(self):
        checkout = CheckoutOrder.objects.create(
            user=self.user,
            status=CheckoutOrder.STATUS_SUBMITTED,
            subtotal=Decimal('20.00'),
            total_price=Decimal('20.00'),
        )
        item = CheckoutOrderItem.objects.create(
            checkout=checkout,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            price=Decimal('10.00'),
            taxable=False,
            quantity=2,
        )
        self.product.quantity_in_stock = 5
        self.product.save(update_fields=['quantity_in_stock'])
        giveaway = record_stock_change(
            self.product, 2, 'giveaway', user=self.user,
            checkout_item=item, note='Original no-sale checkout',
        )
        remove_stock_from_lots(self.product, 2, giveaway)

        self.client.post(
            reverse('giveaway_correction', args=[checkout.pk]),
            {
                'correction_type': 'void',
                'reason': 'Wrong no-sale checkout',
                f'disposition_{item.pk}': 'restock',
            },
        )
        correction = TransactionCorrection.objects.get(checkout=checkout)
        detail = self.client.get(reverse('giveaway_detail', args=[checkout.pk]))
        self.assertContains(detail, 'Undo void')

        response = self.client.post(
            reverse('transaction_correction_undo', args=[correction.pk]),
            follow=True,
        )

        self.product.refresh_from_db()
        self.lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertEqual(self.product.stock_giveaway, 2)
        self.assertEqual(self.lot.quantity_on_hand, 5)
        self.assertContains(response, 'Undone')


@override_settings(AXES_ENABLED=False)
class PermissionAndRecoveryTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user(
            username='gina-test', password='pass1234', is_staff=True,
        )
        self.pu = User.objects.create_user(
            username='pu-test', password='pass1234', is_staff=False,
        )
        self.category = Category.objects.create(name='Permissions')
        self.product = Product.objects.create(
            name='Permission Product', barcode='PERM1001', price=Decimal('4.00'),
            quantity_in_stock=4, category=self.category,
        )
        ProductLot.objects.create(
            product=self.product, lot_number='PERM-LOT', quantity_on_hand=4,
        )

    @staticmethod
    def _special_order():
        return Item.objects.create(
            first_name='Alex', last_name='Patient', item_name='Left wrist brace',
            size='medium', side='left', item_number='SO-1001',
            phone_number='5551234567',
        )

    def test_pu_can_use_operational_pages_but_product_management_prompts_passkey(self):
        order = Order.objects.create(user=self.admin, submitted=True)
        client = Client()
        client.force_login(self.pu)

        for url in [
            reverse('create_order'), reverse('checkout'), reverse('order_view'),
            reverse('order_detail', args=[order.pk]), reverse('label_printing'),
            reverse('expired_products'),
        ]:
            self.assertEqual(client.get(url).status_code, 200, url)

        for url in [
            reverse('new_product'),
            reverse('edit_product', args=[self.product.pk]),
        ]:
            response = client.get(url)
            self.assertEqual(response.status_code, 302)
            self.assertIn(reverse('passkey_unlock'), response.url)

        response = client.post(reverse('delete_item', args=[self.product.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('passkey_unlock'), response.url)
        self.assertTrue(Product.objects.filter(pk=self.product.pk).exists())

    def test_stock_alert_pages_use_timezone_safe_date_filters(self):
        client = Client()
        client.force_login(self.admin)
        self.product.quantity_in_stock = 0
        self.product.save(update_fields=['quantity_in_stock'])
        StockChange.objects.create(
            product=self.product,
            change_type='checkout_unfulfilled',
            quantity=1,
            user=self.admin,
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            self.assertEqual(client.get(reverse('out_of_stock')).status_code, 200)

        self.product.quantity_in_stock = 1
        self.product.save(update_fields=['quantity_in_stock'])
        StockChange.objects.create(
            product=self.product,
            change_type='checkout',
            quantity=1,
            user=self.admin,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            self.assertEqual(client.get(reverse('low_stock_trend')).status_code, 200)

    def test_any_signed_in_user_can_save_checkin_inline_edit(self):
        session = CheckinSession.objects.create(user=self.pu, scanned_by='PU')
        client = Client()
        client.force_login(self.pu)
        response = client.post(
            reverse('checkin_edit_product', args=[session.pk, self.product.pk]),
            {
                'name': 'Inline Updated', 'brand': '', 'item_number': '',
                'price': '4.00', 'barcode': 'PERM1001',
                'quantity_in_stock': '999', 'category': str(self.category.pk),
                'unit_size': '', 'description': '', 'expiry_date': '',
                'price_per_unit': '', 'status': 'on',
                'lot_number': 'PERM-LOT', 'lot_expiry': '', 'lot_quantity': '4',
            },
        )
        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.name, 'Inline Updated')
        self.assertEqual(self.product.quantity_in_stock, 4)

    def test_product_delete_is_recoverable_with_lots_and_counters_intact(self):
        client = Client()
        client.force_login(self.admin)
        response = client.post(reverse('delete_item', args=[self.product.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Product.objects.filter(pk=self.product.pk).exists())
        archived = Product.all_objects.get(pk=self.product.pk)
        self.assertIsNotNone(archived.archived_at)
        self.assertEqual(archived.lots.get().quantity_on_hand, 4)
        self.assertEqual(archived.stock_deleted, 4)

        recovery = client.get(reverse('archive_recovery'))
        self.assertEqual(recovery.content.count(b'<main '), 1)
        self.assertContains(recovery, 'Permission Product')
        client.post(reverse('archive_recovery'), {
            'kind': 'product', 'object_id': self.product.pk,
        })
        restored = Product.objects.get(pk=self.product.pk)
        self.assertTrue(restored.status)
        self.assertEqual(restored.stock_deleted, 0)
        self.assertTrue(StockChange.objects.filter(
            product=restored, change_type='restoration', quantity=4,
        ).exists())

    def test_passkey_unlocked_pu_can_archive_product(self):
        client = Client()
        client.force_login(self.pu)
        session = client.session
        session[PASSKEY_SESSION_KEY] = time.time()
        session.save()

        client.post(reverse('delete_item', args=[self.product.pk]))

        self.assertFalse(Product.objects.filter(pk=self.product.pk).exists())
        self.assertTrue(Product.all_objects.filter(
            pk=self.product.pk, archived_by=self.pu,
        ).exists())

    def test_locked_pu_cannot_archive_special_order(self):
        item = self._special_order()
        client = Client()
        client.force_login(self.pu)

        response = client.post(reverse('item_list'), {
            'delete': '1', 'item_id': item.pk,
        })

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('passkey_unlock'), response.url)
        item.refresh_from_db()
        self.assertIsNone(item.archived_at)
        self.assertIsNone(item.archived_by)
        self.assertFalse(UserAction.objects.filter(
            action='delete_item_list', target=item.item_name,
        ).exists())

    def test_unlocked_pu_can_archive_recover_and_restore_special_order(self):
        item = self._special_order()
        original_values = (
            item.first_name, item.last_name, item.item_name, item.size,
            item.side, item.item_number, item.phone_number, item.is_checked,
        )
        client = Client()
        client.force_login(self.pu)
        session = client.session
        session[PASSKEY_SESSION_KEY] = time.time()
        session.save()

        archive = client.post(reverse('item_list'), {
            'delete': '1', 'item_id': item.pk,
        })

        self.assertRedirects(
            archive, reverse('item_list'), fetch_redirect_response=False,
        )
        item.refresh_from_db()
        self.assertIsNotNone(item.archived_at)
        self.assertEqual(item.archived_by, self.pu)
        self.assertEqual(item.archive_reason, 'Removed from Special Orders')
        self.assertEqual(
            (
                item.first_name, item.last_name, item.item_name, item.size,
                item.side, item.item_number, item.phone_number, item.is_checked,
            ),
            original_values,
        )
        active_page = client.get(reverse('item_list'))
        self.assertNotIn(item, list(active_page.context['items']))
        archive_action = UserAction.objects.get(
            action='delete_item_list', target=item.item_name,
        )
        self.assertEqual(archive_action.user, self.pu)
        self.assertIn('Moved to Recovery', archive_action.detail)

        recovery = client.get(reverse('archive_recovery'), {
            'type': 'special_order', 'q': 'SO-1001',
        })
        self.assertContains(recovery, 'Special order')
        self.assertContains(recovery, item.item_name)
        self.assertContains(recovery, 'Alex Patient')

        restore = client.post(reverse('archive_recovery'), {
            'kind': 'special_order', 'object_id': item.pk,
            'type': 'special_order',
        })

        self.assertEqual(restore.status_code, 302)
        item.refresh_from_db()
        self.assertIsNone(item.archived_at)
        self.assertIsNone(item.archived_by)
        self.assertEqual(item.archive_reason, '')
        self.assertContains(client.get(reverse('item_list')), item.item_name)
        restore_action = UserAction.objects.get(
            action='restore_archived_record', target=item.item_name,
        )
        self.assertEqual(restore_action.user, self.pu)
        self.assertEqual(restore_action.detail, 'special_order')


@override_settings(AXES_ENABLED=False)
class SupplierAndOrderingTrackingTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user(
            username='workflow-admin', password='pass1234', is_staff=True,
        )
        self.pu = User.objects.create_user(
            username='workflow-pu', password='pass1234', is_staff=False,
        )
        category = Category.objects.create(name='Supplier Workflow')
        self.product = Product.objects.create(
            name='Supplier Product', barcode='SUPP1001', price=Decimal('8.00'),
            quantity_in_stock=6, category=category,
        )

    def test_supplier_purchase_order_tracks_plan_without_changing_inventory(self):
        plan = SupplierOrderPlan.objects.create(
            created_by=self.admin, vendor_sequence=['mck'],
        )
        SupplierOrderPlanItem.objects.create(
            plan=plan, product=self.product, product_name=self.product.name,
            barcode=self.product.barcode, quantity=5, position=0,
        )
        client = Client()
        client.force_login(self.admin)
        page = client.get(reverse('supplier_purchase_orders'))
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.content.count(b'<main '), 1)
        create_data = {
            'action': 'create', 'supplier': 'mck',
            'confirmation_number': 'CONF-100',
            'order_date': date.today().isoformat(), 'expected_date': '',
            'status': 'submitted', 'notes': 'Tracking only',
            'plan_id': str(plan.pk),
        }
        response = client.post(reverse('supplier_purchase_orders'), create_data)
        self.assertEqual(response.status_code, 302)
        purchase_order = SupplierPurchaseOrder.objects.get()
        line = purchase_order.lines.get()
        self.assertEqual(line.quantity_ordered, 5)
        self.assertEqual(self.product.quantity_in_stock, 6)
        self.assertEqual(StockChange.objects.count(), 0)

        update = {
            **create_data, 'action': 'update',
            'purchase_order_id': str(purchase_order.pk),
            f'received_{line.pk}': '2',
        }
        client.post(reverse('supplier_purchase_orders'), update)
        purchase_order.refresh_from_db()
        self.assertEqual(purchase_order.status, SupplierPurchaseOrder.STATUS_PARTIAL)

        update[f'received_{line.pk}'] = '5'
        client.post(reverse('supplier_purchase_orders'), update)
        purchase_order.refresh_from_db()
        self.assertEqual(purchase_order.status, SupplierPurchaseOrder.STATUS_RECEIVED)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 6)

    def test_ordering_lifecycle_enforces_progress_and_keeps_status_history(self):
        entry = OrderingSheetEntry.objects.create(
            name='Lifecycle Drug', entry_type=OrderingSheetEntry.ENTRY_DRUG,
            reasoning=OrderingSheetEntry.REASON_STOCK,
            urgency=OrderingSheetEntry.URGENCY_HIGH,
            initials='PU', created_by=self.pu,
        )
        pu_client = Client()
        pu_client.force_login(self.pu)
        pu_client.post(reverse('ordering_sheet'), {
            'action': 'edit', 'entry_id': entry.pk,
            'name': 'Lifecycle Drug Updated', 'initials': 'PU',
            'reasoning': OrderingSheetEntry.REASON_STOCK,
            'urgency': OrderingSheetEntry.URGENCY_HIGH,
        })
        entry.refresh_from_db()
        self.assertEqual(entry.name, 'Lifecycle Drug Updated')

        # A regular user cannot advance shared ordering progress.
        pu_client.post(reverse('ordering_sheet'), {
            'action': 'update_status', 'entry_id': entry.pk,
            'status': OrderingSheetEntry.STATUS_ORDERED,
            'quantity_ordered': '5', 'quantity_received': '0',
        })
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PENDING)

        admin_client = Client()
        admin_client.force_login(self.admin)

        def advance(status, received):
            return admin_client.post(reverse('ordering_sheet'), {
                'action': 'update_status', 'entry_id': entry.pk,
                'status': status, 'supplier_name': 'McKesson',
                'quantity_ordered': '5', 'quantity_received': str(received),
                'expected_date': (date.today() + timedelta(days=2)).isoformat(),
                'order_note': f'Move to {status}',
            })

        advance(OrderingSheetEntry.STATUS_ORDERED, 0)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_ORDERED)
        self.assertIsNotNone(entry.ordered_at)

        # Full-received state cannot be claimed while only part is recorded.
        advance(OrderingSheetEntry.STATUS_RECEIVED, 2)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_ORDERED)

        advance(OrderingSheetEntry.STATUS_PARTIAL_RECEIVED, 2)
        advance(OrderingSheetEntry.STATUS_RECEIVED, 5)
        advance(OrderingSheetEntry.STATUS_READY, 5)
        advance(OrderingSheetEntry.STATUS_PICKED_UP, 5)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PICKED_UP)
        self.assertEqual(entry.quantity_received, 5)
        self.assertIsNotNone(entry.received_at)
        self.assertIsNotNone(entry.completed_at)
        self.assertEqual(OrderingSheetStatusEvent.objects.filter(entry=entry).count(), 5)

        completed = admin_client.get(reverse('ordering_sheet') + '?view=completed')
        self.assertContains(completed, 'Lifecycle Drug Updated')
