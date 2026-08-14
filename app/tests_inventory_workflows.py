import time
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
