from datetime import date
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.template.loader import render_to_string
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import (
    Category,
    CheckinReceivingDraft,
    CheckinSession,
    Product,
    ProductLot,
    ProductLotMovement,
    StockChange,
    UserAction,
)


@override_settings(AXES_ENABLED=False)
class EditProductLotDerivationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='lot-editor-admin', password='test-password', is_staff=True,
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='Lot-derived inventory')
        self.product = Product.objects.create(
            name='Lot Derived Product', barcode='LOT-DERIVED-1',
            price=Decimal('8.99'), price_per_unit=Decimal('4.00'),
            quantity_in_stock=7, expiry_date=date(2031, 1, 15),
            category=self.category,
        )
        self.lot_a = ProductLot.objects.create(
            product=self.product, lot_number='LOT-A',
            expiry_date=date(2031, 1, 15), quantity_on_hand=2,
        )
        self.lot_b = ProductLot.objects.create(
            product=self.product, lot_number='LOT-B',
            expiry_date=date(2031, 3, 20), quantity_on_hand=5,
        )
        self.url = reverse('edit_product', args=[self.product.pk])

    def _base_post(self):
        return {
            'name': self.product.name,
            'brand': '',
            'item_number': '',
            'price': '8.99',
            'barcode': self.product.barcode,
            'category': str(self.category.pk),
            'unit_size': '',
            'description': '',
            'taxable': 'on',
            'status': 'on',
            'price_per_unit': '4.00',
            'next': reverse('inventory_display'),
        }

    def _use_unassigned_inventory(self, quantity=3, expiry=None):
        self.product.lots.all().delete()
        self.product.quantity_in_stock = quantity
        self.product.expiry_date = expiry
        self.product.save(update_fields=['quantity_in_stock', 'expiry_date'])
        return ProductLot.objects.create(
            product=self.product,
            lot_number=ProductLot.UNASSIGNED,
            expiry_date=expiry,
            quantity_on_hand=quantity,
        )

    @staticmethod
    def _identity_baseline(*lots):
        return {
            'lot_original_number': [lot.lot_number for lot in lots],
            'lot_original_expiry': [
                lot.expiry_date.strftime('%d-%m-%Y') if lot.expiry_date else ''
                for lot in lots
            ],
            'lot_original_quantity': [
                str(lot.quantity_on_hand) for lot in lots
            ],
            'lot_removed': ['0' for _lot in lots],
        }

    def test_edit_page_renders_stock_and_expiry_as_derived_non_inputs(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-derived-stock')
        self.assertContains(response, '>7</output>', html=False)
        self.assertContains(response, 'data-derived-expiries')
        self.assertContains(response, '15-01-2031')
        self.assertContains(response, '20-03-2031')
        self.assertNotContains(response, 'name="quantity_in_stock"')
        self.assertNotContains(response, 'name="expiry_date"')
        self.assertContains(response, 'name="lot_quantity"')
        self.assertContains(response, 'name="lot_expiry"')

    def test_edit_page_makes_entire_unassigned_lot_row_editable(self):
        unassigned = self._use_unassigned_inventory(
            quantity=3, expiry=date(2032, 5, 31),
        )

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            'name="lot_number" maxlength="64" value="UNASSIGNED"',
        )
        self.assertContains(
            response,
            f'name="lot_id" value="{unassigned.pk}"',
        )
        self.assertContains(
            response,
            'name="lot_original_number" value="UNASSIGNED"',
        )
        self.assertContains(
            response,
            'name="lot_original_quantity" value="3"',
        )
        self.assertContains(
            response,
            'name="lot_expiry" class="flatpickr-date" value="31-05-2032"',
        )
        self.assertContains(
            response,
            'type="number" name="lot_quantity" min="0" inputmode="numeric" '
            'value="3"',
        )
        self.assertContains(response, 'class="lot-row-remove"')
        self.assertNotContains(response, 'class="lot-row-locked"')
        self.assertContains(response, 'data-track-lot-removals')
        self.assertContains(response, 'name="lot_removed" value="0"')
        self.assertContains(
            response,
            'you can edit its lot number,',
        )
        self.assertContains(response, 'aria-describedby="productLotHelp"')
        self.assertNotContains(
            response,
            '<input type="hidden" name="lot_number" value="UNASSIGNED">',
            html=True,
        )
        self.assertNotContains(response, '<input type="hidden" name="lot_expiry"')
        self.assertNotContains(response, '<input type="hidden" name="lot_quantity"')

    def test_shared_lot_rows_keep_unassigned_locked_without_edit_page_flag(self):
        unassigned = self._use_unassigned_inventory(quantity=3)

        html = render_to_string('includes/_product_lot_rows.html', {
            'lot_rows': [{
                'lot_id': unassigned.pk,
                'lot_number': ProductLot.UNASSIGNED,
                'expiry_date': None,
                'quantity': 3,
                'is_unassigned': True,
                'staff_name': unassigned.staff_name,
            }],
        })

        self.assertIn(
            '<input type="hidden" name="lot_number" value="UNASSIGNED">',
            html,
        )
        self.assertNotIn(
            'name="lot_number" maxlength="64" value="UNASSIGNED"',
            html,
        )
        self.assertIn('<input type="hidden" name="lot_expiry" value="">', html)
        self.assertIn('<input type="hidden" name="lot_quantity" value="3">', html)
        self.assertIn('>Locked</span>', html)
        self.assertNotIn(
            'class="lot-row-remove"', html.split('<template', 1)[0],
        )

    def test_unassigned_full_row_edit_preserves_record_and_audits_unit_delta(self):
        original_expiry = date(2032, 5, 31)
        new_expiry = date(2033, 6, 30)
        unassigned = self._use_unassigned_inventory(
            quantity=3, expiry=original_expiry,
        )
        session = CheckinSession.objects.create(
            user=self.user, scanned_by='Lot editor test',
        )
        draft = CheckinReceivingDraft.objects.create(
            session=session,
            product=self.product,
            existing_lot=unassigned,
            lot_number=ProductLot.UNASSIGNED,
            lot_expiry=original_expiry,
            revision=4,
        )
        historical_change = StockChange.objects.create(
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            change_type='checkin',
            quantity=3,
            user=self.user,
        )
        historical_movement = ProductLotMovement.objects.create(
            stock_change=historical_change,
            lot=unassigned,
            lot_number=ProductLot.UNASSIGNED,
            expiry_date=original_expiry,
            quantity=3,
            direction=ProductLotMovement.DIRECTION_IN,
        )
        original_pk = unassigned.pk
        payload = self._base_post()
        payload.update({
            'lot_id': [str(unassigned.pk)],
            'lot_number': ['supplier-42'],
            'lot_expiry': ['30-06-2033'],
            'lot_quantity': ['4'],
        })
        payload.update(self._identity_baseline(unassigned))

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        unassigned.refresh_from_db()
        self.assertEqual(unassigned.pk, original_pk)
        self.assertEqual(unassigned.lot_number, 'SUPPLIER-42')
        self.assertEqual(unassigned.expiry_date, new_expiry)
        self.assertEqual(unassigned.quantity_on_hand, 4)
        self.assertIsNone(unassigned.archived_at)
        self.assertEqual(self.product.quantity_in_stock, 4)
        self.assertEqual(self.product.expiry_date, new_expiry)

        change = StockChange.objects.get(
            product=self.product,
            change_type='error_add',
        )
        self.assertEqual(change.quantity, 1)
        movement = change.lot_movements.get()
        self.assertEqual(
            (movement.lot_id, movement.direction, movement.lot_number,
             movement.expiry_date, movement.quantity),
            (original_pk, ProductLotMovement.DIRECTION_IN, 'SUPPLIER-42',
             new_expiry, 1),
        )
        historical_movement.refresh_from_db()
        self.assertEqual(historical_movement.lot_id, original_pk)
        self.assertEqual(historical_movement.lot_number, ProductLot.UNASSIGNED)
        self.assertEqual(historical_movement.expiry_date, original_expiry)
        self.assertFalse(StockChange.objects.filter(
            product=self.product,
            change_type='lot_reassignment',
        ).exists())

        draft.refresh_from_db()
        self.assertEqual(draft.existing_lot_id, original_pk)
        self.assertEqual(draft.lot_number, 'SUPPLIER-42')
        self.assertEqual(draft.lot_expiry, new_expiry)
        self.assertEqual(draft.revision, 5)
        self.assertTrue(UserAction.objects.filter(
            user=self.user,
            action='edit_product',
            detail__contains='SUPPLIER-42',
        ).exists())

    def test_expired_unassigned_lot_can_be_named_without_reclassifying_expiry(self):
        expiry = date(2020, 5, 31)
        unassigned = self._use_unassigned_inventory(quantity=3, expiry=expiry)
        payload = self._base_post()
        payload.update({
            'lot_id': [str(unassigned.pk)],
            'lot_number': ['expired-supplier-42'],
            'lot_expiry': ['31-05-2020'],
            'lot_quantity': ['3'],
        })
        payload.update(self._identity_baseline(unassigned))

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        unassigned.refresh_from_db()
        self.assertEqual(unassigned.lot_number, 'EXPIRED-SUPPLIER-42')
        self.assertEqual(unassigned.expiry_date, expiry)
        self.assertEqual(unassigned.quantity_on_hand, 3)
        self.assertEqual(self.product.expiry_date, expiry)
        self.assertEqual(self.product.quantity_in_stock, 3)

    def test_unassigned_lot_units_and_expiry_are_editable(self):
        expiry = date(2032, 5, 31)
        unassigned = self._use_unassigned_inventory(quantity=3, expiry=expiry)
        payload = self._base_post()
        payload.update({
            'lot_id': [str(unassigned.pk)],
            'lot_number': [ProductLot.UNASSIGNED],
            'lot_expiry': ['01-06-2032'],
            'lot_quantity': ['4'],
        })
        payload.update(self._identity_baseline(unassigned))

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        unassigned.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 4)
        self.assertEqual(self.product.expiry_date, date(2032, 6, 1))
        self.assertEqual(unassigned.quantity_on_hand, 4)
        self.assertEqual(unassigned.expiry_date, date(2032, 6, 1))
        self.assertEqual(unassigned.lot_number, ProductLot.UNASSIGNED)
        change = StockChange.objects.get(
            product=self.product, change_type='error_add',
        )
        self.assertEqual(change.quantity, 1)
        movement = change.lot_movements.get()
        self.assertEqual(movement.lot_id, unassigned.pk)
        self.assertEqual(movement.direction, ProductLotMovement.DIRECTION_IN)
        self.assertEqual(movement.quantity, 1)

    def test_named_lot_number_edit_preserves_lot_record_and_draft(self):
        session = CheckinSession.objects.create(
            user=self.user, scanned_by='Named lot editor test',
        )
        historical_change = StockChange.objects.create(
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            change_type='checkin',
            quantity=2,
            user=self.user,
        )
        historical_movement = ProductLotMovement.objects.create(
            stock_change=historical_change,
            lot=self.lot_a,
            lot_number='LOT-A',
            expiry_date=self.lot_a.expiry_date,
            quantity=2,
            direction=ProductLotMovement.DIRECTION_IN,
        )
        draft = CheckinReceivingDraft.objects.create(
            session=session,
            product=self.product,
            existing_lot=self.lot_a,
            lot_number=self.lot_a.lot_number,
            lot_expiry=self.lot_a.expiry_date,
            revision=2,
        )
        original_pk = self.lot_a.pk
        payload = self._base_post()
        payload.update({
            'lot_id': [str(self.lot_a.pk), str(self.lot_b.pk)],
            'lot_number': ['LOT-A-CORRECTED', 'LOT-B'],
            'lot_expiry': ['15-01-2031', '20-03-2031'],
            'lot_quantity': ['2', '5'],
        })
        payload.update(self._identity_baseline(self.lot_a, self.lot_b))

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        self.lot_a.refresh_from_db()
        self.assertEqual(self.lot_a.pk, original_pk)
        self.assertEqual(self.lot_a.lot_number, 'LOT-A-CORRECTED')
        self.assertIsNone(self.lot_a.archived_at)
        historical_movement.refresh_from_db()
        self.assertEqual(historical_movement.lot_id, original_pk)
        self.assertEqual(historical_movement.lot_number, 'LOT-A')
        self.assertEqual(historical_movement.expiry_date, date(2031, 1, 15))
        self.assertFalse(ProductLot.objects.filter(
            product=self.product, lot_number='LOT-A',
        ).exists())
        self.assertFalse(StockChange.objects.filter(
            product=self.product, change_type='lot_reassignment',
        ).exists())
        draft.refresh_from_db()
        self.assertEqual(draft.existing_lot_id, original_pk)
        self.assertEqual(draft.lot_number, 'LOT-A-CORRECTED')
        self.assertEqual(draft.revision, 3)
        self.assertTrue(UserAction.objects.filter(
            user=self.user,
            action='edit_product',
            detail__contains='LOT-A-CORRECTED',
        ).exists())

    def test_lot_identity_collision_is_rejected_without_changes(self):
        payload = self._base_post()
        payload.update({
            'lot_id': [str(self.lot_a.pk), str(self.lot_b.pk)],
            'lot_number': ['LOT-B', 'LOT-B'],
            'lot_expiry': ['20-03-2031', '20-03-2031'],
            'lot_quantity': ['2', '5'],
        })
        payload.update(self._identity_baseline(self.lot_a, self.lot_b))

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'lot identity already exists')
        self.assertContains(response, 'value="LOT-B"')
        self.lot_a.refresh_from_db()
        self.lot_b.refresh_from_db()
        self.assertEqual(
            (self.lot_a.lot_number, self.lot_a.expiry_date),
            ('LOT-A', date(2031, 1, 15)),
        )
        self.assertEqual(self.lot_b.lot_number, 'LOT-B')
        self.assertFalse(StockChange.objects.filter(
            product=self.product, change_type='lot_reassignment',
        ).exists())

    def test_new_row_before_unassigned_duplicate_cannot_bypass_assignment_audit(self):
        expiry = date(2032, 5, 31)
        unassigned = self._use_unassigned_inventory(quantity=3, expiry=expiry)
        payload = self._base_post()
        payload.update({
            'lot_id': ['', str(unassigned.pk)],
            'lot_number': ['SUPPLIER-42', 'SUPPLIER-42'],
            'lot_expiry': ['31-05-2032', '31-05-2032'],
            'lot_quantity': ['0', '3'],
            'lot_original_number': ['', ProductLot.UNASSIGNED],
            'lot_original_expiry': ['', '31-05-2032'],
            'lot_original_quantity': ['', '3'],
            'lot_removed': ['0', '0'],
        })

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'duplicates another lot')
        unassigned.refresh_from_db()
        self.assertEqual(unassigned.lot_number, ProductLot.UNASSIGNED)
        self.assertEqual(unassigned.quantity_on_hand, 3)
        self.assertIsNone(unassigned.archived_at)
        self.assertFalse(ProductLot.objects.filter(
            product=self.product, lot_number='SUPPLIER-42',
        ).exists())
        self.assertFalse(StockChange.objects.filter(
            product=self.product, change_type='lot_reassignment',
        ).exists())

    def test_stale_lot_identity_baseline_is_rejected(self):
        payload = self._base_post()
        payload.update({
            'lot_id': [str(self.lot_a.pk), str(self.lot_b.pk)],
            'lot_number': ['LOT-A-FROM-OLD-PAGE', 'LOT-B'],
            'lot_expiry': ['15-01-2031', '20-03-2031'],
            'lot_quantity': ['2', '5'],
        })
        payload.update(self._identity_baseline(self.lot_a, self.lot_b))
        ProductLot.objects.filter(pk=self.lot_a.pk).update(
            lot_number='LOT-A-CHANGED-ELSEWHERE',
        )

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'changed while this page was open')
        self.lot_a.refresh_from_db()
        self.assertEqual(self.lot_a.lot_number, 'LOT-A-CHANGED-ELSEWHERE')
        self.assertNotContains(response, 'value="LOT-A-CHANGED-ELSEWHERE"')
        self.assertContains(
            response,
            'name="lot_original_number" value="LOT-A"',
        )
        self.assertFalse(UserAction.objects.filter(
            user=self.user,
            action='edit_product',
        ).exists())

    def test_stale_lot_quantity_baseline_is_rejected(self):
        payload = self._base_post()
        payload.update({
            'lot_id': [str(self.lot_a.pk), str(self.lot_b.pk)],
            'lot_number': ['LOT-A', 'LOT-B'],
            'lot_expiry': ['15-01-2031', '20-03-2031'],
            'lot_quantity': ['4', '5'],
        })
        payload.update(self._identity_baseline(self.lot_a, self.lot_b))
        ProductLot.objects.filter(pk=self.lot_a.pk).update(quantity_on_hand=3)
        Product.objects.filter(pk=self.product.pk).update(quantity_in_stock=8)

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'stock changed while this page was open')
        self.lot_a.refresh_from_db()
        self.product.refresh_from_db()
        self.assertEqual(self.lot_a.quantity_on_hand, 3)
        self.assertEqual(self.product.quantity_in_stock, 8)
        self.assertFalse(UserAction.objects.filter(
            user=self.user, action='edit_product',
        ).exists())

    def test_removing_only_unassigned_lot_zeros_and_archives_it(self):
        expiry = date(2032, 5, 31)
        unassigned = self._use_unassigned_inventory(quantity=3, expiry=expiry)
        session = CheckinSession.objects.create(
            user=self.user, scanned_by='Removal test',
        )
        draft = CheckinReceivingDraft.objects.create(
            session=session,
            product=self.product,
            existing_lot=unassigned,
            lot_number=ProductLot.UNASSIGNED,
            lot_expiry=expiry,
            revision=6,
        )
        payload = self._base_post()
        payload.update({
            'lot_id': [str(unassigned.pk)],
            'lot_number': [ProductLot.UNASSIGNED],
            'lot_expiry': ['31-05-2032'],
            'lot_quantity': ['3'],
        })
        payload.update(self._identity_baseline(unassigned))
        payload['lot_removed'] = ['1']

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        unassigned.refresh_from_db()
        draft.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 0)
        self.assertIsNone(self.product.expiry_date)
        self.assertFalse(self.product.expiry_dates.exists())
        self.assertEqual(unassigned.quantity_on_hand, 0)
        self.assertIsNotNone(unassigned.archived_at)
        self.assertEqual(unassigned.archived_by, self.user)
        self.assertIsNone(draft.existing_lot)
        self.assertEqual(draft.lot_number, '')
        self.assertIsNone(draft.lot_expiry)
        self.assertEqual(draft.revision, 7)
        change = StockChange.objects.get(
            product=self.product, change_type='error_subtract',
        )
        self.assertEqual(change.quantity, 3)
        movement = change.lot_movements.get()
        self.assertEqual(
            (movement.lot_id, movement.direction, movement.lot_number,
             movement.expiry_date, movement.quantity),
            (unassigned.pk, ProductLotMovement.DIRECTION_OUT,
             ProductLot.UNASSIGNED, expiry, 3),
        )
        self.assertTrue(UserAction.objects.filter(
            user=self.user,
            action='edit_product',
            detail__contains='removed UNASSIGNED',
        ).exists())

    def test_remove_and_replace_lot_with_same_total_records_both_directions(self):
        expiry = date(2032, 5, 31)
        unassigned = self._use_unassigned_inventory(quantity=3, expiry=expiry)
        payload = self._base_post()
        payload.update({
            'lot_id': [str(unassigned.pk), ''],
            'lot_number': [ProductLot.UNASSIGNED, 'REPLACEMENT-LOT'],
            'lot_expiry': ['31-05-2032', '30-06-2033'],
            'lot_quantity': ['3', '3'],
            'lot_original_number': [ProductLot.UNASSIGNED, ''],
            'lot_original_expiry': ['31-05-2032', ''],
            'lot_original_quantity': ['3', ''],
            'lot_removed': ['1', '0'],
        })

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        unassigned.refresh_from_db()
        replacement = ProductLot.objects.get(
            product=self.product,
            lot_number='REPLACEMENT-LOT',
            archived_at__isnull=True,
        )
        self.assertEqual(self.product.quantity_in_stock, 3)
        self.assertEqual(self.product.expiry_date, date(2033, 6, 30))
        self.assertEqual(unassigned.quantity_on_hand, 0)
        self.assertIsNotNone(unassigned.archived_at)
        self.assertEqual(replacement.quantity_on_hand, 3)
        add_change = StockChange.objects.get(
            product=self.product, change_type='error_add',
        )
        remove_change = StockChange.objects.get(
            product=self.product, change_type='error_subtract',
        )
        self.assertEqual(add_change.quantity, 3)
        self.assertEqual(remove_change.quantity, 3)
        self.assertEqual(
            add_change.lot_movements.get().direction,
            ProductLotMovement.DIRECTION_IN,
        )
        self.assertEqual(
            remove_change.lot_movements.get().direction,
            ProductLotMovement.DIRECTION_OUT,
        )

    def test_validation_rerender_keeps_typed_unassigned_replacement(self):
        unassigned = self._use_unassigned_inventory(quantity=3)
        payload = self._base_post()
        payload.update({
            'price': 'not-a-price',
            'lot_id': [str(unassigned.pk)],
            'lot_number': ['TYPED-LOT-9'],
            'lot_expiry': ['30-06-2033'],
            'lot_quantity': ['5'],
        })
        payload.update(self._identity_baseline(unassigned))

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'value="TYPED-LOT-9"')
        self.assertContains(
            response,
            'name="lot_expiry" class="flatpickr-date" value="30-06-2033"',
        )
        self.assertContains(
            response,
            'name="lot_quantity" min="0" inputmode="numeric" value="5"',
        )
        self.assertContains(response, 'class="lot-row-remove"')
        unassigned.refresh_from_db()
        self.assertEqual(unassigned.lot_number, ProductLot.UNASSIGNED)
        self.assertIsNone(unassigned.archived_at)

    def test_validation_rerender_after_sole_removal_restores_a_blank_row(self):
        unassigned = self._use_unassigned_inventory(quantity=3)
        payload = self._base_post()
        payload.update({
            'price': 'not-a-price',
            'lot_id': [str(unassigned.pk), ''],
            'lot_number': [ProductLot.UNASSIGNED, ''],
            'lot_expiry': ['', ''],
            'lot_quantity': ['3', ''],
            'lot_original_number': [ProductLot.UNASSIGNED, ''],
            'lot_original_expiry': ['', ''],
            'lot_original_quantity': ['3', ''],
            'lot_removed': ['1', '0'],
        })

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'hidden data-lot-removed="true"')
        self.assertContains(response, 'name="lot_removed" value="1"')
        editor_source = (
            Path(settings.BASE_DIR)
            / 'app' / 'templates' / 'includes' / 'product_lot_editor.html'
        ).read_text(encoding='utf-8')
        self.assertIn(
            "editor.hasAttribute('data-track-lot-removals')",
            editor_source,
        )
        self.assertIn(
            "!rows.querySelector('.lot-editor-row:not([data-lot-removed=\"true\"])')",
            editor_source,
        )
        unassigned.refresh_from_db()
        self.assertEqual(unassigned.quantity_on_hand, 3)
        self.assertIsNone(unassigned.archived_at)

    def test_enter_advances_through_fields_and_focuses_save(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-enter-next-fields')
        self.assertContains(response, 'data-enter-next-submit="#saveBtn"')
        self.assertContains(response, 'aria-keyshortcuts="Enter"')
        self.assertContains(response, 'aria-describedby="editProductFormEnterHint"')
        self.assertContains(response, 'id="editProductFormEnterHint"')
        self.assertContains(response, '<kbd>Enter</kbd>', html=True)
        self.assertContains(
            response,
            'Next single-line field. Internal Notes keeps Enter for new lines.',
        )
        self.assertContains(
            response,
            'class="bottom-bar" id="bottomBar" inert aria-hidden="true"',
        )
        self.assertContains(response, 'id="saveBtn" disabled')
        self.assertContains(response, 'form="editProductForm"')

        source = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'edit_product.html'
        ).read_text(encoding='utf-8')
        self.assertIn("saveBtn.disabled = !isChanged;", source)
        self.assertIn("bottomBar.removeAttribute('inert');", source)
        self.assertIn("bottomBar.setAttribute('inert', '');", source)
        self.assertIn("event.target.closest('[data-add-lot], .lot-row-remove')", source)
        self.assertIn('window.requestAnimationFrame(checkChanges);', source)

    def test_lot_rows_override_forged_summary_stock_and_expiry(self):
        payload = self._base_post()
        payload.update({
            'quantity_in_stock': '999',
            'expiry_date': '31-12-2099',
            'extra_expiry_dates': ['30-11-2099'],
            'lot_id': [str(self.lot_a.pk), str(self.lot_b.pk)],
            'lot_number': ['LOT-A', 'LOT-B'],
            'lot_expiry': ['10-02-2032', '25-04-2032'],
            'lot_quantity': ['3', '6'],
        })
        payload.update(self._identity_baseline(self.lot_a, self.lot_b))

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 9)
        self.assertEqual(self.product.expiry_date, date(2032, 2, 10))
        self.assertEqual(
            list(self.product.expiry_dates.order_by('expiry_date').values_list('expiry_date', flat=True)),
            [date(2032, 2, 10), date(2032, 4, 25)],
        )
        self.assertEqual(
            list(self.product.lots.filter(archived_at__isnull=True).order_by('lot_number')
                 .values_list('lot_number', 'quantity_on_hand')),
            [('LOT-A', 3), ('LOT-B', 6)],
        )

    def test_legacy_summary_drift_is_a_separate_audit_reconciliation(self):
        self.product.quantity_in_stock = 6
        self.product.save(update_fields=['quantity_in_stock'])
        payload = self._base_post()
        payload.update({
            'lot_id': [str(self.lot_a.pk), str(self.lot_b.pk)],
            'lot_number': ['LOT-A', 'LOT-B'],
            'lot_expiry': ['15-01-2031', '20-03-2031'],
            'lot_quantity': ['3', '5'],
        })
        payload.update(self._identity_baseline(self.lot_a, self.lot_b))

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        changes = list(
            StockChange.objects.filter(
                product=self.product,
                change_type='error_add',
            ).order_by('pk')
        )
        self.assertEqual([change.quantity for change in changes], [1, 1])
        lot_change, reconciliation = changes
        movement = lot_change.lot_movements.get()
        self.assertEqual(movement.lot_id, self.lot_a.pk)
        self.assertEqual(movement.quantity, lot_change.quantity)
        self.assertFalse(reconciliation.lot_movements.exists())
        self.assertIn('summary reconciled', reconciliation.note)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 8)
        self.assertEqual(self.product.stock_bought, 2)

    def test_save_returns_to_exact_filtered_product_details_url(self):
        origin = (
            f"{reverse('product_details', args=[self.product.pk])}"
            "?start=2026-04-01&end=2026-08-24"
            "&type=line&granularity=week"
        )
        payload = self._base_post()
        payload.update({
            'next': origin,
            'lot_id': [str(self.lot_a.pk), str(self.lot_b.pk)],
            'lot_number': ['LOT-A', 'LOT-B'],
            'lot_expiry': ['15-01-2031', '20-03-2031'],
            'lot_quantity': ['2', '5'],
        })
        payload.update(self._identity_baseline(self.lot_a, self.lot_b))

        response = self.client.post(self.url, payload)

        self.assertRedirects(response, origin, fetch_redirect_response=False)

    def test_unsafe_return_url_falls_back_to_inventory(self):
        unsafe = 'https://example.com/steal-state'
        response = self.client.get(self.url, {'next': unsafe})
        self.assertEqual(response.context['next'], reverse('inventory_display'))

        payload = self._base_post()
        payload.update({
            'next': unsafe,
            'lot_id': [str(self.lot_a.pk), str(self.lot_b.pk)],
            'lot_number': ['LOT-A', 'LOT-B'],
            'lot_expiry': ['15-01-2031', '20-03-2031'],
            'lot_quantity': ['2', '5'],
        })
        payload.update(self._identity_baseline(self.lot_a, self.lot_b))
        response = self.client.post(self.url, payload)
        self.assertRedirects(
            response, reverse('inventory_display'), fetch_redirect_response=False,
        )

    def test_product_details_edit_keeps_separate_archive_inventory_origin(self):
        inventory_origin = (
            f"{reverse('inventory_display')}?q={self.product.barcode}"
            "&sort=quantity_in_stock&direction=desc&page=2"
        )
        details_origin = (
            f"{reverse('product_details', args=[self.product.pk])}"
            "?granularity=month"
        )

        edit_response = self.client.get(self.url, {
            'next': details_origin,
            'archive_next': inventory_origin,
        })

        self.assertEqual(edit_response.context['next'], details_origin)
        self.assertEqual(edit_response.context['archive_next'], inventory_origin)
        self.assertContains(
            edit_response,
            f'name="next" value="{inventory_origin.replace("&", "&amp;")}"',
        )

        response = self.client.post(
            reverse('delete_item', args=[self.product.pk]),
            {'next': inventory_origin},
        )

        self.assertRedirects(
            response, inventory_origin, fetch_redirect_response=False,
        )
        self.assertFalse(Product.objects.filter(pk=self.product.pk).exists())
        self.assertTrue(Product.all_objects.filter(pk=self.product.pk).exists())

    def test_page_script_keeps_derived_summary_synced_to_lot_editor(self):
        source = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'edit_product.html'
        ).read_text(encoding='utf-8')

        self.assertIn('function refreshDerivedInventory()', source)
        self.assertIn("row.dataset.lotRemoved === 'true'", source)
        self.assertIn("row.querySelector('[name=\"lot_quantity\"]')", source)
        self.assertIn("row.querySelector('[name=\"lot_expiry\"]')", source)
        self.assertIn("lotEditor.addEventListener('input', refreshDerivedInventory)", source)
        self.assertIn('window.requestAnimationFrame(refreshDerivedInventory)', source)

