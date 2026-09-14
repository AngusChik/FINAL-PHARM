import json
import time
from datetime import date, timedelta
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from dateutil.relativedelta import relativedelta
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import Product, ProductLot, ProductLotMovement, StockChange, UserAction


@override_settings(AXES_ENABLED=False)
class ExpiredCollectionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='exact-expiry-collector', password='test-pass')
        self.client.force_login(self.user)
        self.url = reverse('expired_products')
        self.product = Product.objects.create(
            name='Same numbered lots', barcode='EXACT-COLLECT-1', price='12.50',
            quantity_in_stock=10, expiry_date=date.today() - timedelta(days=5),
        )
        self.expired = ProductLot.objects.create(
            product=self.product, lot_number='SAME-NUMBER', quantity_on_hand=3,
            expiry_date=date.today() - timedelta(days=5),
        )
        self.next_month = ProductLot.objects.create(
            product=self.product, lot_number='SAME-NUMBER', quantity_on_hand=3,
            expiry_date=date.today() + relativedelta(months=1),
        )
        self.future = ProductLot.objects.create(
            product=self.product, lot_number='TOO-EARLY', quantity_on_hand=4,
            expiry_date=date.today() + relativedelta(months=1) + timedelta(days=1),
        )
        self.barcodeless = Product.objects.create(
            name='No barcode product', price='4.50', quantity_in_stock=4,
        )
        self.other_lot = ProductLot.objects.create(
            product=self.barcodeless, lot_number='OTHER-PRODUCT', quantity_on_hand=4,
            expiry_date=date.today() - timedelta(days=2),
        )

    def _queued(self, lot=None, quantity=1, **overrides):
        lot = lot or self.expired
        return {
            'product_id': str(lot.product_id), 'lot_id': str(lot.pk),
            'quantity': str(quantity), **overrides,
        }

    def _review(self, rows=None, **extra):
        return self.client.post(self.url, {
            'expiry_action': 'review',
            'collected_rows': json.dumps(rows if rows is not None else [self._queued()]),
            **extra,
        })

    def _confirmation(self, review, **overrides):
        rows = review.context['review_rows']
        return {
            'expiry_action': 'confirm', 'review_token': review.context['review_token'],
            'collected': 'yes',
            'selected_lots': [str(row['index']) for row in rows if row['selected']],
            **{
                f'quantity_{row["index"]}': str(row['collected_quantity'])
                for row in rows if row['selected']
            },
            **overrides,
        }

    def _retire(self, product=None, lot_id=None, quantity=1, **extra):
        product = product or self.product
        return self.client.post(self.url, {
            'mode': 'log', 'product_id': str(product.pk),
            'retire_expired': '1',
            'retire_lot_id': str(self.expired.pk if lot_id is None else lot_id),
            'retire_quantity': str(quantity), **extra,
        })

    def _stock_state(self):
        return (
            list(Product.all_objects.order_by('pk').values()),
            list(ProductLot.objects.order_by('pk').values()),
        )

    def _assert_no_retirement(self, before):
        self.assertEqual(self._stock_state(), before)
        self.assertFalse(StockChange.objects.filter(change_type='expired').exists())
        self.assertFalse(ProductLotMovement.objects.exists())
        self.assertFalse(UserAction.objects.filter(action='retire_expired').exists())

    def _scan(self, product):
        return self.client.get(self.url, {'mode': 'log', 'pid': product.pk})

    def test_picker_uses_active_expired_lot_dates_and_quantities_instead_of_product_header(self):
        earliest = date.today() - timedelta(days=20)
        self.product.quantity_in_stock = 13
        self.product.expiry_date = date.today() + timedelta(days=50)
        self.product.item_number = 'EXPIRY-SKU-17'
        self.product.save(update_fields=['quantity_in_stock', 'expiry_date', 'item_number'])
        ProductLot.objects.create(
            product=self.product, lot_number='OLDER-EXPIRED', expiry_date=earliest,
            quantity_on_hand=2,
        )
        ProductLot.objects.create(product=self.product, lot_number='UNDATED', quantity_on_hand=1)
        ProductLot.objects.create(
            product=self.product, lot_number='ARCHIVED-EXPIRED', quantity_on_hand=6,
            expiry_date=date.today() - timedelta(days=50), archived_at=timezone.now(),
        )
        ProductLot.objects.create(
            product=self.product, lot_number='DEPLETED-EXPIRED', quantity_on_hand=0,
            expiry_date=date.today() - timedelta(days=60),
        )
        before = self._stock_state()
        response = self.client.get(self.url, {'mode': 'log'})
        rows = {row['product_id']: row for row in response.context['expiry_picker_products']}
        self.assertEqual(rows[self.product.pk], {
            'product_id': self.product.pk, 'name': self.product.name,
            'barcode': self.product.barcode, 'item_number': 'EXPIRY-SKU-17',
            'expired_quantity': 5, 'earliest_expiry': earliest.isoformat(),
        })
        self.assertEqual(rows[self.barcodeless.pk]['barcode'], '')
        self.assertEqual(rows[self.barcodeless.pk]['item_number'], '')
        self.assertEqual(rows[self.barcodeless.pk]['expired_quantity'], 4)
        self._assert_no_retirement(before)

    def test_picker_excludes_products_without_current_quantity_bearing_expired_lots(self):
        excluded_ids = []
        scenarios = (
            ('Future only', 2, date.today() + timedelta(days=5), 2, False, False),
            ('Expires today', 2, date.today(), 2, False, False),
            ('Named undated only', 2, None, 2, False, False),
            ('Depleted lot only', 2, date.today() - timedelta(days=1), 0, False, False),
            ('Archived lot only', 2, date.today() - timedelta(days=1), 2, True, False),
            ('Archived product', 2, date.today() - timedelta(days=1), 2, False, True),
            ('No product stock', 0, date.today() - timedelta(days=1), 1, False, False),
        )
        for name, stock, expiry, lot_quantity, archived_lot, archived_product in scenarios:
            product = Product.objects.create(
                name=name, price='2.00', quantity_in_stock=stock,
                expiry_date=date.today() - timedelta(days=10),
                archived_at=timezone.now() if archived_product else None,
            )
            ProductLot.objects.create(
                product=product, lot_number='TRACKED', expiry_date=expiry,
                quantity_on_hand=lot_quantity,
                archived_at=timezone.now() if archived_lot else None,
            )
            excluded_ids.append(product.pk)
        before = self._stock_state()
        response = self.client.get(self.url, {'mode': 'log'})
        ids = {row['product_id'] for row in response.context['expiry_picker_products']}
        self.assertEqual(ids, {self.product.pk, self.barcodeless.pk})
        self.assertTrue(ids.isdisjoint(excluded_ids))
        self._assert_no_retirement(before)

    def test_picker_is_independent_of_list_search_and_expiry_filters(self):
        before = self._stock_state()
        expected = self.client.get(self.url, {'mode': 'log'}).context['expiry_picker_products']
        response = self.client.get(self.url, {
            'mode': 'log', 'name_query': 'No product matches this list search',
            'date_filter': 'custom', 'date_from': (date.today() + timedelta(days=90)).isoformat(),
            'date_to': (date.today() + timedelta(days=100)).isoformat(), 'sort': '-name',
        })
        self.assertEqual(response.context['products'], [])
        self.assertEqual(response.context['expiry_picker_products'], expected)
        self.assertEqual(len(expected), 2)
        self._assert_no_retirement(before)

    def test_product_id_loads_barcodeless_product_without_changing_stock(self):
        before = self._stock_state()
        response = self.client.post(self.url, {
            'mode': 'log', 'product_id': str(self.barcodeless.pk),
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['product'].pk, self.barcodeless.pk)
        self.assertEqual(response.context['product_extra']['eligible_lots'][0]['id'], self.other_lot.pk)
        self._assert_no_retirement(before)

    def test_product_id_and_barcode_mismatch_cannot_retire_either_product(self):
        before = self._stock_state()
        response = self._retire(
            product=self.barcodeless, lot_id=self.other_lot.pk,
            barcode=self.product.barcode,
        )
        self.assertEqual(response.status_code, 302)
        self._assert_no_retirement(before)

    def test_invalid_or_duplicate_product_id_never_falls_back_to_barcode(self):
        before = self._stock_state()
        for product_id in ('not-an-id', '-1', '9999999999', [str(self.product.pk)] * 2):
            with self.subTest(product_id=product_id):
                response = self._retire(product_id=product_id, barcode=self.product.barcode)
                self.assertEqual(response.status_code, 302)
                self._assert_no_retirement(before)

    def test_same_lot_number_with_different_expiry_retires_only_exact_id(self):
        token = self._scan(self.product).context['product_extra']['retire_token']
        response = self._retire(lot_id=self.next_month.pk, quantity=2, retire_token=token)
        self.assertEqual(response.status_code, 302)
        self.expired.refresh_from_db()
        self.next_month.refresh_from_db()
        self.future.refresh_from_db()
        self.product.refresh_from_db()
        self.assertEqual((self.expired.quantity_on_hand, self.next_month.quantity_on_hand,
                          self.future.quantity_on_hand), (3, 1, 4))
        self.assertEqual((self.product.quantity_in_stock, self.product.stock_expired), (8, 2))
        movement = ProductLotMovement.objects.get()
        self.assertEqual(movement.lot_id, self.next_month.pk)
        self.assertEqual(movement.expiry_date, self.next_month.expiry_date)
        self.assertEqual(movement.quantity, 2)

    def test_foreign_missing_archived_and_legacy_lot_ids_cannot_substitute(self):
        archived = ProductLot.objects.create(
            product=self.product, lot_number='ARCHIVED', quantity_on_hand=1,
            expiry_date=date.today() - timedelta(days=4), archived_at=timezone.now(),
        )
        before = self._stock_state()
        for lot_id in (self.other_lot.pk, archived.pk, '9999999999', 'legacy', 'invalid'):
            with self.subTest(lot_id=lot_id):
                self.assertEqual(self._retire(lot_id=lot_id).status_code, 302)
                self._assert_no_retirement(before)

    def test_named_undated_stock_cannot_inherit_stale_product_expiry(self):
        product = Product.objects.create(
            name='Undated named stock', barcode='EXACT-UNDATED', price='2.00',
            quantity_in_stock=3, expiry_date=date.today() - timedelta(days=10),
        )
        lot = ProductLot.objects.create(product=product, lot_number='NO-EXPIRY', quantity_on_hand=3)
        before = self._stock_state()
        summary = self._scan(product).context['product_extra']
        self.assertEqual(summary['eligible_lots'], [])
        self.assertEqual(summary['undated_quantity'], 3)
        self.assertEqual(summary['lots'][0]['id'], lot.pk)
        self.assertFalse(summary['lots'][0]['eligible'])
        self.assertNotIn(product.pk, [item.pk for item in self.client.get(self.url).context['products']])
        self.assertEqual(self._retire(product=product, lot_id='legacy').status_code, 302)
        self.assertEqual(self._review([self._queued(lot)]).status_code, 302)
        self._assert_no_retirement(before)

    def test_archived_named_history_does_not_fabricate_legacy_expired_stock(self):
        product = Product.objects.create(
            name='Stale archived header', barcode='EXACT-ARCHIVED', price='2.00',
            quantity_in_stock=3, expiry_date=date.today() - timedelta(days=10),
        )
        ProductLot.objects.create(
            product=product, lot_number='ARCHIVED-HISTORY', quantity_on_hand=3,
            expiry_date=product.expiry_date, archived_at=timezone.now(),
        )
        before = self._stock_state()
        self.assertEqual(self._scan(product).context['product_extra']['eligible_lots'], [])
        self.assertNotIn(product.pk, [item.pk for item in self.client.get(self.url).context['products']])
        self.assertEqual(self._retire(product=product, lot_id='legacy').status_code, 302)
        self.assertEqual(self._review([
            self._queued(product_id=str(product.pk), lot_id='legacy'),
        ]).status_code, 302)
        self._assert_no_retirement(before)

    def test_queue_keeps_exact_selection_and_collected_amounts_across_expiry_windows(self):
        before = self._stock_state()
        review = self._review([
            self._queued(self.next_month, 2), self._queued(self.other_lot, 1),
        ], date_filter='')
        self.assertEqual(review.status_code, 200)
        self.assertTemplateUsed(review, 'expired_log_review.html')
        rows = review.context['review_rows']
        selected = {row['lot_id']: row for row in rows if row['selected']}
        self.assertEqual(set(selected), {self.next_month.pk, self.other_lot.pk})
        self.assertEqual(selected[self.next_month.pk]['quantity'], 3)
        self.assertEqual(selected[self.next_month.pk]['collected_quantity'], 2)
        self.assertEqual(selected[self.other_lot.pk]['collected_quantity'], 1)
        self.assertEqual(review.context['total_units'], 3)
        self.assertEqual(review.context['product_count'], 2)
        self.assertFalse(next(row for row in rows if row['lot_id'] == self.expired.pk)['selected'])
        self.assertFalse(next(row for row in rows if row['lot_id'] == self.future.pk)['eligible'])
        self._assert_no_retirement(before)

    def test_malformed_present_queue_never_falls_back_to_checked_products(self):
        before = self._stock_state()
        for raw in ('', '[', '{}', 'null', '[]', '[null]', '["bad"]'):
            with self.subTest(raw=raw):
                response = self.client.post(self.url, {
                    'expiry_action': 'review', 'collected_rows': raw,
                    'selected_products': [self.product.pk],
                })
                self.assertEqual(response.status_code, 302)
                self._assert_no_retirement(before)

    def test_collection_intent_with_missing_queue_cannot_review_checked_products(self):
        before = self._stock_state()
        response = self.client.post(self.url, {
            'expiry_action': 'review', 'collection_mode': '1',
            'selected_products': [self.product.pk],
        })
        self.assertEqual(response.status_code, 302)
        self._assert_no_retirement(before)

    def test_invalid_duplicate_or_excess_queue_quantities_reject_entire_review(self):
        before = self._stock_state()
        invalid_rows = [
            [self._queued(), self._queued()],
            [self._queued(quantity='')], [self._queued(quantity=0)],
            [self._queued(quantity=-1)], [self._queued(quantity='1.5')],
            [self._queued(quantity=4)], [self._queued(quantity='2147483648')],
            [self._queued(quantity='１')], [self._queued(quantity='1e0')],
            [self._queued(), {'product_id': str(self.product.pk), 'lot_id': str(self.next_month.pk)}],
            [self._queued(), {**self._queued(self.next_month), 'quantity': True}],
            [self._queued(), {**self._queued(self.next_month), 'quantity': 1}],
        ]
        for rows in invalid_rows:
            with self.subTest(rows=rows):
                self.assertEqual(self._review(rows).status_code, 302)
                self._assert_no_retirement(before)

    def test_queue_rejects_foreign_lot_ineligible_lot_and_mismatched_product_selection(self):
        before = self._stock_state()
        cases = (
            ([self._queued(lot_id=str(self.other_lot.pk))], {}),
            ([self._queued(self.future)], {}),
            ([self._queued(lot_id='9999999999')], {}),
            ([self._queued(lot_id='legacy')], {}),
            ([self._queued()], {'selected_products': [self.barcodeless.pk]}),
        )
        for rows, extra in cases:
            with self.subTest(rows=rows, extra=extra):
                self.assertEqual(self._review(rows, **extra).status_code, 302)
                self._assert_no_retirement(before)

    def test_queue_quantities_cannot_exceed_total_product_stock(self):
        self.product.quantity_in_stock = 1
        self.product.save(update_fields=['quantity_in_stock'])
        before = self._stock_state()
        review = self._review([self._queued(self.expired), self._queued(self.next_month)])
        self.assertEqual(review.status_code, 302)
        self._assert_no_retirement(before)

    def test_final_confirmation_updates_exact_queued_lots_and_matching_audit(self):
        review = self._review([self._queued(self.next_month, 2), self._queued(self.other_lot, 4)])
        self.assertEqual(review.status_code, 200)
        response = self.client.post(self.url, self._confirmation(review))
        self.assertEqual(response.status_code, 302)
        for product in (self.product, self.barcodeless):
            product.refresh_from_db()
            self.assertEqual(
                product.quantity_in_stock,
                sum(product.lots.filter(archived_at__isnull=True).values_list('quantity_on_hand', flat=True)),
            )
        self.assertEqual((self.product.quantity_in_stock, self.product.stock_expired), (8, 2))
        self.assertEqual((self.barcodeless.quantity_in_stock, self.barcodeless.stock_expired), (0, 4))
        self.expired.refresh_from_db()
        self.future.refresh_from_db()
        self.other_lot.refresh_from_db()
        self.assertEqual((self.expired.quantity_on_hand, self.future.quantity_on_hand), (3, 4))
        self.assertIsNotNone(self.other_lot.archived_at)
        self.assertEqual(self.other_lot.archived_by, self.user)
        movements = list(ProductLotMovement.objects.select_related('stock_change'))
        self.assertEqual({(row.lot_id, row.quantity, row.expiry_date) for row in movements}, {
            (self.next_month.pk, 2, self.next_month.expiry_date),
            (self.other_lot.pk, 4, self.other_lot.expiry_date),
        })
        self.assertEqual(StockChange.objects.filter(change_type='expired').count(), 2)
        self.assertEqual(UserAction.objects.filter(action='retire_expired').count(), 2)
        for movement in movements:
            self.assertEqual(movement.stock_change.user, self.user)
            self.assertEqual(movement.stock_change.quantity, movement.quantity)
            self.assertEqual(movement.direction, ProductLotMovement.DIRECTION_OUT)

    def test_success_receipt_reports_only_final_checked_and_reduced_quantities(self):
        review = self._review([self._queued(self.expired, 3), self._queued(self.other_lot, 4)])
        chosen = next(row for row in review.context['review_rows'] if row['lot_id'] == self.expired.pk)
        response = self.client.post(self.url, self._confirmation(review, **{
            'selected_lots': [str(chosen['index'])], f'quantity_{chosen["index"]}': '1',
        }))
        self.assertEqual(response.status_code, 302)
        params = parse_qs(urlsplit(response['Location']).query)
        self.assertEqual(params['collection_logged'], ['1'])
        self.assertTrue(params['collection_receipt'][0])
        after_logging = self._stock_state()
        receipt = self.client.get(response['Location']).context['logged_collection']
        self.assertTrue(receipt['id'])
        self.assertEqual(receipt['rows'], [{
            'product_id': str(self.product.pk), 'lot_id': str(self.expired.pk), 'quantity': '1',
        }])
        self.assertEqual(self.client.get(response['Location']).context['logged_collection'], receipt)
        self.assertEqual(self._stock_state(), after_logging)
        self.assertEqual(StockChange.objects.filter(change_type='expired').count(), 1)
        self.barcodeless.refresh_from_db()
        self.assertEqual(self.barcodeless.quantity_in_stock, 4)

    def test_missing_tampered_expired_or_another_users_receipt_cannot_clear_a_collection(self):
        review = self._review()
        response = self.client.post(self.url, self._confirmation(review))
        params = parse_qs(urlsplit(response['Location']).query)
        token = params['collection_receipt'][0]
        after_logging = self._stock_state()
        for receipt_params in (
            {'collection_logged': '1'},
            {'collection_logged': '1', 'collection_receipt': token + 'tampered'},
        ):
            with self.subTest(params=receipt_params):
                page = self.client.get(self.url, {'mode': 'log', **receipt_params})
                self.assertIsNone(page.context['logged_collection'])
        with patch('django.core.signing.time.time', return_value=time.time() + 3601):
            self.assertIsNone(self.client.get(response['Location']).context['logged_collection'])
        other_user = User.objects.create_user(username='different-expiry-collector', password='test-pass')
        self.client.force_login(other_user)
        self.assertIsNone(self.client.get(response['Location']).context['logged_collection'])
        self.assertEqual(self._stock_state(), after_logging)

    def test_lot_oversupply_never_trims_unrelated_unassigned_stock_during_retirement(self):
        ProductLot.objects.create(
            product=self.product, lot_number=ProductLot.UNASSIGNED, quantity_on_hand=2,
        )
        before = self._stock_state()
        self._retire(lot_id=self.expired.pk)
        self._assert_no_retirement(before)
        review = self._review()
        if review.status_code == 200:
            self.client.post(self.url, self._confirmation(review))
        else:
            self.assertEqual(review.status_code, 302)
        self._assert_no_retirement(before)

    def test_lot_archived_after_queue_review_rejects_whole_confirmation(self):
        review = self._review([self._queued(self.next_month), self._queued(self.other_lot)])
        self.other_lot.archived_at = timezone.now()
        self.other_lot.save(update_fields=['archived_at', 'updated_at'])
        before = self._stock_state()
        self.client.post(self.url, self._confirmation(review))
        self._assert_no_retirement(before)

    def test_queue_confirmation_replay_cannot_remove_more_partial_stock(self):
        review = self._review([self._queued(self.next_month)])
        payload = self._confirmation(review)
        self.client.post(self.url, payload)
        after_first = self._stock_state()
        self.client.post(self.url, payload)
        self.assertEqual(self._stock_state(), after_first)
        self.next_month.refresh_from_db()
        self.assertEqual(self.next_month.quantity_on_hand, 2)
        self.assertEqual(StockChange.objects.filter(change_type='expired').count(), 1)
        self.assertEqual(ProductLotMovement.objects.get().quantity, 1)

    def test_single_scan_token_rejects_stale_lot_identity_and_replay(self):
        token = self._scan(self.product).context['product_extra']['retire_token']
        self.next_month.expiry_date -= timedelta(days=1)
        self.next_month.save(update_fields=['expiry_date', 'updated_at'])
        before = self._stock_state()
        self._retire(lot_id=self.next_month.pk, retire_token=token)
        self._assert_no_retirement(before)
        fresh_token = self._scan(self.product).context['product_extra']['retire_token']
        self._retire(lot_id=self.next_month.pk, retire_token=fresh_token)
        after_first = self._stock_state()
        self._retire(lot_id=self.next_month.pk, retire_token=fresh_token)
        self.assertEqual(self._stock_state(), after_first)
        self.assertEqual(StockChange.objects.filter(change_type='expired').count(), 1)

    def test_genuine_undated_unassigned_stock_and_untracked_residual_share_exact_legacy_lot(self):
        product = Product.objects.create(
            name='Unassigned legacy stock', price='1.50', quantity_in_stock=5,
            expiry_date=date.today() - timedelta(days=1),
        )
        lot = ProductLot.objects.create(
            product=product, lot_number=ProductLot.UNASSIGNED, quantity_on_hand=3,
        )
        review = self._review([
            self._queued(lot, 4, lot_id='legacy'),
        ])
        self.assertEqual(review.status_code, 200)
        selected = [row for row in review.context['review_rows'] if row['selected']]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['lot_id'], 'legacy')
        self.assertEqual(selected[0]['collected_quantity'], 4)
        self.client.post(self.url, self._confirmation(review))
        product.refresh_from_db()
        lot.refresh_from_db()
        self.assertEqual((product.quantity_in_stock, product.stock_expired, lot.quantity_on_hand), (1, 4, 1))
        self.assertEqual(ProductLot.objects.filter(product=product).count(), 1)
        movement = ProductLotMovement.objects.get()
        self.assertEqual((movement.lot_id, movement.quantity), (lot.pk, 4))
