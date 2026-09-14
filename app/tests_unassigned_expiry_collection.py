import json
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Product, ProductLot, ProductLotMovement, StockChange, UserAction


@override_settings(AXES_ENABLED=False)
class UnassignedExpiryCollectionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='unassigned-expiry-collector', password='test-pass',
        )
        self.client.force_login(self.user)
        self.url = reverse('expired_products')
        self.product = Product.objects.create(
            name='Unassigned stock without expiry', barcode='UNASSIGNED-EXPIRY-1',
            price='5.00', quantity_in_stock=5,
        )
        self.unassigned = ProductLot.objects.create(
            product=self.product, lot_number=ProductLot.UNASSIGNED,
            expiry_date=None, quantity_on_hand=5,
        )

    def _queued(self, lot=None, quantity=1):
        lot = lot or self.unassigned
        return {
            'product_id': str(lot.product_id), 'lot_id': str(lot.pk),
            'quantity': str(quantity),
        }

    def _review(self, rows=None):
        return self.client.post(self.url, {
            'expiry_action': 'review', 'collection_mode': '1',
            'collected_rows': json.dumps(rows or [self._queued()]),
        })

    def _confirmation(self, response):
        self.assertEqual(response.status_code, 200)
        selected = [row for row in response.context['review_rows'] if row['selected']]
        return {
            'expiry_action': 'confirm', 'review_token': response.context['review_token'],
            'collected': 'yes', 'selected_lots': [str(row['index']) for row in selected],
            **{f'quantity_{row["index"]}': str(row['collected_quantity']) for row in selected},
        }

    def _retire(self, lot=None, quantity=1, **extra):
        lot = lot or self.unassigned
        return self.client.post(self.url, {
            'mode': 'log', 'product_id': str(lot.product_id),
            'retire_expired': '1', 'retire_lot_id': str(lot.pk),
            'retire_quantity': str(quantity), **extra,
        })

    def _assert_unchanged(self):
        self.product.refresh_from_db()
        self.unassigned.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertEqual(self.product.stock_expired, 0)
        self.assertEqual(self.unassigned.quantity_on_hand, 5)
        self.assertIsNone(self.unassigned.expiry_date)
        self.assertIsNone(self.unassigned.archived_at)
        self.assertFalse(StockChange.objects.filter(change_type='expired').exists())
        self.assertFalse(ProductLotMovement.objects.exists())
        self.assertFalse(UserAction.objects.filter(action='retire_expired').exists())

    def test_scan_allows_undated_unassigned_without_inventing_expiry(self):
        response = self.client.get(self.url, {'mode': 'log', 'pid': self.product.pk})

        self.assertEqual(response.status_code, 200)
        summary = response.context['product_extra']
        self.assertEqual(len(summary['eligible_lots']), 1)
        row = summary['eligible_lots'][0]
        self.assertEqual(row['id'], self.unassigned.pk)
        self.assertEqual(row['lot_number'], ProductLot.UNASSIGNED)
        self.assertIsNone(row['date'])
        self.assertTrue(row['is_default'])
        self.assertEqual(summary['retirement_quantity'], 5)
        self.assertEqual(summary['expired_quantity'], 0)
        self.assertContains(response, 'Expiry not recorded')
        self.assertNotContains(response, 'Add expiry first')
        self._assert_unchanged()

    def test_collection_review_and_confirmation_log_exact_undated_lot(self):
        review = self._review([self._queued(quantity=2)])
        confirmation = self._confirmation(review)
        rows = review.context['review_rows']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['lot_id'], self.unassigned.pk)
        self.assertEqual(rows[0]['expiry'], '')
        self.assertTrue(rows[0]['eligible'])
        self.assertNotContains(review, 'cannot log as expired')
        self._assert_unchanged()

        self.assertEqual(self.client.post(self.url, confirmation).status_code, 302)

        self.product.refresh_from_db()
        self.unassigned.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, self.product.stock_expired), (3, 2))
        self.assertEqual(self.unassigned.quantity_on_hand, 3)
        self.assertIsNone(self.product.expiry_date)
        self.assertIsNone(self.unassigned.expiry_date)
        self.assertIsNone(self.unassigned.archived_at)
        self.assertFalse(self.product.expiry_dates.exists())
        change = StockChange.objects.get(change_type='expired')
        self.assertEqual(change.quantity, 2)
        self.assertIn(ProductLot.UNASSIGNED, change.note)
        movement = change.lot_movements.get()
        self.assertEqual((movement.lot_id, movement.quantity), (self.unassigned.pk, 2))
        self.assertEqual(movement.lot_number, ProductLot.UNASSIGNED)
        self.assertIsNone(movement.expiry_date)
        self.assertEqual(UserAction.objects.filter(action='retire_expired').count(), 1)

        self.assertEqual(self.client.post(self.url, confirmation).status_code, 302)
        self.product.refresh_from_db()
        self.unassigned.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, self.unassigned.quantity_on_hand), (3, 3))
        self.assertEqual(StockChange.objects.filter(change_type='expired').count(), 1)
        self.assertEqual(ProductLotMovement.objects.count(), 1)

    def test_direct_retirement_archives_exhausted_undated_lot_and_keeps_audit(self):
        scanned = self.client.get(self.url, {'mode': 'log', 'pid': self.product.pk})
        token = scanned.context['product_extra']['retire_token']

        response = self._retire(quantity=5, retire_token=token)

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.unassigned.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, self.product.stock_expired), (0, 5))
        self.assertEqual(self.unassigned.quantity_on_hand, 0)
        self.assertIsNone(self.unassigned.expiry_date)
        self.assertIsNotNone(self.unassigned.archived_at)
        self.assertEqual(self.unassigned.archived_by, self.user)
        movement = ProductLotMovement.objects.get()
        self.assertEqual((movement.lot_id, movement.quantity), (self.unassigned.pk, 5))
        self.assertIsNone(movement.expiry_date)

    def test_mixed_collection_preserves_unselected_named_and_future_lots(self):
        expired = ProductLot.objects.create(
            product=self.product, lot_number='EXPIRED', quantity_on_hand=3,
            expiry_date=date.today() - timedelta(days=3),
        )
        future = ProductLot.objects.create(
            product=self.product, lot_number='FUTURE', quantity_on_hand=4,
            expiry_date=date.today() + relativedelta(months=2),
        )
        undated_named = ProductLot.objects.create(
            product=self.product, lot_number='NAMED-NO-EXPIRY', quantity_on_hand=2,
        )
        self.product.quantity_in_stock = 14
        self.product.expiry_date = expired.expiry_date
        self.product.save(update_fields=['quantity_in_stock', 'expiry_date'])
        review = self._review([self._queued(quantity=2), self._queued(expired, 1)])

        self.assertEqual(self.client.post(self.url, self._confirmation(review)).status_code, 302)

        self.product.refresh_from_db()
        self.unassigned.refresh_from_db()
        expired.refresh_from_db()
        future.refresh_from_db()
        undated_named.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, self.product.stock_expired), (11, 3))
        self.assertEqual(self.unassigned.quantity_on_hand, 3)
        self.assertEqual(expired.quantity_on_hand, 2)
        self.assertEqual(future.quantity_on_hand, 4)
        self.assertEqual(undated_named.quantity_on_hand, 2)
        self.assertIsNone(self.unassigned.expiry_date)
        self.assertIsNone(undated_named.expiry_date)
        self.assertEqual(
            set(ProductLotMovement.objects.values_list('lot_id', 'quantity')),
            {(self.unassigned.pk, 2), (expired.pk, 1)},
        )
        self.assertFalse(ProductLot.objects.filter(product=self.product, archived_at__isnull=False).exists())

    def test_undated_named_lot_remains_blocked(self):
        self.unassigned.lot_number = 'NAMED-NO-EXPIRY'
        self.unassigned.save(update_fields=['lot_number'])

        scanned = self.client.get(self.url, {'mode': 'log', 'pid': self.product.pk})
        self.assertEqual(scanned.context['product_extra']['eligible_lots'], [])
        self.assertContains(scanned, 'Add expiry first')
        self.assertEqual(self._review().status_code, 302)
        self.assertEqual(self._retire().status_code, 302)
        self._assert_unchanged()

    def test_collection_can_deplete_only_dated_lot_before_undated_unassigned(self):
        expired = ProductLot.objects.create(
            product=self.product, lot_number='MAIN', quantity_on_hand=2,
            expiry_date=date.today() - timedelta(days=3),
        )
        self.product.quantity_in_stock = 7
        self.product.expiry_date = expired.expiry_date
        self.product.save(update_fields=['quantity_in_stock', 'expiry_date'])
        review = self._review([self._queued(expired, 2), self._queued(quantity=5)])

        self.assertEqual(self.client.post(self.url, self._confirmation(review)).status_code, 302)

        self.product.refresh_from_db()
        self.unassigned.refresh_from_db()
        expired.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, self.product.stock_expired), (0, 7))
        self.assertIsNone(self.product.expiry_date)
        self.assertEqual((expired.quantity_on_hand, self.unassigned.quantity_on_hand), (0, 0))
        self.assertIsNotNone(expired.archived_at)
        self.assertIsNotNone(self.unassigned.archived_at)
        self.assertIsNone(self.unassigned.expiry_date)
        self.assertEqual(
            set(ProductLotMovement.objects.values_list('lot_id', 'quantity')),
            {(expired.pk, 2), (self.unassigned.pk, 5)},
        )

    def test_dated_unassigned_still_obeys_one_month_cutoff(self):
        future = date.today() + relativedelta(months=1) + timedelta(days=1)
        self.unassigned.expiry_date = future
        self.unassigned.save(update_fields=['expiry_date'])

        scanned = self.client.get(self.url, {'mode': 'log', 'pid': self.product.pk})
        self.assertEqual(scanned.context['product_extra']['eligible_lots'], [])
        self.assertEqual(self._review().status_code, 302)
        self.assertEqual(self._retire().status_code, 302)
        self.product.refresh_from_db()
        self.unassigned.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, self.unassigned.quantity_on_hand), (5, 5))
        self.assertEqual(self.unassigned.expiry_date, future)
        self.assertFalse(StockChange.objects.filter(change_type='expired').exists())
        self.assertFalse(ProductLotMovement.objects.exists())

    def test_setting_future_expiry_after_review_invalidates_confirmation(self):
        review = self._review([self._queued(quantity=2)])
        confirmation = self._confirmation(review)
        self.unassigned.expiry_date = date.today() + relativedelta(months=2)
        self.unassigned.save(update_fields=['expiry_date'])

        self.assertEqual(self.client.post(self.url, confirmation).status_code, 302)

        self.product.refresh_from_db()
        self.unassigned.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, self.unassigned.quantity_on_hand), (5, 5))
        self.assertEqual(self.product.stock_expired, 0)
        self.assertFalse(StockChange.objects.filter(change_type='expired').exists())
        self.assertFalse(ProductLotMovement.objects.exists())
