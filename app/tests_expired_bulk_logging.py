from datetime import date, timedelta
from decimal import Decimal
import json
import time
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from dateutil.relativedelta import relativedelta
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from .models import Category, Product, ProductLot, ProductLotMovement, StockChange, UserAction
from .views import ExpiredProductView


@override_settings(AXES_ENABLED=False)
class ExpiredBulkLoggingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='expiry-collector', password='test-pass')
        self.client.force_login(self.user)
        self.url = reverse('expired_products')
        self.category = Category.objects.create(name='Expiry collection')
        self.first = Product.objects.create(
            name='Collected first', barcode='COLLECT-1', category=self.category,
            price=Decimal('10'), quantity_in_stock=10,
        )
        self.expired = ProductLot.objects.create(
            product=self.first, lot_number='EXPIRED-A', quantity_on_hand=3,
            expiry_date=date.today() - timedelta(days=3),
        )
        self.also_expired = ProductLot.objects.create(
            product=self.first, lot_number='EXPIRED-B', quantity_on_hand=2,
            expiry_date=date.today() - timedelta(days=2),
        )
        self.future = ProductLot.objects.create(
            product=self.first, lot_number='FUTURE', quantity_on_hand=5,
            expiry_date=date.today() + relativedelta(months=1) + timedelta(days=1),
        )
        self.second = Product.objects.create(
            name='Collected second', barcode='COLLECT-2', category=self.category,
            price=Decimal('6'), quantity_in_stock=4,
        )
        self.second_lot = ProductLot.objects.create(
            product=self.second, lot_number='SECOND', quantity_on_hand=4,
            expiry_date=date.today() - timedelta(days=1),
        )

    def review(self, ids=None, **extra):
        return self.client.post(self.url, {
            'expiry_action': 'review',
            'selected_products': ids if ids is not None else [self.first.pk, self.second.pk],
            **extra,
        })

    def confirmation(self, review, quantities=None, **extra):
        rows = review.context['review_rows']
        return {
            'expiry_action': 'confirm', 'review_token': review.context['review_token'],
            'collected': 'yes',
            'selected_lots': [str(index) for index, row in enumerate(rows) if row['selected']],
            **{f'quantity_{index}': str((quantities or {}).get(row['lot_id'], row['quantity']))
               for index, row in enumerate(rows)},
            **extra,
        }

    def assert_no_logging(self):
        self.assertFalse(StockChange.objects.filter(change_type='expired').exists())
        self.assertFalse(ProductLotMovement.objects.exists())
        self.assertFalse(UserAction.objects.filter(action='retire_expired').exists())

    def stock_and_logging_state(self):
        return (
            list(Product.all_objects.order_by('pk').values()),
            list(ProductLot.objects.order_by('pk').values()),
            list(StockChange.objects.order_by('pk').values()),
            list(ProductLotMovement.objects.order_by('pk').values()),
            list(UserAction.objects.filter(action='retire_expired').order_by('pk').values()),
        )

    def test_selection_and_row_log_are_present_before_checkin(self):
        response = self.client.get(self.url)
        self.assertContains(response, 'id="expSelectAll"')
        self.assertContains(response, '<input type="checkbox" name="selected_products"', count=2)
        self.assertContains(response, 'id="expBulkLog" hidden disabled')
        html = response.content.decode()
        row = html[html.index('<div class="exp-row-actions">'):]
        self.assertLess(row.index('Log expired'), row.index('Check In'))
        self.assert_no_logging()

    def test_review_shows_other_lots_but_only_selects_current_expiry_window(self):
        review = self.review([self.first.pk])
        self.assertEqual(review.status_code, 200)
        self.assertEqual({row['lot_id'] for row in review.context['review_rows'] if row['selected']},
                         {self.expired.pk, self.also_expired.pk})
        future = next(row for row in review.context['review_rows'] if row['lot_id'] == self.future.pk)
        self.assertFalse(future['eligible'])
        self.assertFalse(future['selected'])
        self.assertEqual(review.context['total_units'], 5)
        self.assertContains(review, 'physically collected')
        self.assert_no_logging()

    def test_row_button_reviews_only_that_product_even_if_others_are_checked(self):
        review = self.review(log_product=self.first.pk)
        self.assertEqual({row['product_id'] for row in review.context['review_rows']}, {self.first.pk})
        self.assert_no_logging()

    def test_stale_row_after_complete_logging_refreshes_list_without_logging_again(self):
        listing = self.client.get(self.url)
        self.assertIn(self.second, listing.context['products'])
        review = self.review([self.second.pk])
        self.client.post(self.url, self.confirmation(review))
        before = self.stock_and_logging_state()

        response = self.review(log_product=self.second.pk, name_query='Collected', sort='-name')

        self.assertEqual(response.status_code, 302)
        self.assertEqual(parse_qs(urlsplit(response.url).query), {
            'mode': ['view'], 'name_query': ['Collected'], 'sort': ['-name'],
        })
        refreshed = self.client.get(response.url)
        self.assertContains(refreshed, 'No selected products still have eligible stock. The list has been refreshed.')
        self.assertContains(refreshed, self.second.name)
        self.assertContains(refreshed, 'No stock remains on shelf.')
        self.assertNotIn(self.second, refreshed.context['products'])
        self.assertNotContains(refreshed, 'id="expiryReviewForm"')
        self.assertEqual(self.stock_and_logging_state(), before)

    def test_stale_depleted_product_does_not_block_current_bulk_review(self):
        review = self.review([self.second.pk])
        self.client.post(self.url, self.confirmation(review))
        before = self.stock_and_logging_state()

        review = self.review()

        self.assertEqual(review.status_code, 200)
        self.assertEqual({row['product_id'] for row in review.context['review_rows']}, {self.first.pk})
        self.assertEqual(review.context['product_count'], 1)
        self.assertEqual(review.context['total_units'], 5)
        self.assertEqual(review.context['skipped_products'], [{
            'name': self.second.name, 'reason': 'No stock remains on shelf.',
        }])
        self.assertContains(review, self.second.name)
        self.assertContains(review, 'No stock remains on shelf.')
        self.assertEqual(self.stock_and_logging_state(), before)

    def test_stale_product_with_only_future_stock_is_skipped_in_bulk_review(self):
        self.future.expiry_date = date.today() + timedelta(days=20)
        self.future.save(update_fields=['expiry_date', 'updated_at'])
        review = self.review([self.first.pk])
        self.client.post(self.url, self.confirmation(review))
        before = self.stock_and_logging_state()

        review = self.review()

        self.assertEqual(review.status_code, 200)
        self.assertEqual({row['product_id'] for row in review.context['review_rows']}, {self.second.pk})
        self.assertEqual(review.context['product_count'], 1)
        self.assertEqual(review.context['total_units'], 4)
        self.assertEqual(review.context['skipped_products'], [{
            'name': self.first.name, 'reason': 'No eligible stock remains in this expiry window.',
        }])
        self.assertContains(review, self.first.name)
        self.assertContains(review, 'No eligible stock remains in this expiry window.')
        self.assertEqual(self.stock_and_logging_state(), before)

    def test_all_stale_bulk_selections_refresh_list_instead_of_empty_review(self):
        review = self.review()
        self.client.post(self.url, self.confirmation(review))
        before = self.stock_and_logging_state()

        response = self.review()

        self.assertEqual(response.status_code, 302)
        refreshed = self.client.get(response.url)
        self.assertContains(refreshed, 'No selected products still have eligible stock. The list has been refreshed.')
        self.assertContains(refreshed, self.first.name)
        self.assertContains(refreshed, 'No eligible stock remains in this expiry window.')
        self.assertContains(refreshed, self.second.name)
        self.assertContains(refreshed, 'No stock remains on shelf.')
        self.assertNotContains(refreshed, 'id="expiryReviewForm"')
        self.assertEqual(refreshed.context['products'], [])
        self.assertEqual(self.stock_and_logging_state(), before)

    def test_exact_collection_with_stale_product_still_rejects_entire_review(self):
        review = self.review([self.second.pk])
        self.client.post(self.url, self.confirmation(review))
        before = self.stock_and_logging_state()
        collected = [
            {'product_id': str(self.first.pk), 'lot_id': str(self.expired.pk), 'quantity': '1'},
            {'product_id': str(self.second.pk), 'lot_id': str(self.second_lot.pk), 'quantity': '1'},
        ]

        response = self.review(collected_rows=json.dumps(collected))

        self.assertEqual(response.status_code, 302)
        refreshed = self.client.get(response.url)
        self.assertNotContains(refreshed, 'id="expiryReviewForm"')
        self.assertContains(refreshed, self.second.name)
        self.assertEqual(parse_qs(urlsplit(response.url).query), {'mode': ['log']})
        self.assertEqual(self.stock_and_logging_state(), before)

    def test_only_checked_lot_is_logged_even_when_other_quantities_are_posted(self):
        review = self.review()
        chosen = next(row for row in review.context['review_rows'] if row['lot_id'] == self.also_expired.pk)
        payload = self.confirmation(review, selected_lots=[str(chosen['index'])])
        response = self.client.post(self.url, payload)
        self.assertEqual(response.status_code, 302)
        self.expired.refresh_from_db()
        self.also_expired.refresh_from_db()
        self.second.refresh_from_db()
        self.assertEqual(self.expired.quantity_on_hand, 3)
        self.assertEqual(self.also_expired.quantity_on_hand, 0)
        self.assertEqual(self.second.quantity_in_stock, 4)
        self.assertEqual(ProductLotMovement.objects.get().lot, self.also_expired)

    def test_can_select_another_eligible_lot_outside_the_list_filter(self):
        self.future.expiry_date = date.today() + timedelta(days=20)
        self.future.save(update_fields=['expiry_date', 'updated_at'])
        review = self.review([self.first.pk])
        chosen = next(row for row in review.context['review_rows'] if row['lot_id'] == self.future.pk)
        self.assertTrue(chosen['eligible'])
        self.assertFalse(chosen['selected'])
        self.assertContains(review, 'Outside current expiry filter')
        payload = self.confirmation(review, {self.future.pk: 2}, selected_lots=[str(chosen['index'])])
        self.client.post(self.url, payload)
        self.future.refresh_from_db()
        self.expired.refresh_from_db()
        self.assertEqual(self.future.quantity_on_hand, 3)
        self.assertEqual(self.expired.quantity_on_hand, 3)
        self.assertEqual(ProductLotMovement.objects.get().lot, self.future)

    def test_unchecked_quantities_can_be_omitted(self):
        review = self.review([self.first.pk])
        payload = self.confirmation(review, selected_lots=['0'])
        payload.pop('quantity_1')
        payload.pop('quantity_2')
        self.client.post(self.url, payload)
        self.assertEqual(ProductLotMovement.objects.count(), 1)
        self.assertEqual(ProductLotMovement.objects.get().lot, self.expired)

    def test_invalid_missing_duplicate_or_ineligible_lot_selection_logs_nothing(self):
        review = self.review()
        future = next(row for row in review.context['review_rows'] if row['lot_id'] == self.future.pk)
        for selection in ([], ['bad'], ['999'], ['0', '0'], [str(future['index'])]):
            with self.subTest(selection=selection):
                response = self.client.post(self.url, self.confirmation(review, selected_lots=selection), follow=True)
                self.assertContains(response, 'Nothing was logged.')
                self.assert_no_logging()

    def test_undated_lots_are_shown_but_cannot_be_selected_for_expiry(self):
        undated = ProductLot.objects.create(
            product=self.first, lot_number='UNDATED', quantity_on_hand=1,
        )
        review = self.review([self.first.pk])
        row = next(row for row in review.context['review_rows'] if row['lot_id'] == undated.pk)
        self.assertFalse(row['eligible'])
        self.assertFalse(row['selected'])
        self.assertContains(review, 'No expiry date — cannot log as expired')
        self.client.post(self.url, self.confirmation(review, selected_lots=[str(row['index'])]))
        self.assert_no_logging()

    def test_lot_choices_are_grouped_under_their_product(self):
        review = self.review([self.first.pk])
        self.assertContains(review, 'class="review-product" rowspan="3"')
        self.assertContains(review, 'name="selected_lots" value="0"')
        self.assertContains(review, 'name="selected_lots" value="1"')
        self.assertContains(review, 'Eligible from')
        self.assertContains(review, 'Unchecked lots stay unchanged.')

    def test_bulk_logging_updates_exact_lots_totals_and_audit_history(self):
        review = self.review()
        response = self.client.post(self.url, self.confirmation(review), follow=True)
        self.assertEqual(response.status_code, 200)
        self.first.refresh_from_db()
        self.second.refresh_from_db()
        self.future.refresh_from_db()
        self.assertEqual((self.first.quantity_in_stock, self.first.stock_expired), (5, 5))
        self.assertEqual((self.second.quantity_in_stock, self.second.stock_expired), (0, 4))
        self.assertEqual(self.first.expiry_date, self.future.expiry_date)
        self.assertEqual(self.future.quantity_on_hand, 5)
        self.assertIsNone(self.future.archived_at)
        for lot in (self.expired, self.also_expired, self.second_lot):
            lot.refresh_from_db()
            self.assertEqual(lot.quantity_on_hand, 0)
            self.assertIsNotNone(lot.archived_at)
            self.assertEqual(lot.archived_by, self.user)
        self.assertEqual(StockChange.objects.filter(change_type='expired').count(), 3)
        self.assertEqual(UserAction.objects.filter(action='retire_expired').count(), 3)
        self.assertEqual(set(ProductLotMovement.objects.values_list('lot_number', 'quantity')),
                         {('EXPIRED-A', 3), ('EXPIRED-B', 2), ('SECOND', 4)})
        self.assertTrue(all(change.user == self.user and 'expiry' in change.note
                            for change in StockChange.objects.filter(change_type='expired')))
        self.assertContains(response, 'Logged 9 collected unit(s) across 2 product(s)')

    def test_partial_collection_and_zero_quantity_leave_other_stock_alone(self):
        review = self.review()
        response = self.client.post(self.url, self.confirmation(review, {
            self.expired.pk: 1, self.also_expired.pk: 0, self.second_lot.pk: 0,
        }))
        self.assertEqual(response.status_code, 302)
        self.expired.refresh_from_db()
        self.also_expired.refresh_from_db()
        self.second.refresh_from_db()
        self.assertEqual(self.expired.quantity_on_hand, 2)
        self.assertIsNone(self.expired.archived_at)
        self.assertEqual(self.also_expired.quantity_on_hand, 2)
        self.assertEqual(self.second.quantity_in_stock, 4)
        self.assertEqual(StockChange.objects.get(change_type='expired').quantity, 1)

    def test_legacy_product_collection_archives_unassigned_lot(self):
        legacy = Product.objects.create(
            name='Legacy collected', barcode='COLLECT-LEGACY', category=self.category,
            price=Decimal('5'), quantity_in_stock=2, expiry_date=date.today() - timedelta(days=5),
        )
        review = self.review([legacy.pk])
        self.client.post(self.url, self.confirmation(review))
        legacy.refresh_from_db()
        self.assertEqual(legacy.quantity_in_stock, 0)
        lot = legacy.lots.get(lot_number=ProductLot.UNASSIGNED)
        self.assertIsNotNone(lot.archived_at)
        self.assertEqual(lot.quantity_on_hand, 0)
        self.assertEqual(ProductLotMovement.objects.get().quantity, 2)

    def test_confirmation_is_required_and_invalid_quantities_write_nothing(self):
        review = self.review()
        for extra in ({'collected': ''}, {'quantity_0': '-1'}, {'quantity_0': '1.5'},
                      {'quantity_0': '999'}, {'quantity_0': ''}, {'quantity_0': ['1', '2']}):
            with self.subTest(extra=extra):
                response = self.client.post(self.url, self.confirmation(review, **extra), follow=True)
                self.assertContains(response, 'Nothing was logged.')
                self.assert_no_logging()

    def test_stock_changes_reject_the_entire_batch(self):
        review = self.review()
        self.second_lot.quantity_on_hand = 3
        self.second_lot.save(update_fields=['quantity_on_hand', 'updated_at'])
        response = self.client.post(self.url, self.confirmation(review), follow=True)
        self.assertContains(response, 'changed after review')
        self.first.refresh_from_db()
        self.assertEqual(self.first.quantity_in_stock, 10)
        self.assert_no_logging()

    def test_all_zero_quantities_do_not_create_logs(self):
        review = self.review()
        zeros = {row['lot_id']: 0 for row in review.context['review_rows']}
        response = self.client.post(self.url, self.confirmation(review, zeros), follow=True)
        self.assertContains(response, 'greater than zero')
        self.assert_no_logging()

    def test_collected_total_cannot_exceed_product_stock(self):
        self.first.quantity_in_stock = 1
        self.first.save(update_fields=['quantity_in_stock'])
        review = self.review()
        response = self.client.post(self.url, self.confirmation(review), follow=True)
        self.assertContains(response, 'exceed current product stock')
        self.assert_no_logging()

    def test_product_archived_after_review_rejects_the_batch(self):
        review = self.review()
        from django.utils.timezone import now
        self.second.archived_at = now()
        self.second.save(update_fields=['archived_at'])
        self.client.post(self.url, self.confirmation(review))
        self.assert_no_logging()

    def test_duplicate_selections_are_reviewed_once(self):
        review = self.review([self.first.pk, self.first.pk])
        self.assertEqual(len(review.context['review_rows']), 3)
        self.assertEqual(review.context['product_count'], 1)
        self.assert_no_logging()

    def test_lot_identity_change_with_same_quantity_rejects_confirmation(self):
        review = self.review()
        self.expired.lot_number = 'RENAMED'
        self.expired.save(update_fields=['lot_number', 'updated_at'])
        self.client.post(self.url, self.confirmation(review))
        self.assert_no_logging()

    def test_failure_during_later_retirement_rolls_back_earlier_changes(self):
        review = self.review()
        original = ExpiredProductView._retire_selected_lot
        calls = []
        def retire_then_fail(product, lot, qty, user):
            calls.append(product.pk)
            if len(calls) == 2:
                raise ValidationError('Stock changed during collection.')
            return original(product, lot, qty, user)
        with patch.object(ExpiredProductView, '_retire_selected_lot', side_effect=retire_then_fail):
            response = self.client.post(self.url, self.confirmation(review), follow=True)
        self.assertContains(response, 'Nothing was logged.')
        self.first.refresh_from_db()
        self.expired.refresh_from_db()
        self.assertEqual(self.first.quantity_in_stock, 10)
        self.assertEqual(self.expired.quantity_on_hand, 3)
        self.assertIsNone(self.expired.archived_at)
        self.assert_no_logging()

    def test_repeated_confirmation_does_not_log_twice(self):
        review = self.review()
        payload = self.confirmation(review, {self.expired.pk: 1})
        self.client.post(self.url, payload)
        response = self.client.post(self.url, payload, follow=True)
        self.assertContains(response, 'Nothing was logged.')
        self.assertEqual(StockChange.objects.filter(change_type='expired').count(), 3)
        self.expired.refresh_from_db()
        self.assertEqual(self.expired.quantity_on_hand, 2)

    def test_tampered_and_expired_review_tokens_do_not_log(self):
        review = self.review()
        response = self.client.post(self.url, self.confirmation(review, review_token='invalid'), follow=True)
        self.assertContains(response, 'review expired or is invalid')
        with patch('django.core.signing.time.time', return_value=time.time() + 3601):
            self.client.post(self.url, self.confirmation(review))
        self.assert_no_logging()

    def test_another_user_cannot_confirm_someone_elses_review(self):
        review = self.review()
        other = User.objects.create_user(username='another-collector', password='test-pass')
        self.client.force_login(other)
        self.client.post(self.url, self.confirmation(review))
        self.assert_no_logging()

    def test_review_preserves_filters_and_one_month_eligibility(self):
        cutoff = date.today() + relativedelta(months=1)
        self.second_lot.expiry_date = cutoff
        self.second_lot.save(update_fields=['expiry_date', 'updated_at'])
        filters = {'date_filter': '3_months', 'name_query': 'Collected', 'sort': '-name'}
        review = self.review([self.second.pk], **filters)
        self.assertEqual([row['lot_id'] for row in review.context['review_rows']], [self.second_lot.pk])
        self.assertEqual(parse_qs(urlsplit(review.context['return_url']).query),
                         {'mode': ['view'], **{key: [value] for key, value in filters.items()}})
        rejected = self.review([self.first.pk], **filters)
        self.assertEqual(rejected.status_code, 302)
        self.assert_no_logging()

    def test_get_does_not_log_and_post_requires_authentication_and_csrf(self):
        review = self.review()
        payload = self.confirmation(review)
        self.client.get(self.url, payload)
        self.client.logout()
        self.assertEqual(self.client.post(self.url, payload).status_code, 302)
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(self.user)
        self.assertEqual(csrf_client.post(self.url, payload).status_code, 403)
        self.assert_no_logging()

    def test_empty_invalid_or_archived_selection_does_not_log(self):
        for ids in ([], ['invalid'], ['9' * 100], [999999]):
            with self.subTest(ids=ids):
                self.assertEqual(self.review(ids).status_code, 302)
        self.assert_no_logging()
