"""Manual ordering-list additions must never manufacture a purchase or sale."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal
from threading import Barrier
from time import time
from unittest import skipUnless
from unittest.mock import patch

from django.contrib.auth.models import User
from django.db import close_old_connections, connection
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from . import reporting
from .mckesson import collect_order_items
from .mixins import PASSKEY_SESSION_KEY
from .models import (
    Category,
    CheckoutOrder,
    CheckoutOrderItem,
    Order,
    OrderDetail,
    Product,
    ProductLot,
    ProductLotMovement,
    RecentlyPurchasedProduct,
    StockChange,
    UserAction,
    UserSession,
)


class DashboardReorderHelpers:
    def make_client(self, user=None, *, csrf=False, unlocked=False):
        client = Client(enforce_csrf_checks=csrf)
        user = user or self.user
        client.force_login(user)
        UserSession.objects.get_or_create(
            user=user, session_key=client.session.session_key,
        )
        if unlocked:
            session = client.session
            session[PASSKEY_SESSION_KEY] = time()
            session.save()
        return client

    def make_product(self, name='Manual reorder product', barcode='0123456789012', **kwargs):
        return Product.objects.create(
            name=name, barcode=barcode, price=Decimal('12.50'),
            price_per_unit=Decimal('5.00'), category=self.category,
            **{'quantity_in_stock': 1, **kwargs},
        )

    def add(self, product=None, quantity=3, *, client=None, status=200):
        response = (client or self.client).post(
            reverse('dashboard_reorder_add'),
            {'product_id': (product or self.product).pk, 'quantity': quantity},
            content_type='application/json',
        )
        self.assertEqual(response.status_code, status, response.content[:500])
        return response.json()

    @staticmethod
    def inventory_and_transaction_snapshot():
        models = (
            Product, ProductLot, ProductLotMovement, StockChange,
            Order, OrderDetail, CheckoutOrder, CheckoutOrderItem,
        )
        return {
            model.__name__: list(model.objects.order_by('pk').values())
            for model in models
        }


@override_settings(AXES_ENABLED=False)
class DashboardReorderAddTests(DashboardReorderHelpers, TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='reorder-staff', is_staff=True)
        self.other = User.objects.create_user(username='reorder-other', is_staff=True)
        self.regular = User.objects.create_user(username='reorder-regular')
        self.category = Category.objects.create(name='Health', low_stock_threshold=3)
        self.product = self.make_product()
        self.client = self.make_client()

    def test_creates_one_manual_row_and_audits_the_actor(self):
        payload = self.add(quantity=4)

        self.assertTrue(payload['ok'])
        self.assertFalse(payload['already_added'])
        self.assertEqual(payload['product_id'], self.product.pk)
        self.assertEqual(payload['quantity'], 4)
        self.assertTrue(payload['message'])
        row = RecentlyPurchasedProduct.objects.get(product=self.product)
        self.assertEqual(payload['recent_id'], row.pk)
        self.assertEqual(row.quantity, 0)
        self.assertEqual(row.manual_order_quantity, 4)
        self.assertIsNone(row.archived_at)
        audit = UserAction.objects.get(action='add_recently_purchased')
        self.assertEqual(audit.user, self.user)
        self.assertIn(self.product.name, audit.detail)
        self.assertIn('4', audit.detail)

    def test_manual_add_does_not_change_inventory_history_transactions_or_reports(self):
        ProductLot.objects.create(
            product=self.product, lot_number='UNCHANGED', quantity_on_hand=1,
            expiry_date=timezone.localdate() + timedelta(days=90),
        )
        order = Order.objects.create(user=self.user, submitted=True, total_price='12.50')
        detail = OrderDetail.objects.create(
            order=order, product=self.product, quantity=1, price=self.product.price,
            product_name=self.product.name, product_barcode=self.product.barcode,
        )
        StockChange.objects.create(
            product=self.product, user=self.user, quantity=1,
            change_type='checkout', order_detail=detail,
        )
        snapshot = self.inventory_and_transaction_snapshot()
        digest = reporting.daily_digest()
        totals = reporting.dashboard_kpis()
        totals.pop('reorder_suggestions')

        self.add(quantity=7)

        self.assertEqual(self.inventory_and_transaction_snapshot(), snapshot)
        self.assertEqual(reporting.daily_digest(), digest)
        updated_totals = reporting.dashboard_kpis()
        updated_totals.pop('reorder_suggestions')
        self.assertEqual(updated_totals, totals)
        self.assertEqual(RecentlyPurchasedProduct.objects.get().quantity, 0)

    def test_duplicate_click_preserves_quantity_date_and_creates_no_second_audit(self):
        original = self.add(quantity=2)
        before = RecentlyPurchasedProduct.objects.values().get(pk=original['recent_id'])

        duplicate = self.add(quantity=99)

        self.assertTrue(duplicate['already_added'])
        self.assertEqual(duplicate['recent_id'], original['recent_id'])
        self.assertEqual(duplicate['quantity'], 2)
        self.assertEqual(RecentlyPurchasedProduct.objects.values().get(), before)
        self.assertEqual(UserAction.objects.filter(action='add_recently_purchased').count(), 1)

    def test_existing_fulfilled_entry_is_not_overwritten_or_recounted(self):
        row = RecentlyPurchasedProduct.objects.create(product=self.product, quantity=5)
        old_date = timezone.now() - timedelta(days=12)
        RecentlyPurchasedProduct.objects.filter(pk=row.pk).update(order_date=old_date)
        before = RecentlyPurchasedProduct.objects.values().get(pk=row.pk)

        payload = self.add(quantity=9)

        self.assertTrue(payload['already_added'])
        self.assertEqual(payload['recent_id'], row.pk)
        self.assertEqual(RecentlyPurchasedProduct.objects.values().get(pk=row.pk), before)
        self.assertFalse(UserAction.objects.filter(action='add_recently_purchased').exists())

    def test_archived_generation_stays_archived_and_a_new_active_row_is_created(self):
        archived = RecentlyPurchasedProduct.objects.create(
            product=self.product, quantity=6, manual_order_quantity=11,
            archived_at=timezone.now() - timedelta(days=1),
            archived_by=self.user, archive_reason='Previous ordering cycle completed',
        )
        archived_before = RecentlyPurchasedProduct.objects.values().get(pk=archived.pk)

        payload = self.add(quantity=3)

        self.assertFalse(payload['already_added'])
        self.assertNotEqual(payload['recent_id'], archived.pk)
        self.assertEqual(RecentlyPurchasedProduct.objects.values().get(pk=archived.pk), archived_before)
        self.assertEqual(RecentlyPurchasedProduct.objects.filter(archived_at__isnull=True).count(), 1)
        self.assertEqual(RecentlyPurchasedProduct.objects.count(), 2)

    def test_shared_list_is_visible_and_idempotent_across_staff_accounts(self):
        first = self.add(quantity=6)
        other_client = self.make_client(self.other)

        second = self.add(quantity=8, client=other_client)
        response = other_client.get(reverse('low_stock'))

        self.assertTrue(second['already_added'])
        self.assertEqual(second['recent_id'], first['recent_id'])
        self.assertEqual(second['quantity'], 6)
        self.assertEqual(response.status_code, 200)
        rows = response.context['recently_purchased']
        self.assertEqual([row.product_id for row in rows], [self.product.pk])
        self.assertEqual(rows[0].quantity, 0)
        self.assertEqual(rows[0].bought_60d, 0)
        self.assertEqual(rows[0].manual_order_quantity, 6)
        self.assertContains(response, 'Manual order')
        self.assertContains(response, self.product.barcode)

    def test_ajax_recently_purchased_filter_keeps_manual_entries_visible(self):
        self.add(quantity=5)

        response = self.client.get(
            reverse('low_stock'), {'q': self.product.barcode},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['count'], 1)
        self.assertIn(self.product.name, payload['html'])
        self.assertIn('Manual order', payload['html'])

    def test_anonymous_and_locked_accounts_cannot_add(self):
        self.add(client=Client(), status=401)
        denied = self.add(client=self.make_client(self.regular), status=403)
        self.assertFalse(denied['ok'])
        self.assertTrue(denied['error'])
        self.assertFalse(RecentlyPurchasedProduct.objects.exists())

    def test_passkey_unlocked_regular_account_can_add(self):
        payload = self.add(client=self.make_client(self.regular, unlocked=True))

        self.assertTrue(payload['ok'])
        self.assertEqual(UserAction.objects.get(action='add_recently_purchased').user, self.regular)

    def test_csrf_is_required_and_valid_same_origin_token_is_accepted(self):
        client = self.make_client(csrf=True)
        request = {'product_id': self.product.pk, 'quantity': 3}
        url = reverse('dashboard_reorder_add')
        denied = client.post(url, request, content_type='application/json')
        self.assertEqual(denied.status_code, 403)
        self.assertFalse(RecentlyPurchasedProduct.objects.exists())

        page = client.get(reverse('dashboard'))
        self.assertEqual(page.status_code, 200)
        accepted = client.post(
            url, request, content_type='application/json',
            HTTP_X_CSRFTOKEN=client.cookies['csrftoken'].value,
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertTrue(accepted.json()['ok'])

    def test_get_never_adds_and_success_response_is_not_cacheable(self):
        response = self.client.get(reverse('dashboard_reorder_add'))
        self.assertEqual(response.status_code, 405)
        self.assertFalse(RecentlyPurchasedProduct.objects.exists())

        response = self.client.post(
            reverse('dashboard_reorder_add'),
            {'product_id': self.product.pk, 'quantity': 1},
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('no-store', response['Cache-Control'])

    def test_invalid_body_product_ids_and_quantities_make_no_changes(self):
        valid = {'product_id': self.product.pk, 'quantity': 1}
        invalid = [[], None, {}, {'quantity': 1}, {'product_id': self.product.pk}]
        for product_id in (True, False, 0, -1, 1.5, str(self.product.pk), 2147483648):
            invalid.append({**valid, 'product_id': product_id})
        for quantity in (True, False, None, 0, -1, 1.5, '3', 10000, [], {}):
            invalid.append({**valid, 'quantity': quantity})
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post(
                    reverse('dashboard_reorder_add'), payload,
                    content_type='application/json',
                )
                self.assertEqual(response.status_code, 400, response.content[:300])
                self.assertFalse(response.json()['ok'])
                self.assertTrue(response.json()['error'])
        malformed = self.client.post(
            reverse('dashboard_reorder_add'), '{', content_type='application/json',
        )
        self.assertEqual(malformed.status_code, 400)
        self.assertFalse(RecentlyPurchasedProduct.objects.exists())
        self.assertFalse(UserAction.objects.filter(action='add_recently_purchased').exists())

    def test_quantity_limits_are_inclusive(self):
        second = self.make_product(name='Maximum manual quantity', barcode='9876543210987')
        self.assertEqual(self.add(quantity=1)['quantity'], 1)
        self.assertEqual(self.add(second, quantity=9999)['quantity'], 9999)

    def test_missing_inactive_and_archived_products_are_rejected(self):
        inactive = self.make_product(name='Inactive', barcode='00222', status=False)
        archived = self.make_product(name='Archived', barcode='00333', archived_at=timezone.now())
        missing = Product(product_id=2147483647)
        for product in (inactive, archived, missing):
            with self.subTest(product=product.pk):
                result = self.add(product, status=404)
                self.assertFalse(result['ok'])
        self.assertFalse(RecentlyPurchasedProduct.objects.exists())

    def test_reorder_payload_reports_only_active_list_membership(self):
        other = self.make_product(name='Still available to add', barcode='00444')
        RecentlyPurchasedProduct.objects.create(
            product=other, quantity=1, archived_at=timezone.now(),
        )
        before = {row['product_id']: row for row in reporting.reorder_suggestions()}
        self.assertFalse(before[self.product.pk]['in_recently_purchased'])
        self.assertFalse(before[other.pk]['in_recently_purchased'])

        self.add()
        response = self.client.get(reverse('dashboard_expand'), {'section': 'reorder'})

        self.assertEqual(response.status_code, 200)
        rows = {row['product_id']: row for row in response.json()['items']}
        self.assertTrue(rows[self.product.pk]['in_recently_purchased'])
        self.assertFalse(rows[other.pk]['in_recently_purchased'])
        for field in ('suggested_qty', 'urgency', 'quantity_in_stock', 'threshold'):
            self.assertEqual(rows[self.product.pk][field], before[self.product.pk][field])


@override_settings(AXES_ENABLED=False)
class DashboardManualOrderingIntegrationTests(DashboardReorderHelpers, TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='manual-ordering', is_staff=True)
        self.category = Category.objects.create(name='Ordering')
        self.product = self.make_product()
        self.client = self.make_client()

    def test_manual_quantity_is_ordered_in_both_modes_even_without_any_sales(self):
        self.add(quantity=7)
        expected = [{
            'product_id': self.product.pk, 'name': self.product.name,
            'barcode': self.product.barcode, 'quantity': 7,
        }]
        for mode in ('predicted', 'sold'):
            with self.subTest(mode=mode), patch(
                'app.mckesson.predicted_quantities', return_value={self.product.pk: 25},
            ):
                items, skipped = collect_order_items(qty_mode=mode)
            self.assertEqual(items, expected)
            self.assertEqual(skipped, [])
        self.assertFalse(StockChange.objects.exists())
        self.assertEqual(RecentlyPurchasedProduct.objects.get().quantity, 0)

    def test_existing_sales_entries_keep_prediction_fallback_and_sold_quantities(self):
        self.add(quantity=7)
        sold = self.make_product(name='Actual sales', barcode='00555')
        fallback = self.make_product(name='Prediction fallback', barcode='00666')
        RecentlyPurchasedProduct.objects.create(product=sold, quantity=3)
        RecentlyPurchasedProduct.objects.create(product=fallback, quantity=2)
        with patch('app.mckesson.predicted_quantities', return_value={sold.pk: 10, fallback.pk: 0}):
            items, skipped = collect_order_items(qty_mode='predicted')
        self.assertEqual({item['product_id']: item['quantity'] for item in items}, {
            self.product.pk: 7, sold.pk: 10, fallback.pk: 2,
        })
        self.assertEqual(skipped, [])
        items, skipped = collect_order_items(qty_mode='sold')
        self.assertEqual({item['product_id']: item['quantity'] for item in items}, {
            self.product.pk: 7, sold.pk: 3, fallback.pk: 2,
        })
        self.assertEqual(skipped, [])

    def test_later_real_purchase_retains_manual_quantity_and_records_only_actual_units(self):
        self.add(quantity=7)
        recent = RecentlyPurchasedProduct.objects.get(product=self.product)
        ProductLot.objects.create(
            product=self.product, lot_number='FOR-SALE', quantity_on_hand=1,
        )

        added = self.client.post(
            reverse('add_product_by_id', args=[self.product.pk]), {'quantity': '1'},
        )
        self.assertEqual(added.status_code, 302)
        submitted = self.client.post(reverse('submit_order'))

        self.assertEqual(submitted.status_code, 302)
        recent.refresh_from_db()
        self.product.refresh_from_db()
        self.assertEqual(RecentlyPurchasedProduct.objects.filter(archived_at__isnull=True).count(), 1)
        self.assertEqual(recent.quantity, 1)
        self.assertEqual(recent.manual_order_quantity, 7)
        self.assertEqual(self.product.stock_sold, 1)
        self.assertEqual(self.product.quantity_in_stock, 0)
        self.assertEqual(OrderDetail.objects.get(product=self.product).quantity, 1)
        items, skipped = collect_order_items(qty_mode='sold')
        self.assertEqual(items[0]['quantity'], 7)
        self.assertEqual(skipped, [])

    def test_archived_requests_are_excluded_and_category_and_barcode_rules_still_apply(self):
        self.add(quantity=4)
        no_barcode = self.make_product(name='Needs a barcode', barcode=None)
        self.add(no_barcode, quantity=2)
        archived_product = self.make_product(name='Previous cycle', barcode='00777')
        RecentlyPurchasedProduct.objects.create(
            product=archived_product, manual_order_quantity=9, archived_at=timezone.now(),
        )

        items, skipped = collect_order_items(qty_mode='sold')
        self.assertEqual([item['product_id'] for item in items], [self.product.pk])
        self.assertEqual([(row['product_id'], row['quantity'], row['reason']) for row in skipped], [
            (no_barcode.pk, 2, 'no barcode on product'),
        ])
        items, skipped = collect_order_items(qty_mode='sold', exclude_category_ids=[self.category.pk])
        self.assertEqual(items, [])
        self.assertEqual({row['product_id'] for row in skipped}, {self.product.pk, no_barcode.pk})
        self.assertTrue(all(row['reason'] == 'excluded category' for row in skipped))


@skipUnless(connection.vendor == 'postgresql', 'Requires PostgreSQL row locking.')
@override_settings(AXES_ENABLED=False)
class DashboardReorderConcurrencyTests(DashboardReorderHelpers, TransactionTestCase):
    def test_simultaneous_clicks_from_two_accounts_create_one_shared_request(self):
        self.user = User.objects.create_user(username='reorder-concurrent-1', is_staff=True)
        other = User.objects.create_user(username='reorder-concurrent-2', is_staff=True)
        self.category = Category.objects.create(name='Concurrent reorder')
        self.product = self.make_product()
        clients = [(self.make_client(self.user), 3), (self.make_client(other), 7)]
        barrier = Barrier(2)

        def add_request(args):
            client, quantity = args
            close_old_connections()
            try:
                barrier.wait(timeout=10)
                response = client.post(
                    reverse('dashboard_reorder_add'),
                    {'product_id': self.product.pk, 'quantity': quantity},
                    content_type='application/json',
                )
                return response.status_code, response.json()
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(add_request, clients))

        self.assertEqual([status for status, _ in results], [200, 200])
        self.assertEqual(RecentlyPurchasedProduct.objects.count(), 1)
        row = RecentlyPurchasedProduct.objects.get()
        self.assertEqual(row.quantity, 0)
        self.assertIn(row.manual_order_quantity, (3, 7))
        self.assertEqual({result['recent_id'] for _, result in results}, {row.pk})
        self.assertEqual({result['quantity'] for _, result in results}, {row.manual_order_quantity})
        self.assertEqual(sorted(result['already_added'] for _, result in results), [False, True])
        self.assertEqual(UserAction.objects.filter(action='add_recently_purchased').count(), 1)
        self.assertFalse(StockChange.objects.exists())
        self.assertFalse(Order.objects.exists())
        self.assertFalse(CheckoutOrder.objects.exists())
