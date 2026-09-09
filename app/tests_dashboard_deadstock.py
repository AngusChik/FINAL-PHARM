"""Dashboard suggestions rotate independently of pharmacy-wide dismissals."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from threading import Barrier, BrokenBarrierError
from unittest import skipUnless
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth.models import User
from django.db import close_old_connections, connection
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from . import dashboard_deadstock, reporting
from .models import (
    Category,
    DashboardDeadStockDismissal,
    Product,
    ProductLot,
    ProductLotMovement,
    StockChange,
    UserSession,
)


class DeadStockTestHelpers:
    def make_client(self, user=None, *, csrf=False):
        client = Client(enforce_csrf_checks=csrf)
        user = user or self.user
        client.force_login(user)
        UserSession.objects.get_or_create(
            user=user, session_key=client.session.session_key,
        )
        return client

    def make_products(self, count, *, category=None, prefix='Product', stock=10):
        return Product.objects.bulk_create([
            Product(
                name=f'{prefix} {index:03d}',
                price=Decimal('4.25'),
                price_per_unit=Decimal('2.00'),
                quantity_in_stock=stock,
                category=category or self.category,
            )
            for index in range(count)
        ])

    def post(self, action='next', *, client=None, status=200, **payload):
        response = (client or self.client).post(
            reverse('dashboard_deadstock'),
            {'action': action, **payload},
            content_type='application/json',
        )
        self.assertEqual(response.status_code, status, response.content[:500])
        return response.json()

    @staticmethod
    def ids(payload):
        return [row['product_id'] for row in payload['items']]

    def record_sale(self, product, days_ago):
        sale = StockChange.objects.create(
            product=product, user=self.user, change_type='checkout', quantity=1,
        )
        stamp = timezone.make_aware(datetime.combine(
            date.today() - timedelta(days=days_ago), time(12),
        ))
        StockChange.objects.filter(pk=sale.pk).update(timestamp=stamp)
        return sale


@override_settings(AXES_ENABLED=False)
class DashboardDeadStockEndpointTests(DeadStockTestHelpers, TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='deadstock-staff', is_staff=True)
        self.other = User.objects.create_user(username='deadstock-pharmacy-user')
        self.category = Category.objects.create(name='Health')
        self.client = self.make_client()

    def test_login_is_required_for_listing_and_mutation(self):
        client = Client()
        url = reverse('dashboard_deadstock')
        for response in (
            client.get(url),
            client.post(url, {'action': 'next'}, content_type='application/json'),
        ):
            self.assertEqual(response.status_code, 302)
            self.assertTrue(response.url.startswith(reverse('login')))
        self.assertFalse(DashboardDeadStockDismissal.objects.exists())

    def test_regular_authenticated_pharmacy_user_can_dismiss(self):
        product = self.make_products(1)[0]
        payload = self.post(
            'dismiss', client=self.make_client(self.other), product_id=product.pk,
        )
        self.assertTrue(payload['dismissed'])
        self.assertEqual(
            DashboardDeadStockDismissal.objects.get(product=product).dismissed_by,
            self.other,
        )

    def test_mutations_require_csrf_and_accept_a_valid_token(self):
        product = self.make_products(1)[0]
        client = self.make_client(csrf=True)
        url = reverse('dashboard_deadstock')
        payload = {'action': 'dismiss', 'product_id': product.pk}
        rejected = client.post(url, payload, content_type='application/json')
        self.assertEqual(rejected.status_code, 403)
        self.assertFalse(DashboardDeadStockDismissal.objects.exists())

        # A same-origin page supplies Django's normal CSRF cookie.
        page = client.get(reverse('dashboard'))
        self.assertEqual(page.status_code, 200)
        accepted = client.post(
            url, payload, content_type='application/json',
            HTTP_X_CSRFTOKEN=client.cookies['csrftoken'].value,
        )
        self.assertEqual(accepted.status_code, 200)

    def test_invalid_requests_do_not_create_dismissals(self):
        product = self.make_products(1)[0]
        invalid = [
            [], None, {'action': 'delete'},
            {'action': 'dismiss'},
            {'action': 'dismiss', 'product_id': 0},
            {'action': 'dismiss', 'product_id': -1},
            {'action': 'dismiss', 'product_id': True},
            {'action': 'dismiss', 'product_id': str(product.pk)},
            {'action': 'next', 'exclude_snacks': 'false'},
            {'action': 'next', 'exclude_braces': 1},
            {'action': 'next', 'current_ids': 'invalid'},
            {'action': 'next', 'current_ids': [False]},
            {'action': 'next', 'current_ids': ['1']},
            {'action': 'next', 'current_ids': [-1]},
            {'action': 'next', 'current_ids': list(range(1, 10))},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post(
                    reverse('dashboard_deadstock'), payload,
                    content_type='application/json',
                )
                self.assertEqual(response.status_code, 400, response.content[:300])
        malformed = self.client.post(
            reverse('dashboard_deadstock'), '{', content_type='application/json',
        )
        self.assertEqual(malformed.status_code, 400)
        self.post('dismiss', product_id=2147483647, status=404)
        self.assertFalse(DashboardDeadStockDismissal.objects.exists())

    def test_responses_are_not_cached_and_rows_preserve_product_text(self):
        product = self.make_products(1)[0]
        product.name = '<img src=x onerror=alert(1)> Tablets & Supplements'
        product.save(update_fields=['name'])
        response = self.client.post(
            reverse('dashboard_deadstock'), {'action': 'next'},
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('no-store', response['Cache-Control'])
        self.assertIn('private', response['Cache-Control'])
        row = response.json()['items'][0]
        self.assertEqual(row['name'], product.name)
        self.assertEqual(row['quantity_in_stock'], 10)
        self.assertEqual(Decimal(str(row['capital_tied'])), Decimal('42.5'))
        self.assertEqual(row['days_since_sale'], 'Never')

    def test_dismissal_is_shared_for_thirty_days_and_retries_do_not_extend_it(self):
        product = self.make_products(1)[0]
        timestamp = timezone.now()
        with patch('app.dashboard_deadstock.timezone.now', return_value=timestamp):
            result = self.post('dismiss', product_id=product.pk)
        dismissal = DashboardDeadStockDismissal.objects.get(product=product)
        self.assertEqual(dismissal.dismissed_at, timestamp)
        self.assertEqual(dismissal.expires_at, timestamp + timedelta(days=30))
        self.assertEqual(dismissal.dismissed_by, self.user)
        self.assertEqual(result['available_count'], 0)
        self.assertEqual(result['total_count'], 1)

        other_client = self.make_client(self.other)
        self.assertEqual(self.ids(self.post(client=other_client)), [])
        with patch(
            'app.dashboard_deadstock.timezone.now',
            return_value=timestamp + timedelta(hours=1),
        ):
            retried = self.post('dismiss', client=other_client, product_id=product.pk)
        dismissal.refresh_from_db()
        self.assertEqual(retried['expires_at'], result['expires_at'])
        self.assertEqual(dismissal.dismissed_at, timestamp)
        self.assertEqual(dismissal.expires_at, timestamp + timedelta(days=30))
        self.assertEqual(dismissal.dismissed_by, self.user)
        self.assertEqual(DashboardDeadStockDismissal.objects.count(), 1)

    def test_restore_is_shared_and_allows_a_new_thirty_day_dismissal(self):
        product = self.make_products(1)[0]
        self.post('dismiss', product_id=product.pk)
        other_client = self.make_client(self.other)
        restored = self.post('restore', client=other_client, product_id=product.pk)
        self.assertFalse(restored['dismissed'])
        self.assertIn(product.pk, self.ids(restored))
        self.assertEqual(self.ids(self.post()), [product.pk])
        self.assertEqual(self.client.get(reverse('dashboard_deadstock')).json()['count'], 0)

        later = timezone.now() + timedelta(hours=1)
        with patch('app.dashboard_deadstock.timezone.now', return_value=later):
            self.post('dismiss', client=other_client, product_id=product.pk)
        dismissal = DashboardDeadStockDismissal.objects.get(product=product)
        self.assertEqual(dismissal.dismissed_by, self.other)
        self.assertEqual(dismissal.dismissed_at, later)
        self.assertEqual(dismissal.expires_at, later + timedelta(days=30))

    def test_dismissal_expires_at_the_exact_boundary_without_a_scheduled_job(self):
        product = self.make_products(1)[0]
        expires = timezone.now()
        DashboardDeadStockDismissal.objects.create(
            product=product, dismissed_by=self.user,
            dismissed_at=expires - timedelta(days=30), expires_at=expires,
        )
        with patch(
            'app.dashboard_deadstock.timezone.now',
            return_value=expires - timedelta(microseconds=1),
        ):
            self.assertEqual(self.ids(self.post()), [])
            self.assertEqual(self.client.get(reverse('dashboard_deadstock')).json()['count'], 1)
        with patch('app.dashboard_deadstock.timezone.now', return_value=expires):
            self.assertEqual(self.ids(self.post()), [product.pk])
            self.assertEqual(self.client.get(reverse('dashboard_deadstock')).json()['count'], 0)

    def test_expired_dismissal_does_not_make_a_recently_sold_product_eligible(self):
        product = self.make_products(1)[0]
        self.record_sale(product, 1)
        timestamp = timezone.now()
        DashboardDeadStockDismissal.objects.create(
            product=product, dismissed_by=self.user,
            dismissed_at=timestamp - timedelta(days=31),
            expires_at=timestamp - timedelta(days=1),
        )
        self.assertEqual(self.ids(self.post()), [])

    def test_dismissal_replaces_only_the_selected_row_and_preserves_remaining_order(self):
        self.make_products(16)
        first = self.ids(self.post())
        dismissed_id = first[3]
        result = self.post('dismiss', product_id=dismissed_id, current_ids=first)
        self.assertEqual(self.ids(result)[:3], first[:3])
        self.assertEqual(self.ids(result)[4:], first[4:])
        self.assertEqual(len(self.ids(result)), 8)
        self.assertNotIn(self.ids(result)[3], first)
        restored = self.post(
            'restore', product_id=dismissed_id, current_ids=self.ids(result),
        )
        self.assertEqual(self.ids(restored), self.ids(result))

    def test_retained_rows_are_revalidated_and_deduplicated_before_refill(self):
        self.make_products(16)
        first = self.ids(self.post())
        Product.objects.filter(pk=first[0]).update(quantity_in_stock=0)
        self.post('dismiss', client=self.make_client(self.other), product_id=first[1])
        retained = [first[0], first[1], first[2], first[2], 2147483647, *first[3:6]]
        result = self.post('dismiss', product_id=first[6], current_ids=retained)
        ids = self.ids(result)
        self.assertEqual([pk for pk in ids if pk in first[2:6]], first[2:6])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 8)
        self.assertTrue(set(ids).isdisjoint({first[0], first[1], first[6], 2147483647}))

    def test_dismiss_and_restore_do_not_change_stock_audit_or_report_totals(self):
        product = self.make_products(1)[0]
        self.record_sale(product, 90)
        ProductLot.objects.create(product=product, lot_number='LOT-A', quantity_on_hand=10)
        before_product = Product.objects.values().get(pk=product.pk)
        before_lots = list(ProductLot.objects.values())
        before_changes = list(StockChange.objects.values())
        before_movements = list(ProductLotMovement.objects.values())
        before_report = reporting.daily_digest()['dead_stock']
        before_total = reporting.dashboard_kpis()['dead_stock_count']

        self.post('dismiss', product_id=product.pk)
        self.assertEqual(reporting.daily_digest()['dead_stock'], before_report)
        self.assertEqual(reporting.dashboard_kpis()['dead_stock_count'], before_total)
        self.post('restore', product_id=product.pk)

        self.assertEqual(Product.objects.values().get(pk=product.pk), before_product)
        self.assertEqual(list(ProductLot.objects.values()), before_lots)
        self.assertEqual(list(StockChange.objects.values()), before_changes)
        self.assertEqual(list(ProductLotMovement.objects.values()), before_movements)
        self.assertEqual(reporting.daily_digest()['dead_stock'], before_report)

    def test_dismissed_listing_is_paginated_and_includes_now_ineligible_products(self):
        products = self.make_products(28)
        timestamp = timezone.now()
        DashboardDeadStockDismissal.objects.bulk_create([
            DashboardDeadStockDismissal(
                product=product, dismissed_by=self.user,
                dismissed_at=timestamp, expires_at=timestamp + timedelta(days=30),
            )
            for product in products
        ])
        Product.objects.filter(pk=products[0].pk).update(quantity_in_stock=0)
        Product.objects.filter(pk=products[1].pk).update(archived_at=timestamp)
        pages = [
            self.client.get(reverse('dashboard_deadstock'), {'page': page}).json()
            for page in (1, 2)
        ]
        self.assertEqual([len(page['items']) for page in pages], [25, 3])
        self.assertEqual(pages[0]['count'], 28)
        self.assertTrue(pages[0]['has_next'])
        self.assertFalse(pages[1]['has_next'])
        all_rows = pages[0]['items'] + pages[1]['items']
        self.assertEqual({row['product_id'] for row in all_rows}, {p.pk for p in products})
        self.assertTrue(all(row['dismissed'] and row['expires_at'] for row in all_rows))
        # An archived product is still restorable from this list.
        self.post('restore', product_id=products[1].pk)
        self.assertEqual(self.client.get(reverse('dashboard_deadstock')).json()['count'], 27)

    def test_expanded_full_list_keeps_dismissed_items_with_restore_metadata(self):
        products = self.make_products(3)
        dismissed = self.post('dismiss', product_id=products[1].pk)
        response = self.client.get(reverse('dashboard_expand'), {'section': 'deadstock'})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['count'], 3)
        self.assertEqual(set(self.ids(payload)), {p.pk for p in products})
        row = next(row for row in payload['items'] if row['product_id'] == products[1].pk)
        self.assertTrue(row['dismissed'])
        self.assertEqual(row['expires_at'], dismissed['expires_at'])


@override_settings(AXES_ENABLED=False)
class DashboardDeadStockRotationTests(DeadStockTestHelpers, TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='rotation-staff', is_staff=True)
        self.category = Category.objects.create(name='Health')
        self.client = self.make_client()

    def test_small_pools_fill_without_duplicates_and_minimize_successive_overlap(self):
        for size in (0, 1, 8, 9, 16):
            with self.subTest(size=size):
                products = self.make_products(size, prefix=f'Pool {size}')
                client = self.make_client()
                batches = [self.ids(self.post(client=client)) for _ in range(3)]
                for batch in batches:
                    self.assertEqual(len(batch), min(8, size))
                    self.assertEqual(len(batch), len(set(batch)))
                    self.assertTrue(set(batch).issubset({p.pk for p in products}))
                for previous, current in zip(batches, batches[1:]):
                    minimum_overlap = max(0, 2 * min(8, size) - size)
                    self.assertEqual(len(set(previous) & set(current)), minimum_overlap)
                if size:
                    first_cycle = (batches[0] + batches[1])[:size]
                    self.assertEqual(len(set(first_cycle)), size)
                Product.all_objects.filter(pk__in=[p.pk for p in products]).delete()

    def test_rotation_visits_the_complete_pool_beyond_the_expanded_list_limit(self):
        products = self.make_products(312)
        all_ids = {product.pk for product in products}
        seen = set()
        for _ in range(39):
            payload = self.post()
            batch = self.ids(payload)
            self.assertEqual(payload['available_count'], 312)
            self.assertEqual(len(batch), 8)
            self.assertTrue(seen.isdisjoint(batch))
            seen.update(batch)
        self.assertEqual(seen, all_ids)
        self.assertEqual(len(self.ids(self.post())), 8)

    def test_filters_and_shared_dismissals_apply_before_eight_row_selection(self):
        snacks = Category.objects.create(name='Snacks')
        braces = Category.objects.create(name='Braces')
        self.make_products(305, category=snacks, stock=100)
        self.make_products(10, category=braces, stock=90)
        health = self.make_products(9, stock=1)
        self.post('dismiss', product_id=health[0].pk)
        payload = self.post(exclude_snacks=True, exclude_braces=True)
        self.assertEqual(set(self.ids(payload)), {p.pk for p in health[1:]})
        self.assertEqual(payload['available_count'], 8)
        self.assertEqual(payload['total_count'], 324)
        self.assertEqual(payload['filtered_count'], 315)
        self.assertEqual(payload['dismissed_eligible_count'], 1)

    def test_each_filter_combination_has_an_independent_queue(self):
        health = self.make_products(16)
        self.make_products(16, category=Category.objects.create(name='Snacks'))
        unfiltered_first = self.ids(self.post())
        filtered_first = self.ids(self.post(exclude_snacks=True))
        filtered_second = self.ids(self.post(exclude_snacks=True))
        unfiltered_second = self.ids(self.post())
        self.assertTrue(set(unfiltered_first).isdisjoint(unfiltered_second))
        self.assertTrue(set(filtered_first).isdisjoint(filtered_second))
        self.assertEqual(set(filtered_first + filtered_second), {p.pk for p in health})

    def test_eligibility_changes_reconcile_the_pending_queue(self):
        self.make_products(16)
        first = self.ids(self.post())
        pending = list(Product.objects.exclude(pk__in=first))
        Product.objects.filter(pk=pending[0].pk).update(quantity_in_stock=0)
        Product.objects.filter(pk=pending[1].pk).update(status=False)
        Product.objects.filter(pk=pending[2].pk).update(archived_at=timezone.now())
        self.record_sale(pending[3], 0)
        new_products = self.make_products(4, prefix='New')
        result = self.post()
        expected = {p.pk for p in pending[4:] + new_products}
        self.assertEqual(set(self.ids(result)), expected)
        self.assertEqual(result['available_count'], 16)

    def test_existing_sixty_nine_day_sale_cutoff_and_non_sale_rules_are_preserved(self):
        products = self.make_products(6)
        self.record_sale(products[0], 0)
        self.record_sale(products[1], 69)
        self.record_sale(products[2], 70)
        StockChange.objects.create(
            product=products[3], change_type='giveaway', quantity=1, user=self.user,
        )
        Product.objects.filter(pk=products[4].pk).update(quantity_in_stock=0)
        Product.objects.filter(pk=products[5].pk).update(status=False)
        # Deleted-product audit history must not poison a NOT IN query.
        StockChange.objects.create(product=None, change_type='checkout', quantity=1)
        payload = self.post()
        self.assertEqual(set(self.ids(payload)), {products[2].pk, products[3].pk})
        old_sale = next(row for row in payload['items'] if row['product_id'] == products[2].pk)
        self.assertEqual(old_sale['days_since_sale'], 70)
        self.assertEqual(payload['total_count'], reporting.dead_stock()['count'])

    def test_expansion_listing_pagination_and_reports_do_not_advance_rotation(self):
        products = self.make_products(24)
        first = self.ids(self.post())
        self.assertEqual(self.client.get(
            reverse('dashboard_expand'), {'section': 'deadstock'},
        ).status_code, 200)
        self.assertEqual(self.client.get(reverse('dashboard_deadstock')).status_code, 200)
        self.assertEqual(self.client.get(
            reverse('dashboard_deadstock'), {'page': 2},
        ).status_code, 200)
        reporting.daily_digest()
        second = self.ids(self.post())
        third = self.ids(self.post())
        self.assertEqual(len(set(first + second + third)), 24)
        self.assertEqual(set(first + second + third), {p.pk for p in products})

    def test_rotation_is_independent_between_browser_sessions(self):
        products = self.make_products(16)
        other_browser = self.make_client()
        first_browser = self.ids(self.post()) + self.ids(self.post())
        second_browser = self.ids(self.post(client=other_browser))
        second_browser += self.ids(self.post(client=other_browser))
        expected = {product.pk for product in products}
        self.assertEqual(set(first_browser), expected)
        self.assertEqual(set(second_browser), expected)


@skipUnless(connection.vendor == 'postgresql', 'Requires PostgreSQL row locking.')
@override_settings(AXES_ENABLED=False)
class DashboardDeadStockConcurrentTests(DeadStockTestHelpers, TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='concurrent-first', is_staff=True)
        self.other = User.objects.create_user(username='concurrent-second', is_staff=True)
        self.category = Category.objects.create(name='Concurrency')

    def test_simultaneous_first_dismissals_create_one_unchanged_thirty_day_record(self):
        product = self.make_products(1)[0]
        clients = [self.make_client(self.user), self.make_client(self.other)]
        barrier = Barrier(2)

        def dismiss(client):
            close_old_connections()
            try:
                barrier.wait(timeout=10)
                response = client.post(
                    reverse('dashboard_deadstock'),
                    {'action': 'dismiss', 'product_id': product.pk},
                    content_type='application/json',
                )
                return response.status_code, response.json()
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(dismiss, clients))

        self.assertEqual([status for status, _ in results], [200, 200])
        self.assertEqual(DashboardDeadStockDismissal.objects.count(), 1)
        dismissal = DashboardDeadStockDismissal.objects.get(product=product)
        self.assertEqual(dismissal.expires_at - dismissal.dismissed_at, timedelta(days=30))
        self.assertIn(dismissal.dismissed_by_id, (self.user.pk, self.other.pk))
        self.assertEqual(results[0][1]['expires_at'], results[1][1]['expires_at'])
        self.assertTrue(all(payload['dismissed'] for _, payload in results))

    def test_same_session_page_entries_do_not_consume_the_same_pending_batch(self):
        products = self.make_products(16)
        first_tab = self.make_client(self.user)
        initial_ids = self.ids(self.post(client=first_tab))
        second_tab = Client()
        second_tab.cookies[settings.SESSION_COOKIE_NAME] = first_tab.cookies[
            settings.SESSION_COOKIE_NAME
        ].value
        request_barrier = Barrier(2)
        selection_barrier = Barrier(2, timeout=1)
        choose_ids = dashboard_deadstock._choose_ids

        def overlapping_selection(*args, **kwargs):
            # Without request serialization, both session snapshots reach this
            # point together. With serialization, the first proceeds after the
            # timeout while the second correctly waits outside the session load.
            try:
                selection_barrier.wait()
            except BrokenBarrierError:
                pass
            return choose_ids(*args, **kwargs)

        def next_batch(client):
            close_old_connections()
            try:
                request_barrier.wait(timeout=10)
                response = client.post(
                    reverse('dashboard_deadstock'), {'action': 'next'},
                    content_type='application/json',
                )
                return response.status_code, response.json()
            finally:
                close_old_connections()

        with patch(
            'app.dashboard_deadstock._choose_ids', side_effect=overlapping_selection,
        ), ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(next_batch, [first_tab, second_tab]))

        self.assertEqual([status for status, _ in results], [200, 200])
        batches = [self.ids(payload) for _, payload in results]
        self.assertEqual([len(batch) for batch in batches], [8, 8])
        self.assertTrue(set(batches[0]).isdisjoint(batches[1]))
        self.assertEqual(set(batches[0] + batches[1]), {p.pk for p in products})
        # One request finishes the first cycle; the other starts the next cycle.
        self.assertIn(set(initial_ids), [set(batch) for batch in batches])
