from datetime import date, datetime
from urllib.parse import parse_qs, urlsplit

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import Category, Product, ProductLotMovement, StockChange


@override_settings(AXES_ENABLED=False)
class ExpiredLogTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='expiry-history-staff', password='test-pass')
        self.client.force_login(self.user)
        self.url = reverse('expired_log')
        self.product = Product.objects.create(
            name='Current product name', barcode='CURRENT-1', price='10.00',
            category=Category.objects.create(name='History'), quantity_in_stock=9,
        )
        self.log = StockChange.objects.create(
            product=self.product, product_name='Collected product snapshot',
            product_barcode='HISTORY-123', quantity=3, change_type='expired',
            user=self.user, note='Collected from the shelves',
        )
        StockChange.objects.filter(pk=self.log.pk).update(
            timestamp=timezone.make_aware(datetime(2026, 9, 1, 15, 30)),
        )
        self.movement = ProductLotMovement.objects.create(
            stock_change=self.log, lot_number='HISTORY-LOT', expiry_date=date(2026, 8, 31),
            quantity=3, direction=ProductLotMovement.DIRECTION_OUT,
        )

    def test_login_is_required(self):
        self.client.logout()
        response = self.client.get(self.url)
        self.assertRedirects(response, reverse('login') + '?next=' + self.url)

    def test_page_shows_recorded_identity_lot_and_staff(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'expired_log.html')
        for text in ('Collected product snapshot', 'HISTORY-123', 'HISTORY-LOT',
                     '31/08/2026', '01/09/2026', '15:30', self.user.username,
                     'Collected from the shelves', '3 units removed'):
            self.assertContains(response, text)
        self.assertEqual(response.context['page_obj'][0].removed_units, 3)

    def test_only_expired_changes_are_included(self):
        StockChange.objects.create(product=self.product, quantity=8, change_type='checkin')
        response = self.client.get(self.url)
        self.assertEqual(response.context['page_obj'].paginator.count, 1)
        self.assertEqual(response.context['total_units'], 3)

    def test_archived_products_remain_visible(self):
        self.product.archived_at = timezone.now()
        self.product.save(update_fields=['archived_at'])
        response = self.client.get(self.url)
        self.assertContains(response, 'Collected product snapshot')

    def test_missing_product_user_and_lot_records_keep_audit_snapshots(self):
        self.log.product = None
        self.log.user = None
        self.log.save(update_fields=['product', 'user'])
        response = self.client.get(self.url)
        for text in ('Collected product snapshot', 'HISTORY-123', 'HISTORY-LOT', 'Not recorded'):
            self.assertContains(response, text)

    def test_legacy_entries_fall_back_to_product_identity(self):
        self.log.product_name = ''
        self.log.product_barcode = ''
        self.log.save(update_fields=['product_name', 'product_barcode'])
        self.movement.delete()
        response = self.client.get(self.url)
        for text in ('Current product name', 'CURRENT-1', 'Lot not recorded'):
            self.assertContains(response, text)

    def test_search_by_snapshot_current_identity_lot_and_staff(self):
        for query in ('collected product', 'HISTORY-123', 'current product',
                      'CURRENT-1', 'history-lot', self.user.username):
            with self.subTest(query=query):
                response = self.client.get(self.url, {'q': query})
                self.assertEqual(response.context['page_obj'].paginator.count, 1)
        response = self.client.get(self.url, {'q': 'no-such-product'})
        self.assertContains(response, 'No expired log entries match these filters.')
        self.assertEqual(response.context['total_units'], 0)

    def test_multiple_matching_lots_do_not_duplicate_entries_or_totals(self):
        ProductLotMovement.objects.create(
            stock_change=self.log, lot_number='HISTORY-LOT-2', quantity=1,
            direction=ProductLotMovement.DIRECTION_OUT,
        )
        response = self.client.get(self.url, {'q': 'history-lot'})
        self.assertEqual(response.context['page_obj'].paginator.count, 1)
        self.assertEqual(response.context['total_units'], 3)

    def test_logged_date_filters_are_inclusive_and_not_expiry_dates(self):
        response = self.client.get(self.url, {'from': '2026-09-01', 'to': '2026-09-01'})
        self.assertEqual(response.context['page_obj'].paginator.count, 1)
        response = self.client.get(self.url, {'from': '2026-08-31', 'to': '2026-08-31'})
        self.assertEqual(response.context['page_obj'].paginator.count, 0)
        for filters in ({'from': '2026-09-02'}, {'to': '2026-08-31'}):
            with self.subTest(filters=filters):
                response = self.client.get(self.url, filters)
                self.assertEqual(response.context['page_obj'].paginator.count, 0)

    def test_invalid_dates_show_feedback_instead_of_unfiltered_results(self):
        for filters in ({'from': '2026-02-31'}, {'to': 'invalid'},
                        {'from': '2026-09-02', 'to': '2026-09-01'}):
            with self.subTest(filters=filters):
                response = self.client.get(self.url, filters)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.context['filter_error'])
                self.assertEqual(response.context['page_obj'].paginator.count, 0)

    def test_history_is_paginated_without_a_fifty_entry_cap(self):
        StockChange.objects.bulk_create([
            StockChange(product=self.product, product_name=f'Newer entry {index}',
                        change_type='expired', quantity=1) for index in range(51)
        ])
        first = self.client.get(self.url)
        self.assertEqual(len(first.context['page_obj']), 50)
        self.assertEqual(first.context['page_obj'].paginator.count, 52)
        self.assertEqual(first.context['total_units'], 54)
        second = self.client.get(self.url, {'page': 2})
        self.assertEqual(len(second.context['page_obj']), 2)
        self.assertEqual(second.context['page_obj'][-1].pk, self.log.pk)
        for page in ('invalid', '-1', '999'):
            self.assertEqual(self.client.get(self.url, {'page': page}).status_code, 200)

    def test_filters_and_expired_list_context_are_preserved(self):
        origin = reverse('expired_products') + '?date_filter=custom&date_from=2026-08-01&sort=-name'
        filters = {'q': 'history', 'from': '2026-09-01', 'to': '2026-09-30', 'return_to': origin}
        response = self.client.get(self.url, filters)
        self.assertEqual(response.context['page_return']['url'], origin)
        self.assertEqual(parse_qs(response.context['pagination_query']),
                         {key: [value] for key, value in filters.items()})
        self.assertEqual(parse_qs(urlsplit(response.context['clear_url']).query),
                         {'return_to': [origin]})

    def test_back_link_always_returns_to_expired_stock(self):
        for origin in ('https://example.com/', reverse('inventory_display'), self.url, ''):
            with self.subTest(origin=origin):
                response = self.client.get(self.url, {'return_to': origin})
                self.assertEqual(response.context['page_return']['url'], reverse('expired_products'))

    def test_empty_history_has_a_clear_empty_state(self):
        self.log.delete()
        self.assertContains(self.client.get(self.url), 'No expired stock has been logged yet.')

    def test_history_is_read_only(self):
        before = list(StockChange.objects.values())
        movements = list(ProductLotMovement.objects.values())
        self.client.get(self.url)
        response = self.client.post(self.url, {'expiry_action': 'confirm', 'quantity': '9'})
        self.assertEqual(response.status_code, 405)
        self.assertEqual(list(StockChange.objects.values()), before)
        self.assertEqual(list(ProductLotMovement.objects.values()), movements)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 9)

    def test_saved_text_is_escaped(self):
        self.log.product_name = '<script>alert("name")</script>'
        self.log.note = '<script>alert("note")</script>'
        self.log.save(update_fields=['product_name', 'note'])
        response = self.client.get(self.url)
        self.assertNotContains(response, '<script>alert(')
        self.assertContains(response, '&lt;script&gt;alert(')
