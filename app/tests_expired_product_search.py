from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import Product, ProductLot


@override_settings(AXES_ENABLED=False)
class ExpiredProductSearchTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='expired-search-user')
        self.client.force_login(self.user)
        self.url = reverse('expired_products')
        self.today = date.today()
        self.product = self._product(
            'Caltrate Plus', '00062107055901', '703728',
            self.today - timedelta(days=2),
        )
        self.other = self._product(
            'Tylenol Arthritis', '00062107055902', '703729',
            self.today - timedelta(days=1),
        )

    def _product(self, name, barcode, item_number, expiry, quantity=4):
        return Product.objects.create(
            name=name, barcode=barcode, item_number=item_number,
            expiry_date=expiry, quantity_in_stock=quantity,
            price=Decimal('5.00'),
        )

    def _search(self, query, **filters):
        response = self.client.get(self.url, {'name_query': query, **filters})
        self.assertEqual(response.status_code, 200)
        return response

    def test_search_matches_name_barcode_and_item_number(self):
        for query in ('  caLtRaTe  ', '107055901', '0000062107055901', '3728'):
            with self.subTest(query=query):
                response = self._search(query)
                self.assertEqual(response.context['products'], [self.product])
                self.assertEqual(response.context['name_query'], query.strip())
                self.assertEqual(response.context['total_units_on_shelf'], 4)

    def test_blank_search_keeps_expiry_results_and_sort(self):
        response = self._search('   ', sort='-name')

        self.assertEqual(response.context['products'], [self.other, self.product])
        self.assertEqual(response.context['name_query'], '')

    def test_unmatched_search_returns_no_rows_or_at_risk_totals(self):
        response = self._search('nothing-matches-this')

        self.assertEqual(response.context['products'], [])
        self.assertEqual(response.context['product_count'], 0)
        self.assertEqual(response.context['total_units_on_shelf'], 0)
        self.assertEqual(response.context['value_at_risk'], Decimal('0.00'))

    def test_search_respects_preset_and_custom_expiry_windows(self):
        future = self._product(
            'Caltrate Future', 'SEARCH-FUTURE', '703728-FUTURE',
            self.today + timedelta(days=2),
        )
        after_window = self._product(
            'Caltrate Later', 'SEARCH-LATER', '703728-LATER',
            self.today + timedelta(days=8),
        )

        for query in ('Caltrate', '703728'):
            with self.subTest(query=query, window='expired'):
                response = self._search(query)
                self.assertEqual(response.context['products'], [self.product])
            with self.subTest(query=query, window='1_week'):
                response = self._search(query, date_filter='1_week')
                self.assertEqual(response.context['products'], [future])
            with self.subTest(query=query, window='custom'):
                response = self._search(
                    query, date_filter='custom',
                    date_from=future.expiry_date.isoformat(),
                    date_to=after_window.expiry_date.isoformat(), sort='-name',
                )
                self.assertEqual(response.context['products'], [after_window, future])

    def test_search_uses_matching_lots_without_duplicate_products(self):
        ProductLot.objects.create(
            product=self.product, lot_number='EXPIRED-ONE',
            expiry_date=self.today - timedelta(days=3), quantity_on_hand=1,
        )
        ProductLot.objects.create(
            product=self.product, lot_number='EXPIRED-TWO',
            expiry_date=self.today - timedelta(days=1), quantity_on_hand=1,
        )
        ProductLot.objects.create(
            product=self.product, lot_number='FUTURE',
            expiry_date=self.today + timedelta(days=5), quantity_on_hand=2,
        )

        for query in ('Caltrate', '107055901', '703728'):
            with self.subTest(query=query):
                response = self._search(query)
                self.assertEqual(response.context['products'], [self.product])
                self.assertEqual(response.context['total_units_on_shelf'], 2)
                self.assertEqual(response.context['value_at_risk'], Decimal('10.00'))
                self.assertEqual(len(response.context['products'][0].expiry_lot_rows), 2)

    def test_search_does_not_include_unavailable_or_undated_stock(self):
        self._product('Caltrate Empty', 'SEARCH-EMPTY', '703728-EMPTY',
                      self.today - timedelta(days=1), quantity=0)
        self._product('Caltrate Undated', 'SEARCH-UNDATED', '703728-UNDATED', None)
        archived = self._product(
            'Caltrate Archived', 'SEARCH-ARCHIVED', '703728-ARCHIVED',
            self.today - timedelta(days=1),
        )
        archived.archived_at = timezone.now()
        archived.save(update_fields=['archived_at'])

        for query in ('Caltrate', '703728'):
            with self.subTest(query=query):
                response = self._search(query)
                self.assertEqual(response.context['products'], [self.product])
