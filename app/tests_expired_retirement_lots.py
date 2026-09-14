from datetime import date, timedelta
from decimal import Decimal
from urllib.parse import quote, urlencode

from dateutil.relativedelta import relativedelta
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import (
    Category,
    Product,
    ProductLot,
    StockChange,
    UserAction,
)


@override_settings(AXES_ENABLED=False)
class ExpiredLotRetirementTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='expired-lot-user', password='pass1234',
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='Expiry Lots')
        self.product = Product.objects.create(
            name='Lot Retirement Product',
            barcode='RETIRE-LOT-1001',
            price=Decimal('12.50'),
            quantity_in_stock=9,
            category=self.category,
        )
        self.expired_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='LOT-EXPIRED',
            expiry_date=date.today() - timedelta(days=5),
            quantity_on_hand=2,
        )
        self.cutoff_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='LOT-ONE-MONTH',
            expiry_date=date.today() + relativedelta(months=1),
            quantity_on_hand=3,
        )
        self.future_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='LOT-TOO-EARLY',
            expiry_date=date.today() + relativedelta(months=1) + timedelta(days=1),
            quantity_on_hand=4,
        )

    def _retire(self, lot, quantity, follow=True):
        return self.client.post(
            reverse('expired_products'),
            {
                'mode': 'log',
                'barcode': self.product.barcode,
                'retire_expired': '1',
                'retire_lot_id': str(lot.pk),
                'retire_quantity': str(quantity),
            },
            follow=follow,
        )

    def test_product_name_opens_details_and_returns_to_filtered_expired_list(self):
        origin = reverse('expired_products') + '?' + urlencode({
            'mode': 'view',
            'name_query': self.product.name,
            'date_filter': 'custom',
            'date_from': (date.today() - timedelta(days=7)).isoformat(),
            'date_to': (date.today() - timedelta(days=1)).isoformat(),
            'sort': '-name',
        })
        details_url = (
            reverse('product_details', args=[self.product.pk])
            + '?return_to=' + quote(origin, safe='/')
        )

        listing = self.client.get(origin)
        self.assertContains(listing, f'href="{details_url}"')
        self.assertContains(listing, 'class="exp-product-details-link"', count=1)

        details = self.client.get(details_url)
        self.assertEqual(details.status_code, 200)
        self.assertEqual(details.context['product'].pk, self.product.pk)
        self.assertEqual(details.context['return_to'], origin)
        self.assertEqual(details.context['page_return']['url'], origin)
        self.assertEqual(details.context['page_return']['label'], 'Back to Expired Stock')
        returned = self.client.get(details.context['page_return']['url'])
        self.assertEqual([p.pk for p in returned.context['products']], [self.product.pk])

    def test_scanned_product_shows_lot_numbers_and_one_month_eligibility(self):
        response = self.client.get(
            reverse('expired_products'),
            {'mode': 'log', 'pid': self.product.pk},
        )

        self.assertEqual(response.status_code, 200)
        rows = {row['lot_number']: row for row in response.context['product_extra']['lots']}
        self.assertTrue(rows['LOT-EXPIRED']['eligible'])
        self.assertTrue(rows['LOT-ONE-MONTH']['eligible'])
        self.assertFalse(rows['LOT-TOO-EARLY']['eligible'])
        self.assertEqual(response.context['product_extra']['retirement_quantity'], 5)
        self.assertContains(response, 'Choose the lot you collected')
        self.assertContains(response, 'Lot LOT-ONE-MONTH')
        self.assertContains(response, 'Within one month')

    def test_retirement_removes_only_the_selected_lot_and_audits_it(self):
        response = self._retire(self.cutoff_lot, 2)

        self.assertEqual(response.status_code, 200)
        self.product.refresh_from_db()
        self.expired_lot.refresh_from_db()
        self.cutoff_lot.refresh_from_db()
        self.future_lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 7)
        self.assertEqual(self.product.stock_expired, 2)
        self.assertEqual(self.expired_lot.quantity_on_hand, 2)
        self.assertEqual(self.cutoff_lot.quantity_on_hand, 1)
        self.assertIsNone(self.cutoff_lot.archived_at)
        self.assertEqual(self.future_lot.quantity_on_hand, 4)

        change = StockChange.objects.get(change_type='expired')
        self.assertIn('lot LOT-ONE-MONTH', change.note)
        self.assertEqual(
            list(change.lot_movements.values_list('lot_number', 'quantity')),
            [('LOT-ONE-MONTH', 2)],
        )
        self.assertTrue(
            UserAction.objects.filter(
                user=self.user,
                action='retire_expired',
                detail='2 units retired from lot LOT-ONE-MONTH',
            ).exists()
        )
        self.assertContains(response, 'lot <strong>LOT-ONE-MONTH</strong>')

    def test_retirement_clears_a_fully_depleted_lot_from_active_product_lots(self):
        response = self._retire(self.expired_lot, 2)

        self.assertEqual(response.status_code, 200)
        self.product.refresh_from_db()
        self.expired_lot.refresh_from_db()
        self.cutoff_lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 7)
        self.assertEqual(self.product.expiry_date, self.cutoff_lot.expiry_date)
        self.assertEqual(self.expired_lot.quantity_on_hand, 0)
        self.assertIsNotNone(self.expired_lot.archived_at)
        self.assertEqual(self.expired_lot.archived_by, self.user)
        self.assertFalse(
            self.product.lots.filter(
                pk=self.expired_lot.pk,
                archived_at__isnull=True,
            ).exists()
        )
        self.assertIsNone(self.cutoff_lot.archived_at)

        change = StockChange.objects.get(change_type='expired')
        movement = change.lot_movements.get()
        self.assertEqual(movement.lot, self.expired_lot)
        self.assertEqual(movement.lot_number, 'LOT-EXPIRED')
        self.assertEqual(movement.quantity, 2)

    def test_full_legacy_retirement_clears_generated_unassigned_lot(self):
        legacy_product = Product.objects.create(
            name='Legacy Expired Product',
            barcode='LEGACY-EXPIRED-1001',
            price=Decimal('4.50'),
            quantity_in_stock=3,
            expiry_date=date.today() - timedelta(days=10),
            category=self.category,
        )

        response = self.client.post(
            reverse('expired_products'),
            {
                'mode': 'log',
                'barcode': legacy_product.barcode,
                'retire_expired': '1',
                'retire_lot_id': 'legacy',
                'retire_quantity': '3',
            },
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        legacy_product.refresh_from_db()
        generated_lot = legacy_product.lots.get(
            lot_number=ProductLot.UNASSIGNED,
        )
        self.assertEqual(legacy_product.quantity_in_stock, 0)
        self.assertEqual(generated_lot.quantity_on_hand, 0)
        self.assertIsNotNone(generated_lot.archived_at)
        self.assertEqual(generated_lot.archived_by, self.user)
        self.assertFalse(
            legacy_product.lots.filter(archived_at__isnull=True).exists()
        )

        movement = StockChange.objects.get(
            product=legacy_product,
            change_type='expired',
        ).lot_movements.get()
        self.assertEqual(movement.lot, generated_lot)
        self.assertEqual(movement.lot_number, ProductLot.UNASSIGNED)
        self.assertEqual(movement.quantity, 3)

    def test_lot_more_than_one_month_away_cannot_be_retired(self):
        response = self._retire(self.future_lot, 1)

        self.product.refresh_from_db()
        self.future_lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 9)
        self.assertEqual(self.future_lot.quantity_on_hand, 4)
        self.assertFalse(StockChange.objects.filter(change_type='expired').exists())
        self.assertContains(response, 'Lots become eligible one month before expiry.')

    def test_quantity_cannot_exceed_the_selected_lot(self):
        response = self._retire(self.cutoff_lot, 4)

        self.product.refresh_from_db()
        self.cutoff_lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 9)
        self.assertEqual(self.cutoff_lot.quantity_on_hand, 3)
        self.assertContains(
            response,
            'Only 3 unit(s) remain in that lot.',
        )
