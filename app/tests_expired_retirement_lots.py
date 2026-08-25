from datetime import date, timedelta
from decimal import Decimal

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
        self.assertContains(response, 'Choose the exact lot being removed')
        self.assertContains(response, 'Lot LOT-ONE-MONTH')
        self.assertContains(response, 'Eligible now')

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
