from datetime import date, datetime, time, timedelta
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from .daily_reporting import build_daily_report
from .models import (
    Category, Order, OrderDetail, Product, ProductLot, StockChange,
    TransactionCorrection, TransactionCorrectionLine, TransactionCorrectionUndo,
)
from .reporting import build_daily_report_pdf
from .utils import calculate_order_financials_from_values


class DailyReportDataTests(TestCase):
    def setUp(self):
        self.today = date.today()
        self.day = self.today - timedelta(days=14)
        self.category = Category.objects.create(name='Daily report', low_stock_threshold=4)
        self.user = User.objects.create_user(username='daily-report-user')

    def product(self, name, **kwargs):
        defaults = {
            'price': Decimal('10.00'), 'price_per_unit': Decimal('4.00'),
            'quantity_in_stock': 10, 'category': self.category,
        }
        defaults.update(kwargs)
        return Product.objects.create(name=name, **defaults)

    @staticmethod
    def stamp(day, hour=12, minute=0):
        return timezone.make_aware(datetime.combine(day, time(hour, minute)))

    def sale(self, day, entries, *, seniors=False, submitted=True, discount=None):
        """Entries: (product, quantity, immutable price, immutable cost)."""
        values = calculate_order_financials_from_values(
            [(Decimal(price), qty, False) for _, qty, price, _ in entries],
            seniors_discount=seniors,
        )
        order = Order.objects.create(
            submitted=submitted,
            seniors_discount=seniors,
            subtotal=values['subtotal'],
            discount_amount=Decimal(discount) if discount is not None else values['discount_amount'],
            tax=values['tax'],
            total_price=values['total'],
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        Order.objects.filter(pk=order.pk).update(order_date=self.stamp(day))
        for product, qty, price, cost in entries:
            OrderDetail.objects.create(
                order=order, product=product, product_name=product.name,
                product_barcode=product.barcode or '', quantity=qty,
                price=Decimal(price), cost_per_unit_at_sale=Decimal(cost) if cost is not None else None,
                taxable_at_sale=False,
            )
        return order

    def correction(self, order, quantity, disposition):
        line = order.details.get()
        correction = TransactionCorrection.objects.create(
            order=order, correction_type=TransactionCorrection.TYPE_RETURN,
            reason='Daily report test', created_by=self.user,
        )
        TransactionCorrectionLine.objects.create(
            correction=correction, order_detail=line, product=line.product,
            product_name=line.product_name, quantity=quantity, unit_price=line.price,
            disposition=disposition,
        )
        return correction

    def change(self, product, day, action, qty, *, hour=12, minute=0):
        change = StockChange.objects.create(
            product=product, product_name=product.name, user=self.user,
            change_type=action, quantity=qty, note='Daily report note',
        )
        StockChange.objects.filter(pk=change.pk).update(timestamp=self.stamp(day, hour, minute))
        return change

    def test_empty_report_is_zero_filled_and_pdf_compatible(self):
        digest = build_daily_report(self.day)

        self.assertEqual(digest['inventory_day'], self.today)
        self.assertEqual(digest['sales']['revenue_today'], Decimal('0.00'))
        self.assertEqual(digest['sales']['average_order'], Decimal('0.00'))
        self.assertIsNone(digest['sales']['margin_pct'])
        self.assertEqual(digest['sales']['missing_cost_units'], 0)
        self.assertEqual(len(digest['trend']), 7)
        self.assertEqual(digest['trend'][0]['day'], self.day - timedelta(days=6))
        self.assertEqual(digest['trend'][-1]['day'], self.day)
        self.assertTrue(all(point['revenue'] == point['height_pct'] == 0 for point in digest['trend']))
        self.assertIsNone(digest['comparisons']['previous_day']['pct'])
        self.assertEqual(digest['comparisons']['previous_day']['direction'], 'flat')
        self.assertEqual(digest['activity']['checkin_units'], 0)
        self.assertEqual(digest['top_products'], [])
        self.assertEqual(digest['categories'], [])
        self.assertTrue(build_daily_report_pdf(digest).startswith(b'%PDF'))

    def test_selected_day_comparisons_and_trend_exclude_future_and_drafts(self):
        product = self.product('Bounded product')
        for offset, price in ((-8, '800'), (-7, '20'), (-6, '30'), (-1, '40'), (0, '60'), (1, '900')):
            self.sale(self.day + timedelta(days=offset), [(product, 1, price, '1')])
        self.sale(self.day, [(product, 20, '100', '1')], submitted=False)

        digest = build_daily_report(self.day)

        self.assertEqual(digest['sales']['revenue_today'], Decimal('60.00'))
        self.assertEqual(digest['sales']['orders_today'], 1)
        self.assertEqual(digest['sales']['average_order'], Decimal('60.00'))
        previous = digest['comparisons']['previous_day']
        self.assertEqual((previous['revenue'], previous['delta'], previous['pct'], previous['direction']),
                         (Decimal('40'), Decimal('20'), Decimal('50.0'), 'up'))
        self.assertEqual(digest['comparisons']['previous_week']['pct'], Decimal('200.0'))
        self.assertEqual(sum(point['revenue'] for point in digest['trend']), Decimal('130'))
        self.assertEqual(digest['trend'][-1]['height_pct'], Decimal('100.0'))
        self.assertEqual(digest['top_movers'][0]['total_qty'], 3)
        self.assertEqual(digest['top_products'][0]['units'], 1)

    def test_snapshot_discount_cents_and_costs_reconcile_across_products_and_categories(self):
        product_a = self.product('Sale-time name', barcode='DAILY-A')
        category_b = Category.objects.create(name='Second category')
        product_b = self.product('Second product', category=category_b)
        # A captured discount is authoritative even when it differs from the
        # normal seniors calculation. Three 5-cent lines must total 13 cents.
        self.sale(self.day, [(product_a, 1, '.05', '.02'), (product_b, 2, '.05', '.01')],
                  seniors=True, discount='.02')
        Product.objects.filter(pk=product_a.pk).update(
            name='Renamed later', price=Decimal('99'), price_per_unit=Decimal('80'),
        )

        digest = build_daily_report(self.day)

        self.assertEqual(digest['sales']['revenue_today'], Decimal('.13'))
        self.assertEqual(digest['sales']['cost'], Decimal('.04'))
        self.assertEqual(digest['sales']['profit'], Decimal('.09'))
        self.assertEqual(digest['sales']['margin_pct'], Decimal('69.2'))
        self.assertEqual(sum(row['revenue'] for row in digest['top_products']), Decimal('.13'))
        self.assertEqual(sum(row['profit'] for row in digest['categories']), Decimal('.09'))
        first = next(row for row in digest['top_products'] if row['product_id'] == product_a.pk)
        self.assertEqual((first['name'], first['barcode']), ('Sale-time name', 'DAILY-A'))

    def test_returns_remove_revenue_but_only_restocked_returns_remove_cost(self):
        product = self.product('Return product')
        order = self.sale(self.day, [(product, 3, '10', '4')])
        self.correction(order, 1, TransactionCorrectionLine.DISPOSITION_RESTOCK)
        damaged = self.correction(order, 2, TransactionCorrectionLine.DISPOSITION_DAMAGED)

        digest = build_daily_report(self.day)
        self.assertEqual(digest['sales']['revenue_today'], Decimal('0'))
        self.assertEqual(digest['sales']['orders_today'], 0)
        self.assertEqual(digest['sales']['units_sold'], 0)
        self.assertEqual(digest['sales']['cost'], Decimal('8'))
        self.assertEqual(digest['sales']['profit'], Decimal('-8'))
        self.assertIsNone(digest['sales']['margin_pct'])
        self.assertEqual(digest['top_products'][0]['profit'], Decimal('-8'))

        TransactionCorrectionUndo.objects.create(correction=damaged, created_by=self.user)
        restored = build_daily_report(self.day)
        self.assertEqual(restored['sales']['revenue_today'], Decimal('20'))
        self.assertEqual(restored['sales']['cost'], Decimal('8'))
        self.assertEqual(restored['sales']['units_sold'], 2)
        order.refresh_from_db()
        self.assertEqual(order.subtotal, Decimal('30'))

    def test_missing_sale_costs_are_flagged_without_using_current_product_cost(self):
        product = self.product('Missing snapshot', price_per_unit=Decimal('99'))
        order = self.sale(self.day, [(product, 2, '10', None)])
        self.correction(order, 1, TransactionCorrectionLine.DISPOSITION_DAMAGED)

        digest = build_daily_report(self.day)

        self.assertEqual(digest['sales']['cost'], Decimal('0'))
        self.assertEqual(digest['sales']['missing_cost_units'], 2)
        self.assertEqual(digest['sales']['revenue_today'], Decimal('10'))

    def test_snacks_filter_applies_to_sales_comparisons_stock_expiry_and_activity(self):
        snacks = Category.objects.create(name='sNaCkS')
        snack = self.product('Snack', category=snacks, quantity_in_stock=1,
                             expiry_date=self.today + timedelta(days=1))
        product = self.product('Medicine', quantity_in_stock=2)
        self.sale(self.day, [(snack, 1, '.05', '.01'), (product, 1, '.05', '.02')], seniors=True)
        self.sale(self.day - timedelta(days=1), [(snack, 2, '10', '1')])
        self.change(snack, self.day, 'checkin', 3)
        self.change(product, self.day, 'checkin', 5)

        digest = build_daily_report(self.day, exclude_snacks=True)

        self.assertEqual(digest['sales']['revenue_today'], Decimal('.04'))
        self.assertEqual(digest['sales']['cost'], Decimal('.02'))
        self.assertEqual(digest['sales']['orders_today'], 1)
        self.assertEqual(digest['comparisons']['previous_day']['revenue'], Decimal('0'))
        self.assertEqual([row['product_id'] for row in digest['top_products']], [product.pk])
        self.assertEqual(digest['stock_health']['total_products'], 1)
        self.assertEqual(digest['low_stock']['count'], 1)
        self.assertEqual(digest['expiring_week']['count'], 0)
        self.assertEqual(digest['activity']['checkin_units'], 5)
        self.assertEqual(digest['activity']['count'], 1)

    def test_expiry_uses_current_positive_active_lots_and_legacy_fallback_only(self):
        product = self.product('Mixed lots', expiry_date=self.day, quantity_in_stock=20)
        ProductLot.objects.create(product=product, lot_number='EXPIRED', expiry_date=self.today - timedelta(days=1), quantity_on_hand=2)
        ProductLot.objects.create(product=product, lot_number='TODAY', expiry_date=self.today, quantity_on_hand=3)
        ProductLot.objects.create(product=product, lot_number='WEEK', expiry_date=self.today + timedelta(days=7), quantity_on_hand=4)
        ProductLot.objects.create(product=product, lot_number='LATER', expiry_date=self.today + timedelta(days=8), quantity_on_hand=5)
        ProductLot.objects.create(product=product, lot_number='EMPTY', expiry_date=self.today, quantity_on_hand=0)
        ProductLot.objects.create(product=product, lot_number='ARCHIVED', expiry_date=self.today, quantity_on_hand=6, archived_at=timezone.now())
        depleted = self.product('Depleted lot authority', expiry_date=self.today, quantity_in_stock=9)
        ProductLot.objects.create(product=depleted, lot_number='ZERO', expiry_date=self.today, quantity_on_hand=0)
        undated = self.product('Undated lot authority', expiry_date=self.today)
        ProductLot.objects.create(product=undated, lot_number='UNDATED', quantity_on_hand=1)
        legacy = self.product('Legacy', expiry_date=self.today + timedelta(days=2), quantity_in_stock=7)
        self.product('No units', expiry_date=self.today, quantity_in_stock=0)
        inactive = self.product('Inactive', expiry_date=self.today, status=False)
        ProductLot.objects.create(product=inactive, lot_number='INACTIVE', expiry_date=self.today, quantity_on_hand=2)
        archived = self.product('Archived product', expiry_date=self.today, archived_at=timezone.now())
        ProductLot.objects.create(product=archived, lot_number='ARCHIVED-PRODUCT', expiry_date=self.today, quantity_on_hand=2)

        digest = build_daily_report(self.day)

        self.assertEqual(digest['inventory_day'], self.today)
        self.assertEqual(digest['stock_health']['expired_count'], 1)
        self.assertEqual(digest['expired_stock']['items'][0]['quantity_in_stock'], 2)
        self.assertEqual(digest['expiring_week']['count'], 3)
        self.assertEqual(digest['stock_health']['expiring_soon_count'], 3)
        self.assertEqual({(row['product_id'], row['quantity_in_stock']) for row in digest['expiring_week']['items']},
                         {(product.pk, 3), (product.pk, 4), (legacy.pk, 7)})
        self.assertEqual([row['days_left'] for row in digest['expiring_week']['items']], [0, 2, 7])

    def test_activity_and_stock_lists_are_bounded_without_truncating_counts(self):
        product = self.product('Ledger product', quantity_in_stock=3)
        for i in range(34):
            self.change(product, self.day, 'checkin', 2, hour=9, minute=i)
        self.change(product, self.day, 'error_subtract', -2, hour=10)
        self.change(product, self.day, 'expired', 4, hour=11)
        self.change(product, self.day + timedelta(days=1), 'expired', 100)
        self.change(product, self.day - timedelta(days=1), 'checkin', 100)
        for i in range(26):
            self.product(f'Low {i:02}', quantity_in_stock=1)
            self.product(f'Out {i:02}', quantity_in_stock=0)

        digest = build_daily_report(self.day)

        self.assertEqual(digest['activity']['count'], 36)
        self.assertEqual(digest['activity']['checkin_count'], 34)
        self.assertEqual(digest['activity']['checkin_units'], 68)
        self.assertEqual(digest['activity']['correction_count'], 1)
        self.assertEqual(digest['activity']['expired_units'], 4)
        self.assertEqual(len(digest['activity']['items']), 30)
        self.assertEqual(digest['activity']['items'][0]['time'], '11:00')
        self.assertEqual(digest['activity']['items'][0]['user'], self.user.username)
        self.assertEqual(digest['activity']['items'][0]['product_id'], product.pk)
        self.assertEqual(digest['corrections']['corrections'][0]['qty'], -2)
        self.assertEqual((digest['low_stock']['count'], len(digest['low_stock']['items'])), (27, 25))
        self.assertEqual((digest['out_of_stock']['count'], len(digest['out_of_stock']['items'])), (26, 25))
        self.assertEqual(digest['dead_stock']['lookback_days'], 69)
        self.assertIn('retail_value', digest['dead_stock']['items'][0])

    def test_archived_and_deleted_products_retain_report_snapshots_without_links(self):
        archived = self.product('Archived sale snapshot', barcode='DAILY-ARCHIVED')
        deleted = self.product('Deleted sale snapshot', barcode='DAILY-DELETED')
        active = self.product('Active sale snapshot', barcode='DAILY-ACTIVE')
        self.sale(self.day, [(archived, 2, '10', '4'), (deleted, 1, '6', '2'), (active, 1, '3', '1')])
        self.change(archived, self.day, 'checkin', 2)
        self.change(deleted, self.day, 'checkin', 1)
        self.change(active, self.day, 'checkin', 1)
        Product.objects.filter(pk=archived.pk).update(archived_at=timezone.now())
        deleted.delete()

        digest = build_daily_report(self.day)

        self.assertEqual(digest['sales']['revenue_today'], Decimal('29.00'))
        self.assertEqual(digest['sales']['cost'], Decimal('11.00'))
        products = {item['name']: item for item in digest['top_products']}
        self.assertEqual(products['Archived sale snapshot']['product_id'], archived.pk)
        self.assertEqual(products['Archived sale snapshot']['barcode'], 'DAILY-ARCHIVED')
        self.assertFalse(products['Archived sale snapshot']['can_open_product'])
        self.assertIsNone(products['Deleted sale snapshot']['product_id'])
        self.assertEqual(products['Deleted sale snapshot']['revenue'], Decimal('6.00'))
        self.assertFalse(products['Deleted sale snapshot']['can_open_product'])
        self.assertTrue(products['Active sale snapshot']['can_open_product'])
        activity = {item['name']: item for item in digest['activity']['items']}
        self.assertFalse(activity['Archived sale snapshot']['can_open_product'])
        self.assertFalse(activity['Deleted sale snapshot']['can_open_product'])
        self.assertEqual(activity['Deleted sale snapshot']['qty'], 1)
        self.assertTrue(activity['Active sale snapshot']['can_open_product'])
