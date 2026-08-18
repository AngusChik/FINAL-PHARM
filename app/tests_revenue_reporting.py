from datetime import date
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils.timezone import now

from . import reporting
from .models import (
    Category,
    CheckoutOrder,
    CheckoutOrderItem,
    Order,
    OrderDetail,
    Product,
    TransactionCorrection,
    TransactionCorrectionLine,
    TransactionCorrectionUndo,
)


@override_settings(AXES_ENABLED=False)
class RealizedRevenueReportingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='revenue-admin',
            password='pass1234',
            is_staff=True,
        )
        category = Category.objects.create(name='Revenue reporting')
        self.product = Product.objects.create(
            name='Revenue Product',
            barcode='REV1001',
            price=Decimal('10.00'),
            price_per_unit=Decimal('4.00'),
            quantity_in_stock=7,
            category=category,
            taxable=True,
        )
        self.order = Order.objects.create(
            user=self.user,
            submitted=True,
            subtotal=Decimal('30.00'),
            discount_amount=Decimal('0.00'),
            tax=Decimal('3.90'),
            tax_rate=Decimal('0.13'),
            total_price=Decimal('33.90'),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        self.line = OrderDetail.objects.create(
            order=self.order,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=3,
            price=Decimal('10.00'),
            cost_per_unit_at_sale=Decimal('4.00'),
            taxable_at_sale=True,
        )
        self.client = Client()
        self.client.force_login(self.user)

    def _record_correction(
            self, quantity, correction_type,
            disposition=TransactionCorrectionLine.DISPOSITION_RESTOCK):
        adjustment = (Decimal('11.30') * quantity).quantize(Decimal('0.01'))
        correction = TransactionCorrection.objects.create(
            correction_type=correction_type,
            order=self.order,
            reason='Reporting regression test',
            adjustment_amount=adjustment,
            created_by=self.user,
        )
        TransactionCorrectionLine.objects.create(
            correction=correction,
            order_detail=self.line,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=quantity,
            unit_price=Decimal('10.00'),
            disposition=disposition,
        )
        return correction

    @staticmethod
    def _chart_revenue(chart):
        return Decimal(str(sum(row['revenue'] for row in chart)))

    def _assert_every_summary(self, expected_revenue, expected_units, expected_orders):
        today = date.today()
        expected_revenue = Decimal(expected_revenue)

        summary = reporting.sales_summary(today)
        self.assertEqual(summary['revenue_today'], expected_revenue)
        self.assertEqual(summary['units_sold'], expected_units)
        self.assertEqual(summary['orders_today'], expected_orders)

        chart = reporting.sales_chart(today)
        self.assertEqual(self._chart_revenue(chart), expected_revenue)

        dashboard = reporting.dashboard_kpis(today)
        self.assertEqual(dashboard['revenue_today'], expected_revenue)
        self.assertEqual(dashboard['units_sold_today'], expected_units)
        self.assertEqual(dashboard['orders_today'], expected_orders)
        self.assertEqual(
            self._chart_revenue(dashboard['daily_chart_data']),
            expected_revenue,
        )
        self.assertEqual(
            dashboard['best_sellers'][0]['total_qty']
            if dashboard['best_sellers'] else 0,
            expected_units,
        )

        digest = reporting.daily_digest(today)
        self.assertEqual(digest['sales']['revenue_today'], expected_revenue)
        self.assertEqual(digest['sales']['units_sold'], expected_units)
        self.assertEqual(digest['sales']['orders_today'], expected_orders)
        self.assertEqual(
            digest['top_movers'][0]['total_qty']
            if digest['top_movers'] else 0,
            expected_units,
        )

        transactions = self.client.get(reverse('order_view'))
        self.assertEqual(transactions.status_code, 200)
        self.assertEqual(transactions.context['total_revenue'], expected_revenue)
        self.assertEqual(transactions.context['total_orders'], expected_orders)
        self.assertEqual(
            self._chart_revenue(transactions.context['daily_chart_data']),
            expected_revenue,
        )
        row = transactions.context['page_obj'].object_list[0]
        self.assertEqual(row['total'], expected_revenue)

        analytics = self.client.get(
            reverse('sales_analytics'),
            {'start': today.isoformat(), 'end': today.isoformat(), 'gran': 'day'},
        )
        self.assertEqual(analytics.status_code, 200)
        self.assertEqual(
            Decimal(str(analytics.context['kpi']['revenue'])),
            expected_revenue,
        )
        self.assertEqual(analytics.context['kpi']['items'], expected_units)
        self.assertEqual(analytics.context['kpi']['orders'], expected_orders)
        self.assertEqual(
            self._chart_revenue(analytics.context['revenue_series']),
            expected_revenue,
        )

    def test_full_void_is_removed_from_live_revenue_summaries(self):
        self._record_correction(3, TransactionCorrection.TYPE_VOID)

        self._assert_every_summary(Decimal('0.00'), 0, 0)
        self.order.refresh_from_db()
        self.assertEqual(self.order.subtotal, Decimal('30.00'))
        self.assertEqual(self.order.total_price, Decimal('33.90'))

    def test_partial_return_reduces_live_revenue_and_units(self):
        self._record_correction(2, TransactionCorrection.TYPE_RETURN)

        self._assert_every_summary(Decimal('10.00'), 1, 1)
        self.order.refresh_from_db()
        self.assertEqual(self.order.subtotal, Decimal('30.00'))
        self.assertEqual(self.order.total_price, Decimal('33.90'))

    def test_void_undo_restores_revenue_without_rewriting_snapshot(self):
        correction = self._record_correction(
            3, TransactionCorrection.TYPE_VOID,
        )
        TransactionCorrectionUndo.objects.create(
            correction=correction,
            created_by=self.user,
        )

        self._assert_every_summary(Decimal('30.00'), 3, 1)
        self.order.refresh_from_db()
        self.assertEqual(self.order.subtotal, Decimal('30.00'))
        self.assertEqual(self.order.total_price, Decimal('33.90'))
        self.assertTrue(
            TransactionCorrection.objects.filter(pk=correction.pk).exists(),
        )

    def test_full_non_restocked_return_keeps_inventory_cost_in_profit(self):
        self._record_correction(
            3,
            TransactionCorrection.TYPE_RETURN,
            disposition=TransactionCorrectionLine.DISPOSITION_DAMAGED,
        )

        today = date.today()
        analytics = self.client.get(
            reverse('sales_analytics'),
            {'start': today.isoformat(), 'end': today.isoformat(), 'gran': 'day'},
        )

        self.assertEqual(analytics.status_code, 200)
        self.assertEqual(analytics.context['kpi']['revenue'], 0)
        self.assertEqual(analytics.context['kpi']['orders'], 0)
        self.assertEqual(analytics.context['kpi']['items'], 0)
        self.assertEqual(analytics.context['kpi']['profit'], -12)
        self.assertEqual(analytics.context['revenue_series'][0]['cost'], 12)
        self.assertEqual(analytics.context['revenue_series'][0]['orders'], 0)

    def test_transaction_metrics_follow_source_and_deleted_filters(self):
        checkout = CheckoutOrder.objects.create(
            user=self.user,
            status=CheckoutOrder.STATUS_SUBMITTED,
            subtotal=Decimal('5.00'),
            tax=Decimal('0.65'),
            total_price=Decimal('5.65'),
            submitted_at=now(),
        )
        CheckoutOrderItem.objects.create(
            checkout=checkout,
            product=self.product,
            product_name='No-sale item',
            product_barcode='NOSALE-1',
            price=Decimal('5.00'),
            taxable=True,
            quantity=1,
        )

        giveaway_only = self.client.get(
            reverse('order_view'), {'source': 'giveaway'},
        )
        self.assertEqual(giveaway_only.context['total_orders'], 0)
        self.assertEqual(giveaway_only.context['total_revenue'], Decimal('0.00'))
        self.assertEqual(giveaway_only.context['daily_chart_data'], [])
        self.assertEqual(
            [row['source'] for row in giveaway_only.context['page_obj'].object_list],
            ['giveaway'],
        )

        self.order.is_deleted = True
        self.order.save(update_fields=['is_deleted'])
        active_order = Order.objects.create(
            user=self.user,
            submitted=True,
            subtotal=Decimal('50.00'),
            total_price=Decimal('50.00'),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        OrderDetail.objects.create(
            order=active_order,
            product=self.product,
            product_name='Active sale',
            product_barcode='ACTIVE-1',
            quantity=1,
            price=Decimal('50.00'),
            cost_per_unit_at_sale=Decimal('20.00'),
            taxable_at_sale=False,
        )

        deleted_only = self.client.get(
            reverse('order_view'), {'status': 'deleted', 'source': 'pos'},
        )
        self.assertEqual(deleted_only.context['total_orders'], 1)
        self.assertEqual(deleted_only.context['total_revenue'], Decimal('30.00'))
        self.assertEqual(
            self._chart_revenue(deleted_only.context['daily_chart_data']),
            Decimal('30.00'),
        )
        self.assertEqual(
            [row['id'] for row in deleted_only.context['page_obj'].object_list],
            [self.order.pk],
        )
