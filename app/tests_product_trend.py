from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils.timezone import make_aware

from .models import (
    Order,
    OrderDetail,
    Product,
    ProductLot,
    StockChange,
    TransactionCorrection,
    TransactionCorrectionLine,
)
from .utils import (
    SaleRecord,
    compute_demand_trend,
    get_product_stock_records,
    get_stock_eod,
    recommend_inventory_action,
    stock_change_delta,
)
from .views import ProductTrendView


class ProductTrendLedgerRuleTests(SimpleTestCase):
    def test_all_current_physical_ledger_events_have_explicit_semantics(self):
        self.assertEqual(stock_change_delta("checkin", 3), 3)
        self.assertEqual(stock_change_delta("restoration", 3), 3)
        self.assertEqual(stock_change_delta("return", 3, "restock"), 3)
        self.assertEqual(stock_change_delta("checkout", 3), -3)
        self.assertEqual(stock_change_delta("giveaway", 3), -3)
        self.assertEqual(stock_change_delta("deletion", 3), -3)
        self.assertEqual(stock_change_delta("checkout_unfulfilled", 3), 0)
        self.assertEqual(stock_change_delta("giveaway_unfulfilled", 3), 0)
        self.assertEqual(stock_change_delta("return_no_restock", 3), 0)

    def test_void_and_undo_only_move_stock_when_the_correction_restocked(self):
        self.assertEqual(stock_change_delta("void", 2, "restock"), 2)
        self.assertEqual(stock_change_delta("void", 2, "damaged"), 0)
        self.assertEqual(stock_change_delta("correction_undo", -2, "restock"), -2)
        self.assertEqual(stock_change_delta("correction_undo", -2, "no_restock"), 0)

    def test_partial_calendar_weeks_do_not_create_a_false_trend(self):
        start = datetime(2026, 1, 7)  # Wednesday
        end = datetime(2026, 2, 3)    # Tuesday
        sales = []
        current = start.date()
        while current <= end.date():
            if current.weekday() != 6:
                sales.append(SaleRecord(1, current.isoformat()))
            current += timedelta(days=1)

        slope = compute_demand_trend(sales, [], start, end, {6})

        self.assertAlmostEqual(slope, 0.0, places=6)


class ProductTrendLedgerIntegrationTests(TestCase):
    def setUp(self):
        self.product = Product.objects.create(
            name="Correction-aware product",
            barcode="99110022",
            price=Decimal("10.00"),
            price_per_unit=Decimal("4.00"),
            quantity_in_stock=5,
        )
        self.order = Order.objects.create(submitted=True)
        self.detail = OrderDetail.objects.create(
            order=self.order,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=5,
            price=self.product.price,
        )
        self.correction = TransactionCorrection.objects.create(
            correction_type=TransactionCorrection.TYPE_VOID,
            order=self.order,
            reason="Register correction",
        )
        self.correction_line = TransactionCorrectionLine.objects.create(
            correction=self.correction,
            order_detail=self.detail,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=2,
            unit_price=self.product.price,
            disposition=TransactionCorrectionLine.DISPOSITION_RESTOCK,
        )

        self.day_one = date(2026, 1, 5)
        self.day_two = date(2026, 1, 6)
        self.day_three = date(2026, 1, 7)
        self._change("checkout", 5, self.day_one, order_detail=self.detail)
        self._change(
            "void", 2, self.day_two,
            order_detail=self.detail,
            correction_line=self.correction_line,
        )
        self._change(
            "correction_undo", -2, self.day_three,
            order_detail=self.detail,
            correction_line=self.correction_line,
        )

    def _change(self, change_type, quantity, event_date, **links):
        change = StockChange.objects.create(
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            change_type=change_type,
            quantity=quantity,
            **links,
        )
        StockChange.objects.filter(pk=change.pk).update(
            timestamp=make_aware(datetime.combine(event_date, datetime.min.time()))
        )
        return change

    def test_end_of_day_stock_replays_restocked_void_and_undo(self):
        self.assertEqual(get_stock_eod(self.product, self.day_one), 5)
        self.assertEqual(get_stock_eod(self.product, self.day_two), 7)
        self.assertEqual(get_stock_eod(self.product, self.day_three), 5)

    def test_forecast_sales_net_voids_and_void_undos(self):
        _, sales, _, _ = get_product_stock_records(
            self.product,
            self.day_one.isoformat(),
            self.day_three.isoformat(),
        )

        self.assertEqual([record.quantity for record in sales], [5, -2, 2])
        self.assertEqual(sum(record.quantity for record in sales), 5)

    def test_chart_and_history_share_the_same_ledger_rules(self):
        view = ProductTrendView()
        sold, restocked, _, _, _, _, _ = view._grouped_totals(
            self.product, self.day_one, self.day_three, "week",
        )
        history = view._calculate_historical_stock_levels(
            self.product, self.day_one, self.day_three, "week",
        )

        self.assertEqual(sum(sold), 5)
        self.assertEqual(sum(restocked), 2)
        self.assertEqual(history[-1], 5)

    def test_month_end_range_does_not_skip_february(self):
        _, _, labels, _, _, _, _ = ProductTrendView()._grouped_totals(
            self.product, date(2026, 1, 31), date(2026, 3, 1), "month",
        )

        self.assertEqual(labels, ["Jan 2026", "Feb 2026", "Mar 2026"])


class ProductTrendForecastTests(TestCase):
    def test_quantity_bearing_lots_reduce_usable_forecast_stock(self):
        today = date.today()
        product = Product.objects.create(
            name="Lot-aware product",
            barcode="88110022",
            price=Decimal("12.00"),
            price_per_unit=Decimal("5.00"),
            quantity_in_stock=10,
        )
        ProductLot.objects.create(
            product=product,
            lot_number="EXP-SOON",
            quantity_on_hand=4,
            expiry_date=today + timedelta(days=10),
        )
        ProductLot.objects.create(
            product=product,
            lot_number="SAFE",
            quantity_on_hand=6,
            expiry_date=today + timedelta(days=180),
        )

        result = recommend_inventory_action(
            product=product,
            purchase_history=[],
            sale_history=[],
            expiry_history=[],
            unfulfilled_history=[],
            timeframe_start=(today - timedelta(days=90)).isoformat(),
            timeframe_end=today.isoformat(),
            cost_per_unit=5.0,
            price_per_unit=12.0,
            granularity="month",
        )

        self.assertEqual(result["expiring_stock_units"], 4)
        self.assertEqual(result["expiry_units_at_risk"], 4)
        self.assertEqual(result["usable_stock"], 6)
        self.assertEqual(result["forecast_confidence"], "High")

    def test_forecast_demand_can_consume_stock_before_it_expires(self):
        today = date.today()
        product = Product.objects.create(
            name="FEFO demand product",
            barcode="88110023",
            price=Decimal("12.00"),
            price_per_unit=Decimal("5.00"),
            quantity_in_stock=4,
        )
        ProductLot.objects.create(
            product=product,
            lot_number="SELL-FIRST",
            quantity_on_hand=4,
            expiry_date=today + timedelta(days=10),
        )
        sales = [
            SaleRecord(1, (today - timedelta(days=offset)).isoformat())
            for offset in range(1, 91)
            if (today - timedelta(days=offset)).weekday() != 6
        ]

        result = recommend_inventory_action(
            product=product,
            purchase_history=[],
            sale_history=sales,
            expiry_history=[],
            unfulfilled_history=[],
            timeframe_start=(today - timedelta(days=90)).isoformat(),
            timeframe_end=today.isoformat(),
            cost_per_unit=5.0,
            price_per_unit=12.0,
            granularity="month",
        )

        self.assertEqual(result["expiring_stock_units"], 4)
        self.assertEqual(result["expiry_units_at_risk"], 0)
        self.assertEqual(result["usable_stock"], 4)

    def test_depleted_expired_lot_does_not_override_quantity_bearing_lots(self):
        today = date.today()
        product = Product.objects.create(
            name="Depleted lot product",
            barcode="88110024",
            price=Decimal("12.00"),
            price_per_unit=Decimal("5.00"),
            quantity_in_stock=4,
        )
        ProductLot.objects.create(
            product=product,
            lot_number="EMPTY-EXPIRED",
            quantity_on_hand=0,
            expiry_date=today - timedelta(days=2),
        )
        ProductLot.objects.create(
            product=product,
            lot_number="ACTIVE-FUTURE",
            quantity_on_hand=4,
            expiry_date=today + timedelta(days=180),
        )
        Product.objects.filter(pk=product.pk).update(
            expiry_date=today - timedelta(days=2),
        )
        product.refresh_from_db()

        result = recommend_inventory_action(
            product=product,
            purchase_history=[],
            sale_history=[],
            expiry_history=[],
            unfulfilled_history=[],
            timeframe_start=(today - timedelta(days=90)).isoformat(),
            timeframe_end=today.isoformat(),
            cost_per_unit=5.0,
            price_per_unit=12.0,
            granularity="month",
        )

        self.assertEqual(result["expiring_stock_units"], 0)
        self.assertEqual(result["usable_stock"], 4)


class ProductTrendViewTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="trend-admin",
            password="test-pass",
            is_staff=True,
        )
        self.client.force_login(self.user)
        self.product = Product.objects.create(
            name="Unique Name Lookup",
            barcode="77110022",
            price=Decimal("10.00"),
            price_per_unit=Decimal("4.00"),
            quantity_in_stock=0,
        )
        missed = StockChange.objects.create(
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            change_type="checkout_unfulfilled",
            quantity=3,
        )
        StockChange.objects.filter(pk=missed.pk).update(
            timestamp=make_aware(datetime.combine(date.today(), datetime.min.time()))
        )

    def test_unique_name_search_selects_product_and_uses_unit_cost(self):
        response = self.client.get(reverse("product_trend"), {"q": self.product.name})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["product"], self.product)
        quantity = response.context["recommendation_data"]["suggested_order_quantity"]
        self.assertGreater(
            quantity, 0, response.context["recommendation_data"],
        )
        self.assertEqual(response.context["total_price"], Decimal("4.00") * quantity)

    def test_invalid_chart_options_and_dates_are_normalized(self):
        response = self.client.get(reverse("product_trend"), {
            "type": "pie",
            "granularity": "day",
            "start": "2026-08-20",
            "end": "2026-08-10",
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["chart_type"], "bar")
        self.assertEqual(response.context["granularity"], "month")
        self.assertLess(response.context["start_date"], response.context["end_date"])
        self.assertTrue(response.context["date_range_notice"])


class ProductTrendViewportLayoutTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "product_trend.html"
        ).read_text(encoding="utf-8")

    def test_desktop_result_workspace_is_viewport_fitted(self):
        self.assertIn('trend-page{% if product %} has-product{% endif %}', self.source)
        self.assertIn("height: calc(100vh - 7.5rem);", self.source)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr);", self.source)
        self.assertIn("overflow: hidden;", self.source)
        self.assertIn("maintainAspectRatio: false", self.source)
        self.assertNotIn("height:400px", self.source)

    def test_recommendation_is_in_the_three_column_analysis_workspace(self):
        self.assertIn(".trend-content-row.has-recommendation", self.source)
        self.assertIn("minmax(290px, 0.9fr)", self.source)
        self.assertLess(
            self.source.index('class="trend-chart-panel"'),
            self.source.index('class="trend-reco-panel"'),
        )

    def test_data_typography_and_server_rendered_kpis_are_explicit(self):
        self.assertIn("font-variant-numeric: tabular-nums;", self.source)
        self.assertIn("body.app-shell .trend-page #kpi-strip .kpi-value", self.source)
        self.assertIn('<div class="kpi-label">Net Sold</div>', self.source)
        self.assertNotIn("document.getElementById('kpi-strip').innerHTML", self.source)
