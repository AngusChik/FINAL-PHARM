from datetime import date, datetime, timedelta
from decimal import Decimal
from html import unescape
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils.timezone import make_aware

from .models import (
    Category,
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
from .views import ProductDetailsView


class ProductDetailsLedgerRuleTests(SimpleTestCase):
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


class ProductDetailsLedgerIntegrationTests(TestCase):
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
        view = ProductDetailsView()
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
        _, _, labels, _, _, _, _ = ProductDetailsView()._grouped_totals(
            self.product, date(2026, 1, 31), date(2026, 3, 1), "month",
        )

        self.assertEqual(labels, ["Jan 2026", "Feb 2026", "Mar 2026"])


class ProductDetailsForecastTests(TestCase):
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

    def test_stockout_demand_reports_money_lost_without_becoming_revenue(self):
        today = date.today()
        product = Product.objects.create(
            name="Lost opportunity product",
            barcode="88110025",
            price=Decimal("12.00"),
            price_per_unit=Decimal("5.00"),
            quantity_in_stock=0,
        )
        missed = [
            SaleRecord(3, today.isoformat()),
        ]

        result = recommend_inventory_action(
            product=product,
            purchase_history=[],
            sale_history=[],
            expiry_history=[],
            unfulfilled_history=missed,
            timeframe_start=(today - timedelta(days=30)).isoformat(),
            timeframe_end=today.isoformat(),
            cost_per_unit=5.0,
            price_per_unit=12.0,
            granularity="month",
        )

        self.assertEqual(result["debug"]["true_demand"], 3)
        self.assertEqual(result["actual_profit"], 0.0)
        self.assertEqual(result["estimated_revenue_lost"], 36.0)
        self.assertEqual(result["estimated_gross_profit_lost"], 21.0)
        self.assertIn("$36.00 revenue", result["warnings"][0])
        self.assertGreater(result["suggested_order_quantity"], 0)


class ProductDetailsViewTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="details-user",
            password="test-pass",
            is_staff=False,
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name="Pain Relief")
        self.product = Product.objects.create(
            name="Unique Name Lookup",
            brand="Northwind",
            barcode="77110022",
            item_number="SKU-7711",
            price=Decimal("10.00"),
            price_per_unit=Decimal("4.00"),
            quantity_in_stock=0,
            category=self.category,
            unit_size="24 tablets",
            description="Extended product record details.",
            taxable=False,
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

    def details_url(self, product=None):
        return reverse(
            "product_details",
            kwargs={"product_id": (product or self.product).pk},
        )

    def test_login_is_required(self):
        self.client.logout()

        response = self.client.get(self.details_url())
        legacy = self.client.get(reverse("product_trend"))

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response["Location"])
        self.assertEqual(legacy.status_code, 302)
        self.assertIn(reverse("login"), legacy["Location"])

    def test_normal_user_opens_exact_product_id_and_uses_unit_cost(self):
        other = Product.objects.create(
            name=self.product.name,
            barcode="77110023",
            price=Decimal("7.00"),
            price_per_unit=Decimal("2.00"),
        )

        response = self.client.get(self.details_url(), {"q": other.barcode})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["product"], self.product)
        self.assertTemplateUsed(response, "product_details.html")
        self.assertEqual(
            response.context["estimated_revenue_lost"],
            Decimal("30.00"),
        )
        self.assertEqual(
            response.context["recommendation_data"]["estimated_revenue_lost"],
            30.0,
        )
        self.assertEqual(
            response.context["recommendation_data"]["estimated_gross_profit_lost"],
            18.0,
        )
        self.assertContains(response, "$30.00 estimated revenue lost")
        quantity = response.context["recommendation_data"]["suggested_order_quantity"]
        self.assertGreater(
            quantity, 0, response.context["recommendation_data"],
        )
        self.assertEqual(response.context["total_price"], Decimal("4.00") * quantity)

    def test_header_edit_button_opens_exact_product_and_preserves_returns(self):
        inventory_origin = (
            f"{reverse('inventory_display')}?q=pain&sort=name&direction=asc"
        )
        response = self.client.get(self.details_url(), {
            "return_to": inventory_origin,
            "granularity": "week",
            "type": "line",
        })

        source = response.content.decode()
        header = source[
            source.index('<header class="trend-header">'):
            source.index('</header>')
        ]
        match = re.search(
            r'<a href="([^"]+)"\s+'
            r'class="trend-edit-btn product-details-edit-btn"',
            header,
        )

        self.assertIsNotNone(match)
        edit_url = urlsplit(unescape(match.group(1)))
        edit_query = parse_qs(edit_url.query)
        self.assertEqual(
            edit_url.path,
            reverse("edit_product", args=[self.product.pk]),
        )
        self.assertEqual(
            edit_query["next"],
            [response.wsgi_request.get_full_path()],
        )
        self.assertEqual(edit_query["archive_next"], [inventory_origin])
        self.assertIn('>Edit Product</a>', header)
        self.assertEqual(
            source.count('class="trend-edit-btn product-details-edit-btn"'),
            1,
        )

    def test_missing_and_archived_products_return_404(self):
        missing = self.client.get(reverse(
            "product_details", kwargs={"product_id": self.product.pk + 9999},
        ))

        Product.all_objects.filter(pk=self.product.pk).update(
            archived_at=make_aware(datetime.combine(date.today(), datetime.min.time())),
            archived_by=self.user,
            archive_reason="Archived for Product Details test",
            status=False,
        )
        archived = self.client.get(self.details_url())

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(archived.status_code, 404)

    def test_invalid_chart_options_and_dates_are_normalized(self):
        response = self.client.get(self.details_url(), {
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

    def test_missed_revenue_stays_visible_without_cost_data(self):
        self.product.quantity_in_stock = 6
        self.product.price_per_unit = None
        self.product.save(update_fields=["price_per_unit", "quantity_in_stock"])
        for change_type, quantity in (("checkin", 10), ("checkout", 4)):
            change = StockChange.objects.create(
                product=self.product,
                product_name=self.product.name,
                product_barcode=self.product.barcode,
                change_type=change_type,
                quantity=quantity,
            )
            StockChange.objects.filter(pk=change.pk).update(
                timestamp=make_aware(
                    datetime.combine(date.today(), datetime.min.time())
                ) + timedelta(hours=1 if change_type == "checkin" else 2),
            )

        response = self.client.get(self.details_url())

        self.assertNotIn("recommendation_data", response.context)
        self.assertEqual(
            response.context["price_per_unit_missing_message"],
            "Adjust cost per unit to enable recommendations.",
        )
        self.assertEqual(
            response.context["estimated_revenue_lost"],
            Decimal("30.00"),
        )
        self.assertEqual(response.context["sell_through_rate"], 40.0)
        self.assertContains(response, "40.0%")
        self.assertContains(response, "$30.00 estimated revenue lost")

    def test_complete_record_lots_recent_activity_and_analysis_render(self):
        self.product.quantity_in_stock = 5
        self.product.save(update_fields=["quantity_in_stock"])
        received_at = make_aware(datetime(2025, 12, 1, 9, 30))
        active_lot = ProductLot.objects.create(
            product=self.product,
            lot_number="LOT-ACTIVE",
            quantity_on_hand=5,
            expiry_date=date(2027, 4, 30),
            received_at=received_at,
            notes="Front shelf supply",
        )
        ProductLot.objects.create(
            product=self.product,
            lot_number="LOT-ARCHIVED",
            quantity_on_hand=0,
            expiry_date=date(2026, 2, 1),
            archived_at=make_aware(datetime(2026, 2, 2, 10, 0)),
        )
        changes = []
        for index in range(22):
            change = StockChange.objects.create(
                product=self.product,
                product_name=self.product.name,
                product_barcode=self.product.barcode,
                user=self.user,
                change_type="error_add",
                quantity=1,
                note=f"Activity note {index:02d}",
            )
            StockChange.objects.filter(pk=change.pk).update(
                timestamp=(
                    make_aware(datetime.combine(date.today(), datetime.min.time()))
                    + timedelta(hours=8, minutes=index)
                ),
            )
            changes.append(change)

        response = self.client.get(self.details_url(), {
            "start": "2025-01-01",
            "end": "2025-01-31",
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["product"].brand, "Northwind")
        self.assertEqual(response.context["gross_margin_percent"], Decimal("60"))
        self.assertEqual(response.context["active_lots"], [active_lot])
        recent_changes = list(response.context["recent_changes"])
        self.assertEqual(len(recent_changes), 20)
        self.assertEqual(recent_changes[0].pk, changes[-1].pk)
        self.assertEqual(recent_changes[-1].pk, changes[2].pk)
        self.assertNotIn(changes[1].pk, [change.pk for change in recent_changes])
        self.assertContains(response, "Product Details")
        self.assertContains(response, "Product Information")
        self.assertContains(response, "Inventory Analysis")
        self.assertContains(response, "Active Lots")
        self.assertContains(response, "Recent Activity")
        self.assertContains(response, "Inventory Recommendation")
        for value in (
            self.product.name,
            "Northwind",
            "77110022",
            "SKU-7711",
            "Pain Relief",
            "24 tablets",
            "Extended product record details.",
            "LOT-ACTIVE",
            "Front shelf supply",
            "Activity note 21",
            self.user.username,
        ):
            with self.subTest(value=value):
                self.assertContains(response, value)
        self.assertNotContains(response, "LOT-ARCHIVED")
        self.assertNotContains(response, "Activity note 01")

    def test_inventory_return_url_preserves_filters_and_rejects_other_pages(self):
        origin = (
            f"{reverse('inventory_display')}?q=vitamin+d&category=2&category=7"
            "&sort=quantity_in_stock&direction=desc&page=3#product-14"
        )

        response = self.client.get(self.details_url(), {"return_to": origin})
        wrong_page = self.client.get(self.details_url(), {
            "return_to": f"{reverse('order_view')}?page=4",
        })

        self.assertEqual(response.context["return_to"], origin)
        self.assertEqual(
            wrong_page.context["return_to"], reverse("inventory_display"),
        )
        self.assertEqual(wrong_page.context["page_return"], {
            "url": reverse("inventory_display"),
            "destination": "Inventory",
            "label": "Back to Inventory",
            "source": "explicit",
        })

    def test_legacy_product_trend_redirects_to_inventory_and_keeps_only_query(self):
        response = self.client.get(reverse("product_trend"), {
            "q": "vitamin d & zinc",
            "start": "2026-01-01",
            "end": "2026-03-31",
            "granularity": "week",
            "type": "line",
        })

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response["Location"],
            f"{reverse('inventory_display')}?q=vitamin+d+%26+zinc",
        )

    def test_out_of_stock_revenue_loss_matches_product_details(self):
        response = self.client.get(reverse("out_of_stock"))

        self.assertEqual(response.context["total_missed"], 3)
        self.assertEqual(
            response.context["total_revenue_lost"],
            Decimal("30.00"),
        )


class ProductDetailsResponsiveLayoutTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "product_details.html"
        ).read_text(encoding="utf-8")

    def test_details_page_scrolls_normally_and_chart_is_responsive(self):
        self.assertIn(
            'class="product-details-page trend-page has-product"', self.source,
        )
        self.assertIn("maintainAspectRatio: false", self.source)
        self.assertNotIn("height:400px", self.source)
        self.assertNotIn("height: calc(100vh - 7.5rem);", self.source)
        self.assertIn(".details-table-wrap { overflow-x: auto; }", self.source)
        self.assertIn("height: 510px;", self.source)
        self.assertIn("height: 430px;", self.source)
        self.assertIn("height: 370px;", self.source)
        self.assertIn("box-sizing: border-box;", self.source)
        self.assertIn("padding-right: 3.75rem;", self.source)

    def test_analysis_uses_three_columns_then_collapses_for_narrow_screens(self):
        self.assertIn(
            "grid-template-columns: minmax(270px, 0.8fr) "
            "minmax(430px, 1.9fr) minmax(300px, 0.95fr);",
            self.source,
        )
        self.assertIn("@media (max-width: 1280px)", self.source)
        self.assertIn(".trend-reco-panel { grid-column: 1 / -1; }", self.source)
        self.assertIn("@media (max-width: 980px)", self.source)
        self.assertIn(
            ".trend-content-row,\n  .trend-content-row.no-recommendation,\n"
            "  .product-support-grid { grid-template-columns: 1fr; }",
            self.source,
        )
        self.assertLess(
            self.source.index('class="trend-chart-panel"'),
            self.source.index('class="trend-reco-panel"'),
        )

    def test_data_typography_and_server_rendered_kpis_are_explicit(self):
        self.assertIn("font-variant-numeric: tabular-nums;", self.source)
        self.assertIn("body.app-shell .trend-page #kpi-strip .kpi-value", self.source)
        self.assertIn('<div class="kpi-label">Net Sold</div>', self.source)
        self.assertNotIn("document.getElementById('kpi-strip').innerHTML", self.source)
