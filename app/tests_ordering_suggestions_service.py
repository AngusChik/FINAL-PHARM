from datetime import datetime, time, timedelta
from decimal import Decimal

from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from app.models import (
    Category,
    Order,
    OrderDetail,
    Product,
    ProductLot,
    RecentlyPurchasedProduct,
    StockChange,
    SupplierPurchaseOrder,
    SupplierPurchaseOrderLine,
    TransactionCorrection,
    TransactionCorrectionLine,
    TransactionCorrectionUndo,
)
from app.ordering_suggestions import (
    DemandDay,
    _build_daily_history,
    _history_agreement,
    adaptive_daily_forecast,
    build_ordering_suggestions,
    calculate_dependable_stock,
    calculate_safety_stock,
    calculate_window_metrics,
    classify_demand_pattern,
    classify_momentum,
    simulate_scheduled_coverage,
    sort_suggestions,
    tsb_daily_rate,
)
from app.utils import count_open_days


class OrderingSuggestionMathTests(SimpleTestCase):
    def setUp(self):
        self.today = timezone.localdate()

    def _open_points(self, count, *, end=None, value=0):
        end = end or (self.today - timedelta(days=1))
        points = []
        cursor = end
        while len(points) < count:
            if cursor.weekday() != 6:
                points.append((cursor, value))
            cursor -= timedelta(days=1)
        return list(reversed(points))

    def test_zero_demand_does_not_create_fallback_order(self):
        points = self._open_points(90, value=0)

        result = adaptive_daily_forecast(points)

        self.assertEqual(result["daily_rate"], 0)
        self.assertEqual(result["pattern"]["label"], "No recorded demand")

    def test_tsb_forecast_declines_during_long_zero_run(self):
        active = [0, 0, 4, 0, 0, 3]
        after_short_gap = tsb_daily_rate(active + [0] * 3)
        after_long_gap = tsb_daily_rate(active + [0] * 30)

        self.assertGreater(after_short_gap, after_long_gap)
        self.assertGreater(after_long_gap, 0)

    def test_demand_pattern_selects_plain_intermittent_label(self):
        result = classify_demand_pattern([3] + [0] * 20 + [3] + [0] * 20)

        self.assertEqual(result["internal"], "intermittent")
        self.assertEqual(result["label"], "Occasional")

    def test_adaptive_model_selects_smooth_and_lumpy_methods(self):
        smooth_points = self._open_points(30, value=2)
        lumpy_values = [10] + [0] * 10 + [1] + [0] * 10
        lumpy_points = [
            (day, value)
            for (day, _), value in zip(self._open_points(len(lumpy_values)), lumpy_values)
        ]

        smooth = adaptive_daily_forecast(smooth_points)
        lumpy = adaptive_daily_forecast(lumpy_points)

        self.assertEqual(smooth["pattern"]["internal"], "smooth")
        self.assertEqual(smooth["model"], "recency_weighted")
        self.assertEqual(lumpy["pattern"]["internal"], "lumpy")
        self.assertEqual(lumpy["model"], "intermittent")

    def test_partial_window_uses_only_observable_open_days(self):
        end = self.today - timedelta(days=1)
        available_dates = []
        cursor = end - timedelta(days=39)
        while cursor <= end:
            if cursor.weekday() != 6:
                available_dates.append(cursor)
            cursor += timedelta(days=1)
        history = [
            DemandDay(
                day=day,
                fulfilled=2 if index == 0 else 0,
                observable=index % 2 == 0,
            )
            for index, day in enumerate(available_dates)
        ]
        future_open_days = count_open_days(self.today, self.today + timedelta(days=29))

        result = calculate_window_metrics(
            history,
            end_day=end,
            horizon_days=90,
            label="Recent demand",
            future_open_days=future_open_days,
        )

        expected_observed = len([record for record in history if record.observable])
        self.assertEqual(result["observable_open_days"], expected_observed)
        self.assertEqual(result["total_demand"], 2)
        self.assertAlmostEqual(
            result["monthly_units"],
            round(2 / expected_observed * future_open_days, 1),
        )
        self.assertLess(result["history_days"], result["requested_days"])

    def test_leading_sunday_counts_toward_full_calendar_coverage(self):
        end = self.today - timedelta(days=(self.today.weekday() - 6) % 7)
        start = end - timedelta(days=364)
        self.assertEqual(start.weekday(), 6)
        history = [DemandDay(day=day) for day in self._open_points_between(start, end)]

        result = calculate_window_metrics(
            history,
            end_day=end,
            horizon_days=365,
            label="Long-term view",
            future_open_days=26,
        )

        self.assertEqual(result["history_days"], 365)
        self.assertEqual(result["coverage_label"], "365 of 365 days available")

    def test_momentum_requires_material_and_statistically_credible_change(self):
        future_days = 26
        stable = classify_momentum(
            {"observable_open_days": 78, "total_demand": 31},
            {"observable_open_days": 78, "total_demand": 30},
            future_open_days=future_days,
        )
        rising = classify_momentum(
            {"observable_open_days": 78, "total_demand": 100},
            {"observable_open_days": 78, "total_demand": 20},
            future_open_days=future_days,
        )

        self.assertEqual(stable, "stable")
        self.assertEqual(rising, "rising")

    def test_momentum_detects_credible_fall(self):
        result = classify_momentum(
            {"observable_open_days": 78, "total_demand": 20},
            {"observable_open_days": 78, "total_demand": 100},
            future_open_days=26,
        )

        self.assertEqual(result, "falling")

    def test_history_disagreement_uses_relative_or_absolute_threshold(self):
        def window(monthly):
            return {
                "monthly_units": monthly,
                "observable_open_days": 60,
                "history_days": 90,
                "days": 90,
            }

        _, relative_disagreement = _history_agreement([window(1), window(2)])
        _, absolute_disagreement = _history_agreement([window(10), window(12.1)])

        self.assertTrue(relative_disagreement)
        self.assertTrue(absolute_disagreement)

    def test_fefo_excludes_units_unlikely_to_sell_before_expiry(self):
        result = calculate_dependable_stock(
            [
                {
                    "quantity_on_hand": 10,
                    "expiry_date": self.today + timedelta(days=2),
                },
                {"quantity_on_hand": 4, "expiry_date": None},
            ],
            as_of=self.today,
            daily_rate=1,
        )

        # Three open days at most can be consumed before the dated lot expires;
        # undated stock remains dependable.
        dated_capacity = count_open_days(self.today, self.today + timedelta(days=2))
        self.assertEqual(result["stock_on_hand"], 14)
        self.assertEqual(result["dependable_stock"], dated_capacity + 4)
        self.assertEqual(result["expiry_units_at_risk"], 10 - dated_capacity)

    def test_safety_stock_uses_fallback_without_twelve_rolling_origins(self):
        points = self._open_points(10, value=1)

        safety_stock, origins = calculate_safety_stock(
            points,
            expected_lead_time_demand=4,
        )

        self.assertLess(origins, 12)
        self.assertEqual(safety_stock, 4)

    def test_safety_stock_uses_empirical_errors_with_twelve_origins(self):
        points = self._open_points(90, value=1)

        safety_stock, origins = calculate_safety_stock(
            points,
            expected_lead_time_demand=9,
        )

        self.assertGreaterEqual(origins, 12)
        self.assertEqual(safety_stock, 0)

    def test_sparse_spike_remains_in_positive_error_percentile(self):
        points = self._open_points(90, value=0)
        spike_day, _ = points[60]
        points[60] = (spike_day, 10)

        safety_stock, origins = calculate_safety_stock(
            points,
            expected_lead_time_demand=0,
        )

        self.assertGreaterEqual(origins, 12)
        self.assertEqual(safety_stock, 10)

    def test_future_receipt_does_not_cover_pre_delivery_gap(self):
        due_date = self.today + timedelta(days=7)

        result = simulate_scheduled_coverage(
            dependable_stock=0,
            incoming_schedule=[{"expected_date": due_date, "quantity": 7}],
            daily_rate=1,
            as_of=self.today,
        )

        expected_gap = count_open_days(self.today, self.today + timedelta(days=6))
        self.assertEqual(result["coverage_days"], 0)
        self.assertEqual(result["lead_gap_units"], expected_gap)
        self.assertEqual(result["next_incoming_date"], due_date)

    def test_suggestion_sort_is_action_first_then_stable_by_name_and_id(self):
        unordered = [
            {"classification": "wait_for_now", "name": "Alpha", "product_id": 1},
            {"classification": "needs_attention", "name": "Beta", "product_id": 2},
            {"classification": "order_now", "name": "Zulu", "product_id": 3},
            {"classification": "order_soon", "name": "Gamma", "product_id": 4},
            {"classification": "order_now", "name": "Alpha", "product_id": 5},
        ]

        result = sort_suggestions(unordered)

        self.assertEqual(
            [(item["classification"], item["name"]) for item in result],
            [
                ("order_now", "Alpha"),
                ("order_now", "Zulu"),
                ("order_soon", "Gamma"),
                ("needs_attention", "Beta"),
                ("wait_for_now", "Alpha"),
            ],
        )

    def test_unavailable_zero_days_are_censored_but_recorded_demand_is_observable(self):
        open_days = [day for day, _ in self._open_points(3)]
        start, sale_day, end = open_days
        product = type("ProductStub", (), {"pk": 42})()

        history = _build_daily_history(
            product,
            start=start,
            end=end,
            as_of=self.today,
            current_stock=0,
            fulfilled={(42, sale_day): 1},
            unfilled={},
            movements={},
            positive_inflow={},
        )

        by_day = {record.day: record for record in history}
        self.assertTrue(by_day[sale_day].observable)
        self.assertFalse(by_day[end].observable)

    @staticmethod
    def _open_points_between(start, end):
        cursor = start
        result = []
        while cursor <= end:
            if cursor.weekday() != 6:
                result.append(cursor)
            cursor += timedelta(days=1)
        return result


class OrderingSuggestionDatabaseTests(TestCase):
    def setUp(self):
        self.today = timezone.localdate()
        self.category = Category.objects.create(name="Cold remedies")
        self.product = Product.objects.create(
            name="Test medicine",
            brand="Example",
            barcode="555123",
            item_number="ABC-1",
            price=Decimal("9.99"),
            price_per_unit=Decimal("4.00"),
            quantity_in_stock=6,
            category=self.category,
            status=True,
        )
        old_time = timezone.now() - timedelta(days=120)
        Product.all_objects.filter(pk=self.product.pk).update(created_at=old_time)
        self.product.refresh_from_db()
        ProductLot.objects.create(
            product=self.product,
            lot_number="LOT-1",
            quantity_on_hand=6,
            expiry_date=self.today + timedelta(days=365),
        )
        self.recent = RecentlyPurchasedProduct.objects.create(
            product=self.product,
            quantity=99,  # The advisory service must ignore this legacy counter.
        )

        # This receiving row establishes the trustworthy ledger start and keeps
        # the product available during the recorded period.
        checkin = StockChange.objects.create(
            product=self.product,
            change_type="checkin",
            quantity=6,
        )
        StockChange.objects.filter(pk=checkin.pk).update(
            timestamp=timezone.now() - timedelta(days=100),
        )

    def _dated_timestamp(self, day):
        value = datetime.combine(day, time(hour=12))
        return timezone.make_aware(value) if timezone.is_naive(value) else value

    def _create_demand_product(
        self,
        *,
        name,
        history_days,
        current_stock,
        demand_for_day,
    ):
        start = self.today - timedelta(days=history_days)
        product = Product.objects.create(
            name=name,
            barcode=f"TEST-{name}",
            price=Decimal("5.00"),
            quantity_in_stock=current_stock,
            category=self.category,
            status=True,
        )
        Product.all_objects.filter(pk=product.pk).update(
            created_at=self._dated_timestamp(start),
        )
        product.refresh_from_db()
        if current_stock:
            ProductLot.objects.create(
                product=product,
                lot_number="CURRENT",
                quantity_on_hand=current_stock,
                expiry_date=None,
            )
        recent = RecentlyPurchasedProduct.objects.create(product=product)

        demand_rows = []
        cursor = start + timedelta(days=1)
        while cursor < self.today:
            if cursor.weekday() != 6:
                quantity = max(0, int(demand_for_day(cursor)))
                if quantity:
                    demand_rows.append((cursor, quantity))
            cursor += timedelta(days=1)

        initial_stock = current_stock + sum(quantity for _, quantity in demand_rows)
        checkin = StockChange.objects.create(
            product=product,
            change_type="checkin",
            quantity=initial_stock,
        )
        StockChange.objects.filter(pk=checkin.pk).update(
            timestamp=self._dated_timestamp(start),
        )

        orders = Order.objects.bulk_create([
            Order(submitted=True) for _ in demand_rows
        ])
        for order, (day, _) in zip(orders, demand_rows):
            order.order_date = self._dated_timestamp(day)
        Order.objects.bulk_update(orders, ["order_date"])
        OrderDetail.objects.bulk_create([
            OrderDetail(
                order=order,
                product=product,
                product_name=product.name,
                product_barcode=product.barcode,
                quantity=quantity,
                price=product.price,
                order_date=self._dated_timestamp(day),
            )
            for order, (day, quantity) in zip(orders, demand_rows)
        ])
        checkout_rows = StockChange.objects.bulk_create([
            StockChange(
                product=product,
                change_type="checkout",
                quantity=quantity,
            )
            for _, quantity in demand_rows
        ])
        for stock_change, (day, _) in zip(checkout_rows, demand_rows):
            stock_change.timestamp = self._dated_timestamp(day)
        StockChange.objects.bulk_update(checkout_rows, ["timestamp"])
        return recent, product

    def _completed_sale(self, *, days_ago, quantity):
        order = Order.objects.create(
            submitted=True,
            subtotal=Decimal(quantity * 10),
            total_price=Decimal(quantity * 10),
        )
        detail = OrderDetail.objects.create(
            order=order,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=quantity,
            price=Decimal("10.00"),
        )
        sale_time = timezone.now() - timedelta(days=days_ago)
        Order.objects.filter(pk=order.pk).update(order_date=sale_time)
        OrderDetail.objects.filter(pk=detail.pk).update(order_date=sale_time)
        return order, detail

    def test_service_uses_corrected_original_sale_and_missed_demand_without_writes(self):
        order, detail = self._completed_sale(days_ago=20, quantity=10)
        correction = TransactionCorrection.objects.create(
            correction_type=TransactionCorrection.TYPE_RETURN,
            order=order,
            reason="Customer return",
            adjustment_amount=Decimal("20.00"),
        )
        TransactionCorrectionLine.objects.create(
            correction=correction,
            order_detail=detail,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=2,
            unit_price=Decimal("10.00"),
            disposition=TransactionCorrectionLine.DISPOSITION_RESTOCK,
        )
        missed = StockChange.objects.create(
            product=self.product,
            order_detail=detail,
            change_type="checkout_unfulfilled",
            quantity=3,
        )
        StockChange.objects.filter(pk=missed.pk).update(
            timestamp=timezone.now() - timedelta(days=20),
        )
        purchase_order = SupplierPurchaseOrder.objects.create(
            supplier=SupplierPurchaseOrder.SUPPLIER_MCKESSON,
            status=SupplierPurchaseOrder.STATUS_SUBMITTED,
            expected_date=self.today + timedelta(days=3),
        )
        SupplierPurchaseOrderLine.objects.create(
            purchase_order=purchase_order,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity_ordered=9,
            quantity_received=2,
        )
        tracked_models = (
            RecentlyPurchasedProduct,
            SupplierPurchaseOrder,
            SupplierPurchaseOrderLine,
            StockChange,
        )
        before = {model: model.objects.count() for model in tracked_models}

        result = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=self.recent.pk),
            as_of=self.today,
        )

        suggestion = result["suggestions"][0]
        recent_window = suggestion["windows"][0]
        self.assertEqual(recent_window["fulfilled_units"], 8)
        self.assertEqual(recent_window["unfilled_units"], 3)
        self.assertEqual(recent_window["total_demand"], 11)
        self.assertEqual(suggestion["confirmed_incoming"], 7)
        self.assertEqual(suggestion["timely_incoming"], 7)
        self.assertEqual(suggestion["history_signal"], "Limited history")
        self.assertEqual(suggestion["longer_movement"], "insufficient")
        self.assertEqual(result["summary"]["total"], 1)
        self.assertEqual(
            {model: model.objects.count() for model in tracked_models}, before,
        )

    def test_correction_undo_restores_fulfilled_demand(self):
        order, detail = self._completed_sale(days_ago=10, quantity=5)
        correction = TransactionCorrection.objects.create(
            correction_type=TransactionCorrection.TYPE_VOID,
            order=order,
            reason="Entered in error",
            adjustment_amount=Decimal("50.00"),
        )
        TransactionCorrectionLine.objects.create(
            correction=correction,
            order_detail=detail,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=5,
            unit_price=Decimal("10.00"),
            disposition=TransactionCorrectionLine.DISPOSITION_RESTOCK,
        )
        TransactionCorrectionUndo.objects.create(
            correction=correction,
            reason="Void was accidental",
        )

        result = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=self.recent.pk),
            as_of=self.today,
        )

        self.assertEqual(result["suggestions"][0]["windows"][0]["fulfilled_units"], 5)

    def test_draft_supplier_orders_are_not_counted_as_incoming(self):
        draft = SupplierPurchaseOrder.objects.create(
            supplier=SupplierPurchaseOrder.SUPPLIER_OTHER,
            supplier_name="Example supplier",
            status=SupplierPurchaseOrder.STATUS_DRAFT,
            expected_date=self.today + timedelta(days=2),
        )
        SupplierPurchaseOrderLine.objects.create(
            purchase_order=draft,
            product=self.product,
            product_name=self.product.name,
            quantity_ordered=50,
        )

        suggestion = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=self.recent.pk),
            as_of=self.today,
        )["suggestions"][0]

        self.assertEqual(suggestion["confirmed_incoming"], 0)

    def test_live_missed_demand_never_returns_order_zero_or_gets_hidden(self):
        product = Product.objects.create(
            name="Unavailable medicine",
            barcode="LIVE-1",
            price=Decimal("5.00"),
            quantity_in_stock=0,
            category=self.category,
            status=True,
        )
        recent = RecentlyPurchasedProduct.objects.create(product=product)
        StockChange.objects.create(
            product=product,
            change_type="checkout_unfulfilled",
            quantity=4,
        )

        suggestion = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=recent.pk),
            as_of=self.today,
        )["suggestions"][0]

        self.assertEqual(suggestion["classification"], "order_now")
        self.assertGreaterEqual(suggestion["suggested_quantity"], 4)
        self.assertEqual(suggestion["action_label"], "Order 4 now")
        self.assertIn("today", suggestion["reason"].lower())

    def test_scheduled_day_seven_receipt_keeps_immediate_gap_order_now(self):
        recent, product = self._create_demand_product(
            name="Scheduled receipt",
            history_days=120,
            current_stock=0,
            demand_for_day=lambda _day: 1,
        )
        due_date = self.today + timedelta(days=7)
        purchase_order = SupplierPurchaseOrder.objects.create(
            supplier=SupplierPurchaseOrder.SUPPLIER_MCKESSON,
            status=SupplierPurchaseOrder.STATUS_SUBMITTED,
            expected_date=due_date,
        )
        SupplierPurchaseOrderLine.objects.create(
            purchase_order=purchase_order,
            product=product,
            product_name=product.name,
            product_barcode=product.barcode,
            quantity_ordered=7,
        )

        suggestion = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=recent.pk),
            as_of=self.today,
        )["suggestions"][0]

        self.assertEqual(suggestion["classification"], "order_now")
        expected_cycle_need = count_open_days(
            self.today, self.today + timedelta(days=29),
        ) + suggestion["safety_stock"] - 7
        self.assertEqual(suggestion["suggested_quantity"], expected_cycle_need)
        self.assertEqual(suggestion["coverage_days"], 0)
        self.assertIn("before the next confirmed delivery", suggestion["reason"])
        self.assertIn(f"{due_date.strftime('%b')} {due_date.day}", suggestion["incoming_note"])

    def test_stable_demand_produces_order_soon_and_wait_end_to_end(self):
        soon, _ = self._create_demand_product(
            name="Order soon product",
            history_days=120,
            current_stock=10,
            demand_for_day=lambda _day: 1,
        )
        wait, _ = self._create_demand_product(
            name="Wait product",
            history_days=120,
            current_stock=50,
            demand_for_day=lambda _day: 1,
        )

        result = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk__in=[soon.pk, wait.pk]),
            as_of=self.today,
        )
        by_id = {item["recent_purchase_id"]: item for item in result["suggestions"]}

        self.assertEqual(by_id[soon.pk]["classification"], "order_soon")
        self.assertGreater(by_id[soon.pk]["suggested_quantity"], 0)
        self.assertEqual(by_id[wait.pk]["classification"], "wait_for_now")
        self.assertEqual(by_id[wait.pk]["suggested_quantity"], 0)

    def test_two_week_balance_is_compared_with_reorder_point(self):
        recent, _ = self._create_demand_product(
            name="Fourteen day reorder product",
            history_days=120,
            current_stock=18,
            demand_for_day=lambda _day: 1,
        )
        monday_as_of = self.today + timedelta(days=(7 - self.today.weekday()) % 7)
        if monday_as_of == self.today:
            monday_as_of += timedelta(days=7)

        suggestion = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=recent.pk),
            as_of=monday_as_of,
        )["suggestions"][0]

        self.assertEqual(monday_as_of.weekday(), 0)
        self.assertEqual(suggestion["classification"], "order_soon")
        self.assertGreater(suggestion["suggested_quantity"], 0)

    def test_day_thirty_receipt_is_outside_thirty_day_forecast_window(self):
        expected_date = self.today + timedelta(days=30)
        purchase_order = SupplierPurchaseOrder.objects.create(
            supplier=SupplierPurchaseOrder.SUPPLIER_MCKESSON,
            status=SupplierPurchaseOrder.STATUS_SUBMITTED,
            expected_date=expected_date,
        )
        SupplierPurchaseOrderLine.objects.create(
            purchase_order=purchase_order,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity_ordered=4,
        )

        suggestion = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=self.recent.pk),
            as_of=self.today,
        )["suggestions"][0]

        self.assertEqual(suggestion["confirmed_incoming"], 4)
        self.assertEqual(suggestion["timely_incoming"], 0)

    def test_conflicting_recent_and_steady_history_requires_review(self):
        cutoff = self.today - timedelta(days=90)
        recent, _ = self._create_demand_product(
            name="Mixed history product",
            history_days=180,
            current_stock=45,
            demand_for_day=lambda day: 2 if day >= cutoff else 0,
        )

        suggestion = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=recent.pk),
            as_of=self.today,
        )["suggestions"][0]

        self.assertEqual(suggestion["classification"], "needs_attention")
        self.assertEqual(suggestion["history_signal"], "History is mixed")
        self.assertIn("different actions", suggestion["reason"])

    def test_unknown_and_overdue_incoming_are_displayed_but_excluded(self):
        recent, product = self._create_demand_product(
            name="Uncertain delivery product",
            history_days=120,
            current_stock=10,
            demand_for_day=lambda _day: 1,
        )
        for expected_date in (None, self.today - timedelta(days=1)):
            purchase_order = SupplierPurchaseOrder.objects.create(
                supplier=SupplierPurchaseOrder.SUPPLIER_OTHER,
                supplier_name="Test supplier",
                status=SupplierPurchaseOrder.STATUS_SUBMITTED,
                expected_date=expected_date,
            )
            SupplierPurchaseOrderLine.objects.create(
                purchase_order=purchase_order,
                product=product,
                product_name=product.name,
                product_barcode=product.barcode,
                quantity_ordered=5,
            )

        suggestion = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=recent.pk),
            as_of=self.today,
        )["suggestions"][0]

        self.assertEqual(suggestion["classification"], "needs_attention")
        self.assertEqual(suggestion["confirmed_incoming"], 10)
        self.assertEqual(suggestion["timely_incoming"], 0)
        self.assertIn("5 with no delivery date", suggestion["incoming_note"])
        self.assertIn("5 overdue", suggestion["incoming_note"])
        self.assertIn("not counted yet", suggestion["incoming_note"])

    def test_stock_lot_integrity_mismatch_requires_review(self):
        Product.objects.filter(pk=self.product.pk).update(quantity_in_stock=7)

        suggestion = build_ordering_suggestions(
            RecentlyPurchasedProduct.objects.filter(pk=self.recent.pk),
            as_of=self.today,
        )["suggestions"][0]

        self.assertEqual(suggestion["classification"], "needs_attention")
        self.assertIn("stock and lot totals do not match", suggestion["reason"])
