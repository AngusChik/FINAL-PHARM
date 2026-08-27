from datetime import date
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from . import reporting
from .models import (
    Category,
    Order,
    OrderDetail,
    PagePresence,
    Product,
    TransactionCorrection,
    TransactionCorrectionLine,
)
from .views import build_order_transaction_context


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=10)
class SeniorsDiscountWorkflowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="senior-discount-user",
            password="pass1234",
            is_staff=True,
        )
        self.other_user = User.objects.create_user(
            username="senior-discount-other",
            password="pass1234",
            is_staff=True,
        )
        category = Category.objects.create(name="Senior discount test")
        self.product = Product.objects.create(
            name="Taxable senior discount item",
            barcode="SENIOR-DISCOUNT-001",
            price=Decimal("13.99"),
            quantity_in_stock=10,
            taxable=True,
            category=category,
        )
        self.client.force_login(self.user)
        self.client.post(
            reverse("add_product_by_id", args=[self.product.pk]),
            {"quantity": "1"},
        )
        self.order = Order.objects.get(user=self.user, submitted=False)

    def test_toggle_persists_without_resetting_draft_timer(self):
        original_expiry = self.order.draft_expires_at

        response = self.client.post(
            reverse("create_order"),
            {"action": "toggle_seniors_discount"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertRedirects(response, reverse("create_order"), fetch_redirect_response=False)
        self.order.refresh_from_db()
        self.assertTrue(self.order.seniors_discount)
        self.assertEqual(self.order.draft_expires_at, original_expiry)

        page = self.client.get(reverse("create_order"))
        self.assertEqual(page.context["total_price_before_tax"], Decimal("13.99"))
        self.assertEqual(page.context["seniors_discount_amount"], Decimal("1.40"))
        self.assertEqual(page.context["tax_amount"], Decimal("1.64"))
        self.assertEqual(page.context["total_price_after_tax"], Decimal("14.23"))
        self.assertContains(page, 'aria-pressed="true"')
        self.assertContains(page, "Seniors Discount (&minus;10%)")

        response = self.client.post(
            reverse("create_order"),
            {"action": "toggle_seniors_discount"},
        )
        self.assertEqual(response.status_code, 302)
        self.order.refresh_from_db()
        self.assertFalse(self.order.seniors_discount)

    def test_discount_state_is_scoped_to_the_draft_owner(self):
        self.client.post(
            reverse("create_order"),
            {"action": "toggle_seniors_discount"},
        )

        PagePresence.objects.all().delete()
        other_browser = Client()
        other_browser.force_login(self.other_user)
        page = other_browser.get(reverse("create_order"))

        self.assertIsNone(page.context["order"])
        self.assertFalse(page.context["seniors_discount"])
        self.assertNotContains(
            page,
            '<button type="submit" class="ot-discount-toggle',
        )


class SeamlessFormActionRegressionTests(SimpleTestCase):
    def test_shared_handler_uses_unclobberable_form_attributes(self):
        script = (
            Path(settings.BASE_DIR) / "static" / "js" / "ui-system.js"
        ).read_text(encoding="utf-8")

        self.assertIn("form.getAttribute('action') || window.location.href", script)
        self.assertIn("form.getAttribute('method') || 'POST'", script)
        self.assertNotIn("fetch(form.action || window.location.href", script)

    def test_base_template_busts_stale_shared_script_cache(self):
        template = (
            Path(settings.BASE_DIR) / "app" / "templates" / "base.html"
        ).read_text(encoding="utf-8")
        embedded_template = (
            Path(settings.BASE_DIR)
            / "app"
            / "templates"
            / "ordering_sheet_embed.html"
        ).read_text(encoding="utf-8")

        self.assertIn("ui-system.js' %}?v=20260827-productenter1", template)
        self.assertIn("ui-system.js' %}?v=20260827-productenter1", embedded_template)


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=10)
class SeniorsDiscountSettlementTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="senior-settlement-admin",
            password="pass1234",
            is_staff=True,
        )
        self.client.force_login(self.user)

    def _captured_order(self, *, subtotal, discount, tax, total):
        return Order.objects.create(
            user=self.user,
            submitted=True,
            seniors_discount=True,
            subtotal=Decimal(subtotal),
            discount_amount=Decimal(discount),
            tax=Decimal(tax),
            tax_rate=Decimal("0.13"),
            total_price=Decimal(total),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )

    @staticmethod
    def _line(
            order, *, name, price, quantity=1, taxable=True, cost=None,
            product=None):
        return OrderDetail.objects.create(
            order=order,
            product=product,
            product_name=name,
            product_barcode=f"{name}-BARCODE",
            quantity=quantity,
            price=Decimal(price),
            taxable_at_sale=taxable,
            cost_per_unit_at_sale=(Decimal(cost) if cost is not None else None),
        )

    def test_detail_lines_reconcile_to_captured_mixed_order_totals(self):
        order = self._captured_order(
            subtotal="1.60", discount="0.16", tax="0.12", total="1.56",
        )
        self._line(
            order, name="Taxable", price="1.05", taxable=True, cost="0.40",
        )
        self._line(
            order, name="Exempt", price="0.55", taxable=False, cost="0.20",
        )

        context = build_order_transaction_context(order)
        rows = context["order_details_with_total"]

        self.assertEqual(
            [row["discount_share"] for row in rows],
            [Decimal("0.11"), Decimal("0.05")],
        )
        self.assertEqual(
            [row["item_tax"] for row in rows],
            [Decimal("0.12"), Decimal("0.00")],
        )
        self.assertEqual(
            [row["line_with_tax"] for row in rows],
            [Decimal("1.06"), Decimal("0.50")],
        )
        self.assertEqual(
            [row["profit"] for row in rows],
            [Decimal("0.54"), Decimal("0.30")],
        )
        self.assertEqual(
            sum((row["line_with_tax"] for row in rows), Decimal("0.00")),
            context["total_price_after_tax"],
        )
        self.assertEqual(context["total_profit"], Decimal("0.84"))
        self.assertEqual(context["net_revenue"], Decimal("1.44"))
        self.assertAlmostEqual(float(context["margin_pct"]), 58.333333, places=5)

    def test_corrections_use_order_level_cents_and_void_undo_restores_remainder(self):
        order = self._captured_order(
            subtotal="2.10", discount="0.21", tax="0.25", total="2.14",
        )
        line = self._line(
            order, name="Rounding edge", price="1.05", quantity=2,
            taxable=True, cost="0.40",
        )

        original = reporting.realized_order_financials(order)
        self.assertEqual(original["revenue"], Decimal("1.89"))
        self.assertEqual(original["total"], Decimal("2.14"))

        response = self.client.post(
            reverse("order_correction", args=[order.pk]),
            {
                "correction_type": TransactionCorrection.TYPE_RETURN,
                "reason": "One item returned",
                f"qty_{line.pk}": "1",
                f"disposition_{line.pk}": (
                    TransactionCorrectionLine.DISPOSITION_NO_RESTOCK
                ),
            },
        )
        self.assertRedirects(response, reverse("order_detail", args=[order.pk]))
        returned = TransactionCorrection.objects.get(
            correction_type=TransactionCorrection.TYPE_RETURN,
        )
        self.assertEqual(returned.adjustment_amount, Decimal("1.08"))
        remaining = reporting.realized_order_financials(order)
        self.assertEqual(remaining["subtotal"], Decimal("1.05"))
        self.assertEqual(remaining["discount_amount"], Decimal("0.11"))
        self.assertEqual(remaining["revenue"], Decimal("0.94"))
        self.assertEqual(remaining["tax"], Decimal("0.12"))
        self.assertEqual(remaining["total"], Decimal("1.06"))

        response = self.client.post(
            reverse("order_correction", args=[order.pk]),
            {
                "correction_type": TransactionCorrection.TYPE_VOID,
                "reason": "Void remaining item",
                f"disposition_{line.pk}": (
                    TransactionCorrectionLine.DISPOSITION_NO_RESTOCK
                ),
            },
        )
        self.assertRedirects(response, reverse("order_detail", args=[order.pk]))
        void = TransactionCorrection.objects.get(
            correction_type=TransactionCorrection.TYPE_VOID,
        )
        self.assertEqual(void.adjustment_amount, Decimal("1.06"))
        self.assertEqual(reporting.realized_order_financials(order)["total"], Decimal("0.00"))

        response = self.client.post(
            reverse("transaction_correction_undo", args=[void.pk]),
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["net_total_after_corrections"], Decimal("1.06"),
        )
        self.assertEqual(reporting.realized_order_financials(order)["total"], Decimal("1.06"))

        order.refresh_from_db()
        self.assertEqual(order.subtotal, Decimal("2.10"))
        self.assertEqual(order.discount_amount, Decimal("0.21"))
        self.assertEqual(order.tax, Decimal("0.25"))
        self.assertEqual(order.total_price, Decimal("2.14"))

    def test_reporting_settles_each_senior_order_before_summing(self):
        for index in range(2):
            order = self._captured_order(
                subtotal="1.05", discount="0.11", tax="0.12", total="1.06",
            )
            self._line(
                order,
                name=f"Per-order rounding {index}",
                price="1.05",
                taxable=True,
                cost="0.40",
            )

        annotated = list(
            reporting.annotate_orders_with_realized_sales(
                Order.objects.order_by("pk"),
            )
        )
        self.assertEqual(
            [order.realized_revenue for order in annotated],
            [Decimal("0.94"), Decimal("0.94")],
        )

        summary = reporting.sales_summary(date.today())
        self.assertEqual(summary["revenue_today"], Decimal("1.88"))
        self.assertEqual(summary["units_sold"], 2)
        self.assertEqual(summary["orders_today"], 2)
        self.assertEqual(
            Decimal(str(sum(row["revenue"] for row in reporting.sales_chart(date.today())))),
            Decimal("1.88"),
        )

        transactions = self.client.get(reverse("order_view"))
        self.assertEqual(transactions.status_code, 200)
        self.assertEqual(transactions.context["total_revenue"], Decimal("1.88"))
        self.assertEqual(
            Decimal(str(sum(
                row["revenue"] for row in transactions.context["daily_chart_data"]
            ))),
            Decimal("1.88"),
        )

        analytics = self.client.get(
            reverse("sales_analytics"),
            {
                "start": date.today().isoformat(),
                "end": date.today().isoformat(),
                "gran": "day",
            },
        )
        self.assertEqual(analytics.status_code, 200)
        self.assertEqual(analytics.context["kpi"]["revenue"], 1.88)
        self.assertEqual(
            Decimal(str(sum(
                row["revenue"] for row in analytics.context["revenue_series"]
            ))),
            Decimal("1.88"),
        )
        self.assertEqual(
            Decimal(str(sum(
                row["revenue"] for row in analytics.context["top_products"]
            ))),
            Decimal("1.88"),
        )
        self.assertEqual(
            Decimal(str(sum(
                row["revenue"] for row in analytics.context["category_sales"]
            ))),
            Decimal("1.88"),
        )

    def test_mixed_line_corrections_telescope_to_the_captured_total(self):
        order = self._captured_order(
            subtotal="1.60", discount="0.16", tax="0.12", total="1.56",
        )
        taxable_line = self._line(
            order, name="Mixed taxable", price="1.05", taxable=True,
        )
        exempt_line = self._line(
            order, name="Mixed exempt", price="0.55", taxable=False,
        )

        self.client.post(
            reverse("order_correction", args=[order.pk]),
            {
                "correction_type": TransactionCorrection.TYPE_RETURN,
                "reason": "Return taxable line",
                f"qty_{taxable_line.pk}": "1",
                f"qty_{exempt_line.pk}": "0",
                f"disposition_{taxable_line.pk}": (
                    TransactionCorrectionLine.DISPOSITION_NO_RESTOCK
                ),
            },
        )
        returned = TransactionCorrection.objects.get(
            correction_type=TransactionCorrection.TYPE_RETURN,
        )
        self.assertEqual(returned.adjustment_amount, Decimal("1.07"))
        self.assertEqual(
            reporting.realized_order_financials(order)["total"],
            Decimal("0.49"),
        )

        self.client.post(
            reverse("order_correction", args=[order.pk]),
            {
                "correction_type": TransactionCorrection.TYPE_VOID,
                "reason": "Void exempt remainder",
                f"disposition_{exempt_line.pk}": (
                    TransactionCorrectionLine.DISPOSITION_NO_RESTOCK
                ),
            },
        )
        void = TransactionCorrection.objects.get(
            correction_type=TransactionCorrection.TYPE_VOID,
        )
        self.assertEqual(void.adjustment_amount, Decimal("0.49"))
        self.assertEqual(
            returned.adjustment_amount + void.adjustment_amount,
            order.total_price,
        )

        self.client.post(
            reverse("transaction_correction_undo", args=[void.pk]),
        )
        self.assertEqual(
            reporting.realized_order_financials(order)["total"],
            Decimal("0.49"),
        )

    def test_analytics_preserves_an_uncorrected_captured_snapshot(self):
        order = self._captured_order(
            subtotal="1.05", discount="0.10", tax="0.12", total="1.07",
        )
        self._line(
            order, name="Historical snapshot", price="1.05", taxable=True,
        )
        lines = list(
            reporting.realized_sales_lines(order.details.all())
            .select_related("order", "product__category")
            .order_by("order_id", "pk")
        )

        with self.assertNumQueries(1):
            rows = reporting.settled_realized_sales_rows(lines)

        self.assertEqual(sum(row["revenue"] for row in rows), Decimal("0.95"))
        self.assertEqual(
            reporting.realized_order_financials(order)["total"],
            Decimal("1.07"),
        )

    def test_ignore_snacks_recalculates_discount_on_included_lines(self):
        snacks = Category.objects.create(name=reporting.SNACKS_CATEGORY_NAME)
        medicine = Category.objects.create(name="Non-snack senior reporting")
        snack_product = Product.objects.create(
            name="Snack line",
            barcode="SENIOR-SNACK-001",
            price=Decimal("1.05"),
            quantity_in_stock=1,
            taxable=True,
            category=snacks,
        )
        medicine_product = Product.objects.create(
            name="Medicine line",
            barcode="SENIOR-MED-001",
            price=Decimal("0.55"),
            quantity_in_stock=1,
            taxable=False,
            category=medicine,
        )
        order = self._captured_order(
            subtotal="1.60", discount="0.16", tax="0.12", total="1.56",
        )
        self._line(
            order, name="Snack line", price="1.05", taxable=True,
            product=snack_product,
        )
        self._line(
            order, name="Medicine line", price="0.55", taxable=False,
            product=medicine_product,
        )

        full = reporting.sales_summary(date.today())
        without_snacks = reporting.sales_summary(
            date.today(), exclude_snacks=True,
        )

        self.assertEqual(full["revenue_today"], Decimal("1.44"))
        self.assertEqual(without_snacks["revenue_today"], Decimal("0.49"))
        self.assertEqual(without_snacks["units_sold"], 1)
        self.assertEqual(without_snacks["orders_today"], 1)
