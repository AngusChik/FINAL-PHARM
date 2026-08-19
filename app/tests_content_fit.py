from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .models import Order, OrderDetail


@override_settings(AXES_ENABLED=False, GLOBAL_MAX_SESSIONS=10)
class OrderSuccessContentFitTests(TestCase):
    """Keep long receipt lines inside the enlarged, scroll-safe success card."""

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="content-fit-admin",
            password="pass1234",
            is_staff=True,
        )
        self.long_name = (
            "METAMUCIL SMOOTH TEXT SUGAR FREE BERRY 72 DOSE "
            "ULTRALONGPRODUCTIDENTIFIER1234567890"
            "ULTRALONGPRODUCTIDENTIFIER1234567890"
        )
        self.order = Order.objects.create(
            user=self.user,
            submitted=True,
            subtotal=Decimal("34.99"),
            tax=Decimal("4.55"),
            total_price=Decimal("39.54"),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        OrderDetail.objects.create(
            order=self.order,
            product_name=self.long_name,
            product_barcode="CONTENT-FIT-001",
            quantity=1,
            price=Decimal("34.99"),
            taxable_at_sale=True,
        )
        self.client.force_login(self.user)

    def test_long_product_is_rendered_inside_accessible_receipt_scroller(self):
        response = self.client.get(
            reverse("order_success", args=[self.order.order_id]),
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.long_name)
        html = response.content.decode("utf-8")
        wrapper = html.index('<div class="receipt-table-wrap"')
        table = html.index('<table class="receipt-table"', wrapper)
        product = html.index(self.long_name, table)
        wrapper_end = html.index("</div>", product)

        self.assertLess(wrapper, table)
        self.assertLess(table, product)
        self.assertLess(product, wrapper_end)
        self.assertIn('role="region"', html[wrapper:table])
        self.assertIn('aria-label="Order line items"', html[wrapper:table])
        self.assertIn('tabindex="0"', html[wrapper:table])

    def test_receipt_layout_contract_reserves_numeric_columns_and_wraps_name(self):
        template = (
            Path(settings.BASE_DIR) / "app" / "templates" / "order_success.html"
        ).read_text(encoding="utf-8")

        self.assertIn("max-width: var(--content-confirmation, 1000px)", template)
        self.assertRegex(
            template,
            r"\.receipt-table\s*\{[^}]*min-width:\s*520px;"
            r"[^}]*table-layout:\s*fixed;",
        )
        self.assertRegex(
            template,
            r"\.receipt-table-wrap\s*\{[^}]*overflow-x:\s*auto;",
        )
        self.assertRegex(
            template,
            r"\.receipt-table\s+:is\(th, td\):first-child\s*\{"
            r"[^}]*overflow-wrap:\s*anywhere;",
        )
        self.assertRegex(
            template,
            r"\.receipt-table\s+:is\(th, td\):not\(:first-child\)\s*\{"
            r"[^}]*white-space:\s*nowrap;",
        )
        self.assertIn('<col style="width:56px">', template)
        self.assertEqual(template.count('<col style="width:78px">'), 2)
        self.assertRegex(
            template,
            r"\.success-actions\s*\{[^}]*flex-wrap:\s*wrap;",
        )


class SitewideContentFitContracts(SimpleTestCase):
    def _template(self, name):
        return (
            Path(settings.BASE_DIR) / "app" / "templates" / name
        ).read_text(encoding="utf-8")

    def test_confirmation_rows_reflow_without_clipping_on_phones(self):
        checkout = self._template("checkout_success.html")
        giveaway = self._template("giveaway_detail.html")

        self.assertIn(
            ".cs-line > .cs-name { grid-column:1 / -1;", checkout,
        )
        self.assertIn(
            ".gd-line > .gd-name { grid-column:1 / -1;", giveaway,
        )
        self.assertIn(
            ".cs-total > :last-child { flex:0 0 auto; white-space:nowrap; }",
            checkout,
        )
        self.assertIn(
            ".gd-total > :last-child { flex:0 0 auto; white-space:nowrap; }",
            giveaway,
        )

    def test_workflow_navigation_wraps_above_the_phone_breakpoint(self):
        styles = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        base = self._template("base.html")

        self.assertRegex(
            styles,
            r"@media \(min-width: 769px\)\s*\{[^}]*"
            r"body\.app-shell \.workflow-nav\s*\{[^}]*flex-wrap: wrap;",
        )
        self.assertIn(
            "current_page == 'submit_order' or current_page == 'order_success'",
            base,
        )

    def test_data_heavy_tables_have_direct_scroll_fallbacks(self):
        low_stock = self._template("low_stock_trend.html")
        order_detail = self._template("order_detail.html")

        self.assertIn('<div class="table-scroll ls-table-scroll">', low_stock)
        self.assertIn("grid-template-columns: 340px minmax(0, 1fr);", order_detail)
        self.assertRegex(
            order_detail,
            r"\.order-detail-page \.table-wrap\s*\{[^}]*overflow-x:\s*auto;",
        )
        self.assertRegex(
            order_detail,
            r"\.order-detail-page table\s*\{[^}]*min-width:\s*860px;",
        )
