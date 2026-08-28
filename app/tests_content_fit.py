from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .models import Order, OrderDetail


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=10)
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

    def test_readability_scale_enlarges_type_without_enlarging_spacing(self):
        tokens = (
            Path(settings.BASE_DIR) / "static" / "css" / "tokens.css"
        ).read_text(encoding="utf-8")
        styles = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        base = self._template("base.html")

        self.assertIn("--text-xs: 0.84375rem;", tokens)
        self.assertIn("--text-sm: 0.984375rem;", tokens)
        self.assertIn("--text-base: 1.125rem;", tokens)
        self.assertIn("--space-4: 1rem;", tokens)
        self.assertIn("font-size: var(--text-base);", styles)
        self.assertRegex(
            styles,
            r"body\.app-shell \.nav-content > \.nav-links > li > a \{[^}]+"
            r"font-size: 0\.875rem;",
        )
        self.assertRegex(
            styles,
            r"body\.app-shell \.nav-links li a \{[^}]+font-size: 0\.9525rem;",
        )
        self.assertIn("tokens.css' %}?v=20260826-navtext1", base)
        self.assertIn("ui-system.css' %}?v=20260828-devenv1", base)

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

    def test_shared_workflow_strip_is_removed_without_disabling_shortcuts(self):
        base = self._template("base.html")
        shared_ui = (
            Path(settings.BASE_DIR) / "static" / "js" / "ui-system.js"
        ).read_text(encoding="utf-8")

        self.assertNotIn('class="workflow-nav"', base)
        self.assertNotIn("partials/_workflow_shortcut_decal.html", base)
        self.assertNotIn('class="workflow-guide-button"', base)
        self.assertIn('data-ui-open-shortcuts', base)
        for shortcut in ("Alt + I", "Alt + E", "Alt + R", "Alt + T", "Alt + G", "Alt + L"):
            self.assertIn(shortcut, shared_ui)
        for destination in (
            "inventory_display", "low_stock", "order_view", "ordering_sheet",
            "label_printing",
        ):
            self.assertIn(f'{{% url \'{destination}\' %}}', base)

    def test_closed_side_tabs_share_one_desktop_rail(self):
        styles = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")

        desktop_rail = styles[styles.index("/* Closed desktop pull-out tabs") :]
        desktop_rail = desktop_rail[:desktop_rail.index("@media (max-width: 768px)")]

        for custom_property in (
            "--ui-side-tab-top: clamp(44px, 8vh, 70px);",
            "--ui-side-tab-width: 44px;",
            "--ui-side-tab-height: clamp(86px, 13.25vh, 104px);",
            "--ui-side-tab-gap: clamp(3px, 0.7vh, 6px);",
        ):
            self.assertIn(custom_property, desktop_rail)

        for tab_class in (
            ".ps-slider-toggle",
            ".os-slider-toggle",
            ".ps-home-toggle",
            ".os-home-toggle",
            ".ps-wrap-toggle",
            ".os-wrap-toggle",
            ".sa-slider-toggle",
            ".sl-slider-toggle",
            ".el-home-toggle",
            ".el-slider-toggle",
            ".rs-slider-toggle",
            ".lp-history-tab",
        ):
            self.assertIn(tab_class, desktop_rail)

        self.assertIn("right: 0 !important;", desktop_rail)
        self.assertIn("gap: var(--ui-side-tab-gap) !important;", desktop_rail)
        self.assertIn(
            ".slider-toggles-wrap:has(> button:nth-child(6))",
            desktop_rail,
        )
        self.assertIn(
            "--ui-side-tab-height: clamp(80px, 12vh, 100px);",
            desktop_rail,
        )
        self.assertIn("flex: 0 0 var(--ui-side-tab-height) !important;", desktop_rail)
        self.assertIn("transform: none !important;", desktop_rail)

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

    def test_purchase_summary_keeps_completion_at_the_sticky_top(self):
        purchase = self._template("order_form.html")

        action = purchase.index('<div class="ot-box-primary-action">')
        total = purchase.index('<div class="ot-total-hero">', action)
        line_items = purchase.index('<div class="ot-line-items-wrap">', total)

        self.assertLess(action, total)
        self.assertLess(total, line_items)
        self.assertIn(
            "grid-template-columns: minmax(300px, 3fr) minmax(0, 7fr);",
            purchase,
        )
        self.assertIn("grid-template-rows: auto minmax(0, 1fr);", purchase)
        self.assertIn("grid-row: 1 / span 2;", purchase)
        self.assertIn("LEFT 30%: ORDER SUMMARY BELOW BARCODE", purchase)
        heading = purchase.index('<div class="scan-product-heading">')
        lookup = purchase.index('<div class="form-group autocomplete-wrapper', heading)
        heading_region = purchase[heading:lookup]
        self.assertLess(heading_region.index("Scan product"), heading_region.index("Ready"))
        self.assertEqual(purchase.count('<span class="hint-ready">Ready</span>'), 1)
        self.assertIn("position: sticky; top: 12px;", purchase)
        self.assertIn("max-height: calc(100vh - 24px);", purchase)
        self.assertIn("flex: 1 1 auto; min-height: 0; overflow-y: auto;", purchase)
        self.assertNotIn("order: -1;", purchase)
        self.assertIn(".ot-line-items-wrap { max-height: 360px; }", purchase)
        self.assertIn('aria-keyshortcuts="Shift+Enter"', purchase)
        self.assertIn("!event.shiftKey || event.ctrlKey", purchase)
        self.assertNotIn("const submitOrderForm    =", purchase)
        self.assertNotIn("const submitOrderButton  =", purchase)

        shortcut_start = purchase.index(
            "document.addEventListener('keydown', function (event) {",
            purchase.index("let orderSubmissionStarted = false;"),
        )
        shortcut_end = purchase.index("}, true);", shortcut_start)
        shortcut = purchase[shortcut_start:shortcut_end]
        self.assertIn(
            "const currentForm = document.getElementById('submitOrderForm');",
            shortcut,
        )
        self.assertIn(
            "const currentButton = document.getElementById('submitOrderButton');",
            shortcut,
        )
        self.assertIn("currentButton.form !== currentForm", shortcut)
        self.assertIn(
            "document.body.classList.contains('ui-dialog-open')",
            shortcut,
        )
        self.assertNotIn("if (locked) return;", shortcut)
        self.assertIn("currentForm.requestSubmit(currentButton);", shortcut)
        self.assertIn("event.stopPropagation();", shortcut)

        submit_listener_start = purchase.index(
            "document.addEventListener('submit', function (event) {",
            purchase.index("let orderSubmissionStarted = false;"),
        )
        submit_listener_end = purchase.index("}, true);", submit_listener_start)
        submit_listener = purchase[submit_listener_start:submit_listener_end]
        self.assertIn("currentForm.id !== 'submitOrderForm'", submit_listener)
        self.assertIn("currentForm.querySelector('#submitOrderButton')", submit_listener)
        self.assertNotIn(".ot-box-footer", purchase)
