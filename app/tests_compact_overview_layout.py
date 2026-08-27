import re
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class CompactOverviewLayoutTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        root = Path(settings.BASE_DIR)
        cls.css = (root / "static" / "css" / "ui-system.css").read_text(
            encoding="utf-8"
        )
        cls.base = (root / "app" / "templates" / "base.html").read_text(
            encoding="utf-8"
        )
        cls.order_form = (root / "app" / "templates" / "order_form.html").read_text(
            encoding="utf-8"
        )
        cls.checkout = (root / "app" / "templates" / "checkout.html").read_text(
            encoding="utf-8"
        )

    def css_block(self, selector):
        match = re.search(re.escape(selector) + r"\s*\{(?P<body>[^}]+)\}", self.css)
        self.assertIsNotNone(match, f"Missing CSS block for {selector}")
        return match.group("body")

    def test_sitewide_overview_cards_use_compact_height_tokens(self):
        self.assertIn("--ui-summary-card-min-height: 66px", self.css)
        self.assertIn("--ui-summary-card-padding-block: 0.38rem", self.css)
        for selector in (
            ".home-main > .kpi-strip > .kpi-card",
            ".as-page .as-kpi",
            ".al-page .al-kpi",
            ".dv-page .dv-kpi",
            ".dr-wrap .dr-kpi",
            ".expired-page .exp-kpi-card",
            ".exs-page .exs-kpi-card",
            ".inventory-page .inv-stat-card",
            ".ls-page .ls-kpi-card",
            ".oos-page .oos-kpi-card",
            ".order-detail-page .kpi-card",
            ".tx-page .tx-kpi-card",
            ".trend-page #kpi-strip .kpi-card",
            ".sa-page > .kpi-strip .kpi-card",
        ):
            self.assertIn(selector, self.css)
        self.assertIn("min-height: var(--ui-summary-card-min-height)", self.css)
        self.assertIn("margin-bottom: 0.75rem", self.css)

    def test_shared_workflow_strip_is_removed_sitewide(self):
        self.assertNotIn('class="workflow-nav"', self.base)
        self.assertNotIn("partials/_workflow_shortcut_decal.html", self.base)
        self.assertNotIn('class="workflow-guide-button"', self.base)
        self.assertIn('class="mobile-utility-bar"', self.base)
        self.assertIn('data-ui-open-shortcuts', self.base)
        self.assertIn("?v=20260827-productenter1", self.base)

    def test_order_summary_count_moves_beside_items_and_actions_fill_section(self):
        for source in (self.order_form, self.checkout):
            header_start = source.index('<div class="ot-box-header">')
            header_end = source.index("</div>", header_start)
            self.assertNotIn("otItemCount", source[header_start:header_end])

            label_start = source.index('<div class="ot-section-label">', header_end)
            label_end = source.index("</div>", label_start)
            label = source[label_start:label_end]
            self.assertIn("Items", label)
            self.assertIn('id="otItemCount"', label)
            self.assertEqual(source.count('id="otItemCount"'), 1)
            self.assertIn("min-height: 64px", source)

        self.assertIn(".ot-box-primary-action {\n        padding: 0;", self.order_form)
        self.assertIn(".ot-box-primary-action form { width: 100%; margin: 0; }", self.order_form)
        self.assertIn("border-radius: 0; font-size: 17px", self.order_form)
        self.assertIn("body.app-shell .container .ot-box-primary-action .ot-submit-btn", self.order_form)
        self.assertNotIn('class="ot-line-item-price"', self.order_form)
        self.assertIn(".ot-total-value { font-size: 58px;", self.order_form)
        self.assertIn(".ot-line-item-qty { font-size: 17.5px;", self.order_form)
        self.assertIn(".ot-summary-row > span:last-child { font-size: 19px;", self.order_form)
        self.assertIn(".ot-box-footer { padding: 0;", self.checkout)

    def test_purchase_item_cards_use_seventy_thirty_hierarchy_and_price_color(self):
        self.assertIn(".order-item::before {", self.order_form)
        self.assertIn(
            "grid-template-columns: minmax(0, 7fr) minmax(0, 3fr);",
            self.order_form,
        )
        self.assertIn('grid-template-areas: "main actions";', self.order_form)
        self.assertIn(".order-item-main {", self.order_form)
        self.assertIn(
            "grid-template-rows: minmax(72px, auto) minmax(112px, 1fr) auto;",
            self.order_form,
        )

        main = self.order_form.index('<div class="order-item-main">')
        name = self.order_form.index('<div class="item-top">', main)
        price = self.order_form.index('<div class="col-price item-price-section', main)
        barcode = self.order_form.index('<div class="barcode-display"', main)
        actions = self.order_form.index('<div class="col-actions item-right">', main)
        self.assertLess(main, name)
        self.assertLess(name, price)
        self.assertLess(price, barcode)
        self.assertLess(barcode, actions)
        self.assertNotIn("stock-badge-big", self.order_form)
        self.assertNotIn("stock-badge-label", self.order_form)
        self.assertIn(
            'class="item-quantity" aria-label="Order quantity: {{ item.quantity }}"',
            self.order_form,
        )
        self.assertIn("font-size: clamp(28px, 2.4vw, 36px);", self.order_form)
        self.assertNotIn("background: linear-gradient(135deg, var(--of-primary), #6366f1);", self.order_form)

        self.assertIn(
            ".item-price {\n        max-width: 100%;\n        font-size: clamp(71.3px, 6.9vw, 101.2px);",
            self.order_form,
        )
        self.assertIn("font-size: clamp(59.8px, 16.1vw, 82.8px);", self.order_form)
        self.assertNotIn("font-size: 94px;", self.order_form)
        self.assertIn(
            "grid-template-columns: 1fr;\n        grid-template-rows: auto minmax(0, 1fr);",
            self.order_form,
        )
        self.assertIn(
            "border: 0;\n        border-top: 1px solid #e2e8f0;\n        border-radius: 0;",
            self.order_form,
        )
        self.assertIn("grid-template-areas: \"label full\";", self.order_form)
        self.assertIn(
            "font-size: clamp(17px, 1.65vw, 22px);\n        font-weight: 800;",
            self.order_form,
        )
        self.assertNotIn("barcode-display-last6", self.order_form)
        self.assertNotIn('item.product.barcode|slice:"-6:"', self.order_form)
        self.assertIn("background: transparent;\n        box-shadow: none;", self.order_form)
        self.assertIn(
            ".item-price-section.is-taxable .item-price {\n        color: #047857;",
            self.order_form,
        )
        self.assertIn(
            ".item-price-section.is-tax-free .item-price {\n        color: #b91c1c;",
            self.order_form,
        )
        self.assertNotIn("background: linear-gradient(145deg, #f0fdf4", self.order_form)
        self.assertNotIn("background: linear-gradient(145deg, #fef2f2", self.order_form)
        self.assertIn(
            'item-price-section {% if item.product.taxable %}is-taxable{% else %}is-tax-free{% endif %}',
            self.order_form,
        )
        self.assertNotIn('class="tax-badge', self.order_form)
        self.assertNotIn("Taxable</span>", self.order_form)
        self.assertNotIn("Tax Free</span>", self.order_form)
        self.assertIn("grid-template-columns: minmax(0, 1fr) auto;", self.order_form)
        self.assertIn("max-height: 222px;", self.order_form)
        self.assertIn('class="barcode-display-empty">Not set</div>', self.order_form)
        self.assertIn("Expiries &amp; lots", self.order_form)
        self.assertIn("Lot designation", self.order_form)
        self.assertIn("{% for expiry in item.expiry_lot_rows %}", self.order_form)
        self.assertIn('class="expiry-panel-lot-number"', self.order_form)
        self.assertNotIn("expiry-panel-lot-label", self.order_form)
        self.assertNotIn("Not recorded", self.order_form)
        self.assertIn('<time class="expiry-panel-date"', self.order_form)
        self.assertIn(".expiry-panel.has-warning {", self.order_form)
        self.assertIn(".expiry-panel.has-critical {", self.order_form)
        self.assertIn(".expiry-panel.has-expired {", self.order_form)
        self.assertIn("else if (daysDiff <= 90)", self.order_form)
        self.assertIn("label = '⚠ EXPIRED';", self.order_form)
        self.assertIn("card.classList.add('has-expired-item');", self.order_form)

    def test_purchase_items_container_keeps_exact_workspace_split(self):
        main_grid = re.search(r"\.main-grid\s*\{(?P<body>[^}]+)\}", self.order_form)
        self.assertIsNotNone(main_grid)
        self.assertIn(
            "grid-template-columns: minmax(300px, 3fr) minmax(0, 7fr);",
            main_grid.group("body"),
        )
        self.assertIn(
            "grid-template-rows: auto minmax(0, 1fr);",
            main_grid.group("body"),
        )

        header = self.order_form.index('<div class="items-header">')
        container = self.order_form.index('<div class="items-container">', header)
        summary = self.order_form.index('<div class="ot-box">', container)
        self.assertLess(header, container)
        self.assertLess(container, summary)
        self.assertIn("<h2>Order Items</h2>", self.order_form[header:container])
        self.assertIn(
            'class="items-header-count">{{ order_items|length }} item{{ order_items|length|pluralize }}',
            self.order_form[header:container],
        )

        items_container = re.search(
            r"\.items-container\s*\{(?P<body>[^}]+)\}", self.order_form
        )
        self.assertIsNotNone(items_container)
        container_css = items_container.group("body")
        self.assertIn("display: grid", container_css)
        self.assertIn("gap: 12px", container_css)
        self.assertIn("max-height: calc(100vh - 230px)", container_css)
        self.assertIn("overflow-y: auto", container_css)
        self.assertIn("scrollbar-gutter: stable", container_css)
        self.assertIn("background: #f4f7fb", container_css)
        self.assertIn("margin: 0;", self.order_form)
        self.assertIn(
            ".items-container {\n            max-height: none;\n            overflow: visible;",
            self.order_form,
        )

    def test_purchase_timer_animates_on_lookup_card_bottom_edge(self):
        lookup = self.order_form.index(
            '<div class="control-box ui-workflow-lookup-bar" id="search-box"'
        )
        timer = self.order_form.index('<div id="orderTimerRegion">', lookup)
        workspace = self.order_form.index("<!-- RIGHT 70%: ORDER ITEMS -->", timer)
        self.assertLess(lookup, timer)
        self.assertLess(timer, workspace)
        self.assertNotIn('<div id="orderTimerRegion">', self.order_form[:lookup])

        self.assertIn("#search-box:has(#autoSubmitBar) {", self.order_form)
        self.assertIn("#search-box > #orderTimerRegion {", self.order_form)
        self.assertIn(
            "position: absolute;\n        inset: auto 0 0;",
            self.order_form,
        )
        self.assertIn(
            ".auto-submit-progress {\n        position: absolute;\n        inset: auto 0 0;",
            self.order_form,
        )
        self.assertIn("transition: width 1s linear;", self.order_form)
        self.assertIn('id="autoSubmitCountdown">10:00</strong>', self.order_form)
        self.assertIn("progressBar.style.width = pct + '%';", self.order_form)
        self.assertIn("bar.classList.add('warning');", self.order_form)
        self.assertIn("visualTimer = setInterval(updateDisplay, 1000);", self.order_form)
        self.assertIn("requestAutomaticSubmission()", self.order_form)
        self.assertNotIn('id="autoSubmitReset"', self.order_form)
        self.assertNotIn(".auto-submit-reset", self.order_form)
        self.assertNotIn("resetOrderTimer", self.order_form)

    def test_purchase_seniors_discount_is_right_aligned_header_pill(self):
        summary = self.order_form.index('<div class="ot-box">')
        header = self.order_form.index('<div class="ot-box-header">', summary)
        title = self.order_form.index("<h3>Order Summary</h3>", header)
        discount = self.order_form.index(
            'class="ot-discount-form"', title
        )
        primary_action = self.order_form.index(
            '<div class="ot-box-primary-action">', discount
        )
        self.assertLess(header, title)
        self.assertLess(title, discount)
        self.assertLess(discount, primary_action)
        self.assertEqual(self.order_form.count('class="ot-discount-form"'), 1)

        self.assertIn("display: flex;\n        align-items: center;", self.order_form)
        self.assertIn("justify-content: space-between;", self.order_form)
        self.assertIn(
            ".ot-discount-form {\n        flex: 0 0 auto;\n        margin: 0 0 0 auto;\n        padding: 0;",
            self.order_form,
        )
        self.assertIn("border-radius: 999px;", self.order_form)
        self.assertIn(
            '<span class="ot-discount-text">Seniors Discount &minus; 10%</span>',
            self.order_form,
        )
        text = self.order_form.index('<span class="ot-discount-text">', discount)
        switch = self.order_form.index('<span class="ot-discount-switch"', text)
        self.assertLess(text, switch)
        self.assertNotIn("<small>10% off, before tax</small>", self.order_form)
