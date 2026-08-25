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

    def test_workflow_navigation_and_header_form_one_filled_card(self):
        stack = self.css_block(".workflow-header-stack")
        self.assertIn("overflow: hidden", stack)
        self.assertIn("background: var(--surface-color)", stack)
        self.assertIn("border: 1px solid var(--border-color)", stack)

        nav = self.css_block(".workflow-header-stack > .workflow-nav")
        self.assertIn("align-items: stretch", nav)
        self.assertIn("gap: 0", nav)
        self.assertIn("padding: 0", nav)
        self.assertIn("border: 0", nav)
        self.assertIn("border-bottom: 1px solid var(--border-color)", nav)
        self.assertIn("border-radius: 0", nav)

        self.assertIn("background: transparent", self.css)
        self.assertIn("border-right: 1px solid var(--border-color)", self.css)
        link_blocks = re.findall(
            re.escape(".workflow-nav a") + r"\s*\{([^}]+)\}", self.css
        )
        links = next(block for block in link_blocks if "min-height: 44px" in block)
        self.assertIn("min-height: 44px", links)
        self.assertIn("border-right: 1px solid var(--border-color)", links)
        self.assertIn("border-radius: 0", links)
        self.assertIn("?v=20260824-tableheads2", self.base)

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
        self.assertIn("border-radius: 0; font-size: 15px", self.order_form)
        self.assertIn("body.app-shell .container .ot-box-primary-action .ot-submit-btn", self.order_form)
        self.assertNotIn('class="ot-line-item-price"', self.order_form)
        self.assertIn(".ot-total-value { font-size: 40px;", self.order_form)
        self.assertIn(".ot-line-item-qty { font-size: 15.5px;", self.order_form)
        self.assertIn(".ot-summary-row > span:last-child { font-size: 17px;", self.order_form)
        self.assertIn(".ot-box-footer { padding: 0;", self.checkout)
