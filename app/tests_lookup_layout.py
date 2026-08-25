from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class WidthNeutralProductLookupTests(SimpleTestCase):
    """Guard the scanner/search controls from reclaiming a side column."""

    template_root = Path(settings.BASE_DIR) / "app" / "templates"

    def source(self, name):
        return (self.template_root / name).read_text(encoding="utf-8")

    def test_every_competing_product_lookup_is_marked_width_neutral(self):
        for template_name in (
            "order_form.html",
            "checkout.html",
            "checkin.html",
            "expired_products.html",
            "inventory_display.html",
            "label_printing.html",
            "product_trend.html",
        ):
            with self.subTest(template=template_name):
                self.assertIn("data-width-neutral-lookup", self.source(template_name))

    def test_primary_scanners_render_before_their_information_grids(self):
        for template_name in (
            "order_form.html",
            "checkout.html",
            "expired_products.html",
        ):
            with self.subTest(template=template_name):
                source = self.source(template_name)
                self.assertLess(
                    source.index("data-width-neutral-lookup"),
                    source.index('class="main-grid'),
                )

    def test_checkin_scanner_and_product_share_an_independent_primary_column(self):
        source = self.source("checkin.html")
        css = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        self.assertIn('id="search-box" data-width-neutral-lookup', source)
        self.assertIn('class="checkin-primary-column"', source)
        self.assertLess(
            source.index('class="checkin-primary-column"'),
            source.index('id="search-box" data-width-neutral-lookup'),
        )
        self.assertLess(
            source.index('id="search-box" data-width-neutral-lookup'),
            source.index('class="right-items"'),
        )
        self.assertIn('class="checkin-side-column"', source)
        self.assertLess(
            source.index('class="right-items"'),
            source.index('class="checkin-side-column"'),
        )
        self.assertIn('id="checkinActivityRail"', source)
        self.assertIn(".checkin-page .checkin-primary-column", css)
        self.assertIn(".checkin-page .checkin-side-column", css)
        self.assertIn("align-content: start;", css)
        self.assertNotIn("display: contents;", css)
        self.assertNotIn("grid-row: 1 / span 2;", css)

    def test_legacy_lookup_side_columns_are_removed(self):
        forbidden_by_template = {
            "order_form.html": "grid-template-columns: 380px 1fr 300px",
            "checkout.html": "grid-template-columns: 380px 1fr 300px",
            "inventory_display.html": "grid-template-columns: 280px 1fr",
            "label_printing.html": "grid-template-columns: 340px 1fr",
            "product_trend.html": "grid-template-columns: 260px 1fr",
        }
        for template_name, legacy_rule in forbidden_by_template.items():
            with self.subTest(template=template_name):
                self.assertNotIn(legacy_rule, self.source(template_name))

    def test_product_trend_autocomplete_escapes_the_search_card(self):
        source = self.source("product_trend.html")

        self.assertIn(".trend-grid > .trend-card:first-child {", source)
        self.assertIn("position: relative;", source)
        self.assertIn("z-index: 100;", source)
        self.assertIn("overflow: visible;", source)
        self.assertIn("#trend-autocomplete-results {", source)
        self.assertIn("z-index: 9999;", source)

    def test_label_category_picker_is_inside_the_product_lookup_card(self):
        source = self.source("label_printing.html")
        lookup_start = source.index('class="lp-card lp-lookup-card"')
        preview_start = source.index("<!-- Live Label Sheet Preview -->")
        lookup_region = source[lookup_start:preview_start]

        self.assertIn('class="lp-card-subsection"', lookup_region)
        self.assertIn('id="lp-category-select"', lookup_region)
        self.assertNotIn('<div class="lp-card">', lookup_region)
        self.assertIn(
            "grid-template-columns: minmax(320px, 1.25fr) minmax(300px, 1fr);",
            source,
        )

    def test_label_sidebar_preview_stays_compact_while_expanded_preview_is_full_size(self):
        source = self.source("label_printing.html")

        self.assertIn(
            ".lp-sidebar #lp-sheet-preview { max-height: 170px; overflow: hidden; }",
            source,
        )
        self.assertIn(
            ".lp-sidebar > .lp-card:last-child .lp-card-body { max-height: 190px; overflow: hidden; }",
            source,
        )
        self.assertIn(".lp-preview-sm .lp-sheet-page { width: 100%; }", source)
        self.assertIn(
            ".lp-preview-lg .lp-sheet-page { width: min(100%, 650px); margin: 0 auto; }",
            source,
        )

    def test_inventory_actions_align_in_one_row_and_stack_only_on_phones(self):
        source = self.source("inventory_display.html")

        self.assertIn(
            'class="form-group inv-department-group inv-align-to-product-input"',
            source,
        )
        self.assertIn(
            'class="inv-filter-actions inv-align-to-product-input" role="group" aria-label="Inventory actions"',
            source,
        )
        self.assertNotIn(
            'class="inv-filter-actions" style="margin-top:',
            source,
        )
        self.assertIn(
            "grid-template-columns: repeat(2, minmax(0, 1fr));",
            source,
        )
        self.assertIn("--inv-filter-control-height: 54px;", source)
        self.assertIn("@media (min-width: 1200px)", source)
        self.assertIn("--inv-filter-label-line-height: 1.275rem;", source)
        self.assertIn(
            "--inv-filter-label-offset: calc(var(--inv-filter-label-line-height) + 0.4rem);",
            source,
        )
        self.assertIn("#inventoryFilterForm .inv-align-to-product-input {", source)
        self.assertIn("margin-top: var(--inv-filter-label-offset);", source)
        for selector in (
            "#inventoryFilterForm .inv-search-input-shell,",
            "#inventoryFilterForm .inv-search-input-shell input,",
            "#inventoryFilterForm .ui-product-lookup-submit,",
            "#inventoryFilterForm .inv-cat-disclosure > summary,",
            "#inventoryFilterForm .inv-filter-actions .btn {",
        ):
            with self.subTest(selector=selector):
                self.assertIn(selector, source)
        self.assertIn("height: var(--inv-filter-control-height);", source)
        self.assertIn("min-height: var(--inv-filter-control-height);", source)
        self.assertIn("#inventoryFilterForm .inv-cat-disclosure > summary {", source)
        self.assertIn("padding: 5px 10px;", source)
        self.assertNotIn("margin-top: 1.45rem !important;", source)
        self.assertIn(".inv-filter-actions .btn {", source)
        self.assertIn("min-height: 44px;", source)
        self.assertIn("@media (max-width: 600px)", source)
        self.assertIn(
            ".inv-filter-actions { grid-template-columns: minmax(0, 1fr); }",
            source,
        )

    def test_shared_lookup_bar_is_full_width_and_shrink_safe(self):
        css = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        self.assertIn(".ui-workflow-lookup-bar", css)
        self.assertIn("width: 100%;", css)
        self.assertIn("grid-template-columns: max-content minmax(280px, 1fr) max-content;", css)
        self.assertIn("min-width: 0;", css)
