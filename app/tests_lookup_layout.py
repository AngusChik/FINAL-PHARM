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
        ):
            with self.subTest(template=template_name):
                self.assertIn("data-width-neutral-lookup", self.source(template_name))

    def test_primary_scanners_render_before_their_information_grids(self):
        for template_name in (
            "checkout.html",
            "expired_products.html",
        ):
            with self.subTest(template=template_name):
                source = self.source(template_name)
                self.assertLess(
                    source.index("data-width-neutral-lookup"),
                    source.index('class="main-grid'),
                )

        purchase = self.source("order_form.html")
        purchase_grid = purchase.index('<div class="main-grid">')
        purchase_lookup = purchase.index('id="search-box" data-width-neutral-lookup')
        purchase_items = purchase.index('class="right-items"', purchase_lookup)
        self.assertLess(purchase_grid, purchase_lookup)
        self.assertLess(purchase_lookup, purchase_items)

    def test_checkin_scanner_and_history_share_the_left_rail(self):
        source = self.source("checkin.html")
        self.assertIn('id="search-box" data-width-neutral-lookup', source)
        self.assertIn('class="checkin-primary-column"', source)
        self.assertIn('class="checkin-side-column"', source)
        primary = source.index('class="checkin-primary-column"')
        product = source.index('class="right-items"', primary)
        side = source.index('class="checkin-side-column"', product)
        lookup = source.index('id="search-box" data-width-neutral-lookup', side)
        history = source.index('id="checkinActivityRail"', lookup)
        self.assertLess(primary, product)
        self.assertLess(
            product,
            side,
        )
        self.assertLess(side, lookup)
        self.assertLess(lookup, history)
        self.assertIn(
            "grid-template-columns: clamp(320px, 22vw, 380px) minmax(0, 1fr);",
            source,
        )
        self.assertIn("grid-column: 2;", source)
        self.assertIn("grid-column: 1;", source)

    def test_legacy_lookup_side_columns_are_removed(self):
        forbidden_by_template = {
            "order_form.html": "grid-template-columns: 380px 1fr 300px",
            "checkout.html": "grid-template-columns: 380px 1fr 300px",
            "inventory_display.html": "grid-template-columns: 280px 1fr",
            "label_printing.html": "grid-template-columns: 340px 1fr",
        }
        for template_name, legacy_rule in forbidden_by_template.items():
            with self.subTest(template=template_name):
                self.assertNotIn(legacy_rule, self.source(template_name))

    def test_product_details_header_contains_lookup_period_control_and_edit(self):
        source = self.source("product_details.html")
        header_start = source.index('<header class="trend-header">')
        header_end = source.index("</header>", header_start)
        header = source[header_start:header_end]

        period_start = header.index('id="productPeriodControl"')
        period_end = header.index('</details>', period_start)
        period_control = header[period_start:period_end]
        self.assertIn('id="productDetailsFilters"', period_control)
        self.assertIn('name="start"', period_control)
        self.assertIn('name="end"', period_control)
        self.assertIn('name="granularity"', period_control)
        self.assertIn('id="periodGrouping"', period_control)
        self.assertIn('name="type"', period_control)
        self.assertIn('name="return_to"', period_control)
        self.assertNotIn('id="btnBar"', period_control)
        self.assertNotIn('id="btnLine"', period_control)
        self.assertIn("data-width-neutral-lookup", header)
        self.assertIn('type="search"', header)
        self.assertIn('id="productDetailsSearch"', header)
        self.assertIn('autocomplete="off"', header)
        self.assertIn('role="combobox"', header)
        self.assertIn('aria-controls="productDetailsSearchResults"', header)
        self.assertIn('aria-expanded="false"', header)
        self.assertIn('id="productDetailsSearchResults"', header)
        self.assertIn('role="listbox"', header)
        self.assertIn('id="productDetailsSearchStatus"', header)
        self.assertIn('role="status"', header)
        self.assertLess(
            header.index('id="productDetailsSearch"'),
            period_start,
        )
        self.assertLess(period_end, header.index('>Edit Product</a>'))
        analysis_heading = source[
            source.index('class="card-title analysis-heading"'):
            source.index('<div id="kpi-strip"')
        ]
        self.assertIn('id="btnBar"', analysis_heading)
        self.assertIn('id="btnLine"', analysis_heading)
        self.assertNotIn('name="q"', source)
        self.assertNotIn("Top Sellers", source)

    def test_product_details_lookup_overlays_results_and_opens_exact_product_id(self):
        source = self.source("product_details.html")

        self.assertIn(".product-details-lookup {", source)
        self.assertIn(".product-details-search-results {", source)
        self.assertIn("position: absolute;", source)
        self.assertIn("z-index: 20;", source)
        self.assertIn("data-search-url=\"{% url 'global_search' %}\"", source)
        self.assertIn(
            "data-details-url-template=\"{% url 'product_details' product_id=0 %}\"",
            source,
        )
        self.assertIn("name.textContent = product.name", source)
        self.assertIn("option.tabIndex = -1", source)
        self.assertIn("option.setAttribute('role', 'option')", source)
        self.assertIn("endpoint.searchParams.set('q', query)", source)
        self.assertIn("}, 250);", source)
        self.assertIn("window.location.assign(productDetailsUrl(product.product_id))", source)
        self.assertIn("['start', 'end', 'granularity', 'type']", source)
        self.assertIn("target.searchParams.set('return_to', returnTo)", source)
        self.assertIn("function cancelPendingSearch()", source)
        self.assertGreaterEqual(source.count("cancelPendingSearch();"), 3)
        self.assertIn("lookup.addEventListener('focusout'", source)
        self.assertIn("!lookup.contains(document.activeElement)", source)
        self.assertIn("item.setAttribute('aria-selected', 'false')", source)
        self.assertIn("@media (max-width: 600px)", source)
        mobile_styles = source[source.index("@media (max-width: 600px)"):]
        self.assertIn(".product-details-lookup", mobile_styles)

    def test_retired_product_trend_template_is_removed(self):
        self.assertFalse((self.template_root / "product_trend.html").exists())

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

    def test_label_action_bar_clears_permanent_desktop_navigation(self):
        source = self.source("label_printing.html")

        self.assertIn("left: var(--nav-desktop, 120px);", source)
        self.assertIn(
            "width: calc(100% - var(--nav-desktop, 120px));",
            source,
        )
        self.assertIn(
            ".lp-bottom-bar { left: 0; width: 100%; bottom: 64px; }",
            source,
        )
        self.assertIn(".lp-bottom-bar { display: none !important; }", source)

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
