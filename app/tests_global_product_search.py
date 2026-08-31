from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class GlobalProductSearchCardTests(SimpleTestCase):
    """Guard the site-wide product quick view's compact visual hierarchy."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "base.html"
        ).read_text(encoding="utf-8")

    def test_stock_quantity_is_the_responsive_visual_anchor(self):
        self.assertIn("grid-template-columns: minmax(0, 1fr) 132px;", self.source)
        self.assertIn("min-width: 130px;", self.source)
        self.assertIn("font-size: var(--ps-stock-number-size, 52px);", self.source)
        self.assertIn("stockText.length <= 3", self.source)
        self.assertIn("Math.max(14, Math.floor(104 / (stockText.length * 0.62)))", self.source)
        self.assertIn("--ps-stock-number-size:' + stockNumberSize + 'px", self.source)
        self.assertIn("stockQuantity <= 0", self.source)
        self.assertIn("stockQuantity <= 3", self.source)
        self.assertIn("stockLabel = 'Out of Stock'", self.source)
        self.assertIn("stockLabel = 'Low Stock'", self.source)
        self.assertIn("stockLabel = 'In Stock'", self.source)

    def test_optional_metadata_is_labelled_and_empty_values_are_omitted(self):
        self.assertIn("metaChip('Brand', p.brand)", self.source)
        self.assertIn("metaChip('Barcode', p.barcode, 'is-barcode')", self.source)
        self.assertIn("metaChip('Category', p.category)", self.source)
        self.assertIn("String(value).trim() === ''", self.source)
        self.assertIn("escHtml(String(value))", self.source)
        self.assertNotIn("p.brand || '—'", self.source)
        self.assertNotIn("p.barcode || '—'", self.source)
        self.assertNotIn('class="ps-detail-sep"', self.source)

    def test_expiry_rows_use_plain_language_without_day_abbreviations(self):
        self.assertIn("expiryDates.map(expiryRow).join('')", self.source)
        self.assertIn("lbl = 'Expires today'", self.source)
        self.assertIn("lbl = 'Expires soon'", self.source)
        self.assertIn("dayCount(Math.abs(diffDays)) + ' ago'", self.source)
        self.assertIn("dayCount(diffDays) + ' remaining'", self.source)
        self.assertIn("value + ' day' + (value === 1 ? '' : 's')", self.source)
        self.assertNotIn("Math.abs(diffDays) + 'd ago'", self.source)
        self.assertNotIn("'in ' + diffDays + 'd'", self.source)

    def test_chart_controls_are_labelled_and_keep_aria_state_in_sync(self):
        self.assertIn('aria-label="Sales activity period"', self.source)
        self.assertIn('data-months="6" aria-pressed="true">6 months', self.source)
        self.assertIn('data-months="1" aria-pressed="false">1 month', self.source)
        self.assertIn('for="psDateStart"><span>Start date</span>', self.source)
        self.assertIn('for="psDateEnd"><span>End date</span>', self.source)
        self.assertIn("button.classList.toggle('active', isActive);", self.source)
        self.assertIn("button.setAttribute('aria-pressed', isActive ? 'true' : 'false');", self.source)
        self.assertIn("button.setAttribute('aria-pressed', 'false');", self.source)
        self.assertIn("height: 220px;", self.source)

    def test_product_actions_are_always_visible_and_destinations_are_unchanged(self):
        self.assertIn(".ps-slider-body.has-product-detail", self.source)
        self.assertIn("grid-template-rows: minmax(0, 1fr) auto;", self.source)
        self.assertIn('class="ps-product-scroll"', self.source)
        self.assertIn('class="ps-links" role="group" aria-label="Product actions"', self.source)
        self.assertIn('href="/product-trend/?q=', self.source)
        self.assertIn('href="/product/edit/', self.source)
        self.assertIn('ps-link-btn ps-link-primary">View Full Trend</a>', self.source)
        self.assertIn('ps-link-btn ps-link-secondary">Edit Product</a>', self.source)
        self.assertIn("min-height: 44px;", self.source)
        self.assertIn("background: #b45309;", self.source)
        self.assertIn("body.ps-product-search-open :is(.ui-action-banner, .alert-banner)", self.source)

    def test_narrow_panel_stacks_without_horizontal_overflow(self):
        detail_start = self.source.index(".ps-product-detail {")
        scroll_start = self.source.index(".ps-product-scroll {", detail_start)
        detail_rule = self.source[detail_start:scroll_start]

        self.assertIn("container-type: inline-size;", detail_rule)
        self.assertIn("@container (max-width: 470px)", self.source)
        self.assertIn(".ps-summary-main { grid-template-columns: minmax(0, 1fr); }", self.source)
        self.assertIn("order: -1;", self.source)
        self.assertIn(".ps-activity-groups { grid-template-columns: minmax(0, 1fr); }", self.source)
        self.assertIn(".ps-date-row { grid-template-columns: minmax(0, 1fr); }", self.source)
        self.assertIn(".ps-links { flex-direction: column; }", self.source)
        self.assertIn("overflow-wrap: anywhere;", self.source)
        self.assertIn("developmentBanner.getBoundingClientRect().height", self.source)
        self.assertIn("panel.style.top = topOffset + 'px';", self.source)
        self.assertIn("panel.style.height = 'calc(100vh - ' + topOffset + 'px)'", self.source)

    def test_search_and_scanner_behavior_remains_available(self):
        for token in (
            "searchInput.addEventListener('keydown'",
            "if (e.key !== 'Enter') return;",
            "fetchSearchResults(q, requestId); }, 250",
            "window.openProductTrend",
            "if (e.key === 'Escape'",
            'vendor/chartjs/chart.umd.min.js',
            'aria-label="Close product search"',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)
