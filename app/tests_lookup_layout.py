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

    def test_checkin_scanner_escapes_its_mixed_secondary_wrapper(self):
        source = self.source("checkin.html")
        css = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        self.assertIn('id="search-box" data-width-neutral-lookup', source)
        self.assertIn(".checkin-page .left-controls", css)
        self.assertIn("display: contents;", css)
        self.assertIn("grid-column: 1 / -1;", css)

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

    def test_shared_lookup_bar_is_full_width_and_shrink_safe(self):
        css = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        self.assertIn(".ui-workflow-lookup-bar", css)
        self.assertIn("width: 100%;", css)
        self.assertIn("grid-template-columns: max-content minmax(280px, 1fr) max-content;", css)
        self.assertIn("min-width: 0;", css)
