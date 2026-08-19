from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class InventoryPersistentSearchTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "inventory_display.html"
        ).read_text(encoding="utf-8")

    def test_compact_search_has_unique_accessible_controls(self):
        for element_id in (
            "inventoryStickySearch",
            "inventoryStickySearchForm",
            "inventory-sticky-search-input",
        ):
            with self.subTest(element_id=element_id):
                self.assertEqual(self.source.count(f'id="{element_id}"'), 1)

        self.assertIn(
            'for="inventory-sticky-search-input">Product name, SKU, or barcode',
            self.source,
        )
        self.assertIn('role="search"', self.source)
        self.assertIn('aria-label="Search inventory while viewing results"', self.source)
        self.assertIn('class="inv-sticky-search-shell"', self.source)
        self.assertIn("left: calc(50% + 32px);", self.source)
        self.assertIn("z-index: 950;", self.source)
        self.assertIn(
            "width: min(760px, calc(100vw - 64px - 1.5rem));",
            self.source,
        )
        self.assertIn("@media (max-width: 768px)", self.source)
        self.assertIn("left: 50%;", self.source)
        self.assertIn("width: calc(100vw - 12px);", self.source)

    def test_compact_search_appears_only_after_primary_filters_scroll_away(self):
        self.assertIn(
            "primaryLookup.getBoundingClientRect().bottom <= stickyTopThreshold",
            self.source,
        )
        self.assertIn(
            "document.querySelector('#inventoryFilterForm .ui-product-lookup')",
            self.source,
        )
        self.assertIn(
            "stickySearch.classList.toggle('is-visible', visible)",
            self.source,
        )
        self.assertIn(
            "observe(primaryLookup)",
            self.source,
        )
        self.assertIn(
            "window.addEventListener('scroll', updateStickySearchVisibility",
            self.source,
        )

    def test_compact_search_reuses_live_filter_and_enter_submit_paths(self):
        self.assertIn("lookupInput.value = stickyInput.value;", self.source)
        self.assertIn(
            "lookupInput.dispatchEvent(new Event('input', { bubbles: true }))",
            self.source,
        )
        self.assertIn("filterForm.requestSubmit()", self.source)
        self.assertIn("stickyInput.value = lookupInput.value;", self.source)


class BottomCenteredToastTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        template_root = Path(settings.BASE_DIR) / "app" / "templates"
        cls.base = (template_root / "base.html").read_text(encoding="utf-8")
        cls.delivery = (template_root / "delivery.html").read_text(encoding="utf-8")
        cls.ordering_embed = (
            template_root / "ordering_sheet_embed.html"
        ).read_text(encoding="utf-8")

    def test_global_toast_stack_is_bottom_centered_and_mobile_safe(self):
        self.assertIn("left: 50%;", self.base)
        self.assertIn("transform: translateX(-50%);", self.base)
        self.assertIn("right: auto;", self.base)
        self.assertIn(
            "bottom: calc(64px + 0.75rem + env(safe-area-inset-bottom));",
            self.base,
        )
        self.assertNotIn("from { transform: translateX(100%); opacity: 0; }", self.base)

    def test_delivery_toasts_match_the_site_wide_bottom_center_position(self):
        self.assertIn(
            ".dv-toast-stack { position: fixed; "
            "bottom: calc(24px + env(safe-area-inset-bottom)); left: 50%; right: auto;",
            self.delivery,
        )
        self.assertIn("transform: translateX(-50%);", self.delivery)
        self.assertNotIn(
            ".dv-toast-stack { position: fixed; bottom: 24px; right: 24px;",
            self.delivery,
        )
        self.assertIn("function showDeliveryToast(msg, level)", self.delivery)
        self.assertIn("typeof window.showToast === 'function'", self.delivery)
        self.assertIn("stack.setAttribute('aria-live', 'polite')", self.delivery)
        self.assertIn("t.setAttribute('role', level === 'error' ? 'alert' : 'status')", self.delivery)
        self.assertNotIn("function showToast(msg, level)", self.delivery)

    def test_embedded_ordering_sheet_toasts_are_bottom_centered(self):
        self.assertIn(
            "position: fixed; bottom: calc(14px + env(safe-area-inset-bottom)); "
            "left: 50%; transform: translateX(-50%);",
            self.ordering_embed,
        )
        self.assertNotIn("position: fixed; top: 14px;", self.ordering_embed)
