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
        self.assertIn("left: var(--nav-desktop, 120px);", self.source)
        self.assertIn("right: 0;", self.source)
        self.assertIn("margin-inline: auto;", self.source)
        self.assertIn("z-index: 950;", self.source)
        self.assertIn(
            "calc(100vw - var(--nav-desktop, 120px) - 1.5rem)",
            self.source,
        )
        self.assertIn("transform: translateY(-12px);", self.source)
        self.assertIn("transform: translateY(0);", self.source)
        self.assertIn("@media (max-width: 768px)", self.source)
        self.assertIn("left: 50%;", self.source)
        self.assertIn("right: auto;", self.source)
        self.assertIn("margin-inline: 0;", self.source)
        self.assertIn("width: calc(100vw - 12px);", self.source)
        self.assertIn("transform: translate(-50%, -12px);", self.source)
        self.assertIn("transform: translate(-50%, 0);", self.source)

    def test_compact_search_appears_only_after_primary_filters_scroll_away(self):
        self.assertIn(
            "var lookupBottom = primaryLookup.getBoundingClientRect().bottom;",
            self.source,
        )
        self.assertIn("lookupBottom <= stickyTopThreshold", self.source)
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

    def test_compact_search_stays_focused_when_short_results_clamp_scroll(self):
        self.assertIn("function stickySearchOwnsFocus()", self.source)
        self.assertIn("stickyForm.contains(document.activeElement)", self.source)
        self.assertIn(
            "if (!visible && stickySearchOwnsFocus()) visible = true;",
            self.source,
        )
        self.assertIn("const stickyReleaseThreshold = 56;", self.source)
        self.assertIn(
            "alreadyVisible && lookupBottom <= stickyReleaseThreshold",
            self.source,
        )
        self.assertIn("stickyForm.addEventListener('focusin'", self.source)
        self.assertIn("stickyForm.addEventListener('focusout'", self.source)
        self.assertIn(
            "window.requestAnimationFrame(updateStickySearchVisibility);",
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

    def test_primary_and_sticky_searches_have_accessible_trailing_clear_buttons(self):
        for element_id, controlled_input in (
            ("inventory-search-clear", "product-search"),
            ("inventory-sticky-search-clear", "inventory-sticky-search-input"),
        ):
            with self.subTest(element_id=element_id):
                self.assertEqual(self.source.count(f'id="{element_id}"'), 1)
                self.assertIn(f'aria-controls="{controlled_input}"', self.source)

        self.assertEqual(self.source.count('aria-label="Clear inventory search"'), 2)
        self.assertIn('type="button" class="inv-search-clear"', self.source)
        self.assertIn("width: 44px;", self.source)
        self.assertIn("height: 44px;", self.source)
        self.assertIn(".inv-search-clear[hidden] { display: none; }", self.source)
        self.assertIn('::-webkit-search-cancel-button', self.source)
        self.assertIn('class="inv-sticky-search-submit"', self.source)
        self.assertNotIn(".inv-sticky-search-form button {", self.source)

    def test_clear_button_resets_both_fields_and_refreshes_live_results(self):
        self.assertIn("function clearInventorySearch(focusTarget)", self.source)
        self.assertIn("lookupInput.value = '';", self.source)
        self.assertIn("if (stickyInput) stickyInput.value = '';", self.source)
        self.assertIn("syncSearchClearButtons();", self.source)
        self.assertIn("clearTimeout(invTimer);\n            fetchInventory();", self.source)
        self.assertIn("focusTarget.focus({ preventScroll: true });", self.source)
        self.assertIn("stickyClearButton.disabled = !visible;", self.source)
        self.assertIn("clearInventorySearch(lookupInput);", self.source)
        self.assertIn("clearInventorySearch(stickyInput);", self.source)


class BottomLeftToastTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        template_root = Path(settings.BASE_DIR) / "app" / "templates"
        cls.base = (template_root / "base.html").read_text(encoding="utf-8")
        cls.delivery = (template_root / "delivery.html").read_text(encoding="utf-8")
        cls.ordering_embed = (
            template_root / "ordering_sheet_embed.html"
        ).read_text(encoding="utf-8")
        cls.notifications = (
            template_root / "partials" / "_notifications.html"
        ).read_text(encoding="utf-8")
        cls.notification_script = (
            Path(settings.BASE_DIR) / "static" / "js" / "notifications.js"
        ).read_text(encoding="utf-8")
        cls.styles = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        stack_start = cls.styles.index(".ui-toast-stack,")
        stack_end = cls.styles.index("\n}", stack_start) + len("\n}")
        cls.toast_stack_rule = cls.styles[stack_start:stack_end]

    def test_global_toast_stack_is_bottom_left_and_mobile_safe(self):
        self.assertIn("{% include 'partials/_notifications.html' %}", self.base)
        self.assertIn('class="ui-toast-stack toast-stack"', self.notifications)
        self.assertIn(".ui-toast-stack,", self.styles)
        self.assertIn(
            "left: calc(1rem + env(safe-area-inset-left));",
            self.toast_stack_rule,
        )
        self.assertIn(
            "left: calc(var(--nav-desktop) + 1rem + env(safe-area-inset-left));",
            self.styles,
        )
        self.assertIn("transform: none;", self.toast_stack_rule)
        self.assertIn("right: auto;", self.toast_stack_rule)
        self.assertNotIn("left: 50%;", self.toast_stack_rule)
        self.assertNotIn("translateX(-50%)", self.toast_stack_rule)
        self.assertIn(
            "bottom: calc(64px + 0.75rem + env(safe-area-inset-bottom));",
            self.styles,
        )
        self.assertIn(
            "left: calc(0.75rem + env(safe-area-inset-left));",
            self.styles,
        )
        self.assertIn(
            "width: calc(100vw - 1.5rem - env(safe-area-inset-left) - env(safe-area-inset-right));",
            self.styles,
        )
        self.assertNotIn("translateX(100%)", self.styles)

    def test_delivery_toasts_match_the_site_wide_bottom_left_position(self):
        self.assertIn("function showDeliveryToast(msg, level)", self.delivery)
        self.assertIn("return window.showToast(msg, level || 'info');", self.delivery)
        self.assertNotIn("dv-toast", self.delivery)
        self.assertNotIn("function showToast(msg, level)", self.delivery)

    def test_embedded_ordering_sheet_toasts_are_bottom_left(self):
        self.assertIn(
            "{% include 'partials/_notifications.html' with notifications_toasts_only=True %}",
            self.ordering_embed,
        )
        self.assertIn(
            "bottom: calc(14px + env(safe-area-inset-bottom));",
            self.styles,
        )
        self.assertNotIn("os-msg", self.ordering_embed)
        self.assertIn("window.showToast = showToast;", self.notification_script)
