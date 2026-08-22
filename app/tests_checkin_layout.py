import re
from decimal import Decimal
from html import unescape
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import (
    Category,
    CheckinSession,
    InventoryCountLine,
    Product,
    StockChange,
)


@override_settings(AXES_ENABLED=False)
class CheckinReceiveFirstLayoutTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="checkin-layout-user",
            password="test-password",
            is_staff=True,
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name="Check-in layout")
        self.product = Product.objects.create(
            name="Compact Receiving Product",
            item_number="RECEIVE-101",
            barcode="001122334455",
            price=Decimal("12.50"),
            quantity_in_stock=7,
            category=self.category,
            taxable=True,
        )
        self.session = CheckinSession.objects.create(
            user=self.user,
            scanned_by="Layout tester",
        )
        self.url = reverse("checkin_session", args=[self.session.pk])

    def _render(self, with_product=True):
        query = {"product_id": self.product.pk} if with_product else {}
        response = self.client.get(self.url, query)
        self.assertEqual(response.status_code, 200)
        return response.content.decode("utf-8")

    def _render_inventory(self, counted_qty=3):
        self.session.inventory_mode = True
        self.session.save(update_fields=["inventory_mode"])
        InventoryCountLine.objects.update_or_create(
            session=self.session,
            product=self.product,
            defaults={
                "product_name": self.product.name,
                "product_barcode": self.product.barcode,
                "expected_qty": self.product.quantity_in_stock,
                "counted_qty": counted_qty,
            },
        )
        return self._render()

    @staticmethod
    def _heading_text(rendered_html):
        heading = re.search(r"<h1\b[^>]*>(.*?)</h1>", rendered_html, re.DOTALL)
        if not heading:
            return ""
        without_tags = re.sub(r"<[^>]+>", "", heading.group(1))
        return " ".join(unescape(without_tags).split())

    @staticmethod
    def _media_sections(source, condition):
        marker = f"@media ({condition})"
        sections = []
        cursor = 0
        while True:
            start = source.find(marker, cursor)
            if start == -1:
                return "\n".join(sections)
            opening = source.find("{", start + len(marker))
            if opening == -1:
                return "\n".join(sections)
            depth = 1
            index = opening + 1
            while index < len(source) and depth:
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1
                index += 1
            sections.append(source[opening + 1:index - 1])
            cursor = index

    def test_primary_receiving_controls_keep_existing_behavior_hooks(self):
        html = self._render()

        self.assertEqual(html.count('id="product_lookup"'), 1)
        self.assertIn('id="search-box" data-width-neutral-lookup', html)
        self.assertIn('class="receiving-strip', html)
        for element_id in (
            "qaPrintLabel",
            "toggleEditBtn",
            "qaViewTrend",
            "sbValue",
            "sbSetForm",
            "quickLotAddForm",
            "receivingLotSelect",
            "receivingLotNumber",
            "receivingLotExpiry",
            "receivingLotSaveStatus",
        ):
            self.assertEqual(html.count(f'id="{element_id}"'), 1)

        self.assertLess(html.index('id="search-box"'), html.index('class="right-items"'))
        self.assertNotIn(">⚡ Quick Actions<", html)

    def test_receiving_draft_autosave_is_serialized_debounced_and_flushable(self):
        html = self._render()

        self.assertIn('data-draft-save-url="', html)
        self.assertIn('data-draft-revision="0"', html)
        self.assertIn(
            "const operation = receivingDraftChain.catch(function() {}).then(function()",
            html,
        )
        self.assertIn("scheduleReceivingDraftSave(500)", html)
        self.assertIn("window.flushReceivingDraftSave = function()", html)
        self.assertIn("window.adoptReceivingDraftRevision = function(revision)", html)
        self.assertIn("window.flushReceivingDraftSave().then(function()", html)
        self.assertIn("return receivingDraftChain;", html)
        self.assertNotIn("barcodeSubmitPending", html)
        self.assertNotIn("receivingDraftReadyWithTimeout", html)
        self.assertIn("Product lookup must never wait on or inherit", html)
        self.assertIn("window.location.assign(destination)", html)
        self.assertIn("receivingDraftNeedsRetry", html)
        self.assertIn("window.setReceivingActionLocked = function(locked)", html)
        self.assertIn("window.resetReceivingActionLock = function()", html)
        self.assertIn("receivingActionLockDepth", html)
        self.assertIn("window.isReceivingActionLocked = function()", html)
        self.assertNotIn("control.id === 'product_lookup'", html)
        self.assertIn("window.addEventListener('pageshow'", html)
        self.assertIn("if (searchInput) searchInput.readOnly = false", html)
        self.assertIn("window.setReceivingActionLocked(true)", html)
        self.assertIn("window.setReceivingActionLocked(false)", html)
        self.assertIn("if (intent === receivingDraftIntent)", html)
        self.assertIn("A newer local edit is already queued", html)
        self.assertIn("receivingDraftChain = Promise.resolve();", html)
        self.assertIn("stockForm.querySelectorAll('button')", html)
        self.assertNotIn("stockForm.querySelectorAll('button, input')", html)
        self.assertIn("var payload = new FormData(stockForm);", html)
        self.assertEqual(html.count("new FormData(stockForm)"), 1)
        self.assertIn("navigateAfterReceivingDraft(destination.href)", html)
        self.assertIn("window.navigateAfterReceivingDraft = navigateAfterReceivingDraft", html)
        self.assertIn("window.navigateAfterReceivingDraft(destinations[k])", html)
        self.assertIn('id="checkinNewProductForm"', html)
        self.assertIn("navigateAfterReceivingDraft(destination.href)", html)
        self.assertIn(".order-header form button, #icEndBtn", html)
        self.assertIn("error.field === 'existing_lot_id'", html)
        self.assertIn("resumeAutoScan();", html)
        self.assertIn("forceScannerFocus();", html)
        self.assertIn("applyReceivingLotChoice(false);", html)

    def test_secondary_details_default_open_and_remember_session_choice(self):
        html = self._render()

        details = re.search(
            r'<details class="product-secondary-details" id="productSecondaryDetails"[^>]*>',
            html,
        )
        self.assertIsNotNone(details)
        self.assertIn(f'data-checkin-session-id="{self.session.pk}"', details.group(0))
        self.assertRegex(details.group(0), r"\sopen(?:\s|>)")
        self.assertIn("'checkin-product-details:' + productDetailsSessionId", html)
        self.assertIn("sessionStorage.getItem(productDetailsStorageKey)", html)
        self.assertIn("sessionStorage.setItem(", html)
        self.assertIn("productDetailsPreferredOpen = true", html)
        self.assertIn("setProductDetailsOpen(true)", html)
        self.assertIn("setProductDetailsOpen(productDetailsPreferredOpen)", html)
        self.assertIn("ignoredProgrammaticDetailsState", html)
        self.assertNotIn("productSecondaryDetails.open = false", html)

    def test_lookup_results_stack_above_workspace_and_search_never_locks(self):
        html = self._render()
        template_source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "checkin.html"
        ).read_text(encoding="utf-8")
        shared_css = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        all_styles = template_source + "\n" + shared_css

        self.assertRegex(
            all_styles,
            r"\.checkin-page\s+\.left-controls\s*\{[^}]*"
            r"position:\s*relative;[^}]*z-index:\s*40;",
        )
        self.assertIn("max-height: min(300px, calc(100dvh - 9rem));", template_source)
        self.assertIn("overscroll-behavior: contain;", template_source)
        self.assertIn("scrollbar-gutter: stable;", template_source)
        self.assertNotIn("receivingNavigationPending", html)
        self.assertNotIn("barcodeSubmitPending", html)
        self.assertIn("searchInput.readOnly = false;", html)
        self.assertIn("window.location.assign(destination);", html)
        self.assertIn("window.resetReceivingActionLock()", html)
        self.assertIn("closeLookupResults();\n                navigateAfterReceivingDraft(", html)

    def test_product_movement_graph_precedes_always_visible_session_history(self):
        StockChange.objects.create(
            product=self.product,
            session=self.session,
            user=self.user,
            change_type="checkin",
            quantity=2,
        )
        html = self._render()

        self.assertIn('id="productMovementSummary"', html)
        self.assertIn('id="productMovementTitle"', html)
        self.assertIn('aria-labelledby="productMovementTitle"', html)
        self.assertIn('id="sessionHistoryPanel"', html)
        self.assertIn('id="sessionHistoryTitle"', html)
        self.assertIn('aria-labelledby="sessionHistoryTitle"', html)
        self.assertEqual(html.count('id="phChart"'), 1)
        self.assertEqual(html.count('id="sessionHistoryCard"'), 1)

        self.assertLess(
            html.index('id="productMovementSummary"'),
            html.index('id="sessionHistoryPanel"'),
        )
        canvas = re.search(r'<canvas\b[^>]*\bid="phChart"[^>]*>', html)
        self.assertIsNotNone(canvas)
        self.assertIn('role="img"', canvas.group(0))
        self.assertRegex(canvas.group(0), r'aria-label="[^"]+"')

        self.assertNotIn('role="tablist"', html)
        self.assertNotIn('role="tab"', html)
        self.assertNotIn('role="tabpanel"', html)
        self.assertNotIn('id="activitySessionTab"', html)
        self.assertNotIn('id="activityProductTab"', html)
        self.assertNotIn('id="productHistoryPanel"', html)
        self.assertNotIn('data-table-key="product-history"', html)
        self.assertNotIn("activateActivityTab", html)
        self.assertIn(
            ':is([class*="-slider-panel"],.lp-history-panel)',
            html,
        )
        self.assertNotIn(
            ':is([class*="-slider-panel"],[class*="-history-panel"])',
            html,
        )

    def test_selected_product_without_movement_shows_compact_graph_empty_state(self):
        html = self._render()

        self.assertIn('id="productMovementSummary"', html)
        self.assertIn("No stock movements recorded in the last 90 days.", html)
        self.assertNotIn('id="phChart"', html)
        self.assertLess(
            html.index('id="productMovementSummary"'),
            html.index('id="sessionHistoryPanel"'),
        )

    def test_empty_workspace_still_shows_session_activity_without_dead_trend(self):
        html = self._render(with_product=False)

        self.assertIn('id="checkinActivityRail"', html)
        self.assertIn('id="sessionHistoryPanel"', html)
        self.assertIn('id="sessionHistoryTitle"', html)
        self.assertEqual(html.count('id="sessionHistoryCard"'), 1)
        self.assertNotIn('id="productMovementSummary"', html)
        self.assertNotIn('id="phChart"', html)
        self.assertNotIn('id="activitySessionTab"', html)
        self.assertNotIn('id="activityProductTab"', html)
        self.assertNotIn('id="productHistoryPanel"', html)

    def test_inventory_count_mode_labels_header_and_uses_count_only_side_rail(self):
        normal_html = self._render()
        self.assertEqual(self._heading_text(normal_html), "Check-in")
        self.assertIn('id="checkinActivityRail"', normal_html)
        self.assertNotIn('class="ic-count-col"', normal_html)

        inventory_html = self._render_inventory()
        self.assertEqual(
            self._heading_text(inventory_html),
            "Check-in — Inventory Count",
        )
        self.assertNotIn('id="checkinActivityRail"', inventory_html)
        self.assertNotIn('id="productMovementSummary"', inventory_html)
        self.assertNotIn('id="sessionHistoryPanel"', inventory_html)
        self.assertNotIn('id="phChart"', inventory_html)
        count_regions = re.findall(
            r'class="[^"]*\bic-count-col\b',
            inventory_html,
        )
        self.assertEqual(len(count_regions), 1)
        self.assertRegex(
            inventory_html,
            r'<aside\b[^>]*class="[^"]*\bic-count-col\b[^"]*"'
            r'[^>]*aria-labelledby="inventoryCountTitle"',
        )
        self.assertIn('id="inventoryCountTitle"', inventory_html)
        self.assertIn(
            'class="ic-prog-counted" aria-live="polite" aria-atomic="true"',
            inventory_html,
        )

    def test_inventory_count_table_places_barcode_below_product_name(self):
        html = self._render_inventory()
        scroll_region = re.search(
            r'<div\b[^>]*class="[^"]*\bic-prog-wrap\b[^"]*"[^>]*>',
            html,
        )
        self.assertIsNotNone(scroll_region)
        self.assertIn('role="region"', scroll_region.group(0))
        self.assertIn(
            'aria-label="Inventory count products"',
            scroll_region.group(0),
        )
        self.assertIn('tabindex="0"', scroll_region.group(0))
        table = re.search(
            r'<table\b[^>]*class="[^"]*\bic-prog-table\b[^"]*"[^>]*>'
            r'(.*?)</table>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(table)
        table_html = table.group(1)
        self.assertIn(
            '<caption class="ui-sr-only">Inventory count progress</caption>',
            table_html,
        )
        self.assertEqual(table_html.count('scope="col"'), 4)

        product_row = next(
            (
                row
                for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_html, re.DOTALL)
                if self.product.name in row
            ),
            None,
        )
        self.assertIsNotNone(product_row)
        name = re.search(
            r'class="[^"]*\bic-prog-name\b[^"]*"[^>]*>.*?'
            + re.escape(self.product.name),
            product_row,
            re.DOTALL,
        )
        barcode = re.search(
            r'<small\b[^>]*class="[^"]*\bic-prog-barcode\b[^"]*"[^>]*>'
            r'.*?'
            + re.escape(self.product.barcode)
            + r'.*?</small>',
            product_row,
            re.DOTALL,
        )
        self.assertIsNotNone(name)
        self.assertIsNotNone(barcode)
        self.assertIn(
            '<span class="ui-sr-only">Barcode: </span>',
            barcode.group(0),
        )
        self.assertLess(name.start(), barcode.start())

    def test_inventory_count_panel_fills_desktop_and_resets_on_narrow_screens(self):
        template_source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "checkin.html"
        ).read_text(encoding="utf-8")
        shared_css = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        all_styles = template_source + "\n" + shared_css
        desktop_styles = self._media_sections(all_styles, "min-width: 1050px")
        narrow_styles = self._media_sections(all_styles, "max-width: 1049px")

        self.assertIn("function syncInventoryCountPanelHeight()", template_source)
        self.assertIn(
            "window.innerHeight - grid.getBoundingClientRect().top - 16",
            template_source,
        )
        self.assertRegex(
            template_source,
            r"grid\.style\.setProperty\(\s*"
            r"'--inventory-count-panel-height'",
        )
        self.assertIn(
            "window.addEventListener('resize', syncInventoryCountPanelHeight)",
            template_source,
        )
        self.assertGreaterEqual(
            template_source.count("syncInventoryCountPanelHeight"),
            3,
        )
        self.assertRegex(
            desktop_styles,
            r"\.main-grid\.ic-3col\s+\.ic-progress-card\s*\{[^}]*"
            r"height:\s*var\(--inventory-count-panel-height",
        )
        self.assertRegex(
            all_styles,
            r"\.ic-progress-card\s*\{[^}]*display:\s*flex;[^}]*"
            r"flex-direction:\s*column;",
        )
        self.assertRegex(
            desktop_styles,
            r"\.main-grid\.ic-3col\s+\.ic-prog-wrap\s*\{[^}]*"
            r"flex:\s*1(?:\s+1\s+auto)?;[^}]*min-height:\s*0;[^}]*"
            r"max-height:\s*none;[^}]*overflow-y:\s*auto;",
        )
        self.assertRegex(
            narrow_styles,
            r"\.main-grid\.ic-3col\s+\.ic-progress-card\s*\{[^}]*"
            r"height:\s*auto;[^}]*max-height:\s*none;",
        )
        self.assertRegex(
            narrow_styles,
            r"\.checkin-side-column\s*\{[^}]*grid-column:\s*1;"
            r"[^}]*grid-row:\s*2;[^}]*position:\s*static;",
        )

    def test_product_workspace_stays_directly_below_scanner(self):
        html = self._render()
        template_source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "checkin.html"
        ).read_text(encoding="utf-8")
        shared_css = (
            Path(settings.BASE_DIR) / "static" / "css" / "ui-system.css"
        ).read_text(encoding="utf-8")
        all_styles = template_source + "\n" + shared_css

        primary_start = html.index('class="checkin-primary-column"')
        search_start = html.index('id="search-box"')
        product_start = html.index('class="right-items"')
        side_start = html.index('class="checkin-side-column"')
        rail_start = html.index('id="checkinActivityRail"')
        self.assertLess(primary_start, search_start)
        self.assertLess(search_start, product_start)
        self.assertLess(product_start, side_start)
        self.assertLess(side_start, rail_start)
        self.assertRegex(
            html,
            r'<details class="product-secondary-details" id="productSecondaryDetails"'
            r'[^>]*\sopen(?:\s|>)',
        )

        self.assertRegex(
            all_styles,
            r"\.checkin-primary-column\s*\{[^}]*display:\s*grid;[^}]*"
            r"gap:\s*0\.85rem;[^}]*align-content:\s*start;[^}]*min-width:\s*0;",
        )
        self.assertRegex(
            all_styles,
            r"\.checkin-side-column\s*\{[^}]*grid-column:\s*2;"
            r"[^}]*grid-row:\s*1;",
        )
        self.assertNotIn("grid-row: 1 / span 2;", template_source)
        self.assertNotIn("grid-row: 1 / span 2;", shared_css)
        self.assertIn(
            "grid-template-columns: max-content minmax(0, 1fr) max-content;",
            template_source,
        )
        self.assertIn("text-align: center;", template_source)

        mobile_start = template_source.index("@media (max-width: 1049px)")
        mobile_end = template_source.find("@media", mobile_start + 1)
        mobile_styles = template_source[
            mobile_start: mobile_end if mobile_end != -1 else None
        ]
        self.assertRegex(
            mobile_styles,
            r"\.checkin-primary-column\s*\{[^}]*grid-column:\s*1;"
            r"[^}]*grid-row:\s*1;[^}]*position:\s*static;",
        )
        self.assertRegex(
            mobile_styles,
            r"\.checkin-side-column\s*\{[^}]*grid-column:\s*1;"
            r"[^}]*grid-row:\s*2;[^}]*position:\s*static;",
        )
        self.assertRegex(
            all_styles,
            r"\.session-history-list\s*\{[^}]*overflow-y:\s*auto;",
        )

    def test_stock_adjustment_refreshes_movement_chart_and_session_history(self):
        html = self._render()

        for contract in (
            "var currentMovement = document.getElementById('productMovementSummary');",
            "var nextMovement = nextPage.getElementById('productMovementSummary');",
            "currentMovement.innerHTML = nextMovement.innerHTML;",
            "window.destroyCheckinProductMovementChart();",
            "window.renderCheckinProductMovementChart();",
            "var currentSessionHistory = document.getElementById('sessionHistoryCard');",
            "currentSessionHistory.innerHTML = nextSessionHistory.innerHTML;",
            "var currentCountPanel = document.getElementById('inventoryCountPanel');",
            "var nextCountPanel = nextPage.getElementById('inventoryCountPanel');",
            "var currentCountBadge = currentCountPanel.querySelector('.ic-prog-counted');",
            "currentCountBadge.textContent = nextCountBadge.textContent;",
            "var currentCountRows = currentCountPanel.querySelector('.ic-prog-table tbody');",
            "currentCountRows.innerHTML = nextCountRows.innerHTML;",
            "document.dispatchEvent(new CustomEvent('ui:seamless-updated'",
            "refreshed: [currentCountPanel]",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, html)
        self.assertIn("window.checkinProductMovementChart.destroy();", html)

    def test_product_workspace_no_longer_uses_viewport_height_script(self):
        html = self._render()

        self.assertNotIn("syncCheckinWorkspaceHeight", html)
        self.assertNotIn("syncPhHeight", html)
        self.assertIn(".right-items .product-card {\n    height: auto;", html)
        self.assertIn("overflow: visible !important;", html)

    def test_full_width_shell_stays_clear_of_desktop_and_mobile_navigation(self):
        html = self._render()

        self.assertIn(
            "calc(64px + clamp(0.75rem, 1.4vw, 1.5rem));",
            html,
        )
        self.assertIn(
            "calc(clamp(0.75rem, 1.4vw, 1.5rem) + 2.25rem)",
            html,
        )
        self.assertIn("@media (max-width: 768px)", html)
        self.assertIn(
            "padding: 0.75rem clamp(0.75rem, 1.4vw, 1.5rem) "
            "calc(76px + env(safe-area-inset-bottom));",
            html,
        )
