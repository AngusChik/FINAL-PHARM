import re
from datetime import date
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
    ProductLot,
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

        side_start = html.index('class="checkin-side-column"')
        search_start = html.index('id="search-box"')
        history_start = html.index('id="sessionHistoryPanel"')
        self.assertLess(side_start, search_start)
        self.assertLess(search_start, history_start)
        heading = html.index('<div class="checkin-scan-heading">')
        lookup_form = html.index('<form id="barcodeForm"', heading)
        heading_region = html[heading:lookup_form]
        self.assertLess(heading_region.index("Scan product"), heading_region.index("Ready"))
        self.assertEqual(html.count('<span class="hint-ready">Ready</span>'), 1)
        self.assertNotIn('Click to activate product lookup', html)
        self.assertNotIn(">⚡ Quick Actions<", html)

    def test_header_uses_compact_tax_icon_and_keeps_print_label_in_actions(self):
        html = self._render()
        header = html[
            html.index('<div class="product-card-header'):
            html.index('class="receiving-strip')
        ]
        primary_actions = header[
            header.index('<div class="product-header-primary-actions">'):
            header.index('</details>') + len('</details>')
        ]
        actions_popover = primary_actions[
            primary_actions.index('<div class="product-actions-popover">'):
        ]

        self.assertNotIn('id="qaPrintLabel"', primary_actions[:primary_actions.index('<details')])
        self.assertIn('id="qaPrintLabel"', actions_popover)
        self.assertLess(actions_popover.index('id="qaPrintLabel"'), actions_popover.index('>Inventory</a>'))
        self.assertIn('class="product-tax-status is-taxable view-mode"', header)
        self.assertIn('aria-label="Taxable" title="Taxable">TAX</span>', header)
        self.assertNotIn('✓', header)
        self.assertRegex(
            header,
            r'class="header-status-label active">\s*Active\s*</span>',
        )
        self.assertIn('class="product-sub product-header-meta view-mode"', header)
        self.assertIn('class="product-meta-separator" aria-hidden="true">|</span>', header)
        self.assertNotIn('Visible &amp; Sellable', header)

        self.product.taxable = False
        self.product.status = False
        self.product.save(update_fields=["taxable", "status"])
        inactive_html = self._render()
        inactive_header = inactive_html[
            inactive_html.index('<div class="product-card-header'):
            inactive_html.index('class="receiving-strip')
        ]
        self.assertRegex(
            inactive_header,
            r'class="header-status-label inactive">\s*Inactive\s*</span>',
        )
        self.assertNotIn('Hidden from Orders', inactive_header)
        self.assertIn('class="product-tax-status is-tax-exempt view-mode"', inactive_header)
        self.assertIn('aria-label="Tax exempt" title="Tax exempt">TAX</span>', inactive_header)
        self.assertNotIn('✕', inactive_header)
        self.assertIn('background: #dcfce7;', html)
        self.assertIn('background: #fee2e2;', html)

    def test_price_and_stock_numbers_are_prominent_and_updates_flash_in_place(self):
        html = self._render()

        self.assertIn(".product-price {", html)
        self.assertIn("font-size: 2.25rem;", html)
        self.assertIn(".receive-stock-number {", html)
        self.assertIn("font-size: clamp(6rem, 10vw, 9rem);", html)
        self.assertNotIn('id="checkinLiveUpdate"', html)
        self.assertNotIn("Instant updates ready", html)
        self.assertIn(".checkin-update-flash", html)
        self.assertIn("window.showCheckinLiveUpdate = showCheckinLiveUpdate;", html)
        self.assertIn("window.flashCheckinUpdateTargets = flashCheckinUpdateTargets;", html)

    def test_restock_trend_is_visible_above_details_and_identifiers_are_ordered(self):
        html = self._render()

        workspace = html.index('<div class="receiving-restock-workspace has-restock">')
        receiving = html.index('<section class="receiving-strip')
        restock = html.index('<section class="restock-card')
        details = html.index('<details class="product-secondary-details"')
        barcode = html.index('<div class="detail-label">Barcode / UPC</div>')
        item_number = html.index('<div class="detail-label">Item Number</div>')
        notes = html.index('<div class="detail-label">Internal Notes</div>')

        self.assertLess(workspace, receiving)
        self.assertLess(receiving, restock)
        self.assertLess(restock, details)
        self.assertIn(
            "grid-template-columns: repeat(2, minmax(0, 1fr));",
            html,
        )
        self.assertIn(
            ".receiving-restock-workspace.has-restock { grid-template-columns: minmax(0, 1fr); }",
            html,
        )
        self.assertLess(barcode, item_number)
        self.assertLess(item_number, notes)
        header = html[html.index('<div class="product-card-header'):html.index('class="receiving-strip')]
        self.assertNotIn("Item Number:", header)
        self.assertNotIn("Category:", header)
        status = header.index('class="header-status-label active"')
        category = header.index(f"<strong>{self.category.name}</strong>")
        barcode_value = header.index(
            f'<span class="product-meta-barcode">{self.product.barcode}</span>'
        )
        product_name = header.index(f'<div class="product-name">{self.product.name}</div>')
        self.assertLess(status, category)
        self.assertLess(category, barcode_value)
        self.assertLess(barcode_value, product_name)
        self.assertIn('gap: 0.12rem 0.32rem;', html)
        self.assertIn('padding: 0.55rem 0.75rem;', html)
        self.assertIn("Barcode, item number and notes", html)

    def test_inline_edit_exposes_item_number_and_omits_duplicate_lot_summary(self):
        html = self._render()
        details_start = html.index('<details class="product-secondary-details"')
        details_end = html.index('</details>', details_start)
        details = html[details_start:details_end]

        self.assertEqual(details.count('name="item_number"'), 1)
        self.assertRegex(
            details,
            r'<div class="edit-mode">\s*<input[^>]+name="item_number"',
        )
        self.assertNotIn('type="hidden" name="item_number"', details)
        self.assertNotIn('Product Lot Number(s)', details)
        self.assertNotIn('receiving-lot-summary', details)
        self.assertIn('class="product-lots-editor"', details)
        self.assertIn('data-derive-stock-total', details)

    def test_inline_edit_enter_moves_to_next_field_without_overriding_notes_or_modifiers(self):
        html = self._render()
        shared_ui = (
            Path(settings.BASE_DIR) / "static" / "js" / "ui-system.js"
        ).read_text(encoding="utf-8")

        self.assertIn('data-enter-next-form', html)
        self.assertIn('aria-keyshortcuts="Enter"', html)
        self.assertIn('aria-describedby="inlineEditEnterHint"', html)
        self.assertIn('id="inlineEditEnterHint"', html)
        self.assertIn('<kbd>Enter</kbd>', html)
        self.assertIn('Next field. Internal Notes keeps Enter for new lines.', html)
        self.assertIn('function inlineEnterNavigationControls()', html)
        self.assertIn("if (!productCard.classList.contains('is-editing')) return;", html)
        self.assertIn("event.key !== 'Enter'", html)
        self.assertIn('|| event.shiftKey', html)
        self.assertIn('|| event.ctrlKey', html)
        self.assertIn("current.tagName === 'TEXTAREA'", html)
        self.assertIn("['hidden', 'button', 'submit', 'reset', 'checkbox', 'radio']", html)
        self.assertIn('event.preventDefault();', html)
        self.assertIn("[type=\"submit\"][form=\"inlineEditForm\"]", html)
        self.assertNotIn('workflow-shortcut-decal', html)
        self.assertIn('data-ui-open-shortcuts', html)
        self.assertIn("js/ui-system.js?v=20260830-presence1", html)
        self.assertIn(
            "['Alt + E', 'Expired products']",
            shared_ui,
        )
        self.assertIn(
            "['Enter', 'Next single-line field (Check-in and Add/Edit Product; Notes keeps new lines)']",
            shared_ui,
        )
        self.assertIn(
            "['Shift + Enter', 'Complete current order (Purchase page)']",
            shared_ui,
        )

    def test_receiving_strip_shows_clickable_saved_product_lots(self):
        first = ProductLot.objects.create(
            product=self.product,
            lot_number="VISIBLE-LOT-A",
            expiry_date=date(2031, 1, 31),
            quantity_on_hand=4,
        )
        second = ProductLot.objects.create(
            product=self.product,
            lot_number="VISIBLE-LOT-B",
            expiry_date=date(2032, 6, 30),
            quantity_on_hand=3,
        )

        html = self._render()
        lots_start = html.index('id="receivingSavedLots"')
        lots_end = html.index('class="receiving-lot-feedback"', lots_start)
        visible_lots = html[lots_start:lots_end]

        self.assertIn("Product lots", visible_lots)
        self.assertIn("2 saved", visible_lots)
        self.assertIn(f'data-receiving-lot-id="{first.pk}"', visible_lots)
        self.assertIn(f'data-receiving-lot-id="{second.pk}"', visible_lots)
        self.assertIn('data-lot-quantity="4"', visible_lots)
        self.assertIn('data-lot-quantity="3"', visible_lots)
        self.assertIn('class="receiving-lot-column-head"', visible_lots)
        self.assertIn("Lot name", visible_lots)
        self.assertIn("Stock level", visible_lots)
        self.assertIn("Expiry date", visible_lots)
        self.assertIn('class="receiving-lot-name">VISIBLE-LOT-A', visible_lots)
        self.assertIn('class="receiving-lot-stock">4 in stock', visible_lots)
        self.assertIn('class="receiving-lot-expiry">31-01-2031', visible_lots)
        self.assertIn('class="receiving-lot-name">VISIBLE-LOT-B', visible_lots)
        self.assertIn('class="receiving-lot-stock">3 in stock', visible_lots)
        self.assertIn('class="receiving-lot-expiry">30-06-2032', visible_lots)
        self.assertIn('<ul class="receiving-lot-list"', visible_lots)
        self.assertIn('<li class="receiving-lot-list-item">', visible_lots)
        self.assertIn('class="receiving-lot-row', visible_lots)
        self.assertIn("stock.className = 'receiving-lot-stock';", html)
        self.assertIn("expiry.className = 'receiving-lot-expiry';", html)
        self.assertIn("receivingLotSelect.dispatchEvent(new Event('change'", html)
        self.assertIn("window.renderReceivingLotList = renderReceivingLotList;", html)

    def test_product_lot_list_is_only_saved_picker_and_new_fields_are_on_demand(self):
        html = self._render()
        section_start = html.index('<section class="receiving-strip')
        section_end = html.index('</section>', section_start)
        receiving = html[section_start:section_end]

        product_lots = receiving.index('id="receivingSavedLots"')
        new_lot_button = receiving.index('id="receivingNewLotButton"')
        hidden_select = receiving.index('id="receivingLotSelect"')
        lot_fields = receiving.index('id="receivingNewLotFields"')
        lot_list = receiving.index('id="receivingLotList"')
        self.assertLess(product_lots, new_lot_button)
        self.assertLess(new_lot_button, hidden_select)
        self.assertLess(hidden_select, lot_fields)
        self.assertLess(lot_fields, lot_list)
        self.assertNotIn('<span>Receive into</span>', receiving)
        self.assertRegex(
            receiving,
            r'<select form="quickLotAddForm" id="receivingLotSelect" name="existing_lot_id" hidden',
        )
        self.assertIn('id="receivingNewLotFields" hidden', receiving)
        self.assertIn('aria-controls="receivingNewLotFields" aria-expanded="false"', receiving)
        self.assertIn('>＋ New lot</button>', receiving)
        self.assertRegex(receiving, r'>\s*＋ Receive')
        self.assertIn('id="receivingTargetName">New lot</strong>', receiving)
        self.assertIn('id="receivingTargetMeta">Enter a lot number and optional expiry</small>', receiving)
        self.assertIn(
            'grid-template-columns: minmax(198px, 0.7fr) minmax(0, 1.3fr);',
            html,
        )
        self.assertIn('gap: 0.65rem 0;', html)
        self.assertIn('.receive-stock-summary {', html)
        self.assertIn('align-items: center;', html)
        self.assertIn('justify-content: center;', html)
        self.assertIn('min-height: 300px;', html)
        self.assertIn('font-size: clamp(6rem, 10vw, 9rem);', html)
        self.assertNotIn('.receiving-lot-entry-stack {', html)
        self.assertIn('.receiving-saved-lots {', html)
        self.assertIn('grid-column: 2;', html)
        self.assertIn('grid-row: 2;', html)
        self.assertIn('flex-direction: column;', html)
        self.assertIn('align-self: stretch;', html)
        self.assertIn('max-width: 100%;', html)
        self.assertIn('.receiving-saved-lots-count {', html)
        self.assertIn('flex: 0 0 auto;', html)
        self.assertIn('white-space: nowrap;', html)
        self.assertIn(
            'grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.8fr) minmax(0, 1fr);',
            html,
        )
        self.assertIn('font-size: 1rem;', html)
        self.assertIn('font-size: 0.9rem;', html)
        self.assertIn('overflow-wrap: anywhere;', html)
        lot_row_styles = html[
            html.index('.receiving-lot-row span {'):
            html.index('.receiving-lot-stock {')
        ]
        self.assertIn('overflow: visible;', lot_row_styles)
        self.assertIn('white-space: normal;', lot_row_styles)
        self.assertNotIn('text-overflow: ellipsis;', lot_row_styles)
        self.assertIn('.receiving-lot-list {', html)
        self.assertIn('flex: 1 1 auto;', html)
        self.assertIn('list-style: none;', html)
        self.assertNotIn('repeat(auto-fit, minmax(145px, 1fr))', html)
        self.assertNotIn('class="receiving-lot-chip', html)
        self.assertIn("receivingNewLotFields.hidden = !isNewLot;", html)
        self.assertIn("receivingLotSelect.value = row.dataset.receivingLotId || '';", html)
        self.assertIn('syncReceivingTargetSummary();', html)
        self.assertIn('cancelReceivingNewLot.addEventListener', html)

    def test_inline_edit_derives_stock_from_lots_without_duplicate_expiry_details(self):
        html = self._render()
        details_start = html.index('<details class="product-secondary-details"')
        details_end = html.index('</details>', details_start)
        details = html[details_start:details_end]

        self.assertNotIn('<div class="detail-label">Expiry Date', details)
        self.assertNotIn('expiry-readonly-inline', html)
        self.assertNotIn('saved lots, expiry and notes', details)
        self.assertNotIn('id="checkinAddExpiry"', html)
        self.assertNotIn('id="checkinExpiryContainer"', html)
        self.assertIn('data-derive-stock-total', html)
        self.assertIn('data-lot-derived-total', html)
        self.assertIn('In stock is calculated automatically', html)
        self.assertIn('function refreshInlineLotStock()', html)
        self.assertIn("inlineStockInput.value = String(total);", html)
        self.assertIn("inlineStockDisplay.textContent = String(total);", html)

    def test_inline_edit_warns_when_direct_stock_change_and_lot_total_disagree(self):
        html = self._render()

        self.assertIn('name="inline_stock_baseline" value="7"', html)
        self.assertIn('function showInlineLotMismatch(message)', html)
        self.assertIn('window.noteCheckinStockChangedOutsideLots = function(quantity)', html)
        self.assertIn('inlineStockChangedOutsideLots &&', html)
        self.assertIn('inlineLotStockTotal() !== inlineAuthoritativeStock', html)
        self.assertIn('Update each “In this lot” value so the lots total ', html)
        self.assertIn('product-lots-editor.ui-lot-mismatch', html)
        self.assertIn('window.noteCheckinStockChangedOutsideLots(data.system_quantity)', html)
        self.assertIn('window.refreshCheckinInlineLotRows = function(markup, systemQuantity)', html)
        self.assertIn('window.refreshCheckinInlineLotRows(data.lot_rows_html, data.system_quantity)', html)
        self.assertIn('syncReceivingLotOptionsFromPage(page);', html)
        self.assertIn('function syncReceivingLotOptionsFromPage(page)', html)

    def test_invalid_fields_get_strong_local_highlighting_and_error_summary(self):
        html = self._render()

        self.assertEqual(html.count('id="inlineEditErrorSummary"'), 1)
        self.assertIn("border: 3px solid #dc2626 !important;", html)
        self.assertIn("box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.18) !important;", html)
        self.assertIn("secondaryDetails.open = true;", html)
        self.assertIn("markReceivingFieldInvalid(receivingLotExpiry, error.message);", html)
        self.assertIn("Update blocked — correct the red highlighted fields.", html)

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

    def test_product_movement_graph_is_inside_equal_width_restock_panel(self):
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

        restock_start = html.index('<section class="restock-card')
        restock_end = html.index('</section>', restock_start)
        restock_panel = html[restock_start:restock_end]
        rail_start = html.index('id="checkinActivityRail"')
        rail_end = html.index('</aside>', rail_start)
        activity_rail = html[rail_start:rail_end]

        self.assertIn('class="restock-product-movement"', restock_panel)
        self.assertIn('id="productMovementSummary"', restock_panel)
        self.assertNotIn('id="productMovementSummary"', activity_rail)
        self.assertLess(
            html.index('id="productMovementSummary"'),
            html.index('id="sessionHistoryPanel"'),
        )
        self.assertIn(
            'grid-template-columns: repeat(2, minmax(0, 1fr));',
            html,
        )
        self.assertIn('.restock-card { display: flex; flex-direction: column;', html)
        self.assertIn('min-height: 220px;', html)
        self.assertIn('height: min(820px, calc(100dvh - 8.5rem));', html)
        self.assertIn('.activity-rail-card.ph-card {', html)
        self.assertIn('height: 100%;', html)
        self.assertIn('max-height: none;', html)
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
            r"height:\s*100%;",
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
            r"[^}]*grid-row:\s*1;[^}]*position:\s*static;",
        )

    def test_lookup_and_history_are_left_of_the_product_workspace(self):
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
        self.assertLess(product_start, side_start)
        self.assertLess(primary_start, product_start)
        self.assertLess(side_start, search_start)
        self.assertLess(search_start, rail_start)
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
            template_source,
            r"\.checkin-side-column\s*\{[^}]*grid-column:\s*1;"
            r"[^}]*grid-row:\s*1;",
        )
        self.assertNotIn("grid-row: 1 / span 2;", template_source)
        self.assertNotIn("grid-row: 1 / span 2;", shared_css)
        self.assertIn(
            "grid-template-columns: clamp(320px, 22vw, 380px) minmax(0, 1fr);",
            template_source,
        )
        self.assertIn('grid-template-rows: auto minmax(0, 1fr);', template_source)
        self.assertIn('display: block;', template_source)

        mobile_start = template_source.index("@media (max-width: 1049px)")
        mobile_end = template_source.find("@media", mobile_start + 1)
        mobile_styles = template_source[
            mobile_start: mobile_end if mobile_end != -1 else None
        ]
        self.assertRegex(
            mobile_styles,
            r"\.checkin-primary-column\s*\{[^}]*grid-column:\s*1;"
            r"[^}]*grid-row:\s*2;[^}]*position:\s*static;",
        )
        self.assertRegex(
            mobile_styles,
            r"\.checkin-side-column\s*\{[^}]*grid-column:\s*1;"
            r"[^}]*grid-row:\s*1;[^}]*position:\s*static;",
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
            "width: calc(100% - var(--nav-desktop, 120px));",
            html,
        )
        self.assertIn("margin: 0 0 0 var(--nav-desktop, 120px);", html)
        self.assertIn(
            "top: calc(var(--checkin-sticky-offset, 68px) + 0.75rem);",
            html,
        )
        self.assertIn(
            "body.app-shell .container { width: 100%; margin: 0; padding: 0; }",
            html,
        )
        self.assertNotIn("calc(64px + clamp(0.75rem, 1.4vw, 1.5rem));", html)
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
