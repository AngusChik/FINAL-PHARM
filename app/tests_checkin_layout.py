import re
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Category, CheckinSession, Product, StockChange


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
        self.assertIn("let barcodeSubmitPending = false", html)
        self.assertIn("if (barcodeSubmitPending)", html)
        self.assertIn("submittedValues", html)
        self.assertIn("Keep the scanner pending and controls locked", html)
        self.assertIn("receivingDraftNeedsRetry", html)
        self.assertIn("window.setReceivingActionLocked = function(locked)", html)
        self.assertIn("receivingActionLockDepth", html)
        self.assertIn("window.isReceivingActionLocked = function()", html)
        self.assertIn("A stock update is finishing", html)
        self.assertIn(
            "readOnly: control.id === 'product_lookup' ? !!control.readOnly : null",
            html,
        )
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

    def test_secondary_details_are_collapsed_and_edit_expands_them(self):
        html = self._render()

        self.assertIn(
            '<details class="product-secondary-details" id="productSecondaryDetails">',
            html,
        )
        self.assertNotIn(
            '<details class="product-secondary-details" id="productSecondaryDetails" open>',
            html,
        )
        self.assertIn("productSecondaryDetails.open = true", html)
        self.assertIn("productSecondaryDetails.open = false", html)

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

    def test_collapsed_product_workspace_stays_directly_below_scanner(self):
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
        self.assertNotIn(
            '<details class="product-secondary-details" id="productSecondaryDetails" open>',
            html,
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

        self.session.inventory_mode = True
        self.session.save(update_fields=["inventory_mode"])
        inventory_html = self._render()
        inventory_side = inventory_html.index('class="checkin-side-column"')
        inventory_rail = inventory_html.index('id="checkinActivityRail"')
        inventory_count = inventory_html.index('class="ic-count-col"')
        self.assertLess(inventory_side, inventory_rail)
        self.assertLess(inventory_rail, inventory_count)

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
