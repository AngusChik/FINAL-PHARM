from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Category, CheckinSession, Product


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
        ):
            self.assertEqual(html.count(f'id="{element_id}"'), 1)

        self.assertLess(html.index('id="search-box"'), html.index('class="right-items"'))
        self.assertNotIn(">⚡ Quick Actions<", html)

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

    def test_session_and_product_history_share_keyboard_accessible_tabs(self):
        html = self._render()

        self.assertIn('role="tablist" aria-label="Activity history"', html)
        self.assertIn(
            'id="activitySessionTab" role="tab"\n                  aria-selected="true" aria-controls="sessionHistoryPanel"',
            html,
        )
        self.assertIn('id="activityProductTab" role="tab" tabindex="-1"', html)
        self.assertIn('id="sessionHistoryPanel" role="tabpanel"', html)
        self.assertIn('id="productHistoryPanel" role="tabpanel"', html)
        self.assertEqual(html.count('id="sessionHistoryCard"'), 1)
        self.assertIn("event.key === 'ArrowRight'", html)
        self.assertIn("event.key === 'ArrowLeft'", html)

    def test_empty_workspace_still_shows_session_activity_without_dead_tab(self):
        html = self._render(with_product=False)

        self.assertIn('id="checkinActivityRail"', html)
        self.assertIn('id="activitySessionTab"', html)
        self.assertNotIn('id="activityProductTab"', html)
        self.assertNotIn('id="productHistoryPanel"', html)

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
