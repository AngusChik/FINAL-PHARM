from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .models import Category, Order, PagePresence, Product


@override_settings(AXES_ENABLED=False, GLOBAL_MAX_SESSIONS=10)
class SeniorsDiscountWorkflowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="senior-discount-user",
            password="pass1234",
            is_staff=True,
        )
        self.other_user = User.objects.create_user(
            username="senior-discount-other",
            password="pass1234",
            is_staff=True,
        )
        category = Category.objects.create(name="Senior discount test")
        self.product = Product.objects.create(
            name="Taxable senior discount item",
            barcode="SENIOR-DISCOUNT-001",
            price=Decimal("13.99"),
            quantity_in_stock=10,
            taxable=True,
            category=category,
        )
        self.client.force_login(self.user)
        self.client.post(
            reverse("add_product_by_id", args=[self.product.pk]),
            {"quantity": "1"},
        )
        self.order = Order.objects.get(user=self.user, submitted=False)

    def test_toggle_persists_without_resetting_draft_timer(self):
        original_expiry = self.order.draft_expires_at

        response = self.client.post(
            reverse("create_order"),
            {"action": "toggle_seniors_discount"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertRedirects(response, reverse("create_order"), fetch_redirect_response=False)
        self.order.refresh_from_db()
        self.assertTrue(self.order.seniors_discount)
        self.assertEqual(self.order.draft_expires_at, original_expiry)

        page = self.client.get(reverse("create_order"))
        self.assertEqual(page.context["total_price_before_tax"], Decimal("13.99"))
        self.assertEqual(page.context["seniors_discount_amount"], Decimal("1.40"))
        self.assertEqual(page.context["tax_amount"], Decimal("1.64"))
        self.assertEqual(page.context["total_price_after_tax"], Decimal("14.23"))
        self.assertContains(page, 'aria-pressed="true"')
        self.assertContains(page, "Seniors Discount (&minus;10%)")

        response = self.client.post(
            reverse("create_order"),
            {"action": "toggle_seniors_discount"},
        )
        self.assertEqual(response.status_code, 302)
        self.order.refresh_from_db()
        self.assertFalse(self.order.seniors_discount)

    def test_discount_state_is_scoped_to_the_draft_owner(self):
        self.client.post(
            reverse("create_order"),
            {"action": "toggle_seniors_discount"},
        )

        PagePresence.objects.all().delete()
        other_browser = Client()
        other_browser.force_login(self.other_user)
        page = other_browser.get(reverse("create_order"))

        self.assertIsNone(page.context["order"])
        self.assertFalse(page.context["seniors_discount"])
        self.assertNotContains(
            page,
            '<button type="submit" class="ot-discount-toggle',
        )


class SeamlessFormActionRegressionTests(SimpleTestCase):
    def test_shared_handler_uses_unclobberable_form_attributes(self):
        script = (
            Path(settings.BASE_DIR) / "static" / "js" / "ui-system.js"
        ).read_text(encoding="utf-8")

        self.assertIn("form.getAttribute('action') || window.location.href", script)
        self.assertIn("form.getAttribute('method') || 'POST'", script)
        self.assertNotIn("fetch(form.action || window.location.href", script)

    def test_base_template_busts_stale_shared_script_cache(self):
        template = (
            Path(settings.BASE_DIR) / "app" / "templates" / "base.html"
        ).read_text(encoding="utf-8")

        self.assertIn("ui-system.js' %}?v=20260818-ui15", template)
