from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .models import Product


class NotificationSourceContractTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        root = Path(settings.BASE_DIR)
        templates = root / "app" / "templates"
        cls.base = (templates / "base.html").read_text(encoding="utf-8")
        cls.embed = (templates / "ordering_sheet_embed.html").read_text(
            encoding="utf-8"
        )
        cls.partial = (templates / "partials" / "_notifications.html").read_text(
            encoding="utf-8"
        )
        cls.notifications = (root / "static" / "js" / "notifications.js").read_text(
            encoding="utf-8"
        )
        cls.ui_script = (root / "static" / "js" / "ui-system.js").read_text(
            encoding="utf-8"
        )
        cls.styles = (root / "static" / "css" / "ui-system.css").read_text(
            encoding="utf-8"
        )
        cls.templates = {
            name: (templates / name).read_text(encoding="utf-8")
            for name in (
                "checkout.html",
                "order_form.html",
                "delivery.html",
                "expired_products.html",
                "item_list.html",
                "new_product.html",
                "archive_recovery.html",
                "login.html",
                "checkin.html",
                "partials/_ordering_sheet.html",
            )
        }

    def test_toasts_and_action_banner_use_one_shared_entrypoint(self):
        self.assertEqual(self.base.count("{% include 'partials/_notifications.html' %}"), 1)
        self.assertIn("notifications_toasts_only=True", self.embed)
        self.assertIn('data-ui-toast-stack', self.partial)
        self.assertIn('data-ui-action-banner', self.partial)
        self.assertIn("window.showToast = showToast;", self.notifications)
        self.assertIn("window.showActionBanner", self.notifications)
        self.assertIn("parsed.origin !== window.location.origin", self.notifications)
        self.assertIn(".ui-toast-stack,", self.styles)
        self.assertIn(".ui-action-banner,", self.styles)
        self.assertNotIn(
            "document.querySelectorAll('.toast-msg').forEach",
            self.templates["checkin.html"],
        )

    def test_actionable_workflows_use_the_shared_dialog(self):
        self.assertIn("window.uiDialog = openDialog;", self.ui_script)
        self.assertIn("window.uiConfirm = confirmAction;", self.ui_script)
        self.assertIn("dismissible = options.dismissible !== false", self.ui_script)
        self.assertIn("window.uiConfirm({", self.templates["checkout.html"])
        self.assertIn("window.uiConfirm({", self.templates["order_form.html"])
        self.assertIn("window.uiConfirm({", self.templates["partials/_ordering_sheet.html"])
        self.assertIn("window.uiDialog({", self.templates["expired_products.html"])
        self.assertIn("window.uiDialog({", self.base)
        self.assertNotIn("pl-takeover", self.base)

        combined = "\n".join(self.templates.values())
        for legacy_name in (
            "expiry-confirm-modal",
            "inactive-confirm-modal",
            "active-conflict-modal",
            "nextStepsOverlay",
            "os-confirm-modal",
            "dv-toast",
            "os-msg",
        ):
            with self.subTest(legacy_name=legacy_name):
                self.assertNotIn(legacy_name, combined)

    def test_inline_validation_has_a_shared_accessible_path(self):
        self.assertIn("function attachFieldError(field, source)", self.ui_script)
        self.assertIn("data-ui-generated-field-error", self.ui_script)
        self.assertIn("field.setAttribute('aria-describedby'", self.ui_script)
        self.assertIn("[data-field-error]", self.styles)
        self.assertIn('class="ui-field-error"', self.templates["item_list.html"])
        self.assertIn("np-field-errors ui-field-error", self.templates["new_product.html"])
        self.assertIn("ui-inline-alert--warning", self.templates["archive_recovery.html"])
        self.assertIn("ui-inline-alert--error", self.templates["login.html"])


@override_settings(AXES_ENABLED=False)
class AlertBannerAPIContractTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="notification-contract-user",
            password="test-password",
        )
        self.client.force_login(self.user)

    def test_expiring_product_alert_is_stable_actionable_and_typed(self):
        Product.objects.create(
            name="Expiring Contract Product",
            price=Decimal("1.00"),
            quantity_in_stock=1,
            expiry_date=date.today() + timedelta(days=3),
            status=True,
        )

        response = self.client.get(reverse("alert_banner_api"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "alerts": [
                    {
                        "key": "products-expiring-this-week",
                        "level": "warning",
                        "text": "1 product expiring this week",
                        "url": f"{reverse('expired_products')}?date_filter=1_week",
                    }
                ]
            },
        )


@override_settings(AXES_ENABLED=False)
class InlineFieldValidationRenderTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="inline-validation-user",
            password="test-password",
        )
        self.client.force_login(self.user)

    def test_special_order_errors_are_rendered_beside_their_fields(self):
        response = self.client.post(
            reverse("item_list"),
            {},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-describedby="first_name_error"')
        self.assertContains(
            response,
            'class="ui-field-error" id="first_name_error" data-field-error role="alert"',
        )
