from decimal import Decimal

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from .models import (
    Category,
    CustomLabelQueueItem,
    LabelQueueItem,
    LabelSession,
    LabelSessionItem,
    Product,
)


@override_settings(AXES_ENABLED=False, GLOBAL_MAX_SESSIONS=5)
class CustomLabelPersistenceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="labeluser", password="pass1234", is_staff=True,
        )
        self.other_user = User.objects.create_user(
            username="otherlabeluser", password="pass1234", is_staff=True,
        )
        self.category = Category.objects.create(name="General")
        self.product = Product.objects.create(
            name="Test Product", price=Decimal("9.99"),
            quantity_in_stock=10, category=self.category, barcode="12345",
        )
        self.client = Client()
        self.client.force_login(self.user)

    def _add_custom_label(self, title="Compression Socks", copies=2):
        response = self.client.post(reverse("label_printing"), {
            "add_custom_label": "1",
            "custom_title": title,
            "line_text_0": "Size Medium",
            "line_price_0": "12.50",
            "copies": str(copies),
        })
        self.assertRedirects(response, reverse("label_printing"))
        return CustomLabelQueueItem.objects.get(user=self.user, title=title)

    def test_custom_label_is_database_backed_and_survives_new_login(self):
        label = self._add_custom_label()

        self.assertEqual(label.lines, [{"text": "Size Medium", "price": 12.5}])
        self.assertNotIn("custom_labels", self.client.session)

        self.client.logout()
        second_browser = Client()
        second_browser.force_login(self.user)
        response = second_browser.get(reverse("label_printing"))

        self.assertContains(response, "Compression Socks")
        self.assertEqual(CustomLabelQueueItem.objects.filter(user=self.user).count(), 1)

    def test_surviving_legacy_session_labels_are_migrated_once(self):
        session = self.client.session
        session["custom_labels"] = [{
            "title": "Legacy Label",
            "lines": [{"text": "Old line", "price": 3.25}],
            "copies": 3,
        }]
        session.save()

        self.client.get(reverse("label_printing"))
        self.client.get(reverse("label_printing"))

        label = CustomLabelQueueItem.objects.get(user=self.user)
        self.assertEqual(label.title, "Legacy Label")
        self.assertEqual(label.copies, 3)
        self.assertNotIn("custom_labels", self.client.session)
        self.assertEqual(CustomLabelQueueItem.objects.filter(user=self.user).count(), 1)

    def test_edit_and_remove_use_user_owned_database_ids(self):
        first = self._add_custom_label("First")
        second = self._add_custom_label("Second")
        other = CustomLabelQueueItem.objects.create(
            user=self.other_user, title="Other user's label", lines=[], copies=1,
        )

        self.client.post(reverse("label_printing"), {
            "edit_custom_label": str(second.pk),
            "custom_title": "Second updated",
            "line_text_0": "Updated line",
            "line_price_0": "7.00",
            "copies": "4",
        })
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual(first.title, "First")
        self.assertEqual(second.title, "Second updated")
        self.assertEqual(second.copies, 4)

        self.client.post(reverse("label_printing"), {
            "remove_custom_label": str(first.pk),
        })
        self.client.post(reverse("label_printing"), {
            "remove_custom_label": str(other.pk),
        })
        self.assertFalse(CustomLabelQueueItem.objects.filter(pk=first.pk).exists())
        self.assertTrue(CustomLabelQueueItem.objects.filter(pk=other.pk).exists())

    def test_pdf_history_snapshots_and_restores_custom_labels(self):
        LabelQueueItem.objects.create(user=self.user, product=self.product, qty=2)
        custom = self._add_custom_label("Durable Custom", copies=3)

        response = self.client.get(reverse("generate_label_pdf"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        session = LabelSession.objects.get(user=self.user)
        self.assertEqual(session.label_count, 5)
        custom_snapshot = LabelSessionItem.objects.get(session=session, is_custom=True)
        self.assertIsNone(custom_snapshot.product)
        self.assertEqual(custom_snapshot.product_name, custom.title)
        self.assertEqual(custom_snapshot.custom_lines, custom.lines)
        self.assertEqual(custom_snapshot.qty, 3)

        detail = self.client.get(reverse("label_session_detail", args=[session.pk])).json()
        custom_detail = next(item for item in detail["items"] if item["is_custom"])
        self.assertEqual(custom_detail["custom_lines"], custom.lines)

        LabelQueueItem.objects.filter(user=self.user).delete()
        CustomLabelQueueItem.objects.filter(user=self.user).delete()
        regenerate = self.client.post(
            reverse("label_session_regenerate", args=[session.pk]),
        )
        self.assertTrue(regenerate.json()["ok"])
        self.assertEqual(LabelQueueItem.objects.get(user=self.user).qty, 2)
        restored = CustomLabelQueueItem.objects.get(user=self.user)
        self.assertEqual(restored.title, custom.title)
        self.assertEqual(restored.lines, custom.lines)
        self.assertEqual(restored.copies, 3)

        add_again = self.client.post(
            reverse("label_session_add_to_queue", args=[session.pk]),
        )
        self.assertTrue(add_again.json()["ok"])
        self.assertEqual(CustomLabelQueueItem.objects.filter(user=self.user).count(), 2)

    def test_clear_queue_removes_product_and_custom_labels_for_current_user(self):
        LabelQueueItem.objects.create(user=self.user, product=self.product, qty=1)
        self._add_custom_label()
        other = CustomLabelQueueItem.objects.create(
            user=self.other_user, title="Keep me", lines=[], copies=1,
        )

        self.client.post(reverse("label_printing"), {"clear_queue": "1"})

        self.assertFalse(LabelQueueItem.objects.filter(user=self.user).exists())
        self.assertFalse(CustomLabelQueueItem.objects.filter(user=self.user).exists())
        self.assertTrue(CustomLabelQueueItem.objects.filter(pk=other.pk).exists())
