from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import AnonymousUser, User
from django.db import IntegrityError, transaction
from django.test import RequestFactory, SimpleTestCase, TestCase, Client
from django.urls import reverse
from django.utils.timezone import now

from .models import (
    Product, Category, CheckinSession, StockChange,
    CheckoutOrder, CheckoutOrderItem, UserSession, UserAction,
)
from .views import OrderPDFView


class FakeDetailCollection(list):
    def select_related(self, *_args, **_kwargs):
        return self

    def all(self):
        return self


class OrderPDFViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.order_id = 77
        self.template_dir = Path(__file__).resolve().parent / "templates"
        self.fake_order = SimpleNamespace(
            order_id=self.order_id,
            order_date=SimpleNamespace(strftime=lambda fmt: "June 04, 2026 09:30" if "%H:%M" in fmt else "June 04, 2026"),
            submitted=True,
        )
        self.fake_order.details = FakeDetailCollection(
            [
                SimpleNamespace(
                    product=None,
                    quantity=2,
                    price=Decimal("12.99"),
                    display_name="Vitamin C",
                    display_barcode="123456789012",
                )
            ]
        )

    def test_order_pdf_requires_login(self):
        request = self.factory.get(reverse("order_pdf", args=[self.order_id]))
        request.user = AnonymousUser()

        response = OrderPDFView.as_view()(request, order_id=self.order_id)

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response.url)

    @patch("app.views.get_object_or_404")
    def test_order_pdf_download_returns_pdf_attachment(self, mock_get_object_or_404):
        mock_get_object_or_404.return_value = self.fake_order
        request = self.factory.get(reverse("order_pdf", args=[self.order_id]))
        request.user = SimpleNamespace(is_authenticated=True)

        response = OrderPDFView.as_view()(request, order_id=self.order_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("attachment;", response["Content-Disposition"])
        self.assertIn(f"order_{self.order_id}_transaction_record_", response["Content-Disposition"])
        self.assertTrue(response.content.startswith(b"%PDF"))

    def test_order_templates_reference_pdf_route(self):
        order_view_template = (self.template_dir / "order_view.html").read_text(encoding="utf-8")
        order_detail_template = (self.template_dir / "order_detail.html").read_text(encoding="utf-8")

        self.assertIn("url 'order_pdf' order.order_id", order_view_template)
        self.assertIn("url 'order_pdf' order.order_id", order_detail_template)


class CheckinSessionEditTests(TestCase):
    """Test adjust-line and remove-line flows for session editing."""

    def setUp(self):
        self.staff = User.objects.create_user(
            username="staffuser", password="pass1234", is_staff=True,
        )
        self.regular = User.objects.create_user(
            username="regularuser", password="pass1234", is_staff=False,
        )
        self.category = Category.objects.create(name="General")
        self.product = Product.objects.create(
            name="Test Vitamin",
            price=Decimal("9.99"),
            quantity_in_stock=20,
            category=self.category,
        )
        self.session = CheckinSession.objects.create(
            user=self.staff,
            scanned_by="Test Person",
            ended_at=now(),  # completed session
        )
        # Simulate two stock-change lines in the session
        self.change_add = StockChange.objects.create(
            product=self.product,
            session=self.session,
            user=self.staff,
            change_type="checkin",
            quantity=5,
            note="Original add",
        )
        self.change_sub = StockChange.objects.create(
            product=self.product,
            session=self.session,
            user=self.staff,
            change_type="checkin_delete1",
            quantity=2,
            note="Original remove",
        )
        self.client = Client()

    # ── Reopen ──

    def test_non_staff_cannot_reopen(self):
        self.client.force_login(self.regular, backend="django.contrib.auth.backends.ModelBackend")
        url = reverse("checkin_session_reopen", kwargs={"session_id": self.session.pk})
        resp = self.client.post(url)
        self.session.refresh_from_db()
        self.assertIsNotNone(self.session.ended_at)  # still closed

    def test_staff_can_reopen(self):
        self.client.force_login(self.staff, backend="django.contrib.auth.backends.ModelBackend")
        url = reverse("checkin_session_reopen", kwargs={"session_id": self.session.pk})
        resp = self.client.post(url)
        self.session.refresh_from_db()
        self.assertIsNone(self.session.ended_at)
        self.assertTrue(self.session.is_active)
        self.assertIsNotNone(self.session.reopened_at)
        self.assertTrue(self.session.is_reopened)

    def test_reopen_keeps_other_active_sessions(self):
        """Reopening a session does NOT close other active sessions."""
        self.client.force_login(self.staff, backend="django.contrib.auth.backends.ModelBackend")

        # Create an active session
        active = CheckinSession.objects.create(
            user=self.staff, scanned_by="Staff", note="active one",
        )
        self.assertTrue(active.is_active)

        # Reopen the completed session
        url = reverse("checkin_session_reopen", kwargs={"session_id": self.session.pk})
        self.client.post(url)

        # The other active session should still be active
        active.refresh_from_db()
        self.assertIsNone(active.ended_at)
        self.assertTrue(active.is_active)

        # The reopened session should also be active
        self.session.refresh_from_db()
        self.assertIsNone(self.session.ended_at)
        self.assertTrue(self.session.is_active)

    # ── Adjust line ──

    def test_staff_adjust_line_updates_stock_and_audit(self):
        """Adjusting an add-line from qty 5→8 should increase product stock by 3."""
        self.client.force_login(self.staff, backend="django.contrib.auth.backends.ModelBackend")
        stock_before = self.product.quantity_in_stock  # 20

        url = reverse("checkin_session_adjust", kwargs={
            "session_id": self.session.pk,
            "change_id": self.change_add.pk,
        })
        resp = self.client.post(url, {"new_qty": 8})
        self.assertEqual(resp.status_code, 302)

        self.product.refresh_from_db()
        self.change_add.refresh_from_db()

        # Stock went up by 3 (8 - 5)
        self.assertEqual(self.product.quantity_in_stock, stock_before + 3)
        # Change row updated
        self.assertEqual(self.change_add.quantity, 8)
        # Corrective audit entry created
        corr = StockChange.objects.filter(
            session=self.session,
            note__contains="line adjusted",
        ).first()
        self.assertIsNotNone(corr)
        self.assertEqual(corr.quantity, 3)
        self.assertEqual(corr.change_type, "error_add")

    def test_non_staff_adjust_blocked(self):
        self.client.force_login(self.regular, backend="django.contrib.auth.backends.ModelBackend")
        url = reverse("checkin_session_adjust", kwargs={
            "session_id": self.session.pk,
            "change_id": self.change_add.pk,
        })
        resp = self.client.post(url, {"new_qty": 99})
        self.assertEqual(resp.status_code, 403)
        self.change_add.refresh_from_db()
        self.assertEqual(self.change_add.quantity, 5)  # unchanged

    # ── Remove line ──

    def test_staff_remove_add_line_reverses_stock(self):
        """Removing an add-line of qty 5 should subtract 5 from product stock."""
        self.client.force_login(self.staff, backend="django.contrib.auth.backends.ModelBackend")
        stock_before = self.product.quantity_in_stock  # 20

        url = reverse("checkin_session_remove_line", kwargs={
            "session_id": self.session.pk,
            "change_id": self.change_add.pk,
        })
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 302)

        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, stock_before - 5)
        # Original change row deleted
        self.assertFalse(StockChange.objects.filter(pk=self.change_add.pk).exists())
        # Corrective entry exists
        corr = StockChange.objects.filter(
            session=self.session,
            note__contains="line removed",
        ).first()
        self.assertIsNotNone(corr)
        self.assertEqual(corr.quantity, 5)
        self.assertEqual(corr.change_type, "error_subtract")

    def test_staff_remove_subtract_line_reverses_stock(self):
        """Removing a subtract-line of qty 2 should add 2 back to product stock."""
        self.client.force_login(self.staff, backend="django.contrib.auth.backends.ModelBackend")
        stock_before = self.product.quantity_in_stock  # 20

        url = reverse("checkin_session_remove_line", kwargs={
            "session_id": self.session.pk,
            "change_id": self.change_sub.pk,
        })
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 302)

        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, stock_before + 2)
        self.assertFalse(StockChange.objects.filter(pk=self.change_sub.pk).exists())

    def test_non_staff_remove_blocked(self):
        self.client.force_login(self.regular, backend="django.contrib.auth.backends.ModelBackend")
        url = reverse("checkin_session_remove_line", kwargs={
            "session_id": self.session.pk,
            "change_id": self.change_add.pk,
        })
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 403)
        self.assertTrue(StockChange.objects.filter(pk=self.change_add.pk).exists())

    # ── Combined flow: adjust then remove ──

    def test_adjust_then_remove_flow(self):
        """Full flow: adjust add-line 5→10, then remove subtract-line. Assert final stock."""
        self.client.force_login(self.staff, backend="django.contrib.auth.backends.ModelBackend")
        stock_before = self.product.quantity_in_stock  # 20

        # Step 1: adjust add-line from 5 to 10 (+5 stock)
        self.client.post(
            reverse("checkin_session_adjust", kwargs={
                "session_id": self.session.pk,
                "change_id": self.change_add.pk,
            }),
            {"new_qty": 10},
        )
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, stock_before + 5)  # 25

        # Step 2: remove the subtract-line of qty 2 (+2 stock)
        self.client.post(
            reverse("checkin_session_remove_line", kwargs={
                "session_id": self.session.pk,
                "change_id": self.change_sub.pk,
            }),
        )
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, stock_before + 5 + 2)  # 27

        # The adjusted add-line still exists with qty 10
        self.change_add.refresh_from_db()
        self.assertEqual(self.change_add.quantity, 10)

        # The subtract-line is gone
        self.assertFalse(StockChange.objects.filter(pk=self.change_sub.pk).exists())

        # Audit trail: 2 corrective entries created (1 adjust + 1 remove)
        corrections = StockChange.objects.filter(
            session=self.session,
            change_type__in=["error_add", "error_subtract"],
        ).exclude(pk=self.change_add.pk)
        self.assertEqual(corrections.count(), 2)


class CheckoutTests(TestCase):
    """PU checkout — durable per-user checkout flow."""

    def setUp(self):
        self.pu = User.objects.create_user(username="pu", password="pass1234", is_staff=False)
        self.admin = User.objects.create_user(username="gina", password="pass1234", is_staff=True)
        self.category = Category.objects.create(name="General")
        self.product = Product.objects.create(
            name="Test Vitamin", price=Decimal("10.00"),
            quantity_in_stock=20, category=self.category,
            barcode="12345", taxable=True,
        )
        self.product2 = Product.objects.create(
            name="Bandages", price=Decimal("5.00"),
            quantity_in_stock=10, category=self.category,
            barcode="67890", taxable=False,
        )
        self.client = Client()

    def _register_session(self, user):
        """Mirror what CustomLoginView does so the concurrency middleware is happy."""
        skey = self.client.session.session_key
        UserSession.objects.get_or_create(user=user, session_key=skey)
        return skey

    # ── creation / resume ──
    def test_first_get_creates_single_draft(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        resp = self.client.get(reverse("checkout"))
        self.assertEqual(resp.status_code, 200)
        drafts = CheckoutOrder.objects.filter(user=self.pu, status="draft")
        self.assertEqual(drafts.count(), 1)
        # Reload reuses the same draft
        self.client.get(reverse("checkout"))
        self.assertEqual(CheckoutOrder.objects.filter(user=self.pu, status="draft").count(), 1)

    def test_add_by_barcode_increments_single_line(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        self.client.post(reverse("checkout"), {"barcode": "12345", "quantity": 1})
        self.client.post(reverse("checkout"), {"barcode": "12345", "quantity": 1})
        checkout = CheckoutOrder.objects.get(user=self.pu, status="draft")
        items = checkout.items.all()
        self.assertEqual(items.count(), 1)
        self.assertEqual(items.first().quantity, 2)

    def test_delete_item_decrements_then_removes(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        checkout = CheckoutOrder.objects.create(user=self.pu, status="draft")
        item = CheckoutOrderItem.objects.create(
            checkout=checkout, product=self.product,
            product_name=self.product.name, price=self.product.price,
            taxable=True, quantity=2,
        )
        url = reverse("checkout_delete_item", kwargs={"item_id": item.pk})
        self.client.post(url)
        item.refresh_from_db()
        self.assertEqual(item.quantity, 1)
        self.client.post(url)
        self.assertFalse(CheckoutOrderItem.objects.filter(pk=item.pk).exists())

    def test_submit_decrements_stock_once_and_records_change(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        checkout = CheckoutOrder.objects.create(user=self.pu, status="draft")
        CheckoutOrderItem.objects.create(
            checkout=checkout, product=self.product,
            product_name=self.product.name, price=self.product.price,
            taxable=True, quantity=3,
        )
        resp = self.client.post(reverse("checkout_submit"))
        self.assertEqual(resp.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 17)  # 20 - 3
        self.assertEqual(
            StockChange.objects.filter(product=self.product, change_type="checkout").count(), 1
        )
        checkout.refresh_from_db()
        self.assertEqual(checkout.status, "submitted")
        self.assertIsNotNone(checkout.submitted_at)
        self.assertEqual(checkout.total_price, Decimal("30.00") + Decimal("30.00") * Decimal("0.13"))
        # Items kept as history; a new draft is available next visit
        self.assertEqual(checkout.items.count(), 1)
        self.assertTrue(
            UserAction.objects.filter(user=self.pu, action="checkout_submit").exists()
        )

    def test_submit_empty_blocked(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        CheckoutOrder.objects.create(user=self.pu, status="draft")
        resp = self.client.post(reverse("checkout_submit"))
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(CheckoutOrder.objects.filter(user=self.pu, status="submitted").exists())
        self.assertEqual(StockChange.objects.count(), 0)

    def test_checkout_new_discards_items(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        skey = self._register_session(self.pu)
        checkout = CheckoutOrder.objects.create(
            user=self.pu, status="draft", active_session_key=skey,
        )
        CheckoutOrderItem.objects.create(
            checkout=checkout, product=self.product,
            product_name=self.product.name, price=self.product.price, quantity=2,
        )
        resp = self.client.post(reverse("checkout_new"))
        self.assertEqual(resp.status_code, 302)
        checkout.refresh_from_db()
        self.assertEqual(checkout.items.count(), 0)
        self.assertTrue(UserAction.objects.filter(user=self.pu, action="checkout_new").exists())

    def test_second_live_session_shows_conflict(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        skey = self._register_session(self.pu)
        # A second, still-live session on another computer owns the draft
        UserSession.objects.create(user=self.pu, session_key="otherkey12345678")
        checkout = CheckoutOrder.objects.create(
            user=self.pu, status="draft", active_session_key="otherkey12345678",
        )
        CheckoutOrderItem.objects.create(
            checkout=checkout, product=self.product,
            product_name=self.product.name, price=self.product.price, quantity=1,
        )
        resp = self.client.get(reverse("checkout"))
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.context["show_active_conflict"])
        self.assertTrue(len(resp.context["other_sessions"]) >= 1)

    def test_auto_resume_when_other_session_dead(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        skey = self._register_session(self.pu)
        # active_session_key points to a session that no longer exists → no conflict
        checkout = CheckoutOrder.objects.create(
            user=self.pu, status="draft", active_session_key="ghostkey99999999",
        )
        CheckoutOrderItem.objects.create(
            checkout=checkout, product=self.product,
            product_name=self.product.name, price=self.product.price, quantity=1,
        )
        resp = self.client.get(reverse("checkout"))
        self.assertFalse(resp.context["show_active_conflict"])
        checkout.refresh_from_db()
        self.assertEqual(checkout.active_session_key, skey)  # ownership claimed

    def test_deleted_product_line_survives_and_removable(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        checkout = CheckoutOrder.objects.create(user=self.pu, status="draft")
        item = CheckoutOrderItem.objects.create(
            checkout=checkout, product=None,
            product_name="Ghost Product", product_barcode="000111",
            price=Decimal("4.00"), quantity=1,
        )
        resp = self.client.get(reverse("checkout"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(item.display_name, "Ghost Product")
        self.client.post(reverse("checkout_delete_item", kwargs={"item_id": item.pk}))
        self.assertFalse(CheckoutOrderItem.objects.filter(pk=item.pk).exists())

    def test_partial_unique_prevents_two_drafts(self):
        CheckoutOrder.objects.create(user=self.pu, status="draft")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                CheckoutOrder.objects.create(user=self.pu, status="draft")

    # ── role gating ──
    def test_pu_blocked_from_purchase(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        resp = self.client.get(reverse("create_order"))
        self.assertEqual(resp.status_code, 302)  # AdminRequiredMixin → redirect

    def test_admin_blocked_from_checkout(self):
        self.client.force_login(self.admin, backend="django.contrib.auth.backends.ModelBackend")
        resp = self.client.get(reverse("checkout"))
        self.assertEqual(resp.status_code, 302)  # UserRequiredMixin → redirect
