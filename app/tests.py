from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import AnonymousUser, User
from django.db import IntegrityError, transaction
from django.test import RequestFactory, SimpleTestCase, TestCase, Client, override_settings
from django.urls import reverse
from django.utils.timezone import now

from app import session_limits
from .models import (
    Product, Category, CheckinSession, StockChange,
    CheckoutOrder, CheckoutOrderItem, UserSession, UserAction,
    Order, InventoryCountLine,
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

    # ── purchase page: don't resurrect a deleted order as the current draft ──
    def test_purchase_does_not_resume_single_deleted_order(self):
        self.client.force_login(self.admin, backend="django.contrib.auth.backends.ModelBackend")
        self._register_session(self.admin)
        # First visit creates a draft purchase order and stores it on the session.
        self.client.get(reverse("create_order"))
        order = Order.objects.get(user=self.admin, submitted=False, is_deleted=False)

        # Delete it (soft delete; clears the session's order_id/cart).
        self.client.post(reverse("delete_order", args=[order.order_id]))
        order.refresh_from_db()
        self.assertTrue(order.is_deleted)

        # Returning to the purchase page must start a FRESH draft, not pull the
        # just-deleted one back in.
        self.client.get(reverse("create_order"))
        live = Order.objects.filter(user=self.admin, submitted=False, is_deleted=False)
        self.assertEqual(live.count(), 1)
        self.assertNotEqual(live.first().order_id, order.order_id)
        self.assertEqual(self.client.session.get("order_id"), live.first().order_id)

    def test_purchase_does_not_resume_after_delete_all(self):
        self.client.force_login(self.admin, backend="django.contrib.auth.backends.ModelBackend")
        self._register_session(self.admin)
        self.client.get(reverse("create_order"))
        order = Order.objects.get(user=self.admin, submitted=False, is_deleted=False)

        # Delete-all soft-deletes every visible order (leaves submitted=False).
        self.client.post(reverse("delete_all_orders"))
        order.refresh_from_db()
        self.assertTrue(order.is_deleted)

        # Purchase page must not resume the deleted order.
        self.client.get(reverse("create_order"))
        live = Order.objects.filter(user=self.admin, submitted=False, is_deleted=False)
        self.assertEqual(live.count(), 1)
        self.assertNotEqual(live.first().order_id, order.order_id)


@override_settings(GLOBAL_MAX_SESSIONS=5, SESSION_ACTIVE_WINDOW=300, AXES_ENABLED=False)
class SessionLimitTests(TestCase):
    """Global 'max 5 active computers' cap, dedupe-by-computer, and stale pruning."""

    def setUp(self):
        self.pu = User.objects.create_user(username="pu", password="pass1234", is_staff=False)
        self.admin = User.objects.create_user(username="gina", password="pass1234", is_staff=True)
        self.client = Client()

    def _make_session(self, user, ip, key, age_seconds=0):
        """Create a UserSession row; backdate last_activity via .update() to dodge auto_now."""
        us = UserSession.objects.create(user=user, session_key=key, ip_address=ip)
        if age_seconds:
            UserSession.objects.filter(pk=us.pk).update(
                last_activity=now() - timedelta(seconds=age_seconds)
            )
        return us

    # ── helper-level ──
    def test_prune_removes_stale_keeps_fresh(self):
        self._make_session(self.pu, "192.168.0.10", "fresh1")
        self._make_session(self.pu, "192.168.0.11", "stale1", age_seconds=400)
        self.assertEqual(session_limits.prune_stale(), 1)
        self.assertEqual(UserSession.objects.count(), 1)
        self.assertTrue(UserSession.objects.filter(session_key="fresh1").exists())

    def test_active_count_is_windowed(self):
        self._make_session(self.pu, "192.168.0.10", "fresh1")
        self._make_session(self.pu, "192.168.0.11", "stale1", age_seconds=400)
        self.assertEqual(session_limits.active_count(), 1)

    def test_drop_computer_dedupes_same_user_and_ip(self):
        self._make_session(self.pu, "192.168.0.10", "a")
        self._make_session(self.pu, "192.168.0.10", "b")   # same computer, 2nd row
        self._make_session(self.pu, "192.168.0.99", "c")   # different computer
        self.assertEqual(session_limits.drop_computer(self.pu, "192.168.0.10"), 2)
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 1)
        self.assertTrue(UserSession.objects.filter(session_key="c").exists())

    # ── login flow ──
    def test_regular_login_blocked_at_cap(self):
        for i in range(5):  # 5 PU computers already active
            self._make_session(self.pu, f"192.168.0.{i + 1}", f"cap{i}")
        resp = self.client.post(
            reverse("login"), {"username": "pu", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.50",
        )
        self.assertEqual(resp.status_code, 200)              # re-renders login, no redirect
        self.assertEqual(session_limits.active_count(), 5)   # still 5, the 6th was refused
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 5)
        self.assertFalse(
            UserSession.objects.filter(user=self.pu, ip_address="192.168.0.50").exists()
        )

    def test_regular_login_under_cap_creates_one_slot(self):
        self._make_session(self.pu, "192.168.0.1", "cap0")   # only 1 active
        resp = self.client.post(
            reverse("login"), {"username": "pu", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.50",
        )
        self.assertEqual(resp.status_code, 302)              # logged in
        self.assertEqual(session_limits.active_count(), 2)

    def test_login_replaces_same_computer_row(self):
        # An old session for PU from the same computer (IP) it is logging in from.
        self._make_session(self.pu, "192.168.0.50", "oldkey", age_seconds=20)
        resp = self.client.post(
            reverse("login"), {"username": "pu", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.50",
        )
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 1)  # not 2
        self.assertFalse(UserSession.objects.filter(session_key="oldkey").exists())

    def test_admin_not_blocked_at_cap_and_is_singleton(self):
        for i in range(5):  # cap full of PU computers
            self._make_session(self.pu, f"192.168.0.{i + 1}", f"cap{i}")
        self._make_session(self.admin, "192.168.0.9", "adminold", age_seconds=10)
        resp = self.client.post(
            reverse("login"), {"username": "gina", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.60",
        )
        self.assertEqual(resp.status_code, 302)                       # admin gets in
        self.assertEqual(UserSession.objects.filter(user=self.admin).count(), 1)  # singleton
        self.assertFalse(UserSession.objects.filter(session_key="adminold").exists())
        self.assertLessEqual(session_limits.active_count(), 5)        # cap respected


@override_settings(AXES_ENABLED=False, GLOBAL_MAX_SESSIONS=5, SESSION_ACTIVE_WINDOW=300)
class InventoryCountModeTests(TestCase):
    """Inventory Count Mode: buffer the count, reconcile (apply) at the end."""

    def setUp(self):
        self.user = User.objects.create_user(username="counter", password="pass1234", is_staff=True)
        self.cat = Category.objects.create(name="Aisle 1")
        self.cat2 = Category.objects.create(name="Aisle 2")
        self.p1 = Product.objects.create(name="P1", price=Decimal("1.00"), quantity_in_stock=10, category=self.cat, barcode="111")
        self.p2 = Product.objects.create(name="P2", price=Decimal("1.00"), quantity_in_stock=5, category=self.cat, barcode="222")
        self.other = Product.objects.create(name="Other", price=Decimal("1.00"), quantity_in_stock=7, category=self.cat2, barcode="999")
        self.client = Client()
        self.client.force_login(self.user, backend="django.contrib.auth.backends.ModelBackend")
        UserSession.objects.get_or_create(user=self.user, session_key=self.client.session.session_key)

    def _start_inventory(self, ids):
        return self.client.post(reverse("checkin_start"), {
            "scanned_by": "Me", "note": "", "inventory_mode": "on",
            "count_product_ids": ",".join(str(i) for i in ids),
        })

    def _latest_session(self):
        return CheckinSession.objects.latest("started_at")

    # (a) start creates scope lines, snapshots expected, no stock change
    def test_start_creates_lines_no_stock_change(self):
        resp = self._start_inventory([self.p1.product_id, self.p2.product_id])
        self.assertEqual(resp.status_code, 302)
        session = self._latest_session()
        self.assertTrue(session.inventory_mode)
        lines = InventoryCountLine.objects.filter(session=session)
        self.assertEqual(lines.count(), 2)
        l1 = lines.get(product=self.p1)
        self.assertEqual(l1.expected_qty, 10)
        self.assertEqual(l1.counted_qty, 0)
        self.p1.refresh_from_db()
        self.assertEqual(self.p1.quantity_in_stock, 10)  # untouched
        self.assertEqual(StockChange.objects.filter(session=session).count(), 0)

    # (b) ＋ / − / scan-again adjust counted_qty only, no stock, no checkin ledger row
    def test_plus_minus_adjust_count_not_stock(self):
        self._start_inventory([self.p1.product_id])
        session = self._latest_session()
        self.client.post(reverse("add_quantity", kwargs={"session_id": session.pk, "product_id": self.p1.product_id}), {"amount": 3})
        self.client.post(reverse("delete_one", kwargs={"session_id": session.pk, "product_id": self.p1.product_id}))
        line = InventoryCountLine.objects.get(session=session, product=self.p1)
        self.assertEqual(line.counted_qty, 2)  # +3 then -1
        self.p1.refresh_from_db()
        self.assertEqual(self.p1.quantity_in_stock, 10)
        self.assertEqual(StockChange.objects.filter(session=session, change_type="checkin").count(), 0)

    def test_scan_again_tallies_count(self):
        self._start_inventory([self.p1.product_id])
        session = self._latest_session()
        url = reverse("checkin_session", kwargs={"session_id": session.pk})
        self.client.post(url, {"barcode": "111", "current_barcode": "111"})
        line = InventoryCountLine.objects.get(session=session, product=self.p1)
        self.assertEqual(line.counted_qty, 1)
        self.p1.refresh_from_db()
        self.assertEqual(self.p1.quantity_in_stock, 10)

    # (c) out-of-scope scan auto-creates a line
    def test_out_of_scope_scan_autocreates_line(self):
        self._start_inventory([self.p1.product_id])
        session = self._latest_session()
        url = reverse("checkin_session", kwargs={"session_id": session.pk})
        self.client.post(url, {"barcode": "999", "current_barcode": "999"})
        line = InventoryCountLine.objects.filter(session=session, product=self.other).first()
        self.assertIsNotNone(line)
        self.assertEqual(line.counted_qty, 1)
        self.assertEqual(line.expected_qty, 7)

    # (d) reconcile apply: count is source of truth; unscanned in-scope -> 0; variance recorded; session ends
    def test_reconcile_apply_sets_stock_and_variance(self):
        self._start_inventory([self.p1.product_id, self.p2.product_id])
        session = self._latest_session()
        InventoryCountLine.objects.filter(session=session, product=self.p1).update(counted_qty=8)
        # p2 left unscanned (counted 0)
        resp = self.client.post(reverse("checkin_reconcile", kwargs={"session_id": session.pk}))
        self.assertEqual(resp.status_code, 302)
        self.p1.refresh_from_db()
        self.p2.refresh_from_db()
        self.assertEqual(self.p1.quantity_in_stock, 8)   # 10 -> 8
        self.assertEqual(self.p2.quantity_in_stock, 0)   # unscanned -> 0
        session.refresh_from_db()
        self.assertFalse(session.is_active)
        self.assertTrue(StockChange.objects.filter(session=session, product=self.p1, change_type="error_subtract", quantity=2).exists())
        self.assertTrue(StockChange.objects.filter(session=session, product=self.p2, change_type="error_subtract", quantity=5).exists())

    # (e) non-inventory sessions still mutate live stock
    def test_non_inventory_add_still_changes_stock(self):
        self.client.post(reverse("checkin_start"), {"scanned_by": "Me", "note": ""})
        session = self._latest_session()
        self.assertFalse(session.inventory_mode)
        self.client.post(reverse("add_quantity", kwargs={"session_id": session.pk, "product_id": self.p1.product_id}), {"amount": 2})
        self.p1.refresh_from_db()
        self.assertEqual(self.p1.quantity_in_stock, 12)
        self.assertEqual(InventoryCountLine.objects.filter(session=session).count(), 0)

    # (f) deleting an in-progress inventory count discards the buffer, leaves stock intact
    def test_delete_active_inventory_session_discards_count(self):
        self._start_inventory([self.p1.product_id, self.p2.product_id])
        session = self._latest_session()
        InventoryCountLine.objects.filter(session=session, product=self.p1).update(counted_qty=3)
        self.assertTrue(session.is_active)
        resp = self.client.post(reverse("checkin_session_delete", kwargs={"session_id": session.pk}))
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(CheckinSession.objects.filter(pk=session.pk).exists())
        # count lines cascade-deleted; live stock untouched
        self.assertEqual(InventoryCountLine.objects.filter(session_id=session.pk).count(), 0)
        self.p1.refresh_from_db(); self.p2.refresh_from_db()
        self.assertEqual(self.p1.quantity_in_stock, 10)
        self.assertEqual(self.p2.quantity_in_stock, 5)
