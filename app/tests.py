from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import AnonymousUser, User
from django.test import RequestFactory, SimpleTestCase, TestCase, Client
from django.urls import reverse
from django.utils.timezone import now

from .models import Product, Category, CheckinSession, StockChange
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
        self.client.login(username="regularuser", password="pass1234")
        url = reverse("checkin_session_reopen", kwargs={"session_id": self.session.pk})
        resp = self.client.post(url)
        self.session.refresh_from_db()
        self.assertIsNotNone(self.session.ended_at)  # still closed

    def test_staff_can_reopen(self):
        self.client.login(username="staffuser", password="pass1234")
        url = reverse("checkin_session_reopen", kwargs={"session_id": self.session.pk})
        resp = self.client.post(url)
        self.session.refresh_from_db()
        self.assertIsNone(self.session.ended_at)
        self.assertTrue(self.session.is_active)
        self.assertIsNotNone(self.session.reopened_at)
        self.assertTrue(self.session.is_reopened)

    def test_reopen_keeps_other_active_sessions(self):
        """Reopening a session does NOT close other active sessions."""
        self.client.login(username="staffuser", password="pass1234")

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
        self.client.login(username="staffuser", password="pass1234")
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
        self.client.login(username="regularuser", password="pass1234")
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
        self.client.login(username="staffuser", password="pass1234")
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
        self.client.login(username="staffuser", password="pass1234")
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
        self.client.login(username="regularuser", password="pass1234")
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
        self.client.login(username="staffuser", password="pass1234")
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
