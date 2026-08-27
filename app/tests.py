from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import AnonymousUser, User
from django.test import RequestFactory, SimpleTestCase, TestCase, Client, override_settings
from django.urls import reverse
from django.utils.timezone import now

from app import session_limits
from .models import (
    Product, ProductLot, Category, CheckinSession, StockChange,
    CheckoutOrder, CheckoutOrderItem, LoginAudit, UserSession, UserAction, PagePresence,
    Order, OrderDetail, InventoryCountLine,
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
            order_date=now(),
            submitted=True,
            seniors_discount=False,
            subtotal=Decimal("25.98"),
            discount_amount=Decimal("0.00"),
            tax=Decimal("0.00"),
            tax_rate=Decimal("0.1300"),
            total_price=Decimal("25.98"),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        self.fake_order.details = FakeDetailCollection(
            [
                SimpleNamespace(
                    product=None,
                    quantity=2,
                    price=Decimal("12.99"),
                    taxable_at_sale=False,
                    cost_per_unit_at_sale=None,
                    expiry_at_sale=None,
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
        self.assertIn(f'MPCP-Order-{self.order_id}.pdf', response["Content-Disposition"])
        self.assertTrue(response.content.startswith(b"%PDF"))

    def test_order_templates_reference_pdf_route(self):
        order_view_template = (self.template_dir / "order_view.html").read_text(encoding="utf-8")
        order_detail_template = (self.template_dir / "order_detail.html").read_text(encoding="utf-8")

        self.assertIn('href="{{ row.pdf_url }}"', order_view_template)
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
        # Original ledger row is retained, but removed from the editable session.
        self.change_add.refresh_from_db()
        self.assertIsNone(self.change_add.session_id)
        self.assertIn("Removed from Session", self.change_add.note)
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
        self.change_sub.refresh_from_db()
        self.assertIsNone(self.change_sub.session_id)
        self.assertIn("Removed from Session", self.change_sub.note)

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

        # The subtract-line remains in the audit ledger, detached from the session.
        self.change_sub.refresh_from_db()
        self.assertIsNone(self.change_sub.session_id)

        # Audit trail: 2 corrective entries created (1 adjust + 1 remove)
        corrections = StockChange.objects.filter(
            session=self.session,
            change_type__in=["error_add", "error_subtract"],
        ).exclude(pk=self.change_add.pk)
        self.assertEqual(corrections.count(), 2)

    def test_inline_edit_uses_lot_total_instead_of_hidden_stock_value(self):
        """Inline edits derive stock from lots and ignore stale summary stock."""
        self.session.ended_at = None
        self.session.save(update_fields=["ended_at"])
        self.client.force_login(self.staff, backend="django.contrib.auth.backends.ModelBackend")
        stock_before = self.product.quantity_in_stock
        changes_before = StockChange.objects.count()
        ProductLot.objects.create(
            product=self.product, lot_number="LOT-A", quantity_on_hand=12,
        )
        ProductLot.objects.create(
            product=self.product, lot_number="LOT-B", quantity_on_hand=8,
        )

        response = self.client.post(
            reverse("checkin_edit_product", kwargs={
                "session_id": self.session.pk,
                "product_id": self.product.product_id,
            }),
            {
                "name": self.product.name,
                "brand": "",
                "item_number": "UPDATED-ITEM-42",
                "price": "10.49",
                "barcode": "",
                "quantity_in_stock": "1",  # stale browser value
                "category": self.category.pk,
                "unit_size": "",
                "description": "Updated while adjusting stock",
                "expiry_date": "",
                "taxable": "on",
                "status": "on",
                "price_per_unit": "",
                "lot_number": ["LOT-A", "LOT-B"],
                "lot_expiry": ["", ""],
                "lot_quantity": ["13", "9"],
            },
        )

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, stock_before + 2)
        self.assertEqual(self.product.price, Decimal("10.49"))
        self.assertEqual(self.product.item_number, "UPDATED-ITEM-42")
        self.assertEqual(StockChange.objects.count(), changes_before + 1)
        change = StockChange.objects.latest("pk")
        self.assertEqual(change.change_type, "error_add")
        self.assertEqual(change.quantity, 2)
        self.assertEqual(change.session, self.session)


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

    def _set_current_checkout(self, checkout):
        session = self.client.session
        session['checkout_id'] = checkout.pk
        session.save()

    def _start_checkout(self, user=None):
        user = user or self.pu
        self.client.force_login(user, backend="django.contrib.auth.backends.ModelBackend")
        self.client.post(reverse("checkout_new"))
        return CheckoutOrder.objects.get(pk=self.client.session['checkout_id'])

    # ── creation / resume ──
    def test_chooser_is_lazy_and_start_new_creates_one_draft(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        resp = self.client.get(reverse("checkout"))
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(CheckoutOrder.objects.filter(user=self.pu, status="draft").exists())

        self.client.post(reverse("checkout_new"))
        self.assertEqual(CheckoutOrder.objects.filter(user=self.pu, status="draft").count(), 1)
        # Reloading the chooser does not create another draft.
        self.client.get(reverse("checkout"))
        self.assertEqual(CheckoutOrder.objects.filter(user=self.pu, status="draft").count(), 1)

    def test_checkout_dashboard_resumes_the_exact_selected_purchase_cart(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        remembered = Order.objects.create(user=self.pu, draft_cart={})
        selected = Order.objects.create(
            user=self.pu,
            draft_cart={
                str(self.product.pk): {
                    "quantity": 2,
                    "price": str(self.product.price),
                    "name": self.product.name,
                },
            },
            draft_expires_at=now() - timedelta(hours=1),
        )
        session = self.client.session
        session['order_id'] = remembered.pk
        session.save()

        chooser = self.client.get(reverse("checkout"))
        continue_url = reverse("purchase_continue", args=[selected.pk])
        self.assertContains(chooser, "Active sessions")
        self.assertNotContains(chooser, "Open purchase cart")
        self.assertContains(chooser, continue_url)

        response = self.client.post(continue_url)
        self.assertRedirects(response, reverse("create_order"), fetch_redirect_response=False)
        self.assertEqual(self.client.session['order_id'], selected.pk)
        selected.refresh_from_db()
        self.assertGreater(selected.draft_expires_at, now())
        self.assertEqual(selected.timer_reset_count, 1)

    def test_checkout_dashboard_separates_session_types_with_labels_and_colors(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        checkout = CheckoutOrder.objects.create(user=self.pu, status="draft")
        purchase = Order.objects.create(
            user=self.pu,
            draft_cart={str(self.product.pk): {"quantity": 1}},
        )

        response = self.client.get(reverse("checkout"))

        self.assertEqual(response.context["active_session_count"], 2)
        self.assertContains(response, "Active sessions", count=1)
        self.assertContains(response, "Session type legend")
        self.assertContains(response, "no sale (no charge)")
        self.assertContains(response, "recorded sale")
        self.assertContains(
            response,
            f'Checkout #{checkout.pk}',
        )
        self.assertContains(
            response,
            f'Purchase #{purchase.pk}',
        )
        self.assertContains(response, 'data-session-type="checkout"')
        self.assertContains(response, 'data-session-type="purchase"')
        self.assertContains(response, ".cc-active-list .cc-row-checkout")
        self.assertContains(response, ".cc-active-list .cc-row-purchase")

    def test_purchase_only_dashboard_is_not_reported_empty(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        purchase = Order.objects.create(
            user=self.pu,
            draft_cart={str(self.product.pk): {"quantity": 1}},
        )

        response = self.client.get(reverse("checkout"))

        self.assertEqual(response.context["active_session_count"], 1)
        self.assertContains(response, f'Purchase #{purchase.pk}')
        self.assertNotContains(
            response,
            "No active sessions. Start a new checkout or purchase below.",
        )

    def test_empty_dashboard_has_one_combined_active_session_message(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")

        response = self.client.get(reverse("checkout"))

        self.assertEqual(response.context["active_session_count"], 0)
        self.assertContains(
            response,
            "No active sessions. Start a new checkout or purchase below.",
            count=1,
        )

    def test_admin_can_resume_and_complete_another_users_purchase_cart(self):
        selected = Order.objects.create(
            user=self.pu,
            draft_cart={
                str(self.product.pk): {
                    "quantity": 1,
                    "price": str(self.product.price),
                    "name": self.product.name,
                },
            },
        )
        self.client.force_login(self.admin, backend="django.contrib.auth.backends.ModelBackend")

        dashboard = self.client.get(reverse("checkout"))
        continue_url = reverse("purchase_continue", args=[selected.pk])
        self.assertContains(
            dashboard,
            f'aria-label="Resume purchase cart {selected.pk}"',
        )

        response = self.client.post(continue_url)

        self.assertRedirects(response, reverse("create_order"), fetch_redirect_response=False)
        self.assertEqual(self.client.session.get('order_id'), selected.pk)

        cart = self.client.get(reverse("create_order"))
        self.assertEqual(cart.context["order"].pk, selected.pk)
        self.assertContains(cart, self.product.name)
        selected.refresh_from_db()
        self.assertEqual(selected.timer_reset_count, 1)
        self.client.post(
            reverse("add_product_by_id", args=[self.product2.pk]),
            {"quantity": 2},
        )
        selected.refresh_from_db()
        self.assertEqual(selected.draft_cart[str(self.product2.pk)]["quantity"], 2)
        self.assertEqual(selected.user, self.pu)

        submitted = self.client.post(reverse("submit_order"))
        self.assertEqual(submitted.status_code, 302)
        selected.refresh_from_db()
        self.assertTrue(selected.submitted)
        self.assertEqual(selected.user, self.pu)
        self.assertTrue(
            UserAction.objects.filter(
                user=self.admin,
                action="submit_order",
                target=f"Order #{selected.pk}",
            ).exists()
        )

    def test_non_admin_cannot_resume_another_users_purchase_cart(self):
        other = User.objects.create_user(username="other-pu", password="pass1234")
        selected = Order.objects.create(
            user=self.pu,
            draft_cart={str(self.product.pk): {"quantity": 1}},
            draft_expires_at=now() - timedelta(hours=1),
        )
        self.client.force_login(other, backend="django.contrib.auth.backends.ModelBackend")

        dashboard = self.client.get(reverse("checkout"))
        self.assertContains(dashboard, "Owner only")
        self.assertNotContains(
            dashboard,
            f'aria-label="Resume purchase cart {selected.pk}"',
        )

        response = self.client.post(reverse("purchase_continue", args=[selected.pk]))

        self.assertRedirects(response, reverse("checkout"), fetch_redirect_response=False)
        self.assertIsNone(self.client.session.get('order_id'))
        selected.refresh_from_db()
        self.assertEqual(selected.timer_reset_count, 0)
        self.assertLess(selected.draft_expires_at, now())

    def test_admin_purchase_cart_resume_waits_for_other_computer(self):
        self.client.force_login(self.admin, backend="django.contrib.auth.backends.ModelBackend")
        selected = Order.objects.create(
            user=self.pu,
            draft_cart={str(self.product.pk): {"quantity": 1}},
            draft_expires_at=now() - timedelta(hours=1),
        )
        other_browser = Client()
        other_browser.force_login(
            self.pu, backend="django.contrib.auth.backends.ModelBackend",
        )
        other_session = other_browser.session
        other_session["order_id"] = selected.pk
        other_session.save()
        PagePresence.objects.create(
            page=reverse("create_order"),
            session_key=other_session.session_key,
            user=self.pu,
            ip_address="192.0.2.25",
            user_agent="Chrome on Windows",
        )

        dashboard = self.client.get(reverse("checkout"))
        self.assertContains(dashboard, "In use")
        self.assertContains(dashboard, "Purchase is open on another computer")
        self.assertNotContains(
            dashboard,
            f'aria-label="Resume purchase cart {selected.pk}"',
        )

        response = self.client.post(reverse("purchase_continue", args=[selected.pk]))

        self.assertRedirects(response, reverse("checkout"), fetch_redirect_response=False)
        self.assertIsNone(self.client.session.get('order_id'))
        selected.refresh_from_db()
        self.assertEqual(selected.timer_reset_count, 0)
        self.assertLess(selected.draft_expires_at, now())

    def test_add_by_barcode_increments_single_line(self):
        self._start_checkout()
        self.client.post(reverse("checkout_cart"), {"barcode": "12345", "quantity": 1})
        self.client.post(reverse("checkout_cart"), {"barcode": "12345", "quantity": 1})
        checkout = CheckoutOrder.objects.get(user=self.pu, status="draft")
        items = checkout.items.all()
        self.assertEqual(items.count(), 1)
        self.assertEqual(items.first().quantity, 2)

    def test_checkout_shows_newest_scanned_line_first_and_rescan_moves_it(self):
        self._start_checkout()
        self.client.post(
            reverse("checkout_cart"), {"barcode": self.product.barcode, "quantity": 1},
        )
        self.client.post(
            reverse("checkout_cart"), {"barcode": self.product2.barcode, "quantity": 1},
        )

        response = self.client.get(reverse("checkout_cart"))
        self.assertEqual(
            [row["item"].product_id for row in response.context["order_items"]],
            [self.product2.pk, self.product.pk],
        )

        self.client.post(
            reverse("checkout_cart"), {"barcode": self.product.barcode, "quantity": 1},
        )
        response = self.client.get(reverse("checkout_cart"))
        self.assertEqual(
            [row["item"].product_id for row in response.context["order_items"]],
            [self.product.pk, self.product2.pk],
        )

    def test_purchase_shows_newest_scanned_line_first_and_rescan_moves_it(self):
        self.client.force_login(
            self.pu, backend="django.contrib.auth.backends.ModelBackend",
        )
        self.client.post(
            reverse("create_order"), {"barcode": self.product.barcode, "quantity": 1},
        )
        self.client.post(
            reverse("create_order"), {"barcode": self.product2.barcode, "quantity": 1},
        )

        response = self.client.get(reverse("create_order"))
        self.assertEqual(
            [row["product"].pk for row in response.context["order_items"]],
            [self.product2.pk, self.product.pk],
        )

        self.client.post(
            reverse("create_order"), {"barcode": self.product.barcode, "quantity": 1},
        )
        response = self.client.get(reverse("create_order"))
        self.assertEqual(
            [row["product"].pk for row in response.context["order_items"]],
            [self.product.pk, self.product2.pk],
        )

    def test_pu_can_decrement_then_remove_purchase_cart_item(self):
        self.client.force_login(
            self.pu, backend="django.contrib.auth.backends.ModelBackend",
        )
        self.client.post(
            reverse("create_order"),
            {"barcode": self.product.barcode, "quantity": 2},
        )
        order = Order.objects.get(
            user=self.pu, submitted=False, is_deleted=False,
        )
        url = reverse("delete_order_item", args=[self.product.pk])

        response = self.client.post(url)
        self.assertRedirects(response, reverse("create_order"), fetch_redirect_response=False)
        order.refresh_from_db()
        self.assertEqual(order.draft_cart[str(self.product.pk)]["quantity"], 1)

        response = self.client.post(url)
        self.assertRedirects(response, reverse("create_order"), fetch_redirect_response=False)
        self.assertFalse(
            Order.objects.filter(pk=order.pk, submitted=False, is_deleted=False).exists()
        )
        self.assertNotIn('order_id', self.client.session)

    def test_pu_cannot_remove_item_from_another_users_purchase_cart(self):
        other = User.objects.create_user(
            username="purchase-owner", password="pass1234", is_staff=False,
        )
        order = Order.objects.create(
            user=other,
            draft_cart={
                str(self.product.pk): {
                    "quantity": 2,
                    "price": str(self.product.price),
                    "name": self.product.name,
                },
            },
        )
        self.client.force_login(
            self.pu, backend="django.contrib.auth.backends.ModelBackend",
        )
        session = self.client.session
        session['order_id'] = order.pk
        session.save()

        response = self.client.post(
            reverse("delete_order_item", args=[self.product.pk]),
        )

        self.assertRedirects(response, reverse("create_order"), fetch_redirect_response=False)
        order.refresh_from_db()
        self.assertEqual(order.draft_cart[str(self.product.pk)]["quantity"], 2)

    def test_purchase_card_pairs_active_lots_with_their_expiry_dates(self):
        soon = now().date() + timedelta(days=5)
        later = now().date() + timedelta(days=75)
        ProductLot.objects.create(
            product=self.product,
            lot_number="LOT-SOON",
            expiry_date=soon,
            quantity_on_hand=2,
        )
        ProductLot.objects.create(
            product=self.product,
            lot_number="LOT-LATER",
            expiry_date=later,
            quantity_on_hand=3,
        )
        ProductLot.objects.create(
            product=self.product,
            lot_number="LOT-ZERO",
            expiry_date=soon,
            quantity_on_hand=0,
        )
        ProductLot.objects.create(
            product=self.product,
            lot_number="LOT-ARCHIVED",
            expiry_date=soon,
            quantity_on_hand=4,
            archived_at=now(),
        )
        self.client.force_login(
            self.pu, backend="django.contrib.auth.backends.ModelBackend",
        )
        self.client.post(
            reverse("create_order"),
            {"barcode": self.product.barcode, "quantity": 1},
        )

        response = self.client.get(reverse("create_order"))

        rows = response.context["order_items"][0]["expiry_lot_rows"]
        self.assertEqual(
            [(row["lot_number"], row["date"]) for row in rows],
            [("LOT-SOON", soon), ("LOT-LATER", later)],
        )
        self.assertContains(response, "Expiries &amp; lots")
        self.assertContains(response, "LOT-SOON")
        self.assertContains(response, soon.strftime("%b %d, %Y"))
        self.assertNotContains(response, "LOT-ZERO")
        self.assertNotContains(response, "LOT-ARCHIVED")
        self.assertNotContains(response, "stock-badge-big")

    def test_delete_item_decrements_then_removes(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        checkout = CheckoutOrder.objects.create(user=self.pu, status="draft")
        self._set_current_checkout(checkout)
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
        self._set_current_checkout(checkout)
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
            StockChange.objects.filter(product=self.product, change_type="giveaway").count(), 1
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
        self._set_current_checkout(CheckoutOrder.objects.get(user=self.pu, status="draft"))
        resp = self.client.post(reverse("checkout_submit"))
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(CheckoutOrder.objects.filter(user=self.pu, status="submitted").exists())
        self.assertEqual(StockChange.objects.count(), 0)

    def test_checkout_new_preserves_old_draft_and_starts_separate_draft(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        skey = self._register_session(self.pu)
        checkout = CheckoutOrder.objects.create(
            user=self.pu, status="draft", active_session_key=skey,
        )
        CheckoutOrderItem.objects.create(
            checkout=checkout, product=self.product,
            product_name=self.product.name, price=self.product.price, quantity=2,
        )
        self._set_current_checkout(checkout)
        resp = self.client.post(reverse("checkout_new"))
        self.assertEqual(resp.status_code, 302)
        checkout.refresh_from_db()
        self.assertEqual(checkout.items.count(), 1)
        self.assertEqual(CheckoutOrder.objects.filter(user=self.pu, status="draft").count(), 2)
        self.assertNotEqual(self.client.session['checkout_id'], checkout.pk)
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
        self._set_current_checkout(checkout)
        resp = self.client.get(reverse("checkout_cart"))
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
        self._set_current_checkout(checkout)
        resp = self.client.get(reverse("checkout_cart"))
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
        self._set_current_checkout(checkout)
        resp = self.client.get(reverse("checkout_cart"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(item.display_name, "Ghost Product")
        self.client.post(reverse("checkout_delete_item", kwargs={"item_id": item.pk}))
        self.assertFalse(CheckoutOrderItem.objects.filter(pk=item.pk).exists())

    def test_multiple_drafts_per_user_are_allowed(self):
        CheckoutOrder.objects.create(user=self.pu, status="draft")
        CheckoutOrder.objects.create(user=self.pu, status="draft")
        self.assertEqual(CheckoutOrder.objects.filter(user=self.pu, status="draft").count(), 2)

    # ── role gating ──
    def test_pu_can_use_purchase(self):
        self.client.force_login(self.pu, backend="django.contrib.auth.backends.ModelBackend")
        resp = self.client.get(reverse("create_order"))
        self.assertEqual(resp.status_code, 200)

    def test_admin_can_use_shared_checkout(self):
        self.client.force_login(self.admin, backend="django.contrib.auth.backends.ModelBackend")
        resp = self.client.get(reverse("checkout"))
        self.assertEqual(resp.status_code, 200)

    # ── purchase page: don't resurrect a deleted order as the current draft ──
    def test_purchase_does_not_resume_single_deleted_order(self):
        self.client.force_login(self.admin, backend="django.contrib.auth.backends.ModelBackend")
        self._register_session(self.admin)
        # A draft is created only when the first product is added.
        self.client.post(reverse("add_product_by_id", args=[self.product.pk]), {'quantity': '1'})
        order = Order.objects.get(user=self.admin, submitted=False, is_deleted=False)

        # Delete it (soft delete; clears the session's order_id/cart).
        self.client.post(reverse("delete_order", args=[order.order_id]))
        order.refresh_from_db()
        self.assertTrue(order.is_deleted)

        # A read-only visit stays lazy and never resurrects the deleted draft.
        self.client.get(reverse("create_order"))
        self.assertFalse(Order.objects.filter(user=self.admin, submitted=False, is_deleted=False).exists())

        # The next add creates a fresh draft.
        self.client.post(reverse("add_product_by_id", args=[self.product.pk]), {'quantity': '1'})
        live = Order.objects.filter(user=self.admin, submitted=False, is_deleted=False)
        self.assertEqual(live.count(), 1)
        self.assertNotEqual(live.first().order_id, order.order_id)
        self.assertEqual(self.client.session.get("order_id"), live.first().order_id)

    def test_purchase_does_not_resume_after_delete_all(self):
        self.client.force_login(self.admin, backend="django.contrib.auth.backends.ModelBackend")
        self._register_session(self.admin)
        self.client.post(reverse("add_product_by_id", args=[self.product.pk]), {'quantity': '1'})
        order = Order.objects.get(user=self.admin, submitted=False, is_deleted=False)

        # Delete-all soft-deletes every visible order (leaves submitted=False).
        self.client.post(reverse("delete_all_orders"))
        order.refresh_from_db()
        self.assertTrue(order.is_deleted)

        # Purchase page must not resume the deleted order and stays lazy.
        self.client.get(reverse("create_order"))
        self.assertFalse(Order.objects.filter(user=self.admin, submitted=False, is_deleted=False).exists())

        self.client.post(reverse("add_product_by_id", args=[self.product.pk]), {'quantity': '1'})
        live = Order.objects.filter(user=self.admin, submitted=False, is_deleted=False)
        self.assertEqual(live.count(), 1)
        self.assertNotEqual(live.first().order_id, order.order_id)


@override_settings(MAX_PU_SESSIONS=6, SESSION_ACTIVE_WINDOW=300, AXES_ENABLED=False)
class SessionLimitTests(TestCase):
    """Six shared-PU identities, admin separation, dedupe, and stale pruning."""

    def setUp(self):
        self.pu = User.objects.create_user(username="pu", password="pass1234", is_staff=False)
        self.admin = User.objects.create_user(username="gina", password="pass1234", is_staff=True)
        self.client = Client()

    def _make_session(self, user, ip, key, age_seconds=0, pu_slot=None):
        """Create a UserSession row; backdate last_activity via .update() to dodge auto_now."""
        if not user.is_staff and pu_slot is None:
            used = set(
                UserSession.objects.filter(pu_slot__isnull=False)
                .values_list('pu_slot', flat=True)
            )
            pu_slot = next(slot for slot in range(1, 7) if slot not in used)
        us = UserSession.objects.create(
            user=user,
            session_key=key,
            ip_address=ip,
            pu_slot=pu_slot,
        )
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
        self.assertEqual(session_limits.active_pu_count(), 1)

    def test_drop_computer_dedupes_same_user_and_ip(self):
        self._make_session(self.pu, "192.168.0.10", "a")
        self._make_session(self.pu, "192.168.0.10", "b")   # same computer, 2nd row
        self._make_session(self.pu, "192.168.0.99", "c")   # different computer
        self.assertEqual(session_limits.drop_computer(self.pu, "192.168.0.10"), 2)
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 1)
        self.assertTrue(UserSession.objects.filter(session_key="c").exists())

    # ── login flow ──
    def test_regular_login_blocked_at_cap(self):
        for i in range(6):  # all six PU identities already active
            self._make_session(self.pu, f"192.168.0.{i + 1}", f"cap{i}")
        resp = self.client.post(
            reverse("login"), {"username": "pu", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.50",
        )
        self.assertEqual(resp.status_code, 200)              # re-renders login, no redirect
        self.assertEqual(session_limits.active_pu_count(), 6)  # seventh was refused
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 6)
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
        self.assertEqual(session_limits.active_pu_count(), 2)
        created = UserSession.objects.get(ip_address="192.168.0.50")
        self.assertEqual(created.pu_slot, 2)
        self.assertEqual(created.identity_label, "PU2")

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
        self.assertEqual(UserSession.objects.get(user=self.pu).pu_slot, 1)

    def test_admin_not_blocked_at_cap_and_is_singleton(self):
        for i in range(6):  # PU pool is full
            self._make_session(self.pu, f"192.168.0.{i + 1}", f"cap{i}")
        self._make_session(self.admin, "192.168.0.9", "adminold", age_seconds=10)
        resp = self.client.post(
            reverse("login"), {"username": "gina", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.60",
        )
        self.assertEqual(resp.status_code, 302)                       # admin gets in
        self.assertEqual(UserSession.objects.filter(user=self.admin).count(), 1)  # singleton
        self.assertFalse(UserSession.objects.filter(session_key="adminold").exists())
        self.assertEqual(session_limits.active_pu_count(), 6)         # no PU was evicted
        self.assertEqual(session_limits.active_count(), 7)            # six PU + one admin
        self.assertIsNone(UserSession.objects.get(user=self.admin).pu_slot)
        payload = self.client.get(
            reverse('active_sessions'), {'format': 'json'},
        ).json()
        self.assertEqual(payload['active_slots'], 6)
        self.assertEqual(payload['max_slots'], 6)
        self.assertEqual(
            sorted(row['username'] for row in payload['rows'] if row['role'] == 'Regular'),
            ['PU1', 'PU2', 'PU3', 'PU4', 'PU5', 'PU6'],
        )

    def test_six_shared_pu_logins_receive_distinct_backend_identities(self):
        clients = []
        for i in range(6):
            client = Client()
            response = client.post(
                reverse("login"),
                {"username": "pu", "password": "pass1234"},
                REMOTE_ADDR=f"192.168.0.{i + 1}",
            )
            self.assertEqual(response.status_code, 302)
            clients.append(client)

        sessions = UserSession.objects.filter(user=self.pu).order_by('pu_slot')
        self.assertEqual(list(sessions.values_list('pu_slot', flat=True)), [1, 2, 3, 4, 5, 6])
        self.assertEqual([session.identity_label for session in sessions], [
            'PU1', 'PU2', 'PU3', 'PU4', 'PU5', 'PU6',
        ])
        self.assertEqual(
            sorted(
                LoginAudit.objects.filter(user=self.pu, success=True)
                .values_list('username', flat=True)
            ),
            ['PU1', 'PU2', 'PU3', 'PU4', 'PU5', 'PU6'],
        )
        dashboard = clients[0].get(reverse('dashboard'))
        self.assertEqual(dashboard.status_code, 200)
        self.assertEqual(dashboard.context['ui_access']['identity_label'], 'PU1')
        self.assertContains(dashboard, 'PU1 · Admin locked')

    def test_stale_pu_identity_is_reclaimed(self):
        self._make_session(
            self.pu, "192.168.0.10", "stale1", age_seconds=400, pu_slot=1,
        )
        response = self.client.post(
            reverse("login"),
            {"username": "pu", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.50",
        )
        self.assertEqual(response.status_code, 302)
        replacement = UserSession.objects.get(user=self.pu)
        self.assertEqual(replacement.pu_slot, 1)
        self.assertEqual(self.client.session.get('pu_slot'), 1)


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=6, SESSION_ACTIVE_WINDOW=300)
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

    def test_active_page_history_only_contains_current_session(self):
        self.client.post(reverse("checkin_start"), {"scanned_by": "Me", "note": ""})
        session = self._latest_session()
        self.client.post(
            reverse("add_quantity", kwargs={
                "session_id": session.pk,
                "product_id": self.p1.product_id,
            }),
            {"amount": 2},
        )
        other_session = CheckinSession.objects.create(user=self.user, scanned_by="Other")
        StockChange.objects.create(
            product=self.other, session=other_session, user=self.user,
            change_type="checkin", quantity=9,
        )

        response = self.client.get(reverse("checkin_session", kwargs={"session_id": session.pk}))

        self.assertEqual(response.status_code, 200)
        history = response.context["session_history"]
        self.assertEqual([change.session_id for change in history], [session.pk])
        self.assertEqual(response.context["session_history_action_count"], 1)
        self.assertEqual(response.context["session_history_product_count"], 1)
        self.assertEqual(response.context["session_history_net"], 2)
        self.assertContains(response, "Session History")
        self.assertContains(response, self.p1.name)

    def test_inventory_history_lists_only_counted_products(self):
        self._start_inventory([self.p1.product_id, self.p2.product_id])
        session = self._latest_session()
        self.client.post(
            reverse("add_quantity", kwargs={
                "session_id": session.pk,
                "product_id": self.p1.product_id,
            }),
            {"amount": 3},
        )

        response = self.client.get(reverse("checkin_session", kwargs={"session_id": session.pk}))

        history = response.context["session_history"]
        self.assertEqual([line.product_id for line in history], [self.p1.product_id])
        self.assertTrue(response.context["session_history_is_count"])
        self.assertEqual(response.context["session_history_net"], 3)

    def test_active_page_history_caps_rows_but_keeps_complete_totals(self):
        session = CheckinSession.objects.create(user=self.user, scanned_by="Long session")
        StockChange.objects.bulk_create([
            StockChange(
                product=self.p1,
                session=session,
                user=self.user,
                change_type="checkin",
                quantity=1,
            )
            for _ in range(55)
        ])

        response = self.client.get(reverse("checkin_session", kwargs={"session_id": session.pk}))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["session_history"]), 50)
        self.assertEqual(response.context["session_history_action_count"], 55)
        self.assertEqual(response.context["session_history_net"], 55)
        self.assertTrue(response.context["session_history_has_more"])
        self.assertContains(response, "Showing the latest 50 of 55 actions.")

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
