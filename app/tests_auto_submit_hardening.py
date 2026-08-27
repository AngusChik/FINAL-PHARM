from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils.timezone import now

from .models import Category, Order, OrderDetail, Product, StockChange, UserAction


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=10)
class AutoSubmitBackendHardeningTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="auto-submit-admin",
            password="pass1234",
            is_staff=True,
        )
        category = Category.objects.create(name="Auto-submit")
        self.product = Product.objects.create(
            name="Timer Product",
            barcode="AUTO-SUBMIT-001",
            price=Decimal("7.50"),
            quantity_in_stock=10,
            category=category,
        )
        self.order = self._new_order(quantity=2, generation=3)
        self.client = Client()
        self.client.force_login(self.user)
        self._bind_session(self.order)

    def _new_order(self, *, quantity=1, generation=0, expires_at=None):
        timestamp = now()
        return Order.objects.create(
            user=self.user,
            draft_cart={
                str(self.product.pk): {
                    "name": self.product.name,
                    "price": str(self.product.price),
                    "quantity": quantity,
                },
            },
            draft_expires_at=expires_at or timestamp + timedelta(minutes=10),
            last_timer_reset_at=timestamp,
            timer_reset_count=generation,
        )

    def _bind_session(self, order):
        session = self.client.session
        session["order_id"] = order.pk
        session.save()

    @staticmethod
    def _deadline_ms(order):
        return int(order.draft_expires_at.timestamp() * 1000)

    def _reset(self, *, order=None, generation=None, expected_deadline_ms=None):
        order = order or self.order
        if generation is None:
            generation = order.timer_reset_count
        if expected_deadline_ms is None:
            expected_deadline_ms = self._deadline_ms(order)
        return self.client.post(
            reverse("create_order"),
            {
                "action": "reset_order_timer",
                "order_id": order.pk,
                "timer_reset_count": generation,
                "expected_deadline_ms": expected_deadline_ms,
            },
        )

    def _submit(
        self,
        *,
        order=None,
        generation=None,
        reason="auto",
        expected_deadline_ms=None,
    ):
        order = order or self.order
        if generation is None:
            generation = order.timer_reset_count
        if expected_deadline_ms is None:
            expected_deadline_ms = self._deadline_ms(order)
        return self.client.post(
            reverse("submit_order"),
            {
                "order_id": order.pk,
                "timer_reset_count": generation,
                "submit_reason": reason,
                "expected_deadline_ms": expected_deadline_ms,
            },
        )

    def assert_order_is_draft(self, order):
        order.refresh_from_db()
        self.assertFalse(order.submitted)
        self.assertTrue(order.draft_cart)
        self.assertFalse(order.details.exists())

    def test_auto_submit_binds_to_posted_order_not_session_fallback(self):
        """A stale page must never submit whichever draft the session now names."""
        page_order = self.order
        page_order.draft_expires_at = now() - timedelta(seconds=1)
        page_order.save(update_fields=["draft_expires_at"])
        competing_order = self._new_order(quantity=1, generation=8)
        self._bind_session(competing_order)

        response = self._submit(order=page_order)

        self.assertRedirects(
            response,
            reverse("order_success", args=[page_order.pk]),
            fetch_redirect_response=False,
        )
        page_order.refresh_from_db()
        competing_order.refresh_from_db()
        self.assertTrue(page_order.submitted)
        self.assertFalse(competing_order.submitted)
        self.assertTrue(competing_order.draft_cart)

    def test_reset_binds_to_posted_order_not_session_fallback(self):
        page_order = self.order
        competing_order = self._new_order(quantity=1, generation=8)
        competing_deadline = competing_order.draft_expires_at
        self._bind_session(competing_order)

        response = self._reset(order=page_order)

        self.assertEqual(response.status_code, 200)
        page_order.refresh_from_db()
        competing_order.refresh_from_db()
        self.assertEqual(page_order.timer_reset_count, 4)
        self.assertEqual(competing_order.timer_reset_count, 8)
        self.assertEqual(competing_order.draft_expires_at, competing_deadline)

    def test_reset_rejects_a_stale_timer_generation(self):
        original_deadline = self.order.draft_expires_at

        response = self._reset(generation=self.order.timer_reset_count - 1)

        self.assertEqual(response.status_code, 409)
        self.assertFalse(response.json()["ok"])
        self.order.refresh_from_db()
        self.assertEqual(self.order.timer_reset_count, 3)
        self.assertEqual(self.order.draft_expires_at, original_deadline)

    def test_reset_rejects_a_stale_expected_deadline(self):
        original_deadline = self.order.draft_expires_at

        response = self._reset(
            expected_deadline_ms=self._deadline_ms(self.order) - 1000,
        )

        self.assertEqual(response.status_code, 409)
        self.assertFalse(response.json()["ok"])
        self.order.refresh_from_db()
        self.assertEqual(self.order.timer_reset_count, 3)
        self.assertEqual(self.order.draft_expires_at, original_deadline)

    def test_locked_reset_advances_deadline_and_generation_once(self):
        original_deadline = self.order.draft_expires_at
        original_reset_at = self.order.last_timer_reset_at

        response = self._reset()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.order.refresh_from_db()
        self.assertEqual(self.order.timer_reset_count, 4)
        self.assertEqual(payload["reset_count"], 4)
        self.assertGreater(self.order.draft_expires_at, original_deadline)
        self.assertGreater(self.order.last_timer_reset_at, original_reset_at)
        self.assertEqual(
            payload["expires_at_ms"],
            int(self.order.draft_expires_at.timestamp() * 1000),
        )

    def test_get_renews_expired_saved_draft_once_without_submitting(self):
        """Reopening a saved cart renews its timer without consuming stock."""
        expired_at = now() - timedelta(days=1)
        self.order.draft_expires_at = expired_at
        self.order.last_timer_reset_at = expired_at - timedelta(minutes=10)
        self.order.save(update_fields=[
            "draft_expires_at",
            "last_timer_reset_at",
        ])

        first_response = self.client.get(reverse("create_order"))

        self.assertEqual(first_response.status_code, 200)
        self.order.refresh_from_db()
        renewed_deadline = self.order.draft_expires_at
        self.assertEqual(self.client.session["order_id"], self.order.pk)
        self.assertEqual(self.order.timer_reset_count, 4)
        self.assertGreater(renewed_deadline, now())
        self.assertFalse(self.order.submitted)
        self.assertTrue(self.order.draft_cart)
        self.assertFalse(self.order.details.exists())
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)
        self.assertFalse(
            StockChange.objects.filter(
                order_detail__order=self.order,
                change_type="checkout",
            ).exists(),
        )

        second_response = self.client.get(reverse("create_order"))

        self.assertEqual(second_response.status_code, 200)
        self.order.refresh_from_db()
        self.assertEqual(self.order.timer_reset_count, 4)
        self.assertEqual(self.order.draft_expires_at, renewed_deadline)
        self.assertFalse(self.order.submitted)
        self.assertFalse(self.order.details.exists())
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)

    def test_auto_submit_is_rejected_before_current_deadline(self):
        response = self._submit()

        self.assertRedirects(
            response, reverse("create_order"), fetch_redirect_response=False,
        )
        self.assert_order_is_draft(self.order)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)

    def test_auto_submit_rejects_a_mismatched_expected_deadline(self):
        self.order.draft_expires_at = now() - timedelta(seconds=1)
        self.order.save(update_fields=["draft_expires_at"])

        response = self._submit(
            expected_deadline_ms=self._deadline_ms(self.order) - 1000,
        )

        self.assertRedirects(
            response, reverse("create_order"), fetch_redirect_response=False,
        )
        self.assert_order_is_draft(self.order)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)

    def test_auto_submit_rejects_generation_captured_before_reset(self):
        stale_generation = self.order.timer_reset_count
        reset_response = self._reset(generation=stale_generation)
        self.assertEqual(reset_response.status_code, 200)
        self.order.refresh_from_db()
        self.assertEqual(self.order.timer_reset_count, stale_generation + 1)

        # Keep the deadline check from masking the generation contract.
        self.order.draft_expires_at = now() - timedelta(seconds=1)
        self.order.save(update_fields=["draft_expires_at"])
        response = self._submit(generation=stale_generation)

        self.assertRedirects(
            response, reverse("create_order"), fetch_redirect_response=False,
        )
        self.assert_order_is_draft(self.order)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)

    def test_auto_submit_succeeds_for_matching_expired_generation(self):
        self.order.draft_expires_at = now() - timedelta(seconds=1)
        self.order.save(update_fields=["draft_expires_at"])

        response = self._submit()

        self.assertRedirects(
            response,
            reverse("order_success", args=[self.order.pk]),
            fetch_redirect_response=False,
        )
        self.order.refresh_from_db()
        self.product.refresh_from_db()
        self.assertTrue(self.order.submitted)
        self.assertEqual(self.order.draft_cart, {})
        self.assertEqual(self.product.quantity_in_stock, 8)
        detail = self.order.details.get()
        self.assertEqual(detail.quantity, 2)
        audit = UserAction.objects.get(
            user=self.user,
            action="submit_order",
            target=f"Order #{self.order.pk}",
        )
        self.assertIn("submission_reason=auto", audit.detail)

    def test_duplicate_exact_order_submit_is_idempotent(self):
        self.order.draft_expires_at = now() - timedelta(seconds=1)
        self.order.save(update_fields=["draft_expires_at"])
        generation = self.order.timer_reset_count
        first = self._submit(generation=generation)
        self.assertRedirects(
            first,
            reverse("order_success", args=[self.order.pk]),
            fetch_redirect_response=False,
        )
        self.product.refresh_from_db()
        stock_after_first = self.product.quantity_in_stock

        second = self._submit(generation=generation)

        self.assertRedirects(
            second,
            reverse("order_success", args=[self.order.pk]),
            fetch_redirect_response=False,
        )
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, stock_after_first)
        self.assertEqual(
            OrderDetail.objects.filter(order=self.order).count(), 1,
        )
        self.assertEqual(
            StockChange.objects.filter(
                order_detail__order=self.order,
                change_type="checkout",
            ).count(),
            1,
        )

    def test_legacy_manual_post_still_submits_current_session_order(self):
        """Old clients without the hardening fields remain a manual action."""
        response = self.client.post(reverse("submit_order"))

        self.assertRedirects(
            response,
            reverse("order_success", args=[self.order.pk]),
            fetch_redirect_response=False,
        )
        self.order.refresh_from_db()
        self.product.refresh_from_db()
        self.assertTrue(self.order.submitted)
        self.assertEqual(self.product.quantity_in_stock, 8)
        self.assertEqual(self.order.details.count(), 1)


class AutoSubmitSourceContractTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "order_form.html"
        ).read_text(encoding="utf-8")

    def _slice(self, start, end, *, offset=0):
        start_index = self.source.index(start, offset)
        end_index = self.source.index(end, start_index)
        return self.source[start_index:end_index]

    def test_submit_form_carries_exact_identity_reason_and_generation(self):
        form = self._slice(
            '<form method="post" action="{% url \'submit_order\' %}" id="submitOrderForm">',
            "</form>",
        )
        self.assertIn('name="order_id"', form)
        self.assertIn("{{ order.order_id|default:'' }}", form)
        self.assertIn('name="timer_reset_count"', form)
        self.assertIn("{{ order.timer_reset_count|default:0 }}", form)
        self.assertIn('name="expected_deadline_ms"', form)
        self.assertIn('name="submit_reason"', form)
        self.assertIn('value="manual"', form)

    def test_auto_path_marks_reason_and_uses_all_submission_guards(self):
        auto_path = self._slice(
            "function autoSubmissionIsBlocked() {",
            "function refreshOrderView() {",
        )
        self.assertIn("seamlessMutationInFlight", auto_path)
        self.assertIn("orderSubmissionStarted", auto_path)
        self.assertIn("setSubmitReason('auto');", auto_path)
        self.assertIn("currentTime >= activeExpiryTime", auto_path)
        self.assertIn("Math.ceil((activeExpiryTime - currentTime) / 1000)", auto_path)
        request_submit = auto_path.index("submitForm.requestSubmit")
        self.assertLess(auto_path.index("seamlessMutationInFlight"), request_submit)
        self.assertLess(auto_path.index("orderSubmissionStarted"), request_submit)

    def test_countdown_uses_a_monotonic_server_clock_anchor(self):
        timer = self._slice(
            "var visualTimer = null;",
            "function refreshOrderView() {",
        )
        self.assertIn('data-server-now-ms', self.source)
        self.assertIn("window.performance.now", timer)
        self.assertIn("function estimatedServerNow()", timer)
        self.assertIn("var currentTime = estimatedServerNow();", timer)
        self.assertNotIn("var currentTime = Date.now();", timer)

    def test_manual_reset_control_is_absent_and_countdown_keeps_running(self):
        self.assertNotIn('id="autoSubmitReset"', self.source)
        self.assertNotIn(".auto-submit-reset", self.source)
        self.assertNotIn("function resetOrderTimer", self.source)
        self.assertNotIn("resetInFlight", self.source)
        self.assertNotIn("reset_order_timer", self.source)
        self.assertIn("visualTimer = setInterval(updateDisplay, 1000);", self.source)
        self.assertIn("requestAutomaticSubmission()", self.source)

    def test_seamless_mutations_are_counted_until_success_or_error(self):
        self.assertIn("let seamlessMutationInFlight", self.source)
        submit_listener = self._slice(
            "document.addEventListener('submit', function (event) {",
            "document.addEventListener('keydown', function (event) {",
            offset=self.source.index("let orderSubmissionStarted = false;"),
        )
        seamless_matcher = self._slice(
            "function isOrderSeamlessForm(form) {",
            "function setSubmitReason(reason) {",
        )
        self.assertIn("form[data-seamless]", seamless_matcher)
        self.assertIn("isOrderSeamlessForm(currentForm)", submit_listener)
        self.assertIn("seamlessMutationInFlight", submit_listener)
        seamless_branch = submit_listener[
            submit_listener.index("if (isOrderSeamlessForm(currentForm))") :
        ]
        self.assertIn("if (orderSubmissionStarted)", seamless_branch)
        self.assertLess(
            seamless_branch.index("event.preventDefault()"),
            seamless_branch.index("seamlessMutationInFlight = true"),
        )

        updated = self._slice(
            "document.addEventListener('ui:seamless-updated'",
            "document.addEventListener('ui:seamless-error'",
        )
        error = self.source[
            self.source.index("document.addEventListener('ui:seamless-error'") :
        ]
        self.assertIn("seamlessMutationInFlight", updated)
        self.assertIn("seamlessMutationInFlight", error)

    def test_submit_listener_locks_replaced_forms_against_duplicates(self):
        submit_listener = self._slice(
            "document.addEventListener('submit', function (event) {",
            "document.addEventListener('keydown', function (event) {",
            offset=self.source.index("let orderSubmissionStarted = false;"),
        )
        self.assertIn("currentForm.id !== 'submitOrderForm'", submit_listener)
        self.assertIn("orderSubmissionStarted = true;", submit_listener)
        self.assertIn("currentButton.disabled = true;", submit_listener)
