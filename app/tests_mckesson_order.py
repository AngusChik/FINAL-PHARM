from unittest.mock import Mock, patch

from django.test import SimpleTestCase

import mckesson_order


class McKessonOrderStartupTests(SimpleTestCase):
    def test_exact_pharmaclik_order_controls_are_configured(self):
        self.assertIn(
            "button.select-order-btn.jqsOrderInfoLink",
            mckesson_order.SELECTORS["select_order_button"],
        )
        self.assertIn(
            "#jqsNewOrder[data-action*='orderSelectorStartNew']",
            mckesson_order.SELECTORS["create_order_button"],
        )

    def test_start_sequence_opens_selector_then_creates_and_verifies_order(self):
        events = []
        status = Mock()

        with (
            patch.object(
                mckesson_order, "open_order_selector",
                side_effect=lambda page: events.append("select") or True,
            ),
            patch.object(
                mckesson_order, "click_create_order_button",
                side_effect=lambda page: events.append("create") or True,
            ),
            patch.object(
                mckesson_order, "wait_for_active_order",
                side_effect=lambda page, previous_label="": events.append("verify") or True,
            ),
        ):
            mckesson_order.start_new_order(Mock(), status, no_input=True)

        self.assertEqual(events, ["select", "create", "verify"])
        self.assertIn(
            "Opening Select Order",
            [call.kwargs.get("message") for call in status.update.call_args_list],
        )
        self.assertIn(
            "Clicking Create Order",
            [call.kwargs.get("message") for call in status.update.call_args_list],
        )

    def test_select_order_button_opens_ajax_order_dialog(self):
        control = Mock()
        dialog = Mock()

        with patch.object(
            mckesson_order, "first_visible", side_effect=[control, dialog],
        ):
            opened = mckesson_order.open_order_selector(Mock())

        self.assertTrue(opened)
        control.click.assert_called_once_with(timeout=5000)

    def test_collapsed_create_order_section_is_expanded(self):
        toggle = Mock()
        create = Mock()

        with patch.object(
            mckesson_order, "first_visible",
            side_effect=[None, toggle, create],
        ):
            clicked = mckesson_order.click_create_order_button(Mock())

        self.assertTrue(clicked)
        toggle.click.assert_called_once_with(timeout=5000)
        create.click.assert_called_once_with(timeout=5000)

    def test_existing_order_is_not_mistaken_for_new_order_confirmation(self):
        page = Mock()
        labels = iter([
            "Current Order: 100",
            "Current Order: 100",
            "Current Order: 101",
        ])

        with patch.object(
            mckesson_order, "current_order_label",
            side_effect=lambda page: next(labels),
        ):
            confirmed = mckesson_order.wait_for_active_order(
                page, previous_label="Current Order: 100", timeout_ms=1000,
            )

        self.assertTrue(confirmed)
        self.assertEqual(page.wait_for_timeout.call_count, 2)
