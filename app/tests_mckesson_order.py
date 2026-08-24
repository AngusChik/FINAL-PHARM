import ast
import inspect
import textwrap
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
        self.assertEqual(
            mckesson_order.SELECTORS["product_rows"],
            ["#productList tbody > tr[id^='line']"],
        )
        self.assertEqual(
            mckesson_order.SELECTORS["add_to_current_order"],
            ["#jqsAdd2CurrentOrderLink"],
        )
        self.assertIn(
            "a.jqsAddItem:not([data-suggestion='true'])",
            mckesson_order.SELECTORS["row_cart_button"],
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
            patch.object(mckesson_order, "current_order_id", return_value="101"),
            patch.object(
                mckesson_order, "wait_for_order_selector_closed",
                side_effect=lambda page, expected_order_id, timeout_ms=8000: (
                    events.append(f"closed:{expected_order_id}") or True
                ),
            ),
        ):
            mckesson_order.start_new_order(Mock(), status, no_input=True)

        self.assertEqual(events, ["select", "create", "verify", "closed:101"])
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

    def test_order_selector_must_be_hidden_before_startup_completes(self):
        dialog = Mock()
        dialog.is_visible.side_effect = [True, True, False]
        page = Mock()
        page.locator.return_value.first = dialog

        with patch.object(
            mckesson_order, "current_order_id", return_value="101",
        ):
            closed = mckesson_order.wait_for_order_selector_closed(
                page, expected_order_id="101", timeout_ms=1000,
            )

        self.assertTrue(closed)
        self.assertEqual(dialog.is_visible.call_count, 3)
        self.assertGreaterEqual(page.wait_for_timeout.call_count, 2)

    def test_hidden_selector_does_not_confirm_the_wrong_order(self):
        dialog = Mock()
        dialog.is_visible.return_value = False
        page = Mock()
        page.locator.return_value.first = dialog

        with patch.object(
            mckesson_order, "current_order_id", return_value="100",
        ):
            closed = mckesson_order.wait_for_order_selector_closed(
                page, expected_order_id="101", timeout_ms=1000,
            )

        self.assertFalse(closed)

    def test_order_is_created_once_before_the_product_loop(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(mckesson_order.run)))
        start_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "start_new_order"
        ]
        self.assertEqual(len(start_calls), 1)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            nested_start_calls = [
                child for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "start_new_order"
            ]
            self.assertEqual(
                nested_start_calls, [],
                "Select Order/Create Order must never run inside a product loop",
            )


class McKessonProductSearchTests(SimpleTestCase):
    def test_search_submission_is_bound_to_the_enter_navigation(self):
        source = inspect.getsource(mckesson_order.submit_product_search)

        self.assertIn("expect_navigation", source)
        self.assertLess(source.index("expect_navigation"), source.index("search.press"))
        self.assertIn("input_value", source)
        self.assertIn("barcode", source)
        self.assertNotIn(
            'wait_for_load_state("domcontentloaded")', source,
            "A completed previous document must not satisfy the next product search",
        )

    def test_product_processing_uses_visible_rows_and_not_page_global_stock(self):
        source = inspect.getsource(mckesson_order.add_item_to_cart)
        tree = ast.parse(textwrap.dedent(source))

        visible_rows_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "visible_product_rows"
        ]
        self.assertTrue(
            visible_rows_calls,
            "Search results must be resolved to visible product rows",
        )

        global_availability_checks = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "first_visible"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "page"
            ):
                continue
            selector = ast.unparse(node.args[1])
            if "unavailable_marker" in selector or "row_cart_button" in selector:
                global_availability_checks.append(selector)

        self.assertEqual(
            global_availability_checks, [],
            "Availability and Quick Add must be scoped to the barcode-matched row",
        )

    def test_closed_page_is_a_terminal_error_not_a_skipped_product(self):
        page = Mock()
        item = {"name": "Example", "barcode": "123456789012", "quantity": 1}

        with patch.object(mckesson_order, "page_is_open", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                mckesson_order.add_item_to_cart(page, item)

    def test_result_barcode_accepts_pharmaclik_u_prefix_and_zero_padding(self):
        page = Mock()
        row = Mock()
        toggle = row.locator.return_value.first
        toggle.get_attribute.return_value = "#multi-collapse-catalog-1"
        detail = page.locator.return_value.first
        detail.text_content.return_value = "GTIN: U-00064642025487"

        verified = mckesson_order.verify_result_barcode(
            page, row, "64642025487",
        )

        self.assertTrue(verified)
        page.locator.assert_called_once_with("#multi-collapse-catalog-1")

    def test_result_barcode_rejects_a_different_product(self):
        page = Mock()
        row = Mock()
        toggle = row.locator.return_value.first
        toggle.get_attribute.return_value = "#multi-collapse-catalog-1"
        detail = page.locator.return_value.first
        detail.text_content.return_value = "GTIN: U-00099999999999"

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            mckesson_order.verify_result_barcode(
                page, row, "64642025487",
            )

    def test_available_matching_row_without_quick_add_is_fatal(self):
        page = Mock()
        search = Mock(name="search")
        row = Mock(name="matching_row")
        item = {"name": "Example", "barcode": "64642025487", "quantity": 1}

        with (
            patch.object(mckesson_order, "require_page_open"),
            patch.object(mckesson_order, "current_order_id", return_value="101"),
            patch.object(mckesson_order, "assert_active_order"),
            patch.object(mckesson_order, "modal_is_visible", return_value=False),
            patch.object(mckesson_order, "submit_product_search"),
            patch.object(mckesson_order, "visible_product_rows", return_value=[row]),
            patch.object(mckesson_order, "search_result_count", return_value=1),
            patch.object(mckesson_order, "verify_result_barcode"),
            patch.object(
                mckesson_order, "first_visible",
                side_effect=[search, None, None],
            ),
            patch.object(mckesson_order, "dump_debug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Quick Add"):
                mckesson_order.add_item_to_cart(
                    page, item, expected_order_id="101",
                )


class McKessonDuplicateOrderTests(SimpleTestCase):
    def test_duplicate_in_old_order_uses_verified_add_to_current_link_only(self):
        page = Mock()
        page.locator.return_value.first = Mock(name="modal")
        link = Mock(name="add_to_current")
        link.inner_text.return_value = "Add to current order (163626612)"
        detail = Mock(name="item_order_detail")
        item = {"name": "Example", "barcode": "123456789012", "quantity": 2}

        with (
            patch.object(mckesson_order, "require_page_open"),
            patch.object(mckesson_order, "current_order_id", return_value="163626612"),
            patch.object(
                mckesson_order, "first_visible", side_effect=[link, detail],
            ) as visible,
        ):
            resolved, reason, returned_detail = (
                mckesson_order.resolve_duplicate_order_dialog(
                    page,
                    item,
                    expected_order_id="163626612",
                    cart_count_before=4,
                )
            )

        self.assertIsNone(resolved)
        self.assertEqual(reason, "")
        self.assertIs(returned_detail, detail)
        link.click.assert_called_once_with(timeout=5000)
        self.assertEqual(
            visible.call_args_list[0].args[1],
            ["#jqsAdd2CurrentOrderLink"],
        )

    def test_duplicate_link_for_different_order_is_fatal_and_not_clicked(self):
        page = Mock()
        page.locator.return_value.first = Mock(name="modal")
        link = Mock(name="wrong_add_to_current")
        link.inner_text.return_value = "Add to current order (163465403)"
        item = {"name": "Example", "barcode": "123456789012", "quantity": 1}

        with (
            patch.object(mckesson_order, "require_page_open"),
            patch.object(mckesson_order, "first_visible", return_value=link),
        ):
            with self.assertRaisesRegex(RuntimeError, "wrong transaction"):
                mckesson_order.resolve_duplicate_order_dialog(
                    page,
                    item,
                    expected_order_id="163626612",
                    cart_count_before=4,
                )

        link.click.assert_not_called()

    def test_expected_digits_in_po_column_do_not_match_transaction_column(self):
        page = Mock()
        modal = page.locator.return_value.first
        modal.inner_text.return_value = "Select an Order"
        listed_rows = modal.locator.return_value
        listed_rows.count.return_value = 1
        row = listed_rows.nth.return_value
        row.is_visible.return_value = True
        row.inner_text.return_value = "999999 163626612 Open"
        transaction_cell = row.locator.return_value.nth.return_value
        transaction_cell.inner_text.return_value = "999999"
        item = {"name": "Example", "barcode": "123456789012", "quantity": 1}

        with (
            patch.object(mckesson_order, "require_page_open"),
            patch.object(mckesson_order, "first_visible", return_value=None),
            patch.object(mckesson_order, "dismiss_modal") as dismiss,
        ):
            with self.assertRaisesRegex(RuntimeError, "another McKesson order"):
                mckesson_order.resolve_duplicate_order_dialog(
                    page,
                    item,
                    expected_order_id="163626612",
                    cart_count_before=4,
                )

        row.locator.assert_called_once_with("td")
        row.locator.return_value.nth.assert_called_once_with(1)
        dismiss.assert_not_called()


class McKessonRunSafetyTests(SimpleTestCase):
    def test_run_passes_the_captured_startup_order_to_every_item(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(mckesson_order.run)))

        captured_order_assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "active_order_id"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "start_new_order"
        ]
        self.assertEqual(len(captured_order_assignments), 1)

        add_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "add_item_to_cart"
        ]
        self.assertEqual(len(add_calls), 1)
        expected_keyword = next(
            (
                keyword for keyword in add_calls[0].keywords
                if keyword.arg == "expected_order_id"
            ),
            None,
        )
        self.assertIsNotNone(expected_keyword)
        self.assertIsInstance(expected_keyword.value, ast.Name)
        self.assertEqual(expected_keyword.value.id, "active_order_id")

    def test_fatal_item_exception_is_raised_without_recording_a_skip(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(mckesson_order.run)))
        fatal_handlers = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler)
            and any(isinstance(child, ast.Raise) for child in ast.walk(node))
        ]
        self.assertTrue(fatal_handlers)

        for handler in fatal_handlers:
            calls = [
                child for child in ast.walk(handler)
                if isinstance(child, ast.Call)
            ]
            called_attributes = {
                child.func.attr
                for child in calls
                if isinstance(child.func, ast.Attribute)
            }
            self.assertNotIn("record_result", called_attributes)

            results_appends = [
                child for child in calls
                if isinstance(child.func, ast.Attribute)
                and child.func.attr == "append"
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "results"
            ]
            self.assertEqual(results_appends, [])

    def test_order_guard_immediately_precedes_every_order_mutating_click(self):
        guarded_clicks = (
            (mckesson_order.add_item_to_cart, "cart_button.click"),
            (mckesson_order.add_item_to_cart, "add.click"),
            (mckesson_order.resolve_duplicate_order_dialog, "link.click"),
        )

        for function, click_text in guarded_clicks:
            with self.subTest(click=click_text):
                lines = [
                    line.strip()
                    for line in inspect.getsource(function).splitlines()
                    if line.strip()
                ]
                click_index = next(
                    index for index, line in enumerate(lines)
                    if line.startswith(click_text)
                )
                self.assertEqual(
                    lines[click_index - 1],
                    "wait_for_expected_order(page, expected_order_id)",
                )
