from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from kohlfrisch_order import new_cart_reference, normalized_barcode


class KohlFrischDestinationAutomationTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / 'kohlfrisch_order.py'
        ).read_text(encoding='utf-8')

    def test_session_cart_reference_is_auto_datetime_and_fits_portal_limit(self):
        reference = new_cart_reference(datetime(2026, 8, 24, 13, 45, 57))
        self.assertEqual(reference, 'AUTO(260824-134557)')
        self.assertEqual(len(reference), 19)

    def test_real_catalogue_action_decides_cart_or_watchlist_route(self):
        self.assertIn('"#addToCartModal"', self.source)
        self.assertIn('"#addToWishlistModal"', self.source)
        self.assertIn(
            'destination, modal = open_destination_modal(page, action_btn)',
            self.source,
        )
        self.assertIn(
            'cart_action = first_visible(result_row, SELECTORS["row_cart_action"]',
            self.source,
        )
        self.assertIn('watchlist_action = first_visible(', self.source)
        self.assertNotIn(
            'page.evaluate("(id) => AddToWishlist(id)"',
            self.source,
        )

    def test_search_requires_exact_upc_row_and_available_cart_marker(self):
        self.assertEqual(normalized_barcode('064642025487'), '64642025487')
        self.assertEqual(normalized_barcode('64642025487'), '64642025487')
        self.assertIn('result_row = matching_result_row(page, code)', self.source)
        self.assertIn('SELECTORS["available_marker"]', self.source)
        self.assertIn('SELECTORS["unavailable_marker"]', self.source)
        self.assertIn(
            'Add to Cart was shown, but product availability was not confirmed',
            self.source,
        )
        self.assertIn(
            'Add to Watchlist was shown, but the unavailable status was not confirmed',
            self.source,
        )

    def test_watchlist_is_reused_or_created_with_the_exact_permanent_name(self):
        self.assertIn('WATCHLIST_NAME = "THE WATCHLIST"', self.source)
        self.assertIn(
            'target = find_destination_radio(modal, WATCHLIST_NAME, "Watchlist"',
            self.source,
        )
        self.assertIn(
            'select_destination_radio(page, target, "checkWatchlist")',
            self.source,
        )
        self.assertIn('fill_and_verify(name_input, WATCHLIST_NAME)', self.source)
        self.assertIn('if destinations.watchlist_ready:', self.source)
        self.assertIn('stopped before creating a duplicate', self.source)

    def test_first_cart_product_creates_and_later_products_reuse_session_cart(self):
        self.assertIn('if not destinations.cart_ready:', self.source)
        self.assertIn(
            'fill_and_verify(name_input, destinations.cart_name)',
            self.source,
        )
        self.assertIn(
            'modal, destinations.cart_name, "pendingCarts", timeout_ms=8000',
            self.source,
        )
        self.assertIn(
            'select_destination_radio(page, target, "checkCart")',
            self.source,
        )
        self.assertIn('destinations.cart_ready = True', self.source)

    def test_one_destination_session_is_shared_across_the_entire_item_loop(self):
        session_line = (
            'destinations = KFDestinationSession(cart_name=new_cart_reference())'
        )
        loop_line = 'for i, item in enumerate(items, 1):'
        add_line = 'add_item(page, item, destinations, status=status)'
        self.assertEqual(self.source.count(session_line), 1)
        self.assertLess(self.source.index(session_line), self.source.index(loop_line))
        self.assertGreater(self.source.index(add_line), self.source.index(loop_line))

    def test_add_requires_the_product_specific_success_toast(self):
        self.assertIn('expected_header = "Product Added to Cart"', self.source)
        self.assertIn('expected_header = "Product Added to Watchlist"', self.source)
        self.assertIn('and confirmed_name == product_name', self.source)
        self.assertIn('raise KFDestinationIndeterminate(', self.source)
        self.assertIn(
            '# Leave this row pending: the portal mutation is ambiguous',
            self.source,
        )

    def test_processing_has_no_artificial_delay_between_items(self):
        self.assertNotIn('THROTTLE_SECONDS', self.source)
        self.assertNotIn('time.sleep(THROTTLE_SECONDS)', self.source)
        self.assertIn(
            'Start the next item immediately. The overlay/result waits inside',
            self.source,
        )
