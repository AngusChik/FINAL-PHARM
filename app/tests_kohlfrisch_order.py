from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class KohlFrischWatchlistAutomationTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / 'kohlfrisch_order.py'
        ).read_text(encoding='utf-8')

    def test_every_product_is_routed_to_the_permanent_watchlist(self):
        self.assertIn('WATCHLIST_NAME = "THE WATCHLIST"', self.source)
        self.assertIn(
            'page.evaluate("(id) => AddToWishlist(id)", m.group(1))',
            self.source,
        )
        self.assertNotIn('page.evaluate("(id) => getDetails(id)"', self.source)
        self.assertIn(
            'target_watchlist = radio_for_ref(modal, WATCHLIST_NAME)',
            self.source,
        )
        self.assertIn(
            'set_checkbox(checkbox_for_label(modal, CREATE_NEW_WATCHLIST_LABEL), False)',
            self.source,
        )
        self.assertNotIn('set_checkbox(cb, True)', self.source)

    def test_processing_has_no_artificial_delay_between_items(self):
        self.assertNotIn('THROTTLE_SECONDS', self.source)
        self.assertNotIn('time.sleep(THROTTLE_SECONDS)', self.source)
        self.assertIn(
            'Start the next item immediately. The overlay/result waits inside',
            self.source,
        )
