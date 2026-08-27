from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Category, CheckinSession, InventoryCountLine, Product


@override_settings(AXES_ENABLED=False)
class CheckinFastPathTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='checkin-fast-user', password='test-password', is_staff=True,
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='Fast Check-in')
        self.product = Product.objects.create(
            name='Fast Product', barcode='FAST-100', price=Decimal('5.00'),
            quantity_in_stock=5, category=self.category,
        )
        self.session = CheckinSession.objects.create(
            user=self.user, scanned_by='Fast tester',
        )
        self.ajax = {'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'}

    def test_plus_returns_compact_json_without_redirected_page_render(self):
        response = self.client.post(
            reverse('add_quantity', args=[self.session.pk, self.product.pk]),
            {'amount': 1, 'lot_number': 'FAST-LOT'},
            **self.ajax,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/json')
        payload = response.json()
        self.assertTrue(payload['ok'])
        self.assertTrue(payload['mutated'])
        self.assertEqual(payload['display_quantity'], 6)
        self.assertEqual(payload['system_quantity'], 6)
        self.assertIn('format=checkin_fragments', payload['fragments_url'])
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 6)

    def test_repeat_barcode_scan_is_compact_but_first_scan_only_selects(self):
        url = reverse('checkin_session', args=[self.session.pk])

        first = self.client.post(url, {
            'barcode': self.product.barcode,
            'current_barcode': '',
        }, **self.ajax)
        self.assertFalse(first.json()['mutated'])
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)

        repeated = self.client.post(url, {
            'barcode': self.product.barcode,
            'current_barcode': self.product.barcode,
            'lot_number': 'SCAN-LOT',
        }, **self.ajax)
        self.assertTrue(repeated.json()['mutated'])
        self.assertEqual(repeated.json()['display_quantity'], 6)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 6)

    def test_minus_and_exact_quantity_use_the_same_compact_contract(self):
        exact = self.client.post(
            reverse('set_quantity', args=[self.session.pk, self.product.pk]),
            {'quantity': 9},
            **self.ajax,
        )
        self.assertEqual(exact.json()['display_quantity'], 9)

        minus = self.client.post(
            reverse('delete_one', args=[self.session.pk, self.product.pk]),
            {},
            **self.ajax,
        )
        self.assertEqual(minus.json()['display_quantity'], 8)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 8)

    def test_fragment_refresh_is_smaller_than_the_full_checkin_page(self):
        mutation = self.client.post(
            reverse('add_quantity', args=[self.session.pk, self.product.pk]),
            {'amount': 1, 'lot_number': 'FRAGMENT-LOT'},
            **self.ajax,
        ).json()

        fragments = self.client.get(mutation['fragments_url'], **self.ajax)
        full_page = self.client.get(mutation['navigate_url'])

        self.assertEqual(fragments.status_code, 200)
        payload = fragments.json()
        self.assertNotIn('lot_summary_html', payload)
        self.assertTrue(any(
            lot['lot_number'] == 'FRAGMENT-LOT'
            for lot in payload['receiving_lots']
        ))
        self.assertIn('value="FRAGMENT-LOT"', payload['lot_rows_html'])
        self.assertIn('name="lot_quantity"', payload['lot_rows_html'])
        self.assertIn('Session History', payload['session_history_html'])
        self.assertIn('Product movement', payload['movement_html'])
        self.assertLess(len(fragments.content), len(full_page.content) // 2)

    def test_inventory_count_fast_path_changes_only_the_count_buffer(self):
        count_session = CheckinSession.objects.create(
            user=self.user, scanned_by='Counter', inventory_mode=True,
        )
        InventoryCountLine.objects.create(
            session=count_session, product=self.product,
            product_name=self.product.name, product_barcode=self.product.barcode,
            expected_qty=5, counted_qty=0,
        )

        response = self.client.post(
            reverse('add_quantity', args=[count_session.pk, self.product.pk]),
            {'amount': 1},
            **self.ajax,
        )

        payload = response.json()
        self.assertEqual(payload['display_quantity'], 1)
        self.assertEqual(payload['system_quantity'], 5)
        self.assertEqual(payload['counted_units'], 1)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)

    def test_client_updates_primary_result_before_abortable_fragments(self):
        source = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'checkin.html'
        ).read_text(encoding='utf-8')

        self.assertIn('window.prepareReceivingDraftForStock()', source)
        self.assertIn("contentType.indexOf('application/json')", source)
        self.assertIn('applyCompactMutation(result.data)', source)
        self.assertIn('fragmentRequest.abort()', source)
        self.assertIn("window.submitCheckinMutation(barcodeForm)", source)
        self.assertIn("searchInput.value = '';", source)
        self.assertIn('const restoreScannerFocus = () => {', source)
        self.assertIn('window.requestAnimationFrame(forceScannerFocus);', source)
        self.assertIn("searchInput.value = '';\n                                restoreScannerFocus();", source)
        self.assertGreaterEqual(source.count('return Promise.resolve(false);'), 2)
