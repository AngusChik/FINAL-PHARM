from datetime import date
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import (
    Category, Product, ProductExpiryDate, ProductLot, ProductLotMovement,
    StockChange,
)


@override_settings(AXES_ENABLED=False)
class AddProductLotDerivationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='add-lot-admin', password='test-password', is_staff=True,
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='Opening lots')
        self.url = reverse('new_product')

    def _base_post(self, **overrides):
        payload = {
            'name': 'Lot-created Product',
            'item_number': 'LOT-CREATE-1',
            'brand': 'Generic',
            'barcode': 'LOTCREATE1001',
            'price': '12.99',
            'description': '',
            'category': str(self.category.pk),
            'unit_size': '30 tablets',
            'taxable': 'on',
            'price_per_unit': '6.00',
            'status': 'on',
            'next': 'inventory_display',
        }
        payload.update(overrides)
        return payload

    def test_add_page_renders_stock_and_expiry_as_lot_derived_non_inputs(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-derived-stock')
        self.assertContains(response, 'data-derived-expiry')
        self.assertContains(response, 'Stock Level')
        self.assertContains(response, '>Category</label>', html=False)
        self.assertContains(response, 'Retail Price ($)')
        self.assertContains(response, 'In this lot')
        self.assertContains(response, 'Product Expiration')
        self.assertNotContains(response, 'name="quantity_in_stock"')
        self.assertNotContains(response, 'name="expiry_date"')
        self.assertContains(response, 'name="lot_number"')
        self.assertContains(response, 'name="lot_expiry"')
        self.assertContains(response, 'name="lot_quantity"')

    def test_enter_advances_through_fields_and_focuses_create_without_submitting(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-enter-next-fields')
        self.assertContains(response, 'data-enter-next-submit="#createProductButton"')
        self.assertContains(response, 'aria-keyshortcuts="Enter"')
        self.assertContains(response, 'aria-describedby="productFormEnterHint"')
        self.assertContains(response, 'id="productFormEnterHint"')
        self.assertContains(response, '<kbd>Enter</kbd>', html=True)
        self.assertContains(
            response,
            'Next single-line field. Internal Product Notes keeps Enter for new lines.',
        )
        self.assertContains(response, 'id="createProductButton"')

        shared_ui = (
            Path(settings.BASE_DIR) / 'static' / 'js' / 'ui-system.js'
        ).read_text(encoding='utf-8')
        helper_start = shared_ui.index('function wireEnterNextFields()')
        helper_end = shared_ui.index('function openWorkflowGuide()', helper_start)
        helper = shared_ui[helper_start:helper_end]
        self.assertIn("form[data-enter-next-fields]", helper)
        self.assertIn("event.key !== 'Enter'", helper)
        self.assertIn('event.shiftKey', helper)
        self.assertIn('|| event.ctrlKey', helper)
        self.assertIn('|| event.metaKey', helper)
        self.assertIn('|| event.repeat', helper)
        self.assertIn('event.isComposing', helper)
        self.assertIn("current.tagName === 'TEXTAREA'", helper)
        self.assertIn("['hidden', 'button', 'submit', 'reset', 'checkbox', 'radio']", helper)
        self.assertIn('event.preventDefault();', helper)
        self.assertLess(
            helper.index('event.preventDefault();'),
            helper.index('event.shiftKey'),
        )
        self.assertIn('var next = controls[currentIndex + 1];', helper)
        self.assertIn("form.getAttribute('data-enter-next-submit')", helper)
        self.assertIn("next.focus({ preventScroll: true })", helper)
        self.assertIn("next.closest('[inert], [aria-hidden=\"true\"]')", helper)
        self.assertNotIn('requestSubmit', helper)
        self.assertIn('wireEnterNextFields();', shared_ui)

    def test_lot_rows_override_forged_opening_stock_and_product_expiry(self):
        response = self.client.post(self.url, self._base_post(
            quantity_in_stock='999',
            expiry_date='31-12-2099',
            extra_expiry_dates=['30-11-2099'],
            lot_number=['LATE-LOT', 'EARLY-LOT'],
            lot_expiry=['10-05-2032', '01-12-2031'],
            lot_quantity=['4', '7'],
        ))

        self.assertEqual(response.status_code, 302)
        product = Product.objects.get(barcode='LOTCREATE1001')
        self.assertEqual(product.quantity_in_stock, 11)
        self.assertEqual(product.expiry_date, date(2031, 12, 1))
        self.assertEqual(
            list(ProductExpiryDate.objects.filter(product=product)
                 .values_list('expiry_date', flat=True)),
            [date(2031, 12, 1), date(2032, 5, 10)],
        )
        self.assertEqual(
            list(ProductLot.objects.filter(product=product).order_by('lot_number')
                 .values_list('lot_number', 'expiry_date', 'quantity_on_hand')),
            [
                ('EARLY-LOT', date(2031, 12, 1), 7),
                ('LATE-LOT', date(2032, 5, 10), 4),
            ],
        )
        stock_change = StockChange.objects.get(product=product, change_type='checkin')
        self.assertEqual(stock_change.quantity, 11)
        self.assertEqual(
            sum(ProductLotMovement.objects.filter(stock_change=stock_change)
                .values_list('quantity', flat=True)),
            11,
        )

    def test_zero_quantity_lot_does_not_set_product_expiration(self):
        response = self.client.post(self.url, self._base_post(
            lot_number=['EMPTY-EARLY', 'STOCKED-LATE'],
            lot_expiry=['01-01-2030', '20-06-2033'],
            lot_quantity=['0', '3'],
        ))

        self.assertEqual(response.status_code, 302)
        product = Product.objects.get(barcode='LOTCREATE1001')
        self.assertEqual(product.quantity_in_stock, 3)
        self.assertEqual(product.expiry_date, date(2033, 6, 20))
        self.assertEqual(
            list(product.expiry_dates.values_list('expiry_date', flat=True)),
            [date(2033, 6, 20)],
        )

    def test_missing_lot_rows_create_zero_stock_and_ignore_forged_summaries(self):
        response = self.client.post(self.url, self._base_post(
            quantity_in_stock='500', expiry_date='31-12-2099',
        ))

        self.assertEqual(response.status_code, 302)
        product = Product.objects.get(barcode='LOTCREATE1001')
        self.assertEqual(product.quantity_in_stock, 0)
        self.assertIsNone(product.expiry_date)
        self.assertFalse(product.lots.exists())
        self.assertFalse(product.expiry_dates.exists())
        self.assertFalse(StockChange.objects.filter(product=product).exists())

    def test_invalid_lot_row_is_reported_without_creating_product(self):
        response = self.client.post(self.url, self._base_post(
            lot_number=['BAD-LOT'],
            lot_expiry=['20-06-2033'],
            lot_quantity=['-2'],
        ))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Lot row 1 cannot have negative stock.')
        self.assertFalse(Product.objects.filter(barcode='LOTCREATE1001').exists())

    def test_page_uses_three_column_workspace_and_internal_lot_scrolling(self):
        source = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'new_product.html'
        ).read_text(encoding='utf-8')

        self.assertIn('minmax(320px, 0.95fr)', source)
        self.assertIn('minmax(330px, 0.9fr)', source)
        self.assertIn('minmax(500px, 1.35fr)', source)
        self.assertIn('class="np-card np-identification-card"', source)
        self.assertIn('class="np-card np-commerce-card"', source)
        self.assertIn('class="np-card np-inventory-card"', source)
        self.assertIn('max-height: clamp(96px, 23vh, 210px); overflow-y: auto;', source)

    def test_page_script_keeps_read_only_summaries_synced_to_lot_rows(self):
        source = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'new_product.html'
        ).read_text(encoding='utf-8')

        self.assertIn('function refreshDerivedInventory()', source)
        self.assertIn("row.querySelector('[name=\"lot_quantity\"]')", source)
        self.assertIn("row.querySelector('[name=\"lot_expiry\"]')", source)
        self.assertIn("lotEditor.addEventListener('input'", source)
        self.assertIn("derivedExpiry.textContent = expiries.length", source)
        self.assertNotIn("setVal('id_quantity_in_stock'", source)
        self.assertNotIn("setVal('id_expiry_date'", source)
        self.assertNotIn('Block Enter-to-submit', source)
