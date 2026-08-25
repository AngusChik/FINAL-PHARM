from datetime import date
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Category, Product, ProductLot


@override_settings(AXES_ENABLED=False)
class EditProductLotDerivationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='lot-editor-admin', password='test-password', is_staff=True,
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='Lot-derived inventory')
        self.product = Product.objects.create(
            name='Lot Derived Product', barcode='LOT-DERIVED-1',
            price=Decimal('8.99'), price_per_unit=Decimal('4.00'),
            quantity_in_stock=7, expiry_date=date(2031, 1, 15),
            category=self.category,
        )
        ProductLot.objects.create(
            product=self.product, lot_number='LOT-A',
            expiry_date=date(2031, 1, 15), quantity_on_hand=2,
        )
        ProductLot.objects.create(
            product=self.product, lot_number='LOT-B',
            expiry_date=date(2031, 3, 20), quantity_on_hand=5,
        )
        self.url = reverse('edit_product', args=[self.product.pk])

    def _base_post(self):
        return {
            'name': self.product.name,
            'brand': '',
            'item_number': '',
            'price': '8.99',
            'barcode': self.product.barcode,
            'category': str(self.category.pk),
            'unit_size': '',
            'description': '',
            'taxable': 'on',
            'status': 'on',
            'price_per_unit': '4.00',
            'next': reverse('inventory_display'),
        }

    def test_edit_page_renders_stock_and_expiry_as_derived_non_inputs(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-derived-stock')
        self.assertContains(response, '>7</output>', html=False)
        self.assertContains(response, 'data-derived-expiries')
        self.assertContains(response, '15-01-2031')
        self.assertContains(response, '20-03-2031')
        self.assertNotContains(response, 'name="quantity_in_stock"')
        self.assertNotContains(response, 'name="expiry_date"')
        self.assertContains(response, 'name="lot_quantity"')
        self.assertContains(response, 'name="lot_expiry"')

    def test_lot_rows_override_forged_summary_stock_and_expiry(self):
        payload = self._base_post()
        payload.update({
            'quantity_in_stock': '999',
            'expiry_date': '31-12-2099',
            'extra_expiry_dates': ['30-11-2099'],
            'lot_number': ['LOT-A', 'LOT-B'],
            'lot_expiry': ['10-02-2032', '25-04-2032'],
            'lot_quantity': ['3', '6'],
        })

        response = self.client.post(self.url, payload)

        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 9)
        self.assertEqual(self.product.expiry_date, date(2032, 2, 10))
        self.assertEqual(
            list(self.product.expiry_dates.order_by('expiry_date').values_list('expiry_date', flat=True)),
            [date(2032, 2, 10), date(2032, 4, 25)],
        )
        self.assertEqual(
            list(self.product.lots.filter(archived_at__isnull=True).order_by('lot_number')
                 .values_list('lot_number', 'quantity_on_hand')),
            [('LOT-A', 3), ('LOT-B', 6)],
        )

    def test_save_returns_to_exact_filtered_product_trend_url(self):
        origin = (
            f"{reverse('product_trend')}?q={self.product.barcode}"
            "&start_date=2026-04-01&end_date=2026-08-24"
            "&chart_type=line&granularity=week"
        )
        payload = self._base_post()
        payload.update({
            'next': origin,
            'lot_number': ['LOT-A', 'LOT-B'],
            'lot_expiry': ['15-01-2031', '20-03-2031'],
            'lot_quantity': ['2', '5'],
        })

        response = self.client.post(self.url, payload)

        self.assertRedirects(response, origin, fetch_redirect_response=False)

    def test_unsafe_return_url_falls_back_to_inventory(self):
        unsafe = 'https://example.com/steal-state'
        response = self.client.get(self.url, {'next': unsafe})
        self.assertEqual(response.context['next'], reverse('inventory_display'))

        payload = self._base_post()
        payload.update({
            'next': unsafe,
            'lot_number': ['LOT-A', 'LOT-B'],
            'lot_expiry': ['15-01-2031', '20-03-2031'],
            'lot_quantity': ['2', '5'],
        })
        response = self.client.post(self.url, payload)
        self.assertRedirects(
            response, reverse('inventory_display'), fetch_redirect_response=False,
        )

    def test_delete_returns_to_the_same_origin(self):
        origin = f"{reverse('product_trend')}?q={self.product.barcode}&granularity=month"

        response = self.client.post(
            reverse('delete_item', args=[self.product.pk]),
            {'next': origin},
        )

        self.assertRedirects(response, origin, fetch_redirect_response=False)
        self.assertFalse(Product.objects.filter(pk=self.product.pk).exists())
        self.assertTrue(Product.all_objects.filter(pk=self.product.pk).exists())

    def test_page_script_keeps_derived_summary_synced_to_lot_editor(self):
        source = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'edit_product.html'
        ).read_text(encoding='utf-8')

        self.assertIn('function refreshDerivedInventory()', source)
        self.assertIn("row.querySelector('[name=\"lot_quantity\"]')", source)
        self.assertIn("row.querySelector('[name=\"lot_expiry\"]')", source)
        self.assertIn("lotEditor.addEventListener('input', refreshDerivedInventory)", source)
        self.assertIn('window.requestAnimationFrame(refreshDerivedInventory)', source)
        self.assertIn('cursor: not-allowed;', source)

