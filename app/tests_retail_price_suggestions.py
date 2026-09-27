from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .forms import AddProductForm
from .models import Product
from .views import _retail_price_suggestion_context


@override_settings(AXES_ENABLED=False)
class RetailPriceSuggestionTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username='retail-help-review', is_staff=True,
        )
        cls.product = Product.objects.create(
            name='Retail suggestion product', barcode='1234567890123',
            price=Decimal('8.99'), price_per_unit=Decimal('5.00'),
            quantity_in_stock=0,
        )

    def setUp(self):
        self.client.force_login(self.user)
        self.catalogue = patch('app.views.get_master_catalog_entry', return_value=None).start()
        self.addCleanup(patch.stopall)

    def test_both_forms_show_price_help_without_a_catalogue_match(self):
        for url in (reverse('new_product'), reverse('edit_product', args=[self.product.pk])):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'aria-label="Retail price suggestions"')
                self.assertContains(response, 'Retail Price ($)')
                self.assertContains(response, 'No catalogue suggestion available.')
                self.assertContains(response, 'Cost + 40%')

    def test_edit_uses_catalogue_price_without_replacing_current_retail(self):
        self.catalogue.return_value = {'SUGGESTED RETAIL': '$12.60'}
        response = self.client.get(reverse('edit_product', args=[self.product.pk]))
        self.assertEqual(response.context['suggested_retail'], '12.99')
        self.assertEqual(response.context['suggested_markup'], 160)
        self.assertEqual(response.context['form']['price'].value(), Decimal('8.99'))
        self.product.refresh_from_db()
        self.assertEqual(self.product.price, Decimal('8.99'))

    def test_new_product_looks_up_barcode_without_prefilling_retail(self):
        self.catalogue.return_value = {'SUGGESTED RETAIL': '12.34'}
        response = self.client.get(reverse('new_product'), {
            'barcode': '9991234567890', 'price_per_unit': '4.00',
        })
        self.catalogue.assert_called_with('9991234567890')
        self.assertEqual(response.context['suggested_retail'], '11.99')
        self.assertIsNone(response.context['form']['price'].value())

    def test_scanned_catalogue_hint_still_works_without_a_local_match(self):
        response = self.client.get(reverse('new_product'), {
            'suggested_retail': '12.60', 'price_per_unit': '5.00',
        })
        self.assertEqual(response.context['suggested_retail'], '12.99')

    def test_invalid_catalogue_amounts_are_not_displayed_as_money(self):
        form = AddProductForm(initial={'price_per_unit': '5.00'})
        for amount in ('', '#VALUE!', 'NaN', 'Infinity', '-1', '0', '1e999'):
            with self.subTest(amount=amount):
                self.catalogue.return_value = {'SUGGESTED RETAIL': amount}
                context = _retail_price_suggestion_context(form)
                self.assertEqual(context['suggested_retail'], '')
                self.assertIsNone(context['suggested_markup'])

    def test_validation_errors_keep_price_help_and_catalogue_context(self):
        self.catalogue.return_value = {'SUGGESTED RETAIL': '12.60'}
        for url in (reverse('new_product'), reverse('edit_product', args=[self.product.pk])):
            with self.subTest(url=url):
                response = self.client.post(url, {
                    'name': '', 'barcode': self.product.barcode,
                    'price': '8.99', 'price_per_unit': '5.00',
                })
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.context['form'].errors)
                self.assertEqual(response.context['suggested_retail'], '12.99')
                self.assertContains(response, 'aria-label="Retail price suggestions"')
                self.assertEqual(response.context['form']['price'].value(), '8.99')

    def test_catalogue_suggestion_does_not_require_a_usable_wholesale_cost(self):
        self.catalogue.return_value = {'SUGGESTED RETAIL': '12.60'}
        for cost in ('', '0', 'NaN', 'Infinity', '1e-999999'):
            with self.subTest(cost=cost):
                form = AddProductForm(initial={'price_per_unit': cost})
                context = _retail_price_suggestion_context(form)
                self.assertEqual(context['suggested_retail'], '12.99')
                self.assertIsNone(context['suggested_markup'])
