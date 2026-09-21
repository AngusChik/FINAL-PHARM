from datetime import date
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Category, Order, OrderDetail, Product


@override_settings(AXES_ENABLED=False)
class SalesAnalyticsCategoryFiltersTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='sales-filter-admin', password='pass1234', is_staff=True,
        )
        self.client.force_login(self.user)
        order = Order.objects.create(
            user=self.user, submitted=True,
            subtotal=Decimal('35.00'), discount_amount=Decimal('0.00'),
            tax=Decimal('0.00'), tax_rate=Decimal('0.00'),
            total_price=Decimal('35.00'),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        for index, (category_name, price) in enumerate([
            ('Health', '10.00'), ('bRaCeS', '20.00'), ('Snacks', '5.00'),
        ]):
            product = Product.objects.create(
                name=f'{category_name} item', barcode=f'SALES-FILTER-{index}',
                category=Category.objects.create(name=category_name),
                price=Decimal(price), price_per_unit=Decimal('1.00'),
                quantity_in_stock=4,
            )
            OrderDetail.objects.create(
                order=order, product=product,
                product_name=product.name, product_barcode=product.barcode,
                quantity=1, price=product.price,
                cost_per_unit_at_sale=Decimal('1.00'), taxable_at_sale=False,
            )

    def test_braces_filter_removes_only_braces_from_all_analytics_totals(self):
        response = self.client.get(reverse('sales_analytics'), {
            'start': date.today().isoformat(), 'end': date.today().isoformat(),
            'ignore_braces': '1',
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['ignore_braces'])
        self.assertFalse(response.context['ignore_snacks'])
        self.assertEqual(response.context['kpi']['revenue'], 15)
        self.assertEqual(response.context['kpi']['profit'], 13)
        self.assertEqual(response.context['kpi']['items'], 2)
        self.assertEqual(response.context['kpi']['orders'], 1)
        self.assertEqual(response.context['revenue_series'][0]['revenue'], 15)
        self.assertEqual(
            {row['name'] for row in response.context['category_sales']},
            {'Health', 'Snacks'},
        )
        self.assertEqual(
            {row['name'] for row in response.context['top_products']},
            {'Health item', 'Snacks item'},
        )
        self.assertEqual(set(response.context['top_by_cat']), {'Health', 'Snacks'})

    def test_braces_and_snacks_filters_compose_and_can_be_cleared(self):
        response = self.client.get(reverse('sales_analytics'), {
            'ignore_braces': '1', 'ignore_snacks': '1',
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['kpi']['revenue'], 10)
        self.assertEqual(response.context['kpi']['items'], 1)
        self.assertEqual(response.context['kpi']['profit'], 9)

        cleared = self.client.get(reverse('sales_analytics'))
        self.assertEqual(cleared.context['kpi']['revenue'], 35)
        self.assertEqual(cleared.context['kpi']['items'], 3)
        self.assertFalse(cleared.context['ignore_braces'])

    def test_suggestion_filter_categories_are_independent_of_chart_dates(self):
        response = self.client.get(reverse('sales_analytics'), {
            'tab': 'suggestions', 'start': '2000-01-01', 'end': '2000-01-02',
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['kpi']['items'], 0)
        self.assertEqual(len(response.context['suggestion_categories']), 3)
        self.assertContains(response, 'data-sales-tab="suggestions"')
        self.assertContains(response, 'name="hide_braces"')
        self.assertContains(response, 'sales_suggestions.js')
