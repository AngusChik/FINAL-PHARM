from datetime import date
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Product


@override_settings(AXES_ENABLED=False)
class SalesProductDetailsTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="sales-product-admin", is_staff=True)
        cls.product = Product.objects.create(
            name="Sales analysis product", barcode="SLS-001", price=Decimal("10.00"),
            price_per_unit=Decimal("4.00"), quantity_in_stock=3,
        )
        cls.other = Product.objects.create(
            name="Different product", barcode="SLS-002", price=Decimal("8.00"),
            price_per_unit=Decimal("2.00"), quantity_in_stock=7,
        )

    def setUp(self):
        self.client.force_login(self.user)

    def test_search_tab_starts_without_an_analysis(self):
        response = self.client.get(reverse("sales_analytics"), {"tab": "details"})
        self.assertContains(response, 'data-sales-url="/sales/"')
        self.assertNotContains(response, 'id="inventory-analysis-title"')

    def test_selected_product_renders_same_analysis_inside_sales(self):
        params = {"tab": "details", "product_id": self.product.pk,
                  "start": "2026-01-01", "end": date.today().isoformat(),
                  "granularity": "week", "type": "line", "ignore_snacks": "1"}
        sales = self.client.get(reverse("sales_analytics"), params)
        record = self.client.get(reverse("product_details", args=[self.product.pk]), params)
        self.assertTemplateUsed(sales, "sales_analytics.html")
        self.assertTemplateUsed(sales, "partials/_product_analysis_content.html")
        self.assertContains(sales, 'sales-product-analysis')
        self.assertContains(sales, 'id="trendChart"')
        self.assertContains(sales, 'name="product_id" value="%s"' % self.product.pk)
        self.assertContains(sales, 'name="ignore_snacks" value="1"')
        self.assertEqual(sales.context["chart_type"], "line")
        self.assertEqual(sales.context["granularity"], "week")
        self.assertEqual(sales.context["analysis_start_date"], date(2026, 1, 1))
        self.assertEqual(sales.context["start_date"], "2026-01-01")
        for key in ("product", "sold", "restocked", "expired", "total_sold",
                    "total_missed", "historical_stock_levels", "recommendation_data"):
            self.assertEqual(sales.context[key], record.context[key], key)
        self.assertNotContains(record, 'class="sa-tabs"')
        self.assertNotContains(record, 'data-sales-url=')
        self.assertContains(record, 'data-details-url-template="/product/0/details/"')
        changed = self.client.get(reverse("sales_analytics"), {
            "tab": "details", "product_id": self.other.pk,
        })
        self.assertEqual(changed.context["product"], self.other)

    def test_inventory_product_link_still_opens_the_standalone_record(self):
        inventory = self.client.get(reverse("inventory_display"), {"q": self.product.barcode})
        self.assertContains(inventory, reverse("product_details", args=[self.product.pk]))
        record = self.client.get(reverse("product_details", args=[self.product.pk]))
        self.assertTemplateUsed(record, "product_details.html")
        self.assertNotContains(record, 'sales-product-analysis')

    def test_invalid_product_is_not_replaced_with_another_product(self):
        for product_id in ("missing", "-1", "9999999999999999999999999", self.other.pk + 1000):
            with self.subTest(product_id=product_id):
                response = self.client.get(reverse("sales_analytics"), {
                    "tab": "details", "product_id": product_id,
                })
                self.assertEqual(response.status_code, 404)
