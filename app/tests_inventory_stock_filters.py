import csv
from datetime import date, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from io import StringIO
from urllib.parse import parse_qs, urlencode, urlsplit

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import Category, Product, ProductExpiryDate, ProductLot, UserTablePreference


class _InventoryElements(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.inputs = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a" and "href" in attributes:
            self.links.append(attributes)
        elif tag == "input":
            self.inputs.append(attributes)


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=20)
class InventoryStockFilterTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username="inventory-stock-filter-user", password="test-password", is_staff=True,
        )
        cls.category = Category.objects.create(name="Allergy", low_stock_threshold=6)
        cls.other_category = Category.objects.create(name="Antacid", low_stock_threshold=2)

    def setUp(self):
        self.client.force_login(self.user)
        self.today = date.today()
        self.url = reverse("inventory_display")

    def product(self, name, **kwargs):
        fields = {
            "name": name, "category": self.category, "quantity_in_stock": 10,
            "price": Decimal("5.25"), "price_per_unit": Decimal("2.10"),
        }
        fields.update(kwargs)
        return Product.objects.create(**fields)

    def lot(self, product, number="TRACKED", quantity=10, expiry=None, archived=False):
        return ProductLot.objects.create(
            product=product, lot_number=number, quantity_on_hand=quantity,
            expiry_date=expiry, archived_at=timezone.now() if archived else None,
        )

    def listing(self, stock_filter="all", **params):
        response = self.client.get(self.url, {"stock_filter": stock_filter, **params})
        self.assertEqual(response.status_code, 200)
        return response

    def ajax(self, stock_filter="all", **params):
        response = self.client.get(
            self.url, {"stock_filter": stock_filter, **params},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def ids(self, response):
        return [product.pk for product in response.context["page_obj"].object_list]

    def assert_products(self, stock_filter, expected, **params):
        response = self.listing(stock_filter, **params)
        self.assertCountEqual(self.ids(response), [product.pk for product in expected])
        self.assertEqual(response.context["total_products"], len(expected))
        return response

    def csv_rows(self, **params):
        response = self.client.get(reverse("export_inventory_csv"), params)
        self.assertEqual(response.status_code, 200)
        return list(csv.DictReader(StringIO(response.content.decode("utf-8-sig"))))

    def option_counts(self, options):
        for option in options:
            self.assertTrue(option["label"])
            self.assertTrue(option["description"])
        return {option["value"]: option["count"] for option in options}

    def test_unassigned_requires_positive_unarchived_lot_and_excludes_archived_products(self):
        unassigned = self.product("Unassigned balance")
        self.lot(unassigned, ProductLot.UNASSIGNED, quantity=4)
        self.lot(unassigned, ProductLot.UNASSIGNED, quantity=6, expiry=self.today)
        depleted = self.product("Depleted unassigned balance")
        self.lot(depleted, ProductLot.UNASSIGNED, quantity=0)
        archived = self.product("Archived unassigned lot")
        self.lot(archived, ProductLot.UNASSIGNED, archived=True)
        assigned = self.product("Assigned balance")
        self.lot(assigned)
        stale_summary = self.product("Unassigned despite stale total", quantity_in_stock=0)
        self.lot(stale_summary, ProductLot.UNASSIGNED, quantity=1)
        archived_product = self.product("Archived product", archived_at=timezone.now())
        self.lot(archived_product, ProductLot.UNASSIGNED)

        self.assert_products("unassigned", [unassigned, stale_summary])
        payload = self.ajax("unassigned")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(self.option_counts(payload["stock_filter_options"])["unassigned"], 2)

    def test_expired_uses_live_lots_and_summary_only_without_positive_dated_lots(self):
        yesterday = self.today - timedelta(days=1)
        future = self.today + timedelta(days=90)
        mixed = self.product("Expired and future lots", expiry_date=future)
        self.lot(mixed, "EXPIRED-A", quantity=2, expiry=yesterday)
        self.lot(mixed, "EXPIRED-B", quantity=3, expiry=yesterday)
        self.lot(mixed, "FUTURE", quantity=5, expiry=future)
        stale_summary = self.product("Old summary with fresh stock", expiry_date=yesterday)
        self.lot(stale_summary, expiry=future)
        depleted = self.product("Depleted old lot with fresh stock")
        self.lot(depleted, "OLD", quantity=0, expiry=yesterday)
        self.lot(depleted, "FRESH", expiry=future)
        archived = self.product("Archived old lot with fresh stock")
        self.lot(archived, "OLD", expiry=yesterday, archived=True)
        self.lot(archived, "FRESH", expiry=future)
        legacy = self.product("Legacy summary expiry", expiry_date=yesterday)
        self.lot(legacy, "UNDATED")
        self.lot(legacy, "DEPLETED", quantity=0, expiry=future)
        self.lot(legacy, "ARCHIVED", expiry=future, archived=True)
        empty = self.product("Empty stale expiry", quantity_in_stock=0, expiry_date=yesterday)
        self.lot(empty, expiry=yesterday)
        legacy_dates = self.product("Legacy dates do not override live stock", expiry_date=future)
        ProductExpiryDate.objects.bulk_create([
            ProductExpiryDate(product=legacy_dates, expiry_date=yesterday),
            ProductExpiryDate(product=legacy_dates, expiry_date=yesterday),
        ])

        self.assert_products("expired", [mixed, legacy])

    def test_expiring_soon_includes_today_and_day_30_for_lots_and_legacy_summary(self):
        expected = []
        for days in (-1, 0, 30, 31):
            expiry = self.today + timedelta(days=days)
            product = self.product(f"Dated lot {days}")
            self.lot(product, expiry=expiry)
            legacy = self.product(f"Legacy summary {days}", expiry_date=expiry)
            if days in (0, 30):
                expected.extend([product, legacy])
        stale_summary = self.product("Soon summary with future lot", expiry_date=self.today)
        self.lot(stale_summary, expiry=self.today + timedelta(days=31))
        empty = self.product("Empty soon expiry", quantity_in_stock=0, expiry_date=self.today)
        self.lot(empty, expiry=self.today)
        depleted = self.product("Depleted soon lot")
        self.lot(depleted, quantity=0, expiry=self.today)
        archived = self.product("Archived soon lot")
        self.lot(archived, expiry=self.today, archived=True)

        self.assert_products("expiring_soon", expected)

    def test_low_stock_and_attention_respect_thresholds_and_inactive_status(self):
        low = self.product("At department threshold", quantity_in_stock=6)
        self.product("Above department threshold", quantity_in_stock=7)
        default_low = self.product("At default threshold", category=None, quantity_in_stock=3)
        self.product("Above default threshold", category=None, quantity_in_stock=4)
        empty = self.product("Active empty", quantity_in_stock=0)
        inactive_low = self.product("Inactive low", status=False, quantity_in_stock=1)
        inactive_empty = self.product("Inactive empty", status=False, quantity_in_stock=0)
        inactive_expired = self.product("Inactive expired", status=False)
        self.lot(inactive_expired, expiry=self.today - timedelta(days=1))

        self.assert_products("low_stock", [low, default_low, inactive_low])
        self.assert_products("out_of_stock", [empty, inactive_empty])
        self.assert_products("inactive", [inactive_low, inactive_empty, inactive_expired])
        self.assert_products("attention", [low, default_low, empty, inactive_expired])

    def test_overlapping_attention_reasons_and_duplicate_dates_do_not_inflate_totals(self):
        mixed = self.product("Multiple attention reasons", quantity_in_stock=6)
        self.lot(mixed, ProductLot.UNASSIGNED, quantity=1, expiry=self.today - timedelta(days=1))
        self.lot(mixed, "EXPIRED", quantity=1, expiry=self.today - timedelta(days=1))
        self.lot(mixed, "SOON-A", quantity=2, expiry=self.today)
        self.lot(mixed, "SOON-B", quantity=2, expiry=self.today)
        ProductExpiryDate.objects.bulk_create([
            ProductExpiryDate(product=mixed, expiry_date=self.today),
            ProductExpiryDate(product=mixed, expiry_date=self.today),
        ])
        legacy = self.product(
            "Legacy attention", quantity_in_stock=2, price=Decimal("3.50"),
            price_per_unit=None, expiry_date=self.today - timedelta(days=1),
        )
        self.product("Healthy stock")

        response = self.assert_products("attention", [mixed, legacy])
        self.assertEqual(response.context["total_units"], 8)
        self.assertEqual(response.context["total_retail"], Decimal("38.50"))
        self.assertEqual(response.context["total_cost"], Decimal("12.60"))
        payload = self.ajax("attention")
        self.assertEqual(payload["stats"], {
            "total_products": 2, "total_units": 8,
            "total_retail": "38.50", "total_cost": "12.60",
        })
        counts = self.option_counts(payload["stock_filter_options"])
        self.assertEqual(counts["all"], 3)
        self.assertEqual(counts["attention"], 2)
        self.assertEqual(counts["expired"], 2)
        self.assertEqual(counts["expiring_soon"], 1)

    def test_counts_follow_search_and_departments_before_selected_stock_filter(self):
        unassigned = self.product("Alpha unassigned")
        self.lot(unassigned, ProductLot.UNASSIGNED)
        expired = self.product("Alpha expired", expiry_date=self.today - timedelta(days=1))
        excluded_name = self.product("Beta unassigned")
        self.lot(excluded_name, ProductLot.UNASSIGNED)
        excluded_department = self.product("Alpha other department", category=self.other_category)
        self.lot(excluded_department, ProductLot.UNASSIGNED)
        params = {"q": "Alpha", "category_id": self.category.pk}

        response = self.assert_products("expired", [expired], **params)
        expected = {
            "all": 2, "attention": 2, "unassigned": 1, "expired": 1,
            "expiring_soon": 0, "low_stock": 0, "out_of_stock": 0, "inactive": 0,
        }
        self.assertEqual(self.option_counts(response.context["stock_filter_options"]), expected)
        payload = self.ajax("expired", **params)
        self.assertEqual(payload["stock_filter"], "expired")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(self.option_counts(payload["stock_filter_options"]), expected)

    def test_name_sku_and_barcode_search_compose_with_stock_and_department_filters(self):
        matching = self.product("Searchable medicine", item_number="FILTER-SKU", barcode="12345678901")
        self.lot(matching, ProductLot.UNASSIGNED)
        wrong_stock = self.product("Searchable assigned", item_number="FILTER-SKU-OTHER")
        self.lot(wrong_stock)
        wrong_department = self.product("Searchable elsewhere", category=self.other_category)
        self.lot(wrong_department, ProductLot.UNASSIGNED)

        for query in ("Searchable", "FILTER-SKU", "12345678901"):
            with self.subTest(query=query):
                params = {"q": query, "category_id": self.category.pk}
                self.assert_products("unassigned", [matching], **params)
                self.assertEqual(
                    [row["Name"] for row in self.csv_rows(stock_filter="unassigned", **params)],
                    [matching.name],
                )

    def test_csv_export_matches_each_filter_without_duplicate_products(self):
        expired = self.product("Export expired", expiry_date=self.today - timedelta(days=1))
        soon = self.product("Export soon")
        self.lot(soon, "SOON-A", quantity=5, expiry=self.today)
        self.lot(soon, "SOON-B", quantity=5, expiry=self.today)
        unassigned = self.product("Export unassigned")
        self.lot(unassigned, ProductLot.UNASSIGNED)
        low = self.product("Export low", quantity_in_stock=6)
        empty = self.product("Export empty", quantity_in_stock=0)
        inactive = self.product("Export inactive", status=False)
        healthy = self.product("Export healthy")
        self.product("Excluded search", quantity_in_stock=0)
        self.product("Export excluded department", category=self.other_category, quantity_in_stock=0)
        expected = {
            "all": [expired, soon, unassigned, low, empty, inactive, healthy],
            "attention": [expired, soon, unassigned, low, empty],
            "unassigned": [unassigned], "expired": [expired], "expiring_soon": [soon],
            "low_stock": [low], "out_of_stock": [empty], "inactive": [inactive],
        }

        for stock_filter, products in expected.items():
            with self.subTest(stock_filter=stock_filter):
                params = {"q": "Export", "category_id": self.category.pk}
                self.assert_products(stock_filter, products, **params)
                rows = self.csv_rows(stock_filter=stock_filter, **params)
                self.assertCountEqual([row["Name"] for row in rows], [p.name for p in products])
                self.assertEqual(sum(int(row["Qty In Stock"]) for row in rows), sum(p.quantity_in_stock for p in products))

    def test_invalid_filter_falls_back_to_all_in_full_ajax_and_export(self):
        healthy = self.product("Healthy product")
        empty = self.product("Empty product", quantity_in_stock=0)
        response = self.assert_products("invalid-filter", [healthy, empty])
        self.assertEqual(response.context["stock_filter"], "all")
        self.assertEqual(response.context["stock_filter_qs"], "")
        payload = self.ajax("invalid-filter")
        self.assertEqual(payload["stock_filter"], "all")
        self.assertEqual(payload["count"], 2)
        self.assertCountEqual(
            [row["Name"] for row in self.csv_rows(stock_filter="invalid-filter")],
            [healthy.name, empty.name],
        )

    def test_empty_results_return_zero_stats_but_keep_other_filter_counts(self):
        self.product("Healthy product")
        payload = self.ajax("expired")
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["stats"], {
            "total_products": 0, "total_units": 0, "total_retail": "0.00", "total_cost": "0.00",
        })
        self.assertIn("No products found", payload["html"])
        self.assertEqual(self.option_counts(payload["stock_filter_options"])["all"], 1)

    def test_full_and_ajax_paging_and_product_returns_preserve_stock_filter(self):
        UserTablePreference.objects.create(
            user=self.user, page_key="inventory_display", table_key="main", page_size=25,
        )
        for index in range(30):
            product = self.product(f"Needle product {index:02d}")
            self.lot(product, ProductLot.UNASSIGNED)
        params = {
            "stock_filter": "unassigned", "q": "Needle", "category_id": self.category.pk,
            "sort": "name", "direction": "desc", "page": "2",
        }
        origin = self.url + "?" + urlencode(params)
        response = self.client.get(origin)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["page_obj"].number, 2)
        self.assertEqual(len(self.ids(response)), 5)
        self.assertEqual(response.context["stock_filter_qs"], "&stock_filter=unassigned")
        full_elements = _InventoryElements(response.content.decode())
        selected = [item for item in full_elements.inputs if item.get("name") == "stock_filter" and "checked" in item]
        self.assertEqual([item.get("value") for item in selected], ["unassigned"])

        ajax_response = self.client.get(origin, HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertEqual(ajax_response.status_code, 200)
        payload = ajax_response.json()
        self.assertEqual(payload["count"], 30)
        self.assertEqual(payload["num_pages"], 2)
        self.assertEqual(payload["stats"]["total_units"], 300)
        ajax_elements = _InventoryElements(payload["html"] + payload["pager"])
        edit_paths = {reverse("edit_product", args=[product_id]) for product_id in self.ids(response)}

        for elements in (full_elements, ajax_elements):
            pagers = [link for link in elements.links if "data-page" in link]
            self.assertTrue(pagers)
            for link in pagers:
                query = parse_qs(urlsplit(link["href"]).query)
                for key in ("stock_filter", "q", "category_id", "sort", "direction"):
                    self.assertEqual(query[key], [str(params[key])])
            detail_links = [link for link in elements.links if "inv-product-details-link" in link.get("class", "")]
            self.assertEqual(len(detail_links), 5)
            for link in detail_links:
                self.assertEqual(parse_qs(urlsplit(link["href"]).query)["return_to"], [origin])
            edit_links = [link for link in elements.links if urlsplit(link["href"]).path in edit_paths]
            self.assertEqual(len(edit_links), 5)
            for link in edit_links:
                self.assertEqual(parse_qs(urlsplit(link["href"]).query)["next"], [origin])

        details = self.client.get(detail_links[0]["href"])
        self.assertEqual(details.status_code, 200)
        self.assertEqual(details.context["return_to"], origin)
        self.assertEqual(details.context["page_return"]["url"], origin)
