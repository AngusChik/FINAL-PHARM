import re
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from app.models import (
    Category,
    Product,
    RecentlyPurchasedProduct,
    UserTablePreference,
)


class TablePersonalizationSourceTests(SimpleTestCase):
    def test_every_application_table_is_personalized_or_explicitly_fixed(self):
        template_root = Path(settings.BASE_DIR) / "app" / "templates"
        missing = []
        missing_keys = []

        for path in template_root.rglob("*.html"):
            source = path.read_text(encoding="utf-8")
            for match in re.finditer(r"<table\b([^>]*)>", source, re.I | re.S):
                attributes = match.group(1)
                line = source.count("\n", 0, match.start()) + 1
                if (
                    "data-personalize-table" not in attributes
                    and "data-no-personalize" not in attributes
                ):
                    missing.append(f"{path.relative_to(template_root)}:{line}")
                if (
                    "data-personalize-table" in attributes
                    and "data-table-key=" not in attributes
                ):
                    missing_keys.append(f"{path.relative_to(template_root)}:{line}")

        self.assertEqual(missing, [], f"Unclassified tables: {missing}")
        self.assertEqual(missing_keys, [], f"Personalized tables without keys: {missing_keys}")

        stock_log = (
            Path(settings.BASE_DIR) / "static" / "js" / "stock_log.js"
        ).read_text(encoding="utf-8")
        self.assertIn("data-personalize-table data-table-key=\"stock-log\"", stock_log)
        self.assertIn("data-table-scroll", stock_log)

    def test_shared_client_handles_live_rows_repeated_tables_and_colspans(self):
        script = (
            Path(settings.BASE_DIR) / "static" / "js" / "ui-system.js"
        ).read_text(encoding="utf-8")

        self.assertIn("function applyTablePreferenceToKey(savedPreferences", script)
        self.assertIn("savedPreferences[tableKey] = preference;", script)
        self.assertIn("if (candidateKey !== tableKey) return;", script)
        self.assertIn("if (table.dataset.uiPersonalized === 'true')", script)
        self.assertIn("applyTablePreference(table, columns, preference);", script)
        self.assertIn("record.target.closest('table[data-personalize-table]')", script)
        self.assertIn("cell.dataset.uiOriginalColspan", script)
        self.assertIn("cell.colSpan = Math.max(1, visibleSpan);", script)
        self.assertIn("[data-table-scroll]", script)
        self.assertIn("var sliderBody = table.closest('.sl-slider-body, .rs-slider-body, .el-slider-body')", script)
        self.assertIn("var wrapper = sliderBody || table.closest", script)
        self.assertIn("toolbar._uiTable = table;", script)
        self.assertIn("record.removedNodes", script)
        self.assertIn("if (!toolbar._uiTable || toolbar._uiTable.isConnected) return;", script)
        self.assertIn("scroller._uiTopScrollElement = topScroll", script)
        self.assertIn("scroller._uiTopScrollElement.remove()", script)
        self.assertIn("scroller._uiTopScrollObserver.disconnect()", script)
        self.assertIn("uiFallbackToastStack", script)

    def test_embed_and_responsive_styles_use_the_shared_preference_contract(self):
        root = Path(settings.BASE_DIR)
        embed = (root / "app" / "templates" / "ordering_sheet_embed.html").read_text(
            encoding="utf-8"
        )
        styles = (root / "static" / "css" / "ui-system.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('class="embed-shell"', embed)
        self.assertIn('data-page="{{ request.resolver_match.url_name', embed)
        self.assertIn('data-table-preference-url="{% url \'table_preference_api\' %}"', embed)
        self.assertIn('json_script:"ui-table-preferences"', embed)
        self.assertIn("ui-system.js' %}?v=20260819-ui19", embed)
        self.assertIn("ui-system.css' %}?v=20260819-table1", embed)

        shared_scope = ":is(body.app-shell, body.embed-shell)"
        self.assertIn(f"{shared_scope} .ui-table-view-toolbar", styles)
        self.assertIn(f"{shared_scope} table .ui-column-hidden", styles)
        self.assertIn(f"{shared_scope} table.ui-table-compact", styles)
        self.assertIn(f"{shared_scope} .ui-table-top-scroll", styles)
        self.assertIn("min-height: 44px;", styles)


@override_settings(AXES_ENABLED=False, GLOBAL_MAX_SESSIONS=20)
class TablePersonalizationIntegrationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="table-personalization-admin",
            password="test-password",
            is_staff=True,
        )
        self.client.force_login(self.user)

    def test_ordering_embed_loads_the_same_database_preference_as_full_page(self):
        UserTablePreference.objects.create(
            user=self.user,
            page_key="ordering_sheet",
            table_key="main",
            density="compact",
            page_size=25,
            hidden_columns=["patient"],
        )

        response = self.client.get(reverse("ordering_sheet"), {"embed": "1"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-page="ordering_sheet"')
        self.assertContains(response, 'data-table-preference-url="')
        self.assertContains(response, 'id="ui-table-preferences"')
        self.assertContains(response, '"density": "compact"')
        self.assertContains(response, '"hidden_columns": ["patient"]')

    def test_recently_purchased_ajax_honours_saved_rows_and_refreshes_pager(self):
        category = Category.objects.create(name="Pagination products")
        for index in range(30):
            product = Product.objects.create(
                name=f"Recently purchased {index:02d}",
                price=Decimal("4.99"),
                quantity_in_stock=5,
                category=category,
            )
            RecentlyPurchasedProduct.objects.create(product=product, quantity=1)

        UserTablePreference.objects.create(
            user=self.user,
            page_key="low_stock",
            table_key="main",
            page_size=25,
        )
        url = reverse("low_stock")

        first = self.client.get(
            url,
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        self.assertEqual(first_payload["count"], 30)
        self.assertEqual(first_payload["html"].count('class="rp-product-row"'), 25)
        self.assertIn('id="rp-pagination"', first_payload["pager_html"])
        self.assertIn("Next", first_payload["pager_html"])

        second = self.client.get(
            url,
            {"page_recent": 2},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        second_payload = second.json()
        self.assertEqual(second_payload["html"].count('class="rp-product-row"'), 5)
        self.assertIn('data-current-page="2"', second_payload["pager_html"])
        self.assertIn("2 / 2", second_payload["pager_html"])
