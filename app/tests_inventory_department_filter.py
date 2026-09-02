from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .models import Category, Product


class InventoryDepartmentDisclosureSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "app" / "templates" / "inventory_display.html"
        ).read_text(encoding="utf-8")

    def test_department_filter_is_a_compact_native_disclosure(self):
        self.assertIn(
            '<details class="inv-cat-disclosure" id="inventory-department-filter">',
            self.source,
        )
        self.assertIn('<summary class="inv-cat-tile">', self.source)
        self.assertNotIn(
            '<details class="inv-cat-disclosure" id="inventory-department-filter" open>',
            self.source,
        )
        self.assertEqual(self.source.count('id="inv-cat-hint"'), 1)
        self.assertLess(
            self.source.index('id="inv-cat-hint"'),
            self.source.index('id="category-select"'),
        )
        self.assertIn('<fieldset class="inv-cat-fieldset">', self.source)
        self.assertIn(
            '<legend class="ui-sr-only">Choose departments to include</legend>',
            self.source,
        )

    def test_disclosure_and_category_tiles_are_keyboard_and_touch_safe(self):
        self.assertIn('.inv-cat-disclosure > summary:focus-visible', self.source)
        self.assertIn('min-height: 68px;', self.source)
        self.assertIn('min-height: 44px;', self.source)
        self.assertIn('width: 20px; height: 20px;', self.source)
        self.assertIn('aria-live="polite"', self.source)
        self.assertIn('aria-label="Select all visible departments"', self.source)
        self.assertIn('aria-label="Clear all visible departments"', self.source)
        self.assertIn('.inv-cat-link:focus-visible', self.source)

    def test_existing_live_filter_hooks_remain_inside_the_disclosure(self):
        for element_id in (
            "inv-cat-search",
            "category-select",
            "inv-cat-all",
            "inv-cat-none",
        ):
            with self.subTest(element_id=element_id):
                self.assertEqual(self.source.count(f'id="{element_id}"'), 1)

        self.assertIn("updateCatHint();\n                    fetchInventory();", self.source)
        self.assertIn("if (event.key === 'Enter') event.preventDefault();", self.source)
        self.assertIn("catHint.textContent = 'All departments';", self.source)
        self.assertIn("if (inventoryFetchController) inventoryFetchController.abort();", self.source)
        self.assertIn("requestSequence !== inventoryFetchSequence", self.source)
        self.assertIn("'inventory-department-open'", self.source)
        self.assertIn("body.inventory-department-open .inv-floating-pager", self.source)

    def test_ajax_pager_moves_focus_only_after_a_successful_page_response(self):
        self.assertIn("const paginationRequested = Boolean(options && options.pagination);", self.source)
        self.assertIn("{ pagination: true }", self.source)
        self.assertIn("if (!response.ok) throw new Error", self.source)
        self.assertEqual(self.source.count("resultsPanel.scrollIntoView({"), 1)
        self.assertEqual(self.source.count("resultsTitle.focus({ preventScroll: true });"), 1)
        success_guard = self.source.index(
            "if (requestSequence !== inventoryFetchSequence) return;"
        )
        scroll_call = self.source.index("resultsPanel.scrollIntoView({")
        catch_handler = self.source.index(".catch(function(error)", scroll_call)
        self.assertLess(success_guard, scroll_call)
        self.assertLess(scroll_call, catch_handler)
        self.assertIn("#inventoryResultsPanel { scroll-margin-top: 88px; }", self.source)

    def test_ajax_failure_preserves_results_and_url_and_shows_retry_guidance(self):
        fetch_start = self.source.index('function fetchInventory(page, options)')
        success_guard = self.source.index(
            'if (requestSequence !== inventoryFetchSequence) return;',
            fetch_start,
        )
        row_update = self.source.index('invTbody.innerHTML = data.html;', success_guard)
        url_update = self.source.index('syncFilterUrl(params);', row_update)
        catch_handler = self.source.index('.catch(function(error)', url_update)
        catch_end = self.source.index('            });', catch_handler)
        catch_source = self.source[catch_handler:catch_end]

        self.assertLess(success_guard, row_update)
        self.assertLess(row_update, url_update)
        self.assertLess(url_update, catch_handler)
        self.assertNotIn('syncFilterUrl(', self.source[fetch_start:success_guard])
        self.assertNotIn('innerHTML =', catch_source)
        self.assertNotIn('history.replaceState', catch_source)
        self.assertNotIn('scrollIntoView', catch_source)
        self.assertNotIn('resultsTitle.focus', catch_source)
        self.assertIn(
            'Inventory results could not be updated. Check your connection and try again.',
            catch_source,
        )
        self.assertIn("window.showToast(message, 'error');", catch_source)
        self.assertIn('id="inventoryFetchStatus" role="alert" hidden', self.source)

    def test_inventory_health_uses_staff_facing_unassigned_wording(self):
        self.assertNotIn(
            "Assign positive missing lot balances to MAIN stock?",
            self.source,
        )
        self.assertIn(
            "Assign positive missing lot balances to UNASSIGNED stock?",
            self.source,
        )

    def test_inventory_audit_has_search_filter_and_bounded_batch_review(self):
        self.assertIn('data-audit-search', self.source)
        self.assertIn('data-audit-filter', self.source)
        self.assertIn('data-select-visible-expiry', self.source)
        self.assertIn('var maxExpirySelection = 100;', self.source)
        self.assertIn('run_id: currentAuditRunId', self.source)
        self.assertIn('select:not([disabled])', self.source)


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=20)
class InventoryDepartmentFilterTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="inventory-department-user",
            password="test-password",
            is_staff=True,
        )
        self.client.force_login(self.user)
        self.allergy = Category.objects.create(name="Allergy")
        self.antacid = Category.objects.create(name="Antacid")
        self.allergy_product = Product.objects.create(
            name="Allergy Product",
            item_number="ALLERGY-1",
            price=Decimal("5.00"),
            quantity_in_stock=4,
            category=self.allergy,
        )
        self.antacid_product = Product.objects.create(
            name="Antacid Product",
            item_number="ANTACID-1",
            price=Decimal("6.00"),
            quantity_in_stock=3,
            category=self.antacid,
        )
        self.uncategorized_product = Product.objects.create(
            name="Uncategorized Product",
            item_number="UNCATEGORIZED-1",
            price=Decimal("4.00"),
            quantity_in_stock=2,
            category=None,
        )

    def product_ids(self, response):
        return {product.pk for product in response.context["page_obj"].object_list}

    def test_department_query_filters_with_or_semantics_and_restores_selection(self):
        url = reverse("inventory_display")
        all_products = self.client.get(url)
        self.assertEqual(
            self.product_ids(all_products),
            {
                self.allergy_product.pk,
                self.antacid_product.pk,
                self.uncategorized_product.pk,
            },
        )

        allergy_only = self.client.get(url, {"category_id": self.allergy.pk})
        self.assertEqual(self.product_ids(allergy_only), {self.allergy_product.pk})
        self.assertEqual(
            allergy_only.context["selected_category_ids"],
            [str(self.allergy.pk)],
        )
        html = allergy_only.content.decode("utf-8")
        self.assertRegex(
            html,
            rf'name="category_id" value="{self.allergy.pk}"\s+checked',
        )
        self.assertContains(allergy_only, "1 department selected")

        both = self.client.get(
            f"{url}?category_id={self.allergy.pk}&category_id={self.antacid.pk}"
        )
        self.assertEqual(
            self.product_ids(both),
            {
                self.allergy_product.pk,
                self.antacid_product.pk,
                self.uncategorized_product.pk,
            },
        )

        exported = self.client.get(
            reverse("export_inventory_csv")
            + f"?category_id={self.allergy.pk}&category_id={self.antacid.pk}"
        )
        export_text = exported.content.decode("utf-8")
        self.assertIn("Allergy Product", export_text)
        self.assertIn("Antacid Product", export_text)
        self.assertIn("Uncategorized Product", export_text)

    def test_ajax_department_filter_updates_results_without_replacing_disclosure(self):
        response = self.client.get(
            reverse("inventory_display"),
            {"category_id": self.antacid.pk},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)
        self.assertIn("Antacid Product", payload["html"])
        self.assertNotIn("Allergy Product", payload["html"])
        self.assertNotIn("inventory-department-filter", payload["html"])
