import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import re
from unittest.mock import patch

from django.contrib.auth.models import User
from django.db import connection
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from .mixins import PASSKEY_SESSION_KEY
from .models import (
    Category,
    Product,
    RecentlyPurchasedProduct,
    SupplierOrderPlan,
    SupplierOrderRun,
    SupplierPurchaseOrder,
    SupplierPurchaseOrderLine,
    UserSession,
)


@override_settings(AXES_ENABLED=False)
class OrderingSuggestionsEndpointTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user(
            username='suggestion-admin',
            password='pass1234',
            is_staff=True,
        )
        self.pu = User.objects.create_user(
            username='suggestion-pu',
            password='pass1234',
        )
        self.health = Category.objects.create(name='Health')
        self.snacks = Category.objects.create(name='Snacks')
        self.braces = Category.objects.create(name='Braces')
        self.product = Product.objects.create(
            name='Alpha Tablets',
            brand='North Brand',
            barcode='SUGGEST-001',
            price=Decimal('9.99'),
            quantity_in_stock=0,
            category=self.health,
        )
        RecentlyPurchasedProduct.objects.create(
            product=self.product,
            quantity=3,
        )

    @staticmethod
    def _service_result(suggestions=None):
        generated_at = timezone.make_aware(datetime(2026, 8, 30, 10, 15))
        suggestions = suggestions or []
        return {
            'suggestions': suggestions,
            'summary': {
                'total': len(suggestions),
                'order_now': len(suggestions),
                'order_soon': 0,
                'wait': 0,
                'needs_attention': 0,
            },
            'generated_at': generated_at,
        }

    @staticmethod
    def _register_session(client, user):
        session_key = client.session.session_key
        UserSession.objects.get_or_create(
            user=user,
            session_key=session_key,
        )

    def _staff_client(self):
        client = Client()
        client.force_login(self.staff)
        self._register_session(client, self.staff)
        return client

    def test_endpoint_requires_admin_access(self):
        url = reverse('ordering_suggestions')

        anonymous = Client().get(url)
        self.assertEqual(anonymous.status_code, 302)
        self.assertEqual(anonymous.url, reverse('login'))

        locked_client = Client()
        locked_client.force_login(self.pu)
        self._register_session(locked_client, self.pu)
        locked = locked_client.get(url)
        self.assertEqual(locked.status_code, 302)
        self.assertTrue(locked.url.startswith(reverse('passkey_unlock')))
        self.assertIn('next=%2Flow-stock%2Fsuggestions%2F', locked.url)

    @patch('app.views.render_to_string', return_value='<section>Suggestions</section>')
    @patch('app.ordering_suggestions.build_ordering_suggestions')
    def test_passkey_unlocked_user_can_access(self, build_suggestions, _render):
        build_suggestions.return_value = self._service_result()
        client = Client()
        client.force_login(self.pu)
        self._register_session(client, self.pu)
        session = client.session
        session[PASSKEY_SESSION_KEY] = time.time()
        session.save()

        response = client.get(reverse('ordering_suggestions'))

        self.assertEqual(response.status_code, 200)
        build_suggestions.assert_called_once()

    @patch('app.views.render_to_string', return_value='<section>Suggestions</section>')
    @patch('app.ordering_suggestions.build_ordering_suggestions')
    def test_response_contract_is_private_and_never_cached(
        self,
        build_suggestions,
        _render,
    ):
        build_suggestions.return_value = self._service_result([
            {'product_id': self.product.pk, 'action_label': 'Order 3 now'},
        ])

        response = self._staff_client().get(reverse('ordering_suggestions'))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload),
            {'html', 'summary', 'count', 'generated_at', 'filters'},
        )
        self.assertEqual(payload['html'], '<section>Suggestions</section>')
        self.assertEqual(payload['count'], 1)
        self.assertEqual(payload['summary']['order_now'], 1)
        self.assertEqual(payload['filters'], {
            'q': '',
            'category': '',
            'hide_snacks': '',
            'hide_braces': '',
        })
        self.assertIn('no-store', response['Cache-Control'])
        self.assertIn('private', response['Cache-Control'])
        self.assertEqual(response['Pragma'], 'no-cache')
        self.assertIn('Cookie', response.get('Vary', ''))

    def test_real_endpoint_renders_the_suggestion_partial(self):
        response = self._staff_client().get(reverse('ordering_suggestions'))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['count'], 1)
        self.assertIn('Alpha Tablets', payload['html'])
        self.assertIn('How we worked this out', payload['html'])
        self.assertIn('Last 90 days', payload['html'])
        self.assertIn('Last 180 days', payload['html'])
        self.assertIn('Last year', payload['html'])

    def test_sales_tab_exposes_the_suggestion_endpoint(self):
        response = self._staff_client().get(reverse('sales_analytics'), {'tab': 'suggestions'})

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'data-suggestions-url="{reverse("ordering_suggestions")}"',
        )

    def test_recently_purchased_page_no_longer_contains_the_suggestion_board(self):
        response = self._staff_client().get(reverse('low_stock'))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'id="rp-suggestions-panel"')

    @patch('app.views.render_to_string', return_value='<section>Suggestions</section>')
    @patch('app.ordering_suggestions.build_ordering_suggestions')
    def test_filters_apply_to_all_matches_independent_of_table_pagination(
        self,
        build_suggestions,
        _render,
    ):
        products = Product.objects.bulk_create([
            Product(
                name=f'Batch Product {index:03d}',
                brand='Batch Brand',
                barcode=f'BATCH-{index:03d}',
                price=Decimal('1.00'),
                quantity_in_stock=0,
                category=self.health,
            )
            for index in range(105)
        ])
        RecentlyPurchasedProduct.objects.bulk_create([
            RecentlyPurchasedProduct(product=product, quantity=1)
            for product in products
        ])
        other_category = Category.objects.create(name='Other')
        excluded_category = Product.objects.create(
            name='Batch Product Other', price=Decimal('1.00'),
            quantity_in_stock=0, category=other_category,
        )
        excluded_snack = Product.objects.create(
            name='Batch Product Snack', price=Decimal('1.00'),
            quantity_in_stock=0, category=self.snacks,
        )
        RecentlyPurchasedProduct.objects.create(
            product=excluded_category, quantity=1,
        )
        RecentlyPurchasedProduct.objects.create(
            product=excluded_snack, quantity=1,
        )

        captured_ids = []

        def capture_queryset(recent_products, as_of=None):
            captured_ids.extend(
                recent_products.values_list('product_id', flat=True)
            )
            return self._service_result([
                {'product_id': product_id, 'action_label': 'Review first'}
                for product_id in captured_ids
            ])

        build_suggestions.side_effect = capture_queryset
        client = self._staff_client()
        url = reverse('ordering_suggestions')
        params = {
            'q': 'Batch Product',
            'category': str(self.health.pk),
            'hide_snacks': '1',
            'hide_braces': '1',
            # Bookmarked page parameters from the former paginated list are
            # harmless: suggestions still cover the complete filtered set.
            'page_recent': '2',
        }

        with CaptureQueriesContext(connection) as queries:
            response = client.get(url, params)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(captured_ids), {product.pk for product in products})
        self.assertEqual(len(captured_ids), 105)
        self.assertEqual(response.json()['count'], 105)
        self.assertEqual(response.json()['filters'], {
            'q': 'Batch Product',
            'category': str(self.health.pk),
            'hide_snacks': '1',
            'hide_braces': '1',
        })
        # Auth/session checks plus one evaluation of the filtered queryset stay
        # constant as the number of matching products grows.
        self.assertLessEqual(len(queries), 5)

    @patch('app.views.render_to_string', return_value='<section>Suggestions</section>')
    @patch('app.ordering_suggestions.build_ordering_suggestions')
    def test_ignore_braces_and_snacks_apply_independently_and_together(
        self,
        build_suggestions,
        _render,
    ):
        snack = Product.objects.create(
            name='Snack', price=Decimal('1.00'), category=self.snacks,
        )
        brace = Product.objects.create(
            name='Wrist support', price=Decimal('1.00'), category=self.braces,
        )
        RecentlyPurchasedProduct.objects.bulk_create([
            RecentlyPurchasedProduct(product=snack, quantity=1),
            RecentlyPurchasedProduct(product=brace, quantity=1),
        ])
        client = self._staff_client()
        selections = (
            ({}, {self.product.pk, snack.pk, brace.pk}),
            ({'hide_snacks': '1'}, {self.product.pk, brace.pk}),
            ({'hide_braces': '1'}, {self.product.pk, snack.pk}),
            ({'hide_snacks': '1', 'hide_braces': '1'}, {self.product.pk}),
        )
        build_suggestions.return_value = self._service_result()

        for filters, expected in selections:
            with self.subTest(filters=filters):
                response = client.get(reverse('ordering_suggestions'), filters)
                self.assertEqual(response.status_code, 200)
                rows = build_suggestions.call_args.args[0]
                self.assertEqual(set(rows.values_list('product_id', flat=True)), expected)
                self.assertEqual(response.json()['filters']['hide_braces'], filters.get('hide_braces', ''))

    @patch('app.views.render_to_string', return_value='<section>Suggestions</section>')
    def test_real_suggestion_request_does_not_create_or_change_supplier_work(
        self,
        _render,
    ):
        plan = SupplierOrderPlan.objects.create(
            created_by=self.staff,
            vendor_sequence=['mck'],
            status=SupplierOrderPlan.STATUS_COMPLETED,
        )
        run = SupplierOrderRun.objects.create(
            plan=plan,
            created_by=self.staff,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_DONE,
        )
        purchase_order = SupplierPurchaseOrder.objects.create(
            plan=plan,
            supplier=SupplierPurchaseOrder.SUPPLIER_MCKESSON,
            status=SupplierPurchaseOrder.STATUS_SUBMITTED,
            created_by=self.staff,
        )
        line = SupplierPurchaseOrderLine.objects.create(
            purchase_order=purchase_order,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity_ordered=4,
            quantity_received=1,
        )
        before = {
            'plans': SupplierOrderPlan.objects.count(),
            'runs': SupplierOrderRun.objects.count(),
            'purchase_orders': SupplierPurchaseOrder.objects.count(),
            'purchase_lines': SupplierPurchaseOrderLine.objects.count(),
            'plan_state': plan.status,
            'run_state': run.state,
            'purchase_state': purchase_order.status,
            'received': line.quantity_received,
        }

        response = self._staff_client().get(reverse('ordering_suggestions'))

        self.assertEqual(response.status_code, 200)
        plan.refresh_from_db()
        run.refresh_from_db()
        purchase_order.refresh_from_db()
        line.refresh_from_db()
        after = {
            'plans': SupplierOrderPlan.objects.count(),
            'runs': SupplierOrderRun.objects.count(),
            'purchase_orders': SupplierPurchaseOrder.objects.count(),
            'purchase_lines': SupplierPurchaseOrderLine.objects.count(),
            'plan_state': plan.status,
            'run_state': run.state,
            'purchase_state': purchase_order.status,
            'received': line.quantity_received,
        }
        self.assertEqual(after, before)

    @patch('app.views.render_to_string', return_value='<section>Suggestions</section>')
    def test_real_service_query_count_does_not_grow_per_product(self, _render):
        client = self._staff_client()
        url = reverse('ordering_suggestions')

        with CaptureQueriesContext(connection) as single_product_queries:
            single_response = client.get(url)
        self.assertEqual(single_response.status_code, 200)

        products = Product.objects.bulk_create([
            Product(
                name=f'Query Bound Product {index:02d}',
                price=Decimal('1.00'),
                quantity_in_stock=0,
                category=self.health,
            )
            for index in range(20)
        ])
        RecentlyPurchasedProduct.objects.bulk_create([
            RecentlyPurchasedProduct(product=product, quantity=1)
            for product in products
        ])

        with CaptureQueriesContext(connection) as many_product_queries:
            many_response = client.get(url)
        self.assertEqual(many_response.status_code, 200)
        self.assertEqual(many_response.json()['count'], 21)
        self.assertLessEqual(
            len(many_product_queries),
            len(single_product_queries) + 1,
        )
        self.assertLessEqual(len(many_product_queries), 14)


class OrderingSuggestionsTemplateContractTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        template_root = Path(__file__).resolve().parent / 'templates'
        cls.page = (template_root / 'low_stock.html').read_text(encoding='utf-8')
        cls.sales = (template_root / 'sales_analytics.html').read_text(encoding='utf-8')
        cls.partial = (
            template_root / 'partials' / 'rp_suggestions.html'
        ).read_text(encoding='utf-8')
        cls.rows = (
            template_root / 'partials' / 'rp_rows.html'
        ).read_text(encoding='utf-8')

    def test_review_suggestions_is_a_sales_tab_instead_of_a_flipping_board(self):
        self.assertIn('data-sales-tab="suggestions"', self.sales)
        self.assertIn('id="sa-panel-suggestions"', self.sales)
        self.assertNotIn('id="rp-suggestions-btn"', self.page)
        self.assertNotIn('id="rp-suggestions-panel"', self.page)

    def test_suggestions_tab_has_accessible_status_and_independent_filters(self):
        self.assertRegex(
            self.sales,
            re.compile(
                r'<section[^>]*id="sa-panel-suggestions"[^>]*'
                r'aria-labelledby="sa-tab-suggestions"',
                re.DOTALL,
            ),
        )
        self.assertIn('id="sa-suggestions-filters"', self.sales)
        self.assertIn('name="hide_snacks"', self.sales)
        self.assertIn('name="hide_braces"', self.sales)
        self.assertIn('independent of the chart dates', self.sales)
        self.assertIn('role="status" aria-live="polite"', self.sales)

    def test_confirmed_incoming_includes_plain_timing_note(self):
        self.assertIn('Confirmed incoming', self.partial)
        self.assertIn('{{ suggestion.incoming_note }}', self.partial)
        self.assertIn('rp-suggestion-incoming-note', self.partial)

    def test_product_movement_chart_fills_the_detail_card(self):
        self.assertIn('class="rp-chart-stage"', self.rows)
        self.assertRegex(
            self.page,
            re.compile(
                r'\.rp-chart-wrap\s*\{[^}]*display:flex;\s*flex-direction:column;',
                re.DOTALL,
            ),
        )
        self.assertRegex(
            self.page,
            re.compile(
                r'\.rp-chart-stage\s*\{[^}]*flex:1 1 220px;[^}]*min-height:220px;',
                re.DOTALL,
            ),
        )
        self.assertIn(
            "var stage = wrap.querySelector('.rp-chart-stage') || wrap;",
            self.page,
        )
        self.assertIn('offset: weeks.length === 1,', self.page)

    def test_empty_suggestions_explains_current_filter_controls(self):
        self.assertIn('clear the filters above', self.partial)
        self.assertNotIn('Return to Recently Purchased', self.partial)
