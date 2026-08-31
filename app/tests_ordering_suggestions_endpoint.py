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

    def test_recently_purchased_page_exposes_the_suggestion_endpoint(self):
        response = self._staff_client().get(reverse('low_stock'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'data-suggestions-url="{reverse("ordering_suggestions")}"',
        )

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
            # Suggestions cover the complete filtered set, regardless of which
            # table page the browser currently has open.
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
        })
        # Auth/session checks plus one evaluation of the filtered queryset stay
        # constant as the number of matching products grows.
        self.assertLessEqual(len(queries), 5)

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
        cls.partial = (
            template_root / 'partials' / 'rp_suggestions.html'
        ).read_text(encoding='utf-8')

    def test_review_button_immediately_follows_automation_and_controls_back_face(self):
        self.assertIn('class="rp-header-actions"', self.page)
        self.assertIn('.rp-header .rp-header-actions', self.page)
        self.assertRegex(
            self.page,
            re.compile(
                r'id="rp-ao-btn"[^>]*>.*?</button>\s*'
                r'<button[^>]*id="rp-suggestions-btn"[^>]*'
                r'aria-controls="rp-suggestions-panel"[^>]*'
                r'aria-expanded="false"',
                re.DOTALL,
            ),
        )

    def test_front_and_back_faces_keep_accessible_initial_state(self):
        self.assertRegex(
            self.page,
            re.compile(
                r'<section[^>]*id="rp-products-panel"[^>]*'
                r'aria-labelledby="rp-page-heading"',
                re.DOTALL,
            ),
        )
        self.assertRegex(
            self.page,
            re.compile(
                r'<section[^>]*id="rp-suggestions-panel"[^>]*'
                r'aria-labelledby="rp-suggestions-title"[^>]*'
                r'aria-hidden="true"[^>]*\binert\b',
                re.DOTALL,
            ),
        )

    def test_confirmed_incoming_includes_plain_timing_note(self):
        self.assertIn('Confirmed incoming', self.partial)
        self.assertIn('{{ suggestion.incoming_note }}', self.partial)
        self.assertIn('rp-suggestion-incoming-note', self.partial)

    def test_history_close_restores_scroll_after_native_traversal(self):
        restore_match = re.search(
            r'function restoreFrontScrollPosition\(generation\) \{.*?\n  \}',
            self.page,
            re.DOTALL,
        )
        self.assertIsNotNone(restore_match)
        restore_source = restore_match.group(0)
        self.assertIn('generation !== boardStateGeneration', restore_source)
        self.assertIn('window.requestAnimationFrame(restore);', restore_source)
        self.assertIn('restoreTimer = window.setTimeout(function() {', restore_source)
        self.assertIn("var scrollHistoryKey = 'rpSuggestionsFrontScroll';", self.page)
        self.assertIn('window.history.replaceState(currentState', self.page)
