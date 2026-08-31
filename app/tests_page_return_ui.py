from decimal import Decimal
from pathlib import Path

from django.contrib.auth import get_user_model
from django.template.loader import get_template, render_to_string
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .models import Category, CheckinSession, Product


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = PROJECT_ROOT / 'app' / 'templates'


class PageReturnTemplateTests(SimpleTestCase):
    direct_templates = (
        'inventory_display.html',
        'checkin_dashboard.html',
        'order_view.html',
        'delivery.html',
        'item_list.html',
        'label_printing.html',
        'supplier_purchase_orders.html',
        'activity_log.html',
        'active_sessions.html',
        'archive_recovery.html',
        'new_product.html',
        'edit_product.html',
        'product_trend.html',
        'expired_products.html',
        'expiring_soon.html',
        'out_of_stock.html',
        'low_stock_trend.html',
        'low_stock.html',
        'order_detail.html',
        'giveaway_detail.html',
        'transaction_correction.html',
        'sales_analytics.html',
        'daily_report.html',
        'checkin_session_detail.html',
        'checkin_reconcile.html',
        'checkin.html',
    )

    excluded_templates = (
        'home.html',
        'login.html',
        'signup.html',
        'passkey_unlock.html',
        'page_busy.html',
        'checkin_needs_review.html',
        'order_form.html',
        'checkout_chooser.html',
        'checkout.html',
        'order_success.html',
        'checkout_success.html',
        '404.html',
        '500.html',
        'ordering_sheet_embed.html',
    )

    def source(self, relative_path):
        return (TEMPLATES / relative_path).read_text(encoding='utf-8')

    def test_partial_is_a_normal_destination_labelled_anchor(self):
        rendered = render_to_string('partials/_page_return.html', {
            'page_return': {
                'url': '/inventory/?q=vitamin&page=3',
                'destination': 'Inventory',
                'label': 'Back to Inventory',
                'source': 'referrer',
            },
        })

        self.assertIn('class="ui-page-return"', rendered)
        self.assertIn('href="/inventory/?q=vitamin&amp;page=3"', rendered)
        self.assertIn('aria-label="Back to Inventory"', rendered)
        self.assertIn('data-page-return-source="referrer"', rendered)
        self.assertIn('aria-hidden="true">&larr;</span>', rendered)
        self.assertIn('>Inventory</span>', rendered)
        self.assertNotIn('history.back', rendered)

    def test_partial_renders_nothing_without_context(self):
        self.assertEqual(
            render_to_string('partials/_page_return.html', {}).strip(),
            '',
        )

    def test_eligible_page_headers_use_the_shared_title_row(self):
        include = "{% include 'partials/_page_return.html' %}"
        for template_name in self.direct_templates:
            with self.subTest(template_name=template_name):
                source = self.source(template_name)
                self.assertEqual(source.count(include), 1)
                self.assertIn('ui-page-title-row', source)

    def test_eligible_templates_compile(self):
        for template_name in self.direct_templates:
            with self.subTest(template_name=template_name):
                get_template(template_name)
        get_template('ordering_sheet.html')
        get_template('ordering_sheet_embed.html')

    def test_ordering_return_is_inside_the_non_embedded_header_only(self):
        source = self.source('partials/_ordering_sheet.html')
        start = source.index('{% if not embed %}')
        include = source.index(
            "{% include 'partials/_page_return.html' %}", start,
        )
        end = source.index('{% endif %}', include)

        self.assertLess(start, include)
        self.assertLess(include, end)

    def test_checkin_return_is_gated_to_inline_edit_rerenders(self):
        source = self.source('checkin.html')
        gate = "request.resolver_match.url_name == 'checkin_edit_product'"
        include = "{% include 'partials/_page_return.html' %}"

        self.assertGreaterEqual(source.count(gate), 3)
        self.assertLess(source.index(gate), source.index(include))
        self.assertNotIn('header-left ui-page-title-row', source)

    def test_excluded_pages_do_not_include_the_shared_return(self):
        include = "partials/_page_return.html"
        for template_name in self.excluded_templates:
            with self.subTest(template_name=template_name):
                self.assertNotIn(include, self.source(template_name))


class PageReturnAssetContractTests(SimpleTestCase):
    def source(self, relative_path):
        return (PROJECT_ROOT / relative_path).read_text(encoding='utf-8')

    def test_script_keeps_only_valid_per_tab_destinations(self):
        source = self.source('static/js/ui-system.js')

        self.assertIn("'pharmacy.page-return.v1:'", source)
        self.assertIn("source === 'explicit' || source === 'referrer'", source)
        self.assertIn("source !== 'same-page'", source)
        self.assertIn("source === 'direct-fallback'", source)
        self.assertIn('target.origin !== window.location.origin', source)
        self.assertIn("target.pathname.replace(/\\/+$/, '') || '/'", source)
        self.assertIn('targetPath === currentPath', source)
        self.assertIn('destinationNode.textContent = stored.destination', source)
        self.assertNotIn('history.back(', source)

    def test_styles_provide_touch_focus_and_mobile_stack_contracts(self):
        source = self.source('static/css/ui-system.css')

        self.assertIn('.ui-page-return {', source)
        self.assertIn('min-height: var(--touch-target, 44px);', source)
        self.assertIn('.ui-page-return:focus-visible', source)
        self.assertIn('.ui-page-title-row {', source)
        self.assertIn('flex-direction: column;', source)

    def test_base_assets_have_matching_return_navigation_cache_version(self):
        source = self.source('app/templates/base.html')
        embedded = self.source('app/templates/ordering_sheet_embed.html')

        self.assertEqual(source.count('?v=20260830-presence1'), 2)
        self.assertEqual(embedded.count('?v=20260830-presence1'), 2)


@override_settings(AXES_ENABLED=False)
class PageReturnStaticRouteRenderTests(TestCase):
    included_routes = (
        'inventory_display',
        'checkin_dashboard',
        'order_view',
        'delivery',
        'item_list',
        'label_printing',
        'ordering_sheet',
        'supplier_purchase_orders',
        'activity_log',
        'active_sessions',
        'archive_recovery',
        'new_product',
        'product_trend',
        'expired_products',
        'expiring_soon',
        'out_of_stock',
        'low_stock_trend',
        'low_stock',
        'sales_analytics',
        'daily_report',
    )

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='page-return-route-matrix',
            password='pass1234',
            is_staff=True,
        )
        self.client.force_login(self.user)

    def test_each_static_header_page_renders_one_return_control(self):
        for route_name in self.included_routes:
            with self.subTest(route_name=route_name):
                response = self.client.get(reverse(route_name))
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'class="ui-page-return"', count=1)

    def test_dashboard_and_active_purchase_workspace_render_no_return(self):
        for route_name in ('dashboard', 'create_order', 'checkout'):
            with self.subTest(route_name=route_name):
                response = self.client.get(reverse(route_name))
                self.assertEqual(response.status_code, 200)
                self.assertNotContains(response, 'class="ui-page-return"')

    def test_embedded_ordering_sheet_renders_no_return(self):
        response = self.client.get(reverse('ordering_sheet'), {'embed': '1'})

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'class="ui-page-return"')


@override_settings(AXES_ENABLED=False)
class CheckinPageReturnRenderTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='page-return-checkin',
            password='pass1234',
            is_staff=True,
        )
        self.category = Category.objects.create(name='Page Return Check-in')
        self.product = Product.objects.create(
            name='Page Return Product',
            barcode='PAGE-RETURN-001',
            price=Decimal('7.50'),
            quantity_in_stock=1,
            category=self.category,
        )
        self.session = CheckinSession.objects.create(
            user=self.user,
            scanned_by='Page Return Tester',
        )
        self.client.force_login(self.user)

    def test_active_checkin_keeps_original_header_without_return(self):
        response = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': self.product.pk},
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'data-page-return')
        rendered = response.content.decode('utf-8')
        self.assertIn('<div class="header-left">', rendered)
        self.assertNotIn('<div class="header-left ui-page-title-row">', rendered)

    def test_invalid_inline_edit_rerender_shows_session_return(self):
        response = self.client.post(
            reverse(
                'checkin_edit_product',
                args=[self.session.pk, self.product.pk],
            ),
            {},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-page-return')
        self.assertContains(response, 'Back to Check-in Session')
        self.assertContains(response, 'header-left ui-page-title-row')
