"""Protect the approved spacing scope and the retained dashboard entry points."""

from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.template.loader import render_to_string
from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings
from django.urls import resolve, reverse

from .context_processors import ui_context


class CompactSpacingScopeTests(SimpleTestCase):
    def render_shell(self, path):
        request = RequestFactory().get(path)
        request.user = AnonymousUser()
        request.session = {}
        request.resolver_match = resolve(path)
        return render_to_string('base.html', {
            'request': request,
            **ui_context(request),
        })

    def test_protected_workflows_do_not_load_spacing_overrides(self):
        paths = (
            '/inventory/', '/checkin/session/1/',
            '/checkin/session/1/detail/', '/checkin/session/1/product/1/edit/',
            '/order/', '/order/submit/', '/order/success/1/', '/orders/',
            '/orders/1/', '/orders/1/correct/', '/giveaways/1/correct/',
            '/checkout/purchase/1/continue/', '/checkout/success/1/',
            '/labels/', '/labels/sessions/', '/labels/sessions/1/',
            '/labels/sessions/1/regenerate/',
        )
        for path in paths:
            with self.subTest(path=path):
                html = self.render_shell(path)
                self.assertIn('navigation-layout.css?v=20260925-type1" media="screen"', html)
                self.assertNotIn('compact-spacing.css', html)
                self.assertNotIn('app-shell ui-compact', html)

    def test_remaining_workspaces_opt_in_without_changing_print_styles(self):
        paths = (
            '/dashboard/', '/ordering-sheet/', '/delivery/', '/low-stock/',
            '/checkin/', '/expired-products/', '/out-of-stock/', '/sales/',
            '/history/', '/recovery/', '/supplier-orders/', '/new-product/',
            '/prescription-drugs/',
        )
        for path in paths:
            with self.subTest(path=path):
                html = self.render_shell(path)
                self.assertIn('navigation-layout.css?v=20260925-type1" media="screen"', html)
                self.assertIn('app-shell ui-compact', html)
                self.assertIn('compact-spacing.css?v=20260925-spacing2" media="screen"', html)


@override_settings(AXES_ENABLED=False)
class CompactDashboardEntryPointTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.staff = get_user_model().objects.create_user(
            username='spacing-review', password='test-only', is_staff=True,
        )

    def test_only_header_ordering_button_is_removed(self):
        self.client.force_login(self.staff)
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'id="orderingSheetBtn"')
        self.assertContains(response, 'class="daily-report-btn"')
        self.assertContains(response, 'aria-keyshortcuts="Alt+G"')
        self.assertContains(response, 'data-ui-open-shortcuts')
        for element_id in (
            'notepadInput', 'notepadAddBtn', 'notepadClearDone',
            'miniCalendar', 'reorderExpand', 'deadstockExpand',
            'ds-ignore-snacks', 'ds-ignore-braces', 'connectPhoneBtn',
            'psHomeToggle', 'osHomeToggle', 'saSliderToggle',
            'slSliderToggle', 'elHomeToggle', 'rsSliderToggle',
        ):
            with self.subTest(element_id=element_id):
                self.assertContains(response, f'id="{element_id}"')
        for route in (
            'inventory_display', 'checkin_dashboard', 'create_order',
            'checkout', 'expired_products', 'label_printing', 'archive_recovery',
            'low_stock', 'order_view', 'sales_analytics', 'history',
            'supplier_purchase_orders', 'prescription_drugs', 'active_sessions',
        ):
            with self.subTest(route=route):
                self.assertContains(response, f'href="{reverse(route)}"')
