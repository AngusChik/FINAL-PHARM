from urllib.parse import urlencode

from django.contrib.auth.models import AnonymousUser, User
from django.test import RequestFactory, TestCase, override_settings
from django.urls import resolve, reverse

from app.context_processors import ui_context
from app.navigation import page_return_candidate


@override_settings(ALLOWED_HOSTS=['testserver'])
class PageReturnNavigationTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username='page-return-user', password='pass1234', is_staff=True,
        )

    def request(self, path, *, data=None, method='get', referrer=None,
                authenticated=True):
        extra = {'HTTP_HOST': 'testserver'}
        if referrer is not None:
            extra['HTTP_REFERER'] = referrer
        factory_method = getattr(self.factory, method)
        request = factory_method(path, data=data or {}, **extra)
        request.user = self.user if authenticated else AnonymousUser()
        request.session = {}
        request.resolver_match = resolve(request.path_info)
        return request

    def page_return(self, *args, **kwargs):
        return ui_context(self.request(*args, **kwargs))['page_return']

    def test_candidate_preserves_exact_safe_url_and_names_destination(self):
        current = reverse('product_trend')
        origin = (
            f"http://testserver{reverse('inventory_display')}?"
            + urlencode({
                'q': 'vitamin d & zinc',
                'category': 'OTC',
                'sort': '-quantity_in_stock',
                'page': 3,
            })
        )

        candidate = page_return_candidate(self.request(current), origin)

        self.assertEqual(candidate, {
            'url': origin,
            'destination': 'Inventory',
            'label': 'Back to Inventory',
            'is_current': False,
        })

    def test_candidate_allows_active_workspace_as_a_destination(self):
        origin = f"{reverse('checkout_cart')}?terminal=PU4"

        candidate = page_return_candidate(
            self.request(reverse('giveaway_detail', args=[14])), origin,
        )

        self.assertEqual(candidate['url'], origin)
        self.assertEqual(candidate['destination'], 'Checkout Cart')
        self.assertFalse(candidate['is_current'])

    def test_explicit_product_next_wins_and_preserves_query(self):
        origin = (
            f"{reverse('inventory_display')}?q=aspirin&category=12"
            "&sort=name&page=4"
        )
        result = self.page_return(
            reverse('edit_product', args=[41]),
            data={'next': origin},
            referrer=f"http://testserver{reverse('order_view')}?page=2",
        )

        self.assertEqual(result, {
            'url': origin,
            'destination': 'Inventory',
            'label': 'Back to Inventory',
            'source': 'explicit',
        })

    def test_posted_product_next_survives_validation_rerender(self):
        origin = (
            f"{reverse('product_trend')}?q=acetaminophen"
            "&start_date=2026-08-01&granularity=week"
        )
        result = self.page_return(
            reverse('edit_product', args=[42]),
            data={'next': origin},
            method='post',
            referrer=f"http://testserver{reverse('edit_product', args=[42])}",
        )

        self.assertEqual(result['url'], origin)
        self.assertEqual(result['destination'], 'Product Trend')
        self.assertEqual(result['source'], 'explicit')

    def test_return_to_is_an_explicit_destination(self):
        origin = f"{reverse('order_view')}?q=receipt-27&page=2"

        result = self.page_return(
            reverse('archive_recovery'), data={'return_to': origin},
        )

        self.assertEqual(result, {
            'url': origin,
            'destination': 'Transactions',
            'label': 'Back to Transactions',
            'source': 'explicit',
        })

    def test_next_is_ignored_outside_product_forms(self):
        result = self.page_return(
            reverse('activity_log'),
            data={'next': f"{reverse('order_view')}?page=8"},
        )

        self.assertEqual(result, {
            'url': reverse('dashboard'),
            'destination': 'Dashboard',
            'label': 'Back to Dashboard',
            'source': 'direct-fallback',
        })

    def test_checkin_product_edit_accepts_explicit_next(self):
        origin = f"{reverse('inventory_display')}?q=bandages&page=3"

        result = self.page_return(
            reverse('checkin_edit_product', args=[17, 29]),
            data={'next': origin},
        )

        self.assertEqual(result['url'], origin)
        self.assertEqual(result['destination'], 'Inventory')
        self.assertEqual(result['source'], 'explicit')

    def test_checkin_product_edit_is_not_a_return_destination(self):
        post_only_url = reverse('checkin_edit_product', args=[17, 29])

        self.assertIsNone(page_return_candidate(
            self.request(reverse('activity_log')), post_only_url,
        ))
        for source in ('explicit', 'referrer'):
            with self.subTest(source=source):
                kwargs = (
                    {'data': {'return_to': post_only_url}}
                    if source == 'explicit'
                    else {'referrer': post_only_url}
                )
                result = self.page_return(reverse('activity_log'), **kwargs)
                self.assertEqual(result, {
                    'url': reverse('dashboard'),
                    'destination': 'Dashboard',
                    'label': 'Back to Dashboard',
                    'source': 'direct-fallback',
                })

    def test_invalid_checkin_product_edit_post_falls_back_to_session(self):
        current = reverse('checkin_edit_product', args=[17, 29])
        request = self.request(
            current,
            method='post',
            data={'next': reverse('global_search')},
            referrer=f'http://testserver{current}',
        )

        context = ui_context(request)

        self.assertEqual(context['workflow_parent'], {
            'url': reverse('checkin_session', args=[17]),
            'label': 'Back to Session',
        })
        self.assertEqual(context['page_return'], {
            'url': reverse('checkin_session', args=[17]),
            'destination': 'Check-in Session',
            'label': 'Back to Check-in Session',
            'source': 'direct-fallback',
        })

    def test_product_non_page_next_falls_back_to_inventory(self):
        api_url = f"{reverse('global_search')}?q=aspirin"

        for route_name, path in (
            ('new_product', reverse('new_product')),
            ('edit_product', reverse('edit_product', args=[47])),
        ):
            with self.subTest(route_name=route_name):
                request = self.request(path, data={'next': api_url})
                context = ui_context(request)

                self.assertEqual(context['workflow_parent']['url'], api_url)
                self.assertEqual(context['page_return'], {
                    'url': reverse('inventory_display'),
                    'destination': 'Inventory',
                    'label': 'Back to Inventory',
                    'source': 'direct-fallback',
                })

    def test_safe_referrer_preserves_its_complete_url(self):
        origin = (
            f"http://testserver{reverse('inventory_display')}?"
            "q=insulin&stock=low&page=5"
        )

        result = self.page_return(reverse('product_trend'), referrer=origin)

        self.assertEqual(result['url'], origin)
        self.assertEqual(result['destination'], 'Inventory')
        self.assertEqual(result['source'], 'referrer')

    def test_unsafe_explicit_value_does_not_override_safe_referrer(self):
        origin = f"http://testserver{reverse('inventory_display')}?page=7"

        result = self.page_return(
            reverse('edit_product', args=[43]),
            data={'next': 'https://example.com/outside'},
            referrer=origin,
        )

        self.assertEqual(result['url'], origin)
        self.assertEqual(result['source'], 'referrer')

    def test_invalid_return_to_with_same_page_referrer_forces_direct_fallback(self):
        current = reverse('activity_log')

        result = self.page_return(
            current,
            data={'return_to': 'https://example.com/outside'},
            referrer=f'http://testserver{current}?page=2',
        )

        self.assertEqual(result, {
            'url': reverse('dashboard'),
            'destination': 'Dashboard',
            'label': 'Back to Dashboard',
            'source': 'direct-fallback',
        })

    def test_invalid_product_next_with_same_page_referrer_forces_direct_fallback(self):
        current = reverse('edit_product', args=[44])

        result = self.page_return(
            current,
            data={'next': reverse('global_search')},
            referrer=f'http://testserver{current}',
        )

        self.assertEqual(result, {
            'url': reverse('inventory_display'),
            'destination': 'Inventory',
            'label': 'Back to Inventory',
            'source': 'direct-fallback',
        })

    def test_same_path_referrer_uses_workflow_fallback_and_marks_source(self):
        current = f"{reverse('product_trend')}?q=new&page=2"
        prior_filter = (
            f"http://testserver{reverse('product_trend')}?q=old&page=1"
        )

        result = self.page_return(current, referrer=prior_filter)

        self.assertEqual(result, {
            'url': reverse('inventory_display'),
            'destination': 'Inventory',
            'label': 'Back to Inventory',
            'source': 'same-page',
        })

    def test_current_explicit_value_is_treated_as_same_page(self):
        current = f"{reverse('product_trend')}?q=current"

        result = self.page_return(
            current, data={'return_to': f"{reverse('product_trend')}?q=old"},
        )

        self.assertEqual(result['url'], reverse('inventory_display'))
        self.assertEqual(result['source'], 'same-page')

    def test_non_page_referrers_are_rejected(self):
        current = reverse('product_trend')
        rejected = (
            'https://example.com/inventory/',
            '/not-a-real-page/?q=inventory',
            reverse('login'),
            reverse('passkey_unlock'),
            reverse('order_success', args=[5]),
            reverse('global_search'),
            reverse('expired_products_pdf'),
            reverse('export_inventory_csv'),
            reverse('submit_order'),
            reverse('delete_item', args=[5]),
            reverse('label_sessions'),
        )

        for referrer in rejected:
            with self.subTest(referrer=referrer):
                result = self.page_return(current, referrer=referrer)
                self.assertEqual(result, {
                    'url': reverse('inventory_display'),
                    'destination': 'Inventory',
                    'label': 'Back to Inventory',
                    'source': 'direct-fallback',
                })

    def test_workflow_fallbacks_are_destination_labelled(self):
        cases = (
            (
                reverse('expired_products'),
                reverse('inventory_display'),
                'Inventory',
            ),
            (
                reverse('checkin_session_detail', args=[11]),
                reverse('checkin_dashboard'),
                'Check-in',
            ),
            (
                reverse('checkin_reconcile', args=[12]),
                reverse('checkin_session', args=[12]),
                'Check-in Session',
            ),
            (
                reverse('order_correction', args=[13]),
                reverse('order_detail', args=[13]),
                'Transaction',
            ),
            (
                reverse('giveaway_correction', args=[14]),
                reverse('giveaway_detail', args=[14]),
                'Transaction',
            ),
            (
                reverse('supplier_purchase_orders'),
                reverse('ordering_sheet'),
                'Ordering',
            ),
            (
                reverse('item_list'),
                reverse('delivery'),
                'Delivery',
            ),
        )

        for current, expected_url, destination in cases:
            with self.subTest(current=current):
                result = self.page_return(current)
                self.assertEqual(result['url'], expected_url)
                self.assertEqual(result['destination'], destination)
                self.assertEqual(result['label'], f'Back to {destination}')
                self.assertEqual(result['source'], 'direct-fallback')

    def test_workflow_root_falls_back_to_dashboard(self):
        for route_name in (
            'inventory_display',
            'checkin_dashboard',
            'delivery',
            'label_printing',
            'ordering_sheet',
            'activity_log',
            'active_sessions',
            'archive_recovery',
        ):
            with self.subTest(route_name=route_name):
                result = self.page_return(reverse(route_name))
                self.assertEqual(result, {
                    'url': reverse('dashboard'),
                    'destination': 'Dashboard',
                    'label': 'Back to Dashboard',
                    'source': 'direct-fallback',
                })

    def test_dashboard_and_anonymous_requests_have_no_page_return(self):
        self.assertIsNone(self.page_return(reverse('dashboard')))
        self.assertIsNone(self.page_return(
            reverse('inventory_display'), authenticated=False,
        ))
