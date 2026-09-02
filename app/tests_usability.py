import json
import re
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils.timezone import now

from .models import (
    Category, CheckinSession, DeliveryCheckIn, Product, ProductExpiryDate, UserAction,
    UserTablePreference,
)


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=20)
class SharedUsabilityTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user(
            username='usability-admin', password='pass1234', is_staff=True,
        )
        self.pu = User.objects.create_user(
            username='usability-pu', password='pass1234', is_staff=False,
        )
        self.category = Category.objects.create(name='Usability')
        self.client = Client()

    def product(self, name, archived=False):
        product = Product.all_objects.create(
            name=name,
            price=Decimal('4.99'),
            quantity_in_stock=3,
            category=self.category,
        )
        if archived:
            Product.all_objects.filter(pk=product.pk).update(
                archived_at=now(), archived_by=self.admin,
                archive_reason='Test recovery record', status=False,
            )
            product.refresh_from_db()
        return product

    def test_access_and_help_are_visible_in_shared_layout(self):
        self.client.force_login(self.pu)
        response = self.client.get(reverse('inventory_display'))
        self.assertContains(response, 'PU1 · Admin locked')
        self.assertContains(
            response,
            'Select to unlock protected actions with the separate admin passkey '
            'for 5 minutes in this browser.',
        )
        self.assertNotContains(response, 'with the admin password')
        self.assertContains(response, 'data-ui-open-shortcuts')
        self.assertContains(response, 'data-ui-open-guide')
        self.assertContains(response, 'data-requires-admin')
        self.assertContains(response, 'ui-workflow-help')
        self.assertContains(response, 'data-personalize-table')
        self.assertEqual(response.context['workflow_help']['title'], 'Product records')

        self.client.force_login(self.admin)
        response = self.client.get(reverse('inventory_display'))
        self.assertNotContains(response, 'Staff admin')
        self.assertContains(response, 'id="navPresence"')
        self.assertContains(response, 'data-ui-open-shortcuts')
        self.assertContains(response, 'class="logout-form"')
        self.assertContains(response, 'data-can-administer="true"')

    def test_workflow_strip_is_absent_while_parent_context_remains_available(self):
        session = CheckinSession.objects.create(
            user=self.pu, scanned_by='Usability tester',
        )
        self.client.force_login(self.pu)
        response = self.client.get(reverse(
            'checkin_session_detail', kwargs={'session_id': session.pk},
        ))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['workflow_parent'], {
            'url': reverse('checkin_dashboard'),
            'label': 'Back to Check-in',
        })
        self.assertNotContains(response, 'class="workflow-nav"')
        self.assertNotContains(response, 'workflow-shortcut-decal')

    def test_product_form_parent_preserves_safe_checkin_origin(self):
        session = CheckinSession.objects.create(
            user=self.admin, scanned_by='Usability tester',
        )
        session_url = reverse('checkin_session', kwargs={'session_id': session.pk})
        self.client.force_login(self.admin)

        response = self.client.get(reverse('new_product'), {'next': session_url})
        self.assertEqual(response.context['workflow_parent'], {
            'url': session_url,
            'label': 'Back to Check-in',
        })

        unsafe = self.client.get(
            reverse('new_product'), {'next': 'https://example.com/outside'},
        )
        self.assertEqual(unsafe.context['next'], reverse('inventory_display'))
        self.assertEqual(unsafe.context['workflow_parent'], {
            'url': reverse('inventory_display'),
            'label': 'Back to Inventory',
        })

    def test_product_form_parent_preserves_exact_product_trend_origin(self):
        product = self.product('Trend return product')
        origin = (
            f"{reverse('product_trend')}?q={product.pk}&start_date=2026-04-01"
            "&end_date=2026-08-24&chart_type=line&granularity=week"
        )
        self.client.force_login(self.admin)

        response = self.client.get(
            reverse('edit_product', args=[product.pk]),
            {'next': origin},
        )

        self.assertEqual(response.context['workflow_parent'], {
            'url': origin,
            'label': 'Back to Product Trend',
        })
        self.assertEqual(response.context['next'], origin)
        self.assertContains(response, 'data-page-return')
        self.assertContains(response, 'aria-label="Back to Product Trend"')
        self.assertContains(response, '>Product Trend</span>')

    def test_edit_sources_pass_their_complete_current_url(self):
        template_root = Path(settings.BASE_DIR) / 'app' / 'templates'
        for relative_path in (
            'product_trend.html',
            'expired_products.html',
            'expiring_soon.html',
            'low_stock_trend.html',
            'out_of_stock.html',
            'partials/inv_rows.html',
            'partials/rp_rows.html',
        ):
            with self.subTest(template=relative_path):
                source = (template_root / relative_path).read_text(encoding='utf-8')
                self.assertIn('request.get_full_path|urlencode', source)

    def test_legacy_header_navigation_is_removed_from_page_templates(self):
        template_root = Path(settings.BASE_DIR) / 'app' / 'templates'
        source = '\n'.join(
            path.read_text(encoding='utf-8')
            for path in template_root.rglob('*.html')
            if path.name != 'base.html'
        )
        self.assertIsNone(re.search(
            r'<a[^>]+class="[^"]*btn-back-dashboard', source,
        ))

        removed_parent_controls = {
            'checkin_session_detail.html': 'class="sd-back"',
            'checkin.html': 'Check-in Dashboard</a>',
            'checkout.html': 'Checkout Sessions</a>',
            'edit_product.html': 'class="btn-back"',
            'new_product.html': 'href="{{ next|default:',
            'order_detail.html': 'Back to Orders</a>',
            'order_form.html': '>📋 Orders</a>',
            'transaction_correction.html': 'class="tc-back"',
            'giveaway_detail.html': 'Back to Transactions</a>',
            'checkout_success.html': 'Checkout Dashboard</a>',
        }
        for template_name, marker in removed_parent_controls.items():
            with self.subTest(template=template_name):
                template = (template_root / template_name).read_text(encoding='utf-8')
                self.assertNotIn(marker, template)

    def test_table_preference_api_persists_validated_user_settings(self):
        self.client.force_login(self.pu)
        url = reverse('table_preference_api')
        response = self.client.post(
            url,
            data=json.dumps({
                'page_key': 'inventory_display',
                'table_key': 'main',
                'density': 'compact',
                'page_size': 25,
                'hidden_columns': ['price', 'status'],
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        preference = UserTablePreference.objects.get(user=self.pu)
        self.assertEqual(preference.density, 'compact')
        self.assertEqual(preference.page_size, 25)
        self.assertEqual(preference.hidden_columns, ['price', 'status'])

        get_response = self.client.get(url, {
            'page_key': 'inventory_display', 'table_key': 'main',
        })
        self.assertEqual(get_response.json()['preference']['page_size'], 25)

        invalid = self.client.post(
            url,
            data=json.dumps({
                'page_key': 'inventory_display', 'table_key': 'main',
                'density': 'compact', 'page_size': 30, 'hidden_columns': [],
            }),
            content_type='application/json',
        )
        self.assertEqual(invalid.status_code, 400)

        reset = self.client.post(
            url,
            data=json.dumps({
                'page_key': 'inventory_display', 'table_key': 'main', 'reset': True,
            }),
            content_type='application/json',
        )
        self.assertTrue(reset.json()['reset'])
        self.assertFalse(UserTablePreference.objects.filter(user=self.pu).exists())

    def test_saved_page_size_controls_inventory_pagination(self):
        for index in range(30):
            self.product(f'Pagination product {index:02d}')
        UserTablePreference.objects.create(
            user=self.pu, page_key='inventory_display', table_key='main', page_size=25,
        )
        self.client.force_login(self.pu)
        response = self.client.get(reverse('inventory_display'))
        self.assertEqual(len(response.context['page_obj'].object_list), 25)
        self.assertEqual(response.context['page_obj'].paginator.per_page, 25)

    def test_inventory_ajax_pagination_uses_the_saved_page_size(self):
        products = [
            self.product(f'Ajax pagination product {index:02d}')
            for index in range(30)
        ]
        UserTablePreference.objects.create(
            user=self.pu, page_key='inventory_display', table_key='main', page_size=25,
        )
        self.client.force_login(self.pu)

        response = self.client.get(
            reverse('inventory_display'),
            {'page': 2},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['count'], 30)
        self.assertEqual(payload['num_pages'], 2)
        self.assertIn(products[-1].name, payload['html'])
        self.assertEqual(payload['html'].count('<tr>'), 5)
        self.assertIn('2 of 2', payload['pager'])

    def test_inventory_pagination_uses_pk_to_break_tied_sort_values(self):
        products = [
            self.product(f'Tied price product {index:02d}')
            for index in range(30)
        ]
        UserTablePreference.objects.create(
            user=self.pu, page_key='inventory_display', table_key='main', page_size=25,
        )
        self.client.force_login(self.pu)

        first_page = self.client.get(reverse('inventory_display'), {
            'sort': 'price', 'direction': 'desc', 'page': 1,
        })
        second_page = self.client.get(reverse('inventory_display'), {
            'sort': 'price', 'direction': 'desc', 'page': 2,
        })
        paged_ids = [
            product.pk
            for response in (first_page, second_page)
            for product in response.context['page_obj'].object_list
        ]

        self.assertEqual(paged_ids, [product.pk for product in products])
        self.assertEqual(len(paged_ids), len(set(paged_ids)))

    def test_inventory_labels_zero_stock_expiry_as_no_stock(self):
        expiry = now().date() + timedelta(days=60)
        zero_stock = Product.objects.create(
            name='No stock expiry display', price=Decimal('5.00'),
            quantity_in_stock=0, category=self.category, expiry_date=expiry,
        )
        ProductExpiryDate.objects.create(
            product=zero_stock, expiry_date=expiry,
        )
        in_stock = Product.objects.create(
            name='In stock expiry display', price=Decimal('5.00'),
            quantity_in_stock=1, category=self.category, expiry_date=expiry,
        )
        ProductExpiryDate.objects.create(product=in_stock, expiry_date=expiry)
        self.client.force_login(self.pu)

        zero_response = self.client.get(
            reverse('inventory_display'), {'q': zero_stock.name},
        )
        in_stock_response = self.client.get(
            reverse('inventory_display'), {'q': in_stock.name},
        )

        self.assertContains(zero_response, 'class="inv-no-stock-expiry">No stock</span>')
        self.assertNotContains(zero_response, expiry.isoformat())
        self.assertContains(in_stock_response, expiry.isoformat())

    def test_recovery_can_search_filter_paginate_and_restore(self):
        target = self.product('Needle Search Target', archived=True)
        self.product('Unrelated Archived Product', archived=True)
        DeliveryCheckIn.objects.create(
            barcode='DEL-100', first_name='Recovery', last_name='Visitor',
            archived_at=now() - timedelta(days=1), archived_by=self.admin,
            archive_reason='Old delivery record',
        )
        for index in range(27):
            self.product(f'Archived pagination {index:02d}', archived=True)

        UserTablePreference.objects.create(
            user=self.admin, page_key='archive_recovery', table_key='main', page_size=25,
        )
        self.client.force_login(self.admin)
        response = self.client.get(reverse('archive_recovery'), {'type': 'product'})
        self.assertEqual(response.context['page_obj'].paginator.per_page, 25)
        self.assertGreater(response.context['page_obj'].paginator.num_pages, 1)
        self.assertContains(response, 'data-personalize-table')

        filtered = self.client.get(reverse('archive_recovery'), {
            'type': 'product', 'q': 'Needle Search Target',
        })
        self.assertContains(filtered, 'Needle Search Target')
        self.assertNotContains(filtered, 'Unrelated Archived Product')
        self.assertEqual(filtered.context['page_obj'].paginator.count, 1)

        restore = self.client.post(reverse('archive_recovery'), {
            'kind': 'product', 'object_id': target.pk,
            'type': 'product', 'q': 'Needle Search Target',
        })
        self.assertEqual(restore.status_code, 302)
        self.assertIn('type=product', restore['Location'])
        self.assertIn('q=Needle+Search+Target', restore['Location'])
        target.refresh_from_db()
        self.assertIsNone(target.archived_at)

    def test_application_templates_do_not_use_native_browser_confirmations(self):
        template_root = Path(settings.BASE_DIR) / 'app' / 'templates'
        source = '\n'.join(
            path.read_text(encoding='utf-8')
            for path in template_root.rglob('*.html')
        )
        self.assertIsNone(re.search(r'(?<!ui)confirm\s*\(', source))
        self.assertNotIn('window.confirm', source)

    def test_delivery_average_uses_all_completed_visible_records(self):
        reference = now()

        first = DeliveryCheckIn.objects.create(
            barcode='AVG-1', first_name='Average', last_name='Twenty',
        )
        DeliveryCheckIn.objects.filter(pk=first.pk).update(
            checked_in_at=reference - timedelta(minutes=80),
            checked_out_at=reference - timedelta(minutes=60),
        )
        second = DeliveryCheckIn.objects.create(
            barcode='AVG-2', first_name='Average', last_name='Forty',
        )
        DeliveryCheckIn.objects.filter(pk=second.pk).update(
            checked_in_at=reference - timedelta(days=2, minutes=40),
            checked_out_at=reference - timedelta(days=2),
        )
        DeliveryCheckIn.objects.create(
            barcode='AVG-ACTIVE', first_name='Still', last_name='Onsite',
        )
        archived = DeliveryCheckIn.objects.create(
            barcode='AVG-ARCHIVED', first_name='Archived', last_name='Record',
            archived_at=reference,
        )
        DeliveryCheckIn.objects.filter(pk=archived.pk).update(
            checked_in_at=reference - timedelta(hours=3),
            checked_out_at=reference - timedelta(hours=1),
        )

        self.client.force_login(self.pu)
        response = self.client.get(reverse('delivery'))

        self.assertEqual(response.context['avg_minutes_on_site'], 30)
        self.assertContains(response, 'id="kpiAvgOnsite">30m</span>')

    def test_pu_can_undo_delivery_checkout_without_destructive_controls(self):
        record = DeliveryCheckIn.objects.create(
            barcode='UNDO-1', first_name='Undo', last_name='Visitor',
        )
        DeliveryCheckIn.objects.filter(pk=record.pk).update(checked_out_at=now())
        record.refresh_from_db()

        self.client.force_login(self.pu)
        page = self.client.get(reverse('delivery'))
        self.assertContains(
            page,
            f'onclick="deliveryUndo({record.pk})">Undo</button>',
        )
        self.assertNotContains(
            page,
            f'onclick="deliveryDelete({record.pk})"',
        )

        response = self.client.post(
            reverse('delivery'),
            {'action': 'undo_checkout', 'record_id': str(record.pk)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')
        record.refresh_from_db()
        self.assertIsNone(record.checked_out_at)
        self.assertTrue(UserAction.objects.filter(
            user=self.pu, action='delivery_undo_checkout',
            target='Undo Visitor',
        ).exists())

        delete_response = self.client.post(
            reverse('delivery'),
            {'action': 'delete_record', 'record_id': str(record.pk)},
        )
        self.assertEqual(delete_response.status_code, 403)
        record.refresh_from_db()
        self.assertIsNone(record.archived_at)

        clear_response = self.client.post(
            reverse('delivery'), {'action': 'clear_history'},
        )
        self.assertEqual(clear_response.status_code, 302)
        self.assertIn(reverse('passkey_unlock'), clear_response.url)
