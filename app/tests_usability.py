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
    Category, CheckinSession, DeliveryCheckIn, Product, UserTablePreference,
)


@override_settings(AXES_ENABLED=False, GLOBAL_MAX_SESSIONS=20)
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
        self.assertContains(response, 'PU · Admin locked')
        self.assertContains(response, 'data-ui-open-shortcuts')
        self.assertContains(response, 'data-ui-open-guide')
        self.assertContains(response, 'data-requires-admin')
        self.assertContains(response, 'ui-workflow-help')
        self.assertContains(response, 'data-personalize-table')
        self.assertEqual(response.context['workflow_help']['title'], 'Product records')

        self.client.force_login(self.admin)
        response = self.client.get(reverse('inventory_display'))
        self.assertContains(response, 'Staff admin')
        self.assertContains(response, 'data-can-administer="true"')

    def test_workflow_parent_follows_dashboard_on_checkin_session_detail(self):
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
        workflow = re.search(
            r'<div class="workflow-nav".*?</div>',
            response.content.decode('utf-8'),
            re.DOTALL,
        ).group(0)
        self.assertLess(
            workflow.index('workflow-dashboard-link'),
            workflow.index('workflow-parent-link'),
        )
        self.assertLess(
            workflow.index('workflow-parent-link'),
            workflow.index('workflow-nav-label'),
        )
        self.assertIn('Back to Check-in', workflow)

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
        self.assertEqual(unsafe.context['workflow_parent'], {
            'url': reverse('inventory_display'),
            'label': 'Back to Inventory',
        })

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
