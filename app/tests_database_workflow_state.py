import json
import os
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from .models import (
    Category, DashboardTask, LabelPrintOverride, LabelQueueItem, Order,
    Product, SupplierOrderPlan, SupplierOrderRun, SupplierOrderRunItem,
)
from .supplier_orders import DatabaseRunStatus


@override_settings(AXES_ENABLED=False, GLOBAL_MAX_SESSIONS=10)
class DatabaseWorkflowStateTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='workflow-admin', password='pass1234', is_staff=True,
        )
        self.other_user = User.objects.create_user(
            username='workflow-other', password='pass1234', is_staff=True,
        )
        self.category = Category.objects.create(name='Print Label')
        self.product = Product.objects.create(
            name='Database Product', price=Decimal('12.50'), barcode='123456',
            quantity_in_stock=20, category=self.category,
        )
        self.client = Client()
        self.client.force_login(self.user)

    def test_dashboard_tasks_are_shared_and_soft_archived(self):
        created = self.client.post(
            reverse('dashboard_tasks'),
            data='{"action":"add","text":"Call supplier"}',
            content_type='application/json',
        ).json()['item']

        self.client.logout()
        second_browser = Client()
        second_browser.force_login(self.other_user)
        visible = second_browser.get(reverse('dashboard_tasks')).json()['items']
        self.assertEqual([item['text'] for item in visible], ['Call supplier'])

        second_browser.post(
            reverse('dashboard_tasks'),
            data=f'{{"action":"toggle","id":{created["id"]}}}',
            content_type='application/json',
        )
        task = DashboardTask.objects.get(pk=created['id'])
        self.assertTrue(task.completed)
        self.assertEqual(task.completed_by, self.other_user)
        self.assertIsNotNone(task.completed_at)

        second_browser.post(
            reverse('dashboard_tasks'),
            data='{"action":"clear_completed"}',
            content_type='application/json',
        )
        task.refresh_from_db()
        self.assertIsNotNone(task.archived_at)
        self.assertEqual(task.archived_by, self.other_user)

    def test_legacy_dashboard_tasks_can_be_imported(self):
        response = self.client.post(
            reverse('dashboard_tasks'),
            data='{"action":"import_legacy","items":[{"text":"Old task","done":true,"user":"GINA"}]}',
            content_type='application/json',
        )
        self.assertEqual(response.json()['imported'], 1)
        task = DashboardTask.objects.get()
        self.assertEqual(task.created_by_name, 'GINA')
        self.assertTrue(task.completed)

        duplicate = self.client.post(
            reverse('dashboard_tasks'),
            data='{"action":"import_legacy","items":[{"text":"Old task","done":true,"user":"GINA"}]}',
            content_type='application/json',
        )
        self.assertEqual(duplicate.json()['imported'], 0)
        self.assertEqual(DashboardTask.objects.count(), 1)

    def test_label_override_migrates_from_session_and_survives_new_browser(self):
        session = self.client.session
        session['label_overrides'] = {
            f'p{self.product.pk}': {
                'name': 'Special print name', 'price': '8.75', 'barcode': '',
            },
        }
        session.save()

        self.client.get(reverse('label_printing'))
        override = LabelPrintOverride.objects.get(user=self.user, product=self.product)
        self.assertEqual(override.name, 'Special print name')
        self.assertEqual(override.price, Decimal('8.75'))
        self.assertTrue(override.barcode_overridden)
        self.assertEqual(override.barcode, '')
        self.assertNotIn('label_overrides', self.client.session)

        self.client.logout()
        second_browser = Client()
        second_browser.force_login(self.user)
        response = second_browser.get(reverse('label_printing'))
        self.assertContains(response, 'Special print name')

    def test_queue_override_cascades_when_queue_item_is_removed(self):
        queue_item = LabelQueueItem.objects.create(user=self.user, product=self.product)
        self.client.post(reverse('label_printing'), {
            'save_label_override': '1',
            'override_key': f'q{queue_item.pk}',
            'override_name': 'Queued custom name',
            'override_price': '10.00',
            'override_barcode': '999',
        })
        self.assertTrue(LabelPrintOverride.objects.filter(queue_item=queue_item).exists())

        self.client.post(reverse('label_printing'), {'remove_item': str(queue_item.pk)})
        self.assertFalse(LabelPrintOverride.objects.filter(user=self.user).exists())

    def test_purchase_deadline_is_stored_on_order_and_reset_is_recorded(self):
        session = self.client.session
        session['cart'] = {
            str(self.product.pk): {
                'name': self.product.name,
                'price': str(self.product.price),
                'quantity': 1,
            },
        }
        session.save()

        self.client.get(reverse('create_order'))
        order = Order.objects.get(user=self.user, submitted=False)
        first_deadline = order.draft_expires_at
        self.assertIsNotNone(first_deadline)
        self.assertIsNotNone(order.last_timer_reset_at)

        response = self.client.post(reverse('create_order'), {'action': 'reset_order_timer'})
        self.assertTrue(response.json()['ok'])
        order.refresh_from_db()
        self.assertGreaterEqual(order.draft_expires_at, first_deadline)
        self.assertEqual(order.timer_reset_count, 1)

        self.client.logout()
        second_browser = Client()
        second_browser.force_login(self.user)
        second_browser.get(reverse('create_order'))
        order.refresh_from_db()
        self.assertEqual(Order.objects.filter(user=self.user, submitted=False).count(), 1)
        self.assertEqual(order.timer_reset_count, 1)

    def test_add_by_id_writes_purchase_draft_before_redirect_finishes(self):
        response = self.client.post(
            reverse('add_product_by_id', args=[self.product.pk]),
            {'quantity': '2'},
        )
        self.assertEqual(response.status_code, 302)
        order = Order.objects.get(user=self.user, submitted=False)
        self.assertEqual(order.draft_cart[str(self.product.pk)]['quantity'], 2)
        self.assertIsNotNone(order.draft_expires_at)
        self.assertEqual(self.client.session['order_id'], order.pk)

    def _create_plan(self):
        response = self.client.post(
            reverse('supplier_order_plan'),
            data=json.dumps({
                'action': 'create',
                'seq': ['mck', 'kf'],
                'items': [{
                    'product_id': self.product.pk,
                    'name': 'Database Product',
                    'barcode': '123456',
                    'quantity': 3,
                }],
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        return SupplierOrderPlan.objects.get(pk=response.json()['plan']['id'])

    def test_supplier_plan_and_item_results_are_durable(self):
        plan = self._create_plan()
        self.client.logout()
        second_browser = Client()
        second_browser.force_login(self.user)
        saved = second_browser.get(reverse('supplier_order_plan')).json()['plan']
        self.assertEqual(saved['id'], plan.pk)
        self.assertEqual(saved['items'][0]['quantity'], 3)

        run = SupplierOrderRun.objects.create(
            plan=plan, created_by=self.user, vendor='mck', total=1,
        )
        row = SupplierOrderRunItem.objects.create(
            run=run, product=self.product, product_name=self.product.name,
            barcode=self.product.barcode, quantity_requested=3, position=0,
        )
        status = DatabaseRunStatus('mck', run.pk)
        item = status.pending_items()[0]
        status.record_result(item, True, 'added x3')

        row.refresh_from_db()
        self.assertEqual(row.outcome, SupplierOrderRunItem.OUTCOME_ADDED)
        self.assertEqual(row.reason, 'added x3')
        self.assertIsNotNone(row.processed_at)
        self.assertEqual(status.payload()['added'][0]['product_id'], self.product.pk)

        second_browser.post(
            reverse('order_control'),
            data=f'{{"vendor":"mck","action":"pause","plan_id":{plan.pk}}}',
            content_type='application/json',
        )
        run.refresh_from_db()
        self.assertTrue(run.pause_requested)

    @patch('app.views._launch_order_process', return_value=SimpleNamespace(pid=os.getpid()))
    def test_supplier_start_passes_only_a_database_run_id_to_worker(self, launch):
        plan = self._create_plan()
        response = self.client.post(
            reverse('mckesson_order_start'),
            data=json.dumps({
                'plan_id': plan.pk,
                'items': [{
                    'product_id': self.product.pk,
                    'name': 'Database Product',
                    'barcode': '123456',
                    'quantity': 4,
                }],
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        run = SupplierOrderRun.objects.get(pk=response.json()['run_id'])
        self.assertEqual(run.items.get().quantity_requested, 4)
        command = launch.call_args.args[0]
        self.assertIn('--run-id', command)
        self.assertNotIn('--status-file', command)
        self.assertNotIn('--control-file', command)
        self.assertNotIn('--items-file', command)
