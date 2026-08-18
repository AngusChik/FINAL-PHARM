import asyncio
import json
import importlib
import os
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.apps import apps
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import (
    Category, DashboardTask, LabelPrintOverride, LabelQueueItem, Order,
    Product, RecentlyPurchasedProduct, StockChange, SupplierOrderPlan,
    SupplierOrderRun, SupplierOrderRunItem,
)
from .supplier_orders import DatabaseRunStatus


class SupplierOrderAsyncDatabaseTests(TransactionTestCase):
    def test_status_database_calls_are_safe_inside_playwright_event_loop(self):
        async def exercise(vendor):
            status = DatabaseRunStatus(vendor)
            status.update(
                state=SupplierOrderRun.STATE_LOGIN,
                message='Waiting for supplier login',
            )
            pending = status.ensure_items(
                [{
                    'name': 'Async supplier item',
                    'barcode': '123456789',
                    'quantity': 2,
                }],
                [{
                    'name': 'Pre-skipped item',
                    'barcode': '',
                    'quantity': 1,
                    'reason': 'Missing barcode',
                }],
            )
            status.record_result(pending[0], True, 'added x2')
            return status.run.pk, status.control(), status.payload()

        for vendor in (
            SupplierOrderRun.VENDOR_MCKESSON,
            SupplierOrderRun.VENDOR_KOHLFRISCH,
        ):
            with self.subTest(vendor=vendor):
                run_id, control, payload = asyncio.run(exercise(vendor))
                run = SupplierOrderRun.objects.get(pk=run_id)

                self.assertEqual(run.state, SupplierOrderRun.STATE_LOGIN)
                self.assertEqual(run.items.count(), 2)
                self.assertFalse(control['pause_requested'])
                self.assertFalse(control['cancel_requested'])
                self.assertEqual(payload['added'][0]['name'], 'Async supplier item')
                self.assertEqual(payload['skipped'][0]['reason'], 'Missing barcode')


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
        self.assertNotIn('cart', self.client.session)
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
        self.assertNotIn('cart', self.client.session)

    def test_submitted_purchase_financials_survive_product_changes_and_deletion(self):
        self.product.price_per_unit = Decimal('5.00')
        self.product.taxable = True
        self.product.save(update_fields=['price_per_unit', 'taxable'])

        self.client.post(
            reverse('add_product_by_id', args=[self.product.pk]),
            {'quantity': '2'},
        )
        response = self.client.post(reverse('submit_order'))
        self.assertEqual(response.status_code, 302)

        order = Order.objects.get(user=self.user, submitted=True)
        line = order.details.get()
        self.assertEqual(order.financial_snapshot_source, Order.SNAPSHOT_CAPTURED)
        self.assertEqual(order.subtotal, Decimal('25.00'))
        self.assertEqual(order.discount_amount, Decimal('0.00'))
        self.assertEqual(order.tax, Decimal('3.25'))
        self.assertEqual(order.total_price, Decimal('28.25'))
        self.assertTrue(line.taxable_at_sale)
        self.assertEqual(line.cost_per_unit_at_sale, Decimal('5.00'))
        self.assertNotIn('cart', self.client.session)
        self.assertNotIn('order_id', self.client.session)

        self.product.price = Decimal('99.99')
        self.product.price_per_unit = Decimal('80.00')
        self.product.taxable = False
        self.product.save(update_fields=['price', 'price_per_unit', 'taxable'])
        self.product.delete()

        history = self.client.get(reverse('order_detail', args=[order.pk]))
        self.assertEqual(history.context['total_price_before_tax'], Decimal('25.00'))
        self.assertEqual(history.context['total_tax'], Decimal('3.25'))
        self.assertEqual(history.context['total_price_after_tax'], Decimal('28.25'))
        line.refresh_from_db()
        self.assertIsNone(line.product)
        self.assertTrue(line.taxable_at_sale)
        self.assertEqual(line.cost_per_unit_at_sale, Decimal('5.00'))

    def test_short_stock_bills_and_records_only_fulfilled_units(self):
        self.product.taxable = True
        self.product.save(update_fields=['taxable'])
        self.client.post(
            reverse('add_product_by_id', args=[self.product.pk]),
            {'quantity': '5'},
        )
        # Simulate stock being consumed at another terminal after this cart
        # was built but before it is submitted.
        self.product.quantity_in_stock = 2
        self.product.save(update_fields=['quantity_in_stock'])

        response = self.client.post(reverse('submit_order'))

        self.assertEqual(response.status_code, 302)
        order = Order.objects.get(user=self.user, submitted=True)
        line = order.details.get()
        self.assertEqual(line.quantity, 2)
        self.assertEqual(order.subtotal, Decimal('25.00'))
        self.assertEqual(order.tax, Decimal('3.25'))
        self.assertEqual(order.total_price, Decimal('28.25'))

        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 0)
        self.assertEqual(self.product.stock_sold, 2)
        self.assertEqual(self.product.stock_unfulfilled, 3)
        fulfilled = StockChange.objects.get(change_type='checkout')
        unfulfilled = StockChange.objects.get(change_type='checkout_unfulfilled')
        self.assertEqual((fulfilled.quantity, fulfilled.order_detail), (2, line))
        self.assertEqual((unfulfilled.quantity, unfulfilled.order_detail), (3, line))
        self.assertEqual(
            RecentlyPurchasedProduct.objects.get(product=self.product).quantity,
            2,
        )

    def test_zero_stock_records_shortfall_without_a_billed_order_line(self):
        self.product.taxable = True
        self.product.save(update_fields=['taxable'])
        self.client.post(
            reverse('add_product_by_id', args=[self.product.pk]),
            {'quantity': '3'},
        )
        # The final units were consumed elsewhere before submission.
        self.product.quantity_in_stock = 0
        self.product.save(update_fields=['quantity_in_stock'])

        response = self.client.post(reverse('submit_order'))

        self.assertEqual(response.status_code, 302)
        order = Order.objects.get(user=self.user, submitted=True)
        self.assertFalse(order.details.exists())
        self.assertEqual(order.subtotal, Decimal('0.00'))
        self.assertEqual(order.tax, Decimal('0.00'))
        self.assertEqual(order.total_price, Decimal('0.00'))

        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 0)
        self.assertEqual(self.product.stock_sold, 0)
        self.assertEqual(self.product.stock_unfulfilled, 3)
        self.assertFalse(StockChange.objects.filter(change_type='checkout').exists())
        unfulfilled = StockChange.objects.get(change_type='checkout_unfulfilled')
        self.assertEqual(unfulfilled.quantity, 3)
        self.assertIsNone(unfulfilled.order_detail)
        self.assertFalse(
            RecentlyPurchasedProduct.objects.filter(product=self.product).exists(),
        )

    def test_item_numbers_may_repeat_when_products_are_added(self):
        self.product.item_number = 'SHARED-ITEM'
        self.product.save(update_fields=['item_number'])
        response = self.client.post(reverse('new_product'), {
            'name': 'Second Database Product',
            'item_number': 'SHARED-ITEM',
            'brand': 'Generic',
            'barcode': '654321',
            'price': '9.99',
            'quantity_in_stock': '1',
            'description': '',
            'category': str(self.category.pk),
            'unit_size': '',
            'expiry_date': '',
            'taxable': 'on',
            'price_per_unit': '4.00',
            'status': 'on',
            'next': 'inventory_display',
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Product.objects.filter(item_number='SHARED-ITEM').count(), 2)

    def test_product_search_indexes_are_declared_on_model(self):
        self.assertEqual(
            {index.name for index in Product._meta.indexes},
            {
                'product_barcode_idx', 'product_name_idx',
                'product_stock_status_idx', 'product_cat_status_idx',
                'product_expiry_idx',
            },
        )

    def test_legacy_purchase_backfill_marks_derived_snapshot(self):
        order = Order.objects.create(user=self.user, submitted=True, seniors_discount=True)
        order.details.create(
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=2,
            price=Decimal('12.50'),
        )
        migration = importlib.import_module(
            'app.migrations.0052_durable_purchase_financials_and_product_indexes'
        )
        migration.backfill_purchase_financial_snapshots(apps, None)

        order.refresh_from_db()
        line = order.details.get()
        self.assertEqual(order.financial_snapshot_source, Order.SNAPSHOT_BACKFILLED)
        self.assertEqual(order.subtotal, Decimal('25.00'))
        self.assertEqual(order.discount_amount, Decimal('2.50'))
        self.assertEqual(order.tax, Decimal('2.93'))
        self.assertEqual(order.total_price, Decimal('25.43'))
        self.assertTrue(line.taxable_at_sale)

    def test_recent_purchase_shortfall_migration_repairs_only_owning_generation(self):
        migration = importlib.import_module(
            'app.migrations.0062_repair_recent_purchase_shortfalls'
        )
        base = timezone.now() - timedelta(days=10)

        def product(suffix):
            return Product.objects.create(
                name=f'RP migration {suffix}',
                price=Decimal('1.00'),
                barcode=f'RP-MIG-{suffix}',
                quantity_in_stock=0,
                category=self.category,
            )

        def ledger(target, change_type, quantity, timestamp):
            change = StockChange.objects.create(
                product=target,
                product_name=target.name,
                product_barcode=target.barcode,
                change_type=change_type,
                quantity=quantity,
                user=self.user,
            )
            StockChange.objects.filter(pk=change.pk).update(timestamp=timestamp)
            return change

        def recent(target, quantity, order_date, archived_at=None):
            row = RecentlyPurchasedProduct.objects.create(
                product=target,
                quantity=quantity,
                archived_at=archived_at,
            )
            RecentlyPurchasedProduct.objects.filter(pk=row.pk).update(
                order_date=order_date,
            )
            row.refresh_from_db()
            return row

        inflated = product('inflated')
        ledger(inflated, 'checkout', 3, base)
        ledger(
            inflated, 'checkout_unfulfilled', 2,
            base + timedelta(milliseconds=10),
        )
        inflated_row = recent(
            inflated, 5, base + timedelta(milliseconds=50),
        )

        already_correct = product('correct')
        ledger(already_correct, 'checkout', 3, base)
        ledger(
            already_correct, 'checkout_unfulfilled', 2,
            base + timedelta(milliseconds=10),
        )
        correct_row = recent(
            already_correct, 3, base + timedelta(milliseconds=50),
        )

        regenerated = product('regenerated')
        ledger(regenerated, 'checkout', 2, base)
        ledger(
            regenerated, 'checkout_unfulfilled', 1,
            base + timedelta(milliseconds=10),
        )
        old_row = recent(
            regenerated,
            3,
            base + timedelta(milliseconds=50),
            archived_at=base + timedelta(days=1),
        )
        ledger(regenerated, 'checkout', 4, base + timedelta(days=2))
        new_row = recent(
            regenerated, 4,
            base + timedelta(days=2, milliseconds=50),
        )

        missing_row = product('missing')
        ledger(missing_row, 'checkout_unfulfilled', 7, base)

        repairs = migration.plan_recent_purchase_repairs(apps)
        self.assertEqual(
            repairs,
            [
                {
                    'row_id': inflated_row.pk,
                    'product_id': inflated.pk,
                    'before': 5,
                    'after': 3,
                    'deduction': 2,
                    'archived': False,
                },
                {
                    'row_id': old_row.pk,
                    'product_id': regenerated.pk,
                    'before': 3,
                    'after': 2,
                    'deduction': 1,
                    'archived': True,
                },
            ],
        )

        migration.repair_recent_purchase_shortfalls(apps, None)
        inflated_row.refresh_from_db()
        correct_row.refresh_from_db()
        old_row.refresh_from_db()
        new_row.refresh_from_db()
        self.assertEqual(inflated_row.quantity, 3)
        self.assertEqual(correct_row.quantity, 3)
        self.assertEqual(old_row.quantity, 2)
        self.assertEqual(new_row.quantity, 4)
        self.assertFalse(
            RecentlyPurchasedProduct.objects.filter(product=missing_row).exists()
        )
        self.assertEqual(migration.plan_recent_purchase_repairs(apps), [])

    def test_recent_purchase_generation_match_skips_restoration_overlap(self):
        migration = importlib.import_module(
            'app.migrations.0062_repair_recent_purchase_shortfalls'
        )
        event_at = timezone.now() - timedelta(days=1)
        generations = [
            SimpleNamespace(
                order_date=event_at - timedelta(days=2), archived_at=None,
            ),
            SimpleNamespace(
                order_date=event_at - timedelta(hours=1),
                archived_at=event_at + timedelta(hours=1),
            ),
        ]

        self.assertIsNone(
            migration.match_recent_purchase_generation(generations, event_at)
        )

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
