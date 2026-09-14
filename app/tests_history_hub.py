"""Integration coverage for the central, read-only History browser."""

import time
from datetime import timedelta
from urllib.parse import parse_qs, urlencode, urlsplit

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .mixins import PASSKEY_SESSION_KEY
from .models import (
    CheckinSession, CheckoutOrder, CheckoutOrderItem, DailyReportArchive,
    DeliveryCheckIn, InventoryAuditIssue, InventoryAuditRun, Item, LabelSession,
    LabelSessionItem, LoginAudit, Order, OrderDetail, OrderingSheetEntry, PrescriptionDrug,
    PrescriptionDrugChange, PrescriptionDrugLearningRecord,
    PrescriptionDrugRequestRevision, Product, RecentlyPurchasedProduct, StockChange,
    SupplierPurchaseOrder, SupplierPurchaseOrderLine, TransactionCorrection,
    TransactionCorrectionLine, TransactionCorrectionUndo, UserAction,
)


@override_settings(AXES_ENABLED=False)
class HistoryHubTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.staff = get_user_model().objects.create_user(
            username='history-hub-admin', is_staff=True,
        )
        cls.other = get_user_model().objects.create_user(
            username='history-hub-other', is_staff=True,
        )
        cls.pu = get_user_model().objects.create_user(username='history-hub-pu')
        cls.now = timezone.now()
        cls.product = Product.objects.create(
            name='Original history product', barcode='700001', price='4.50',
            quantity_in_stock=6,
        )
        cls.session_record = CheckinSession.objects.create(
            user=cls.staff, scanned_by='AB', ended_at=cls.now,
            note='Receiving note retained',
        )
        cls.stock = StockChange.objects.create(
            user=cls.staff, session=cls.session_record, product=cls.product,
            product_name='Original history product', product_barcode='700001',
            change_type='checkin', quantity=6, note='Received six units',
        )
        cls.expired = StockChange.objects.create(
            user=cls.staff, product=cls.product,
            product_name='Expired history product', product_barcode='700001',
            change_type='expired', quantity=-1, note='Retired expired unit',
        )
        cls.login = LoginAudit.objects.create(
            user=cls.staff, username=cls.staff.username, success=False,
            ip_address='127.0.0.2',
        )
        cls.action = UserAction.objects.create(
            user=cls.staff, action='edit_product', target='Recorded product edit',
            detail='Original edit reason',
        )
        cls.transaction = Order.objects.create(
            user=cls.staff, submitted=True, subtotal='9.00', tax='1.17',
            total_price='10.17', financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        OrderDetail.objects.create(
            order=cls.transaction, product=cls.product,
            product_name='Sale-time history product', product_barcode='700002',
            quantity=2, price='4.50',
        )
        cls.checkout = CheckoutOrder.objects.create(
            user=cls.staff, status=CheckoutOrder.STATUS_SUBMITTED,
            submitted_at=cls.now, subtotal='4.50', tax='0.59', total_price='5.09',
        )
        CheckoutOrderItem.objects.create(
            checkout=cls.checkout, product=cls.product,
            product_name='Checkout-time history product', product_barcode='700003',
            quantity=1, price='4.50',
        )
        cls.labels = LabelSession.objects.create(
            user=cls.staff, label_count=2, note='Saved label run',
        )
        LabelSessionItem.objects.create(
            session=cls.labels, product=cls.product,
            product_name='Print-time history product', product_barcode='700004',
            product_price='4.50', qty=2,
        )
        cls.report = DailyReportArchive.objects.create(
            report_date=timezone.localdate(), pdf=b'%PDF-history-test',
            summary='Retained daily report summary',
            snapshot_data={'summary': {'revenue_today': '10.17', 'units_sold': 2}},
        )
        cls.drug = PrescriptionDrug.objects.create(
            name='History medicine', brand='History brand', strength='10 mg',
        )
        cls.request_record = PrescriptionDrugLearningRecord.objects.create(
            drug=cls.drug, source_name='Original medicine request',
            requested_at=cls.now, quantity_needed='2', quantity_unit='bottle',
            source_snapshot={
                'name': 'Original medicine request', 'status': 'pending',
                'quantity_needed': '2 bottles', 'quantity_remaining': '1 bottle',
                'initials': 'AB', 'is_deleted': True,
            },
        )
        PrescriptionDrugRequestRevision.objects.create(
            learning_record=cls.request_record, drug=cls.drug,
            snapshot={'name': 'Previous medicine request', 'quantity_needed': '1 bottle'},
        )
        cls.drug_change = PrescriptionDrugChange.objects.create(
            drug=cls.drug, entity_type='drug', entity_id=cls.drug.pk,
            action='update', actor=cls.staff, actor_label='Recorded staff name',
            before={'strength': '5 mg'}, after={'strength': '10 mg'},
        )
        cls.delivery = DeliveryCheckIn.objects.create(
            barcode='DEL-HISTORY', first_name='History', last_name='Delivery',
            checked_out_at=cls.now, comment='Delivery handover note',
        )
        cls.audit = InventoryAuditRun.objects.create(
            created_by=cls.staff, status=InventoryAuditRun.STATUS_ISSUES,
            completed_at=cls.now, issue_count=1, summary='One retained audit issue',
        )
        InventoryAuditIssue.objects.create(
            run=cls.audit, product=cls.product, product_name='Audit-time product',
            code='test_history_issue', title='Recorded audit finding',
            detail='Expected six units', expected_value='6', actual_value='5',
        )
        cls.supplier_order = SupplierPurchaseOrder.objects.create(
            supplier=SupplierPurchaseOrder.SUPPLIER_OTHER,
            supplier_name='History supplier', confirmation_number='HISTORY-PO-1',
            status=SupplierPurchaseOrder.STATUS_RECEIVED, created_by=cls.staff,
            notes='Supplier receipt note',
        )
        SupplierPurchaseOrderLine.objects.create(
            purchase_order=cls.supplier_order, product=cls.product,
            product_name='Supplier-time product', product_barcode='700005',
            quantity_ordered=3, quantity_received=3, unit_cost='2.00',
        )

    def setUp(self):
        self.client.force_login(self.staff)

    def detail_url(self, kind, record):
        return reverse('history_detail', args=[kind, record.pk])

    def records_by_kind(self):
        return {
            'stock': self.stock, 'expired': self.expired, 'scans': self.stock,
            'checkins': self.session_record, 'transactions': self.transaction,
            'checkouts': self.checkout, 'labels': self.labels, 'reports': self.report,
            'prescription_requests': self.request_record,
            'prescription_changes': self.drug_change, 'deliveries': self.delivery,
            'inventory_audits': self.audit, 'supplier_orders': self.supplier_order,
        }

    def test_each_history_category_lists_records_and_opens_a_detail_table(self):
        for kind, record in self.records_by_kind().items():
            with self.subTest(kind=kind):
                response = self.client.get(reverse('history'), {'kind': kind})
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, '<table')
                row = next(row for row in response.context['rows'] if row['id'] == record.pk)
                self.assertContains(response, self.detail_url(kind, record))
                detail = self.client.get(row['url'])
                self.assertEqual(detail.status_code, 200)
                self.assertTrue(detail.context['detail']['fields'])

    def test_default_activity_and_old_address_both_link_to_individual_events(self):
        for name in ('history', 'activity_log'):
            with self.subTest(route=name):
                response = self.client.get(reverse(name))
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'History')
                for kind, record in (('login', self.login), ('action', self.action), ('stock', self.stock)):
                    self.assertContains(response, self.detail_url(kind, record))
                    self.assertEqual(self.client.get(self.detail_url(kind, record)).status_code, 200)

    def test_activity_search_opens_the_matching_original_action(self):
        response = self.client.get(reverse('history'), {'q': 'Original edit reason'})
        self.assertEqual(response.context['page_obj'].paginator.count, 1)
        row = response.context['rows'][0]
        self.assertEqual(urlsplit(row['url']).path, self.detail_url('action', self.action))
        detail = self.client.get(row['url'])
        self.assertContains(detail, 'Recorded product edit')
        self.assertContains(detail, 'Original edit reason')

    def test_anonymous_and_locked_sessions_cannot_browse_or_open_history(self):
        urls = (reverse('history'), reverse('activity_log'), self.detail_url('labels', self.labels))
        self.client.logout()
        for url in urls:
            with self.subTest(access='anonymous', url=url):
                self.assertRedirects(self.client.get(url), reverse('login'), fetch_redirect_response=False)
        self.client.force_login(self.pu)
        for url in urls:
            with self.subTest(access='locked', url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertEqual(urlsplit(response.url).path, reverse('passkey_unlock'))
        session = self.client.session
        session[PASSKEY_SESSION_KEY] = time.time()
        session.save()
        self.assertEqual(self.client.get(reverse('history')).status_code, 200)
        self.assertEqual(self.client.get(self.detail_url('stock', self.stock)).status_code, 200)

    def test_label_history_keeps_owner_scope_in_list_and_direct_detail(self):
        other_labels = LabelSession.objects.create(
            user=self.other, label_count=1, note='Other staff private label run',
        )
        LabelSessionItem.objects.create(
            session=other_labels, product_name='Other staff label content', product_price='1.00',
        )
        response = self.client.get(reverse('history'), {'kind': 'labels'})
        self.assertEqual([row['id'] for row in response.context['rows']], [self.labels.pk])
        self.assertNotContains(response, 'Other staff private label run')
        self.assertEqual(self.client.get(self.detail_url('labels', other_labels)).status_code, 404)

    def test_archived_records_remain_browsable_and_readable(self):
        records = {
            'checkins': self.session_record, 'labels': self.labels,
            'reports': self.report, 'deliveries': self.delivery,
            'supplier_orders': self.supplier_order,
        }
        for record in records.values():
            record.archived_at = self.now
            record.save(update_fields=['archived_at'])
        self.transaction.is_deleted = True
        self.transaction.deleted_at = self.now
        self.transaction.save(update_fields=['is_deleted', 'deleted_at'])
        self.checkout.hidden_from_history = True
        self.checkout.save(update_fields=['hidden_from_history'])
        records.update(transactions=self.transaction, checkouts=self.checkout)
        for kind, record in records.items():
            with self.subTest(kind=kind):
                response = self.client.get(reverse('history'), {'kind': kind})
                self.assertIn(record.pk, [row['id'] for row in response.context['rows']])
                retained_status = 'Hidden from checkout history' if kind == 'checkouts' else 'Archived'
                self.assertContains(self.client.get(self.detail_url(kind, record)), retained_status)

    def test_archived_records_category_opens_each_recovery_record_type(self):
        archived = {
            'product': self.product, 'checkin': self.session_record,
            'delivery': self.delivery, 'supplier_order': self.supplier_order,
        }
        for record in archived.values():
            record.archived_at = self.now
            record.archived_by = self.staff
            record.save(update_fields=['archived_at', 'archived_by'])
        self.transaction.is_deleted = True
        self.transaction.deleted_at = self.now
        self.transaction.deleted_by = self.staff
        self.transaction.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])
        archived['order'] = self.transaction
        archived['ordering'] = OrderingSheetEntry.objects.create(
            name='Archived ordering request', initials='AB', is_deleted=True,
            deleted_at=self.now, deleted_by=self.staff,
        )
        archived['recent_purchase'] = RecentlyPurchasedProduct.objects.create(
            product=self.product, quantity=2, archived_at=self.now, archived_by=self.staff,
        )
        archived['special_order'] = Item.objects.create(
            first_name='Test', last_name='Customer', item_name='Archived special item',
            size='na', side='na', item_number='SPECIAL-HISTORY', phone_number='',
            archived_at=self.now, archived_by=self.staff,
        )
        response = self.client.get(reverse('history'), {'kind': 'recovery'})
        self.assertEqual(response.status_code, 200)
        rows = response.context['rows']
        self.assertEqual({row['kind'] for row in rows}, {f'recovery-{kind}' for kind in archived})
        self.assertEqual(response.context['page_obj'].paginator.count, len(archived))
        for kind, record in archived.items():
            with self.subTest(kind=kind):
                row = next(row for row in rows if row['kind'] == f'recovery-{kind}')
                self.assertEqual(urlsplit(row['url']).path, self.detail_url(f'recovery-{kind}', record))
                detail = self.client.get(row['url'])
                self.assertContains(detail, 'Archived')
                self.assertEqual(parse_qs(urlsplit(detail.context['history_return']).query)['kind'], ['recovery'])
        active = Product.objects.create(name='Active history product', price='1.00')
        self.assertEqual(self.client.get(self.detail_url('recovery-product', active)).status_code, 404)

    def test_recovery_pagination_keeps_mixed_sources_and_legacy_missing_dates(self):
        orders = Order.objects.bulk_create([
            Order(user=self.staff, submitted=True, is_deleted=True,
                  deleted_at=self.now, deleted_by=self.staff)
            for _ in range(65)
        ])
        requests = OrderingSheetEntry.objects.bulk_create([
            OrderingSheetEntry(name=f'Archived page request {i}', initials='AB',
                               is_deleted=True, deleted_at=self.now, deleted_by=self.staff)
            for i in range(65)
        ])
        legacy_order = Order.objects.create(submitted=True, is_deleted=True, deleted_by=self.staff)
        legacy_request = OrderingSheetEntry.objects.create(
            name='Legacy removal without date', initials='AB', is_deleted=True, deleted_by=self.staff,
        )
        expected = (
            {('recovery-order', record.pk) for record in [*orders, legacy_order]}
            | {('recovery-ordering', record.pk) for record in [*requests, legacy_request]}
        )
        seen = []
        for number in (1, 2, 3):
            response = self.client.get(reverse('history'), {'kind': 'recovery', 'page': number})
            self.assertEqual(response.context['page_obj'].paginator.count, 132)
            self.assertEqual(response.context['page_obj'].number, number)
            rows = response.context['rows']
            seen.extend((row['kind'], row['id']) for row in rows)
            if number < 3:
                self.assertTrue(all(row['timestamp'] is not None for row in rows))
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), expected)
        self.assertEqual({(row['kind'], row['id']) for row in rows[-2:]}, {
            ('recovery-order', legacy_order.pk), ('recovery-ordering', legacy_request.pk),
        })
        self.assertTrue(all(row['timestamp'] is None for row in rows[-2:]))
        repeated = self.client.get(reverse('history'), {'kind': 'recovery', 'page': 3})
        self.assertEqual([(row['kind'], row['id']) for row in repeated.context['rows']], seen[100:])
        self.assertEqual(self.client.get(rows[-1]['url']).status_code, 200)
        dated = self.client.get(reverse('history'), {
            'kind': 'recovery', 'date_from': timezone.localtime(self.now).date().isoformat(),
        })
        self.assertEqual(dated.context['page_obj'].paginator.count, 130)

    def test_saved_product_names_and_barcodes_survive_later_inventory_edits(self):
        self.product.name = 'Changed current catalogue name'
        self.product.barcode = '799999'
        self.product.price = '99.00'
        self.product.save(update_fields=['name', 'barcode', 'price'])
        expected = (
            ('stock', self.stock, 'Original history product', '700001'),
            ('transactions', self.transaction, 'Sale-time history product', '700002'),
            ('checkouts', self.checkout, 'Checkout-time history product', '700003'),
            ('labels', self.labels, 'Print-time history product', '700004'),
            ('supplier_orders', self.supplier_order, 'Supplier-time product', '700005'),
        )
        for kind, record, name, barcode in expected:
            with self.subTest(kind=kind):
                response = self.client.get(self.detail_url(kind, record))
                self.assertContains(response, name)
                self.assertContains(response, barcode)
                self.assertNotContains(response, 'Changed current catalogue name')
        stock_list = self.client.get(reverse('history'), {'kind': 'stock', 'q': 'Original history product'})
        self.assertIn(self.stock.pk, [row['id'] for row in stock_list.context['rows']])
        self.assertContains(stock_list, 'Original history product')
        activity_list = self.client.get(reverse('history'), {'kind': 'activity', 'type': 'all_stock'})
        self.assertContains(activity_list, 'Original history product')
        self.assertNotContains(activity_list, 'Changed current catalogue name')

    def test_deleted_product_does_not_break_record_detail(self):
        self.product.delete()
        for kind, record in (
            ('stock', self.stock), ('transactions', self.transaction),
            ('checkouts', self.checkout), ('labels', self.labels),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(self.client.get(self.detail_url(kind, record)).status_code, 200)

    def test_stock_history_distinguishes_recorded_quantity_from_stock_effect(self):
        correction = TransactionCorrection.objects.create(
            order=self.transaction, correction_type='void', reason='Stock effect audit',
        )
        source = self.transaction.details.get()
        restocked = TransactionCorrectionLine.objects.create(
            correction=correction, order_detail=source, product_name=source.product_name,
            quantity=1, unit_price=source.price, disposition='restock',
        )
        damaged = TransactionCorrectionLine.objects.create(
            correction=correction, order_detail=source, product_name=source.product_name,
            quantity=1, unit_price=source.price, disposition='damaged',
        )
        cases = (
            ('expired', 3, None, '-3 units'),
            ('checkout', 2, None, '-2 units'),
            ('checkout_unfulfilled', 4, None, '4 units · no stock change'),
            ('return_no_restock', 2, None, '2 units · no stock change'),
            ('void', 1, restocked, '+1 units'),
            ('void', 1, damaged, '1 units · no stock change'),
            ('correction_undo', 1, restocked, '-1 units'),
            ('correction_undo', 1, damaged, '1 units · no stock change'),
            ('lot_reassignment', 2, None, '2 units moved between lots'),
        )
        expected = {}
        for change_type, quantity, correction_line, effect in cases:
            record = StockChange.objects.create(
                product_name='Stock effect case', change_type=change_type,
                quantity=quantity, correction_line=correction_line,
            )
            expected[record.pk] = effect
            with self.subTest(change_type=change_type, effect=effect):
                response = self.client.get(self.detail_url('stock', record))
                fields = {field['label']: field['value'] for field in response.context['detail']['fields']}
                self.assertEqual(fields['Recorded quantity'], str(quantity))
                self.assertEqual(fields['Stock effect'], effect)
        response = self.client.get(reverse('history'), {'kind': 'stock', 'q': 'Stock effect case'})
        self.assertEqual(len(response.context['rows']), len(expected))
        for row in response.context['rows']:
            self.assertTrue(row['summary'].startswith(expected[row['id']]))

    def test_scan_history_includes_receiving_additions_and_removals(self):
        records = [
            StockChange.objects.create(
                user=self.staff, session=self.session_record,
                product_name=f'Scan coverage {change_type}', change_type=change_type, quantity=1,
            )
            for change_type in ('checkin', 'error_add', 'error_subtract', 'checkin_delete1')
        ]
        response = self.client.get(reverse('history'), {'kind': 'scans', 'q': 'Scan coverage'})
        self.assertEqual({row['id'] for row in response.context['rows']}, {record.pk for record in records})
        for record in records:
            with self.subTest(change_type=record.change_type):
                self.assertEqual(self.client.get(self.detail_url('scans', record)).status_code, 200)

    def test_transaction_detail_keeps_original_money_and_shows_active_adjustment(self):
        source = self.transaction.details.get()
        source.taxable_at_sale = True
        source.save(update_fields=['taxable_at_sale'])
        correction = TransactionCorrection.objects.create(
            order=self.transaction, correction_type='void', reason='One unit corrected',
            adjustment_amount='5.08',
        )
        TransactionCorrectionLine.objects.create(
            correction=correction, order_detail=source, product_name=source.product_name,
            quantity=1, unit_price=source.price, disposition='restock',
        )
        response = self.client.get(self.detail_url('transactions', self.transaction))
        fields = {field['label']: field['value'] for field in response.context['detail']['fields']}
        self.assertEqual(fields['Original total'], '$10.17')
        self.assertEqual(fields['Original subtotal'], '$9.00')
        self.assertEqual(fields['Current total'], '$5.09')
        self.assertEqual(fields['Current subtotal'], '$4.50')
        self.assertContains(response, 'One unit corrected')
        TransactionCorrectionUndo.objects.create(correction=correction, reason='Correction entered by mistake')
        response = self.client.get(self.detail_url('transactions', self.transaction))
        fields = {field['label']: field['value'] for field in response.context['detail']['fields']}
        self.assertEqual(fields['Original total'], '$10.17')
        self.assertNotIn('Current total', fields)
        self.assertContains(response, 'Correction entered by mistake')
        self.transaction.refresh_from_db()
        self.assertEqual(str(self.transaction.total_price), '10.17')

    def test_legacy_transaction_money_uses_saved_lines_when_snapshot_is_missing(self):
        legacy = Order.objects.create(user=self.staff, submitted=True, total_price='99.00')
        OrderDetail.objects.create(
            order=legacy, product=self.product, product_name='Legacy sale product',
            quantity=2, price='3.00', taxable_at_sale=True,
        )
        response = self.client.get(self.detail_url('transactions', legacy))
        fields = {field['label']: field['value'] for field in response.context['detail']['fields']}
        self.assertEqual(fields['Original subtotal'], '$6.00')
        self.assertEqual(fields['Original total'], '$6.78')
        self.assertEqual(fields['Financial basis'], 'Reconstructed from saved sale items')

    def test_search_staff_and_dates_apply_before_pagination(self):
        stamp = self.now - timedelta(days=12)
        day = timezone.localtime(stamp).date().isoformat()
        matches = StockChange.objects.bulk_create([
            StockChange(user=self.staff, product_name=f'Needle history {i}', change_type='return', quantity=1)
            for i in range(65)
        ])
        StockChange.objects.filter(pk__in=[row.pk for row in matches]).update(timestamp=stamp)
        StockChange.objects.create(
            user=self.other, product_name='Needle history other user', change_type='return', quantity=1,
        )
        params = {
            'kind': 'stock', 'q': 'Needle history', 'user': self.staff.username,
            'date_from': day, 'date_to': day, 'page': 2,
        }
        response = self.client.get(reverse('history'), params)
        self.assertEqual(response.status_code, 200)
        page = response.context['page_obj']
        self.assertEqual(page.paginator.count, 65)
        self.assertEqual(page.number, 2)
        self.assertEqual(len(response.context['rows']), 15)
        self.assertTrue(all(row['title'].startswith('Needle history') for row in response.context['rows']))
        detail = self.client.get(response.context['rows'][0]['url'])
        return_params = parse_qs(urlsplit(detail.context['history_return']).query)
        self.assertEqual(return_params, {key: [str(value)] for key, value in params.items()})
        no_matches = self.client.get(reverse('history'), {**params, 'date_from': (timezone.localtime(stamp).date() + timedelta(days=1)).isoformat()})
        self.assertEqual(no_matches.context['page_obj'].paginator.count, 0)

    def test_detail_return_keeps_history_filters_and_rejects_external_destinations(self):
        history_return = reverse('history') + '?' + urlencode({
            'kind': 'stock', 'q': 'Original history', 'page': 2,
        })
        response = self.client.get(self.detail_url('stock', self.stock), {'return_to': history_return})
        self.assertEqual(response.context['history_return'], history_return)
        for unsafe in ('https://example.invalid/history/', '//example.invalid/history/', '/purchase/'):
            with self.subTest(return_to=unsafe):
                response = self.client.get(self.detail_url('stock', self.stock), {'return_to': unsafe})
                target = response.context['history_return']
                self.assertEqual(urlsplit(target).path, reverse('history'))
                self.assertEqual(parse_qs(urlsplit(target).query).get('kind'), ['stock'])

    def test_invalid_dates_show_filter_errors_and_cannot_export_unfiltered_records(self):
        invalid_filters = (
            {'date_from': 'not-a-date'},
            {'date_to': '2026-02-31'},
            {'date_from': '2026-03-02', 'date_to': '2026-03-01'},
        )
        for filters in invalid_filters:
            for kind in ('activity', 'stock', 'recovery'):
                with self.subTest(kind=kind, filters=filters):
                    response = self.client.get(reverse('history'), {'kind': kind, **filters})
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.context['filter_errors'])
                    self.assertEqual(response.context['rows'], [])
                    self.assertEqual(response.context['page_obj'].paginator.count, 0)
                    self.assertContains(response, 'role="alert"')
            export = self.client.get(reverse('history'), {'export': 'pdf', **filters})
            self.assertEqual(export.status_code, 400)
            self.assertNotEqual(export['Content-Type'], 'application/pdf')

    def test_detail_rejects_unknown_kinds_missing_rows_and_wrong_subcategory(self):
        for kind, pk in (('not_a_history_type', self.stock.pk), ('stock', 99999999), ('expired', self.stock.pk)):
            with self.subTest(kind=kind, pk=pk):
                self.assertEqual(self.client.get(reverse('history_detail', args=[kind, pk])).status_code, 404)

    def test_drafts_and_onsite_deliveries_are_not_completed_history(self):
        drafts = {
            'transactions': Order.objects.create(user=self.staff),
            'checkouts': CheckoutOrder.objects.create(user=self.staff),
            'deliveries': DeliveryCheckIn.objects.create(
                barcode='STILL-ONSITE', first_name='Still', last_name='Onsite',
            ),
        }
        for kind, record in drafts.items():
            with self.subTest(kind=kind):
                response = self.client.get(reverse('history'), {'kind': kind})
                self.assertNotIn(record.pk, [row['id'] for row in response.context['rows']])
                self.assertEqual(self.client.get(self.detail_url(kind, record)).status_code, 404)

    def test_history_endpoints_do_not_accept_record_changes(self):
        for url in (reverse('history'), self.detail_url('stock', self.stock)):
            with self.subTest(url=url):
                self.assertEqual(self.client.post(url, {'quantity': 999, 'delete': '1'}).status_code, 405)
        self.stock.refresh_from_db()
        self.assertEqual(self.stock.quantity, 6)
