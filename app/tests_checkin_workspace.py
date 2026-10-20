from datetime import date
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import (
    Category, CheckinReceivingDraft, CheckinSession, InventoryCountLine,
    Product, ProductLot, ProductLotMovement, StockChange,
)


@override_settings(AXES_ENABLED=False)
class CheckinWorkspaceTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user('workspace-user', password='test', is_staff=True)
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='Workspace')
        self.product = Product.objects.create(name='Vitamin D3', price=Decimal('12.49'), barcode='WORKSPACE01', category=self.category, quantity_in_stock=14)
        self.lots = [
            ProductLot.objects.create(product=self.product, lot_number='LOT-A', expiry_date=date(2031,1,31), quantity_on_hand=5),
            ProductLot.objects.create(product=self.product, lot_number='LOT-B', expiry_date=date(2032,1,31), quantity_on_hand=6),
            ProductLot.objects.create(product=self.product, lot_number=ProductLot.UNASSIGNED, quantity_on_hand=3),
        ]
        self.session = CheckinSession.objects.create(user=self.user, scanned_by='Workspace tester')
        self.url = reverse('checkin_edit_product', args=[self.session.pk, self.product.pk])

    def payload(self):
        return {
            'name': self.product.name, 'price': '12.49', 'category': str(self.category.pk),
            'barcode': self.product.barcode, 'item_number': '', 'description': '',
            'brand': '', 'price_per_unit': '', 'unit_size': '', 'status': 'on',
            'quantity_in_stock': '999', 'inline_stock_baseline': '14',
            'lot_id': [str(l.pk) for l in self.lots],
            'lot_number': [l.lot_number for l in self.lots],
            'lot_expiry': [l.expiry_date.strftime('%d-%m-%Y') if l.expiry_date else '' for l in self.lots],
            'lot_quantity': [str(l.quantity_on_hand) for l in self.lots],
            'lot_original_number': [l.lot_number for l in self.lots],
            'lot_original_expiry': [l.expiry_date.strftime('%d-%m-%Y') if l.expiry_date else '' for l in self.lots],
            'lot_original_quantity': [str(l.quantity_on_hand) for l in self.lots],
            'lot_removed': ['0'] * len(self.lots),
        }

    def test_rename_retains_identity_history_and_drafts(self):
        lot = self.lots[2]
        change = StockChange.objects.create(product=self.product, quantity=3, change_type='checkin', user=self.user, session=self.session)
        movement = ProductLotMovement.objects.create(stock_change=change, lot=lot, lot_number=lot.lot_number, quantity=3, direction=ProductLotMovement.DIRECTION_IN)
        draft = CheckinReceivingDraft.objects.create(session=self.session, product=self.product, existing_lot=lot, lot_number=lot.lot_number)
        payload = self.payload()
        payload['lot_number'][2] = 'NAMED'
        payload['lot_expiry'][2] = '30-06-2033'
        response = self.client.post(self.url, payload)
        self.assertEqual(response.status_code, 302)
        lot.refresh_from_db(); draft.refresh_from_db(); movement.refresh_from_db()
        self.assertEqual(lot.lot_number, 'NAMED')
        self.assertEqual(draft.existing_lot_id, lot.pk)
        self.assertEqual(draft.lot_number, 'NAMED')
        self.assertEqual(movement.lot_id, lot.pk)
        self.assertEqual(movement.lot_number, ProductLot.UNASSIGNED)
        self.assertEqual(StockChange.objects.count(), 1)

    def test_lot_corrections_audit_both_sides_with_session_without_duplicates(self):
        payload = self.payload()
        payload['lot_quantity'] = ['3','8','3']
        response = self.client.post(self.url, payload)
        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 14)
        changes = list(StockChange.objects.filter(session=self.session).order_by('change_type'))
        self.assertEqual([(c.change_type, c.quantity) for c in changes], [('error_add',2), ('error_subtract',2)])
        self.assertEqual(ProductLotMovement.objects.count(), 2)
        self.assertEqual(set(ProductLotMovement.objects.values_list('lot_id',flat=True)), {self.lots[0].pk,self.lots[1].pk})

    def test_removal_archives_identity_and_clears_receiving_selection(self):
        lot = self.lots[0]
        draft = CheckinReceivingDraft.objects.create(session=self.session, product=self.product, existing_lot=lot, lot_number=lot.lot_number, lot_expiry=lot.expiry_date)
        payload = self.payload(); payload['lot_removed'][0] = '1'
        self.assertEqual(self.client.post(self.url, payload).status_code, 302)
        lot.refresh_from_db(); draft.refresh_from_db(); self.product.refresh_from_db()
        self.assertIsNotNone(lot.archived_at)
        self.assertEqual(lot.quantity_on_hand, 0)
        self.assertIsNone(draft.existing_lot_id)
        self.assertEqual(self.product.quantity_in_stock, 9)
        self.assertEqual(StockChange.objects.get().quantity, 5)
        self.assertEqual(ProductLotMovement.objects.get().lot_id, lot.pk)

    def test_new_lot_and_unassigned_quantity_change(self):
        payload = self.payload()
        for name,value in {'lot_id':'','lot_number':'NEW','lot_expiry':'','lot_quantity':'4','lot_original_number':'','lot_original_expiry':'','lot_original_quantity':'','lot_removed':'0'}.items():
            payload[name].append(value)
        payload['lot_quantity'][2] = '5'
        self.assertEqual(self.client.post(self.url, payload).status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 20)
        self.assertEqual(StockChange.objects.get().quantity, 6)
        self.assertEqual(ProductLotMovement.objects.count(), 2)

    def test_bad_rows_keep_inputs_and_do_not_save_product_fields(self):
        for field,value,message in [('lot_expiry','32-13-2031','invalid expiry'), ('lot_quantity','-1','negative stock')]:
            with self.subTest(field=field):
                payload=self.payload(); payload[field][0]=value; payload['price']='25.00'
                response=self.client.post(self.url,payload)
                self.assertEqual(response.status_code,200)
                self.assertContains(response,message)
                self.assertContains(response,f'value="{value}"')
                self.assertContains(response,'id="checkinTrends"')
                self.product.refresh_from_db()
                self.assertEqual(self.product.price,Decimal('12.49'))
        payload=self.payload(); payload['lot_number'][1]='LOT-A'; payload['lot_expiry'][1]='31-01-2031'
        self.assertContains(self.client.post(self.url,payload),'lot identity already exists')
        self.assertFalse(StockChange.objects.exists())

    def test_stale_transfer_is_rejected_even_with_unchanged_total(self):
        payload=self.payload()
        ProductLot.objects.filter(pk=self.lots[0].pk).update(quantity_on_hand=3)
        ProductLot.objects.filter(pk=self.lots[1].pk).update(quantity_on_hand=8)
        response=self.client.post(self.url,payload)
        self.assertContains(response,'stock changed while this page was open')
        self.assertEqual(list(ProductLot.objects.filter(product=self.product).order_by('pk').values_list('quantity_on_hand',flat=True)),[3,8,3])
        self.assertFalse(StockChange.objects.exists())

    def test_missing_lot_rows_cannot_silently_clear_saved_stock(self):
        payload = {name: value for name, value in self.payload().items() if not name.startswith('lot_')}
        response = self.client.post(self.url, payload)
        self.assertContains(response, 'lot list changed or is missing')
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 14)
        self.assertEqual(ProductLot.objects.filter(product=self.product, archived_at__isnull=True).count(), 3)
        self.assertFalse(StockChange.objects.exists())

    def test_inventory_count_remains_separate_from_system_lot_correction(self):
        self.session.inventory_mode=True; self.session.save()
        line=InventoryCountLine.objects.create(session=self.session, product=self.product, expected_qty=14, counted_qty=2)
        payload=self.payload(); payload['lot_quantity'][2]='4'
        self.assertEqual(self.client.post(self.url,payload).status_code,302)
        line.refresh_from_db(); self.product.refresh_from_db()
        self.assertEqual(line.counted_qty,2)
        self.assertEqual(self.product.quantity_in_stock,15)

    def test_initial_and_fragment_render_one_shared_list(self):
        url=reverse('checkin_session',args=[self.session.pk])
        response=self.client.get(url,{'product_id':self.product.pk})
        self.assertContains(response,'id="receivingLotList"',count=1)
        self.assertNotContains(response,'<fieldset class="product-lots-editor"')
        self.assertContains(response,'form="inlineEditForm" type="hidden" name="lot_original_quantity" value="3"')
        fragment=self.client.get(url,{'product_id':self.product.pk,'format':'checkin_fragments'},HTTP_X_REQUESTED_WITH='XMLHttpRequest').json()
        self.assertIn('unified-lot-cells',fragment['lot_rows_html'])
        self.assertIn('unified-lot-cells',fragment['product_lots_html'])
