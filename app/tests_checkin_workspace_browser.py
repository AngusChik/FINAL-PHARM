import os
from datetime import date
from pathlib import Path
from unittest import skipUnless

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import override_settings
from django.urls import reverse
from playwright.sync_api import expect, sync_playwright

from .models import Category, CheckinSession, InventoryCountLine, Product, ProductLot, StockChange


@skipUnless(os.environ.get('PHARMACY_BROWSER_TESTS') == '1', 'Opt-in browser suite')
@override_settings(AXES_ENABLED=False)
class CheckinWorkspaceBrowserTests(StaticLiveServerTestCase):
    def setUp(self):
        self.user=get_user_model().objects.create_user('checkin-browser',password='test',is_staff=True)
        self.client.force_login(self.user)
        category=Category.objects.create(name='Vitamins & supplements')
        self.product=Product.objects.create(name='Vitamin D3 1000 IU',barcode='CHECKINBROWSER',price='12.49',quantity_in_stock=44,category=category,description='Store on the vitamin shelf.')
        self.lots=[ProductLot.objects.create(product=self.product,lot_number=number,expiry_date=expiry,quantity_on_hand=qty) for number,expiry,qty in [('VD2604',date(2031,11,30),24),('VD2610',date(2032,5,31),12),(ProductLot.UNASSIGNED,None,6),('VD2501',date(2020,9,30),2)]]
        if 'two_lots' in self._testMethodName:
            ProductLot.objects.filter(pk__in=[lot.pk for lot in self.lots[2:]]).delete()
            self.lots=self.lots[:2]
            self.product.quantity_in_stock=36; self.product.save()
        self.session=CheckinSession.objects.create(user=self.user,scanned_by='Browser tester',inventory_mode='inventory_count' in self._testMethodName)
        self.engine=sync_playwright().start(); self.browser=self.engine.chromium.launch(headless=True)
        self.context=self.browser.new_context(viewport={'width':1440,'height':1000})
        self.context.add_cookies([{'name':settings.SESSION_COOKIE_NAME,'value':self.client.cookies[settings.SESSION_COOKIE_NAME].value,'url':self.live_server_url}])
        self.page=self.context.new_page(); self.page.set_default_timeout(10000)
        self.errors=[]; self.page.on('pageerror',lambda error:self.errors.append(str(error)))
        self.output=Path(settings.BASE_DIR)/'.runtime/checkin-workspace-qa'; self.output.mkdir(exist_ok=True)

    def tearDown(self):
        self.context.close(); self.browser.close(); self.engine.stop()
        self.assertEqual(self.errors,[])
        super().tearDown()

    def visit(self):
        self.page.goto(self.live_server_url+reverse('checkin_session',args=[self.session.pk])+'?product_id='+str(self.product.pk))
        self.page.wait_for_function('window.isCheckinInlineEditing && window.uiExpiryDates')
        self.page.evaluate('document.fonts.ready')

    def row(self,lot):
        return self.page.locator(f'.lot-editor-row[data-assignment-lot-id="{lot.pk}"]')

    def capture(self,name):
        self.page.screenshot(path=str(self.output/name),full_page=True)

    def assert_layout(self):
        metrics=self.page.evaluate('''() => ({width:innerWidth,scroll:document.documentElement.scrollWidth,
          overflow:[...document.querySelectorAll('.product-card,.receiving-strip,.checkin-trends-panel,.unified-lot-cells')].filter(el=>{
            const r=el.getBoundingClientRect(); return r.width && r.height && (r.left < -1 || r.right > innerWidth+1);
          }).map(el=>el.className)})''')
        self.assertLessEqual(metrics['scroll'],metrics['width'],metrics)
        self.assertFalse(metrics['overflow'],metrics)
        expect(self.page.locator('#receivingLotList')).to_have_count(1)

    def test_receiving_editing_errors_and_responsive_layout(self):
        self.visit()
        self.capture('receiving-desktop.png'); self.assert_layout()
        expect(self.page.locator('#productSecondaryDetails')).not_to_have_attribute('open','')
        self.page.locator('#checkinTrends > summary').click()
        expect(self.page.locator('.checkin-trends-body')).to_be_hidden()
        self.page.locator('#checkinTrends > summary').press('Enter')
        expect(self.page.locator('.checkin-trends-body')).to_be_visible()
        self.page.locator('#toggleEditBtn').click()
        expect(self.row(self.lots[2]).locator('[name=lot_number]')).to_be_editable()
        expect(self.page.locator('#product_lookup')).to_be_disabled()
        self.capture('editing-desktop.png'); self.assert_layout()
        self.row(self.lots[1]).locator('[name=lot_number]').fill('VD2604')
        self.row(self.lots[1]).locator('[name=lot_expiry]').fill('30-11-2031')
        self.page.locator('[form=inlineEditForm][type=submit]').click()
        expect(self.row(self.lots[1]).locator('.inline-field-error')).to_contain_text('lot identity already exists')
        self.capture('error-desktop.png')
        for width in (320,390,768,1024,1920):
            self.page.set_viewport_size({'width':width,'height':844 if width<769 else 1000})
            self.assert_layout()
            if width==390:self.capture('error-phone.png')
        self.page.locator('#cancelInlineEdit').click()
        expect(self.page.locator('#product_lookup')).to_be_enabled()
        for width in (320,390,768,1024,1440,1920):
            self.page.set_viewport_size({'width':width,'height':844 if width<769 else 1000}); self.assert_layout()
            if width==390:self.capture('receiving-phone.png')

    def test_pause_cancel_add_remove_save_and_receiving_refresh(self):
        self.visit()
        self.page.locator(f'[data-receiving-lot-id="{self.lots[0].pk}"]').click()
        self.page.locator('#toggleEditBtn').click()
        self.page.clock.install(); self.page.clock.fast_forward(31000)
        expect(self.page.locator('#product_lookup')).to_be_disabled()
        self.row(self.lots[2]).locator('[name=lot_quantity]').fill('8')
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('46')
        self.row(self.lots[2]).locator('[name=lot_number]').fill('NAMED')
        self.page.locator('[data-add-lot]').click()
        new_row=self.page.locator('.lot-editor-row[data-assignment-lot-id=""]')
        new_row.locator('[name=lot_number]').fill('NEW')
        new_row.locator('[name=lot_quantity]').fill('3')
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('49')
        new_row.locator('.lot-row-remove').click()
        self.page.locator('#cancelInlineEdit').click()
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('44')
        expect(self.page.locator('#receivingLotSelect')).to_have_value(str(self.lots[0].pk))
        self.page.clock.resume()
        self.page.locator('#toggleEditBtn').click()
        self.row(self.lots[2]).locator('[name=lot_number]').fill('NAMED')
        self.row(self.lots[2]).locator('[name=lot_quantity]').fill('8')
        self.page.locator('[form=inlineEditForm][type=submit]').click()
        expect(self.page.locator('#product_lookup')).to_be_enabled()
        expect(self.page.locator('#product_lookup')).to_be_focused()
        expect(self.page.locator('#sessionHistoryCard')).to_contain_text('+2')
        expect(self.row(self.lots[2]).locator('.receiving-lot-name')).to_have_text('NAMED')
        self.page.locator('#quickLotAddForm button').click()
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('47')
        expect(self.row(self.lots[0]).locator('[name=lot_quantity]')).to_have_value('25')
        self.addCleanup(lambda: self.assertEqual(Product.objects.get(pk=self.product.pk).quantity_in_stock,47))

    def test_inventory_count_keeps_count_and_system_stock_distinct(self):
        self.visit()
        self.page.locator('#quickLotAddForm button').click()
        expect(self.page.locator('#sbValue')).to_have_text('1')
        self.page.locator('#toggleEditBtn').click()
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('44')
        self.row(self.lots[0]).locator('[name=lot_quantity]').fill('25')
        self.page.locator('[form=inlineEditForm][type=submit]').click()
        expect(self.page.locator('#product_lookup')).to_be_enabled()
        expect(self.page.locator('#sbValue')).to_have_text('1')
        expect(self.page.locator('[data-count-system-stock]')).to_have_text('45')
        self.addCleanup(lambda: self.assertEqual(InventoryCountLine.objects.get().counted_qty,1))
        self.assert_layout(); self.capture('inventory-count-desktop.png')
        self.page.set_viewport_size({'width':390,'height':844}); self.assert_layout()

    def test_phone_stock_controls_and_new_lots_survive_fragment_refresh(self):
        self.page.set_viewport_size({'width':390,'height':900})
        self.visit()
        self.page.locator(f'[data-receiving-lot-id="{self.lots[0].pk}"]').click()
        self.page.locator('#quickLotAddForm button').click()
        expect(self.row(self.lots[0]).locator('[name=lot_quantity]')).to_have_value('25')
        self.page.locator('.btn-stock-minus').click()
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('44')
        expect(self.row(self.lots[3]).locator('[name=lot_quantity]')).to_have_value('1')
        self.page.locator('#receivingNewLotButton').click()
        self.page.locator('#receivingLotNumber').fill('PHONE-RESTOCK')
        self.page.locator('#receivingLotExpiry').fill('30-06-2033')
        self.page.locator('#quickLotAddForm button').click()
        expect(self.page.locator('#receivingLotList')).to_contain_text('PHONE-RESTOCK')
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('45')
        expect(self.page.locator('#receivingNewLotButton')).to_have_count(1)
        expect(self.page.locator('#assignLotStockButton')).to_have_count(1)
        self.page.locator('#toggleEditBtn').click()
        expect(self.page.locator('.btn-stock-minus')).to_be_hidden()
        expect(self.page.locator('#quickLotAddForm button')).to_be_hidden()
        self.page.locator('[data-add-lot]').click()
        new_row=self.page.locator('.lot-editor-row[data-assignment-lot-id=""]')
        new_row.locator('[name=lot_number]').fill('PHONE-EDIT')
        new_row.locator('[name=lot_quantity]').fill('2')
        self.page.locator('[form=inlineEditForm][type=submit]').click()
        expect(self.page.locator('#receivingLotList')).to_contain_text('PHONE-EDIT')
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('47')
        self.page.locator('#assignLotStockButton').click()
        expect(self.page.locator('#cancelLotAssignment')).to_be_visible()
        self.page.locator('#cancelLotAssignment').click()
        self.assert_layout()

    def test_two_lots_keyboard_calendar_and_unsaved_warning(self):
        self.visit()
        self.page.locator('#toggleEditBtn').click()
        self.assertFalse(self.page.evaluate("(() => {const e=new Event('beforeunload',{cancelable:true});window.dispatchEvent(e);return e.defaultPrevented;})()"))
        first=self.row(self.lots[0]).locator('[name=lot_number]')
        first.fill('RENAMED'); first.press('Enter')
        expiry=self.row(self.lots[0]).locator('[name=lot_expiry]')
        expect(expiry).to_be_focused()
        expect(self.page.locator('.flatpickr-calendar.open')).to_be_visible()
        expiry.press('Escape')
        self.assertTrue(self.page.evaluate("(() => {const e=new Event('beforeunload',{cancelable:true});window.dispatchEvent(e);return e.defaultPrevented;})()"))
        self.page.evaluate("window.refreshCheckinFragments(document.getElementById('inlineEditForm').dataset.refreshUrl)")
        expect(first).to_have_value('RENAMED')
        for width in (320,390,768,1024,1440,1920):
            self.page.set_viewport_size({'width':width,'height':844 if width<769 else 1000});self.assert_layout()
        self.page.set_viewport_size({'width':390,'height':844})
        self.row(self.lots[0]).scroll_into_view_if_needed()
        self.page.screenshot(path=str(self.output/'editing-phone-viewport.png'))
        self.page.locator('#cancelInlineEdit').click()
        for width in (320,390,768,1024,1440,1920):
            self.page.set_viewport_size({'width':width,'height':844 if width<769 else 1000});self.assert_layout()

    def test_assignment_expired_confirmation_and_selected_lot_removal(self):
        self.visit()
        self.page.locator('#assignLotStockButton').click()
        self.page.locator(f'[data-lot-assignment-choice="{self.lots[2].pk}"]').click()
        self.page.locator(f'[data-lot-assignment-choice="{self.lots[0].pk}"]').click()
        self.page.locator('#assignLotStockButton').click()
        expect(self.page.locator('#cancelLotAssignment')).to_be_hidden()
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('44')
        expect(self.row(self.lots[0]).locator('[name=lot_quantity]')).to_have_value('30')
        self.page.locator(f'[data-receiving-lot-id="{self.lots[3].pk}"]').click()
        expect(self.page.get_by_role('button',name='Select expired lot',exact=True)).to_be_visible()
        self.page.get_by_role('button',name='Select expired lot',exact=True).click()
        expect(self.page.locator('#receivingLotSelect')).to_have_value(str(self.lots[3].pk))
        self.page.locator('#toggleEditBtn').click()
        self.row(self.lots[3]).locator('.lot-row-remove').click()
        self.page.locator('[form=inlineEditForm][type=submit]').click()
        expect(self.page.locator('#product_lookup')).to_be_enabled()
        expect(self.page.locator('#receivingLotSelect')).to_have_value('')
        expect(self.page.locator('[data-system-stock-display]')).to_have_text('42')

    def test_native_validation_and_cancel_after_full_error_response(self):
        self.visit()
        self.page.locator('#toggleEditBtn').click()
        quantity = self.row(self.lots[0]).locator('[name=lot_quantity]')
        quantity.fill('-1')
        self.page.locator('[form=inlineEditForm][type=submit]').click()
        expect(quantity).to_have_attribute('aria-invalid', 'true')
        expect(self.row(self.lots[0]).locator('.inline-field-error')).to_be_visible()
        expect(self.page.locator('#product_lookup')).to_be_disabled()
        self.page.locator('#cancelInlineEdit').click()
        expect(quantity).to_have_value('24')
        self.page.locator('#toggleEditBtn').click()
        self.row(self.lots[1]).locator('[name=lot_number]').fill('VD2604')
        self.row(self.lots[1]).locator('[name=lot_expiry]').fill('30-11-2031')
        self.page.once('dialog', lambda dialog: dialog.accept())
        with self.page.expect_navigation():
            self.page.locator('#inlineEditForm').evaluate('form => form.submit()')
        expect(self.row(self.lots[1]).locator('.inline-field-error')).to_contain_text('lot identity already exists')
        expect(self.page.locator('#product_lookup')).to_be_disabled()
        self.page.locator('#cancelInlineEdit').click()
        expect(self.page.locator('#product_lookup')).to_be_enabled()
        expect(self.row(self.lots[1]).locator('[name=lot_number]')).to_have_value('VD2610')
