"""Opt-in real browser regressions against Django's disposable test database.

Run with PHARMACY_BROWSER_TESTS=1 and inventory.settings_development.
"""
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

from .models import Category, CheckinSession, DeliveryCheckIn, Product, ProductLot


@skipUnless(os.environ.get('PHARMACY_BROWSER_TESTS') == '1', 'Opt-in browser suite')
@override_settings(AXES_ENABLED=False)
class PopupAndExpiryBrowserTests(StaticLiveServerTestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='ui-browser-admin', password='test-password', is_staff=True,
        )
        self.client.force_login(self.user)
        self.product = self.make_product()
        self.session = CheckinSession.objects.create(user=self.user, scanned_by='Browser tester')
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(viewport={'width': 1440, 'height': 1000})
        self.context.add_cookies([{
            'name': settings.SESSION_COOKIE_NAME,
            'value': self.client.cookies[settings.SESSION_COOKIE_NAME].value,
            'url': self.live_server_url,
        }])
        self.page = self.context.new_page()
        self.page.set_default_timeout(8000)
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))

    def tearDown(self):
        self.context.close()
        self.browser.close()
        self.playwright.stop()
        self.assertEqual(self.errors, [], 'Browser JavaScript errors')
        super().tearDown()

    def visit(self, name, args=None, query=''):
        self.page.goto(self.live_server_url + reverse(name, args=args) + query)
        self.page.wait_for_function('window.uiDialog && window.uiExpiryDates')

    def screenshot(self, name):
        output = Path(settings.BASE_DIR) / '.runtime' / 'dialog-expiry-qa'
        output.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(output / name), full_page=True)

    def test_delivery_success_closes_and_updates_records(self):
        self.visit('delivery')
        self.page.locator('#delivery-open-checkin').click()
        self.page.locator('#ci-barcode').fill('UITEST100')
        self.page.locator('#ci-first').fill('Browser')
        self.page.locator('#ci-last').fill('Visitor')
        self.page.locator('#checkin-form button[type=submit]').click()
        expect(self.page.locator('.ui-dialog')).to_have_count(0)
        expect(self.page.locator('#onsite-tbody')).to_contain_text('Browser Visitor')
        expect(self.page.locator('#delivery-open-checkin')).to_be_focused()
        self.page.locator('#delivery-open-checkout').click()
        self.page.locator('#co-barcode').fill('UITEST100')
        self.page.locator('#checkout-form button[type=submit]').click()
        expect(self.page.locator('.ui-dialog')).to_have_count(0)
        expect(self.page.locator('#history-tbody')).to_contain_text('Browser Visitor')
        expect(self.page.locator('#delivery-open-checkout')).to_be_focused()
        self.addCleanup(lambda: self.assertIsNotNone(
            DeliveryCheckIn.objects.get(barcode='UITEST100').checked_out_at,
        ))
        self.screenshot('delivery-saved-desktop.png')

    def test_delivery_errors_and_all_dismissal_methods(self):
        self.visit('delivery')
        self.page.set_viewport_size({'width': 390, 'height': 844})
        for close in ('button', 'escape', 'backdrop'):
            self.page.locator('#delivery-open-checkin').click()
            self.page.locator('#ci-barcode').fill('KEEP100')
            self.page.locator('#checkin-form button[type=submit]').click()
            expect(self.page.locator('#checkin-form button[type=submit]')).to_be_enabled()
            expect(self.page.locator('.ui-dialog')).to_be_visible()
            expect(self.page.locator('#ci-barcode')).to_have_value('KEEP100')
            if close == 'button':
                self.screenshot('delivery-dialog-mobile.png')
                self.page.locator('.ui-dialog-close').click()
            elif close == 'escape':
                self.page.keyboard.press('Escape')
            else:
                self.page.locator('.ui-dialog-backdrop').click(position={'x': 5, 'y': 180})
            expect(self.page.locator('.ui-dialog')).to_have_count(0)
            expect(self.page.locator('#delivery-open-checkin')).to_be_focused()
        self.addCleanup(lambda: self.assertEqual(DeliveryCheckIn.objects.count(), 0))
        self.page.locator('#delivery-open-checkout').click()
        self.page.locator('#co-barcode').fill('MISSING100')
        self.page.locator('#checkout-form button[type=submit]').click()
        expect(self.page.locator('#checkout-form button[type=submit]')).to_be_enabled()
        expect(self.page.locator('.ui-dialog')).to_be_visible()
        expect(self.page.locator('#co-barcode')).to_have_value('MISSING100')

    def test_shared_dialog_keyboard_does_not_close_underlying_drawer(self):
        self.visit('delivery')
        self.page.locator('#psSliderToggle').click()
        expect(self.page.locator('#psSliderPanel')).to_have_class('ps-slider-panel open')
        self.page.evaluate("""() => {
            const form = document.createElement('form');
            form.innerHTML = '<input type="hidden"><input id="visible"><button id="last">Save</button>';
            window.testDialog = window.uiDialog({title: 'Keyboard test', content: form});
        }""")
        self.page.locator('.ui-dialog-close').focus()
        self.page.keyboard.press('Shift+Tab')
        expect(self.page.locator('#last')).to_be_focused()
        self.page.keyboard.press('Tab')
        expect(self.page.locator('.ui-dialog-close')).to_be_focused()
        self.page.keyboard.press('Escape')
        expect(self.page.locator('.ui-dialog')).to_have_count(0)
        expect(self.page.locator('#psSliderPanel')).to_have_class('ps-slider-panel open')
        self.page.evaluate("window.testDialog = window.uiDialog({title: 'Locked', dismissible: false})")
        self.page.keyboard.press('Escape')
        expect(self.page.locator('.ui-dialog')).to_be_visible()
        expect(self.page.locator('#psSliderPanel')).to_have_class('ps-slider-panel open')
        self.page.evaluate("window.testDialog.close('done')")
        self.page.evaluate("""() => {
            const native = document.createElement('dialog');
            native.id = 'native-popup';
            native.textContent = 'Native popup';
            document.body.appendChild(native);
            native.showModal();
        }""")
        self.page.keyboard.press('Escape')
        expect(self.page.locator('#native-popup')).not_to_be_visible()
        expect(self.page.locator('#psSliderPanel')).to_have_class('ps-slider-panel open')
        self.page.keyboard.press('Escape')
        expect(self.page.locator('#psSliderPanel')).not_to_have_class('ps-slider-panel open')

    def test_delivery_pending_and_network_failure_keep_one_submission(self):
        self.visit('delivery')
        pending = []
        self.page.route('**/delivery/', lambda route: pending.append(route))
        self.page.locator('#delivery-open-checkin').click()
        self.page.locator('#ci-barcode').fill('PENDING100')
        self.page.locator('#ci-first').fill('Pending')
        self.page.locator('#ci-last').fill('Visitor')
        self.page.locator('#checkin-form button[type=submit]').click()
        self.page.locator('#checkin-form').evaluate('(form) => form.requestSubmit()')
        expect(self.page.locator('#checkin-form button[type=submit]')).to_be_disabled()
        self.assertEqual(len(pending), 1)
        pending[0].abort('failed')
        expect(self.page.locator('#checkin-form button[type=submit]')).to_be_enabled()
        expect(self.page.locator('.ui-dialog')).to_be_visible()
        expect(self.page.locator('#ci-first')).to_have_value('Pending')

    def test_existing_popups_close_and_restore_focus(self):
        for page_name, opener, popup in (
            ('low_stock', '#rp-export-btn', '#rp-export-modal'),
            ('label_printing', '#lp-custom-open', '#lp-custom-modal'),
            ('inventory_display', '#invHealthButton', '#invAuditOverlay'),
        ):
            with self.subTest(page=page_name):
                self.visit(page_name)
                self.page.locator(opener).focus()
                self.page.locator(opener).press('Enter')
                expect(self.page.locator(popup)).to_be_visible()
                self.page.keyboard.press('Escape')
                expect(self.page.locator(popup)).not_to_be_visible()
                expect(self.page.locator(opener)).to_be_focused()

    def test_label_modal_closes_after_successful_save(self):
        self.visit('label_printing')
        self.page.locator('#lp-custom-open').focus()
        self.page.locator('#lp-custom-open').press('Enter')
        self.page.locator('#lp-custom-title').fill('Browser label')
        self.page.locator('.lp-custom-name:visible').fill('Test line')
        self.page.locator('.lp-custom-price:visible').fill('5.00')
        self.page.locator('#lp-custom-submit').click()
        expect(self.page.locator('#lp-custom-modal')).not_to_be_visible()
        expect(self.page.locator('#lp-queue-region')).to_contain_text('Browser label')
        expect(self.page.locator('#lp-custom-open')).to_be_focused()

    def check_date_entry(self, field):
        field.fill('')
        field.press_sequentially('12')
        expect(field).to_have_value('12-')
        field.press_sequentially('03')
        expect(field).to_have_value('12-03-')
        field.press_sequentially('2031')
        expect(field).to_have_value('12-03-2031')
        self.assertEqual(field.evaluate('(el) => el.validationMessage'), '')
        field.press('ControlOrMeta+A')
        field.press('Backspace')
        expect(field).to_have_value('')
        field.press_sequentially('12')
        field.press('Backspace')
        expect(field).to_have_value('1')
        field.fill('29022032')
        expect(field).to_have_value('29-02-2032')
        field.fill('2031-12-25')
        expect(field).to_have_value('25-12-2031')
        field.evaluate('(el) => el.setSelectionRange(3, 5)')
        field.press_sequentially('11')
        expect(field).to_have_value('25-11-2031')
        field.fill('31-02-2031')
        field.press('Tab')
        expect(field).to_have_value('31-02-2031')
        self.assertIn('DD-MM-YYYY', field.evaluate('(el) => el.validationMessage'))
        field.fill('')
        self.assertEqual(field.evaluate('(el) => el.validationMessage'), '')
        field.evaluate("el => el._flatpickr.setDate(new Date(2032, 5, 9), true)")
        expect(field).to_have_value('09-06-2032')
        field.evaluate('el => el._flatpickr.close()')

    def test_new_product_existing_and_new_lot_dates(self):
        self.visit('new_product')
        self.check_date_entry(self.page.locator('input[name=lot_expiry]:visible').first)
        self.page.locator('[data-add-lot]').click()
        field = self.page.locator('input[name=lot_expiry]:visible').last
        self.check_date_entry(field)
        field.click()
        self.page.locator('.flatpickr-calendar.open [aria-label="June 15, 2032"]').click()
        expect(field).to_have_value('15-06-2032')
        field.click()
        field.evaluate('el => el._flatpickr.currentYearElement.focus()')
        self.page.keyboard.press('Escape')
        self.assertFalse(field.evaluate('el => el._flatpickr.isOpen'))
        self.screenshot('new-product-expiry-desktop.png')
        self.page.set_viewport_size({'width': 390, 'height': 844})
        field = self.page.locator('input[name=lot_expiry]:visible').last
        self.check_date_entry(field)
        self.assertTrue(field.evaluate("""el => {
            const style = getComputedStyle(el);
            const ctx = document.createElement('canvas').getContext('2d');
            ctx.font = style.font;
            return el.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
                >= ctx.measureText(el.value).width;
        }"""), 'The complete expiry date must fit on mobile')
        self.screenshot('new-product-expiry-mobile.png')

    def make_product(self):
        category = Category.objects.create(name='Browser lots')
        product = Product.objects.create(
            name='Browser test product', barcode='DATE100', category=category,
            price='5.00', quantity_in_stock=2,
        )
        ProductLot.objects.create(
            product=product, lot_number='BROWSER', quantity_on_hand=2,
            expiry_date=date(2031, 1, 1),
        )
        return product

    def test_edit_product_dynamic_dates(self):
        self.visit('edit_product', [self.product.pk])
        self.check_date_entry(self.page.locator('input[name=lot_expiry]:visible').first)
        self.page.locator('[data-add-lot]').click()
        self.check_date_entry(self.page.locator('input[name=lot_expiry]:visible').last)

    def test_checkin_receiving_dates_and_replaced_fields(self):
        self.visit('checkin_session', [self.session.pk], '?product_id=' + str(self.product.pk))
        self.page.locator('#receivingLotSelect').select_option('', force=True)
        self.check_date_entry(self.page.locator('#receivingLotExpiry'))
        self.page.locator('#receivingLotExpiry').evaluate("""el => {
            el._flatpickr.destroy();
            el.replaceWith(el.cloneNode(true));
        }""")
        self.check_date_entry(self.page.locator('#receivingLotExpiry'))
