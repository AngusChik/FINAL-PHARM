"""Opt-in browser check: six PU browsers plus four phones, using a disposable DB."""

import os
from pathlib import Path
from unittest import skipUnless

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import override_settings
from django.urls import reverse
from playwright.sync_api import expect, sync_playwright

from app.models import UserSession


@skipUnless(os.environ.get('PHARMACY_BROWSER_TESTS') == '1', 'Opt-in browser suite')
@override_settings(
    AXES_ENABLED=False, MAX_PU_SESSIONS=6, MAX_PU_PHONE_SESSIONS=4,
    PHARMACY_HOST='192.168.0.22', SECURE=True,
    # Exercise the production page layout; the runner still uses development
    # credentials, disabled integrations, and its disposable PostgreSQL DB.
    PHARMACY_ENVIRONMENT='production',
    PASSWORD_HASHERS=['django.contrib.auth.hashers.MD5PasswordHasher'],
)
class PUWorkstationBrowserTests(StaticLiveServerTestCase):
    def test_six_computers_four_phones_qr_and_capacity_in_separate_browser_contexts(self):
        get_user_model().objects.create_user('PU', password='browser-test-password')
        output = Path(settings.BASE_DIR) / '.runtime' / 'pu-session-phone-qa'
        output.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as engine:
            browser = engine.chromium.launch(headless=True)
            pages = []

            def login(phone=False):
                context = browser.new_context(
                    viewport={'width': 390, 'height': 844} if phone else {'width': 1440, 'height': 1000},
                    is_mobile=phone, has_touch=phone,
                )
                page = context.new_page()
                page.set_default_timeout(10000)
                page.goto(self.live_server_url + reverse('connect_phone' if phone else 'login'))
                page.locator('input[name="username"]').fill('PU')
                page.locator('input[name="password"]').fill('browser-test-password')
                page.locator('button[type="submit"]').click()
                return page

            for _ in range(6):
                page = login()
                page.wait_for_url(self.live_server_url + reverse('dashboard'))
                pages.append(page)

            extra = login()
            expect(extra).to_have_url(self.live_server_url + reverse('login'))
            expect(extra.locator('body')).to_contain_text('All 6 PU slots are already in use.')

            phones = []
            for _ in range(4):
                phone = login(phone=True)
                phone.wait_for_url(self.live_server_url + reverse('dashboard'))
                phones.append(phone)

            extra_phone = login(phone=True)
            expect(extra_phone).to_have_url(self.live_server_url + reverse('login'))
            expect(extra_phone.locator('body')).to_contain_text('Phone access is currently full.')

            for page in pages + phones:
                page.reload()
                expect(page.locator('#connectPhoneBtn')).to_be_visible()

            phone = phones[0]
            for page, filename in ((pages[0], 'connect-phone-desktop.png'), (phone, 'connect-phone-mobile.png')):
                page.locator('#connectPhoneBtn').scroll_into_view_if_needed()
                if page is phone:
                    page.screenshot(path=str(output / 'phone-dashboard-before-connect.png'))
                    page.wait_for_function("""() =>
                        document.querySelector('#connectPhoneBtn').getBoundingClientRect().bottom <=
                        document.querySelector('.app-nav').getBoundingClientRect().top
                    """)
                page.locator('#connectPhoneBtn').click()
                expect(page.locator('#cpModalOverlay')).to_be_visible()
                expect(page.locator('#cpUrl')).to_have_text('https://192.168.0.22/connect-phone/')
                expect(page.locator('.cp-qr-img')).to_have_js_property('complete', True)
                self.assertGreater(page.locator('.cp-qr-img').evaluate('(image) => image.naturalWidth'), 0)
                page.locator('.cp-modal').screenshot(path=str(output / filename))
                page.locator('#cpModalClose').click()

            browser.close()

        # Playwright owns an event loop while running; inspect the database
        # after leaving that context rather than bypassing Django's async guard.
        self.assertEqual(UserSession.objects.filter(device_type=UserSession.DEVICE_COMPUTER).count(), 6)
        self.assertEqual(UserSession.objects.filter(device_type=UserSession.DEVICE_PHONE).count(), 4)
