"""Opt-in Ordering Sheet browser coverage on the disposable Django test DB.

PHARMACY_BROWSER_TESTS=1, with inventory.settings_development. All sync calls
are mocked; these checks never read or pull the real Google Sheet.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.db import connections
from django.test import override_settings
from django.urls import reverse
from playwright.sync_api import expect, sync_playwright

from .models import OrderingSheetEntry, OrderingSheetStatusEvent, UserAction


@skipUnless(os.environ.get('PHARMACY_BROWSER_TESTS') == '1', 'Opt-in browser suite')
@override_settings(AXES_ENABLED=False, TIME_ZONE='America/Toronto')
class OrderingRedesignBrowserTests(StaticLiveServerTestCase):
    def setUp(self):
        self.staff = get_user_model().objects.create_user('sheet-staff', is_staff=True)
        self.pu = get_user_model().objects.create_user('sheet-pu')
        names = ['Apo-atorvastatin 20mg', 'pms-perindopril-amlodipine 3.5mg/2.5mg',
                 'Jamp-ciprofloxacin 500mg', 'pms-olanzapine 5mg', 'Clobazam 10mg',
                 'Vitamin D 1000 IU']
        self.entries = []
        for i in range(85):
            self.entries.append(OrderingSheetEntry.objects.create(
                name=names[i] if i < len(names) else f'Sample medicine {i:03d} 20mg',
                patient_name='Sample patient' if i % 3 == 1 else '',
                reasoning='basket' if i % 3 == 1 else 'stock',
                urgency='high' if i % 3 == 2 else 'na',
                entry_type='otc' if i == 5 else 'drug', side='na',
                status='backordered' if i == 4 else ('ordered' if i == 3 else 'pending'),
                quantity_needed='3', quantity_remaining='1', initials='AB',
                order_note='Check supplier availability' if i == 3 else '',
                created_by=self.staff,
            ))
        baseline = datetime.now(timezone.utc)
        for i, entry in enumerate(self.entries):
            entry.created_at = baseline - timedelta(minutes=i)
        OrderingSheetEntry.objects.bulk_update(self.entries, ['created_at'])
        OrderingSheetStatusEvent.objects.create(
            entry=self.entries[3], from_status='pending', to_status='ordered', changed_by=self.staff,
        )
        for name, result in (
            ('app.gsheet_sync.is_configured', True),
            ('app.gsheet_sync.load_state', {'last_sync': datetime(2026, 10, 18, 13, 42, tzinfo=timezone.utc).timestamp(), 'errors': []}),
            ('app.scheduled_jobs.next_gsheet_pull', None),
        ):
            mock = patch(name, return_value=result)
            mock.start()
            self.addCleanup(mock.stop)
        self.sync = patch('app.scheduled_jobs.run_google_sheet_sync').start()
        self.addCleanup(patch.stopall)
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(viewport={'width': 1920, 'height': 1080})
        self.page = self.context.new_page()
        self.page.set_default_timeout(8000)
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.login(self.staff)

    def tearDown(self):
        self.context.close()
        self.browser.close()
        self.playwright.stop()
        self.sync.assert_not_called()
        self.assertEqual(self.errors, [], 'Browser JavaScript errors')
        super().tearDown()

    def login(self, user):
        # Playwright owns an event loop even with its synchronous API. Keep
        # Django's synchronous session writes on a separate, short-lived thread.
        def session_cookie():
            try:
                self.client.force_login(user)
                return self.client.cookies[settings.SESSION_COOKIE_NAME].value
            finally:
                connections.close_all()
        with ThreadPoolExecutor(max_workers=1) as executor:
            cookie = executor.submit(session_cookie).result()
        self.context.clear_cookies()
        self.context.add_cookies([{
            'name': settings.SESSION_COOKIE_NAME,
            'value': cookie,
            'url': self.live_server_url,
        }])

    def visit(self, embed=False):
        self.page.goto(self.live_server_url + reverse('ordering_sheet') + ('?embed=1' if embed else ''))
        self.page.wait_for_function('Boolean(window.uiConfirm && window.showToast)', timeout=8000)
        expect(self.page.locator('.os-personalize-slot button')).to_be_visible()

    def row(self, index):
        return self.page.locator(f'#os-tbody tr[data-entry-id="{self.entries[index].pk}"]')

    def screenshot(self, name):
        output = Path(settings.BASE_DIR) / '.runtime' / 'ordering-sheet-qa'
        output.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(output / name), full_page=True)

    def assert_toolbar_fits(self):
        sizes = self.page.locator('#os-workspace .table-card-header').evaluate('''el => ({
            width: el.clientWidth, scroll: el.scrollWidth,
            documentWidth: document.documentElement.clientWidth,
            documentScroll: document.documentElement.scrollWidth,
            overflowing: [...el.querySelectorAll('select, input, button')].filter(control => {
                if (!control.getClientRects().length) return false;
                const r = control.getBoundingClientRect(), p = el.getBoundingClientRect();
                return r.left < p.left - 1 || r.right > p.right + 1;
            }).map(control => control.id)
        })''')
        self.assertLessEqual(sizes['scroll'], sizes['width'] + 1, sizes)
        self.assertLessEqual(sizes['documentScroll'], sizes['documentWidth'] + 1, sizes)
        self.assertEqual(sizes['overflowing'], [], sizes)

    def test_layout_roles_filters_keyboard_and_responsive_views(self):
        for user in (self.staff, self.pu):
            self.login(user)
            for embed in (False, True):
                with self.subTest(user=user.username, embed=embed):
                    self.visit(embed)
                    expect(self.page.locator('.os-account-badge')).to_have_text('Staff account' if user.is_staff else 'PU account')
                    expect(self.page.locator('#os-last-pulled')).to_contain_text('18/10/2026, 09:42')
                    expect(self.page.locator('.os-delete-form')).to_have_count(85)
                    expect(self.page.locator('.os-status-select')).to_have_count(85 if user.is_staff else 0)
                    pull = self.page.locator('.os-embed-pull-btn')
                    if user.is_staff:
                        expect(pull).to_be_enabled()
                    else:
                        expect(pull).to_be_disabled()
                    for width in (1920, 1440, 1024, 768, 390):
                        self.page.set_viewport_size({'width': width, 'height': 1000})
                        self.assert_toolbar_fits()
                        if width in (1920, 1024, 390):
                            self.screenshot(f'{user.username}-{"embed" if embed else "full"}-{width}.png')
                    self.page.set_viewport_size({'width': 1920, 'height': 1080})
                    self.page.locator('body').evaluate("el => el.style.zoom = '1.5'")
                    self.assert_toolbar_fits()
                    self.screenshot(f'{user.username}-{"embed" if embed else "full"}-zoom150.png')
                    self.page.locator('body').evaluate("el => el.style.zoom = ''")
                    self.page.locator('#id_drug-patient_name').focus()
                    for name in ('patient_name', 'reasoning', 'urgency', 'name', 'quantity_needed', 'quantity_remaining', 'initials'):
                        expect(self.page.locator(f'#id_drug-{name}')).to_be_focused()
                        self.page.keyboard.press('Tab')
                    expect(self.page.locator('#os-form button[type=submit]')).to_be_focused()
                    self.page.locator('#os-type-filter').select_option('otc')
                    expect(self.page.locator('#os-tbody tr:visible')).to_have_count(1)
                    self.page.locator('#os-status-filter').select_option('ordered')
                    expect(self.page.locator('#os-nomatch')).to_be_visible()
                    self.page.reload()
                    expect(self.page.locator('#os-type-filter')).to_have_value('otc')
                    expect(self.page.locator('#os-status-filter')).to_have_value('ordered')
                    self.page.locator('#os-clear-filters').click()
                    expect(self.page.locator('#os-tbody tr:visible')).to_have_count(85)
                    self.page.locator('.os-mode-label[data-mode=otc]').click()
                    expect(self.page.locator('#id_otc-side')).to_have_value('na')
                    self.page.locator('.os-mode-label[data-mode=drug]').click()

    def test_solid_colours_and_compact_rows(self):
        self.visit(True)
        self.page.mouse.move(0, 0)
        for index, background, reason in (
            (0, 'rgb(255, 241, 242)', 'rgb(255, 255, 0)'),
            (1, 'rgb(255, 241, 242)', 'rgb(255, 153, 0)'),
            (2, 'rgb(255, 241, 242)', 'rgb(147, 197, 253)'),
            (3, 'rgb(236, 253, 245)', 'rgb(255, 255, 0)'),
            (4, 'rgb(252, 165, 165)', 'rgb(252, 165, 165)'),
        ):
            row = self.row(index)
            expect(row.locator('.os-patient-col')).to_have_css('background-color', background)
            expect(row.locator('.os-reason-col')).to_have_css('background-color', reason)
            self.assertEqual(row.locator('td').evaluate_all('(cells) => cells.every(c => getComputedStyle(c).backgroundImage === "none")'), True)
        self.assertLessEqual(self.row(0).bounding_box()['height'], 36)
        self.assertGreaterEqual(self.row(0).bounding_box()['height'], 32)
        self.assertGreater(self.row(0).locator('.os-name-col').bounding_box()['width'], 500)
        meta = self.row(3).locator('.os-row-meta')
        expect(meta).to_contain_text('Check supplier availability')
        expect(meta).to_contain_text('Updated')
        self.assertLessEqual(meta.bounding_box()['height'], 16)
        self.assertIn('Check supplier availability', meta.get_attribute('title'))
        self.assertIn('by sheet-staff', meta.get_attribute('title'))

    def test_delete_in_place_preserves_scroll_selection_sort_and_drafts(self):
        for user, embed in ((self.staff, False), (self.pu, True)):
            self.login(user)
            self.visit(embed)
            self.page.locator('#os-type-filter').select_option('drug')
            self.page.locator('#os-status-filter').select_option('pending')
            self.page.locator('#os-search').fill('Sample medicine')
            self.page.get_by_role('button', name='Sort by drug name', exact=True).click()
            self.page.locator('#id_drug-name').fill('Unsubmitted medicine')
            self.row(35).locator('.os-note-toggle').click()
            self.row(35).locator('.os-note-input').fill('Keep this unsaved note')
            self.row(36).locator('.os-row-check').check()
            target = self.row(40 if user.is_staff else 41)
            target.locator('.delete-btn').scroll_into_view_if_needed()
            self.page.evaluate('''() => {
                window.savedTbody = document.querySelector('#os-tbody');
                window.savedRows = [...savedTbody.rows];
                window.savedScroll = {x: scrollX, y: scrollY,
                    top: document.querySelector('.active-table-wrap').scrollTop,
                    left: document.querySelector('.active-table-wrap').scrollLeft};
            }''')
            navigations = []
            handler = lambda request: navigations.append(request.url) if request.is_navigation_request() else None
            self.page.on('request', handler)
            target.locator('.delete-btn').click()
            expect(target).to_have_count(0)
            expect(self.page.locator('.ui-dialog')).to_have_count(0)
            expect(self.page.locator('#id_drug-name')).to_have_value('Unsubmitted medicine')
            expect(self.row(35).locator('.os-note-input')).to_have_value('Keep this unsaved note')
            expect(self.row(36).locator('.os-row-check')).to_be_checked()
            self.assertTrue(self.page.evaluate('''() => {
                const wrap = document.querySelector('.active-table-wrap');
                return window.savedTbody === document.querySelector('#os-tbody') &&
                    [...savedTbody.rows].every(row => savedRows.includes(row)) &&
                    Math.abs(savedScroll.top - wrap.scrollTop) <= 1 &&
                    savedScroll.left === wrap.scrollLeft && savedScroll.y === scrollY && savedScroll.x === scrollX;
            }'''))
            self.assertEqual(navigations, [])
            self.page.remove_listener('request', handler)
            expect(self.page.locator('#os-type-filter')).to_have_value('drug')
            expect(self.page.locator('#os-status-filter')).to_have_value('pending')
            expect(self.page.locator('.os-name-col[aria-sort]')).to_have_attribute('aria-sort', 'ascending')
            self.page.locator('#os-clear-filters').click()

    def test_rapid_delete_network_and_server_errors_keep_rows_and_focus(self):
        self.visit()
        target = self.row(0)
        pending = []
        self.page.route('**/ordering-sheet/', lambda route: pending.append(route))
        target.locator('.delete-btn').click()
        target.locator('.os-delete-form').evaluate('(form) => { form.requestSubmit(); form.requestSubmit(); }')
        expect(target.locator('.delete-btn')).to_be_disabled()
        self.assertEqual(len(pending), 1)
        self.page.locator('#id_drug-name').fill('Preserve focus and input')
        pending.pop().abort('failed')
        expect(target.locator('.delete-btn')).to_be_enabled()
        expect(self.page.locator('#id_drug-name')).to_be_focused()
        target.locator('.os-delete-form').evaluate('(form) => form.requestSubmit()')
        expect(target.locator('.delete-btn')).to_be_disabled()
        self.page.wait_for_timeout(100)
        pending.pop().fulfill(status=500, content_type='application/json', body='{"ok":false,"error":"Test server failure"}')
        expect(target.locator('.delete-btn')).to_be_enabled()
        expect(self.page.locator('#id_drug-name')).to_be_focused()
        expect(self.page.get_by_text('Test server failure', exact=True)).to_be_visible()
        self.page.unroute('**/ordering-sheet/')
        target.locator('.delete-btn').click()
        expect(target).to_have_count(0)
        self.addCleanup(lambda: self.assertEqual(UserAction.objects.filter(action='ordering_delete').count(), 1))

    def test_last_filtered_and_last_total_row_and_bulk_confirmation(self):
        self.login(self.pu)
        self.visit()
        self.page.locator('#os-search').fill('Clobazam')
        self.row(4).locator('.delete-btn').click()
        expect(self.page.locator('#os-nomatch')).to_be_visible()
        expect(self.page.locator('#os-status-filter option[value=backordered]')).to_have_text('Back-Ordered (0)')
        self.page.locator('#os-clear-filters').click()
        self.page.locator('#id_drug-name').fill('Draft after deleting all entries')
        self.page.locator('#os-select-all').check()
        self.row(0).locator('.os-row-check').uncheck()
        self.page.locator('#os-bulk-delete').click()
        expect(self.page.locator('.ui-dialog')).to_be_visible()
        self.page.get_by_role('button', name='Cancel', exact=True).click()
        expect(self.page.locator('#os-tbody tr')).to_have_count(84)
        self.page.locator('#os-bulk-delete').click()
        self.page.get_by_role('button', name='Move to Recovery', exact=True).click()
        expect(self.page.locator('#os-tbody tr')).to_have_count(1)
        self.row(0).locator('.delete-btn').click()
        expect(self.page.locator('#os-empty')).to_be_visible()
        expect(self.page.locator('#os-table')).not_to_be_visible()
        expect(self.page.locator('#os-count')).to_have_text('0')
        expect(self.page.locator('#id_drug-name')).to_have_value('Draft after deleting all entries')
        self.addCleanup(lambda: self.assertEqual(OrderingSheetEntry.objects.filter(is_deleted=True, deleted_by=self.pu).count(), 85))

    def test_custom_status_keeps_fast_inline_save(self):
        self.visit(True)
        target = self.row(0)
        target.locator('.os-status-select').select_option('custom')
        expect(target.locator('.os-custom-status-input')).to_be_focused()
        target.locator('.os-custom-status-input').fill('Awaiting supplier')
        target.locator('.os-custom-status-input').press('Enter')
        expect(target.locator('.os-custom-status-form')).not_to_be_visible()
        expect(target.locator('.os-status-select option:checked')).to_have_text('Awaiting supplier')
        expect(target.locator('.os-status-select')).to_have_attribute('data-current', 'custom')
        expect(target.locator('.os-status-select')).to_be_enabled()
        self.addCleanup(lambda: self.assertEqual(OrderingSheetEntry.objects.get(pk=self.entries[0].pk).custom_status_text, 'Awaiting supplier'))

    def test_actual_embedded_drawer_for_staff_and_pu(self):
        for user in (self.staff, self.pu):
            self.login(user)
            self.visit()
            self.page.locator('#osSliderToggle').click()
            panel = self.page.locator('#osSliderPanel')
            expect(panel).to_have_class('os-slider-panel open')
            expect(panel.locator('.os-slider-header')).to_have_css('background-image', 'none')
            self.assertLessEqual(panel.locator('.os-slider-header').bounding_box()['height'], 40)
            frame = self.page.frame_locator('#osSliderFrame')
            expect(frame.locator('.os-account-badge')).to_have_text('Staff account' if user.is_staff else 'PU account')
            expect(frame.locator('#os-type-filter')).to_be_visible()
            expect(frame.locator('.os-delete-form')).to_have_count(85)
            for width in (1440, 768, 390):
                self.page.set_viewport_size({'width': width, 'height': 1000})
                fit = frame.locator('.table-card-header').evaluate('(el) => el.scrollWidth <= el.clientWidth + 1')
                self.assertTrue(fit)
                if width != 768:
                    self.screenshot(f'{user.username}-drawer-{width}.png')
            self.page.locator('#osSliderClose').click()
            expect(panel).not_to_have_class('os-slider-panel open')
            self.page.set_viewport_size({'width': 1920, 'height': 1080})

    def test_personalization_keeps_column_keys_after_reordering(self):
        self.visit()
        before = self.row(0).bounding_box()['height']
        self.page.locator('.os-personalize-slot button').click()
        dialog = self.page.locator('.ui-dialog')
        dialog.locator('input[value=reasoning]').uncheck()
        dialog.locator('input[value=compact]').check()
        dialog.get_by_role('button', name='Save view', exact=True).click()
        expect(dialog).to_have_count(0)
        expect(self.row(0).locator('.os-reason-col')).not_to_be_visible()
        expect(self.row(0).locator('.os-name-col')).to_be_visible()
        self.assertLessEqual(self.row(0).bounding_box()['height'], before)
        self.visit(True)
        expect(self.row(0).locator('.os-reason-col')).not_to_be_visible()
        expect(self.row(0).locator('.os-name-col')).to_be_visible()
        expect(self.page.locator('#os-table')).to_have_class('active-table ui-table-compact')
        self.row(0).locator('.delete-btn').click()
        expect(self.row(0)).to_have_count(0)
        expect(self.row(1).locator('.os-reason-col')).not_to_be_visible()
        expect(self.row(1).locator('.os-name-col')).to_be_visible()

    def test_add_edit_comment_and_status_keep_existing_input_methods(self):
        for user, embed in ((self.staff, False), (self.pu, True)):
            self.login(user)
            self.visit(embed)
            self.page.locator('#id_drug-patient_name').fill('Example patient')
            self.page.locator('#id_drug-reasoning').select_option('stock')
            self.page.locator('#id_drug-urgency').select_option('high')
            self.page.locator('#id_drug-name').fill('New test medicine')
            self.page.locator('#id_drug-quantity_needed').fill('2')
            self.page.locator('#id_drug-quantity_remaining').fill('0')
            self.page.locator('#id_drug-initials').fill('XY')
            self.page.locator('#os-form button[type=submit]').click()
            row = self.page.locator('#os-tbody tr[data-name="New test medicine"]')
            expect(row).to_have_count(1)
            expect(self.page.locator('#id_drug-name')).to_have_value('')
            row.locator('.os-note-toggle').click()
            row.locator('.os-note-input').fill('Availability requested')
            row.locator('.os-note-save').click()
            expect(row.locator('.os-row-meta')).to_contain_text('Availability requested')
            row.locator('.os-edit-btn').click()
            self.page.locator('#os-edit-name').fill('Updated test medicine')
            self.page.locator('#os-edit-form button[type=submit]').click()
            row = self.page.locator('#os-tbody tr[data-name="Updated test medicine"]')
            expect(row).to_have_count(1)
            expect(self.page.locator('#os-edit-overlay')).not_to_be_visible()
            if user.is_staff:
                row.locator('.os-status-select').select_option('ordered')
                expect(row).to_have_attribute('data-status', 'ordered')
                expect(row.locator('.os-row-meta')).to_contain_text('Updated')
            row.locator('.delete-btn').click()
            expect(row).to_have_count(0)
