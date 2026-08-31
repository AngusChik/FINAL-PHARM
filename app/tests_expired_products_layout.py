from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(AXES_ENABLED=False)
class ExpiredProductsLayoutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='expired-layout-user',
            password='pass1234',
        )
        self.client.force_login(self.user)

    def test_view_controls_are_grouped_in_the_sticky_left_rail(self):
        response = self.client.get(reverse('expired_products'))

        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        left_start = html.index('<div class="left-controls">')
        right_start = html.index('<div class="right-panel">', left_start)
        left_rail = html[left_start:right_start]

        log_button = left_rail.index('Log Expired Products')
        pdf_button = left_rail.index('id="printReportBtn"')
        quick_filter = left_rail.index('Quick Filter')
        expiry_window = left_rail.index('Expiry Window')
        date_range = left_rail.index('Expiry Date Range')

        self.assertLess(log_button, pdf_button)
        self.assertLess(pdf_button, quick_filter)
        self.assertLess(quick_filter, expiry_window)
        self.assertLess(expiry_window, date_range)
        self.assertNotContains(response, 'Preset filters for common time ranges.')

    def test_log_mode_keeps_a_way_back_to_the_expired_products_view(self):
        response = self.client.get(reverse('expired_products'), {'mode': 'log'})

        self.assertContains(response, '← View Expired Products')
        self.assertNotContains(response, '<div class="left-controls">', html=True)

    def test_pdf_modal_inherits_custom_range_and_sort(self):
        response = self.client.get(reverse('expired_products'), {
            'mode': 'view',
            'date_filter': 'custom',
            'date_from': '2026-08-01',
            'date_to': '2026-08-31',
            'sort': '-name',
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'role="dialog"')
        self.assertContains(
            response,
            '<option value="custom" selected>Custom date range</option>',
            html=True,
        )
        self.assertContains(
            response,
            '<input type="date" id="printDateFrom" value="2026-08-01">',
            html=True,
        )
        self.assertContains(
            response,
            '<input type="date" id="printDateTo" value="2026-08-31">',
            html=True,
        )
        self.assertContains(
            response,
            '<option value="-name" selected>Name (Z–A)</option>',
            html=True,
        )

    def test_pdf_modal_url_contract_includes_custom_dates(self):
        response = self.client.get(reverse('expired_products'))

        self.assertContains(response, "url.searchParams.set('date_filter', filter)")
        self.assertContains(response, "url.searchParams.set('sort', sort)")
        self.assertContains(response, "url.searchParams.set('date_from', from)")
        self.assertContains(response, "url.searchParams.set('date_to', to)")
        self.assertContains(response, 'Choose at least one expiry date for the PDF report.')
