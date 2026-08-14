import re
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .middleware import CONTENT_SECURITY_POLICY
from .models import OrderingSheetEntry


class LocalBrowserAssetTests(SimpleTestCase):
    def test_runtime_templates_do_not_reference_public_cdns(self):
        disallowed_hosts = (
            'cdn.jsdelivr.net',
            'cdnjs.cloudflare.com',
            'fonts.googleapis.com',
            'fonts.gstatic.com',
        )
        source_roots = (
            Path(settings.BASE_DIR) / 'app' / 'templates',
            Path(settings.BASE_DIR) / 'static' / 'css',
            Path(settings.BASE_DIR) / 'static' / 'js',
        )
        violations = []
        for source_root in source_roots:
            for source_path in source_root.rglob('*'):
                if not source_path.is_file():
                    continue
                content = source_path.read_text(encoding='utf-8')
                if any(host in content for host in disallowed_hosts):
                    violations.append(str(source_path.relative_to(settings.BASE_DIR)))
        self.assertEqual(violations, [])

    def test_required_browser_assets_are_committed_locally(self):
        expected = (
            'chartjs/chart.umd.min.js',
            'flatpickr/flatpickr.min.css',
            'flatpickr/flatpickr.min.js',
            'jspdf/jspdf.umd.min.js',
            'jspdf/jspdf.plugin.autotable.min.js',
            'libre-barcode-128/libre-barcode-128.css',
            'libre-barcode-128/libre-barcode-128.woff2',
        )
        vendor_root = Path(settings.BASE_DIR) / 'static' / 'vendor'
        for relative_path in expected:
            with self.subTest(asset=relative_path):
                asset = vendor_root / relative_path
                self.assertTrue(asset.is_file())
                self.assertGreater(asset.stat().st_size, 0)

    def test_every_base_page_defines_a_specific_title(self):
        template_root = Path(settings.BASE_DIR) / 'app' / 'templates'
        missing_titles = []
        generic_titles = []
        title_pattern = re.compile(
            r'{%\s*block\s+title\s*%}(.*?){%\s*endblock\s*%}',
            re.IGNORECASE | re.DOTALL,
        )
        for template_path in template_root.rglob('*.html'):
            content = template_path.read_text(encoding='utf-8')
            if 'extends' not in content or 'base.html' not in content:
                continue
            match = title_pattern.search(content)
            relative_path = str(template_path.relative_to(settings.BASE_DIR))
            if not match:
                missing_titles.append(relative_path)
            elif match.group(1).strip() in {'', 'MPCP'}:
                generic_titles.append(relative_path)
        self.assertEqual(missing_titles, [])
        self.assertEqual(generic_titles, [])

    def test_caddy_and_django_apply_the_same_content_policy(self):
        caddyfile = (Path(settings.BASE_DIR) / 'Caddyfile').read_text(encoding='utf-8')
        self.assertIn(f'Content-Security-Policy "{CONTENT_SECURITY_POLICY}"', caddyfile)

    def test_checkin_session_rows_expose_keyboard_link_semantics(self):
        template = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'checkin_dashboard.html'
        ).read_text(encoding='utf-8')
        self.assertIn('data-session-url=', template)
        self.assertIn('tabindex="0" role="link"', template)
        self.assertNotRegex(template, r'<tr class="cd-clickable[^>]+onclick=')


class OrderingAccessibilityTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='ordering-a11y-user',
            password='test-password',
        )
        self.client.force_login(self.user)
        self.url = reverse('ordering_sheet')

    def test_ordering_forms_have_unique_prefixed_field_ids(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        markup = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
        ids = re.findall(r'\bid="([^"]+)"', markup)
        duplicate_ids = sorted({field_id for field_id in ids if ids.count(field_id) > 1})
        self.assertEqual(duplicate_ids, [])
        self.assertEqual(markup.count('<main '), 1)
        self.assertIn('id="id_drug-name"', html)
        self.assertIn('id="id_otc-name"', html)

    def test_rejected_drug_submission_keeps_field_error_highlighting(self):
        response = self.client.post(self.url, {
            'action': 'add',
            'drug-name': '',
            'drug-reasoning': OrderingSheetEntry.REASON_STOCK,
            'drug-urgency': OrderingSheetEntry.URGENCY_LOW,
            'drug-initials': 'AB',
        })

        self.assertEqual(response.status_code, 422)
        self.assertTrue(response.context['form'].is_bound)
        self.assertIn('name', response.context['form'].errors)
        self.assertContains(response, 'id="id_drug-name"', status_code=422)
        self.assertContains(response, 'aria-invalid="true"', status_code=422)

    def test_rejected_otc_submission_reopens_otc_form_with_highlighting(self):
        response = self.client.post(self.url, {
            'action': 'add_otc',
            'otc-name': '',
            'otc-side': OrderingSheetEntry.SIDE_LEFT,
            'otc-initials': 'AB',
        })

        self.assertEqual(response.status_code, 422)
        self.assertTrue(response.context['otc_form'].is_bound)
        self.assertIn('name', response.context['otc_form'].errors)
        self.assertContains(response, 'id="id_otc-name"', status_code=422)
        self.assertContains(response, 'id="os-mode-toggle" aria-label="Show OTC product form" checked', status_code=422)

    def test_prefixed_drug_submission_still_creates_an_entry(self):
        response = self.client.post(self.url, {
            'action': 'add',
            'drug-name': 'Amoxicillin 500 mg',
            'drug-reasoning': OrderingSheetEntry.REASON_STOCK,
            'drug-urgency': OrderingSheetEntry.URGENCY_LOW,
            'drug-initials': 'AB',
        })

        self.assertRedirects(response, self.url)
        entry = OrderingSheetEntry.objects.get()
        self.assertEqual(entry.name, 'Amoxicillin 500 mg')
        self.assertEqual(entry.created_by, self.user)

    def test_site_responses_include_the_local_only_content_policy(self):
        response = self.client.get(self.url)

        self.assertEqual(response.headers['Content-Security-Policy'], CONTENT_SECURITY_POLICY)
        self.assertNotIn('https:', response.headers['Content-Security-Policy'])
