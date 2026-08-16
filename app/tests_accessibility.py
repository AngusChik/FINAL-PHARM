import re
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .middleware import CONTENT_SECURITY_POLICY
from .models import (
    Category, CheckinSession, CheckoutOrder, OrderingSheetEntry, Product,
)


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

    def test_recently_purchased_row_details_are_keyboard_operable(self):
        template_root = Path(settings.BASE_DIR) / 'app' / 'templates'
        page = (template_root / 'low_stock.html').read_text(encoding='utf-8')
        rows = (template_root / 'partials' / 'rp_rows.html').read_text(
            encoding='utf-8'
        )

        self.assertIn('class="rp-row-expand" aria-expanded="false"', rows)
        self.assertIn('aria-controls="rp-detail-{{ recent.id }}"', rows)
        self.assertIn('id="rp-detail-{{ recent.id }}" class="rp-detail-row"', rows)
        self.assertIn("var expandButton = e.target.closest('.rp-row-expand');", page)
        self.assertIn("expandButton.setAttribute('aria-expanded', 'true');", page)
        self.assertIn(
            "tbody.addEventListener('click', async function(e) {\n"
            "    const btn = e.target.closest('.rp-delete-one');",
            page,
        )

    def test_delivery_table_switcher_defaults_to_onsite_then_offers_both_views(self):
        template = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'delivery.html'
        ).read_text(encoding='utf-8')

        onsite_button = template.index('data-delivery-view="onsite"')
        checkedout_button = template.index('data-delivery-view="checkedout"')
        both_button = template.index('data-delivery-view="both"')

        self.assertLess(onsite_button, checkedout_button)
        self.assertLess(checkedout_button, both_button)
        self.assertIn(
            'data-delivery-view="onsite"\n              aria-pressed="true"',
            template,
        )
        self.assertIn('id="delivery-onsite-panel" data-delivery-panel="onsite"', template)
        self.assertIn(
            'id="delivery-history-panel" data-delivery-panel="checkedout" '
            'aria-labelledby="history-table-heading" hidden',
            template,
        )
        self.assertIn("setDeliveryView('onsite');", template)

    def test_sidebar_omits_checkout_but_keeps_its_keyboard_shortcut(self):
        template = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'base.html'
        ).read_text(encoding='utf-8')
        sidebar = re.search(
            r'<ul class="nav-links">(.*?)</ul>', template, re.DOTALL,
        ).group(1)

        self.assertNotIn("{% url 'checkout' %}", sidebar)
        self.assertIn('o: "{% url \'checkout\' %}"', template)

    def test_checkout_card_has_one_full_size_shell(self):
        template = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'checkout_chooser.html'
        ).read_text(encoding='utf-8')

        self.assertIn('.cc-shell {', template)
        self.assertIn('max-width: 592px', template)
        self.assertIn('<div class="cc-shell">', template)
        self.assertNotIn('<div class="cc-modal">', template)

    def test_wide_tables_receive_an_accessible_top_scrollbar(self):
        script = (
            Path(settings.BASE_DIR) / 'static' / 'js' / 'ui-system.js'
        ).read_text(encoding='utf-8')
        styles = (
            Path(settings.BASE_DIR) / 'static' / 'css' / 'ui-system.css'
        ).read_text(encoding='utf-8')

        self.assertIn('function wireTableOverflowScrollers()', script)
        self.assertIn("topScroll.setAttribute('aria-label', 'Horizontal table scroll')", script)
        self.assertIn('scroller.scrollLeft = topScroll.scrollLeft', script)
        self.assertIn('topScroll.scrollLeft = scroller.scrollLeft', script)
        self.assertIn('wireTableOverflowScrollers();', script)
        self.assertIn('body.app-shell .ui-table-top-scroll {', styles)

    def test_permission_markers_do_not_repeat_admin_badges_for_staff(self):
        script = (
            Path(settings.BASE_DIR) / 'static' / 'js' / 'ui-system.js'
        ).read_text(encoding='utf-8')
        styles = (
            Path(settings.BASE_DIR) / 'static' / 'css' / 'ui-system.css'
        ).read_text(encoding='utf-8')

        self.assertIn("if (canAdminister) {", script)
        self.assertIn("visual.classList.add('ui-admin-available');\n          return;", script)
        self.assertIn("marker.textContent = '🔒';", script)
        self.assertIn("marker.setAttribute('aria-label', 'Admin password required')", script)
        self.assertNotIn("marker.textContent = canAdminister ? 'Admin'", script)
        self.assertNotIn('.ui-admin-locked { border-style: dashed', styles)

    def test_sidebar_open_state_waits_for_real_pointer_exit_after_navigation(self):
        template = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'base.html'
        ).read_text(encoding='utf-8')

        self.assertIn("var navSurface = nav.querySelector('.nav-content') || nav;", template)
        self.assertIn('var pointerObserved = !restoredOpen;', template)
        self.assertIn("nav.addEventListener('pointerenter'", template)
        self.assertIn("nav.addEventListener('pointerleave'", template)
        self.assertIn("document.addEventListener('pointermove'", template)
        self.assertIn('if (!desktopNav.matches || !pointerObserved || navLinkDown) return;', template)
        self.assertNotIn("if (!nav.matches(':hover'))", template)
        self.assertNotIn('}, 50);', template)

    def test_alt_x_closes_an_open_sidebar_before_navigating(self):
        template = (
            Path(settings.BASE_DIR) / 'app' / 'templates' / 'base.html'
        ).read_text(encoding='utf-8')
        script = (
            Path(settings.BASE_DIR) / 'static' / 'js' / 'ui-system.js'
        ).read_text(encoding='utf-8')
        styles = (
            Path(settings.BASE_DIR) / 'static' / 'css' / 'ui-system.css'
        ).read_text(encoding='utf-8')

        self.assertIn("if (k === 'x')", template)
        self.assertIn("sidebar.classList.contains('nav-force-open')", template)
        self.assertIn("sidebar.dispatchEvent(new CustomEvent('ui:nav-close'))", template)
        self.assertIn("nav.addEventListener('ui:nav-close'", template)
        self.assertIn("event.key !== 'Escape'", template)
        self.assertIn("nav.contains(event.target)", template)
        self.assertIn('closeNav(false);', template)
        self.assertIn('nav.nav-force-closed:hover', template)
        self.assertIn('body.app-shell nav.nav-force-closed:focus-within', styles)
        self.assertIn('body.app-shell nav.nav-force-closed .nav-label', styles)
        self.assertIn("['Alt + X', 'Dashboard / close sidebar']", script)

    def test_table_view_save_button_submits_its_footer_form(self):
        script = (
            Path(settings.BASE_DIR) / 'static' / 'js' / 'ui-system.js'
        ).read_text(encoding='utf-8')

        self.assertIn('var tablePreferenceDialogId = 0;', script)
        self.assertIn("form.id = 'ui-table-settings-form-'", script)
        self.assertIn("save.setAttribute('form', form.id);", script)
        self.assertIn("save.textContent = 'Saving…';", script)

    def test_product_workflows_use_one_name_sku_or_barcode_field(self):
        template_root = Path(settings.BASE_DIR) / 'app' / 'templates'
        lookup_templates = (
            'order_form.html',
            'checkout.html',
            'checkin.html',
            'expired_products.html',
        )

        for template_name in lookup_templates:
            with self.subTest(template=template_name):
                source = (template_root / template_name).read_text(encoding='utf-8')
                self.assertEqual(source.count('id="product_lookup"'), 1)
                self.assertEqual(source.count('ui-product-lookup-submit'), 1)
                self.assertNotIn('id="name_query"', source)
                self.assertNotIn('id="barcode"', source)
                self.assertIn('ProductLookup.', source)

        inventory = (template_root / 'inventory_display.html').read_text(encoding='utf-8')
        self.assertEqual(inventory.count('id="product-search"'), 1)
        self.assertEqual(inventory.count('name="q"'), 1)
        self.assertNotIn('id="barcode-search"', inventory)
        self.assertNotIn('id="name-search"', inventory)

        base = (template_root / 'base.html').read_text(encoding='utf-8')
        self.assertIn("{% static 'js/product_lookup.js' %}", base)
        self.assertTrue(
            (Path(settings.BASE_DIR) / 'static' / 'js' / 'product_lookup.js').is_file()
        )


class UnifiedProductLookupTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='lookup-user', password='test-password', is_staff=True,
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='Lookup category')
        self.product = Product.objects.create(
            name='Unified Aspirin', item_number='SKU-ALPHA', barcode='001234567890',
            price='5.00', quantity_in_stock=7, category=self.category,
        )
        self.other = Product.objects.create(
            name='Different Product', item_number='SKU-BETA', barcode='998877665544',
            price='8.00', quantity_in_stock=2, category=self.category,
        )
        self.checkin_session = CheckinSession.objects.create(
            user=self.user, scanned_by='Lookup tester',
        )
        checkout = CheckoutOrder.objects.create(
            user=self.user, active_session_key=self.client.session.session_key,
        )
        session = self.client.session
        session['checkout_id'] = checkout.pk
        session.save()

    def _inventory_product_ids(self, query):
        response = self.client.get(reverse('inventory_display'), {'q': query})
        self.assertEqual(response.status_code, 200)
        return [product.pk for product in response.context['page_obj'].object_list]

    def test_inventory_combined_query_searches_name_sku_and_barcode(self):
        queries = ('aspirin', 'SKU-ALPHA', '1234567890')
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(self._inventory_product_ids(query), [self.product.pk])

    def test_inventory_combined_query_supports_ajax_export_and_old_links(self):
        ajax = self.client.get(
            reverse('inventory_display'), {'q': 'SKU-ALPHA'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )
        self.assertEqual(ajax.status_code, 200)
        self.assertEqual(ajax.json()['count'], 1)
        self.assertIn('Unified Aspirin', ajax.json()['html'])

        exported = self.client.get(reverse('export_inventory_csv'), {'q': '1234567890'})
        csv_text = exported.content.decode('utf-8')
        self.assertIn('Unified Aspirin', csv_text)
        self.assertNotIn('Different Product', csv_text)

        legacy = self.client.get(
            reverse('inventory_display'), {'barcode_query': '001234567890'},
        )
        self.assertEqual(
            [product.pk for product in legacy.context['page_obj'].object_list],
            [self.product.pk],
        )

    def test_all_five_workflow_pages_render_one_lookup_control(self):
        urls = (
            reverse('create_order'),
            reverse('checkout_cart'),
            reverse('checkin_session', args=[self.checkin_session.pk]),
            reverse('expired_products') + '?mode=log',
        )
        for url in urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'id="product_lookup"', count=1)
                self.assertContains(response, '>Go</button>', count=1)
                self.assertNotContains(response, 'id="name_query"')
                self.assertNotContains(response, 'id="barcode"')

        inventory = self.client.get(reverse('inventory_display'))
        self.assertContains(inventory, 'id="product-search"', count=1)
        self.assertContains(inventory, 'name="q"', count=1)


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
