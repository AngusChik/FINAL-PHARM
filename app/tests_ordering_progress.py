from datetime import date, timedelta
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from app.models import OrderingSheetEntry, OrderingSheetStatusEvent


class OrderingProgressDetailsTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='ordering-progress-admin',
            password='test-password',
            is_staff=True,
        )
        self.client.force_login(self.user)
        self.url = reverse('ordering_sheet')

    def create_entry(self, *, name='Progress Drug', status=OrderingSheetEntry.STATUS_PENDING,
                     ordered=None, received=0):
        return OrderingSheetEntry.objects.create(
            name=name,
            entry_type=OrderingSheetEntry.ENTRY_DRUG,
            reasoning=OrderingSheetEntry.REASON_STOCK,
            urgency=OrderingSheetEntry.URGENCY_LOW,
            initials='AB',
            status=status,
            quantity_ordered=ordered,
            quantity_received=received,
            created_by=self.user,
        )

    def post_progress(self, entry, status, *, supplier='McKesson', ordered='5', received=None):
        data = {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': status,
            'supplier_name': supplier,
            'quantity_ordered': ordered,
            'expected_date': '',
            'order_note': '',
        }
        if received is not None:
            data['quantity_received'] = received
        return self.client.post(self.url, data)

    def test_full_and_embedded_views_render_supplier_dropdown_and_one_ordered_field(self):
        self.create_entry()

        for suffix in ('', '?embed=1'):
            with self.subTest(suffix=suffix):
                response = self.client.get(f'{self.url}{suffix}')
                html = response.content.decode()

                self.assertEqual(response.status_code, 200)
                self.assertIn('<select name="supplier_name"', html)
                self.assertNotIn('<input type="text" name="supplier_name"', html)
                self.assertIn('<option value="McKesson"', html)
                self.assertIn('<option value="K&amp;F"', html)
                self.assertIn('<option value="Direct"', html)
                self.assertEqual(
                    html.count('<input type="number" name="quantity_ordered"'),
                    1,
                )
                self.assertEqual(html.count('name="status_only" value="0"'), 1)
                self.assertIn('>Qty ordered</span>', html)
                self.assertIn('>Qty received so far</span>', html)
                self.assertIn('os-quantity-received-field wide" hidden', html)

    def test_every_status_is_exposed_on_its_rendered_row(self):
        entries = []
        for index, (status, _label) in enumerate(OrderingSheetEntry.STATUS_CHOICES):
            entries.append(self.create_entry(name=f'Status Drug {index}', status=status))

        response = self.client.get(f'{self.url}?view=all')
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        for entry in entries:
            with self.subTest(status=entry.status):
                row_start = html.index(f'<tr data-entry-id="{entry.pk}"')
                row = html[row_start:html.index('</tr>', row_start)]
                self.assertIn(f'data-status="{entry.status}"', row)
                self.assertNotIn('class="row-', row)

    def test_reasoning_is_plain_text_for_drug_and_otc_rows(self):
        drug = self.create_entry(name='Plain Reason Drug')
        otc = OrderingSheetEntry.objects.create(
            name='Plain Reason OTC',
            entry_type=OrderingSheetEntry.ENTRY_OTC,
            side=OrderingSheetEntry.SIDE_LEFT,
            urgency=OrderingSheetEntry.URGENCY_NA,
            initials='AB',
            created_by=self.user,
        )

        html = self.client.get(f'{self.url}?view=all').content.decode()
        for entry, expected in ((drug, 'Order for stock'), (otc, 'OTC &middot; Left')):
            with self.subTest(entry=entry.name):
                row_start = html.index(f'<tr data-entry-id="{entry.pk}"')
                row = html[row_start:html.index('</tr>', row_start)]
                self.assertIn('class="os-reason-text"', row)
                self.assertIn(expected, row)
                self.assertNotIn('class="pill reason-', row)
                self.assertNotIn('otc-pill', row)

    def test_google_marker_precedes_name_without_a_form_badge(self):
        entry = self.create_entry(name='Google Imported Drug')
        entry.source = OrderingSheetEntry.SOURCE_GSHEET
        entry.save(update_fields=['source'])

        for suffix in ('?view=all', '?embed=1&view=all'):
            with self.subTest(suffix=suffix):
                html = self.client.get(f'{self.url}{suffix}').content.decode()
                row_start = html.index(f'<tr data-entry-id="{entry.pk}"')
                row = html[row_start:html.index('</tr>', row_start)]
                self.assertIn(
                    'class="os-google-source" role="img" aria-label="Added via Google Sheet"',
                    row,
                )
                self.assertLess(row.index('os-google-source'), row.index('os-drug-name'))
                self.assertNotIn('gsheet-pill', row)
                self.assertNotIn('>Form</span>', row)

    def test_each_supported_supplier_is_saved_exactly(self):
        for index, supplier in enumerate(('McKesson', 'K&F', 'Direct')):
            with self.subTest(supplier=supplier):
                entry = self.create_entry(name=f'Supplier Drug {index}')
                response = self.post_progress(
                    entry,
                    OrderingSheetEntry.STATUS_ORDERED,
                    supplier=supplier,
                    ordered='5',
                )

                self.assertEqual(response.status_code, 302)
                entry.refresh_from_db()
                self.assertEqual(entry.status, OrderingSheetEntry.STATUS_ORDERED)
                self.assertEqual(entry.supplier_name, supplier)
                self.assertEqual(entry.quantity_ordered, 5)
                self.assertEqual(entry.quantity_received, 0)

    def test_forged_supplier_is_rejected_without_mutating_progress(self):
        entry = self.create_entry()

        response = self.post_progress(
            entry,
            OrderingSheetEntry.STATUS_ORDERED,
            supplier='Unknown wholesaler',
        )

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PENDING)
        self.assertEqual(entry.supplier_name, '')
        self.assertIsNone(entry.quantity_ordered)

    def test_existing_noncanonical_supplier_is_preserved_until_reselected(self):
        entry = self.create_entry()
        entry.supplier_name = 'Kohl & Frisch legacy'
        entry.save(update_fields=['supplier_name'])

        page = self.client.get(self.url)
        self.assertContains(page, 'Kohl &amp; Frisch legacy (existing)')

        response = self.post_progress(
            entry,
            OrderingSheetEntry.STATUS_ORDERED,
            supplier='Kohl & Frisch legacy',
            ordered='5',
        )

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.supplier_name, 'Kohl & Frisch legacy')
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_ORDERED)

    def test_ordered_status_requires_a_positive_ordered_quantity(self):
        entry = self.create_entry()

        self.post_progress(
            entry,
            OrderingSheetEntry.STATUS_ORDERED,
            supplier='Direct',
            ordered='0',
        )

        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PENDING)
        self.assertIsNone(entry.quantity_ordered)
        self.assertEqual(entry.supplier_name, '')

    def test_status_change_without_required_progress_persists_and_records_event(self):
        entry = self.create_entry()

        response = self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_NOT_FOR_SALE,
            'status_only': '1',
        })

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_NOT_FOR_SALE)
        event = OrderingSheetStatusEvent.objects.get(entry=entry)
        self.assertEqual(event.from_status, OrderingSheetEntry.STATUS_PENDING)
        self.assertEqual(event.to_status, OrderingSheetEntry.STATUS_NOT_FOR_SALE)
        self.assertEqual(event.changed_by, self.user)

    def test_not_for_sale_dropdown_exposes_every_admin_status(self):
        entry = self.create_entry(status=OrderingSheetEntry.STATUS_NOT_FOR_SALE)

        for suffix in ('', '?embed=1'):
            with self.subTest(suffix=suffix):
                response = self.client.get(f'{self.url}{suffix}')
                rendered_entry = next(
                    item for item in response.context['entries'] if item.pk == entry.pk
                )
                self.assertEqual(
                    [value for value, _label in rendered_entry.status_options],
                    OrderingSheetEntry.ADMIN_STATUS_CHOICES,
                )

    def test_not_for_sale_can_be_reopened_and_clears_completion(self):
        entry = self.create_entry()

        self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_NOT_FOR_SALE,
            'status_only': '1',
        })
        entry.refresh_from_db()
        self.assertIsNotNone(entry.completed_at)

        response = self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_PENDING,
            'status_only': '1',
        })

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PENDING)
        self.assertIsNone(entry.completed_at)
        self.assertTrue(
            OrderingSheetStatusEvent.objects.filter(
                entry=entry,
                from_status=OrderingSheetEntry.STATUS_NOT_FOR_SALE,
                to_status=OrderingSheetEntry.STATUS_PENDING,
                changed_by=self.user,
            ).exists()
        )

    def test_not_for_sale_can_change_to_ordered_without_progress_details(self):
        entry = self.create_entry(status=OrderingSheetEntry.STATUS_NOT_FOR_SALE)
        entry.supplier_name = OrderingSheetEntry.SUPPLIER_DIRECT
        entry.order_note = 'Preserve these details'
        entry.save(update_fields=['supplier_name', 'order_note'])

        response = self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_ORDERED,
            'status_only': '1',
        })

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_ORDERED)
        self.assertIsNone(entry.quantity_ordered)
        self.assertEqual(entry.supplier_name, OrderingSheetEntry.SUPPLIER_DIRECT)
        self.assertEqual(entry.order_note, 'Preserve these details')
        self.assertIsNone(entry.completed_at)
        self.assertTrue(
            OrderingSheetStatusEvent.objects.filter(
                entry=entry,
                from_status=OrderingSheetEntry.STATUS_NOT_FOR_SALE,
                to_status=OrderingSheetEntry.STATUS_ORDERED,
            ).exists()
        )

    def test_other_terminal_statuses_remain_locked(self):
        for status in (
            OrderingSheetEntry.STATUS_PICKED_UP,
            OrderingSheetEntry.STATUS_CANCELLED,
        ):
            with self.subTest(status=status):
                entry = self.create_entry(status=status)
                self.assertFalse(
                    entry.can_transition_to(OrderingSheetEntry.STATUS_PENDING)
                )

    def test_status_only_ordered_change_needs_no_quantity_and_preserves_details(self):
        expected_date = date.today() + timedelta(days=4)
        entry = self.create_entry()
        entry.supplier_name = OrderingSheetEntry.SUPPLIER_MCKESSON
        entry.expected_date = expected_date
        entry.order_note = 'Keep these saved details'
        entry.save(update_fields=['supplier_name', 'expected_date', 'order_note'])

        response = self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_ORDERED,
            'status_only': '1',
            # Status-only saves must ignore unfinished/stale detail controls.
            'supplier_name': 'Unknown wholesaler',
            'quantity_ordered': '',
            'expected_date': 'not-a-date',
            'order_note': 'Do not overwrite',
        })

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_ORDERED)
        self.assertIsNone(entry.quantity_ordered)
        self.assertEqual(entry.supplier_name, OrderingSheetEntry.SUPPLIER_MCKESSON)
        self.assertEqual(entry.expected_date, expected_date)
        self.assertEqual(entry.order_note, 'Keep these saved details')
        event = OrderingSheetStatusEvent.objects.get(entry=entry)
        self.assertEqual(event.from_status, OrderingSheetEntry.STATUS_PENDING)
        self.assertEqual(event.to_status, OrderingSheetEntry.STATUS_ORDERED)

    def test_status_change_missing_required_quantity_does_not_persist_or_record_event(self):
        entry = self.create_entry()

        response = self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_ORDERED,
        })

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PENDING)
        self.assertIsNone(entry.quantity_ordered)
        self.assertFalse(OrderingSheetStatusEvent.objects.filter(entry=entry).exists())

    def test_full_received_status_uses_ordered_quantity_automatically(self):
        entry = self.create_entry(
            status=OrderingSheetEntry.STATUS_ORDERED,
            ordered=5,
        )

        response = self.post_progress(
            entry,
            OrderingSheetEntry.STATUS_RECEIVED,
            supplier='K&F',
            ordered='5',
        )

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_RECEIVED)
        self.assertEqual(entry.quantity_received, 5)

    def test_partial_received_requires_a_partial_value_and_preserves_it_afterward(self):
        entry = self.create_entry(
            status=OrderingSheetEntry.STATUS_ORDERED,
            ordered=5,
        )

        self.post_progress(
            entry,
            OrderingSheetEntry.STATUS_PARTIAL_RECEIVED,
            ordered='5',
        )
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_ORDERED)
        self.assertEqual(entry.quantity_received, 0)

        self.post_progress(
            entry,
            OrderingSheetEntry.STATUS_PARTIAL_RECEIVED,
            ordered='5',
            received='2',
        )
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PARTIAL_RECEIVED)
        self.assertEqual(entry.quantity_received, 2)

        self.post_progress(
            entry,
            OrderingSheetEntry.STATUS_BACKORDERED,
            ordered='5',
        )
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_BACKORDERED)
        self.assertEqual(entry.quantity_received, 2)


class OrderingProgressClientContractTests(SimpleTestCase):
    def test_received_field_visibility_is_reinitialized_after_seamless_refresh(self):
        template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

        self.assertIn("status.value === 'partial_received'", template)
        self.assertIn('input.disabled = !isPartial;', template)
        self.assertIn('input.required = isPartial;', template)
        self.assertIn('orderedInput.required = needsOrdered;', template)
        self.assertGreaterEqual(template.count('syncAllReceivedQuantityFields(osTbody);'), 2)

    def test_status_selection_autosaves_status_only_without_detail_validation(self):
        template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

        helper_start = template.index('function saveSelectedStatus(statusSelect) {')
        helper_end = template.index('\n    syncAllReceivedQuantityFields(osTbody);', helper_start)
        helper = template[helper_start:helper_end]

        self.assertIn('syncReceivedQuantityField(form);', helper)
        self.assertIn(
            'if (statusSelect.value === statusSelect.dataset.current) return;',
            helper,
        )
        self.assertIn("form.querySelector('input[name=\"status_only\"]')", helper)
        self.assertIn("statusOnly.value = '1';", helper)
        self.assertIn("form.querySelectorAll('.os-progress-details input,", helper)
        self.assertIn('control.disabled = true;', helper)
        self.assertIn('control.disabled = disabledStates[index];', helper)
        self.assertIn('form.requestSubmit();', helper)
        self.assertNotIn('form.checkValidity()', helper)
        self.assertNotIn('form.reportValidity()', helper)
        self.assertNotIn("form.querySelector('.os-progress-details')", helper)

        delegated_change = template.index(
            "if (e.target.classList.contains('os-status-select')) {"
        )
        delegated_end = template.index(
            "if (e.target.classList.contains('os-row-check'))",
            delegated_change,
        )
        self.assertIn(
            'saveSelectedStatus(e.target);',
            template[delegated_change:delegated_end],
        )
        self.assertIn("document.addEventListener('ui:seamless-error'", template)
        self.assertIn("action.value !== 'update_status'", template)
        self.assertIn("statusOnly.value = '0';", template)
        self.assertIn('statusSelect.value = statusSelect.dataset.current;', template)


class OrderingRowPresentationContractTests(SimpleTestCase):
    def setUp(self):
        self.template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

    def test_all_statuses_define_whole_row_colors_including_sticky_cells(self):
        for status, _label in OrderingSheetEntry.STATUS_CHOICES:
            with self.subTest(status=status):
                selector = f'.active-table tbody tr[data-status="{status}"]'
                selector_start = self.template.index(selector)
                selector_end = self.template.index('}', selector_start)
                self.assertIn('--os-row-bg:', self.template[selector_start:selector_end])
        self.assertIn(
            '.active-table tbody tr > td { background: var(--os-row-bg); }',
            self.template,
        )
        self.assertGreaterEqual(
            self.template.count('background: var(--os-row-bg, #fff);'),
            2,
        )
        self.assertNotIn('tr.row-high   td', self.template)
        self.assertNotIn('tr.row-medium td', self.template)
        self.assertNotIn('tr.row-low    td', self.template)

    def test_reason_boxes_and_left_edge_accents_are_removed(self):
        self.assertNotIn('.reason-stock', self.template)
        self.assertNotIn('.reason-basket', self.template)
        self.assertNotIn('td:first-child { box-shadow: inset 3px', self.template)
        self.assertNotIn('class="pill reason-', self.template)
        self.assertNotIn('class="pill otc-pill"', self.template)
        self.assertIn('class="os-reason-text"', self.template)
        reason_start = self.template.index('.os-reason-text {')
        reason_end = self.template.index('\n    }', reason_start)
        reason_css = self.template[reason_start:reason_end]
        self.assertNotIn('background:', reason_css)
        self.assertNotIn('border:', reason_css)
        self.assertNotIn('padding:', reason_css)

    def test_name_receives_space_while_reason_and_urgency_stay_compact(self):
        self.assertIn(
            '.active-table .os-name-col { width: 28%; min-width: 300px; }',
            self.template,
        )
        self.assertIn(
            '.active-table .os-reason-col { width: 140px; min-width: 140px; max-width: 140px; }',
            self.template,
        )
        self.assertIn(
            '.active-table .os-urgency-col { width: 150px; min-width: 150px; max-width: 150px; }',
            self.template,
        )
        drug_name_start = self.template.index('.os-drug-name {')
        drug_name_end = self.template.index('\n    }', drug_name_start)
        drug_name_css = self.template[drug_name_start:drug_name_end]
        self.assertIn('white-space: nowrap;', drug_name_css)
        self.assertNotIn('text-overflow:', drug_name_css)
        self.assertNotIn('overflow: hidden;', drug_name_css)
        self.assertIn('min-width: max-content;', self.template)
        self.assertIn('os-sortable os-name-col', self.template)
        self.assertIn('os-sortable os-reason-col', self.template)
        self.assertIn('os-sortable os-urgency-col', self.template)
