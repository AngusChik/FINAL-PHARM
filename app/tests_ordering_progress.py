from datetime import date, datetime, timedelta, timezone as datetime_timezone
from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone

from app.models import OrderingSheetEntry, OrderingSheetStatusEvent, UserAction


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
                     ordered=None, received=0, custom_status_text=''):
        if status == OrderingSheetEntry.STATUS_CUSTOM and not custom_status_text:
            custom_status_text = 'Custom status'
        return OrderingSheetEntry.objects.create(
            name=name,
            entry_type=OrderingSheetEntry.ENTRY_DRUG,
            reasoning=OrderingSheetEntry.REASON_STOCK,
            urgency=OrderingSheetEntry.URGENCY_LOW,
            initials='AB',
            status=status,
            custom_status_text=custom_status_text,
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

    def test_full_and_embedded_views_render_status_without_order_details(self):
        entry = self.create_entry()
        entry.order_note = 'Call supplier before ordering'
        entry.save(update_fields=['order_note'])
        OrderingSheetStatusEvent.objects.create(
            entry=entry,
            from_status=OrderingSheetEntry.STATUS_PENDING,
            to_status=OrderingSheetEntry.STATUS_PENDING,
            changed_by=self.user,
        )

        for suffix in ('', '?embed=1'):
            with self.subTest(suffix=suffix):
                response = self.client.get(f'{self.url}{suffix}')
                html = response.content.decode()

                self.assertEqual(response.status_code, 200)
                self.assertIn('<select name="status"', html)
                self.assertEqual(html.count('name="status_only" value="1"'), 2)
                self.assertIn(
                    'data-column-key="status-actions" '
                    'data-column-label="Status and actions">Status / Actions',
                    html,
                )
                self.assertNotIn('>Status<span class="sort-ind"></span></th>', html)
                row_start = html.index(f'<tr data-entry-id="{entry.pk}"')
                row = html[row_start:html.index('</tr>', row_start)]
                actions_start = row.index('<td class="td-actions2 os-actions-cell"')
                actions_cell = row[actions_start:row.index('</td>', actions_start)]
                self.assertNotIn('<select name="status"', row[:actions_start])
                self.assertIn('<select name="status"', actions_cell)
                self.assertIn('class="os-action-buttons"', actions_cell)
                self.assertIn('Call supplier before ordering', actions_cell)
                self.assertIn('class="os-status-updated"', actions_cell)
                self.assertIn('class="os-note-form"', actions_cell)
                self.assertNotIn('class="os-progress-details"', html)
                self.assertNotIn('<summary>Order details</summary>', html)
                self.assertNotIn('<select name="supplier_name"', html)
                self.assertNotIn('<input type="number" name="quantity_ordered"', html)
                self.assertNotIn('<input type="date" name="expected_date"', html)
                self.assertNotIn('>Qty received so far</span>', html)

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

    def test_initial_precedes_date_in_one_column_for_full_and_embedded_views(self):
        entry = self.create_entry()

        for suffix in ('?view=all', '?embed=1&view=all'):
            with self.subTest(suffix=suffix):
                html = self.client.get(f'{self.url}{suffix}').content.decode()
                self.assertIn(
                    'data-column-key="initial-date" '
                    'data-column-label="Initial - Date">',
                    html,
                )
                self.assertEqual(html.count('<th class="os-sortable '), 3)
                for label in ('drug name', 'reasoning', 'entry date'):
                    self.assertIn(
                        'class="os-sort-button" '
                        f'aria-label="Sort by {label}"',
                        html,
                    )
                row_start = html.index(f'<tr data-entry-id="{entry.pk}"')
                row = html[row_start:html.index('</tr>', row_start)]
                cell_start = row.index('<td class="os-initial-date-col"')
                cell = row[cell_start:row.index('</td>', cell_start)]
                self.assertIn('class="os-initials">AB</span>', cell)
                self.assertIn('class="os-time">', cell)
                self.assertLess(cell.index('os-initials'), cell.index('os-time'))

    def test_entry_date_display_and_sort_share_the_local_calendar_day(self):
        early = self.create_entry(name='Early entry')
        late = self.create_entry(name='Late entry')
        OrderingSheetEntry.objects.filter(pk=early.pk).update(
            created_at=datetime(2026, 9, 12, 9, 0, tzinfo=datetime_timezone.utc),
            initials='ZZ',
        )
        OrderingSheetEntry.objects.filter(pk=late.pk).update(
            created_at=datetime(2026, 9, 13, 2, 30, tzinfo=datetime_timezone.utc),
            initials='AA',
        )
        with timezone.override('America/Toronto'):
            for suffix in ('?view=all', '?embed=1&view=all'):
                with self.subTest(suffix=suffix):
                    html = self.client.get(f'{self.url}{suffix}').content.decode()
                    self.assertNotIn('os-entry-date-filter', html)
                    self.assertIn('aria-label="Sort by entry date"', html)
                    for entry in (early, late):
                        start = html.index(f'<tr data-entry-id="{entry.pk}"')
                        row = html[start:html.index('</tr>', start)]
                        self.assertIn('class="os-initial-date-col" data-sort="20260912"', row)
                        self.assertIn('class="os-time">12/09/2026</span>', row)

    def test_all_views_start_with_latest_additions_regardless_of_type_or_urgency(self):
        entries = []
        for name, day, hour, entry_type, urgency, status in (
            ('Old urgent drug', 11, 12, 'drug', 'high', 'pending'),
            ('Earlier urgent drug', 12, 12, 'drug', 'high', 'ordered'),
            ('Later OTC item', 12, 18, 'otc', 'na', 'pending'),
            ('Latest low urgency drug', 13, 12, 'drug', 'low', 'pending'),
            ('Latest tied OTC item', 13, 12, 'otc', 'na', 'pending'),
            ('Recent completed item', 14, 12, 'otc', 'na', 'picked_up'),
            ('Old completed item', 10, 12, 'drug', 'high', 'cancelled'),
        ):
            entry = self.create_entry(name=name, status=status)
            OrderingSheetEntry.objects.filter(pk=entry.pk).update(
                created_at=datetime(2026, 9, day, hour, tzinfo=datetime_timezone.utc),
                entry_type=entry_type, urgency=urgency,
            )
            entries.append(entry)

        expected_indexes = {
            'active': [4, 3, 2, 1, 0],
            'completed': [5, 6],
            'all': [5, 4, 3, 2, 1, 0, 6],
        }
        for view, indexes in expected_indexes.items():
            for embed in ('', '&embed=1'):
                with self.subTest(view=view, embed=embed):
                    response = self.client.get(f'{self.url}?view={view}{embed}')
                    self.assertEqual(response.status_code, 200)
                    expected = [entries[index].pk for index in indexes]
                    self.assertEqual([entry.pk for entry in response.context['entries']], expected)
                    html = response.content.decode()
                    positions = [html.index(f'<tr data-entry-id="{pk}"') for pk in expected]
                    self.assertEqual(positions, sorted(positions))

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
                self.assertLess(row.index('os-google-source'), row.index('os-urgency-meter'))
                self.assertLess(row.index('os-urgency-meter'), row.index('os-drug-name'))
                self.assertNotIn('gsheet-pill', row)
                self.assertNotIn('>Form</span>', row)

    def test_three_tier_urgency_meter_precedes_name_for_every_level(self):
        for index, (urgency, label) in enumerate(OrderingSheetEntry.URGENCY_CHOICES):
            entry = self.create_entry(name=f'Urgency Drug {index}')
            entry.urgency = urgency
            entry.save(update_fields=['urgency'])

            for suffix in ('?view=all', '?embed=1&view=all'):
                with self.subTest(urgency=urgency, suffix=suffix):
                    html = self.client.get(f'{self.url}{suffix}').content.decode()
                    row_start = html.index(f'<tr data-entry-id="{entry.pk}"')
                    row = html[row_start:html.index('</tr>', row_start)]
                    self.assertIn(
                        f'class="os-urgency-meter os-urgency-meter--{urgency}"',
                        row,
                    )
                    self.assertIn(f'aria-label="Urgency: {label}"', row)
                    self.assertLess(row.index('os-urgency-meter'), row.index('os-drug-name'))
                    self.assertEqual(row.count('<span aria-hidden="true"></span>'), 3)
                    self.assertNotIn('os-urgency-col', row)

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

    def test_existing_noncanonical_supplier_is_hidden_but_backend_preserves_it(self):
        entry = self.create_entry()
        entry.supplier_name = 'Kohl & Frisch legacy'
        entry.save(update_fields=['supplier_name'])

        page = self.client.get(self.url)
        self.assertNotContains(page, 'Kohl &amp; Frisch legacy (existing)')

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

    def test_custom_text_option_and_editor_render_in_full_and_embedded_views(self):
        entry = self.create_entry()

        for suffix in ('?view=all', '?embed=1&view=all'):
            with self.subTest(suffix=suffix):
                response = self.client.get(f'{self.url}{suffix}')
                rendered_entry = next(
                    item for item in response.context['entries'] if item.pk == entry.pk
                )
                self.assertIn(
                    (OrderingSheetEntry.STATUS_CUSTOM, 'Custom text'),
                    rendered_entry.status_options,
                )
                html = response.content.decode()
                row_start = html.index(f'<tr data-entry-id="{entry.pk}"')
                row = html[row_start:html.index('</tr>', row_start)]
                self.assertIn('<option value="custom"', row)
                self.assertIn(f'id="os-custom-status-{entry.pk}"', row)
                self.assertIn(f'id="os-custom-status-input-{entry.pk}"', row)
                self.assertIn('placeholder="Type custom status" maxlength="80"', row)
                self.assertIn('class="os-custom-status-form"', row)
                self.assertIn('hidden', row)

    def test_custom_status_is_trimmed_audited_and_keeps_progress_details(self):
        expected_date = date.today() + timedelta(days=3)
        entry = self.create_entry()
        entry.supplier_name = OrderingSheetEntry.SUPPLIER_DIRECT
        entry.quantity_ordered = 7
        entry.expected_date = expected_date
        entry.order_note = 'Keep this note'
        entry.save(update_fields=[
            'supplier_name', 'quantity_ordered', 'expected_date', 'order_note',
        ])

        response = self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_CUSTOM,
            'custom_status_text': '  Waiting for insurance approval  ',
            'status_only': '1',
            # Custom status saves remain status-only and ignore stale details.
            'supplier_name': 'Forged supplier',
            'quantity_ordered': '999',
            'expected_date': 'not-a-date',
            'order_note': 'Do not replace',
        })

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_CUSTOM)
        self.assertEqual(entry.custom_status_text, 'Waiting for insurance approval')
        self.assertEqual(entry.status_display, 'Waiting for insurance approval')
        self.assertEqual(entry.supplier_name, OrderingSheetEntry.SUPPLIER_DIRECT)
        self.assertEqual(entry.quantity_ordered, 7)
        self.assertEqual(entry.expected_date, expected_date)
        self.assertEqual(entry.order_note, 'Keep this note')
        self.assertIsNone(entry.completed_at)
        event = OrderingSheetStatusEvent.objects.get(entry=entry)
        self.assertEqual(event.from_status, OrderingSheetEntry.STATUS_PENDING)
        self.assertEqual(event.to_status, OrderingSheetEntry.STATUS_CUSTOM)
        self.assertEqual(event.note, 'Custom status: Waiting for insurance approval')
        self.assertTrue(UserAction.objects.filter(
            user=self.user,
            action='ordering_status_update',
            target=entry.name,
            detail='Status → Waiting for insurance approval',
        ).exists())
        for suffix in ('?view=all', '?embed=1&view=all'):
            with self.subTest(suffix=suffix):
                html = self.client.get(f'{self.url}{suffix}').content.decode()
                row_start = html.index(f'<tr data-entry-id="{entry.pk}"')
                row = html[row_start:html.index('</tr>', row_start)]
                self.assertIn(
                    '<option value="custom" selected>Waiting for insurance approval</option>',
                    row,
                )
                self.assertIn('<option value="edit_custom">Edit custom text…</option>', row)
                self.assertIn('aria-expanded="false"', row)
                self.assertRegex(
                    row,
                    r'<form[^>]*class="os-custom-status-form"[^>]*\s+hidden>',
                )
                self.assertRegex(
                    row,
                    r'<input[^>]*class="os-custom-status-input"[^>]*\s+disabled>',
                )

    def post_fast_custom_status(self, entry, text, **kwargs):
        return self.client.post(self.url, {
            'action': 'update_status', 'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_CUSTOM, 'status_only': '1',
            'custom_status_text': text,
        }, HTTP_X_ORDERING_STATUS='custom', **kwargs)

    def test_fast_custom_status_returns_small_json_without_rendering_the_table(self):
        entry = self.create_entry()
        with patch('app.views.OrderingSheetView._render_page', side_effect=AssertionError('Full render')):
            response = self.post_fast_custom_status(entry, '  Call supplier  ')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/json')
        self.assertLess(len(response.content), 1000)
        self.assertEqual(response.json()['custom_status_text'], 'Call supplier')
        self.assertEqual(response.json()['entry_id'], entry.pk)
        self.assertEqual(response.json()['status'], OrderingSheetEntry.STATUS_CUSTOM)
        self.assertIn(self.user.username, response.json()['updated_text'])
        entry.refresh_from_db()
        self.assertEqual(entry.custom_status_text, 'Call supplier')
        self.assertEqual(entry.status_events.get().note, 'Custom status: Call supplier')
        self.assertEqual(UserAction.objects.filter(action='ordering_status_update').count(), 1)

    def test_fast_custom_status_rejects_invalid_text_and_preserves_old_value(self):
        entry = self.create_entry(status=OrderingSheetEntry.STATUS_CUSTOM, custom_status_text='Keep me')
        for text in ('  ', 'X' * 81):
            with self.subTest(text=text):
                response = self.post_fast_custom_status(entry, text)
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.json()['ok'])
                self.assertTrue(response.json()['error'])
        entry.refresh_from_db()
        self.assertEqual(entry.custom_status_text, 'Keep me')
        self.assertFalse(entry.status_events.exists())

    def test_fast_custom_status_keeps_permissions_and_transition_validation(self):
        entry = self.create_entry(status=OrderingSheetEntry.STATUS_PICKED_UP)
        response = self.post_fast_custom_status(entry, 'Must not reopen')
        self.assertEqual(response.status_code, 400)
        self.user.is_staff = False
        self.user.save(update_fields=['is_staff'])
        response = self.post_fast_custom_status(entry, 'Must not change')
        self.assertEqual(response.status_code, 403)
        self.assertFalse(response.json()['ok'])
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PICKED_UP)
        self.assertFalse(entry.status_events.exists())

    def test_fast_custom_status_noop_does_not_repeat_audit_and_works_embedded(self):
        entry = self.create_entry(status=OrderingSheetEntry.STATUS_CUSTOM, custom_status_text='Already saved')
        self.url += '?embed=1'
        response = self.post_fast_custom_status(entry, ' Already saved ')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['ok'])
        self.assertEqual(response.json()['custom_status_text'], 'Already saved')
        self.assertFalse(entry.status_events.exists())

    def test_custom_status_rejects_blank_and_oversized_text_without_mutating(self):
        max_length = OrderingSheetEntry._meta.get_field('custom_status_text').max_length
        for index, custom_text in enumerate(('   ', 'x' * (max_length + 1))):
            with self.subTest(custom_text_length=len(custom_text)):
                entry = self.create_entry(name=f'Invalid Custom {index}')
                response = self.client.post(self.url, {
                    'action': 'update_status',
                    'entry_id': str(entry.pk),
                    'status': OrderingSheetEntry.STATUS_CUSTOM,
                    'custom_status_text': custom_text,
                    'status_only': '1',
                })

                self.assertEqual(response.status_code, 302)
                entry.refresh_from_db()
                self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PENDING)
                self.assertEqual(entry.custom_status_text, '')
                self.assertIsNone(entry.status_updated_at)
                self.assertFalse(OrderingSheetStatusEvent.objects.filter(entry=entry).exists())
                self.assertFalse(UserAction.objects.filter(
                    action='ordering_status_update', target=entry.name,
                ).exists())

    def test_custom_status_accepts_its_exact_maximum_length(self):
        entry = self.create_entry()
        max_length = OrderingSheetEntry._meta.get_field('custom_status_text').max_length
        custom_text = 'x' * max_length

        self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_CUSTOM,
            'custom_status_text': custom_text,
            'status_only': '1',
        })

        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_CUSTOM)
        self.assertEqual(entry.custom_status_text, custom_text)

    def test_existing_custom_status_can_be_edited_and_unchanged_text_is_a_noop(self):
        entry = self.create_entry(
            status=OrderingSheetEntry.STATUS_CUSTOM,
            custom_status_text='Waiting for prescriber',
        )

        self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_CUSTOM,
            'custom_status_text': 'Waiting for patient',
            'status_only': '1',
        })
        entry.refresh_from_db()
        changed_at = entry.status_updated_at
        self.assertEqual(entry.custom_status_text, 'Waiting for patient')
        self.assertEqual(OrderingSheetStatusEvent.objects.filter(entry=entry).count(), 1)

        self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_CUSTOM,
            'custom_status_text': '  Waiting for patient  ',
            'status_only': '1',
        })
        entry.refresh_from_db()
        self.assertEqual(entry.status_updated_at, changed_at)
        self.assertEqual(OrderingSheetStatusEvent.objects.filter(entry=entry).count(), 1)

    def test_leaving_custom_status_clears_its_text(self):
        entry = self.create_entry(
            status=OrderingSheetEntry.STATUS_CUSTOM,
            custom_status_text='Waiting for patient',
        )

        response = self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_PENDING,
            'status_only': '1',
        })

        self.assertEqual(response.status_code, 302)
        entry.refresh_from_db()
        self.assertEqual(entry.status, OrderingSheetEntry.STATUS_PENDING)
        self.assertEqual(entry.custom_status_text, '')
        self.assertIsNone(entry.completed_at)

    def test_non_admin_sees_escaped_custom_label_and_cannot_change_it(self):
        custom_text = '<img src=x onerror=alert(1)>'
        entry = self.create_entry(
            status=OrderingSheetEntry.STATUS_CUSTOM,
            custom_status_text=custom_text,
        )
        self.user.is_staff = False
        self.user.save(update_fields=['is_staff'])

        response = self.client.get(f'{self.url}?view=all')
        html = response.content.decode()
        self.assertNotIn(custom_text, html)
        self.assertIn('&lt;img src=x onerror=alert(1)&gt;', html)
        self.assertNotIn('class="os-custom-status-form"', html)

        self.client.post(self.url, {
            'action': 'update_status',
            'entry_id': str(entry.pk),
            'status': OrderingSheetEntry.STATUS_CUSTOM,
            'custom_status_text': 'Changed without permission',
            'status_only': '1',
        })
        entry.refresh_from_db()
        self.assertEqual(entry.custom_status_text, custom_text)

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
                self.assertFalse(
                    entry.can_transition_to(OrderingSheetEntry.STATUS_CUSTOM)
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
    def test_full_page_and_embed_toolbar_keep_search_in_desktop_grid(self):
        template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

        toolbar_start = template.index('<div class="table-card-header">')
        table_start = template.index(
            '<div class="active-table-wrap"',
            toolbar_start,
        )
        toolbar = template[toolbar_start:table_start]

        for marker in (
            'class="os-toolbar-heading"',
            'class="os-toolbar-title-row"',
            'id="os-items-title"',
            'id="os-count"',
            'class="os-personalize-slot" data-table-action-slot',
            'class="os-type-filter"',
            'class="os-insights"',
            'id="os-high-flag"',
            'class="os-search-tools"',
            'id="os-search"',
            'id="os-clear-filters"',
            'id="os-bulk-delete"',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, toolbar)

        self.assertIn(
            '<div class="os-toolbar-heading">\n'
            '            <div class="os-toolbar-title-row">\n'
            '                <h2 id="os-items-title">Ordering</h2>\n'
            '                <span class="badge-count" id="os-count">{{ entries|length }}</span>\n'
            '            </div>\n'
            '        </div>',
            toolbar,
        )

        insights_start = toolbar.index('<div class="os-insights"')
        insights_end = toolbar.index('</div>', insights_start)
        self.assertNotIn('id="os-clear-filters"', toolbar[insights_start:insights_end])
        search_tools_start = toolbar.index('<div class="os-search-tools">')
        search_tools_end = toolbar.index('</div>', search_tools_start)
        search_tools = toolbar[search_tools_start:search_tools_end]
        self.assertIn('class="os-personalize-slot" data-table-action-slot', search_tools)
        self.assertEqual(toolbar.count('data-table-action-slot'), 1)
        self.assertLess(search_tools.index('id="os-search"'), search_tools.index('class="os-personalize-slot"'))
        self.assertLess(search_tools.index('id="os-search"'), search_tools.index('id="os-clear-filters"'))
        self.assertRegex(
            search_tools,
            r'id="os-search"[^>]*>\s*<button[^>]*id="os-clear-filters"',
        )
        self.assertLess(toolbar.index('id="os-clear-filters"'), toolbar.index('id="os-bulk-delete"'))

        clear_css_start = template.index('\n    .os-clear-filters {')
        clear_css_end = template.index('\n    }', clear_css_start)
        clear_css = template[clear_css_start:clear_css_end]
        self.assertIn('flex: 0 0 auto;', clear_css)
        self.assertIn('margin-left: 0;', clear_css)
        self.assertIn('white-space: nowrap;', clear_css)
        self.assertNotIn('margin-left: auto;', clear_css)

        search_tools_css_start = template.index('.os-search-tools {')
        search_tools_css_end = template.index('\n    }', search_tools_css_start)
        search_tools_css = template[search_tools_css_start:search_tools_css_end]
        self.assertIn('flex: 0 1 auto;', search_tools_css)
        self.assertIn('display: inline-grid;', search_tools_css)
        self.assertIn('grid-template-columns: minmax(0, 1fr) auto;', search_tools_css)
        self.assertIn('max-width: 100%;', search_tools_css)
        self.assertIn('.os-search-tools .os-clear-filters { grid-column: 2; grid-row: 1; }', template)
        self.assertIn('.os-search-tools .os-personalize-slot { grid-column: 1; grid-row: 2; justify-self: start; }', template)

        search_input_css_start = template.index('.os-search-tools .table-search {')
        search_input_css_end = template.index('\n    }', search_input_css_start)
        search_input_css = template[search_input_css_start:search_input_css_end]
        self.assertIn('min-width: 0;', search_input_css)
        self.assertIn('margin-left: 0;', search_input_css)

        self.assertNotIn(
            '</div>\n    <div class="os-insights"',
            template,
        )
        shared_header_selector = (
            ':is(body.app-shell, body.embed-shell)[data-page="ordering_sheet"] '
            '.table-card-header {'
        )
        self.assertIn(shared_header_selector, template)
        header_start = template.index(
            shared_header_selector
        )
        header_end = template.index('\n        }', header_start)
        header_css = template[header_start:header_end]
        self.assertIn('flex-wrap: wrap;', header_css)
        self.assertIn('column-gap: 0.5rem;', header_css)
        self.assertIn('overflow: visible;', header_css)
        self.assertNotIn('overflow-x: auto;', header_css)

        insights_css_start = template.index(
            ':is(body.app-shell, body.embed-shell)[data-page="ordering_sheet"] '
            '.table-card-header .os-insights {'
        )
        insights_css_end = template.index('\n        }', insights_css_start)
        insights_css = template[insights_css_start:insights_css_end]
        self.assertIn('flex: 1 1 620px;', insights_css)
        self.assertIn('flex-wrap: wrap;', insights_css)

        desktop_grid_start = template.index('@media screen and (min-width: 1024px)')
        desktop_grid_end = template.index('\n    }\n\n    .delete-btn', desktop_grid_start)
        desktop_grid_css = template[desktop_grid_start:desktop_grid_end]
        self.assertIn(shared_header_selector, desktop_grid_css)
        self.assertIn('display: grid;', desktop_grid_css)
        self.assertIn(
            'grid-template-columns: max-content max-content minmax(150px, 1fr) '
            'minmax(210px, max-content);',
            desktop_grid_css,
        )
        self.assertIn('align-items: center;', desktop_grid_css)
        self.assertIn('overflow: visible;', desktop_grid_css)
        self.assertNotIn('overflow-x: auto;', desktop_grid_css)
        for expected in (
            '.table-card-header .os-toolbar-heading {\n            grid-column: 1;',
            '.table-card-header .os-type-filter {\n            grid-column: 2;\n            margin-left: 0;',
            '.table-card-header .os-insights {\n            grid-column: 3;\n            min-width: 0;\n            flex: initial;\n            flex-wrap: wrap;',
            '.table-card-header .os-search-tools {\n            grid-column: 4;\n            width: 100%;\n            max-width: none;\n            margin-left: 0;',
            '.table-card-header .os-bulk-delete.visible {\n            grid-column: 4;\n            grid-row: 2;\n            justify-self: end;',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, desktop_grid_css)

        for selector in ('.os-stat {', '.os-urg-flag {'):
            with self.subTest(selector=selector):
                rule_start = template.index(selector)
                rule_end = template.index('\n    }', rule_start)
                rule_css = template[rule_start:rule_end]
                self.assertIn('flex: 0 0 auto;', rule_css)
                self.assertIn('white-space: nowrap;', rule_css)

    def test_full_page_uses_viewport_height_with_table_owned_scrolling(self):
        template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

        desktop_start = template.index('@media screen and (min-width: 769px)')
        desktop_end = template.index('</style>', desktop_start)
        desktop_css = template[desktop_start:desktop_end]

        self.assertIn('body.app-shell[data-page="ordering_sheet"] {', desktop_css)
        self.assertIn('height: 100dvh;', desktop_css)
        self.assertIn(
            'height: calc(100dvh - var(--ui-development-banner-height, 0px));',
            desktop_css,
        )
        self.assertIn('> .container > .table-card {', desktop_css)
        self.assertIn('flex: 1 1 0;', desktop_css)
        self.assertIn('.active-table-wrap {', desktop_css)
        self.assertIn('max-height: none;', desktop_css)
        self.assertIn('overflow: auto;', desktop_css)
        self.assertIn(
            '<h2 id="os-items-title">Ordering</h2>',
            template,
        )
        self.assertIn(
            'class="active-table-wrap" role="region" '
            'aria-labelledby="os-items-title" tabindex="0"',
            template,
        )
        self.assertIn(
            'data-personalize-table data-table-key="main" data-table-label="Ordering"',
            template,
        )

    def test_filters_and_initial_date_sort_survive_targeted_row_actions(self):
        template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

        self.assertIn("'orderingSheetTableState:v3:' + tableView", template)
        self.assertIn('sessionStorage.getItem(TABLE_STATE_KEY)', template)
        self.assertIn('sessionStorage.setItem(TABLE_STATE_KEY', template)
        for field in (
            'search: search ? search.value',
            'type: activeType',
            'status: activeStatus',
            'urgencyOnly: urgencyOnly',
        ):
            self.assertIn(field, template)

        self.assertIn(
            'data-column-key="initial-date" data-column-label="Initial - Date">',
            template,
        )
        self.assertNotIn('>Initial<span class="sort-ind"></span></th>', template)
        self.assertNotIn('>Date<span class="sort-ind"></span></th>', template)
        self.assertIn("data-sort=\"{{ entry.created_at|date:'Ymd' }}\"", template)
        self.assertNotIn('data-entry-date=', template)
        self.assertNotIn('os-entry-date-filter', template)
        self.assertNotIn('entryDateFilter', template)
        self.assertNotIn('savedTableState.entryDate', template)
        self.assertIn('aria-label="Sort by entry date"', template)
        self.assertIn('class="os-time">{{ entry.created_at|date:"d/m/Y" }}', template)
        self.assertNotIn('entry.created_at|date:"d M Y, H:i"', template)
        seamless_start = template.index(
            "document.addEventListener('ui:seamless-updated'"
        )
        seamless_end = template.index('// ── Click-to-sort column headers', seamless_start)
        seamless_handler = template[seamless_start:seamless_end]
        self.assertIn("selectors.indexOf('#os-tbody')", seamless_handler)
        self.assertLess(
            seamless_handler.index('applyCurrentSort();'),
            seamless_handler.index('applyFilters();'),
        )
        self.assertIn('data-seamless-refresh="#os-tbody"', template)
        self.assertIn('activeSortIndex = idx;', template)
        self.assertNotIn('savedTableState.sortIndex', template)
        self.assertNotIn('savedTableState.sortDirection', template)
        self.assertIn("activeHeader.setAttribute(\n            'aria-sort'", template)

    def test_removed_order_details_have_no_client_side_handlers(self):
        template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

        self.assertNotIn('os-progress-details', template)
        self.assertNotIn('os-progress-grid', template)
        self.assertNotIn('os-quantity-received-field', template)
        self.assertNotIn('syncReceivedQuantityField', template)
        self.assertNotIn('syncAllReceivedQuantityFields', template)

    def test_status_selection_autosaves_status_only_without_detail_validation(self):
        template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

        helper_start = template.index('function saveSelectedStatus(statusSelect) {')
        helper_end = template.index('\n\n    if (osTbody)', helper_start)
        helper = template[helper_start:helper_end]

        self.assertIn(
            'if (statusSelect.value === statusSelect.dataset.current) return;',
            helper,
        )
        self.assertIn("form.querySelector('input[name=\"status_only\"]')", helper)
        self.assertIn("statusOnly.value = '1';", helper)
        self.assertIn('form.requestSubmit();', helper)
        self.assertNotIn('form.checkValidity()', helper)
        self.assertNotIn('form.reportValidity()', helper)
        self.assertNotIn('detailControls', helper)

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
        self.assertIn('statusSelect.value = statusSelect.dataset.current;', template)

    def test_custom_status_reveals_a_labelled_editor_before_submitting(self):
        template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

        self.assertIn("statusSelect.value === 'custom'", template)
        self.assertIn('syncCustomStatusEditor(statusSelect, true, false);', template)
        self.assertIn('customForm.hidden = !isEditing;', template)
        self.assertIn('input.disabled = !isEditing;', template)
        self.assertIn('input.required = isEditing;', template)
        self.assertIn("statusSelect.setAttribute('aria-expanded'", template)
        self.assertIn('aria-controls="os-custom-status-{{ entry.pk }}"', template)
        self.assertIn(
            'for="os-custom-status-input-{{ entry.pk }}">Custom status for {{ entry.name }}',
            template,
        )
        self.assertIn('name="custom_status_text"', template)
        self.assertIn('data-server-value="{{ entry.custom_status_text }}"', template)
        self.assertIn('placeholder="Type custom status" maxlength="80"', template)
        self.assertIn('input.value = input.value.trim();', template)
        self.assertIn("input.setCustomValidity(input.value ? '' : 'Enter custom status text.');", template)
        self.assertIn("e.key !== 'Escape'", template)

        helper_start = template.index('function saveSelectedStatus(statusSelect) {')
        helper_end = template.index('\n\n    if (osTbody)', helper_start)
        helper = template[helper_start:helper_end]
        custom_branch = helper.index("statusSelect.value === 'custom'")
        custom_return = helper.index('return;', custom_branch)
        request_submit = helper.index('form.requestSubmit();')
        self.assertLess(custom_branch, custom_return)
        self.assertLess(custom_return, request_submit)
        self.assertIn("form.classList.contains('os-custom-status-form')", template)
        self.assertIn('initializeCustomStatusEditors();', template)


class OrderingRowPresentationContractTests(SimpleTestCase):
    def setUp(self):
        self.template = (
            Path(settings.BASE_DIR)
            / 'app'
            / 'templates'
            / 'partials'
            / '_ordering_sheet.html'
        ).read_text(encoding='utf-8')

    def test_reason_colors_only_fill_reasoning_cell_except_when_backordered(self):
        row = '.active-table tbody tr:not([data-status="backordered"])'
        stock = f'{row}[data-reasoning="stock"] > td.os-reason-col'
        basket = f'{row}[data-reasoning="basket"] > td.os-reason-col'
        high = f'{row}[data-urgency="high"] > td.os-reason-col'
        for selector, background, hover in (
            (stock, '#ffff00', '#ffeb00'),
            (basket, '#ff9900', '#f59e0b'),
            (high, '#93c5fd', '#7cb7f6'),
        ):
            with self.subTest(selector=selector):
                start = self.template.index(selector)
                rule = self.template[start:self.template.index('}', start)]
                self.assertIn(f'--os-row-bg: {background};', rule)
                self.assertIn(f'--os-row-hover-bg: {hover};', rule)
        self.assertGreater(self.template.index(high), self.template.index(stock))
        self.assertGreater(self.template.index(high), self.template.index(basket))
        for attribute in ('data-reasoning="stock"', 'data-reasoning="basket"', 'data-urgency="high"'):
            self.assertNotIn(f'.active-table tbody tr[{attribute}] {{', self.template)
        self.assertIn('data-reasoning="{{ entry.reasoning }}" data-urgency="{{ entry.urgency }}"', self.template)
        self.assertIn('--os-row-muted: #334155;', self.template)

    def test_pending_and_ordered_keep_status_colors_and_backordered_is_red(self):
        for status, background, hover in (
            ('pending', '#fff1f2', '#ffe4e6'),
            ('ordered', '#ecfdf5', '#d1fae5'),
            ('backordered', '#fca5a5', '#f87171'),
        ):
            with self.subTest(status=status):
                start = self.template.index(f'.active-table tbody tr[data-status="{status}"]')
                rule = self.template[start:self.template.index('}', start)]
                self.assertIn(f'--os-row-bg: {background};', rule)
                self.assertIn(f'--os-row-hover-bg: {hover};', rule)

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
        self.assertEqual(
            self.template.count('background: var(--os-row-bg, #fff);'),
            1,
        )
        self.assertNotIn('tr.row-high   td', self.template)
        self.assertNotIn('tr.row-medium td', self.template)
        self.assertNotIn('tr.row-low    td', self.template)

    def test_status_and_actions_share_one_pinned_column_without_sorting(self):
        self.assertIn(
            '<th class="os-actions-col" data-column-key="status-actions" '
            'data-column-label="Status and actions">Status / Actions</th>',
            self.template,
        )
        self.assertNotIn('>Status<span class="sort-ind"></span></th>', self.template)
        self.assertIn('<td class="td-actions2 os-actions-cell"', self.template)
        self.assertIn('class="os-actions-primary"', self.template)
        self.assertIn('class="os-actions-status"', self.template)
        self.assertIn('class="os-action-buttons" role="group"', self.template)
        self.assertIn('width: 320px; min-width: 320px; max-width: 320px;', self.template)
        self.assertIn(
            'display: grid; grid-template-columns: minmax(156px, 1fr) auto;',
            self.template,
        )
        self.assertNotIn('nth-last-child(2)', self.template)
        self.assertNotIn('nth-last-child(-n+2)', self.template)
        self.assertIn(
            '.active-table tbody tr:hover td.os-actions-cell { '
            'background: var(--os-row-hover-bg); }',
            self.template,
        )

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

    def test_compact_supporting_columns_leave_more_room_for_drug_name(self):
        self.assertIn(
            '.active-table .os-patient-col { width: 100px; min-width: 90px; '
            'max-width: 100px; padding-inline: 0.375rem; }',
            self.template,
        )
        self.assertIn(
            '.active-table .os-name-col { width: 44%; min-width: 500px; '
            'max-width: 840px; padding-left: 0.375rem; }',
            self.template,
        )
        self.assertIn(
            '.active-table .os-reason-col { width: 128px; min-width: 128px; max-width: 128px; }',
            self.template,
        )
        self.assertIn(
            '.active-table .os-qty-needed-col { width: 1%; min-width: 76px; '
            'padding-right: 0.375rem; white-space: nowrap; }',
            self.template,
        )
        self.assertIn(
            '.active-table .os-qty-remaining-col { width: 1%; min-width: 100px; '
            'padding-left: 0.375rem; white-space: nowrap; }',
            self.template,
        )
        self.assertIn(
            '<th class="os-qty-needed-col" data-column-key="qty-needed" '
            'data-column-label="Needed">Needed</th>',
            self.template,
        )
        self.assertIn(
            '<th class="os-qty-remaining-col" data-column-key="qty-remaining" '
            'data-column-label="Remaining">Remaining</th>',
            self.template,
        )
        table_head = self.template[
            self.template.index('<thead>'):self.template.index('</thead>')
        ]
        self.assertNotIn('Qty Needed', table_head)
        self.assertNotIn('Qty Remaining', table_head)
        self.assertIn('<td class="os-qty-needed-col">', self.template)
        self.assertIn('<td class="os-qty-remaining-col">', self.template)
        self.assertNotIn('os-urgency-col', self.template)
        self.assertNotIn('>Urgency<span class="sort-ind"></span></th>', self.template)
        drug_name_start = self.template.index('.os-drug-name {')
        drug_name_end = self.template.index('\n    }', drug_name_start)
        drug_name_css = self.template[drug_name_start:drug_name_end]
        self.assertIn('white-space: normal;', drug_name_css)
        self.assertIn('overflow-wrap: anywhere;', drug_name_css)
        self.assertIn('font-size: calc(22px * var(--ui-type-scale, 1));', drug_name_css)
        self.assertNotIn('text-overflow:', drug_name_css)
        self.assertNotIn('overflow: hidden;', drug_name_css)
        self.assertIn(
            '.os-name-line { display: flex; align-items: flex-start; gap: 0.4rem; '
            'min-width: 0; max-width: 840px; }',
            self.template,
        )
        self.assertIn('<th class="os-patient-col"', self.template)
        self.assertIn('<td class="os-patient-col">', self.template)
        self.assertIn('os-sortable os-name-col', self.template)
        self.assertIn('os-sortable os-reason-col', self.template)
        self.assertIn(
            'class="os-urgency-meter os-urgency-meter--{{ entry.urgency }}" '
            'role="img" aria-label="Urgency: {{ entry.get_urgency_display }}"',
            self.template,
        )
        self.assertIn('.os-urgency-meter--high > span,', self.template)
        self.assertIn('.os-urgency-meter--medium > span:nth-child(-n+2),', self.template)
        self.assertIn('.os-urgency-meter--low > span:first-child {', self.template)
        self.assertIn('.os-urgency-meter--na::after {', self.template)
