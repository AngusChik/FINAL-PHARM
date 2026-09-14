import time

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from app.mixins import PASSKEY_SESSION_KEY
from app.models import OrderingSheetEntry, OrderingSheetStatusEvent, UserAction


class OrderingSheetPUCommentTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.owner = get_user_model().objects.create_user(
            username='ordering-comment-owner', is_staff=True,
        )
        cls.pu = get_user_model().objects.create_user(username='PU', is_staff=False)

    def setUp(self):
        self.client.force_login(self.pu)
        self.url = reverse('ordering_sheet')

    def create_entry(self, entry_type=OrderingSheetEntry.ENTRY_DRUG):
        return OrderingSheetEntry.objects.create(
            name=f'Shared {entry_type} request', entry_type=entry_type,
            created_by=self.owner, initials='AD',
            status=OrderingSheetEntry.STATUS_ORDERED,
            quantity_ordered=5, quantity_received=1,
        )

    def test_pu_can_add_and_edit_shared_comments_in_full_and_embedded_views(self):
        for entry_type in (OrderingSheetEntry.ENTRY_DRUG, OrderingSheetEntry.ENTRY_OTC):
            entry = self.create_entry(entry_type)
            for suffix in ('', '?embed=1'):
                url = f'{self.url}{suffix}'
                with self.subTest(entry_type=entry_type, suffix=suffix):
                    response = self.client.get(url)
                    self.assertEqual(response.status_code, 200)
                    self.assertFalse(response.context['can_administer'])
                    self.assertFalse(response.context['can_delete_entries'])
                    html = response.content.decode()
                    start = html.index(f'<tr data-entry-id="{entry.pk}"')
                    row = html[start:html.index('</tr>', start)]
                    self.assertIn(f'aria-label="Add or edit comment for {entry.name}"', row)
                    self.assertIn('class="os-note-form"', row)
                    self.assertNotIn('<select name="status"', row)
                    self.assertIn('class="edit-btn os-edit-btn"', row)
                    self.assertNotIn('name="action" value="delete"', row)
                    self.assertNotIn('class="os-check os-row-check"', row)
                    self.assertNotIn('id="os-select-all"', html)
                    self.assertNotIn('id="os-bulk-delete"', html)

                    for comment in ('Please call before pickup.', 'Please call & confirm the quantity.'):
                        response = self.client.post(url, {
                            'action': 'update_note', 'entry_id': entry.pk,
                            'order_note': f'  {comment}  ',
                        })
                        self.assertRedirects(response, url, fetch_redirect_response=False)
                        entry.refresh_from_db()
                        self.assertEqual(entry.order_note, comment)

                    html = self.client.get(url).content.decode()
                    self.assertIn('os-note-toggle has-note', html)
                    self.assertIn('Please call &amp; confirm the quantity.', html)

    def test_comment_save_updates_only_the_comment(self):
        entry = self.create_entry()
        before = OrderingSheetEntry.objects.values().get(pk=entry.pk)

        self.client.post(self.url, {
            'action': 'update_note', 'entry_id': entry.pk, 'order_note': 'PU comment',
            'status': 'received', 'status_only': '1', 'name': 'Changed name',
            'quantity_ordered': '100', 'quantity_received': '100',
            'is_deleted': '1', 'created_by': self.pu.pk,
        })

        self.assertEqual(
            OrderingSheetEntry.objects.values().get(pk=entry.pk),
            {**before, 'order_note': 'PU comment'},
        )
        self.assertFalse(OrderingSheetStatusEvent.objects.filter(entry=entry).exists())

    def test_pu_can_edit_shared_drug_and_otc_entries_without_changing_progress(self):
        for entry_type in (OrderingSheetEntry.ENTRY_DRUG, OrderingSheetEntry.ENTRY_OTC):
            for suffix in ('', '?embed=1'):
                with self.subTest(entry_type=entry_type, suffix=suffix):
                    entry = self.create_entry(entry_type)
                    before = OrderingSheetEntry.objects.values().get(pk=entry.pk)
                    fields = {
                        'name': 'Edited shared request', 'initials': 'PU',
                        'patient_name': 'Updated patient', 'quantity_needed': '3',
                        'quantity_remaining': '1',
                    }
                    if entry_type == OrderingSheetEntry.ENTRY_OTC:
                        fields.update(side='right', phone_number='123-456-7890')
                    else:
                        fields.update(reasoning='basket', urgency='high')

                    response = self.client.post(f'{self.url}{suffix}', {
                        'action': 'edit', 'entry_id': entry.pk, **fields,
                        'status': 'received', 'order_note': 'Do not overwrite',
                        'quantity_ordered': '100', 'quantity_received': '100',
                        'is_deleted': '1', 'created_by': self.pu.pk,
                    })

                    self.assertRedirects(response, f'{self.url}{suffix}', fetch_redirect_response=False)
                    self.assertEqual(
                        OrderingSheetEntry.objects.values().get(pk=entry.pk),
                        {**before, **fields},
                    )
                    self.assertTrue(UserAction.objects.filter(
                        user=self.pu, action='ordering_edit', target=fields['name'],
                    ).exists())
                    self.assertFalse(OrderingSheetStatusEvent.objects.filter(entry=entry).exists())

    def test_pu_cannot_change_order_status(self):
        entry = self.create_entry()
        before = OrderingSheetEntry.objects.values().get(pk=entry.pk)
        self.client.post(self.url, {
            'action': 'update_status', 'entry_id': entry.pk,
            'status': 'received', 'status_only': '1',
        })
        self.assertEqual(OrderingSheetEntry.objects.values().get(pk=entry.pk), before)

    def test_pu_cannot_delete_own_or_shared_rows_even_with_an_admin_passkey(self):
        own = self.create_entry()
        own.created_by = self.pu
        own.status = OrderingSheetEntry.STATUS_PENDING
        own.save(update_fields=['created_by', 'status'])
        shared = self.create_entry()
        before = list(OrderingSheetEntry.objects.filter(pk__in=[own.pk, shared.pk]).order_by('pk').values())

        for unlocked in (False, True):
            session = self.client.session
            if unlocked:
                session[PASSKEY_SESSION_KEY] = time.time()
            else:
                session.pop(PASSKEY_SESSION_KEY, None)
            session.save()
            for suffix in ('', '?embed=1'):
                with self.subTest(unlocked=unlocked, suffix=suffix):
                    response = self.client.get(f'{self.url}{suffix}')
                    self.assertFalse(response.context['can_delete_entries'])
                    html = response.content.decode()
                    self.assertNotIn('name="action" value="delete"', html)
                    self.assertNotIn('id="os-bulk-delete"', html)
                    self.assertNotIn('id="os-select-all"', html)
                    self.assertNotIn('class="os-check os-row-check"', html)
                    for payload in (
                        {'action': 'delete', 'entry_id': own.pk},
                        {'action': 'delete', 'entry_id': shared.pk},
                        {'action': 'delete_selected', 'entry_ids': f'{own.pk},{shared.pk}'},
                    ):
                        response = self.client.post(f'{self.url}{suffix}', payload, follow=True)
                        self.assertContains(response, 'Only staff accounts can delete ordering-sheet entries.')
                        self.assertEqual(
                            list(OrderingSheetEntry.objects.filter(pk__in=[own.pk, shared.pk]).order_by('pk').values()),
                            before,
                        )
        self.assertFalse(UserAction.objects.filter(action='ordering_delete').exists())

    def test_staff_can_still_remove_single_and_selected_entries_to_recovery(self):
        self.client.force_login(self.owner)
        single = self.create_entry()
        selected = self.create_entry(OrderingSheetEntry.ENTRY_OTC)
        response = self.client.get(self.url)
        self.assertTrue(response.context['can_delete_entries'])
        self.assertContains(response, 'id="os-bulk-delete"')
        self.assertContains(response, 'id="os-select-all"')

        for entry, payload in (
            (single, {'action': 'delete', 'entry_id': single.pk}),
            (selected, {'action': 'delete_selected', 'entry_ids': str(selected.pk)}),
        ):
            self.client.post(self.url, payload)
            entry.refresh_from_db()
            self.assertTrue(entry.is_deleted)
            self.assertIsNotNone(entry.deleted_at)
            self.assertEqual(entry.deleted_by_id, self.owner.pk)

    def test_pu_cannot_edit_deleted_entries(self):
        entry = self.create_entry()
        entry.is_deleted = True
        entry.save(update_fields=['is_deleted'])
        before = OrderingSheetEntry.objects.values().get(pk=entry.pk)
        self.client.post(self.url, {
            'action': 'edit', 'entry_id': entry.pk, 'name': 'Changed name', 'initials': 'PU',
        })
        self.assertEqual(OrderingSheetEntry.objects.values().get(pk=entry.pk), before)

    def test_pu_cannot_comment_on_deleted_rows(self):
        entry = self.create_entry()
        entry.is_deleted = True
        entry.order_note = 'Retained comment'
        entry.save(update_fields=['is_deleted', 'order_note'])

        response = self.client.post(self.url, {
            'action': 'update_note', 'entry_id': entry.pk, 'order_note': 'New comment',
        }, follow=True)

        self.assertContains(response, 'Ordering-sheet entry not found.')
        entry.refresh_from_db()
        self.assertEqual(entry.order_note, 'Retained comment')
        self.assertTrue(entry.is_deleted)

    def test_edit_and_comment_require_sign_in(self):
        entry = self.create_entry()
        self.client.logout()
        before = OrderingSheetEntry.objects.values().get(pk=entry.pk)
        for action in ('edit', 'update_note'):
            response = self.client.post(self.url, {
                'action': action, 'entry_id': entry.pk, 'order_note': 'Anonymous comment',
                'name': 'Anonymous edit', 'initials': 'AA',
            })
            self.assertEqual(response.status_code, 302)
            self.assertIn(reverse('login'), response.url)
            self.assertEqual(OrderingSheetEntry.objects.values().get(pk=entry.pk), before)
