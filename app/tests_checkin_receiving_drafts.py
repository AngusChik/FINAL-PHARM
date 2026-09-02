from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import (
    Category,
    CheckinReceivingDraft,
    CheckinSession,
    Product,
    ProductLot,
)


@override_settings(AXES_ENABLED=False)
class CheckinReceivingDraftTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='receiving-draft-user', password='test-password', is_staff=True,
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='Receiving drafts')
        self.product = Product.objects.create(
            name='Draft Product', barcode='DRAFT-100', price=Decimal('8.50'),
            quantity_in_stock=5, category=self.category,
        )
        self.session = CheckinSession.objects.create(
            user=self.user, scanned_by='Draft tester',
        )

    def draft_url(self, product=None, session=None):
        return reverse(
            'checkin_receiving_draft',
            args=[(session or self.session).pk, (product or self.product).pk],
        )

    def test_typed_pair_autosaves_without_creating_or_changing_inventory(self):
        response = self.client.post(self.draft_url(), {
            'lot_number': ' new-77 ',
            'lot_expiry': '31-12-2030',
            'revision': '0',
        })

        self.assertEqual(response.status_code, 200)
        draft = CheckinReceivingDraft.objects.get(
            session=self.session, product=self.product,
        )
        self.assertEqual(draft.lot_number, 'NEW-77')
        self.assertEqual(draft.lot_expiry, date(2030, 12, 31))
        self.assertIsNone(draft.existing_lot_id)
        self.assertEqual(draft.revision, 1)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertFalse(ProductLot.objects.filter(product=self.product).exists())

    def test_typed_draft_restores_after_page_navigation(self):
        self.client.post(self.draft_url(), {
            'lot_number': 'restore-8',
            'lot_expiry': '30-11-2031',
            'revision': '0',
        })

        response = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': self.product.pk},
        )

        self.assertEqual(response.context['receiving_draft_lot_number'], 'RESTORE-8')
        self.assertEqual(response.context['receiving_draft_lot_expiry'], '30-11-2031')
        self.assertEqual(response.context['receiving_draft_revision'], 1)
        self.assertContains(response, 'value="RESTORE-8"')
        self.assertContains(response, 'value="30-11-2031"')

    def test_saved_lot_selection_autosaves_and_restores_authoritatively(self):
        saved_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='KNOWN-LOT',
            expiry_date=date.today() + timedelta(days=90),
            quantity_on_hand=5,
        )

        response = self.client.post(self.draft_url(), {
            'existing_lot_id': str(saved_lot.pk),
            'lot_number': 'ignored',
            'lot_expiry': '01-01-2099',
            'revision': '0',
        })

        self.assertEqual(response.status_code, 200)
        draft = CheckinReceivingDraft.objects.get(session=self.session, product=self.product)
        self.assertEqual(draft.existing_lot, saved_lot)
        self.assertEqual(draft.lot_number, saved_lot.lot_number)
        self.assertEqual(draft.lot_expiry, saved_lot.expiry_date)
        saved_lot.refresh_from_db()
        self.assertEqual(saved_lot.quantity_on_hand, 5)

        page = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': self.product.pk},
        )
        self.assertEqual(page.context['selected_receiving_lot_id'], saved_lot.pk)
        # The saved receiving selector and the separate MAIN-assignment
        # destination selector both offer this recorded lot.
        self.assertContains(page, f'value="{saved_lot.pk}"', count=2)

    def test_invalid_expiry_preserves_last_valid_draft_and_inventory(self):
        first = self.client.post(self.draft_url(), {
            'lot_number': 'VALID-1', 'lot_expiry': '31-12-2030', 'revision': '0',
        })
        self.assertEqual(first.status_code, 200)

        response = self.client.post(self.draft_url(), {
            'lot_number': 'INVALID-2', 'lot_expiry': '31-99-2030', 'revision': '1',
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['field'], 'lot_expiry')
        draft = CheckinReceivingDraft.objects.get(session=self.session, product=self.product)
        self.assertEqual((draft.lot_number, draft.lot_expiry, draft.revision),
                         ('VALID-1', date(2030, 12, 31), 1))
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertFalse(ProductLot.objects.filter(product=self.product).exists())

    def test_past_expiry_is_rejected_without_saving_or_receiving_stock(self):
        yesterday = date.today() - timedelta(days=1)

        autosave = self.client.post(self.draft_url(), {
            'lot_number': 'EXPIRED-LOT',
            'lot_expiry': yesterday.strftime('%d-%m-%Y'),
            'revision': '0',
        })
        receive = self.client.post(
            reverse('add_quantity', args=[self.session.pk, self.product.pk]),
            {
                'amount': '1',
                'lot_number': 'EXPIRED-LOT',
                'lot_expiry': yesterday.strftime('%d-%m-%Y'),
            },
        )

        self.assertEqual(autosave.status_code, 400)
        self.assertEqual(autosave.json()['field'], 'lot_expiry')
        self.assertEqual(receive.status_code, 302)
        self.assertFalse(CheckinReceivingDraft.objects.exists())
        self.assertFalse(ProductLot.objects.filter(product=self.product).exists())
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)

    def test_blank_fields_clear_existing_draft(self):
        created = self.client.post(self.draft_url(), {
            'lot_number': 'CLEAR-ME', 'lot_expiry': '', 'revision': '0',
        })
        self.assertEqual(created.status_code, 200)

        response = self.client.post(self.draft_url(), {
            'lot_number': '', 'lot_expiry': '', 'revision': '1',
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['cleared'])
        draft = CheckinReceivingDraft.objects.get(
            session=self.session, product=self.product,
        )
        self.assertEqual(
            (draft.existing_lot_id, draft.lot_number, draft.lot_expiry, draft.revision),
            (None, '', None, 2),
        )
        self.assertEqual(response.json()['draft']['revision'], 2)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)

    def test_clear_keeps_revision_tombstone_against_stale_recreate(self):
        created = self.client.post(self.draft_url(), {
            'lot_number': 'FIRST', 'lot_expiry': '', 'revision': '0',
        })
        self.assertEqual(created.status_code, 200)
        cleared = self.client.post(self.draft_url(), {
            'lot_number': '', 'lot_expiry': '', 'revision': '1',
        })
        self.assertEqual(cleared.status_code, 200)
        recreated = self.client.post(self.draft_url(), {
            'lot_number': 'LATEST', 'lot_expiry': '', 'revision': '2',
        })
        self.assertEqual(recreated.status_code, 200)

        stale = self.client.post(self.draft_url(), {
            'lot_number': 'STALE', 'lot_expiry': '', 'revision': '1',
        })

        self.assertEqual(stale.status_code, 409)
        draft = CheckinReceivingDraft.objects.get(
            session=self.session, product=self.product,
        )
        self.assertEqual((draft.lot_number, draft.revision), ('LATEST', 3))

    def test_ended_session_rejects_autosave(self):
        self.session.ended_at = timezone.now()
        self.session.save(update_fields=['ended_at'])

        response = self.client.post(self.draft_url(), {
            'lot_number': 'TOO-LATE', 'lot_expiry': '', 'revision': '0',
        })

        self.assertEqual(response.status_code, 409)
        self.assertFalse(CheckinReceivingDraft.objects.exists())

    def test_oversized_lot_number_is_rejected_without_partial_save(self):
        response = self.client.post(self.draft_url(), {
            'lot_number': 'X' * 65, 'lot_expiry': '', 'revision': '0',
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['field'], 'lot_number')
        self.assertFalse(CheckinReceivingDraft.objects.exists())
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)

    def test_saved_lot_must_belong_to_product(self):
        other = Product.objects.create(
            name='Other Draft Product', barcode='DRAFT-200', price=Decimal('4.00'),
            quantity_in_stock=1, category=self.category,
        )
        other_lot = ProductLot.objects.create(
            product=other, lot_number='OTHER-LOT', quantity_on_hand=1,
        )

        response = self.client.post(self.draft_url(), {
            'existing_lot_id': str(other_lot.pk),
            'confirm_expired_lot': 'yes',
            'revision': '0',
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['field'], 'existing_lot_id')
        self.assertFalse(CheckinReceivingDraft.objects.exists())
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)

    def test_expired_confirmation_cannot_restore_an_archived_lot(self):
        archived_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='ARCHIVED-LOT',
            expiry_date=date.today() - timedelta(days=1),
            quantity_on_hand=5,
            archived_at=timezone.now(),
            archived_by=self.user,
        )

        response = self.client.post(self.draft_url(), {
            'existing_lot_id': str(archived_lot.pk),
            'confirm_expired_lot': 'yes',
            'revision': '0',
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['field'], 'existing_lot_id')
        self.assertFalse(CheckinReceivingDraft.objects.exists())
        self.product.refresh_from_db()
        archived_lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertEqual(archived_lot.quantity_on_hand, 5)

    def test_expired_named_saved_lot_needs_confirmation_and_restores(self):
        expired_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='EXPIRED-SAVED',
            expiry_date=date.today() - timedelta(days=1),
            quantity_on_hand=5,
        )

        rejected = self.client.post(self.draft_url(), {
            'existing_lot_id': str(expired_lot.pk),
            'revision': '0',
        })

        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(rejected.json()['field'], 'existing_lot_id')
        self.assertIn('Confirm that you want to receive stock', rejected.json()['error'])
        self.assertFalse(CheckinReceivingDraft.objects.exists())

        accepted = self.client.post(self.draft_url(), {
            'existing_lot_id': str(expired_lot.pk),
            'confirm_expired_lot': 'yes',
            'revision': '0',
        })

        self.assertEqual(accepted.status_code, 200)
        draft = CheckinReceivingDraft.objects.get(
            session=self.session, product=self.product,
        )
        self.assertEqual(draft.existing_lot, expired_lot)
        page = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': self.product.pk},
        )
        self.assertEqual(page.context['selected_receiving_lot_id'], expired_lot.pk)
        self.assertIn(
            expired_lot.pk,
            page.context['expired_receiving_lot_ids'],
        )
        self.assertTrue(
            page.context['selected_receiving_lot_requires_confirmation'],
        )
        fragment = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {
                'product_id': self.product.pk,
                'format': 'checkin_fragments',
            },
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )
        self.assertEqual(fragment.status_code, 200)
        payload = fragment.json()
        self.assertTrue(
            payload['selected_receiving_lot_requires_confirmation'],
        )
        expired_payload = next(
            lot for lot in payload['receiving_lots']
            if lot['id'] == expired_lot.pk
        )
        self.assertTrue(expired_payload['requires_expired_confirmation'])

    def test_expired_lot_query_parameter_cannot_create_a_trusted_selection(self):
        expired_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='EXPIRED-QUERY',
            expiry_date=date.today() - timedelta(days=1),
            quantity_on_hand=5,
        )

        page = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {
                'product_id': self.product.pk,
                'receiving_lot_id': expired_lot.pk,
            },
        )

        self.assertIsNone(page.context['selected_receiving_lot_id'])
        self.assertFalse(
            page.context['selected_receiving_lot_requires_confirmation'],
        )
        self.assertFalse(CheckinReceivingDraft.objects.exists())

    def test_expired_saved_lot_confirmation_guards_every_receiving_action(self):
        expired_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='EXPIRED-ACTIONS',
            expiry_date=date.today() - timedelta(days=2),
            quantity_on_hand=5,
        )
        selected = self.client.post(self.draft_url(), {
            'existing_lot_id': str(expired_lot.pk),
            'confirm_expired_lot': 'yes',
            'revision': '0',
        })
        self.assertEqual(selected.status_code, 200)

        add_url = reverse('add_quantity', args=[self.session.pk, self.product.pk])
        self.client.post(add_url, {
            'amount': '1',
            'existing_lot_id': str(expired_lot.pk),
        })
        self.product.refresh_from_db()
        expired_lot.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, expired_lot.quantity_on_hand), (5, 5))

        self.client.post(add_url, {
            'amount': '1',
            'existing_lot_id': str(expired_lot.pk),
            'confirm_expired_lot': 'yes',
        })
        self.product.refresh_from_db()
        expired_lot.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, expired_lot.quantity_on_hand), (6, 6))

        scan_url = reverse('checkin_session', args=[self.session.pk])
        scan_payload = {
            'barcode': self.product.barcode,
            'current_barcode': self.product.barcode,
            'existing_lot_id': str(expired_lot.pk),
        }
        self.client.post(scan_url, scan_payload)
        self.product.refresh_from_db()
        expired_lot.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, expired_lot.quantity_on_hand), (6, 6))

        self.client.post(scan_url, {
            **scan_payload,
            'confirm_expired_lot': 'yes',
        })
        self.product.refresh_from_db()
        expired_lot.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, expired_lot.quantity_on_hand), (7, 7))

        set_url = reverse('set_quantity', args=[self.session.pk, self.product.pk])
        self.client.post(set_url, {
            'quantity': '9',
            'existing_lot_id': str(expired_lot.pk),
        })
        self.product.refresh_from_db()
        expired_lot.refresh_from_db()
        self.assertEqual((self.product.quantity_in_stock, expired_lot.quantity_on_hand), (7, 7))

        self.client.post(set_url, {
            'quantity': '9',
            'existing_lot_id': str(expired_lot.pk),
            'confirm_expired_lot': 'yes',
        })
        self.product.refresh_from_db()
        expired_lot.refresh_from_db()
        draft = CheckinReceivingDraft.objects.get(
            session=self.session, product=self.product,
        )
        self.assertEqual((self.product.quantity_in_stock, expired_lot.quantity_on_hand), (9, 9))
        self.assertEqual(draft.existing_lot, expired_lot)

    def test_literal_main_can_be_selected_as_a_receiving_destination(self):
        main_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='MAIN',
            expiry_date=date.today() + timedelta(days=90),
            quantity_on_hand=5,
        )

        draft_response = self.client.post(self.draft_url(), {
            'existing_lot_id': str(main_lot.pk),
            'revision': '0',
        })
        receive_response = self.client.post(
            reverse('add_quantity', args=[self.session.pk, self.product.pk]),
            {
                'amount': '1',
                'existing_lot_id': str(main_lot.pk),
            },
            follow=True,
        )

        self.assertEqual(draft_response.status_code, 200)
        self.assertEqual(receive_response.status_code, 200)
        draft = CheckinReceivingDraft.objects.get(
            session=self.session,
            product=self.product,
        )
        self.assertEqual(draft.existing_lot, main_lot)
        self.assertContains(receive_response, 'lot MAIN')
        self.product.refresh_from_db()
        main_lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 6)
        self.assertEqual(main_lot.quantity_on_hand, 6)

    def test_literal_main_can_be_created_as_a_new_receiving_lot(self):
        ProductLot.objects.create(
            product=self.product,
            lot_number='BASE-LOT',
            expiry_date=date.today() + timedelta(days=60),
            quantity_on_hand=5,
        )
        expiry = date.today() + timedelta(days=120)

        draft_response = self.client.post(self.draft_url(), {
            'lot_number': 'MAIN',
            'lot_expiry': expiry.strftime('%d-%m-%Y'),
            'revision': '0',
        })
        receive_response = self.client.post(
            reverse('add_quantity', args=[self.session.pk, self.product.pk]),
            {
                'amount': '1',
                'lot_number': 'MAIN',
                'lot_expiry': expiry.strftime('%d-%m-%Y'),
            },
            follow=True,
        )

        self.assertEqual(draft_response.status_code, 200)
        self.assertEqual(receive_response.status_code, 200)
        main_lot = ProductLot.objects.get(
            product=self.product,
            lot_number='MAIN',
            expiry_date=expiry,
        )
        self.assertEqual(main_lot.quantity_on_hand, 1)
        self.assertNotEqual(main_lot.lot_number, ProductLot.UNASSIGNED)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 6)
        draft = CheckinReceivingDraft.objects.get(
            session=self.session,
            product=self.product,
        )
        self.assertEqual(draft.existing_lot, main_lot)

    def test_internal_main_cannot_be_selected_with_expired_confirmation(self):
        internal_main = ProductLot.objects.create(
            product=self.product,
            lot_number=ProductLot.UNASSIGNED,
            quantity_on_hand=5,
        )

        draft_response = self.client.post(self.draft_url(), {
            'existing_lot_id': str(internal_main.pk),
            'confirm_expired_lot': 'yes',
            'revision': '0',
        })
        receive_response = self.client.post(
            reverse('add_quantity', args=[self.session.pk, self.product.pk]),
            {
                'amount': '1',
                'existing_lot_id': str(internal_main.pk),
                'confirm_expired_lot': 'yes',
            },
            follow=True,
        )

        self.assertEqual(draft_response.status_code, 400)
        self.assertEqual(draft_response.json()['field'], 'existing_lot_id')
        self.assertContains(
            receive_response,
            'saved lot is no longer available for this product',
        )
        self.assertFalse(CheckinReceivingDraft.objects.exists())
        self.product.refresh_from_db()
        internal_main.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
        self.assertEqual(internal_main.quantity_on_hand, 5)

    def test_zero_balance_named_lot_can_be_reused_by_repeat_scan(self):
        depleted_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='KNOWN-DEPLETED',
            expiry_date=date.today() + timedelta(days=90),
            quantity_on_hand=0,
        )
        selected = self.client.post(self.draft_url(), {
            'existing_lot_id': str(depleted_lot.pk), 'revision': '0',
        })

        response = self.client.post(
            reverse('checkin_session', args=[self.session.pk]),
            {
                'barcode': self.product.barcode,
                'current_barcode': self.product.barcode,
                'existing_lot_id': str(depleted_lot.pk),
            },
        )

        self.assertEqual(selected.status_code, 200)
        self.assertEqual(response.status_code, 302)
        self.product.refresh_from_db()
        depleted_lot.refresh_from_db()
        draft = CheckinReceivingDraft.objects.get(
            session=self.session, product=self.product,
        )
        self.assertEqual(self.product.quantity_in_stock, 6)
        self.assertEqual(depleted_lot.quantity_on_hand, 1)
        self.assertEqual(draft.existing_lot, depleted_lot)

    def test_successful_receive_promotes_lot_and_rejects_late_stale_autosave(self):
        first = self.client.post(self.draft_url(), {
            'lot_number': 'SHIP-22', 'lot_expiry': '31-12-2032', 'revision': '0',
        })
        self.assertEqual(first.status_code, 200)

        received = self.client.post(
            reverse('add_quantity', args=[self.session.pk, self.product.pk]),
            {'amount': '1', 'lot_number': 'SHIP-22', 'lot_expiry': '31-12-2032'},
        )
        self.assertEqual(received.status_code, 302)
        lot = ProductLot.objects.get(
            product=self.product, lot_number='SHIP-22', expiry_date=date(2032, 12, 31),
        )
        draft = CheckinReceivingDraft.objects.get(session=self.session, product=self.product)
        self.assertEqual(draft.existing_lot, lot)
        self.assertEqual(draft.revision, 2)

        stale = self.client.post(self.draft_url(), {
            'lot_number': 'LATE-OVERWRITE',
            'lot_expiry': '01-01-2035',
            'revision': '1',
        })
        self.assertEqual(stale.status_code, 409)
        self.assertTrue(stale.json()['conflict'])
        draft.refresh_from_db()
        self.assertEqual((draft.existing_lot, draft.lot_number, draft.revision),
                         (lot, 'SHIP-22', 2))

        page = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': self.product.pk},
        )
        self.assertEqual(page.context['selected_receiving_lot_id'], lot.pk)

    def test_exact_increase_retains_selected_saved_lot_as_draft(self):
        lot = ProductLot.objects.create(
            product=self.product, lot_number='EXACT-LOT',
            expiry_date=date.today() + timedelta(days=120), quantity_on_hand=5,
        )
        self.client.post(self.draft_url(), {
            'existing_lot_id': str(lot.pk), 'revision': '0',
        })

        response = self.client.post(
            reverse('set_quantity', args=[self.session.pk, self.product.pk]),
            {'quantity': '7', 'existing_lot_id': str(lot.pk)},
        )

        self.assertEqual(response.status_code, 302)
        draft = CheckinReceivingDraft.objects.get(session=self.session, product=self.product)
        self.assertEqual((draft.existing_lot, draft.revision), (lot, 2))
        lot.refresh_from_db()
        self.assertEqual(lot.quantity_on_hand, 7)

    def test_repeat_barcode_scan_retains_selected_saved_lot_as_draft(self):
        lot = ProductLot.objects.create(
            product=self.product, lot_number='SCAN-LOT',
            expiry_date=date.today() + timedelta(days=120), quantity_on_hand=5,
        )
        self.client.post(self.draft_url(), {
            'existing_lot_id': str(lot.pk), 'revision': '0',
        })

        response = self.client.post(
            reverse('checkin_session', args=[self.session.pk]),
            {
                'barcode': self.product.barcode,
                'current_barcode': self.product.barcode,
                'existing_lot_id': str(lot.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        draft = CheckinReceivingDraft.objects.get(session=self.session, product=self.product)
        self.assertEqual((draft.existing_lot, draft.revision), (lot, 2))
        lot.refresh_from_db()
        self.assertEqual(lot.quantity_on_hand, 6)

    def test_unassigned_receive_does_not_restore_literal_unassigned(self):
        self.client.post(self.draft_url(), {
            'lot_number': 'OLD-DRAFT', 'lot_expiry': '', 'revision': '0',
        })

        response = self.client.post(
            reverse('add_quantity', args=[self.session.pk, self.product.pk]),
            {'amount': '1', 'lot_number': '', 'lot_expiry': ''},
        )

        self.assertEqual(response.status_code, 302)
        draft = CheckinReceivingDraft.objects.get(
            session=self.session, product=self.product,
        )
        self.assertEqual(
            (draft.existing_lot_id, draft.lot_number, draft.lot_expiry, draft.revision),
            (None, '', None, 2),
        )
        page = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': self.product.pk},
        )
        self.assertEqual(page.context['receiving_draft_lot_number'], '')
        self.assertIsNone(page.context['selected_receiving_lot_id'])

    def test_inventory_count_rejects_receiving_draft_without_changing_stock(self):
        count_session = CheckinSession.objects.create(
            user=self.user, scanned_by='Counter', inventory_mode=True,
        )

        response = self.client.post(self.draft_url(session=count_session), {
            'lot_number': 'COUNT-LOT', 'lot_expiry': '31-12-2030', 'revision': '0',
        })

        self.assertEqual(response.status_code, 400)
        self.assertFalse(CheckinReceivingDraft.objects.filter(session=count_session).exists())
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 5)
