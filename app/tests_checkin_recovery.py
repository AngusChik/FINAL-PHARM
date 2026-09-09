from datetime import date, timedelta
from decimal import Decimal
import threading
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.db import close_old_connections, connection
from django.test import RequestFactory, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import (
    Category, CheckinReceivingDraft, CheckinSession, InventoryCountLine,
    Product, StockChange, UserAction,
)


@override_settings(AXES_ENABLED=False)
class CheckinRecoveryTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='checkin-recovery', password='test-password', is_staff=True,
        )
        self.client.force_login(self.user)
        self.product = Product.objects.create(
            name='Retained receiving product', barcode='RECOVERY-CHECKIN-1',
            price=Decimal('4.50'), quantity_in_stock=12,
            category=Category.objects.create(name='Recovery check-in'),
        )
        self.session = CheckinSession.objects.create(
            user=self.user, scanned_by='AB', note='Keep this receiving note',
            ended_at=timezone.now(),
        )
        self.change = StockChange.objects.create(
            product=self.product, session=self.session, user=self.user,
            change_type='checkin', quantity=2, product_name=self.product.name,
            product_barcode=self.product.barcode,
        )
        self.count = InventoryCountLine.objects.create(
            session=self.session, product=self.product, product_name=self.product.name,
            product_barcode=self.product.barcode, expected_qty=10, counted_qty=12,
        )
        self.draft = CheckinReceivingDraft.objects.create(
            session=self.session, product=self.product, lot_number='KEEP-LOT',
            lot_expiry=date(2030, 12, 31), revision=3,
        )

    def archive(self, session=None):
        return self.client.post(reverse(
            'checkin_session_delete', args=[(session or self.session).pk],
        ))

    def restore(self, session=None):
        return self.client.post(reverse('archive_recovery'), {
            'kind': 'checkin', 'object_id': (session or self.session).pk,
            'type': 'checkin',
        })

    def assert_contents_retained(self):
        self.session.refresh_from_db()
        self.change.refresh_from_db()
        self.count.refresh_from_db()
        self.draft.refresh_from_db()
        self.product.refresh_from_db()
        self.assertEqual(self.change.session_id, self.session.pk)
        # The unfiltered base manager keeps historical foreign keys resolvable.
        self.assertEqual(StockChange.objects.get(pk=self.change.pk).session.pk, self.session.pk)
        self.assertEqual(self.count.session_id, self.session.pk)
        self.assertEqual((self.count.expected_qty, self.count.counted_qty), (10, 12))
        self.assertEqual(self.draft.session_id, self.session.pk)
        self.assertEqual(self.draft.lot_number, 'KEEP-LOT')
        self.assertEqual(self.draft.lot_expiry, date(2030, 12, 31))
        self.assertEqual(self.draft.revision, 3)
        self.assertEqual(self.session.note, 'Keep this receiving note')
        self.assertEqual(self.product.quantity_in_stock, 12)
        self.assertEqual(StockChange.objects.count(), 1)

    def test_removal_archives_whole_session_without_detaching_or_replaying_stock(self):
        started_at, ended_at = self.session.started_at, self.session.ended_at

        response = self.archive()

        self.assertRedirects(response, reverse('checkin_dashboard'), fetch_redirect_response=False)
        self.assert_contents_retained()
        self.assertIsNotNone(self.session.archived_at)
        self.assertEqual(self.session.archived_by, self.user)
        self.assertEqual((self.session.started_at, self.session.ended_at), (started_at, ended_at))
        self.assertFalse(CheckinSession.objects.filter(pk=self.session.pk).exists())
        self.assertTrue(CheckinSession.all_objects.filter(pk=self.session.pk).exists())
        self.assertTrue(UserAction.objects.filter(
            action='delete_session', target=f'Session #{self.session.pk}',
            detail__contains='Moved to Recovery',
        ).exists())

    def test_clear_history_archives_only_visible_completed_sessions_and_is_repeatable(self):
        active = CheckinSession.objects.create(user=self.user, scanned_by='ACTIVE')
        already_archived = CheckinSession.all_objects.create(
            user=self.user, ended_at=timezone.now(),
            archived_at=timezone.now() - timedelta(days=2),
            archive_reason='Earlier archive',
        )
        original_archive_time = already_archived.archived_at

        for _ in range(2):
            response = self.client.post(reverse('checkin_clear_history'))
            self.assertEqual(response.status_code, 302)

        self.assert_contents_retained()
        active.refresh_from_db()
        already_archived.refresh_from_db()
        self.assertIsNotNone(self.session.archived_at)
        self.assertIsNone(active.archived_at)
        self.assertTrue(active.is_active)
        self.assertEqual(already_archived.archived_at, original_archive_time)
        self.assertEqual(already_archived.archive_reason, 'Earlier archive')
        self.assertTrue(UserAction.objects.filter(
            action='clear_session_history', target='0 sessions cleared',
        ).exists())

    def test_recovery_lists_and_filters_archived_sessions(self):
        self.archive()

        for query in (str(self.session.pk), 'AB', self.product.name, self.product.barcode):
            with self.subTest(query=query):
                response = self.client.get(reverse('archive_recovery'), {
                    'type': 'checkin', 'q': query,
                    'date_from': timezone.localdate().isoformat(),
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.context['page_obj'].paginator.count, 1)
                self.assertContains(response, f'Session #{self.session.pk}')
                self.assertContains(response, 'Completed')

    def test_restore_completed_session_retains_original_dates_and_history(self):
        original_dates = self.session.started_at, self.session.ended_at
        self.archive()

        response = self.restore()

        self.assertEqual(response.status_code, 302)
        self.assert_contents_retained()
        self.assertIsNone(self.session.archived_at)
        self.assertIsNone(self.session.archived_by)
        self.assertEqual(self.session.archive_reason, '')
        self.assertEqual((self.session.started_at, self.session.ended_at), original_dates)
        self.assertFalse(self.session.is_active)
        self.assertTrue(UserAction.objects.filter(
            action='restore_archived_record', detail='checkin',
        ).exists())
        self.assertEqual(self.restore().status_code, 404)
        self.assertEqual(StockChange.objects.count(), 1)

    def test_active_count_restores_its_buffer_without_applying_the_count(self):
        CheckinSession.objects.filter(pk=self.session.pk).update(
            ended_at=None, inventory_mode=True,
        )
        self.archive()
        self.session.refresh_from_db()
        self.assertFalse(self.session.is_active)

        self.restore()

        self.assert_contents_retained()
        self.assertTrue(self.session.is_active)
        self.assertIsNone(self.session.ended_at)

    def test_archived_sessions_disappear_from_active_and_history_views(self):
        self.archive()
        active = CheckinSession.objects.create(user=self.user, scanned_by='archived-active')
        self.archive(active)

        response = self.client.get(reverse('checkin_dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['active_sessions'])
        self.assertFalse(response.context['sessions_page'].object_list)
        self.assertContains(response, reverse('archive_recovery') + '?type=checkin')

    def test_archived_session_urls_cannot_read_or_modify_hidden_session(self):
        self.archive()
        for name in ('checkin_session', 'checkin_session_detail', 'checkin_reconcile', 'checkin_session_pdf'):
            with self.subTest(method='get', route=name):
                response = self.client.get(reverse(name, args=[self.session.pk]))
                self.assertEqual(response.status_code, 404)

        routes = [
            ('checkin_session', [self.session.pk]),
            ('checkin_end', [self.session.pk]),
            ('checkin_session_delete', [self.session.pk]),
            ('checkin_session_reopen', [self.session.pk]),
            ('checkin_reconcile', [self.session.pk]),
            ('checkin_session_adjust', [self.session.pk, self.change.pk]),
            ('checkin_session_remove_line', [self.session.pk, self.change.pk]),
            ('add_quantity', [self.session.pk, self.product.pk]),
            ('delete_one', [self.session.pk, self.product.pk]),
            ('set_quantity', [self.session.pk, self.product.pk]),
            ('checkin_receiving_draft', [self.session.pk, self.product.pk]),
            ('checkin_reassign_lot', [self.session.pk, self.product.pk]),
            ('checkin_edit_product', [self.session.pk, self.product.pk]),
            ('checkin_add_by_id', [self.session.pk, self.product.pk]),
        ]
        for name, args in routes:
            with self.subTest(method='post', route=name):
                response = self.client.post(reverse(name, args=args), {
                    'amount': 1, 'new_qty': 99, 'quantity': 99,
                    'lot_number': 'MUST-NOT-SAVE', 'revision': 3,
                })
                self.assertEqual(response.status_code, 404)
        self.assert_contents_retained()

    def test_regular_user_cannot_archive_or_restore_without_admin_access(self):
        self.archive()
        regular = get_user_model().objects.create_user(username='regular-checkin-recovery')
        self.client.force_login(regular)

        response = self.restore()

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('passkey_unlock'), response.url)
        self.session.refresh_from_db()
        self.assertIsNotNone(self.session.archived_at)
        visible = CheckinSession.objects.create(user=regular)
        response = self.archive(visible)
        self.assertEqual(response.status_code, 302)
        visible.refresh_from_db()
        self.assertIsNone(visible.archived_at)


@override_settings(AXES_ENABLED=False)
class CheckinRecoveryConcurrencyTests(TransactionTestCase):
    def test_archive_waits_for_inflight_count_and_keeps_its_final_progress(self):
        if connection.vendor != 'postgresql':
            self.skipTest('PostgreSQL session-row locking contract')
        from . import views

        user = get_user_model().objects.create_user(username='archive-race', is_staff=True)
        product = Product.objects.create(
            name='Concurrent count', barcode='COUNT-ARCHIVE-RACE',
            price=Decimal('1.00'), quantity_in_stock=12,
            category=Category.objects.create(name='Concurrent recovery'),
        )
        session = CheckinSession.objects.create(user=user, inventory_mode=True)
        count = InventoryCountLine.objects.create(
            session=session, product=product, expected_qty=12, counted_qty=0,
        )
        count_entered = threading.Event()
        release_count = threading.Event()
        archive_started = threading.Event()
        archive_finished = threading.Event()
        failures, results = [], []
        adjust_count = views._adjust_inventory_count

        def paused_count(*args, **kwargs):
            count_entered.set()
            if not release_count.wait(5):
                raise AssertionError('Count was not released in time.')
            return adjust_count(*args, **kwargs)

        def request_for(url, data=None):
            request = RequestFactory().post(url, data or {})
            request.user = user
            request.session = {}
            request._messages = FallbackStorage(request)
            return request

        def receive_count():
            close_old_connections()
            try:
                response = views.AddQuantityView(
                    request_for(reverse('add_quantity', args=[session.pk, product.pk]), {'amount': 1}),
                    session_id=session.pk, product_id=product.pk,
                )
                results.append(response.status_code)
            except BaseException as exc:
                failures.append(exc)
            finally:
                close_old_connections()

        def archive_session():
            close_old_connections()
            try:
                archive_started.set()
                response = views.DeleteCheckinSessionView.as_view()(
                    request_for(reverse('checkin_session_delete', args=[session.pk])),
                    session_id=session.pk,
                )
                results.append(response.status_code)
            except BaseException as exc:
                failures.append(exc)
            finally:
                archive_finished.set()
                close_old_connections()

        writer = threading.Thread(target=receive_count, daemon=True)
        archiver = threading.Thread(target=archive_session, daemon=True)
        with patch('app.views._adjust_inventory_count', side_effect=paused_count):
            try:
                writer.start()
                self.assertTrue(count_entered.wait(5), 'Count did not enter its write.')
                archiver.start()
                self.assertTrue(archive_started.wait(5), 'Archive did not start.')
                self.assertFalse(archive_finished.wait(.2), 'Archive overtook an active count write.')
            finally:
                release_count.set()
                writer.join(10)
                if archiver.ident is not None:
                    archiver.join(10)
        self.assertFalse(writer.is_alive())
        self.assertFalse(archiver.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(results, [302, 302])
        session.refresh_from_db()
        count.refresh_from_db()
        product.refresh_from_db()
        self.assertIsNotNone(session.archived_at)
        self.assertEqual(count.counted_qty, 1)
        self.assertEqual(product.quantity_in_stock, 12)
        self.assertFalse(StockChange.objects.exists())
