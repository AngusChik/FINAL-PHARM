from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import threading
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import close_old_connections, connection, transaction
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .inventory_services import (
    reassign_lot_stock,
    repair_lot_balance_to_main,
)
from .models import (
    Category,
    CheckinSession,
    Product,
    ProductLot,
    ProductLotMovement,
    StockChange,
    UserAction,
    display_lot_name,
    is_reserved_new_lot_name,
)
from .views import ActivityLogView, CheckinProductView


class RecordingCanvas:
    strings = []

    def __init__(self, buffer, pagesize=None):
        self.buffer = buffer
        type(self).strings = []

    def drawString(self, _x, _y, value):
        type(self).strings.append(str(value))

    def drawRightString(self, _x, _y, value):
        type(self).strings.append(str(value))

    def drawCentredString(self, _x, _y, value):
        type(self).strings.append(str(value))

    def setFont(self, *_args):
        pass

    def setFillColor(self, *_args):
        pass

    def setStrokeColor(self, *_args):
        pass

    def setLineWidth(self, *_args):
        pass

    def line(self, *_args):
        pass

    def rect(self, *_args, **_kwargs):
        pass

    def showPage(self):
        pass

    def save(self):
        self.buffer.write(b'%PDF-1.4\n% lot reassignment test\n%%EOF')


class LotReassignmentFixtureMixin:
    def setUp(self):
        self.user = User.objects.create_user(
            username='checkin-lot-user',
            password='test-password',
            is_staff=False,
        )
        self.admin = User.objects.create_user(
            username='checkin-lot-admin',
            password='test-password',
            is_staff=True,
        )
        self.category = Category.objects.create(name='Lot Assignment Tests')
        self.product = self.make_product()
        self.main_lot = ProductLot.objects.create(
            product=self.product,
            lot_number=ProductLot.UNASSIGNED,
            quantity_on_hand=6,
        )
        self.source_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='SUPPLIER-A',
            expiry_date=date.today() + timedelta(days=90),
            quantity_on_hand=4,
        )
        self.destination_lot = ProductLot.objects.create(
            product=self.product,
            lot_number='SUPPLIER-B',
            expiry_date=date.today() + timedelta(days=120),
            quantity_on_hand=0,
        )
        self.session = CheckinSession.objects.create(
            user=self.user,
            scanned_by='Check-in user',
        )

    def make_product(
        self,
        *,
        name='Lot Assignment Product',
        barcode='LOT-MOVE-100',
        quantity=10,
    ):
        return Product.objects.create(
            name=name,
            barcode=barcode,
            price=Decimal('8.00'),
            quantity_in_stock=quantity,
            category=self.category,
            stock_bought=13,
            stock_sold=3,
            stock_expired=1,
            stock_unfulfilled=2,
            stock_giveaway=1,
            stock_deleted=1,
        )

    @staticmethod
    def counters(product):
        return (
            product.stock_bought,
            product.stock_sold,
            product.stock_expired,
            product.stock_unfulfilled,
            product.stock_giveaway,
            product.stock_deleted,
        )

    def move(
        self,
        *,
        product=None,
        source=None,
        quantity=1,
        expected=None,
        destination_kind='existing',
        session=None,
        **destination,
    ):
        product = product or self.product
        source = source or self.main_lot
        expected = source.quantity_on_hand if expected is None else expected
        if destination_kind == 'existing' and 'destination_lot_id' not in destination:
            destination['destination_lot_id'] = self.destination_lot.pk
        return reassign_lot_stock(
            product,
            source_lot_id=source.pk,
            expected_source_balance=expected,
            quantity=quantity,
            destination_kind=destination_kind,
            user=self.user,
            session=self.session if session is None else session,
            **destination,
        )

    def endpoint_payload(
        self,
        *,
        source=None,
        quantity=1,
        expected=None,
        destination_kind='existing',
        **extra,
    ):
        source = source or self.main_lot
        expected = source.quantity_on_hand if expected is None else expected
        payload = {
            'source_lot_id': str(source.pk),
            'source_expected_balance': str(expected),
            'quantity': str(quantity),
            'destination_kind': destination_kind,
        }
        if destination_kind == 'existing' and 'destination_lot_id' not in extra:
            extra['destination_lot_id'] = str(self.destination_lot.pk)
        payload.update(extra)
        return payload


@override_settings(AXES_ENABLED=False)
class LotReassignmentServiceTests(LotReassignmentFixtureMixin, TestCase):
    def test_main_to_named_is_neutral_paired_and_preserves_totals(self):
        before_counters = self.counters(self.product)

        result = self.move(
            source=self.main_lot,
            quantity=2,
            destination_lot_id=self.destination_lot.pk,
        )

        self.product.refresh_from_db()
        self.main_lot.refresh_from_db()
        self.destination_lot.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)
        self.assertEqual(self.counters(self.product), before_counters)
        self.assertEqual(self.main_lot.quantity_on_hand, 4)
        self.assertEqual(self.destination_lot.quantity_on_hand, 2)
        change = result['stock_change']
        self.assertEqual(change.change_type, 'lot_reassignment')
        self.assertEqual(change.get_change_type_display(), 'Moved')
        self.assertEqual(str(change), f'{self.product.name}: 2 moved')
        self.assertIn('from UNASSIGNED', change.note)
        self.assertCountEqual(
            list(change.lot_movements.values_list(
                'direction', 'lot_number', 'quantity',
            )),
            [
                (ProductLotMovement.DIRECTION_OUT, ProductLot.UNASSIGNED, 2),
                (ProductLotMovement.DIRECTION_IN, 'SUPPLIER-B', 2),
            ],
        )

    def test_named_to_named_supports_partial_then_full_transfer(self):
        first = self.move(
            source=self.source_lot,
            quantity=2,
            destination_lot_id=self.destination_lot.pk,
        )
        self.source_lot.refresh_from_db()
        self.destination_lot.refresh_from_db()
        self.assertEqual((self.source_lot.quantity_on_hand, self.destination_lot.quantity_on_hand), (2, 2))
        self.assertEqual(first['source_before'], 4)
        self.assertEqual(first['destination_before'], 0)

        second = self.move(
            source=self.source_lot,
            quantity=2,
            expected=2,
            destination_lot_id=self.destination_lot.pk,
        )

        self.source_lot.refresh_from_db()
        self.destination_lot.refresh_from_db()
        self.product.refresh_from_db()
        self.assertEqual((self.source_lot.quantity_on_hand, self.destination_lot.quantity_on_hand), (0, 4))
        self.assertEqual(second['source_before'], 2)
        self.assertEqual(self.product.quantity_in_stock, 10)
        self.assertEqual(
            StockChange.objects.filter(change_type='lot_reassignment').count(),
            2,
        )

    def test_named_to_main_creates_then_reuses_expiry_matching_main(self):
        first = self.move(
            source=self.source_lot,
            quantity=1,
            destination_kind='main',
        )
        destination = first['destination']

        self.assertTrue(first['destination_created'])
        self.assertEqual(destination.lot_number, ProductLot.UNASSIGNED)
        self.assertEqual(destination.expiry_date, self.source_lot.expiry_date)
        self.assertEqual(destination.quantity_on_hand, 1)
        self.assertNotEqual(destination.pk, self.main_lot.pk)

        self.source_lot.refresh_from_db()
        second = self.move(
            source=self.source_lot,
            quantity=1,
            expected=3,
            destination_kind='main',
        )
        destination.refresh_from_db()
        self.assertFalse(second['destination_created'])
        self.assertEqual(second['destination'].pk, destination.pk)
        self.assertEqual(destination.quantity_on_hand, 2)

    def test_expected_source_balance_rejects_stale_then_allows_full_move(self):
        with self.assertRaisesMessage(ValidationError, 'stock changed'):
            self.move(
                source=self.source_lot,
                expected=5,
                quantity=1,
                destination_lot_id=self.destination_lot.pk,
            )
        self.source_lot.refresh_from_db()
        self.destination_lot.refresh_from_db()
        self.assertEqual((self.source_lot.quantity_on_hand, self.destination_lot.quantity_on_hand), (4, 0))

        self.move(
            source=self.source_lot,
            quantity=4,
            destination_lot_id=self.destination_lot.pk,
        )
        self.source_lot.refresh_from_db()
        self.destination_lot.refresh_from_db()
        self.assertEqual((self.source_lot.quantity_on_hand, self.destination_lot.quantity_on_hand), (0, 4))

    def test_zero_quantity_and_expired_saved_destinations_are_valid_choices(self):
        self.move(
            source=self.source_lot,
            quantity=1,
            destination_lot_id=self.destination_lot.pk,
        )
        self.destination_lot.refresh_from_db()
        self.assertEqual(self.destination_lot.quantity_on_hand, 1)

        expired = ProductLot.objects.create(
            product=self.product,
            lot_number='EXPIRED-DEST',
            expiry_date=date.today() - timedelta(days=1),
            quantity_on_hand=0,
        )
        self.source_lot.refresh_from_db()
        with self.assertRaisesMessage(ValidationError, 'already expired'):
            self.move(
                source=self.source_lot,
                expected=3,
                quantity=1,
                destination_lot_id=expired.pk,
            )
        expired.refresh_from_db()
        self.assertEqual(expired.quantity_on_hand, 0)

        self.move(
            source=self.source_lot,
            expected=3,
            quantity=1,
            destination_lot_id=expired.pk,
            confirm_past_expiry=True,
        )
        expired.refresh_from_db()
        self.assertEqual(expired.quantity_on_hand, 1)

    def test_expired_source_reclassification_requires_separate_confirmation(self):
        product = self.make_product(
            name='Expired source product',
            barcode='LOT-MOVE-EXPIRED',
            quantity=2,
        )
        expired_source = ProductLot.objects.create(
            product=product,
            lot_number='EXPIRED-SOURCE',
            expiry_date=date.today() - timedelta(days=2),
            quantity_on_hand=2,
        )
        future_destination = ProductLot.objects.create(
            product=product,
            lot_number='FUTURE-DEST',
            expiry_date=date.today() + timedelta(days=100),
            quantity_on_hand=0,
        )

        with self.assertRaisesMessage(ValidationError, 'currently belong to an expired lot'):
            self.move(
                product=product,
                source=expired_source,
                quantity=1,
                destination_lot_id=future_destination.pk,
            )
        self.move(
            product=product,
            source=expired_source,
            quantity=1,
            destination_lot_id=future_destination.pk,
            confirm_expiry_reclassification=True,
        )
        expired_source.refresh_from_db()
        future_destination.refresh_from_db()
        self.assertEqual((expired_source.quantity_on_hand, future_destination.quantity_on_hand), (1, 1))

    def test_same_identity_wrong_product_and_archived_rows_are_rejected(self):
        with self.assertRaisesMessage(ValidationError, 'different from the source'):
            self.move(
                source=self.source_lot,
                destination_lot_id=self.source_lot.pk,
            )

        other = self.make_product(
            name='Other lot product',
            barcode='LOT-MOVE-OTHER',
            quantity=1,
        )
        other_lot = ProductLot.objects.create(
            product=other,
            lot_number='OTHER-LOT',
            quantity_on_hand=1,
        )
        with self.assertRaisesMessage(ValidationError, 'source lot is no longer active'):
            self.move(source=other_lot, destination_lot_id=self.destination_lot.pk)
        with self.assertRaisesMessage(ValidationError, 'destination lot is no longer active'):
            self.move(source=self.main_lot, destination_lot_id=other_lot.pk)

        archived_destination = ProductLot.objects.create(
            product=self.product,
            lot_number='ARCHIVED-DEST',
            quantity_on_hand=0,
            archived_at=timezone.now(),
            archived_by=self.admin,
        )
        with self.assertRaisesMessage(ValidationError, 'destination lot is no longer active'):
            self.move(
                source=self.main_lot,
                destination_lot_id=archived_destination.pk,
            )

    def test_product_and_lot_total_mismatches_fail_without_writes(self):
        self.product.quantity_in_stock = 11
        self.product.save(update_fields=['quantity_in_stock'])
        with self.assertRaisesMessage(ValidationError, 'Run Inventory Audit'):
            self.move(source=self.main_lot, destination_lot_id=self.destination_lot.pk)
        self.assertFalse(
            StockChange.objects.filter(change_type='lot_reassignment').exists()
        )

        archived_product = self.make_product(
            name='Archived move product',
            barcode='LOT-MOVE-ARCHIVED',
            quantity=1,
        )
        archived_source = ProductLot.objects.create(
            product=archived_product,
            lot_number='ARCHIVED-SOURCE',
            quantity_on_hand=1,
        )
        archived_destination = ProductLot.objects.create(
            product=archived_product,
            lot_number='ACTIVE-DEST',
            quantity_on_hand=0,
        )
        archived_product.archived_at = timezone.now()
        archived_product.archived_by = self.admin
        archived_product.save(update_fields=['archived_at', 'archived_by'])
        with self.assertRaisesMessage(ValidationError, 'moved to Recovery'):
            self.move(
                product=archived_product,
                source=archived_source,
                destination_lot_id=archived_destination.pk,
            )

    def test_new_destination_allows_main_but_reserves_unassigned(self):
        with self.assertRaisesMessage(ValidationError, 'UNASSIGNED is reserved'):
            self.move(
                source=self.main_lot,
                destination_kind='new',
                destination_lot_number=ProductLot.UNASSIGNED,
            )

        result = self.move(
            source=self.main_lot,
            destination_kind='new',
            destination_lot_number='MAIN',
        )

        self.assertTrue(result['destination_created'])
        self.assertEqual(result['destination'].lot_number, 'MAIN')
        self.assertEqual(result['destination'].quantity_on_hand, 1)
        self.assertIn('to MAIN;', result['stock_change'].note)
        self.assertNotIn('MAIN (recorded lot)', result['stock_change'].note)

    def test_new_destination_rejects_existing_or_archived_identity(self):
        self.main_lot.refresh_from_db()

        with self.assertRaisesMessage(ValidationError, 'already exists'):
            self.move(
                source=self.main_lot,
                destination_kind='new',
                destination_lot_number=self.destination_lot.lot_number,
                destination_expiry_date=self.destination_lot.expiry_date,
            )

        archived = ProductLot.objects.create(
            product=self.product,
            lot_number='ARCHIVED-IDENTITY',
            expiry_date=date.today() + timedelta(days=30),
            quantity_on_hand=0,
            archived_at=timezone.now(),
            archived_by=self.admin,
        )
        with self.assertRaisesMessage(ValidationError, 'Restore it through Inventory Audit'):
            self.move(
                source=self.main_lot,
                destination_kind='new',
                destination_lot_number=archived.lot_number,
                destination_expiry_date=archived.expiry_date,
            )

    def test_main_is_a_plain_named_lot_while_unassigned_remains_reserved(self):
        self.assertEqual(display_lot_name('MAIN'), 'MAIN')
        self.assertFalse(is_reserved_new_lot_name('MAIN'))
        self.assertEqual(
            display_lot_name(ProductLot.UNASSIGNED),
            ProductLot.UNASSIGNED,
        )
        self.assertTrue(is_reserved_new_lot_name(ProductLot.UNASSIGNED))

    def test_session_state_is_revalidated_by_the_service(self):
        ended = CheckinSession.objects.create(
            user=self.user,
            scanned_by='Ended',
            ended_at=timezone.now(),
        )
        count = CheckinSession.objects.create(
            user=self.user,
            scanned_by='Counter',
            inventory_mode=True,
        )
        with self.assertRaisesMessage(ValidationError, 'session ended'):
            self.move(session=ended)
        with self.assertRaisesMessage(ValidationError, 'inventory count'):
            self.move(session=count)

    def test_residual_repair_creates_main_and_generic_audit_action(self):
        product = self.make_product(
            name='Residual stock product',
            barcode='LOT-RESIDUAL',
            quantity=5,
        )
        ProductLot.objects.create(
            product=product,
            lot_number='TRACKED',
            quantity_on_hand=2,
        )

        result = repair_lot_balance_to_main(
            product,
            user=self.user,
            session=self.session,
        )

        product.refresh_from_db()
        self.assertEqual(result['quantity'], 3)
        self.assertEqual(product.quantity_in_stock, 5)
        self.assertEqual(result['destination'].lot_number, ProductLot.UNASSIGNED)
        self.assertEqual(result['destination'].quantity_on_hand, 3)
        self.assertEqual(result['action'].action, 'repair_lot_balance')
        self.assertEqual(
            result['action'].get_action_display(),
            'Assigned Missing Stock to UNASSIGNED',
        )
        self.assertIn('previously untracked', result['action'].detail)

        no_op = repair_lot_balance_to_main(
            product,
            user=self.user,
            session=self.session,
        )
        self.assertEqual(no_op['quantity'], 0)
        self.assertIsNone(no_op['destination'])
        self.assertIsNone(no_op['action'])

    def test_residual_repair_refuses_negative_and_archived_main_identity(self):
        excessive = self.make_product(
            name='Excessive tracked stock',
            barcode='LOT-EXCESS',
            quantity=1,
        )
        ProductLot.objects.create(
            product=excessive,
            lot_number='TOO-MUCH',
            quantity_on_hand=2,
        )
        with self.assertRaisesMessage(ValidationError, 'exceed'):
            repair_lot_balance_to_main(excessive, session=self.session)

        archived_main_product = self.make_product(
            name='Archived MAIN residual',
            barcode='LOT-ARCH-MAIN',
            quantity=3,
        )
        ProductLot.objects.create(
            product=archived_main_product,
            lot_number=ProductLot.UNASSIGNED,
            quantity_on_hand=0,
            archived_at=timezone.now(),
            archived_by=self.admin,
        )
        with self.assertRaisesMessage(ValidationError, 'archived'):
            repair_lot_balance_to_main(
                archived_main_product,
                user=self.user,
                session=self.session,
            )
        self.assertFalse(
            UserAction.objects.filter(
                action='repair_lot_balance',
                target=archived_main_product.name,
            ).exists()
        )


@override_settings(AXES_ENABLED=False)
class LotReassignmentEndpointTests(LotReassignmentFixtureMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.user)
        self.url = reverse(
            'checkin_reassign_lot',
            args=[self.session.pk, self.product.pk],
        )

    def test_non_staff_lock_holder_can_move_named_stock_and_action_is_generic(self):
        response = self.client.post(
            self.url,
            self.endpoint_payload(
                source=self.source_lot,
                quantity=2,
                destination_lot_id=str(self.destination_lot.pk),
            ),
        )

        self.assertEqual(response.status_code, 302)
        self.source_lot.refresh_from_db()
        self.destination_lot.refresh_from_db()
        self.assertEqual((self.source_lot.quantity_on_hand, self.destination_lot.quantity_on_hand), (2, 2))
        action = UserAction.objects.get(action='reassign_product_lot')
        self.assertEqual(action.user, self.user)
        self.assertEqual(action.get_action_display(), 'Moved Stock Between Lots')
        self.assertIn('from SUPPLIER-A', action.detail)
        self.assertIn('to SUPPLIER-B', action.detail)

    def test_ajax_move_returns_compact_response_then_refreshes_inline_lots(self):
        mutation = self.client.post(
            self.url,
            self.endpoint_payload(
                source=self.source_lot,
                quantity=1,
                destination_lot_id=str(self.destination_lot.pk),
            ),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

        self.assertEqual(mutation.status_code, 200)
        payload = mutation.json()
        self.assertTrue(payload['ok'])
        self.assertIn('format=checkin_fragments', payload['fragments_url'])
        fragment = self.client.get(
            payload['fragments_url'],
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )
        self.assertEqual(fragment.status_code, 200)
        fragment_payload = fragment.json()
        self.assertIn('product_lots_html', fragment_payload)
        self.assertNotIn('lot_reassignment_html', fragment_payload)
        self.assertIn('id="assignLotStockButton"', fragment_payload['product_lots_html'])
        self.assertIn('SUPPLIER-A', fragment_payload['product_lots_html'])
        self.assertIn('SUPPLIER-B', fragment_payload['product_lots_html'])

    def test_double_submit_is_rejected_by_expected_source_balance(self):
        payload = self.endpoint_payload(
            source=self.source_lot,
            quantity=2,
            destination_lot_id=str(self.destination_lot.pk),
        )
        self.assertEqual(self.client.post(self.url, payload).status_code, 302)
        self.assertEqual(self.client.post(self.url, payload).status_code, 302)

        self.source_lot.refresh_from_db()
        self.destination_lot.refresh_from_db()
        self.assertEqual((self.source_lot.quantity_on_hand, self.destination_lot.quantity_on_hand), (2, 2))
        self.assertEqual(
            StockChange.objects.filter(change_type='lot_reassignment').count(),
            1,
        )

    def test_ended_and_inventory_count_sessions_are_rejected(self):
        ended = CheckinSession.objects.create(
            user=self.user,
            scanned_by='Ended',
            ended_at=timezone.now(),
        )
        count = CheckinSession.objects.create(
            user=self.user,
            scanned_by='Counter',
            inventory_mode=True,
        )
        for session in (ended, count):
            with self.subTest(session=session.pk):
                response = self.client.post(
                    reverse(
                        'checkin_reassign_lot',
                        args=[session.pk, self.product.pk],
                    ),
                    self.endpoint_payload(
                        source=self.source_lot,
                        destination_lot_id=str(self.destination_lot.pk),
                    ),
                )
                self.assertEqual(response.status_code, 302)
        self.source_lot.refresh_from_db()
        self.assertEqual(self.source_lot.quantity_on_hand, 4)
        self.assertFalse(
            StockChange.objects.filter(change_type='lot_reassignment').exists()
        )

    def test_wrong_product_lots_and_product_total_mismatch_are_rejected(self):
        other = self.make_product(
            name='Endpoint other product',
            barcode='LOT-ENDPOINT-OTHER',
            quantity=1,
        )
        other_lot = ProductLot.objects.create(
            product=other,
            lot_number='OTHER-ENDPOINT',
            quantity_on_hand=1,
        )
        for payload in (
            self.endpoint_payload(
                source=other_lot,
                destination_lot_id=str(self.destination_lot.pk),
            ),
            self.endpoint_payload(
                source=self.source_lot,
                destination_lot_id=str(other_lot.pk),
            ),
        ):
            with self.subTest(payload=payload):
                response = self.client.post(self.url, payload)
                self.assertEqual(response.status_code, 302)

        self.product.quantity_in_stock = 9
        self.product.save(update_fields=['quantity_in_stock'])
        mismatch = self.client.post(
            self.url,
            self.endpoint_payload(
                source=self.source_lot,
                destination_lot_id=str(self.destination_lot.pk),
            ),
        )
        self.assertEqual(mismatch.status_code, 302)
        self.assertFalse(
            StockChange.objects.filter(change_type='lot_reassignment').exists()
        )

    def test_archived_product_is_not_mutated(self):
        self.product.archived_at = timezone.now()
        self.product.archived_by = self.admin
        self.product.save(update_fields=['archived_at', 'archived_by'])

        response = self.client.post(
            self.url,
            self.endpoint_payload(
                source=self.source_lot,
                destination_lot_id=str(self.destination_lot.pk),
            ),
        )

        self.assertEqual(response.status_code, 404)
        self.source_lot.refresh_from_db()
        self.assertEqual(self.source_lot.quantity_on_hand, 4)

    def test_other_computer_page_lock_blocks_mutation(self):
        holder_user = User.objects.create_user(
            username='other-checkin-user',
            password='test-password',
        )
        holder = Client()
        holder.force_login(holder_user)
        session_url = reverse('checkin_session', args=[self.session.pk])
        self.assertEqual(holder.get(session_url).status_code, 200)

        blocked = self.client.post(
            self.url,
            self.endpoint_payload(
                source=self.source_lot,
                destination_lot_id=str(self.destination_lot.pk),
            ),
        )

        self.assertEqual(blocked.status_code, 409)
        self.source_lot.refresh_from_db()
        self.assertEqual(self.source_lot.quantity_on_hand, 4)

    def test_selecting_product_repairs_only_positive_untracked_residual(self):
        product = self.make_product(
            name='Inline residual product',
            barcode='LOT-INLINE-RESIDUAL',
            quantity=5,
        )
        ProductLot.objects.create(
            product=product,
            lot_number='TRACKED-INLINE',
            quantity_on_hand=2,
        )

        response = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': product.pk},
        )

        self.assertEqual(response.status_code, 200)
        main = ProductLot.objects.get(
            product=product,
            lot_number=ProductLot.UNASSIGNED,
            expiry_date__isnull=True,
        )
        self.assertEqual(main.quantity_on_hand, 3)
        action = UserAction.objects.get(
            action='repair_lot_balance',
            target=product.name,
        )
        self.assertEqual(
            action.get_action_display(),
            'Assigned Missing Stock to UNASSIGNED',
        )
        product.refresh_from_db()
        self.assertEqual(product.quantity_in_stock, 5)


@override_settings(AXES_ENABLED=False)
class InlineLotAssignmentPresentationTests(LotReassignmentFixtureMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.user)

    def test_product_lots_panel_shows_all_active_rows_but_receive_picker_stays_safe(self):
        expired = ProductLot.objects.create(
            product=self.product,
            lot_number='EXPIRED-ZERO',
            expiry_date=date.today() - timedelta(days=1),
            quantity_on_hand=0,
        )
        literal_main = ProductLot.objects.create(
            product=self.product,
            lot_number='MAIN',
            expiry_date=date.today() + timedelta(days=30),
            quantity_on_hand=0,
        )
        response = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': self.product.pk},
        )

        self.assertEqual(response.status_code, 200)
        display_ids = {
            lot.pk for lot in response.context['display_product_lots']
        }
        receivable_ids = {
            lot.pk for lot in response.context['saved_receiving_lots']
        }
        self.assertTrue({
            self.main_lot.pk,
            self.source_lot.pk,
            self.destination_lot.pk,
            expired.pk,
            literal_main.pk,
        }.issubset(display_ids))
        self.assertNotIn(self.main_lot.pk, receivable_ids)
        self.assertIn(expired.pk, receivable_ids)
        self.assertIn(
            expired.pk,
            response.context['expired_receiving_lot_ids'],
        )
        self.assertIn(literal_main.pk, receivable_ids)
        self.assertIn(self.source_lot.pk, receivable_ids)
        self.assertIn(self.destination_lot.pk, receivable_ids)
        self.assertContains(response, 'data-lot-display-name="MAIN"', html=False)
        self.assertNotContains(response, 'MAIN (recorded lot)')
        self.assertContains(response, 'id="assignLotStockButton"', html=False)
        self.assertContains(response, 'data-lot-assignment-choice', html=False)
        self.assertNotContains(response, 'Assign MAIN stock')

    def test_inline_partial_contract_is_accessible_mobile_and_replaces_standalone_panel(self):
        base = Path(settings.BASE_DIR)
        partial_path = (
            base / 'app' / 'templates' / 'partials'
            / '_checkin_product_lots_panel.html'
        )
        old_partial_path = (
            base / 'app' / 'templates' / 'partials'
            / '_checkin_lot_reassignment.html'
        )
        checkin_path = base / 'app' / 'templates' / 'checkin.html'
        partial = partial_path.read_text(encoding='utf-8')
        checkin = checkin_path.read_text(encoding='utf-8')

        self.assertFalse(old_partial_path.exists())
        self.assertNotIn('_checkin_lot_reassignment.html', checkin)
        self.assertEqual(checkin.count('_checkin_product_lots_panel.html'), 1)
        for hook in (
            'id="receivingSavedLots"',
            'id="assignLotStockButton"',
            'aria-pressed="false"',
            'id="cancelLotAssignment"',
            'data-lot-assignment-choice',
            'data-lot-assignment-quantity',
            'inputmode="none" autocomplete="off" readonly disabled',
            'aria-expanded="false"',
            'data-assignment-synthetic-main',
            'data-lot-assignment-main-choice',
            'id="lotAssignmentForm"',
            'data-lot-assignment-form',
            'id="lotAssignmentGuidance"',
            'role="status"',
            'aria-live="polite"',
        ):
            with self.subTest(hook=hook):
                self.assertIn(hook, partial)
        for field_name in (
            'source_lot_id',
            'source_expected_balance',
            'quantity',
            'destination_kind',
            'destination_lot_id',
            'destination_lot_number',
            'destination_lot_expiry',
            'confirm_past_expiry',
            'confirm_expiry_reclassification',
        ):
            with self.subTest(field=field_name):
                self.assertIn(f'name="{field_name}"', partial)
        self.assertIn('.lot-assignment-toggle', checkin)
        self.assertIn('min-height: 44px;', checkin)
        self.assertIn('.lot-assignment-row-button:focus-visible', checkin)
        self.assertIn('.lot-assignment-numpad-key:focus-visible', checkin)
        self.assertIn('.lot-assignment-inline-numpad {', checkin)
        self.assertIn('width: min(200px, 100%);', checkin)
        self.assertIn('.lot-assignment-numpad-key {', checkin)
        self.assertIn('.lot-assignment-inline-numpad-actions {', checkin)
        self.assertIn('.lot-assignment-inline-numpad-actions [data-lot-numpad-enter] {', checkin)
        self.assertIn('@media (max-width: 640px)', checkin)
        self.assertIn('.lot-assignment-row-controls { grid-template-columns:', checkin)
        self.assertIn('is-assignment-mode', checkin)
        self.assertIn("panel.closest('.receiving-strip')", checkin)
        self.assertNotIn("panel.closest('.receiving-restock-workspace')", checkin)
        self.assertIn('product_lots_html', checkin)
        for hook in (
            'function closeLotAssignmentNumpad(options)',
            'function openLotAssignmentNumpad(quantityInput)',
            "data-lot-numpad-digit",
            "data-lot-numpad-clear",
            "data-lot-numpad-backspace",
            "data-lot-numpad-cancel",
            "data-lot-numpad-enter",
            "function applyLotAssignmentNumpadValue()",
            "function selectOnlyUnassignedAssignmentSource()",
            "positiveSources.length !== 1",
            "dataset.assignmentInternalMain !== 'true'",
            "autoSelectedUnassigned = selectOnlyUnassignedAssignmentSource();",
            "UNASSIGNED selected as the source.",
            "controls.appendChild(pad);",
            "quantityInput.setAttribute('aria-controls', pad.id);",
            "quantityInput.setAttribute('aria-expanded', 'true');",
            "event.key === 'Backspace'",
            "event.key === 'Delete'",
            "event.key === 'Escape'",
            "event.key === 'Enter'",
            'event.stopPropagation();',
            'if (event.defaultPrevented) return;',
        ):
            with self.subTest(numpad_hook=hook):
                self.assertIn(hook, checkin)
        self.assertEqual(checkin.count("setAttribute('data-lot-numpad-digit'"), 2)
        self.assertIn("['1', '2', '3', '4', '5', '6', '7', '8', '9']", checkin)
        self.assertIn("form.querySelector('[name=\"quantity\"]')", checkin)
        self.assertIn("if (event.key === 'Escape' && lotAssignmentNumpad)", checkin)
        self.assertNotIn('aria-haspopup="dialog"', partial)
        self.assertNotIn("window.uiDialog({", checkin)
        self.assertNotIn('data-lot-numpad-display', checkin)
        self.assertNotIn('window.prompt(', checkin)

    def test_lot_mismatch_disables_assignment_and_directs_staff_to_audit(self):
        # A positive residual is deliberately repaired into MAIN on selection.
        # Use excess tracked lot stock so the unsafe mismatch remains blocked.
        self.product.quantity_in_stock = 9
        self.product.save(update_fields=['quantity_in_stock'])
        response = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {'product_id': self.product.pk},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Run Inventory Audit')
        self.assertContains(
            response,
            'id="assignLotStockButton"',
            html=False,
        )
        self.assertContains(
            response,
            'disabled aria-describedby="lotAssignmentBlocked"',
            html=False,
        )

    def test_fragment_refresh_is_read_only_and_does_not_repair_residual_stock(self):
        self.product.quantity_in_stock = 12
        self.product.save(update_fields=['quantity_in_stock'])

        response = self.client.get(
            reverse('checkin_session', args=[self.session.pk]),
            {
                'product_id': self.product.pk,
                'format': 'checkin_fragments',
            },
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

        self.assertEqual(response.status_code, 200)
        self.main_lot.refresh_from_db()
        self.assertEqual(self.main_lot.quantity_on_hand, 6)
        self.assertFalse(
            UserAction.objects.filter(action='repair_lot_balance').exists()
        )


@override_settings(AXES_ENABLED=False)
class LotReassignmentLockCompatibilityTests(
    LotReassignmentFixtureMixin,
    TransactionTestCase,
):
    def test_session_no_key_lock_does_not_deadlock_product_then_fk_write(self):
        if connection.vendor != 'postgresql':
            self.skipTest('PostgreSQL row-lock compatibility test')

        product_locked = threading.Event()
        session_locked = threading.Event()
        errors = []

        def product_then_session_fk():
            close_old_connections()
            try:
                with transaction.atomic():
                    product = Product.objects.select_for_update().get(
                        pk=self.product.pk,
                    )
                    product_locked.set()
                    if not session_locked.wait(5):
                        raise AssertionError('Session lock was not acquired in time.')
                    StockChange.objects.create(
                        product=product,
                        session_id=self.session.pk,
                        change_type='checkin',
                        quantity=1,
                        note='Concurrent FK lock contract',
                    )
            except BaseException as exc:
                errors.append(exc)
            finally:
                close_old_connections()

        def session_then_product():
            close_old_connections()
            try:
                if not product_locked.wait(5):
                    raise AssertionError('Product lock was not acquired in time.')
                with transaction.atomic():
                    CheckinSession.objects.select_for_update(no_key=True).get(
                        pk=self.session.pk,
                    )
                    session_locked.set()
                    Product.objects.select_for_update().get(pk=self.product.pk)
            except BaseException as exc:
                errors.append(exc)
            finally:
                close_old_connections()

        first = threading.Thread(target=product_then_session_fk, daemon=True)
        second = threading.Thread(target=session_then_product, daemon=True)
        first.start()
        second.start()
        first.join(10)
        second.join(10)

        self.assertFalse(first.is_alive(), 'Product-first mutation did not finish.')
        self.assertFalse(second.is_alive(), 'Session-first mutation did not finish.')
        self.assertEqual(errors, [])
        self.assertEqual(
            StockChange.objects.filter(
                note='Concurrent FK lock contract',
            ).count(),
            1,
        )


@override_settings(AXES_ENABLED=False)
class LotReassignmentReportingTests(LotReassignmentFixtureMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.change = self.move(
            source=self.source_lot,
            quantity=2,
            destination_lot_id=self.destination_lot.pk,
            session=self.session,
        )['stock_change']
        UserAction.objects.create(
            user=self.admin,
            action='reassign_product_lot',
            target=self.product.name,
            detail=(
                f'Session #{self.session.pk}: moved 2 unit(s) from '
                f'{self.source_lot.destination_name} '
                f'({self.source_lot.expiry_date.isoformat()}) to '
                f'{self.destination_lot.destination_name} '
                f'({self.destination_lot.expiry_date.isoformat()}).'
            ),
        )
        self.client.force_login(self.admin)

    def test_session_detail_and_feed_are_neutral_and_immutable(self):
        detail_url = reverse(
            'checkin_session_detail',
            args=[self.session.pk],
        )
        response = self.client.get(detail_url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['net_totals'], {})
        self.assertContains(response, '2 moved')
        self.assertNotContains(
            response,
            reverse(
                'checkin_session_adjust',
                args=[self.session.pk, self.change.pk],
            ),
        )
        self.assertNotContains(
            response,
            reverse(
                'checkin_session_remove_line',
                args=[self.session.pk, self.change.pk],
            ),
        )

        before = self.change.quantity
        adjust = self.client.post(
            reverse(
                'checkin_session_adjust',
                args=[self.session.pk, self.change.pk],
            ),
            {'new_qty': 9},
        )
        remove = self.client.post(
            reverse(
                'checkin_session_remove_line',
                args=[self.session.pk, self.change.pk],
            ),
        )
        self.assertEqual((adjust.status_code, remove.status_code), (302, 302))
        self.change.refresh_from_db()
        self.assertEqual(self.change.quantity, before)
        self.assertEqual(self.change.session, self.session)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 10)

        context = CheckinProductView._session_history_context(self.session)
        self.assertEqual(context['session_history_action_count'], 1)
        self.assertEqual(context['session_history_net'], 0)

    def test_stock_log_and_activity_use_generic_moved_labels(self):
        stock_log = self.client.get(reverse('stock_log_api')).json()
        entry = next(
            item for item in stock_log['entries']
            if item['action'] == 'Moved'
        )
        self.assertTrue(entry['neutral'])
        self.assertFalse(entry['positive'])

        stock_activity = self.client.get(
            reverse('activity_log'),
            {'type': 'lot_reassignment'},
        )
        self.assertContains(stock_activity, 'Moved')
        self.assertContains(stock_activity, 'SUPPLIER-A')
        self.assertContains(stock_activity, 'SUPPLIER-B')

        action_activity = self.client.get(
            reverse('activity_log'),
            {'type': 'reassign_product_lot'},
        )
        self.assertContains(action_activity, 'Moved Stock Between Lots')
        self.assertContains(action_activity, f'Session #{self.session.pk}')
        view = ActivityLogView()
        self.assertEqual(
            view._filter_label('lot_reassignment'),
            'Moved Between Lots',
        )
        self.assertEqual(
            view._filter_label('reassign_product_lot'),
            'Moved Stock Between Lots',
        )

    def test_historical_unassigned_text_remains_visible_without_rewrite(self):
        legacy_change = StockChange.objects.create(
            product=self.product,
            session=self.session,
            user=self.admin,
            change_type='lot_reassignment',
            quantity=1,
            note='Moved 1 from UNASSIGNED to LEGACY-LOT',
        )
        legacy_action = UserAction.objects.create(
            user=self.admin,
            action='reassign_product_lot',
            target='UNASSIGNED stock review',
            detail='Moved from UNASSIGNED to LEGACY-LOT',
        )
        self.session.note = 'Legacy UNASSIGNED receiving session'
        self.session.save(update_fields=['note'])

        responses = (
            self.client.get(reverse('stock_log_api')),
            self.client.get(reverse('stock_log_api'), {'export': 'csv'}),
            self.client.get(
                reverse('checkin_session_detail', args=[self.session.pk]),
            ),
            self.client.get(
                reverse('activity_log'),
                {'type': 'lot_reassignment'},
            ),
            self.client.get(
                reverse('activity_log'),
                {'type': 'reassign_product_lot'},
            ),
        )
        for response in responses:
            with self.subTest(response=response.status_code):
                text = response.content.decode()
                self.assertIn(ProductLot.UNASSIGNED, text)

        legacy_change.refresh_from_db()
        legacy_action.refresh_from_db()
        self.session.refresh_from_db()
        self.assertIn(ProductLot.UNASSIGNED, legacy_change.note)
        self.assertIn(ProductLot.UNASSIGNED, legacy_action.detail)
        self.assertIn(ProductLot.UNASSIGNED, self.session.note)

    @patch('app.views.canvas.Canvas', RecordingCanvas)
    def test_checkin_pdf_reports_named_source_destination_and_neutral_quantity(self):
        response = self.client.get(
            reverse('checkin_session_pdf', args=[self.session.pk]),
        )

        self.assertEqual(response.status_code, 200)
        drawn = ' '.join(RecordingCanvas.strings)
        self.assertIn('Moved', drawn)
        self.assertIn('Moved 2', drawn)
        self.assertIn(
            f'Session #{self.session.pk}: SUPPLIER-A 2 -> SUPPLIER-B 2',
            drawn,
        )
        self.assertIn(
            f'expiry {self.destination_lot.expiry_date.isoformat()}',
            drawn,
        )

    @patch('app.views.canvas.Canvas', RecordingCanvas)
    def test_checkin_pdf_reports_unassigned_source_without_main_alias(self):
        self.move(
            source=self.main_lot,
            quantity=1,
            destination_lot_id=self.destination_lot.pk,
            session=self.session,
        )

        response = self.client.get(
            reverse('checkin_session_pdf', args=[self.session.pk]),
        )

        self.assertEqual(response.status_code, 200)
        drawn = ' '.join(RecordingCanvas.strings)
        self.assertIn(
            f'Session #{self.session.pk}: UNASSIGNED 1 -> SUPPLIER-B 1',
            drawn,
        )
        self.assertNotIn(
            f'Session #{self.session.pk}: MAIN 1 -> SUPPLIER-B 1',
            drawn,
        )

    @patch('app.views.canvas.Canvas', RecordingCanvas)
    def test_all_checkin_pdf_wraps_generic_move_details(self):
        self.session.ended_at = timezone.now()
        self.session.save(update_fields=['ended_at'])

        response = self.client.get(reverse('checkin_all_sessions_pdf'))

        self.assertEqual(response.status_code, 200)
        drawn = ' '.join(RecordingCanvas.strings)
        self.assertIn(
            f'Session #{self.session.pk}: SUPPLIER-A 2 -> SUPPLIER-B 2',
            drawn,
        )

    @patch('reportlab.pdfgen.canvas.Canvas', RecordingCanvas)
    def test_activity_pdf_uses_generic_move_filter_and_action(self):
        response = self.client.get(
            reverse('activity_log'),
            {'type': 'reassign_product_lot', 'export': 'pdf'},
        )

        self.assertEqual(response.status_code, 200)
        drawn = ' '.join(RecordingCanvas.strings)
        self.assertIn('Filter: Moved Stock Between Lots', drawn)
        self.assertIn('Moved Stock Between Lots', drawn)
        self.assertIn(f'Session #{self.session.pk}', drawn)

    def test_inventory_csv_uses_unassigned_lot_name(self):
        response = self.client.get(reverse('export_inventory_csv'))
        content = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertIn('UNASSIGNED:6', content)
