"""Transactional helpers that keep Product stock and quantity-bearing lots aligned."""

from datetime import date

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import F, Q, Sum

from .models import (
    Product,
    ProductExpiryDate,
    ProductLot,
    ProductLotMovement,
    StockChange,
    UserAction,
    display_lot_name,
    is_reserved_new_lot_name,
)


def _clean_lot_number(value):
    return (value or ProductLot.UNASSIGNED).strip().upper() or ProductLot.UNASSIGNED


def lot_total(product):
    return (
        ProductLot.objects.filter(product=product, archived_at__isnull=True)
        .aggregate(total=Sum('quantity_on_hand'))['total']
        or 0
    )


def get_or_create_lot(product, lot_number=None, expiry_date=None, session=None):
    """Return one exact lot, avoiding nullable-unique backend differences."""
    lot_number = _clean_lot_number(lot_number)
    lot = (
        ProductLot.objects.select_for_update()
        .filter(
            product=product,
            lot_number=lot_number,
            expiry_date=expiry_date,
        )
        .first()
    )
    if lot:
        update_fields = []
        if lot.archived_at is not None:
            lot.archived_at = None
            lot.archived_by = None
            update_fields.extend(['archived_at', 'archived_by'])
        if session and not lot.checkin_session_id:
            lot.checkin_session = session
            update_fields.append('checkin_session')
        if update_fields:
            lot.save(update_fields=update_fields + ['updated_at'])
        return lot
    return ProductLot.objects.create(
        product=product,
        lot_number=lot_number,
        expiry_date=expiry_date,
        checkin_session=session,
    )


def ensure_lot_balance(product, expected_total=None):
    """Put legacy/unallocated stock in UNASSIGNED without guessing a real lot."""
    expected = product.quantity_in_stock if expected_total is None else expected_total
    current = lot_total(product)
    difference = int(expected) - int(current)
    if difference == 0:
        return
    unassigned = get_or_create_lot(product)
    if unassigned.quantity_on_hand + difference < 0:
        raise ValidationError(
            f'Lot quantities for {product.name} exceed the product total by '
            f'{abs(difference)}. Reconcile the lots before continuing.'
        )
    unassigned.quantity_on_hand += difference
    unassigned.save(update_fields=['quantity_on_hand', 'updated_at'])


def _record_movement(stock_change, lot, quantity, direction):
    if not stock_change or quantity <= 0:
        return None
    return ProductLotMovement.objects.create(
        stock_change=stock_change,
        lot=lot,
        lot_number=lot.lot_number,
        expiry_date=lot.expiry_date,
        quantity=quantity,
        direction=direction,
    )


@transaction.atomic
def add_stock_to_lot(
    product, quantity, stock_change=None, lot_number=None, expiry_date=None,
    session=None,
):
    quantity = int(quantity)
    if quantity <= 0:
        return None
    product = Product.objects.select_for_update().get(pk=product.pk)
    # The caller normally updates Product first. Balance against the old total.
    ensure_lot_balance(product, expected_total=product.quantity_in_stock - quantity)
    lot = get_or_create_lot(product, lot_number, expiry_date, session)
    lot.quantity_on_hand = F('quantity_on_hand') + quantity
    lot.save(update_fields=['quantity_on_hand', 'updated_at'])
    lot.refresh_from_db(fields=['quantity_on_hand'])
    _record_movement(stock_change, lot, quantity, ProductLotMovement.DIRECTION_IN)
    _refresh_product_expiry(product)
    return lot


@transaction.atomic
def remove_stock_from_lots(product, quantity, stock_change=None, as_of_date=None):
    """Remove stock from lots and record the exact allocations.

    Normal inventory corrections retain strict FEFO ordering. A sale can pass
    its purchase date to consume the nearest valid expiry first, then undated
    stock, and finally the nearest expired lot only when an expired-sale
    override has allowed that inventory to be sold.
    """
    quantity = int(quantity)
    if quantity <= 0:
        return []
    product = Product.objects.select_for_update().get(pk=product.pk)
    # The caller normally updates Product first. Balance against the old total.
    ensure_lot_balance(product, expected_total=product.quantity_in_stock + quantity)
    lot_query = ProductLot.objects.select_for_update().filter(
        product=product, archived_at__isnull=True, quantity_on_hand__gt=0,
    )
    if as_of_date is None:
        lots = list(
            lot_query.order_by(
                F('expiry_date').asc(nulls_last=True), 'received_at', 'pk',
            )
        )
    else:
        if hasattr(as_of_date, 'date'):
            as_of_date = as_of_date.date()
        # Lock in primary-key order for deterministic concurrent checkouts,
        # then apply purchase-date priority in memory.
        lots = list(lot_query.order_by('pk'))

        def purchase_expiry_priority(lot):
            if lot.expiry_date is None:
                return (1, 0, lot.received_at, lot.pk)
            if lot.expiry_date >= as_of_date:
                return (0, (lot.expiry_date - as_of_date).days, lot.received_at, lot.pk)
            return (2, (as_of_date - lot.expiry_date).days, lot.received_at, lot.pk)

        lots.sort(key=purchase_expiry_priority)
    remaining = quantity
    allocations = []
    for lot in lots:
        take = min(remaining, lot.quantity_on_hand)
        if take <= 0:
            continue
        lot.quantity_on_hand -= take
        lot.save(update_fields=['quantity_on_hand', 'updated_at'])
        _record_movement(stock_change, lot, take, ProductLotMovement.DIRECTION_OUT)
        allocations.append((lot, take))
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValidationError(
            f'Only {quantity - remaining} lot-tracked units were available for {product.name}.'
        )
    _refresh_product_expiry(product)
    return allocations


def _positive_whole_number(value, message):
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise ValidationError(message)
    if normalized <= 0:
        raise ValidationError(message)
    return normalized


def _lot_identity(lot_number, expiry_date):
    return (_clean_lot_number(lot_number), expiry_date)


def _lock_active_checkin_session(session):
    """Lock and revalidate an optional Check-in session before stock rows."""
    if session is None:
        return None
    session_pk = getattr(session, 'pk', None)
    if session_pk is None:
        raise ValidationError('This Check-in session is no longer available.')
    session_model = type(session)
    try:
        locked_session = session_model.objects.select_for_update(
            no_key=True,
        ).get(pk=session_pk)
    except session_model.DoesNotExist:
        raise ValidationError('This Check-in session is no longer available.')
    if not locked_session.is_active:
        raise ValidationError(
            'This Check-in session ended while the form was open. Stock was not moved.'
        )
    if locked_session.inventory_mode:
        raise ValidationError(
            'Finish or reconcile the inventory count before moving stock between lots.'
        )
    return locked_session


def _lock_product_and_active_lots(product):
    product_pk = getattr(product, 'pk', product)
    try:
        locked_product = Product.all_objects.select_for_update().get(pk=product_pk)
    except (Product.DoesNotExist, TypeError, ValueError):
        raise ValidationError('This product is no longer available.')
    if locked_product.archived_at is not None:
        raise ValidationError(
            'This product moved to Recovery while the form was open. Stock was not moved.'
        )
    locked_lots = list(
        ProductLot.objects.select_for_update()
        .filter(product=locked_product, archived_at__isnull=True)
        .order_by('pk')
    )
    product_total = int(locked_product.quantity_in_stock or 0)
    lot_total_before = sum(int(lot.quantity_on_hand or 0) for lot in locked_lots)
    return locked_product, locked_lots, product_total, lot_total_before


def _validate_lot_balance(product, product_total, lot_total_before):
    if lot_total_before != product_total:
        raise ValidationError(
            f'Lot quantities total {lot_total_before}, but {product.name} has '
            f'{product_total} in stock. Run Inventory Audit before moving stock '
            'between lots.'
        )


def _create_new_destination_lot(
    product,
    active_lots,
    *,
    lot_number,
    expiry_date,
    session,
    allow_internal_main=False,
):
    """Create one unused identity without reviving an archived lot."""
    typed_name = _clean_lot_number(lot_number)
    lot_number_limit = ProductLot._meta.get_field('lot_number').max_length
    if len(typed_name) > lot_number_limit:
        raise ValidationError(
            f'Lot numbers must be {lot_number_limit} characters or fewer.'
        )
    if not allow_internal_main and is_reserved_new_lot_name(typed_name):
        raise ValidationError(
            'UNASSIGNED is reserved for stock not connected to a supplier lot. '
            'Choose UNASSIGNED as the destination, or enter the supplier lot number.'
        )
    identity = (typed_name, expiry_date)
    existing = next(
        (
            lot for lot in active_lots
            if _lot_identity(lot.lot_number, lot.expiry_date) == identity
        ),
        None,
    )
    if existing is not None:
        return existing, False
    archived = (
        ProductLot.objects.select_for_update()
        .filter(
            product=product,
            lot_number=typed_name,
            expiry_date=expiry_date,
            archived_at__isnull=False,
        )
        .first()
    )
    if archived is not None:
        raise ValidationError(
            f'{display_lot_name(typed_name)} with '
            'that expiry is archived. Restore it through Inventory Audit before '
            'using that exact lot again.'
        )
    destination = ProductLot.objects.create(
        product=product,
        lot_number=typed_name,
        expiry_date=expiry_date,
        checkin_session=session,
    )
    active_lots.append(destination)
    return destination, True


@transaction.atomic
def reassign_lot_stock(
    product,
    *,
    source_lot_id,
    expected_source_balance,
    quantity,
    destination_kind,
    destination_lot_id=None,
    destination_lot_number=None,
    destination_expiry_date=None,
    confirm_past_expiry=False,
    confirm_expiry_reclassification=False,
    user=None,
    session=None,
):
    """Move stock from one active lot to one destination without changing totals."""
    session = _lock_active_checkin_session(session)
    product, active_lots, product_total, lot_total_before = (
        _lock_product_and_active_lots(product)
    )
    _validate_lot_balance(product, product_total, lot_total_before)
    active_by_id = {lot.pk: lot for lot in active_lots}

    try:
        source_id = int(source_lot_id)
        expected_balance = int(expected_source_balance)
    except (TypeError, ValueError):
        raise ValidationError('Choose a valid source lot and review its quantity.')
    move_quantity = _positive_whole_number(
        quantity,
        'Enter a whole-number quantity greater than zero.',
    )
    source = active_by_id.get(source_id)
    if source is None or source.archived_at is not None:
        raise ValidationError(
            'The selected source lot is no longer active for this product.'
        )
    source_balance = int(source.quantity_on_hand or 0)
    if source_balance != expected_balance:
        raise ValidationError(
            f'{source.destination_name} stock changed while this form was open. '
            'Review the current quantity and try again.'
        )
    if source_balance <= 0:
        raise ValidationError('Choose a source lot that has stock available.')
    if move_quantity > source_balance:
        raise ValidationError(
            f'Only {source_balance} unit(s) remain in {source.destination_name}.'
        )

    normalized_kind = str(destination_kind or '').strip().lower()
    if normalized_kind not in {'existing', 'new', 'main'}:
        raise ValidationError('Choose an existing lot, a new lot, or UNASSIGNED.')

    destination_created = False
    if normalized_kind == 'existing':
        try:
            destination_id = int(destination_lot_id)
        except (TypeError, ValueError):
            raise ValidationError('Choose a valid destination lot.')
        destination = active_by_id.get(destination_id)
        if destination is None or destination.archived_at is not None:
            raise ValidationError(
                'The selected destination lot is no longer active for this product.'
            )
    elif normalized_kind == 'new':
        typed_name = str(destination_lot_number or '').strip().upper()
        if not typed_name:
            raise ValidationError('Enter a name for the new destination lot.')
        destination, destination_created = _create_new_destination_lot(
            product,
            active_lots,
            lot_number=typed_name,
            expiry_date=destination_expiry_date,
            session=session,
        )
        if not destination_created:
            raise ValidationError(
                'That destination lot already exists. Select it from the saved '
                'Product Lots table instead.'
            )
    else:
        # A synthetic UNASSIGNED destination preserves the source expiry so moving
        # stock never silently changes whether it is expired.
        destination, destination_created = _create_new_destination_lot(
            product,
            active_lots,
            lot_number=ProductLot.UNASSIGNED,
            expiry_date=source.expiry_date,
            session=session,
            allow_internal_main=True,
        )

    if destination.pk == source.pk:
        raise ValidationError('Choose a destination different from the source lot.')
    today = date.today()
    if (
        destination.expiry_date is not None
        and destination.expiry_date < today
        and not confirm_past_expiry
    ):
        raise ValidationError(
            'This destination lot is already expired. Confirm that you want '
            'these units to appear in Expired Products.'
        )
    if (
        source.expiry_date is not None
        and source.expiry_date < today
        and (
            destination.expiry_date is None
            or destination.expiry_date >= today
        )
        and not confirm_expiry_reclassification
    ):
        raise ValidationError(
            'These units currently belong to an expired lot. Confirm that moving '
            'them to an undated or future-dated lot is accurate.'
        )

    source_before = source_balance
    destination_before = int(destination.quantity_on_hand or 0)
    source.quantity_on_hand = source_before - move_quantity
    source.save(update_fields=['quantity_on_hand', 'updated_at'])
    destination.quantity_on_hand = destination_before + move_quantity
    destination.save(update_fields=['quantity_on_hand', 'updated_at'])

    source_label = display_lot_name(source.lot_number)
    destination_label = display_lot_name(destination.lot_number)
    source_expiry = source.expiry_date.isoformat() if source.expiry_date else 'no expiry'
    destination_expiry = (
        destination.expiry_date.isoformat() if destination.expiry_date else 'no expiry'
    )
    stock_change = StockChange.objects.create(
        product=product,
        product_name=product.name,
        product_barcode=product.barcode or '',
        change_type='lot_reassignment',
        quantity=move_quantity,
        note=(
            f'Moved {move_quantity} unit(s) from {source_label}; {source_expiry} '
            f'to {destination_label}; {destination_expiry}. Product stock unchanged.'
        ),
        user=user,
        session=session,
    )
    _record_movement(
        stock_change,
        source,
        move_quantity,
        ProductLotMovement.DIRECTION_OUT,
    )
    _record_movement(
        stock_change,
        destination,
        move_quantity,
        ProductLotMovement.DIRECTION_IN,
    )
    _refresh_product_expiry(product)

    tracked_after = sum(int(lot.quantity_on_hand or 0) for lot in active_lots)
    if tracked_after != product_total:
        raise ValidationError(
            'The lot movement did not preserve the product stock total. Nothing '
            'was changed.'
        )
    return {
        'product': product,
        'source': source,
        'destination': destination,
        'destination_created': destination_created,
        'source_before': source_before,
        'destination_before': destination_before,
        'quantity': move_quantity,
        'stock_change': stock_change,
    }


@transaction.atomic
def repair_lot_balance_to_main(product, *, user=None, session=None):
    """Assign only a positive untracked residual to an undated UNASSIGNED bucket."""
    session = _lock_active_checkin_session(session)
    product, active_lots, product_total, lot_total_before = (
        _lock_product_and_active_lots(product)
    )
    residual = product_total - lot_total_before
    if residual < 0:
        raise ValidationError(
            f'Lot quantities exceed {product.name} stock by {abs(residual)}. '
            'Nothing was changed; review this product in Inventory Audit.'
        )
    if residual == 0:
        return {
            'product': product,
            'destination': None,
            'quantity': 0,
            'product_total': product_total,
            'lot_total_before': lot_total_before,
            'action': None,
        }

    destination, _created = _create_new_destination_lot(
        product,
        active_lots,
        lot_number=ProductLot.UNASSIGNED,
        expiry_date=None,
        session=session,
        allow_internal_main=True,
    )
    destination.quantity_on_hand = int(destination.quantity_on_hand or 0) + residual
    destination.save(update_fields=['quantity_on_hand', 'updated_at'])
    session_detail = f'Session #{session.pk}: ' if session is not None else ''
    action = UserAction.objects.create(
        user=user,
        action='repair_lot_balance',
        target=product.name,
        detail=(
            f'{session_detail}assigned {residual} previously untracked unit(s) '
            f'to UNASSIGNED. Product stock {product_total}; tracked lots before repair '
            f'{lot_total_before}; tracked lots after repair {product_total}.'
        ),
    )
    _refresh_product_expiry(product)
    return {
        'product': product,
        'destination': destination,
        'quantity': residual,
        'product_total': product_total,
        'lot_total_before': lot_total_before,
        'action': action,
    }


@transaction.atomic
def remove_stock_from_lot(product, lot, quantity, stock_change=None):
    """Remove stock from one explicitly selected lot and record that allocation."""
    quantity = int(quantity)
    if quantity <= 0:
        return []
    product = Product.objects.select_for_update().get(pk=product.pk)
    # The caller normally updates Product first. Balance against the old total
    # before reducing only the lot the staff member selected.
    ensure_lot_balance(product, expected_total=product.quantity_in_stock + quantity)
    lot_id = getattr(lot, 'pk', lot)
    selected_lot = (
        ProductLot.objects.select_for_update()
        .filter(
            pk=lot_id,
            product=product,
            archived_at__isnull=True,
            quantity_on_hand__gt=0,
        )
        .first()
    )
    if selected_lot is None:
        raise ValidationError('The selected lot is no longer available.')
    if selected_lot.quantity_on_hand < quantity:
        raise ValidationError(
            f'Only {selected_lot.quantity_on_hand} unit(s) remain in lot '
            f'{selected_lot.destination_name}.'
        )
    selected_lot.quantity_on_hand -= quantity
    selected_lot.save(update_fields=['quantity_on_hand', 'updated_at'])
    _record_movement(
        stock_change, selected_lot, quantity, ProductLotMovement.DIRECTION_OUT,
    )
    _refresh_product_expiry(product)
    return [(selected_lot, quantity)]


@transaction.atomic
def remove_stock_from_recorded_lots(
    product, quantity, source_stock_changes, stock_change=None,
):
    """Reverse stock previously added by known ledger rows.

    Unlike a normal FEFO removal, an undo must take units back out of the exact
    lots populated by the correction. If those units have since been consumed,
    fail safely instead of silently removing unrelated inventory.
    """
    quantity = int(quantity)
    if quantity <= 0:
        return []
    product = Product.all_objects.select_for_update().get(pk=product.pk)
    # The caller has already decremented Product; align lots against the total
    # that existed immediately before the undo.
    ensure_lot_balance(product, expected_total=product.quantity_in_stock + quantity)
    movements = list(
        ProductLotMovement.objects.filter(
            stock_change__in=source_stock_changes,
            direction=ProductLotMovement.DIRECTION_IN,
        ).order_by('created_at', 'pk')
    )
    remaining = quantity
    removed = []
    for movement in movements:
        if remaining <= 0:
            break
        lot = None
        if movement.lot_id:
            lot = (
                ProductLot.objects.select_for_update()
                .filter(pk=movement.lot_id, product=product)
                .first()
            )
        if lot is None:
            lot = (
                ProductLot.objects.select_for_update()
                .filter(
                    product=product,
                    lot_number=movement.lot_number,
                    expiry_date=movement.expiry_date,
                )
                .first()
            )
        take = min(remaining, movement.quantity)
        if lot is None or lot.quantity_on_hand < take:
            lot_label = display_lot_name(
                movement.lot_number or ProductLot.UNASSIGNED,
            )
            raise ValidationError(
                f'Undo unavailable: {product.name} no longer has the '
                f'{take} returned unit(s) in lot {lot_label}. Use an inventory '
                'correction if those units have already been used.'
            )
        lot.quantity_on_hand -= take
        lot.save(update_fields=['quantity_on_hand', 'updated_at'])
        _record_movement(
            stock_change, lot, take, ProductLotMovement.DIRECTION_OUT,
        )
        removed.append((lot, take))
        remaining -= take
    if remaining:
        raise ValidationError(
            f'Undo unavailable: the void history for {product.name} only '
            f'identifies {quantity - remaining} of {quantity} returned unit(s).'
        )
    _refresh_product_expiry(product)
    return removed


@transaction.atomic
def restore_stock_to_original_lots(product, quantity, source_stock_changes, stock_change=None):
    """Restock a return into the original depleted lots where that history exists."""
    quantity = int(quantity)
    if quantity <= 0:
        return []
    product = Product.objects.select_for_update().get(pk=product.pk)
    # Caller has already incremented Product; align against its pre-return total.
    ensure_lot_balance(product, expected_total=product.quantity_in_stock - quantity)
    remaining = quantity
    restored = []
    source_rows = list(
        source_stock_changes.values('pk', 'order_detail_id', 'checkout_item_id')
    )
    source_ids = [row['pk'] for row in source_rows]
    order_detail_ids = [row['order_detail_id'] for row in source_rows if row['order_detail_id']]
    checkout_item_ids = [row['checkout_item_id'] for row in source_rows if row['checkout_item_id']]
    returned_source = Q(pk__in=[])
    if order_detail_ids:
        returned_source |= Q(
            stock_change__correction_line__order_detail_id__in=order_detail_ids,
        )
    if checkout_item_ids:
        returned_source |= Q(
            stock_change__correction_line__checkout_item_id__in=checkout_item_ids,
        )
    movements = (
        ProductLotMovement.objects.select_related('lot')
        .filter(
            stock_change_id__in=source_ids,
            direction=ProductLotMovement.DIRECTION_OUT,
        )
        .order_by('created_at', 'pk')
    )
    for movement in movements:
        if remaining <= 0:
            break
        lot = movement.lot
        if not lot or lot.product_id != product.pk:
            lot = get_or_create_lot(product, movement.lot_number, movement.expiry_date)
        returned_already = (
            ProductLotMovement.objects.filter(
                returned_source,
                stock_change__correction_line__isnull=False,
                stock_change__correction_line__correction__undo__isnull=True,
                lot_number=movement.lot_number,
                expiry_date=movement.expiry_date,
                direction=ProductLotMovement.DIRECTION_IN,
            ).aggregate(total=Sum('quantity'))['total']
            or 0
        )
        capacity = max(0, movement.quantity - returned_already)
        put_back = min(remaining, capacity)
        if put_back <= 0:
            continue
        lot.quantity_on_hand = F('quantity_on_hand') + put_back
        lot.save(update_fields=['quantity_on_hand', 'updated_at'])
        lot.refresh_from_db(fields=['quantity_on_hand'])
        _record_movement(stock_change, lot, put_back, ProductLotMovement.DIRECTION_IN)
        restored.append((lot, put_back))
        remaining -= put_back
    if remaining:
        lot = get_or_create_lot(product)
        lot.quantity_on_hand = F('quantity_on_hand') + remaining
        lot.save(update_fields=['quantity_on_hand', 'updated_at'])
        lot.refresh_from_db(fields=['quantity_on_hand'])
        _record_movement(stock_change, lot, remaining, ProductLotMovement.DIRECTION_IN)
        restored.append((lot, remaining))
    _refresh_product_expiry(product)
    return restored


def _refresh_product_expiry(product):
    dated = list(
        ProductLot.objects.filter(
            product=product,
            archived_at__isnull=True,
            quantity_on_hand__gt=0,
            expiry_date__isnull=False,
        )
        .order_by('expiry_date')
        .values_list('expiry_date', flat=True)
        .distinct()
    )
    if dated:
        # ProductExpiryDate is retained for legacy screens/exports, but it must
        # mirror quantity-bearing lots. A depleted lot must never keep a product
        # blocked as expired after FEFO has moved on to a newer lot.
        ProductExpiryDate.objects.filter(product=product).delete()
        ProductExpiryDate.objects.bulk_create([
            ProductExpiryDate(product=product, expiry_date=value)
            for value in dated
        ])
        earliest = dated[0]
    else:
        has_dated_lot_history = ProductLot.objects.filter(
            product=product,
            expiry_date__isnull=False,
        ).exists()
        if not has_dated_lot_history:
            # Products that have never adopted lot tracking still use their
            # legacy product-level expiry until staff assigns a dated lot.
            return
        ProductExpiryDate.objects.filter(product=product).delete()
        earliest = None

    if product.expiry_date != earliest:
        Product.objects.filter(pk=product.pk).update(expiry_date=earliest)
        product.expiry_date = earliest


def lot_balance_issue(product):
    tracked = lot_total(product)
    expected = int(product.quantity_in_stock or 0)
    if tracked == expected:
        return None
    return {'product_total': expected, 'lot_total': tracked, 'difference': expected - tracked}
