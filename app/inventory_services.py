"""Transactional helpers that keep Product stock and quantity-bearing lots aligned."""

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import F, Q, Sum

from .models import Product, ProductLot, ProductLotMovement


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
def remove_stock_from_lots(product, quantity, stock_change=None):
    """Remove stock FEFO (dated lots first, then undated) and record allocations."""
    quantity = int(quantity)
    if quantity <= 0:
        return []
    product = Product.objects.select_for_update().get(pk=product.pk)
    # The caller normally updates Product first. Balance against the old total.
    ensure_lot_balance(product, expected_total=product.quantity_in_stock + quantity)
    lots = list(
        ProductLot.objects.select_for_update()
        .filter(product=product, archived_at__isnull=True, quantity_on_hand__gt=0)
        .order_by(F('expiry_date').asc(nulls_last=True), 'received_at', 'pk')
    )
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
            lot_label = movement.lot_number or ProductLot.UNASSIGNED
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
    earliest = (
        ProductLot.objects.filter(
            product=product,
            archived_at__isnull=True,
            quantity_on_hand__gt=0,
            expiry_date__isnull=False,
        )
        .order_by('expiry_date')
        .values_list('expiry_date', flat=True)
        .first()
    )
    # Preserve a legacy expiry until a dated quantity-bearing lot is known.
    if earliest is not None and product.expiry_date != earliest:
        Product.objects.filter(pk=product.pk).update(expiry_date=earliest)
        product.expiry_date = earliest


def lot_balance_issue(product):
    tracked = lot_total(product)
    expected = int(product.quantity_in_stock or 0)
    if tracked == expected:
        return None
    return {'product_total': expected, 'lot_total': tracked, 'difference': expected - tracked}
