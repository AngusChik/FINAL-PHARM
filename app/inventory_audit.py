"""Reusable inventory-integrity audit used by the UI and management command."""

from django.db import transaction
from django.db.models import Count, F, IntegerField, Q, Sum, Value
from django.db.models.functions import Coalesce
from django.urls import reverse
from django.utils import timezone

from .inventory_services import get_or_create_lot
from .models import (
    InventoryAuditIssue,
    InventoryAuditRun,
    Product,
    ProductExpiryDate,
    ProductLot,
    SupplierPurchaseOrderLine,
    display_lot_text,
    normalize_barcode_key,
)


ZERO_STOCK_EXPIRY_ISSUE = 'zero_stock_current_expiry'
CLEAR_ZERO_STOCK_EXPIRY_ACTION = 'clear_zero_stock_expiry'


def _expiry_rows_snapshot(rows):
    return [
        {'id': row.pk, 'date': row.expiry_date.isoformat()}
        for row in sorted(rows, key=lambda row: (row.expiry_date, row.pk))
    ]


def _zero_stock_expiry_snapshot(product, *, active_lot_total, expiry_rows):
    return {
        'quantity_in_stock': int(product.quantity_in_stock or 0),
        'active_lot_total': int(active_lot_total or 0),
        'product_expiry_date': (
            product.expiry_date.isoformat() if product.expiry_date else ''
        ),
        'expiry_rows': _expiry_rows_snapshot(expiry_rows),
    }


def _expiry_snapshot_label(snapshot):
    labels = []
    if snapshot.get('product_expiry_date'):
        labels.append(f'Product: {snapshot["product_expiry_date"]}')
    row_dates = [row.get('date', '') for row in snapshot.get('expiry_rows', [])]
    if row_dates:
        labels.append(f'Recorded: {", ".join(row_dates)}')
    label = '; '.join(labels) or 'No current expiry'
    return label if len(label) <= 100 else label[:97] + '...'


def _clear_selected_zero_stock_expiries(run, issue_ids, source_run_id):
    """Clear selected current-expiry fields only when the audit snapshot is current.

    ProductLot rows are durable history and are intentionally retained. The
    complete selection is validated while its products, lots, and expiry rows
    are locked; if any selection changed, none of the selected records changes.
    """
    requested_ids = sorted(set(int(issue_id) for issue_id in issue_ids))
    if not requested_ids:
        return {'cleared': [], 'stale': []}

    with transaction.atomic():
        sources = list(
            InventoryAuditIssue.objects.select_for_update()
            .filter(pk__in=requested_ids)
        )
        source_by_id = {source.pk: source for source in sources}
        validations = []
        stale = []
        locked_products = {}
        selected_product_ids = set()

        for issue_id in requested_ids:
            source = source_by_id.get(issue_id)
            if source is None:
                stale.append({
                    'source_issue_id': issue_id,
                    'product': None,
                    'reason': 'The selected audit finding no longer exists.',
                })
                continue
            action = (source.metadata or {}).get('action', '')
            if (
                source.code != ZERO_STOCK_EXPIRY_ISSUE
                or action != CLEAR_ZERO_STOCK_EXPIRY_ACTION
                or source.repaired
                or source.product_id is None
                or source.run_id != source_run_id
            ):
                stale.append({
                    'source_issue_id': issue_id,
                    'source_run_id': source.run_id,
                    'product': source.product,
                    'reason': 'The selected finding is not an open zero-stock expiry review.',
                })
                continue
            if source.product_id in selected_product_ids:
                stale.append({
                    'source_issue_id': issue_id,
                    'source_run_id': source.run_id,
                    'product': source.product,
                    'reason': 'The same product was selected more than once.',
                })
                continue
            selected_product_ids.add(source.product_id)

            product = locked_products.get(source.product_id)
            if product is None:
                product = Product.all_objects.select_for_update().get(
                    pk=source.product_id,
                )
                locked_products[source.product_id] = product
            active_lots = list(
                ProductLot.objects.select_for_update()
                .filter(product=product, archived_at__isnull=True)
                .order_by('pk')
            )
            expiry_rows = list(
                ProductExpiryDate.objects.select_for_update()
                .filter(product=product)
                .order_by('expiry_date', 'pk')
            )
            current_snapshot = _zero_stock_expiry_snapshot(
                product,
                active_lot_total=sum(lot.quantity_on_hand for lot in active_lots),
                expiry_rows=expiry_rows,
            )
            source_snapshot = (source.metadata or {}).get('snapshot') or {}
            reason = ''
            if product.archived_at is not None:
                reason = 'The product moved to Recovery after this audit.'
            elif current_snapshot['quantity_in_stock'] != 0:
                reason = 'The product now has stock.'
            elif current_snapshot['active_lot_total'] != 0:
                reason = 'The product now has quantity-bearing lot stock.'
            elif not (
                current_snapshot['product_expiry_date']
                or current_snapshot['expiry_rows']
            ):
                reason = 'The current expiry was already cleared or changed.'
            elif current_snapshot != source_snapshot:
                reason = 'The stock or expiry information changed after this audit.'

            if reason:
                stale.append({
                    'source_issue_id': issue_id,
                    'source_run_id': source.run_id,
                    'product': product,
                    'reason': reason,
                    'snapshot': current_snapshot,
                })
                continue
            validations.append({
                'source': source,
                'product': product,
                'expiry_rows': expiry_rows,
                'snapshot': current_snapshot,
            })

        if stale:
            # The selection is one reviewed action. Do not partly apply it when
            # even one row changed after the audit was displayed.
            stale_product_ids = {
                item['product'].pk for item in stale if item.get('product')
            }
            for item in validations:
                if item['product'].pk not in stale_product_ids:
                    stale.append({
                        'source_issue_id': item['source'].pk,
                        'source_run_id': item['source'].run_id,
                        'product': item['product'],
                        'reason': (
                            'Nothing was cleared because another selected '
                            'product changed after review.'
                        ),
                        'snapshot': item['snapshot'],
                    })
            return {'cleared': [], 'stale': stale}

        cleared = []
        resolved_at = timezone.now()
        for item in validations:
            product = item['product']
            ProductExpiryDate.objects.filter(product=product).delete()
            product.expiry_date = None
            product.save(update_fields=['expiry_date', 'updated_at'])
            source = item['source']
            source_metadata = dict(source.metadata or {})
            source_metadata.update({
                'result': 'cleared',
                'resolved_by_run_id': run.pk,
                'resolved_by_user_id': run.created_by_id,
                'resolved_at': resolved_at.isoformat(),
            })
            source.repaired = True
            source.metadata = source_metadata
            source.save(update_fields=['repaired', 'metadata'])
            cleared.append({
                'source_issue_id': source.pk,
                'source_run_id': source.run_id,
                'product': product,
                'snapshot': item['snapshot'],
                'resolved_by_run_id': run.pk,
            })
        return {'cleared': cleared, 'stale': []}


def _issue(
    issues,
    *,
    code,
    title,
    detail,
    product=None,
    expected='',
    actual='',
    repairable=False,
    repaired=False,
    metadata=None,
):
    issues.append(InventoryAuditIssue(
        code=code,
        title=title,
        detail=detail,
        product=product,
        product_name=product.name if product else '',
        expected_value=str(expected),
        actual_value=str(actual),
        repairable=repairable,
        repaired=repaired,
        metadata=metadata or {},
    ))


@transaction.atomic
def _run_inventory_audit_transaction(
    run, *, repair_unassigned=False,
    clear_zero_stock_expiry_issue_ids=None,
    clear_zero_stock_expiry_run_id=None,
):
    """Complete one audit and all requested repairs in one transaction.

    The optional repair only adds a positive missing balance to the product's
    UNASSIGNED lot. It never changes the product total or reduces a named lot.
    """
    issues = []
    check_counts = {
        'barcode_uniqueness': 0,
        'barcode_normalization': 0,
        'lot_balances': 0,
        'zero_stock_expiry': 0,
        'negative_values': 0,
        'supplier_receiving': 0,
    }
    repaired_count = 0

    try:
        expiry_action = _clear_selected_zero_stock_expiries(
            run,
            clear_zero_stock_expiry_issue_ids or [],
            clear_zero_stock_expiry_run_id,
        )
        stale_expiry_by_product = {
            item['product'].pk: item
            for item in expiry_action['stale']
            if item.get('product')
        }
        for item in expiry_action['cleared']:
            check_counts['zero_stock_expiry'] += 1
            repaired_count += 1
            snapshot = item['snapshot']
            _issue(
                issues,
                code=ZERO_STOCK_EXPIRY_ISSUE,
                title='Zero-stock current expiry cleared',
                detail=(
                    'The product-level current expiry was cleared after stock '
                    'and lot balances were rechecked. Depleted lot history was retained.'
                ),
                product=item['product'],
                expected='No current expiry while stock is zero',
                actual=_expiry_snapshot_label(snapshot),
                repairable=True,
                repaired=True,
                metadata={
                    'action': CLEAR_ZERO_STOCK_EXPIRY_ACTION,
                    'result': 'cleared',
                    'source_issue_id': item['source_issue_id'],
                    'source_run_id': item['source_run_id'],
                    'resolved_by_run_id': item['resolved_by_run_id'],
                    'snapshot': snapshot,
                },
            )

        duplicate_keys = list(
            Product.all_objects.exclude(normalized_barcode__isnull=True)
            .values('normalized_barcode')
            .annotate(count=Count('pk'))
            .filter(count__gt=1)
        )
        for row in duplicate_keys:
            products = Product.all_objects.filter(
                normalized_barcode=row['normalized_barcode'],
            ).order_by('pk')
            for product in products:
                check_counts['barcode_uniqueness'] += 1
                _issue(
                    issues,
                    code='duplicate_normalized_barcode',
                    title='Duplicate barcode identity',
                    detail=(
                        f'{row["count"]} products share normalized barcode '
                        f'{row["normalized_barcode"]}.'
                    ),
                    product=product,
                    expected='Unique barcode',
                    actual=row['normalized_barcode'],
                )

        products = list(
            Product.all_objects.annotate(
                active_lot_total=Coalesce(
                    Sum(
                        'lots__quantity_on_hand',
                        filter=Q(lots__archived_at__isnull=True),
                    ),
                    Value(0),
                    output_field=IntegerField(),
                ),
            ).prefetch_related('expiry_dates').order_by('pk')
        )
        for product in products:
            expected_key = normalize_barcode_key(product.barcode)
            if product.normalized_barcode != expected_key:
                check_counts['barcode_normalization'] += 1
                _issue(
                    issues,
                    code='barcode_key_mismatch',
                    title='Barcode normalization mismatch',
                    detail='The stored barcode search key does not match the product barcode.',
                    product=product,
                    expected=expected_key or 'blank',
                    actual=product.normalized_barcode or 'blank',
                )

            expected_stock = int(product.quantity_in_stock or 0)
            tracked_stock = int(product.active_lot_total or 0)
            if expected_stock != tracked_stock:
                difference = expected_stock - tracked_stock
                repaired = False
                repairable = difference > 0
                if repair_unassigned and repairable:
                    with transaction.atomic():
                        locked = Product.all_objects.select_for_update().get(pk=product.pk)
                        active_total = (
                            ProductLot.objects.filter(
                                product=locked, archived_at__isnull=True,
                            ).aggregate(total=Sum('quantity_on_hand'))['total'] or 0
                        )
                        live_difference = int(locked.quantity_in_stock or 0) - int(active_total)
                        if live_difference > 0:
                            lot = get_or_create_lot(locked)
                            ProductLot.objects.filter(pk=lot.pk).update(
                                quantity_on_hand=F('quantity_on_hand') + live_difference,
                                updated_at=timezone.now(),
                            )
                            difference = live_difference
                            tracked_stock = int(active_total)
                            repaired = True
                            repaired_count += 1
                check_counts['lot_balances'] += 1
                _issue(
                    issues,
                    code='lot_total_mismatch',
                    title='Product and lot totals differ',
                    detail=(
                        f'Product stock is {expected_stock}, while active lots total '
                        f'{tracked_stock}. Missing positive stock can be assigned to '
                        'UNASSIGNED stock without changing inventory.'
                    ),
                    product=product,
                    expected=expected_stock,
                    actual=tracked_stock,
                    repairable=repairable,
                    repaired=repaired,
                    metadata={'difference': difference},
                )

            expiry_rows = list(product.expiry_dates.all())
            if (
                product.archived_at is None
                and expected_stock == 0
                and (product.expiry_date is not None or expiry_rows)
            ):
                check_counts['zero_stock_expiry'] += 1
                snapshot = _zero_stock_expiry_snapshot(
                    product,
                    active_lot_total=tracked_stock,
                    expiry_rows=expiry_rows,
                )
                stale_review = stale_expiry_by_product.pop(product.pk, None)
                safe_to_review = tracked_stock == 0
                if safe_to_review:
                    detail = (
                        'This product has no stock or quantity-bearing active lots, '
                        'but it still presents a current expiry date.'
                    )
                    metadata = {
                        'action': CLEAR_ZERO_STOCK_EXPIRY_ACTION,
                        'result': 'review_required',
                        'snapshot': snapshot,
                    }
                else:
                    detail = (
                        'This product is recorded as having no stock and a current '
                        f'expiry, but its active lots still total {tracked_stock}. '
                        'Reconcile the stock and lot mismatch before reviewing the expiry.'
                    )
                    metadata = {
                        'result': 'blocked_by_lot_mismatch',
                        'snapshot': snapshot,
                    }
                if stale_review and safe_to_review:
                    detail += (
                        ' Nothing was cleared because the selected review changed; '
                        'review this fresh finding before trying again.'
                    )
                    metadata.update({
                        'result': 'changed_since_review',
                        'source_issue_id': stale_review['source_issue_id'],
                        'source_run_id': stale_review.get('source_run_id'),
                    })
                _issue(
                    issues,
                    code=ZERO_STOCK_EXPIRY_ISSUE,
                    title='Zero-stock product has a current expiry',
                    detail=detail,
                    product=product,
                    expected='No current expiry while stock is zero',
                    actual=_expiry_snapshot_label(snapshot),
                    repairable=safe_to_review,
                    metadata=metadata,
                )

            negative_fields = [
                ('quantity_in_stock', product.quantity_in_stock),
                ('price', product.price),
                ('price_per_unit', product.price_per_unit),
            ]
            for field, value in negative_fields:
                if value is not None and value < 0:
                    check_counts['negative_values'] += 1
                    _issue(
                        issues,
                        code='negative_product_value',
                        title='Negative product value',
                        detail=f'{field} cannot be negative.',
                        product=product,
                        expected='0 or greater',
                        actual=value,
                        metadata={'field': field},
                    )

        for item in expiry_action['stale']:
            product = item.get('product')
            if product and product.pk not in stale_expiry_by_product:
                # A fresh zero-stock finding above already explains this stale
                # selection and remains selectable for another review.
                continue
            check_counts['zero_stock_expiry'] += 1
            _issue(
                issues,
                code='zero_stock_expiry_review_changed',
                title='Expiry review changed before clearing',
                detail=(
                    f'{item["reason"]} No expiry dates were cleared. Run or '
                    'review the latest audit before trying again.'
                ),
                product=product,
                expected='The reviewed zero-stock expiry snapshot',
                actual=(
                    _expiry_snapshot_label(item['snapshot'])
                    if item.get('snapshot') else 'Selection unavailable'
                ),
                metadata={
                    'action': CLEAR_ZERO_STOCK_EXPIRY_ACTION,
                    'result': 'not_cleared',
                    'source_issue_id': item['source_issue_id'],
                    'source_run_id': item.get('source_run_id'),
                },
            )

        for line in SupplierPurchaseOrderLine.objects.filter(
            quantity_received__gt=F('quantity_ordered'),
        ).select_related('purchase_order', 'product'):
            check_counts['supplier_receiving'] += 1
            _issue(
                issues,
                code='supplier_over_received',
                title='Supplier receipt exceeds order',
                detail=(
                    f'{line.purchase_order}: {line.product_name} records more received '
                    'units than were ordered.'
                ),
                product=line.product,
                expected=line.quantity_ordered,
                actual=line.quantity_received,
                metadata={
                    'purchase_order_id': line.purchase_order_id,
                    'line_id': line.pk,
                },
            )

        for issue in issues:
            issue.run = run
        InventoryAuditIssue.objects.bulk_create(issues)

        issue_count = len(issues)
        if issue_count == 0:
            status = InventoryAuditRun.STATUS_PASSED
            summary = 'Inventory integrity audit passed.'
        elif repaired_count == issue_count:
            status = InventoryAuditRun.STATUS_REPAIRED
            summary = f'{repaired_count} inventory issue(s) repaired.'
        else:
            status = InventoryAuditRun.STATUS_ISSUES
            remaining = issue_count - repaired_count
            summary = f'{issue_count} issue(s) found; {remaining} still need attention.'

        checks = [
            {
                'key': key,
                'label': label,
                'issues': check_counts[key],
                'status': 'passed' if check_counts[key] == 0 else 'issues',
            }
            for key, label in [
                ('barcode_uniqueness', 'Barcode uniqueness'),
                ('barcode_normalization', 'Barcode search keys'),
                ('lot_balances', 'Product and lot totals'),
                ('zero_stock_expiry', 'Zero-stock current expiry'),
                ('negative_values', 'Non-negative stock and prices'),
                ('supplier_receiving', 'Supplier receiving limits'),
            ]
        ]
        run.status = status
        run.issue_count = issue_count
        run.repaired_count = repaired_count
        run.checks = checks
        run.summary = summary
        run.completed_at = timezone.now()
        run.save(update_fields=[
            'status', 'issue_count', 'repaired_count', 'checks', 'summary',
            'completed_at',
        ])
    except Exception:
        # The decorator rolls back repairs and result rows together. The public
        # wrapper records a durable error only after this transaction exits.
        raise

    return InventoryAuditRun.objects.prefetch_related('issues').select_related(
        'created_by',
    ).get(pk=run.pk)


def run_inventory_audit(
    *, created_by=None, repair_unassigned=False,
    clear_zero_stock_expiry_issue_ids=None,
    clear_zero_stock_expiry_run_id=None,
):
    """Run every durable inventory check and retain structured results."""
    run = InventoryAuditRun.objects.create(
        created_by=created_by,
        repair_requested=(
            repair_unassigned or bool(clear_zero_stock_expiry_issue_ids)
        ),
    )
    try:
        return _run_inventory_audit_transaction(
            run,
            repair_unassigned=repair_unassigned,
            clear_zero_stock_expiry_issue_ids=(
                clear_zero_stock_expiry_issue_ids or []
            ),
            clear_zero_stock_expiry_run_id=clear_zero_stock_expiry_run_id,
        )
    except Exception as exc:
        # The transaction above has fully rolled back, including any expiry or
        # lot mutation. Persist only the failed run so staff can see the error.
        InventoryAuditRun.objects.filter(pk=run.pk).update(
            status=InventoryAuditRun.STATUS_ERROR,
            issue_count=0,
            repaired_count=0,
            checks=[],
            error=str(exc)[:4000],
            summary='Inventory audit could not be completed.',
            completed_at=timezone.now(),
        )
        return InventoryAuditRun.objects.prefetch_related('issues').select_related(
            'created_by',
        ).get(pk=run.pk)


def serialize_audit_run(run, *, include_issues=True):
    if run is None:
        return None
    issues = list(run.issues.all()) if include_issues else []
    return {
        'id': run.pk,
        'status': run.status,
        'status_label': run.get_status_display(),
        'summary': display_lot_text(run.summary),
        'error': display_lot_text(run.error),
        'issue_count': run.issue_count,
        'repaired_count': run.repaired_count,
        'repairable_count': sum(
            1 for issue in issues
            if issue.repairable
            and not issue.repaired
            and (issue.metadata or {}).get('action') != CLEAR_ZERO_STOCK_EXPIRY_ACTION
        ),
        'clearable_expiry_count': sum(
            1 for issue in issues
            if issue.repairable
            and not issue.repaired
            and (issue.metadata or {}).get('action') == CLEAR_ZERO_STOCK_EXPIRY_ACTION
        ),
        'repair_requested': run.repair_requested,
        'checks': run.checks,
        'started_at': run.started_at.isoformat(),
        'completed_at': run.completed_at.isoformat() if run.completed_at else None,
        'created_by': (
            run.created_by.get_short_name() or run.created_by.get_username()
            if run.created_by else 'System'
        ),
        'issues': [
            {
                'id': issue.pk,
                'code': issue.code,
                'severity': issue.severity,
                'title': display_lot_text(issue.title),
                'detail': issue.staff_detail,
                'product_id': issue.product_id,
                'product_name': issue.product_name,
                'product_url': (
                    reverse('edit_product', args=[issue.product_id])
                    if issue.product_id else ''
                ),
                'expected': issue.staff_expected_value,
                'actual': issue.staff_actual_value,
                'repairable': issue.repairable,
                'repaired': issue.repaired,
                'action': (issue.metadata or {}).get('action', ''),
                'action_result': display_lot_text(
                    (issue.metadata or {}).get('result', '')
                ),
            }
            for issue in issues
        ],
    }
