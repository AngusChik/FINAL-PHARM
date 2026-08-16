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
    ProductLot,
    SupplierPurchaseOrderLine,
    normalize_barcode_key,
)


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


def run_inventory_audit(*, created_by=None, repair_unassigned=False):
    """Run every durable inventory check and retain structured results.

    The optional repair only adds a positive missing balance to the product's
    UNASSIGNED lot. It never changes the product total or reduces a named lot.
    """
    run = InventoryAuditRun.objects.create(
        created_by=created_by,
        repair_requested=repair_unassigned,
    )
    issues = []
    check_counts = {
        'barcode_uniqueness': 0,
        'barcode_normalization': 0,
        'lot_balances': 0,
        'negative_values': 0,
        'supplier_receiving': 0,
    }
    repaired_count = 0

    try:
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
            ).order_by('pk')
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
                        'UNASSIGNED without changing inventory.'
                    ),
                    product=product,
                    expected=expected_stock,
                    actual=tracked_stock,
                    repairable=repairable,
                    repaired=repaired,
                    metadata={'difference': difference},
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
    except Exception as exc:
        run.status = InventoryAuditRun.STATUS_ERROR
        run.error = str(exc)[:4000]
        run.summary = 'Inventory audit could not be completed.'
        run.completed_at = timezone.now()
        run.save(update_fields=['status', 'error', 'summary', 'completed_at'])

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
        'summary': run.summary,
        'error': run.error,
        'issue_count': run.issue_count,
        'repaired_count': run.repaired_count,
        'repairable_count': sum(1 for issue in issues if issue.repairable and not issue.repaired),
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
                'title': issue.title,
                'detail': issue.detail,
                'product_id': issue.product_id,
                'product_name': issue.product_name,
                'product_url': (
                    reverse('edit_product', args=[issue.product_id])
                    if issue.product_id else ''
                ),
                'expected': issue.expected_value,
                'actual': issue.actual_value,
                'repairable': issue.repairable,
                'repaired': issue.repaired,
            }
            for issue in issues
        ],
    }
