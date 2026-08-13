"""Database storage and worker helpers for supplier-order automation.

The browser automation runs in a separate process.  This module gives both the
web process and the worker a small shared API backed by normal Django models,
so progress, controls, inputs, and results survive restarts and remain useful
for reporting.
"""

import os

from django.db import transaction
from django.utils import timezone

from .models import SupplierOrderRun, SupplierOrderRunItem


ACTIVE_STATES = {
    SupplierOrderRun.STATE_STARTING,
    SupplierOrderRun.STATE_LOGIN,
    SupplierOrderRun.STATE_WAITING_USER,
    SupplierOrderRun.STATE_RUNNING,
    SupplierOrderRun.STATE_PAUSED,
    SupplierOrderRun.STATE_REVIEW,
}
TERMINAL_STATES = {
    SupplierOrderRun.STATE_DONE,
    SupplierOrderRun.STATE_ERROR,
    SupplierOrderRun.STATE_CANCELLED,
}


def _safe_product_id(value):
    try:
        return int(value) if value not in (None, '') else None
    except (TypeError, ValueError):
        return None


def serialize_run(run):
    """Return the status shape consumed by the Recently Purchased page."""
    rows = list(run.items.select_related('product').all())
    added = [
        {
            'product_id': row.product_id,
            'name': row.product_name,
            'qty': row.quantity_requested,
            'barcode': row.barcode,
        }
        for row in rows if row.outcome == SupplierOrderRunItem.OUTCOME_ADDED
    ]
    skipped = [
        {
            'product_id': row.product_id,
            'name': row.product_name,
            'qty': row.quantity_requested,
            'barcode': row.barcode,
            'reason': row.reason,
        }
        for row in rows if row.outcome == SupplierOrderRunItem.OUTCOME_SKIPPED
    ]
    return {
        'run_id': run.pk,
        'plan_id': run.plan_id,
        'vendor': run.vendor,
        'state': run.state,
        'message': run.message,
        'current': run.current,
        'total': run.total,
        'pid': run.process_id,
        'paused': run.pause_requested,
        'cancel_requested': run.cancel_requested,
        'added': added,
        'skipped': skipped,
        'updated_at': run.updated_at.timestamp() if run.updated_at else None,
    }


class DatabaseRunStatus:
    """Status/control adapter used inside each supplier worker process."""

    def __init__(self, vendor, run_id=None):
        if run_id:
            self.run = SupplierOrderRun.objects.get(pk=run_id, vendor=vendor)
        else:
            self.run = SupplierOrderRun.objects.create(
                vendor=vendor, source=SupplierOrderRun.SOURCE_CLI,
            )
        self.update(process_id=os.getpid())

    def update(self, **values):
        allowed = {'state', 'message', 'current', 'total', 'process_id'}
        changes = {key: value for key, value in values.items() if key in allowed}
        timestamp = timezone.now()
        if changes.get('state') in TERMINAL_STATES:
            changes['completed_at'] = timestamp
        if (changes.get('state') not in (None, SupplierOrderRun.STATE_STARTING)
                and self.run.started_at is None):
            changes['started_at'] = timestamp
        changes['updated_at'] = timestamp
        SupplierOrderRun.objects.filter(pk=self.run.pk).update(**changes)
        self.run.refresh_from_db()

    def ensure_items(self, items, pre_skipped=None):
        """Persist an input list once and return pending rows as worker dicts."""
        pre_skipped = pre_skipped or []
        with transaction.atomic():
            run = SupplierOrderRun.objects.select_for_update().get(pk=self.run.pk)
            if not run.items.exists():
                rows = []
                position = 0
                for item in items:
                    rows.append(SupplierOrderRunItem(
                        run=run,
                        product_id=_safe_product_id(item.get('product_id')),
                        product_name=str(item.get('name') or '')[:200],
                        barcode=str(item.get('barcode') or '')[:64],
                        quantity_requested=max(1, int(item.get('quantity') or 1)),
                        position=position,
                    ))
                    position += 1
                for item in pre_skipped:
                    rows.append(SupplierOrderRunItem(
                        run=run,
                        product_id=_safe_product_id(item.get('product_id')),
                        product_name=str(item.get('name') or '')[:200],
                        barcode=str(item.get('barcode') or '')[:64],
                        quantity_requested=max(1, int(item.get('quantity') or 1)),
                        position=position,
                        outcome=SupplierOrderRunItem.OUTCOME_SKIPPED,
                        reason=str(item.get('reason') or '')[:500],
                        processed_at=timezone.now(),
                    ))
                    position += 1
                SupplierOrderRunItem.objects.bulk_create(rows)
                run.total = len(items)
                run.save(update_fields=['total', 'updated_at'])
        return self.pending_items()

    def pending_items(self):
        return [
            {
                '_run_item_id': row.pk,
                'product_id': row.product_id,
                'name': row.product_name,
                'barcode': row.barcode,
                'quantity': row.quantity_requested,
            }
            for row in self.run.items.filter(
                outcome=SupplierOrderRunItem.OUTCOME_PENDING,
            ).order_by('position', 'pk')
        ]

    def record_result(self, item, added, reason):
        item_id = item.get('_run_item_id')
        if not item_id:
            return
        SupplierOrderRunItem.objects.filter(
            pk=item_id, run_id=self.run.pk,
        ).update(
            outcome=(SupplierOrderRunItem.OUTCOME_ADDED if added
                     else SupplierOrderRunItem.OUTCOME_SKIPPED),
            reason=str(reason or '')[:500],
            processed_at=timezone.now(),
        )
        self.update()

    def control(self):
        return SupplierOrderRun.objects.filter(pk=self.run.pk).values(
            'pause_requested', 'cancel_requested',
        ).first() or {}

    def payload(self):
        self.run.refresh_from_db()
        return serialize_run(self.run)
