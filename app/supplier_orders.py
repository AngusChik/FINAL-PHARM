"""Database storage and worker helpers for supplier-order automation.

The browser automation runs in a separate process.  This module gives both the
web process and the worker a small shared API backed by normal Django models,
so progress, controls, inputs, and results survive restarts and remain useful
for reporting.
"""

import asyncio
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import connections, transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

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

SUPPLIER_ORDER_TASK_NAME = 'Pharmacy Supplier Ordering'
SUPPLIER_WORKER_SCRIPTS = {
    SupplierOrderRun.VENDOR_MCKESSON: 'mckesson_order.py',
    SupplierOrderRun.VENDOR_KOHLFRISCH: 'kohlfrisch_order.py',
}
SCHEDULED_LAUNCH_MAX_AGE = timedelta(minutes=15)
SCHEDULED_LAUNCH_START_TIMEOUT = timedelta(seconds=45)
_LAUNCH_MARKER_RE = re.compile(r'^supplier-order-(\d+)\.launch$')


_DATABASE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix='supplier-order-database',
)


def _is_windows():
    return os.name == 'nt'


def _no_window_creationflags():
    if not _is_windows():
        return 0
    return getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)


def _launch_marker_path(base_dir, run_id):
    return Path(base_dir) / '.runtime' / f'supplier-order-{int(run_id)}.launch'


def scheduled_launcher_repair_hint():
    return (
        'Keep the pharmacy Windows user signed in, then run '
        'setup-main-computer.bat to repair and test the supplier launcher.'
    )


def _discard_launch_file(path):
    """Best-effort cleanup that never hides the actionable launcher failure."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def queue_scheduled_supplier_launch(run, base_dir=None):
    """Ask Task Scheduler to broker one supplier worker outside Waitress's job.

    The marker contains only a database run id and its expected vendor. The
    scheduled dispatcher reconstructs the worker command from fixed constants,
    rather than executing command text supplied by the web process.
    """
    if not _is_windows():
        raise OSError('The scheduled supplier launcher is only available on Windows.')
    if run.vendor not in SUPPLIER_WORKER_SCRIPTS:
        raise OSError('Unknown supplier for the scheduled launcher.')

    base = Path(base_dir or settings.BASE_DIR)
    marker = _launch_marker_path(base, run.pk)
    temporary = marker.with_name(f'{marker.name}.{os.getpid()}.tmp')
    payload = {
        'run_id': run.pk,
        'vendor': run.vendor,
        'attempt': run.attempt,
        'requested_at': timezone.now().isoformat(),
    }
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload), encoding='utf-8')
        os.replace(temporary, marker)
    except OSError as exc:
        _discard_launch_file(temporary)
        raise OSError(
            'Windows could not create the supplier-launch request. '
            f'{scheduled_launcher_repair_hint()} Windows reported: {exc}'
        ) from exc

    try:
        result = subprocess.run(
            [
                'schtasks.exe', '/Run', '/TN', SUPPLIER_ORDER_TASK_NAME,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=_no_window_creationflags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _discard_launch_file(marker)
        raise OSError(
            f'Windows could not start the {SUPPLIER_ORDER_TASK_NAME} task. '
            f'{scheduled_launcher_repair_hint()} Windows reported: {exc}'
        ) from exc

    if result.returncode != 0:
        _discard_launch_file(marker)
        detail = (result.stderr or result.stdout or '').strip()
        if detail:
            detail = f' {detail}'
        raise OSError(
            f'The {SUPPLIER_ORDER_TASK_NAME} task is unavailable.{detail} '
            f'{scheduled_launcher_repair_hint()}'
        )

    if isinstance(run, SupplierOrderRun):
        SupplierOrderRun.objects.filter(
            pk=run.pk,
            attempt=run.attempt,
            state=SupplierOrderRun.STATE_STARTING,
            process_id__isnull=True,
        ).update(
            message='Waiting for the Windows supplier launcher...',
            updated_at=timezone.now(),
        )
    return marker


def _mark_launch_error(run_id, message, attempt=None):
    runs = SupplierOrderRun.objects.filter(
        pk=run_id,
        state=SupplierOrderRun.STATE_STARTING,
        process_id__isnull=True,
    )
    if attempt is not None:
        runs = runs.filter(attempt=attempt)
    runs.update(
        state=SupplierOrderRun.STATE_ERROR,
        message=str(message)[:500],
        completed_at=timezone.now(),
        updated_at=timezone.now(),
    )


def _launch_supplier_worker(run_id, base_dir=None, popen=None, attempt=None):
    """Launch exactly one validated pending run from the scheduled broker."""
    base = Path(base_dir or settings.BASE_DIR)
    python = base / 'env' / 'Scripts' / 'python.exe'
    popen = popen or subprocess.Popen

    with transaction.atomic():
        run = SupplierOrderRun.objects.select_for_update().filter(pk=run_id).first()
        if run is None or run.state != SupplierOrderRun.STATE_STARTING:
            return None, None
        if attempt is not None and run.attempt != attempt:
            return None, None
        if run.process_id:
            return run.process_id, None
        script_name = SUPPLIER_WORKER_SCRIPTS.get(run.vendor)
        script = base / script_name if script_name else None
        if not python.exists() or script is None or not script.exists():
            run.state = SupplierOrderRun.STATE_ERROR
            run.message = 'The supplier worker or application environment was not found.'
            run.completed_at = timezone.now()
            run.save(update_fields=[
                'state', 'message', 'completed_at', 'updated_at',
            ])
            return None, None

        logs_dir = base / 'logs'
        logs_dir.mkdir(exist_ok=True)
        try:
            with open(
                logs_dir / f'{run.vendor}_order.log',
                'a',
                encoding='utf-8',
            ) as logf:
                process = popen(
                    [
                        str(python), str(script), '--no-input',
                        '--run-id', str(run.pk), '--attempt', str(run.attempt),
                    ],
                    cwd=str(base),
                    stdin=subprocess.DEVNULL,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    creationflags=_no_window_creationflags(),
                    close_fds=True,
                )
        except OSError as exc:
            run.state = SupplierOrderRun.STATE_ERROR
            run.message = f'Could not start scheduled supplier worker: {exc}'[:500]
            run.completed_at = timezone.now()
            run.save(update_fields=[
                'state', 'message', 'completed_at', 'updated_at',
            ])
            return None, None

        run.process_id = process.pid
        run.save(update_fields=['process_id', 'updated_at'])
        return process.pid, process


def dispatch_scheduled_supplier_launches(
        base_dir=None, at=None, popen=None, wait_for_workers=False):
    """Consume validated launch markers and start their database-backed runs."""
    base = Path(base_dir or settings.BASE_DIR)
    runtime_dir = base / '.runtime'
    if not runtime_dir.exists():
        return []

    at = at or timezone.now()
    results = []
    processes = []
    for marker in sorted(runtime_dir.glob('supplier-order-*.launch')):
        match = _LAUNCH_MARKER_RE.fullmatch(marker.name)
        if not match:
            continue
        run_id = int(match.group(1))
        claimed = marker.with_name(f'{marker.name}.{os.getpid()}.claimed')
        try:
            os.replace(marker, claimed)
        except FileNotFoundError:
            continue

        try:
            requested_attempt = None
            payload = json.loads(claimed.read_text(encoding='utf-8'))
            requested_at = parse_datetime(str(payload.get('requested_at') or ''))
            if requested_at is None:
                raise ValueError('Supplier launch request has no valid timestamp.')
            if timezone.is_naive(requested_at):
                requested_at = timezone.make_aware(
                    requested_at, timezone.get_current_timezone(),
                )
            if at - requested_at > SCHEDULED_LAUNCH_MAX_AGE:
                raise ValueError('Supplier launch request expired before it could start.')
            if int(payload.get('run_id')) != run_id:
                raise ValueError('Supplier launch request id does not match its marker.')

            requested_attempt = int(payload.get('attempt'))
            run = SupplierOrderRun.objects.filter(pk=run_id).only(
                'vendor', 'attempt',
            ).first()
            if run is None:
                continue
            if payload.get('vendor') != run.vendor:
                raise ValueError('Supplier launch request vendor does not match the run.')
            if requested_attempt != run.attempt:
                raise ValueError('Supplier launch request belongs to an old attempt.')

            pid, process = _launch_supplier_worker(
                run_id, base, popen=popen, attempt=requested_attempt,
            )
            error = ''
            if pid is None:
                error = (
                    SupplierOrderRun.objects.filter(pk=run_id)
                    .values_list('message', flat=True)
                    .first()
                    or 'Scheduled supplier worker did not start.'
                )
            elif process is not None:
                processes.append((run_id, requested_attempt, process))
            results.append({'run_id': run_id, 'pid': pid, 'error': error})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            message = f'Could not validate scheduled supplier launch: {exc}'
            _mark_launch_error(run_id, message, attempt=requested_attempt)
            results.append({'run_id': run_id, 'pid': None, 'error': message})
        finally:
            claimed.unlink(missing_ok=True)
    if wait_for_workers:
        # Keep the Task Scheduler action alive for the lifetime of its browser
        # workers. This prevents the scheduler from considering the action
        # finished and potentially cleaning up its process tree prematurely.
        for run_id, attempt, process in processes:
            exit_code = process.wait()
            # A hard browser/worker crash can bypass its final status write.
            # Reconcile it here so the Control Manager never remains active
            # solely because the stored PID belonged to a departed child.
            SupplierOrderRun.objects.filter(
                pk=run_id,
                attempt=attempt,
                state__in=ACTIVE_STATES,
            ).update(
                state=SupplierOrderRun.STATE_ERROR,
                message=(
                    'The supplier worker exited before reporting completion '
                    f'(exit code {exit_code}).'
                ),
                process_id=None,
                cancel_requested=True,
                completed_at=timezone.now(),
                updated_at=timezone.now(),
            )
    return results


def _has_running_event_loop():
    """Return whether this thread is currently controlled by an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _database_worker(operation):
    """Run one ORM operation with a connection owned by the worker thread."""
    try:
        return operation()
    finally:
        # Supplier automation is a long-running process outside Django's normal
        # request lifecycle, so it must release its thread-local DB connection.
        connections.close_all()


def _run_database_operation(operation):
    """Keep synchronous ORM calls out of Playwright's running event loop."""
    if not _has_running_event_loop():
        return operation()
    return _DATABASE_EXECUTOR.submit(_database_worker, operation).result()


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
    pending_count = sum(
        1 for row in rows if row.outcome == SupplierOrderRunItem.OUTCOME_PENDING
    )
    return {
        'run_id': run.pk,
        'attempt': run.attempt,
        'plan_id': run.plan_id,
        'vendor': run.vendor,
        'state': run.state,
        'message': run.message,
        'current': run.current,
        'total': run.total,
        'pid': run.process_id,
        'paused': run.pause_requested,
        'cancel_requested': run.cancel_requested,
        'pending_count': pending_count,
        'added': added,
        'skipped': skipped,
        'updated_at': run.updated_at.timestamp() if run.updated_at else None,
        'heartbeat_at': run.heartbeat_at.timestamp() if run.heartbeat_at else None,
    }


class DatabaseRunStatus:
    """Status/control adapter used inside each supplier worker process."""

    def __init__(self, vendor, run_id=None, attempt=None):
        def load_or_create_run():
            if run_id:
                return SupplierOrderRun.objects.get(pk=run_id, vendor=vendor)
            return SupplierOrderRun.objects.create(
                vendor=vendor, source=SupplierOrderRun.SOURCE_CLI,
            )

        self.run = _run_database_operation(load_or_create_run)
        self.attempt = int(attempt or self.run.attempt)
        if self.run.attempt != self.attempt:
            raise RuntimeError('This supplier worker belongs to an expired attempt.')
        if self.update(process_id=os.getpid()) is False:
            raise RuntimeError('This supplier worker lease is no longer active.')

    def update(self, **values):
        return _run_database_operation(lambda: self._update(values))

    def _update(self, values):
        allowed = {'state', 'message', 'current', 'total', 'process_id'}
        changes = {key: value for key, value in values.items() if key in allowed}
        timestamp = timezone.now()
        with transaction.atomic():
            current = SupplierOrderRun.objects.select_for_update().get(pk=self.run.pk)

            # A status request may reclaim a worker whose PID is alive but
            # whose browser/Playwright loop has stopped heartbeating.  Never
            # let a late callback from that abandoned process resurrect the
            # terminal database row.
            if current.attempt != self.attempt or current.state in TERMINAL_STATES:
                self.run = current
                return False

            if changes.get('state') in TERMINAL_STATES:
                changes['completed_at'] = timestamp
            if (changes.get('state') not in (None, SupplierOrderRun.STATE_STARTING)
                    and current.started_at is None):
                changes['started_at'] = timestamp
            changes['heartbeat_at'] = timestamp
            changes['updated_at'] = timestamp
            SupplierOrderRun.objects.filter(pk=current.pk).update(**changes)
        self.run.refresh_from_db()
        return True

    def ensure_items(self, items, pre_skipped=None):
        """Persist an input list once and return pending rows as worker dicts."""
        return _run_database_operation(
            lambda: self._ensure_items(items, pre_skipped),
        )

    def _ensure_items(self, items, pre_skipped=None):
        pre_skipped = pre_skipped or []
        with transaction.atomic():
            run = SupplierOrderRun.objects.select_for_update().get(pk=self.run.pk)
            if run.attempt != self.attempt or run.state in TERMINAL_STATES:
                self.run = run
                return []
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
        return self._pending_items()

    def pending_items(self):
        return _run_database_operation(self._pending_items)

    def _pending_items(self):
        run = SupplierOrderRun.objects.filter(
            pk=self.run.pk,
            attempt=self.attempt,
        ).first()
        if run is None or run.state in TERMINAL_STATES:
            return []
        self.run = run
        return [
            {
                '_run_item_id': row.pk,
                'product_id': row.product_id,
                'name': row.product_name,
                'barcode': row.barcode,
                'quantity': row.quantity_requested,
            }
            for row in run.items.filter(
                outcome=SupplierOrderRunItem.OUTCOME_PENDING,
            ).order_by('position', 'pk')
        ]

    def record_result(self, item, added, reason):
        return _run_database_operation(
            lambda: self._record_result(item, added, reason),
        )

    def _record_result(self, item, added, reason):
        item_id = item.get('_run_item_id')
        if not item_id:
            return
        with transaction.atomic():
            run = SupplierOrderRun.objects.select_for_update().get(pk=self.run.pk)
            if run.attempt != self.attempt or run.state in TERMINAL_STATES:
                self.run = run
                return False
            SupplierOrderRunItem.objects.filter(
                pk=item_id, run_id=run.pk,
                outcome=SupplierOrderRunItem.OUTCOME_PENDING,
            ).update(
                outcome=(SupplierOrderRunItem.OUTCOME_ADDED if added
                         else SupplierOrderRunItem.OUTCOME_SKIPPED),
                reason=str(reason or '')[:500],
                processed_at=timezone.now(),
            )
        self._update({})
        return True

    def control(self):
        return _run_database_operation(self._control)

    def _control(self):
        control = SupplierOrderRun.objects.filter(
            pk=self.run.pk,
            attempt=self.attempt,
        ).values(
            'state', 'pause_requested', 'cancel_requested',
        ).first() or {}
        control['lease_active'] = control.get('state') in ACTIVE_STATES
        if not control['lease_active']:
            control['cancel_requested'] = True
        return control

    def payload(self):
        return _run_database_operation(self._payload)

    def _payload(self):
        self.run.refresh_from_db()
        return serialize_run(self.run)
