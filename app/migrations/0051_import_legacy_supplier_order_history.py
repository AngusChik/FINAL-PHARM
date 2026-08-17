import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.db import migrations, models


ACTIVE_STATES = {'starting', 'login', 'waiting_user', 'running', 'paused', 'review'}
TERMINAL_STATES = {'done', 'error', 'cancelled'}


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return None


def _file_time(path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.now(tz=timezone.utc)


def _positive_int(value, default=1):
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _product_id(Product, row):
    raw_id = row.get('product_id')
    try:
        raw_id = int(raw_id)
    except (TypeError, ValueError):
        raw_id = None
    if raw_id and Product.objects.filter(pk=raw_id).exists():
        return raw_id
    barcode = str(row.get('barcode') or '').strip()
    if barcode:
        return Product.objects.filter(barcode=barcode).values_list('pk', flat=True).first()
    return None


def _set_run_times(SupplierOrderRun, run, created_at, updated_at=None, completed=False):
    values = {
        'created_at': created_at,
        'started_at': created_at,
        'updated_at': updated_at or created_at,
    }
    if completed:
        values['completed_at'] = updated_at or created_at
    SupplierOrderRun.objects.filter(pk=run.pk).update(**values)


def _import_report(apps, base, vendor, prefix):
    SupplierOrderRun = apps.get_model('app', 'SupplierOrderRun')
    SupplierOrderRunItem = apps.get_model('app', 'SupplierOrderRunItem')
    Product = apps.get_model('app', 'Product')
    path = base / f'{prefix}_order_report.csv'
    if not path.exists():
        return
    marker = f'Imported legacy CSV report: {path.name}'
    if SupplierOrderRun.objects.filter(vendor=vendor, source='legacy_report', message=marker).exists():
        return
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return
    if not rows:
        return
    timestamp = _file_time(path)
    run = SupplierOrderRun.objects.create(
        vendor=vendor,
        source='legacy_report',
        state='done',
        message=marker,
        current=len(rows),
        total=len(rows),
    )
    items = []
    for position, row in enumerate(rows):
        status = str(row.get('status') or '').strip().lower()
        outcome = 'added' if status == 'added' else 'skipped'
        items.append(SupplierOrderRunItem(
            run=run,
            product_id=_product_id(Product, row),
            product_name=str(row.get('name') or '')[:200],
            barcode=str(row.get('barcode') or '')[:64],
            quantity_requested=_positive_int(row.get('quantity')),
            position=position,
            outcome=outcome,
            reason=str(row.get('reason') or '')[:500],
            processed_at=timestamp,
        ))
    SupplierOrderRunItem.objects.bulk_create(items)
    _set_run_times(SupplierOrderRun, run, timestamp, completed=True)


def _result_key(row):
    barcode = str(row.get('barcode') or '').strip()
    if barcode:
        return ('barcode', barcode)
    return ('name', str(row.get('name') or '').strip().casefold())


def _import_status(apps, base, vendor, prefix):
    SupplierOrderRun = apps.get_model('app', 'SupplierOrderRun')
    SupplierOrderRunItem = apps.get_model('app', 'SupplierOrderRunItem')
    Product = apps.get_model('app', 'Product')
    status_path = base / f'{prefix}_order_status.json'
    items_path = base / f'{prefix}_order_items.json'
    status = _read_json(status_path)
    item_data = _read_json(items_path)
    rows = item_data.get('items', []) if isinstance(item_data, dict) else []
    if not isinstance(status, dict) and not rows:
        return
    marker = f'Imported legacy status: {status_path.name}'
    if SupplierOrderRun.objects.filter(vendor=vendor, source='legacy_status', message__startswith=marker).exists():
        return

    status = status if isinstance(status, dict) else {}
    original_state = str(status.get('state') or 'error')
    state = original_state if original_state in ACTIVE_STATES | TERMINAL_STATES else 'error'
    if state in ACTIVE_STATES:
        state = 'error'
    legacy_message = str(status.get('message') or '').strip()
    message = f'{marker}. {legacy_message}' if legacy_message else marker
    updated_at = None
    try:
        updated_at = datetime.fromtimestamp(float(status.get('updated_at')), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        updated_at = _file_time(status_path if status_path.exists() else items_path)
    created_at = min(
        [_file_time(path) for path in (status_path, items_path) if path.exists()] or [updated_at]
    )

    added = status.get('added') if isinstance(status.get('added'), list) else []
    skipped = status.get('skipped') if isinstance(status.get('skipped'), list) else []
    result_map = {}
    for result in added:
        if isinstance(result, dict):
            result_map[_result_key(result)] = ('added', str(result.get('reason') or ''))
    for result in skipped:
        if isinstance(result, dict):
            result_map[_result_key(result)] = ('skipped', str(result.get('reason') or ''))

    if not rows:
        rows = [row for row in added + skipped if isinstance(row, dict)]
    total = max(int(status.get('total') or 0), len(rows))
    current = min(max(int(status.get('current') or 0), 0), total)
    run = SupplierOrderRun.objects.create(
        vendor=vendor,
        source='legacy_status',
        state=state,
        message=message[:500],
        current=current,
        total=total,
        cancel_requested=(original_state == 'cancelled'),
    )
    items = []
    for position, row in enumerate(rows):
        outcome, reason = result_map.get(_result_key(row), ('pending', ''))
        items.append(SupplierOrderRunItem(
            run=run,
            product_id=_product_id(Product, row),
            product_name=str(row.get('name') or '')[:200],
            barcode=str(row.get('barcode') or '')[:64],
            quantity_requested=_positive_int(row.get('quantity', row.get('qty'))),
            position=position,
            outcome=outcome,
            reason=reason[:500],
            processed_at=updated_at if outcome != 'pending' else None,
        ))
    SupplierOrderRunItem.objects.bulk_create(items)
    _set_run_times(
        SupplierOrderRun, run, created_at, updated_at,
        completed=(state in TERMINAL_STATES),
    )


def import_legacy_supplier_history(apps, schema_editor):
    base = Path(settings.BASE_DIR)
    for vendor, prefix in (('mck', 'mckesson'), ('kf', 'kohlfrisch')):
        _import_report(apps, base, vendor, prefix)
        _import_status(apps, base, vendor, prefix)


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0050_order_draft_expires_at_order_last_timer_reset_at_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='supplierorderrun',
            name='source',
            field=models.CharField(
                choices=[
                    ('web', 'Web ordering workflow'),
                    ('cli', 'Command line'),
                    ('legacy_status', 'Imported status file'),
                    ('legacy_report', 'Imported CSV report'),
                ],
                default='web',
                max_length=20,
            ),
        ),
        migrations.RunPython(import_legacy_supplier_history, migrations.RunPython.noop),
    ]
