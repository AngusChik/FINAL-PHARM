"""Read-only adapters for the shared history browser.

List querysets stay ordered and unsliced until the view paginates them. Detail
pages use retained records and saved identities, never rebuild old transactions
or printed labels from today's product catalogue.
"""

from copy import copy
from datetime import date, datetime
from decimal import Decimal

from django.db.models import Count, Q
from django.db.models.functions import Coalesce
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.utils import timezone
from django.utils.dateparse import parse_date

from .models import (
    CheckinSession, CheckoutOrder, DailyReportArchive, DeliveryCheckIn,
    InventoryAuditRun, LabelSession, Order,
    PrescriptionDrugChange, PrescriptionDrugLearningRecord, StockChange,
    SupplierPurchaseOrder, display_lot_text,
)
from .reporting import realized_order_financials
from .utils import stock_change_delta


HISTORY_CATEGORIES = [
    {'key': 'transactions', 'label': 'Transactions', 'description': 'Submitted purchases, saved sale details, and adjustments.'},
    {'key': 'checkouts', 'label': 'Checkouts', 'description': 'Completed no-sale checkouts and their items.'},
    {'key': 'stock', 'label': 'Stock changes', 'description': 'Stock additions, removals, adjustments, and product activity.'},
    {'key': 'expired', 'label': 'Expired stock', 'description': 'Retired stock with quantities, notes, and lot details.'},
    {'key': 'scans', 'label': 'Scans', 'description': 'Recorded check-in scans, removals, and manual receiving adjustments.'},
    {'key': 'checkins', 'label': 'Check-in sessions', 'description': 'Receiving sessions, inventory counts, and archived sessions.'},
    {'key': 'labels', 'label': 'Label printing', 'description': 'Your saved print runs, product labels, and custom labels.'},
    {'key': 'reports', 'label': 'Daily reports', 'description': 'Saved report versions and their original PDFs.', 'user_filter': False},
    {'key': 'prescription_requests', 'label': 'Prescription requests', 'description': 'Ordering requests and their observed quantity and status changes.'},
    {'key': 'prescription_changes', 'label': 'Prescription record changes', 'description': 'Changes to prescription drugs, packs, and supplier details.'},
    {'key': 'deliveries', 'label': 'Deliveries', 'description': 'Completed deliveries, including archived records.', 'user_filter': False},
    {'key': 'inventory_audits', 'label': 'Inventory audits', 'description': 'Inventory checks, findings, and recorded repairs.'},
    {'key': 'supplier_orders', 'label': 'Supplier orders', 'description': 'Supplier order records, receipt quantities, and archived orders.'},
]

_SEARCH_FIELDS = {
    'stock': ('product_name', 'product_barcode', 'product__name', 'product__barcode', 'note', 'change_type'),
    'checkins': ('note', 'scanned_by', 'stock_changes__product_name', 'stock_changes__product_barcode', 'count_lines__product_name', 'count_lines__product_barcode'),
    'transactions': ('details__product_name', 'details__product_barcode'),
    'checkouts': ('items__product_name', 'items__product_barcode'),
    'labels': ('note', 'items__product_name', 'items__product_barcode', 'items__custom_lines'),
    'reports': ('summary',),
    'prescription_requests': ('source_name', 'drug__name', 'drug__brand', 'drug__din', 'reason'),
    'prescription_changes': ('drug__name', 'drug__brand', 'drug__din', 'entity_type', 'action', 'actor_label'),
    'deliveries': ('first_name', 'last_name', 'barcode', 'comment'),
    'inventory_audits': ('summary', 'status', 'issues__title', 'issues__product_name', 'issues__detail'),
    'supplier_orders': ('supplier', 'supplier_name', 'confirmation_number', 'notes', 'lines__product_name', 'lines__product_barcode'),
}
_USER_FIELDS = {
    'stock': ('user__username',),
    'checkins': ('user__username', 'scanned_by'),
    'transactions': ('user__username',),
    'checkouts': ('user__username',),
    'labels': ('user__username',),
    'prescription_requests': ('entry__created_by__username', 'entry__initials'),
    'prescription_changes': ('actor__username', 'actor_label'),
    'inventory_audits': ('created_by__username',),
    'supplier_orders': ('created_by__username',),
}
_TIME_FIELDS = {
    'stock': 'timestamp', 'checkins': 'started_at', 'transactions': 'order_date',
    'checkouts': '_history_time', 'labels': 'created_at', 'reports': 'report_date',
    'prescription_requests': '_history_time', 'prescription_changes': 'created_at',
    'deliveries': 'checked_out_at', 'inventory_audits': 'started_at',
    'supplier_orders': 'order_date',
}


def _base_queryset(key, request):
    if key in {'stock', 'expired', 'scans'}:
        queryset = StockChange.objects.select_related('product', 'user', 'session', 'correction_line')
        if key == 'expired':
            queryset = queryset.filter(change_type='expired')
        elif key == 'scans':
            queryset = queryset.filter(change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract'])
        return queryset
    if key == 'checkins':
        return CheckinSession.all_objects.select_related('user').annotate(
            history_scan_count=Count('stock_changes', distinct=True, filter=Q(
                stock_changes__change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract'],
            )),
            history_count_products=Count('count_lines', distinct=True),
        )
    if key == 'transactions':
        return Order.objects.filter(submitted=True).select_related('user')
    if key == 'checkouts':
        return CheckoutOrder.objects.filter(status=CheckoutOrder.STATUS_SUBMITTED).select_related('user').annotate(
            _history_time=Coalesce('submitted_at', 'created_at'),
        )
    if key == 'labels':
        # Match existing print-history ownership, including staff accounts.
        return LabelSession.objects.filter(user=request.user).select_related('user')
    if key == 'reports':
        return DailyReportArchive.objects.defer('pdf', 'snapshot_data')
    if key == 'prescription_requests':
        return PrescriptionDrugLearningRecord.objects.select_related('drug', 'entry__created_by').annotate(
            _history_time=Coalesce('requested_at', 'processed_at'),
        )
    if key == 'prescription_changes':
        return PrescriptionDrugChange.objects.select_related('drug', 'actor')
    if key == 'deliveries':
        return DeliveryCheckIn.objects.filter(checked_out_at__isnull=False)
    if key == 'inventory_audits':
        return InventoryAuditRun.objects.select_related('created_by')
    if key == 'supplier_orders':
        return SupplierPurchaseOrder.objects.select_related('created_by', 'plan')
    raise Http404('Unknown history category.')


def _date_filter(value):
    try:
        return parse_date(str(value or ''))
    except (TypeError, ValueError):
        return None


def _contains_any(fields, value):
    condition = Q()
    for field in fields:
        condition |= Q(**{f'{field}__icontains': value})
    return condition


def get_history_queryset(key, request):
    """Return the complete, filtered queryset for database pagination."""
    queryset = _base_queryset(key, request)
    family = 'stock' if key in {'expired', 'scans'} else key
    term = request.GET.get('q', '').strip()[:200]
    if term:
        condition = _contains_any(_SEARCH_FIELDS[family], term)
        numeric_term = term.lstrip('#')
        if numeric_term.isdecimal() and len(numeric_term) <= 18:
            condition |= Q(pk=int(numeric_term))
        queryset = queryset.filter(condition).distinct()
    actor = request.GET.get('user', '').strip()[:150]
    if actor:
        fields = _USER_FIELDS.get(family, ())
        # These sources do not record an author. Do not imply an unrelated
        # archive operator created the original report or delivery.
        queryset = queryset.filter(_contains_any(fields, actor)).distinct() if fields else queryset.none()
    time_field = _TIME_FIELDS[family]
    lookup = time_field if family in {'reports', 'supplier_orders'} else f'{time_field}__date'
    start = _date_filter(request.GET.get('date_from'))
    end = _date_filter(request.GET.get('date_to'))
    if start:
        queryset = queryset.filter(**{f'{lookup}__gte': start})
    if end:
        queryset = queryset.filter(**{f'{lookup}__lte': end})
    ordering = [f'-{time_field}']
    if family == 'reports':
        ordering.append('-updated_at')
    elif family == 'supplier_orders':
        ordering.append('-created_at')
    return queryset.order_by(*ordering, '-pk')


def get_history_record(key, pk, request):
    """Load one in-scope record independently of temporary list filters."""
    queryset = _base_queryset(key, request)
    if key in {'stock', 'expired', 'scans'}:
        queryset = queryset.select_related('order_detail', 'checkout_item').prefetch_related('lot_movements')
    elif key == 'checkins':
        queryset = queryset.prefetch_related(
            'count_lines', 'stock_changes__product', 'stock_changes__user',
            'stock_changes__correction_line', 'stock_changes__lot_movements',
        )
    elif key in {'transactions', 'checkouts'}:
        relation = 'details' if key == 'transactions' else 'items'
        queryset = queryset.prefetch_related(
            relation, f'{relation}__stock_changes__product', f'{relation}__stock_changes__user',
            f'{relation}__stock_changes__lot_movements', f'{relation}__stock_changes__correction_line',
            'corrections__created_by',
            'corrections__lines', 'corrections__undo__created_by',
        )
    elif key == 'labels':
        queryset = queryset.prefetch_related('items')
    elif key == 'prescription_requests':
        queryset = queryset.prefetch_related('revisions__drug')
    elif key == 'inventory_audits':
        queryset = queryset.prefetch_related('issues')
    elif key == 'supplier_orders':
        queryset = queryset.prefetch_related('lines')
    return get_object_or_404(queryset, pk=pk)


def _user(user, fallback='—'):
    return user.get_username() if user else fallback


def _value(value):
    if value is None or value == '':
        return '—'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, datetime):
        local = timezone.localtime(value) if timezone.is_aware(value) else value
        return local.strftime('%b %d, %Y %I:%M %p')
    if isinstance(value, date):
        return value.strftime('%b %d, %Y')
    if isinstance(value, Decimal):
        return format(value.normalize(), 'f')
    if isinstance(value, dict):
        return '\n'.join(f'{_label(key)}: {_value(item)}' for key, item in value.items()) or '—'
    if isinstance(value, (tuple, list)):
        return '\n'.join(_value(item) for item in value) or '—'
    return display_lot_text(value)


def _label(name):
    return str(name).replace('_', ' ').capitalize()


def _money(amount):
    return f'${amount:,.2f}' if amount is not None else 'Not recorded'


def _name(record):
    return record.product_name or (record.product.name if record.product_id else 'Product no longer available')


def _barcode(record):
    # An existing snapshot name identifies a saved record; a deliberately
    # empty saved barcode must not be replaced by a later catalogue barcode.
    if record.product_name:
        return record.product_barcode
    return record.product_barcode or (record.product.barcode if record.product_id else '')


def _retained_status(record, status):
    if getattr(record, 'archived_at', None) or getattr(record, 'is_deleted', False):
        return f'Archived · {status}'
    if getattr(record, 'hidden_from_history', False):
        return f'Hidden from checkout history · {status}'
    return status


def _stock_effect(record):
    """Use the ledger's event/disposition rules, not the stored quantity sign."""
    if record.change_type == 'lot_reassignment':
        return f'{abs(record.quantity)} units moved between lots'
    disposition = record.correction_line.disposition if record.correction_line_id else None
    delta = stock_change_delta(record.change_type, record.quantity, disposition)
    if delta:
        return f'{delta:+d} units'
    return f'{abs(record.quantity)} units · no stock change'


def history_row(key, record):
    """Normalize a single saved record without fetching its full detail."""
    row = {'id': record.pk, 'timestamp': None, 'title': '', 'summary': '', 'user': '—', 'status': 'Recorded'}
    if key in {'stock', 'expired', 'scans'}:
        row.update(timestamp=record.timestamp, title=_name(record), user=_user(record.user),
                   summary=f'{_stock_effect(record)} · {record.staff_note or _barcode(record) or "No note"}',
                   status=record.get_change_type_display())
    elif key == 'checkins':
        count = getattr(record, 'history_scan_count', None)
        products = getattr(record, 'history_count_products', None)
        row.update(timestamp=record.started_at, title=f'Check-in session #{record.pk}',
                   user=record.scanned_by or _user(record.user),
                   summary=(f'Inventory count · {products} products' if record.inventory_mode and products is not None
                            else f'{count} recorded scans' if count is not None else record.staff_note),
                   status='Active' if record.ended_at is None else 'Completed')
    elif key == 'transactions':
        row.update(timestamp=record.order_date, title=f'Transaction #{record.pk}', user=_user(record.user),
                   summary=(f'{_money(record.total_price)} original total' if record.financial_snapshot_source
                            else 'Saved purchase details'), status='Submitted')
    elif key == 'checkouts':
        row.update(timestamp=record.submitted_at or record.created_at, title=f'Checkout #{record.pk}',
                   user=_user(record.user), summary='No-sale checkout', status=record.get_status_display())
    elif key == 'labels':
        row.update(timestamp=record.created_at, title=f'Label print run #{record.pk}', user=_user(record.user),
                   summary=f'{record.label_count} labels' + (f' · {record.note}' if record.note else ''), status='Printed')
    elif key == 'reports':
        row.update(timestamp=record.updated_at, title=f'Daily report · {_value(record.report_date)}',
                   summary=record.summary or f'Saved version #{record.pk}', status='Saved')
    elif key == 'prescription_requests':
        snapshot = record.source_snapshot or {}
        row.update(timestamp=record.requested_at or record.processed_at, title=record.source_name,
                   user=_user(record.entry.created_by, record.entry.initials or '—') if record.entry_id else '—',
                   summary=f'Requested: {_value(snapshot.get("quantity_needed"))}',
                   status=record.status_display if record.drug_id else record.get_reason_display() or 'Not matched')
        if snapshot.get('is_deleted'):
            row['status'] = f'Archived request · {row["status"]}'
    elif key == 'prescription_changes':
        row.update(timestamp=record.created_at, title=str(record.drug),
                   user=record.actor_label or _user(record.actor),
                   summary=f'{_label(record.entity_type)} · {_label(record.action)}', status='Recorded')
    elif key == 'deliveries':
        row.update(timestamp=record.checked_out_at, title=f'{record.first_name} {record.last_name}'.strip() or f'Delivery #{record.pk}',
                   summary=record.comment or record.barcode, status='Completed')
    elif key == 'inventory_audits':
        row.update(timestamp=record.started_at, title=f'Inventory audit #{record.pk}',
                   user=_user(record.created_by, 'System'), summary=record.summary or f'{record.issue_count} findings',
                   status=record.get_status_display())
    elif key == 'supplier_orders':
        row.update(timestamp=record.created_at, title=f'{record.display_supplier} · {record.confirmation_number or "#" + str(record.pk)}',
                   user=_user(record.created_by), summary=f'Ordered {_value(record.order_date)}', status=record.get_status_display())
    else:
        raise Http404('Unknown history category.')
    row['status'] = _retained_status(record, row['status'])
    return row


def _fields(*pairs):
    return [{'label': label, 'value': _value(value)} for label, value in pairs]


def _section(title, columns, rows):
    return {'title': title, 'columns': columns, 'rows': [[_value(value) for value in row] for row in rows]}


def _stock_section(changes, title='Stock activity'):
    return _section(title, ['Date', 'Product', 'Barcode', 'Action', 'Stock effect', 'User', 'Note'], [
        [change.timestamp, _name(change), _barcode(change), change.get_change_type_display(),
         _stock_effect(change), _user(change.user), change.staff_note]
        for change in changes
    ])


def _lot_section(changes):
    return _section('Lot movements', ['Product', 'Lot', 'Expiry', 'Movement', 'Units'], [
        [_name(change), movement.staff_lot_name, movement.expiry_date, movement.get_direction_display(), movement.quantity]
        for change in changes for movement in change.lot_movements.all()
    ])


def _snapshot_sections(snapshot, title='Saved report'):
    """Render structured saved report data as fields/tables, including older versions."""
    sections = []
    if not isinstance(snapshot, dict):
        return sections
    # Report snapshots also contain link targets and chart sizing for their
    # original layout. Keep the history detail focused on recorded values.
    def visible_key(key):
        return key not in {'can_open_product', 'height_pct'} and not key.endswith(('_id', '_url'))

    scalars = [(key, value) for key, value in snapshot.items()
               if visible_key(key) and not isinstance(value, (dict, list))]
    if scalars:
        sections.append(_section(title, ['Field', 'Saved value'], [[_label(key), value] for key, value in scalars]))
    for key, value in snapshot.items():
        section_title = f'{title} · {_label(key)}'
        if isinstance(value, dict):
            sections.extend(_snapshot_sections(value, section_title))
        elif isinstance(value, list) and value:
            if all(isinstance(item, dict) for item in value):
                columns = list(dict.fromkeys(field for item in value for field in item if visible_key(field)))
                if columns:
                    sections.append(_section(section_title, [_label(field) for field in columns], [
                        [item.get(field) for field in columns] for item in value
                    ]))
            else:
                sections.append(_section(section_title, ['Saved value'], [[item] for item in value]))
    return sections


def history_detail(key, record, request):
    """Readable saved detail with no editing, regeneration, or restore side effects."""
    row = history_row(key, record)
    detail = {'title': row['title'], 'subtitle': row['summary'], 'fields': _fields(
        ('Record', f'#{record.pk}'), ('Date', row['timestamp']), ('User', row['user']), ('Status', row['status']),
    ), 'sections': [], 'actions': []}
    fields, sections, actions = detail['fields'], detail['sections'], detail['actions']
    if key in {'stock', 'expired', 'scans'}:
        fields.extend(_fields(('Product', _name(record)), ('Barcode', _barcode(record)),
                              ('Recorded quantity', record.quantity), ('Stock effect', _stock_effect(record)), ('Note', record.staff_note),
                              ('Check-in session', f'#{record.session_id}' if record.session_id else None)))
        sections.append(_lot_section([record]))
        if record.product_id and not record.product.archived_at:
            actions.append({'label': 'View product', 'url': reverse('product_details', args=[record.product_id])})
        if record.session_id and not record.session.archived_at:
            actions.append({'label': 'View check-in session', 'url': reverse('checkin_session_detail', args=[record.session_id])})
        if record.order_detail_id:
            actions.append({'label': 'View transaction', 'url': reverse('order_detail', args=[record.order_detail.order_id])})
        if record.checkout_item_id:
            actions.append({'label': 'View checkout', 'url': reverse('giveaway_detail', args=[record.checkout_item.checkout_id])})
    elif key == 'checkins':
        fields.extend(_fields(('Mode', 'Inventory count' if record.inventory_mode else 'Receiving'),
                              ('Started', record.started_at), ('Ended', record.ended_at),
                              ('Reopened', record.reopened_at), ('Scanned by', record.scanned_by), ('Note', record.staff_note)))
        changes = sorted(record.stock_changes.all(), key=lambda item: (item.timestamp, item.pk), reverse=True)
        sections.append(_stock_section(changes))
        sections.append(_lot_section(changes))
        if record.inventory_mode:
            counts = list(record.count_lines.all())
            fields.extend(_fields(('Products in count', len(counts)), ('Units counted', sum(line.counted_qty for line in counts))))
            sections.append(_section('Inventory count', ['Product', 'Barcode', 'Expected', 'Counted', 'Variance'], [
                [line.product_name, line.product_barcode, line.expected_qty, line.counted_qty, f'{line.variance:+d}'] for line in counts
            ]))
        if not record.archived_at:
            actions.append({'label': 'View session PDF', 'url': reverse('checkin_session_pdf', args=[record.pk])})
    elif key in {'transactions', 'checkouts'}:
        is_sale = key == 'transactions'
        lines = list((record.details if is_sale else record.items).all())
        corrections = list(record.corrections.all())
        active_corrections = [correction for correction in corrections if not hasattr(correction, 'undo')]
        adjusted_quantities = {}
        for correction in active_corrections:
            for line in correction.lines.all():
                line_id = line.order_detail_id if is_sale else line.checkout_item_id
                adjusted_quantities[line_id] = adjusted_quantities.get(line_id, 0) + line.quantity
        if is_sale:
            financial_lines = [copy(line) for line in lines]
            for line in financial_lines:
                line.realized_quantity = line.quantity
            original = realized_order_financials(record, financial_lines)
            fields.extend(_fields(('Original subtotal', _money(original['subtotal'])),
                                  ('Original discount', _money(original['discount_amount'])),
                                  ('Original tax', _money(original['tax'])), ('Original total', _money(original['total'])),
                                  ('Financial basis', record.get_financial_snapshot_source_display()
                                   or 'Reconstructed from saved sale items')))
            if active_corrections:
                for line in financial_lines:
                    line.realized_quantity = max(0, line.quantity - adjusted_quantities.get(line.pk, 0))
                current = realized_order_financials(record, financial_lines)
                fields.extend(_fields(('Current subtotal', _money(current['subtotal'])),
                                      ('Current discount', _money(current['discount_amount'])),
                                      ('Current tax', _money(current['tax'])), ('Current total', _money(current['total']))))
        else:
            fields.extend(_fields(('Recorded subtotal (no sale)', _money(record.subtotal)),
                                  ('Recorded tax value (no sale)', _money(record.tax)),
                                  ('Recorded item value (no sale)', _money(record.total_price))))
        sections.append(_section('Purchased items' if is_sale else 'Checkout items',
                                 ['Product', 'Barcode', 'Fulfilled', 'Adjusted', 'Remaining', 'Unit price', 'Original line value', 'Taxable'], [
            [line.product_name, line.product_barcode, line.quantity, adjusted_quantities.get(line.pk, 0),
             max(0, line.quantity - adjusted_quantities.get(line.pk, 0)), _money(line.price), _money(line.line_total),
             line.taxable_at_sale if is_sale else line.taxable] for line in lines
        ]))
        sections.append(_section('Transaction adjustments', ['Date', 'Type', 'Reason', 'Note', 'Amount', 'User', 'Void undone'], [
            [correction.created_at, correction.get_correction_type_display(), correction.reason, correction.note,
             _money(correction.adjustment_amount), _user(correction.created_by),
             correction.undo.created_at if hasattr(correction, 'undo') else None] for correction in corrections
        ]))
        sections.append(_section('Adjusted items', ['Adjustment', 'Product', 'Barcode', 'Quantity', 'Unit price', 'Disposition'], [
            [f'#{correction.pk}', line.product_name, line.product_barcode, line.quantity, _money(line.unit_price), line.get_disposition_display()]
            for correction in corrections for line in correction.lines.all()
        ]))
        sections.append(_section('Void reversals', ['Adjustment', 'Reversed at', 'Reason', 'User'], [
            [f'#{correction.pk}', correction.undo.created_at, correction.undo.reason, _user(correction.undo.created_by)]
            for correction in corrections if hasattr(correction, 'undo')
        ]))
        changes = sorted(
            (change for line in lines for change in line.stock_changes.all()),
            key=lambda item: (item.timestamp, item.pk), reverse=True,
        )
        sections.append(_stock_section(changes))
        sections.append(_lot_section(changes))
        actions.append({'label': 'View transaction details' if is_sale else 'View checkout details',
                        'url': reverse('order_detail' if is_sale else 'giveaway_detail', args=[record.pk])})
        if is_sale:
            actions.append({'label': 'View receipt PDF', 'url': reverse('order_pdf', args=[record.pk])})
    elif key == 'labels':
        fields.extend(_fields(('Labels printed', record.label_count), ('Print note', record.note)))
        sections.append(_section('Saved labels', ['Label', 'Type', 'Copies', 'Barcode', 'Price', 'Brand', 'Item number', 'Custom text'], [
            [item.product_name, 'Custom label' if item.is_custom else 'Product label', item.qty,
             item.product_barcode, None if item.is_custom else _money(item.product_price), item.product_brand,
             item.product_item_number, item.custom_lines if item.is_custom else None] for item in record.items.all()
        ]))
    elif key == 'reports':
        fields.extend(_fields(('Report date', record.report_date), ('Saved at', record.updated_at), ('Summary', record.summary)))
        sections.extend(_snapshot_sections(record.snapshot_data))
        url = reverse('daily_report_archive_pdf', args=[record.pk])
        actions.extend([{'label': 'View saved PDF', 'url': url}, {'label': 'Download saved PDF', 'url': f'{url}?download=1'}])
    elif key == 'prescription_requests':
        snapshot = record.source_snapshot or {}
        fields.extend(_fields(('Source label', record.source_name), ('Matched drug', str(record.drug) if record.drug_id else None),
                              ('Requested at', record.requested_at), ('Last observed', record.processed_at),
                              ('Parsed quantity', record.quantity_needed), ('Quantity unit', record.quantity_unit),
                              ('Catalogue result', record.get_reason_display() or 'Matched')))
        sections.append(_section('Latest saved request', ['Field', 'Saved value'], [
            ['Source label', snapshot.get('name', record.source_name)],
            ['Quantity needed', snapshot.get('quantity_needed')], ['Quantity remaining', snapshot.get('quantity_remaining')],
            ['Quantity ordered', snapshot.get('quantity_ordered')], ['Quantity received', snapshot.get('quantity_received')],
            ['Status', record.status_display], ['Archived request', snapshot.get('is_deleted', False)],
        ]))
        sections.append(_section('Observed revisions', ['Observed', 'Source label', 'Matched drug', 'Needed', 'Parsed quantity', 'Unit', 'Remaining', 'Ordered', 'Received', 'Status', 'Archived'], [
            [revision.observed_at, revision.snapshot.get('name'), str(revision.drug) if revision.drug_id else None,
             revision.snapshot.get('quantity_needed'), revision.snapshot.get('quantity_needed_value'),
             revision.snapshot.get('quantity_unit'), revision.snapshot.get('quantity_remaining'),
             revision.snapshot.get('quantity_ordered'), revision.snapshot.get('quantity_received'),
             revision.status_display, revision.snapshot.get('is_deleted', False)] for revision in record.revisions.all()
        ]))
        if record.drug_id:
            actions.append({'label': 'View prescription drug', 'url': reverse('prescription_drug_detail', args=[record.drug_id])})
    elif key == 'prescription_changes':
        fields.extend(_fields(('Drug', str(record.drug)), ('Record type', _label(record.entity_type)),
                              ('Record ID', record.entity_id), ('Action', _label(record.action))))
        before, after = record.before or {}, record.after or {}
        sections.append(_section('Recorded changes', ['Field', 'Before', 'After'], [
            [_label(name), before.get(name), after.get(name)]
            for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)
        ]))
        actions.append({'label': 'View prescription drug', 'url': reverse('prescription_drug_detail', args=[record.drug_id])})
    elif key == 'deliveries':
        fields.extend(_fields(('Name', f'{record.first_name} {record.last_name}'.strip()), ('Barcode', record.barcode),
                              ('Checked in', record.checked_in_at), ('Checked out', record.checked_out_at),
                              ('Time onsite', record.checked_out_at - record.checked_in_at if record.checked_out_at else None),
                              ('Comment', record.comment)))
    elif key == 'inventory_audits':
        fields.extend(_fields(('Completed', record.completed_at), ('Findings', record.issue_count),
                              ('Repaired', record.repaired_count), ('Repair requested', record.repair_requested),
                              ('Summary', record.summary), ('Error', record.error)))
        sections.append(_section('Checks performed', ['Check', 'Result', 'Findings'], [
            [check.get('label', check.get('key')), _label(check.get('status', '')), check.get('issues')]
            for check in (record.checks or []) if isinstance(check, dict)
        ]))
        sections.append(_section('Recorded findings', ['Product', 'Finding', 'Severity', 'Details', 'Expected', 'Actual', 'Repaired'], [
            [issue.product_name, issue.title, issue.get_severity_display(), issue.staff_detail,
             issue.staff_expected_value, issue.staff_actual_value, issue.repaired] for issue in record.issues.all()
        ]))
    elif key == 'supplier_orders':
        fields.extend(_fields(('Supplier', record.display_supplier), ('Confirmation number', record.confirmation_number),
                              ('Order date', record.order_date), ('Expected date', record.expected_date),
                              ('Last updated', record.updated_at), ('Notes', record.notes)))
        sections.append(_section('Supplier order items', ['Product', 'Barcode', 'Ordered', 'Received', 'Remaining', 'Unit cost'], [
            [line.product_name, line.product_barcode, line.quantity_ordered, line.quantity_received, line.remaining, _money(line.unit_cost)]
            for line in record.lines.all()
        ]))
    if getattr(record, 'archived_at', None):
        fields.extend(_fields(('Archived at', record.archived_at), ('Archive reason', getattr(record, 'archive_reason', ''))))
    if getattr(record, 'is_deleted', False):
        fields.extend(_fields(('Archived at', record.deleted_at)))
    # A section with no saved rows adds no information to the detail page.
    detail['sections'] = [section for section in sections if section['rows']]
    return detail
