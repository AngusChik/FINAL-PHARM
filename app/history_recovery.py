"""Read-only, uncapped archived-record browsing for the central History page."""

from datetime import date, datetime
from itertools import islice
from urllib.parse import urlencode

from django.db.models import CharField, F, Value
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.utils.dateparse import parse_date
from django.utils.timezone import is_aware, localtime


def _recovery_view():
    # views imports the History adapters; defer this existing archive policy
    # dependency until a request, after all view classes have been defined.
    from .views import ArchiveRecoveryView

    return ArchiveRecoveryView()


def _date_filter(value):
    try:
        return parse_date(value) if value else None
    except (TypeError, ValueError):
        return None


def get_recovery_history(request):
    """Return an ordered sequence compatible with Django's Paginator.

    Root History views apply the same administrator requirement as Recovery.
    The user filter refers to the person who removed the record.
    """
    view = _recovery_view()
    selected_type = request.GET.get('archive_type', 'all')
    kinds = [selected_type] if selected_type in view.TYPE_LABELS else view.TYPE_LABELS
    query = request.GET.get('q', '').strip()[:200]
    username = request.GET.get('user', '').strip()[:150]
    start = _date_filter(request.GET.get('date_from', ''))
    end = _date_filter(request.GET.get('date_to', ''))
    sources = {}
    for kind in kinds:
        queryset = view._queryset(kind, query, start, end)
        if username:
            user_field = 'deleted_by' if kind in ('order', 'ordering') else 'archived_by'
            queryset = queryset.filter(**{f'{user_field}__username__icontains': username})
        sources[kind] = queryset
    return RecoveryHistory(view, sources)


class RecoveryHistory:
    """Page lightweight UNION references before loading any archived objects."""

    ordered = True

    def __init__(self, view, sources):
        self.view = view
        self.sources = sources
        references = []
        for kind, queryset in sources.items():
            date_field = 'deleted_at' if kind in ('order', 'ordering') else 'archived_at'
            references.append(queryset.order_by().annotate(
                history_timestamp=F(date_field),
                history_kind=Value(kind, output_field=CharField()),
                history_pk=F('pk'),
            ).values('history_timestamp', 'history_kind', 'history_pk'))
        self.references = references[0].union(*references[1:], all=True).order_by(
            F('history_timestamp').desc(nulls_last=True), 'history_kind', '-history_pk',
        )

    def count(self):
        return self.references.count()

    def __len__(self):
        return self.count()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._hydrate(list(self.references[key]))
        if key < 0:
            key += len(self)
        if key < 0:
            raise IndexError(key)
        rows = self._hydrate(list(self.references[key:key + 1]))
        if not rows:
            raise IndexError(key)
        return rows[0]

    def __iter__(self):
        references = self.references.iterator(chunk_size=500)
        while batch := list(islice(references, 500)):
            yield from self._hydrate(batch)

    def _hydrate(self, references):
        records = {}
        for kind, queryset in self.sources.items():
            ids = [ref['history_pk'] for ref in references if ref['history_kind'] == kind]
            if ids:
                records[kind] = queryset.in_bulk(ids)
        rows = []
        for ref in references:
            kind = ref['history_kind']
            obj = records.get(kind, {}).get(ref['history_pk'])
            if obj is None:
                continue
            row = self.view._row(kind, obj)
            rows.append({
                'id': obj.pk,
                'kind': f'recovery-{kind}',
                'timestamp': row['archived_at'],
                'title': row['title'],
                'summary': ' · '.join(str(value) for value in (
                    row['type_label'], row['reference'], row['detail'], row['reason'],
                ) if value),
                'user': row['archived_by'] or '—',
                'status': 'Archived',
            })
        return rows


def _text(value):
    if value is None or value == '':
        return '—'
    if isinstance(value, datetime):
        return (localtime(value) if is_aware(value) else value).strftime('%b %d, %Y %H:%M')
    if isinstance(value, date):
        return value.strftime('%b %d, %Y')
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    return str(value)


def _fields(*pairs):
    return [{'label': label, 'value': _text(value)} for label, value in pairs]


def _section(title, columns, rows):
    return {'title': title, 'columns': columns, 'rows': [[_text(cell) for cell in row] for row in rows]}


def _money(value):
    return f'${value:.2f}' if value is not None else '—'


def _username(user):
    return user.get_username() if user else '—'


def _product_detail(obj):
    fields = _fields(
        ('Product', obj.name), ('Brand', obj.brand), ('Barcode', obj.barcode),
        ('Item number', obj.item_number), ('Department', obj.category),
        ('Unit size', obj.unit_size), ('Description', obj.description),
        ('Stock retained', obj.quantity_in_stock), ('Retail price', _money(obj.price)),
        ('Unit cost', _money(obj.price_per_unit)), ('Taxable', obj.taxable),
        ('Earliest expiry', obj.expiry_date), ('Created', obj.created_at),
        ('Last updated', obj.updated_at),
    )
    lots = _section('Retained lots', ['Lot', 'Expiry', 'Quantity', 'Received', 'Note'], (
        [lot.staff_name, lot.expiry_date, lot.quantity_on_hand, lot.received_at, lot.notes]
        for lot in obj.lots.order_by('expiry_date', 'pk')
    ))
    changes = _section('Stock history', ['Date', 'Action', 'Units', 'Staff', 'Note'], (
        [change.timestamp, change.get_change_type_display(), change.quantity,
         _username(change.user), change.staff_note]
        for change in obj.stock_changes.select_related('user').order_by('-timestamp', '-pk')
    ))
    return fields, [lots, changes]


def _ordering_detail(obj):
    fields = _fields(
        ('Item', obj.name), ('Type', obj.get_entry_type_display()),
        ('Status', obj.status_display), ('Patient', obj.patient_name),
        ('Phone', obj.phone_number), ('Side', obj.get_side_display()),
        ('Reason', obj.get_reasoning_display()), ('Urgency', obj.get_urgency_display()),
        ('Quantity needed', obj.quantity_needed), ('Quantity remaining', obj.quantity_remaining),
        ('Quantity ordered', obj.quantity_ordered), ('Quantity received', obj.quantity_received),
        ('Supplier', obj.supplier_name), ('Expected date', obj.expected_date),
        ('Note', obj.order_note), ('Initials', obj.initials),
        ('Created', obj.created_at), ('Created by', _username(obj.created_by)),
        ('Ordered', obj.ordered_at), ('Received', obj.received_at),
        ('Contacted', obj.contacted_at), ('Completed', obj.completed_at),
    )
    events = _section('Status history', ['Date', 'Previous status', 'New status', 'Staff', 'Note'], (
        [event.created_at, event.get_from_status_display(), event.get_to_status_display(),
         _username(event.changed_by), event.note]
        for event in obj.status_events.select_related('changed_by').order_by('-created_at', '-pk')
    ))
    return fields, [events]


def _delivery_detail(obj):
    return _fields(
        ('Name', f'{obj.first_name} {obj.last_name}'.strip()), ('Barcode', obj.barcode),
        ('Comment', obj.comment), ('Checked in', obj.checked_in_at),
        ('Checked out', obj.checked_out_at),
        ('Status when removed', 'Checked out' if obj.checked_out_at else 'On site'),
    ), []


def _recent_purchase_detail(obj):
    return _fields(
        ('Product', obj.product.name), ('Barcode', obj.product.barcode),
        ('Item number', obj.product.item_number), ('Quantity purchased', obj.quantity),
        ('Manual order quantity', obj.manual_order_quantity), ('Recorded', obj.order_date),
    ), []


def _special_order_detail(obj):
    return _fields(
        ('Item', obj.item_name), ('Item number', obj.item_number),
        ('Customer', f'{obj.first_name} {obj.last_name}'.strip()),
        ('Phone', obj.phone_number), ('Size', obj.get_size_display()),
        ('Side', obj.get_side_display()), ('Checked', obj.is_checked),
    ), []


def get_recovery_detail(kind, pk, request):
    """Describe one archived record, without exposing any mutation action."""
    view = _recovery_view()
    prefix = 'recovery-'
    if not kind.startswith(prefix) or kind[len(prefix):] not in view.TYPE_LABELS:
        raise Http404('Unknown archived record type.')
    archive_kind = kind[len(prefix):]
    obj = get_object_or_404(view._queryset(archive_kind), pk=pk)
    archive = view._row(archive_kind, obj)
    shared_kind = {
        'checkin': 'checkins', 'order': 'transactions', 'supplier_order': 'supplier_orders',
    }.get(archive_kind)
    if shared_kind:
        from .history_data import history_detail

        detail = history_detail(shared_kind, obj, request)
    else:
        builders = {
            'product': _product_detail, 'ordering': _ordering_detail,
            'delivery': _delivery_detail, 'recent_purchase': _recent_purchase_detail,
            'special_order': _special_order_detail,
        }
        fields, sections = builders[archive_kind](obj)
        detail = {'title': archive['title'], 'fields': fields, 'sections': sections}
    detail['subtitle'] = f"Archived {archive['type_label'].lower()}"
    detail['fields'] = _fields(
        ('Removed', archive['archived_at']), ('Removed by', archive['archived_by']),
        ('Removal note', archive['reason']),
    ) + detail.get('fields', [])
    filters = {'type': archive_kind}
    for key in ('q', 'date_from', 'date_to'):
        if request.GET.get(key):
            filters[key] = request.GET[key]
    detail['actions'] = [{
        'label': 'Open Recovery',
        'url': f"{reverse('archive_recovery')}?{urlencode(filters)}",
    }]
    return detail
