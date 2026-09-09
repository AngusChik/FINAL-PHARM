"""Uncapped activity history, ordered and paginated in the database."""

from itertools import islice
import re

from django.db.models import CharField, F, Value
from django.urls import reverse

from .models import (
    CheckinSession, LoginAudit, Product, StockChange, UserAction, display_lot_text,
)


def build_activity_history(view, event_type, user_filter, date_from, date_to):
    stock_types = dict(StockChange.CHANGE_TYPE_CHOICES)
    action_types = dict(UserAction.ACTION_CHOICES)
    direct_stock = event_type.removeprefix('stock:') if event_type.startswith('stock:') else None
    direct_action = event_type.removeprefix('action:') if event_type.startswith('action:') else None
    valid = (
        event_type in view.LOGIN_TYPES + view.STOCK_TYPES + view.ACTION_TYPES
        or event_type in view.STOCK_TYPE_MAP or event_type in view.ACTION_TYPE_MAP
        or event_type in ('all_sessions', 'all_delivery', 'all_item_list')
        or direct_stock in stock_types or direct_action in action_types
    )
    if not valid:
        event_type = ''
    sources = []
    if event_type in view.LOGIN_TYPES:
        queryset = LoginAudit.objects.select_related('user').all()
        if user_filter:
            queryset = queryset.filter(username__icontains=user_filter)
        if event_type in ('login_success', 'login_failed'):
            queryset = queryset.filter(success=event_type == 'login_success')
        sources.append(('login', queryset))
    if event_type in view.STOCK_TYPES or event_type in view.STOCK_TYPE_MAP or direct_stock in stock_types:
        queryset = StockChange.objects.select_related('product', 'user').all()
        if user_filter:
            queryset = queryset.filter(user__username__icontains=user_filter)
        if direct_stock in stock_types:
            queryset = queryset.filter(change_type=direct_stock)
        elif event_type in view.STOCK_TYPE_MAP:
            queryset = queryset.filter(change_type__in=view.STOCK_TYPE_MAP[event_type])
        sources.append(('stock', queryset))
    if event_type in view.ACTION_TYPES or event_type in view.ACTION_TYPE_MAP or event_type in ('all_sessions', 'all_delivery', 'all_item_list') or direct_action in action_types:
        queryset = UserAction.objects.select_related('user').all()
        if user_filter:
            queryset = queryset.filter(user__username__icontains=user_filter)
        if direct_action in action_types:
            queryset = queryset.filter(action=direct_action)
        elif event_type in view.ACTION_TYPE_MAP:
            queryset = queryset.filter(action__in=view.ACTION_TYPE_MAP[event_type])
        elif event_type == 'all_sessions':
            queryset = queryset.filter(action__in=view.SESSION_ACTIONS)
        elif event_type == 'all_delivery':
            queryset = queryset.filter(action__in=view.DELIVERY_ACTIONS)
        elif event_type == 'all_item_list':
            queryset = queryset.filter(action__in=view.ACTION_TYPE_MAP['item_list_ops'])
        sources.append(('action', queryset))
    filtered = []
    for source, queryset in sources:
        if date_from:
            queryset = queryset.filter(timestamp__date__gte=date_from)
        if date_to:
            queryset = queryset.filter(timestamp__date__lte=date_to)
        filtered.append((source, queryset))
    return ActivityHistory(filtered, view.SESSION_ACTIONS, view.DELIVERY_ACTIONS)


class ActivityHistory:
    """A paginator-compatible sequence that hydrates only the requested events.

    The UNION contains lightweight references from each source. HTML pages fetch
    one slice; exports iterate bounded batches without a per-source history cap.
    """

    ordered = True

    def __init__(self, sources, session_actions, delivery_actions):
        self.sources = dict(sources)
        self.session_actions = session_actions
        self.delivery_actions = delivery_actions
        references = [
            queryset.order_by().annotate(
                event_source=Value(source, output_field=CharField()),
                event_id=F('pk'),
            ).values('timestamp', 'event_source', 'event_id')
            for source, queryset in sources
        ]
        self.references = references[0].union(*references[1:], all=True).order_by(
            '-timestamp', 'event_source', '-event_id',
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
        for source, queryset in self.sources.items():
            ids = [row['event_id'] for row in references if row['event_source'] == source]
            if ids:
                records[source] = queryset.in_bulk(ids)
        actions = records.get('action', {}).values()
        product_names = {record.target for record in actions if record.action in {
            'add_product', 'edit_product', 'update_product_settings',
        }}
        products = dict(Product.objects.filter(name__in=product_names).values_list('name', 'pk'))
        session_ids = {
            int(match.group(1)) for record in actions
            if record.action in self.session_actions
            and (match := re.search(r'#(\d+)', record.target))
        }
        sessions = set(CheckinSession.objects.filter(pk__in=session_ids).values_list('pk', flat=True))
        events = []
        for ref in references:
            source = ref['event_source']
            record = records.get(source, {}).get(ref['event_id'])
            if record is None:
                continue
            event = getattr(self, f'_{source}_event')(record, products, sessions)
            event['timestamp'] = record.timestamp
            events.append(event)
        return events

    @staticmethod
    def _login_event(record, products, sessions):
        return {
            'category': 'Login', 'user': record.username,
            'action': 'Login Success' if record.success else 'Login Failed',
            'detail': f'IP: {record.ip_address or "unknown"}',
            'badge': 'success' if record.success else 'failed', 'link': '',
        }

    @staticmethod
    def _stock_event(record, products, sessions):
        note = record.staff_note
        if record.change_type == 'lot_reassignment':
            detail = f'Session #{record.session_id}; {note}' if record.session_id and note else note
            detail = detail or f'Quantity: {record.quantity}'
        else:
            detail = f'Quantity: {record.quantity}'
            if note:
                detail += f'; {note}'
        badge = {
            'checkin': 'checkin', 'error_add': 'checkin', 'checkout': 'checkout',
            'expired': 'expired', 'deletion': 'deletion',
        }.get(record.change_type, 'other')
        link = ''
        if record.product_id and record.product and not record.product.archived_at:
            link = reverse('product_details', args=[record.product_id])
        return {
            'category': 'Stock', 'user': record.user.username if record.user else '—',
            'action': record.get_change_type_display(),
            'detail': f'{record.display_name} — {detail}', 'badge': badge, 'link': link,
        }

    def _action_event(self, record, products, sessions):
        action = record.action
        badge = 'other'
        if any(word in action for word in ('delete', 'clear', 'remove')):
            badge = 'deletion'
        elif action == 'submit_order':
            badge = 'checkout'
        elif action in ('add_product', 'create_account'):
            badge = 'checkin'
        elif action in self.session_actions:
            badge = 'session'
        elif action in self.delivery_actions:
            badge = 'delivery'
        category = 'Session' if action in self.session_actions else 'Delivery' if action in self.delivery_actions else 'Action'
        link = ''
        match = re.search(r'#(\d+)', record.target)
        if action == 'submit_order' and match:
            link = reverse('order_detail', args=[int(match.group(1))])
        elif action in ('add_product', 'edit_product', 'update_product_settings') and record.target in products:
            link = reverse('product_details', args=[products[record.target]])
        elif action in self.session_actions and match and int(match.group(1)) in sessions:
            link = reverse('checkin_session_detail', args=[int(match.group(1))])
        elif action in self.delivery_actions and action != 'delivery_clear_history':
            link = reverse('delivery')
        detail = display_lot_text(record.target)
        if record.detail:
            detail += f' — {display_lot_text(record.detail)}'
        return {
            'category': category, 'user': record.user.username if record.user else '—',
            'action': record.get_action_display(), 'detail': detail,
            'badge': badge, 'link': link,
        }
