"""Browse retained workflow histories and their individual records."""

from urllib.parse import urlencode, urlsplit

from django.core.paginator import Paginator
from django.http import Http404, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, render
from django.urls import Resolver404, resolve, reverse
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.utils.http import url_has_allowed_host_and_scheme
from django.views import View

from .activity_history import build_activity_history
from .history_data import (
    HISTORY_CATEGORIES, get_history_queryset, get_history_record,
    history_detail, history_row,
)
from .mixins import AdminRequiredMixin
from .models import LoginAudit, StockChange, UserAction, display_lot_text
from .views import ActivityLogView, preferred_table_page_size


CATEGORIES = [
    {'key': 'activity', 'label': 'Activity',
     'description': 'Logins, stock changes, and recorded staff actions.'},
    *HISTORY_CATEGORIES,
    {'key': 'recovery', 'label': 'Archived records',
     'description': 'Records removed from working pages and retained in Recovery.'},
]


def history_dates(request):
    dates = []
    errors = []
    for key, label in (('date_from', 'From'), ('date_to', 'To')):
        raw = request.GET.get(key, '')
        try:
            value = parse_date(raw) if raw else None
        except (ValueError, OverflowError):
            value = None
        if raw and (value is None or value.year < 1900):
            errors.append(f'{label}: enter a valid date from 1900 onward.')
        dates.append(value)
    if all(dates) and dates[0] > dates[1]:
        errors.append('The From date must be on or before the To date.')
    return (*dates, errors)


def history_list_url(kind, params=None):
    values = {'kind': kind, **(params or {})}
    return reverse('history') + '?' + urlencode({k: v for k, v in values.items() if v})


def safe_history_return(request, kind):
    raw = request.GET.get('return_to', '')
    if raw and url_has_allowed_host_and_scheme(
        raw, allowed_hosts={request.get_host()}, require_https=request.is_secure(),
    ):
        try:
            path = urlsplit(raw).path
            if path.startswith('/') and resolve(path).url_name in {'history', 'activity_log'}:
                return raw
        except (Resolver404, ValueError):
            pass
    return history_list_url(kind)


class HistoryView(AdminRequiredMixin, View):
    template_name = 'activity_log.html'

    def get(self, request):
        kind = request.GET.get('kind', 'activity')
        category = next((item for item in CATEGORIES if item['key'] == kind), None)
        if category is None:
            raise Http404('Unknown history type.')
        date_from, date_to, errors = history_dates(request)
        query = request.GET.get('q', '').strip()[:200]
        user_filter = request.GET.get('user', '').strip()[:150]
        event_type = request.GET.get('type', '') if kind == 'activity' else ''
        legacy = ActivityLogView()
        if kind == 'activity':
            records = [] if errors else build_activity_history(
                legacy, event_type, user_filter, date_from, date_to, query=query,
            )
            if request.GET.get('export') == 'pdf':
                if errors:
                    return HttpResponseBadRequest(' '.join(errors))
                return legacy._render_pdf(
                    records, event_type, user_filter,
                    request.GET.get('date_from', ''), request.GET.get('date_to', ''),
                )
        elif kind == 'recovery':
            from .history_recovery import get_recovery_history
            records = [] if errors else get_recovery_history(request)
        else:
            records = [] if errors else get_history_queryset(kind, request)

        page = Paginator(records, preferred_table_page_size(request, 50)).get_page(request.GET.get('page'))
        params = {
            'q': query, 'user': user_filter,
            'date_from': request.GET.get('date_from', ''),
            'date_to': request.GET.get('date_to', ''),
            'type': event_type,
            'archive_type': request.GET.get('archive_type', '') if kind == 'recovery' else '',
        }
        return_url = history_list_url(kind, {**params, 'page': page.number})
        rows = []
        for record in page:
            if kind == 'activity':
                row = {
                    'id': record['id'], 'kind': record['source'],
                    'timestamp': record['timestamp'], 'title': record['action'],
                    'summary': record['detail'], 'user': record['user'],
                    'status': record['category'],
                }
            elif kind == 'recovery':
                row = dict(record)
            else:
                row = history_row(kind, record)
            row['url'] = reverse('history_detail', args=[row.get('kind', kind), row['id']]) + '?' + urlencode({'return_to': return_url})
            rows.append(row)

        tabs = []
        for item in CATEGORIES:
            tab_params = {key: value for key, value in params.items() if key not in {'type', 'archive_type'}}
            if item.get('user_filter') is False:
                tab_params.pop('user', None)
            tabs.append({**item, 'url': history_list_url(item['key'], tab_params)})
        filter_params = {key: value for key, value in {'kind': kind, **params}.items() if value}
        filter_query = urlencode(filter_params)
        if kind == 'recovery':
            from .views import ArchiveRecoveryView
            archive_types = ArchiveRecoveryView.TYPE_LABELS.items()
        else:
            archive_types = []
        return render(request, self.template_name, {
            'category': category, 'categories': tabs, 'kind': kind,
            'rows': rows, 'page_obj': page,
            'query': query, 'user_filter': user_filter, 'event_type': event_type,
            'date_from': params['date_from'], 'date_to': params['date_to'],
            'filter_errors': errors, 'filter_query': filter_query,
            'has_filters': any(params.values()), 'clear_url': history_list_url(kind),
            'archive_types': archive_types, 'archive_type': params['archive_type'],
            'export_url': reverse('history') + '?' + filter_query + '&export=pdf',
            'stock_event_choices': [('stock:' + key, label) for key, label in StockChange.CHANGE_TYPE_CHOICES],
            'action_event_choices': [('action:' + key, label) for key, label in UserAction.ACTION_CHOICES],
            'legacy_event_filter': legacy._filter_label(event_type) if event_type and not event_type.startswith(('stock:', 'action:')) else '',
        })


class HistoryDetailView(AdminRequiredMixin, View):
    def get(self, request, kind, pk):
        if kind in {'login', 'action'}:
            model = LoginAudit if kind == 'login' else UserAction
            record = get_object_or_404(model.objects.select_related('user'), pk=pk)
            stamp = timezone.localtime(record.timestamp).strftime('%b %d, %Y %H:%M')
            if kind == 'login':
                detail = {
                    'title': 'Successful login' if record.success else 'Failed login',
                    'subtitle': f'Login record #{record.pk}',
                    'fields': [
                        {'label': 'Date and time', 'value': stamp},
                        {'label': 'Username', 'value': record.username},
                        {'label': 'Result', 'value': 'Success' if record.success else 'Failed'},
                        {'label': 'IP address', 'value': record.ip_address or 'Not recorded'},
                    ], 'sections': [], 'actions': [],
                }
            else:
                detail = {
                    'title': record.get_action_display(),
                    'subtitle': f'Activity record #{record.pk}',
                    'fields': [
                        {'label': 'Date and time', 'value': stamp},
                        {'label': 'Staff', 'value': record.user.get_username() if record.user else 'Not recorded'},
                        {'label': 'Record', 'value': display_lot_text(record.target) or 'Not recorded'},
                        {'label': 'Details', 'value': display_lot_text(record.detail) or 'No additional details recorded.'},
                    ], 'sections': [], 'actions': [],
                }
            list_kind = 'activity'
        elif kind.startswith('recovery-'):
            from .history_recovery import get_recovery_detail
            detail = get_recovery_detail(kind, pk, request)
            list_kind = 'recovery'
        else:
            record = get_history_record(kind, pk, request)
            detail = history_detail(kind, record, request)
            list_kind = kind
        return_url = safe_history_return(request, list_kind)
        # Existing detail links can return here without discarding the original table filters.
        for action in detail.get('actions', []):
            if action.get('url') and not any(word in action['url'] for word in ('/pdf/', '/regenerate/')):
                separator = '&' if '?' in action['url'] else '?'
                action['url'] += separator + urlencode({'return_to': request.get_full_path()})
        return render(request, 'history_detail.html', {
            'detail': detail, 'history_return': return_url,
            'page_return': {'url': return_url, 'destination': 'History', 'label': 'Back to History', 'source': 'explicit'},
        })
