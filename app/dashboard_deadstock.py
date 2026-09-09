"""Dashboard recommendations and shared 30-day dismissals, separate from reports."""

import json
import random
from datetime import date, timedelta

from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.sessions.models import Session
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Max
from django.http import JsonResponse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.cache import never_cache

from .models import DashboardDeadStockDismissal, Product, StockChange


BATCH_SIZE = 8
DISMISSAL_DAYS = 30
SESSION_KEY = 'dashboard_deadstock_rotation_v1'


class DashboardDeadStockSessionMiddleware:
    """Serialize this endpoint's session queue from initial load through save.

    This must wrap Django's SessionMiddleware: a view-only lock releases before
    that middleware saves and permits another tab to consume the same queue.
    The application uses database-backed sessions, so its existing session row
    provides the lock without a separate queue model or cross-process mutex.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method != 'POST' or request.path_info != '/api/dashboard/deadstock/':
            return self.get_response(request)
        session_key = request.COOKIES.get(settings.SESSION_COOKIE_NAME)
        if not session_key:
            return self.get_response(request)
        with transaction.atomic():
            Session.objects.select_for_update().only('session_key').filter(
                session_key=session_key,
            ).first()
            # SessionMiddleware both initializes and persists the session inside
            # this lock. Concurrent requests then read the committed next queue.
            return self.get_response(request)


def _dismissal_fields(dismissal):
    if dismissal is None:
        return {
            'dismissed': False, 'expires_at': None,
            'dismissed_at': None, 'dismissed_by': '',
        }
    actor = dismissal.dismissed_by
    return {
        'dismissed': True,
        'expires_at': dismissal.expires_at.isoformat(),
        'dismissed_at': dismissal.dismissed_at.isoformat(),
        'dismissed_by': (actor.get_short_name() or actor.get_username()) if actor else '',
    }


def decorate_items(items, *, at=None):
    """Copy reporting rows and add live dismissal metadata, keeping every row."""
    items = list(items)
    active = {
        row.product_id: row
        for row in DashboardDeadStockDismissal.objects.filter(
            product_id__in=[item['product_id'] for item in items],
            expires_at__gt=at or timezone.now(),
        ).select_related('dismissed_by')
    }
    return [dict(item, **_dismissal_fields(active.get(item['product_id']))) for item in items]


def _product_items(products, *, day=None):
    products = list(products)
    today = day or date.today()
    sale_dates = {
        row['product_id']: row['last']
        for row in StockChange.objects.filter(
            product_id__in=[product.pk for product in products],
            change_type='checkout',
        ).values('product_id').annotate(last=Max('timestamp'))
    }
    items = []
    for product in products:
        last_sale = sale_dates.get(product.pk)
        items.append({
            'product_id': product.pk,
            'name': product.name,
            'barcode': product.barcode or '',
            'quantity_in_stock': product.quantity_in_stock,
            'capital_tied': float(product.price * product.quantity_in_stock),
            'days_since_sale': (today - last_sale.date()).days if last_sale else 'Never',
            'category_name': product.category.name if product.category else '',
        })
    return items


def _pool(exclude_snacks=False, exclude_braces=False, *, at=None):
    """Read all candidate IDs without imposing the report's display limit."""
    at = at or timezone.now()
    # Keep the reporting module's local date convention and checkout-only cutoff.
    recently_sold = StockChange.objects.filter(
        change_type='checkout', timestamp__date__gte=date.today() - timedelta(days=69),
        product_id__isnull=False,
    ).order_by().values('product_id')
    base = Product.objects.filter(status=True, quantity_in_stock__gt=0).exclude(
        product_id__in=recently_sold,
    )
    total_count = base.count()
    if exclude_snacks:
        base = base.exclude(category__name__iexact='Snacks')
    if exclude_braces:
        base = base.exclude(category__name__iexact='Braces')
    category_ids = list(base.order_by('product_id').values_list('product_id', flat=True))
    dismissed_ids = set(DashboardDeadStockDismissal.objects.filter(
        expires_at__gt=at,
    ).values_list('product_id', flat=True))
    available_ids = [pk for pk in category_ids if pk not in dismissed_ids]
    return base, available_ids, {
        'count': len(available_ids),
        'available_count': len(available_ids),
        'total_count': total_count,
        'filtered_count': total_count - len(category_ids),
        'dismissed_count': len(dismissed_ids),
        'dismissed_eligible_count': len(set(category_ids) & dismissed_ids),
    }


def _valid_stored_ids(value):
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(pk for pk in value if type(pk) is int and pk > 0))


def _choose_ids(session, available_ids, *, filter_key, current_ids=()):
    """Consume a shuffled session queue, keeping valid displayed rows in place."""
    available = set(available_ids)
    chosen = [pk for pk in current_ids if pk in available][:BATCH_SIZE]
    # Mutations outside a full current batch must not advance its rotation.
    if len(chosen) == BATCH_SIZE:
        return chosen

    states = session.get(SESSION_KEY, {})
    if not isinstance(states, dict):
        states = {}
    state = states.get(filter_key, {})
    if not isinstance(state, dict):
        state = {}
    known = set(_valid_stored_ids(state.get('known')))
    remaining = [pk for pk in _valid_stored_ids(state.get('remaining')) if pk in available]
    remaining_set = set(remaining)
    newly_eligible = [pk for pk in available_ids if pk not in known and pk not in remaining_set]
    random.shuffle(newly_eligible)
    remaining.extend(newly_eligible)
    previous_batch = set(_valid_stored_ids(state.get('last')))

    while len(chosen) < min(BATCH_SIZE, len(available_ids)):
        remaining = [pk for pk in remaining if pk not in chosen]
        if not remaining:
            remaining = [pk for pk in available_ids if pk not in chosen]
            random.shuffle(remaining)
            # At a cycle boundary, prefer products not in the previous batch.
            remaining.sort(key=lambda pk: pk in previous_batch)
        needed = min(BATCH_SIZE - len(chosen), len(remaining))
        chosen.extend(remaining[:needed])
        remaining = remaining[needed:]

    states[filter_key] = {
        'known': available_ids, 'remaining': remaining, 'last': chosen,
    }
    session[SESSION_KEY] = states
    return chosen


def _batch(request, *, exclude_snacks=False, exclude_braces=False, current_ids=(), at=None):
    at = at or timezone.now()
    base, available_ids, counts = _pool(exclude_snacks, exclude_braces, at=at)
    selected_ids = _choose_ids(
        request.session, available_ids,
        filter_key=f'{int(exclude_snacks)}{int(exclude_braces)}',
        current_ids=current_ids,
    )
    if current_ids:
        # Refill missing slots in place so dismissing row two does not move rows
        # three through eight. Remaining additions append to a short batch.
        selected = set(selected_ids)
        retained = set(current_ids) & selected
        replacements = iter(pk for pk in selected_ids if pk not in retained)
        reordered = []
        for pk in current_ids:
            replacement = pk if pk in retained else next(replacements, None)
            if replacement is not None:
                reordered.append(replacement)
        selected_ids = reordered + list(replacements)
    products = {
        product.pk: product for product in base.filter(
            product_id__in=selected_ids,
        ).select_related('category')
    }
    items = _product_items(products[pk] for pk in selected_ids if pk in products)
    items = [dict(item, **_dismissal_fields(None)) for item in items]
    return {'ok': True, 'items': items, **counts}


def _positive_id(value):
    return type(value) is int and 0 < value <= 2147483647


def _parse_post(request):
    try:
        payload = json.loads(request.body or '{}')
    except (ValueError, UnicodeDecodeError):
        raise ValueError('Invalid request.') from None
    if not isinstance(payload, dict):
        raise ValueError('Invalid request.')
    if payload.get('action') not in ('next', 'dismiss', 'restore'):
        raise ValueError('Unknown action.')
    for key in ('exclude_snacks', 'exclude_braces'):
        if type(payload.get(key, False)) is not bool:
            raise ValueError('Invalid category filter.')
    current_ids = payload.get('current_ids', [])
    if not isinstance(current_ids, list) or len(current_ids) > BATCH_SIZE:
        raise ValueError('Invalid current product list.')
    if not all(_positive_id(pk) for pk in current_ids):
        raise ValueError('Invalid current product list.')
    payload['current_ids'] = list(dict.fromkeys(current_ids))
    if payload['action'] != 'next' and not _positive_id(payload.get('product_id')):
        raise ValueError('Invalid product.')
    return payload


@method_decorator(never_cache, name='dispatch')
class DashboardDeadStockAPIView(LoginRequiredMixin, View):
    """Shared dismissal controls with independent recommendation rotation per session."""

    def get(self, request):
        filters = {}
        for key in ('exclude_snacks', 'exclude_braces'):
            value = request.GET.get(key, '0')
            if value not in ('0', '1', 'false', 'true'):
                return JsonResponse({'ok': False, 'error': 'Invalid category filter.'}, status=400)
            filters[key] = value in ('1', 'true')
        at = timezone.now()
        _, _, counts = _pool(**filters, at=at)
        active = DashboardDeadStockDismissal.objects.filter(expires_at__gt=at).select_related(
            'product__category', 'dismissed_by',
        ).order_by('expires_at', 'pk')
        page = Paginator(active, 25).get_page(request.GET.get('page', 1))
        dismissals = list(page.object_list)
        rows = _product_items(row.product for row in dismissals)
        items = [dict(item, **_dismissal_fields(row)) for item, row in zip(rows, dismissals)]
        return JsonResponse({
            'ok': True, **counts, 'items': items, 'count': page.paginator.count,
            'page': page.number, 'num_pages': page.paginator.num_pages,
            'has_previous': page.has_previous(), 'has_next': page.has_next(),
        })

    def post(self, request):
        try:
            payload = _parse_post(request)
        except ValueError as exc:
            return JsonResponse({'ok': False, 'error': str(exc)}, status=400)
        filters = {
            key: payload.get(key, False)
            for key in ('exclude_snacks', 'exclude_braces')
        }
        if payload['action'] == 'next':
            return JsonResponse(_batch(request, **filters))

        with transaction.atomic():
            # Lock the durable parent even when the one-to-one row does not exist.
            product = Product.all_objects.select_for_update().filter(
                pk=payload['product_id'],
            ).first()
            if product is None:
                return JsonResponse({'ok': False, 'error': 'Product not found.'}, status=404)
            at = timezone.now()
            dismissal = DashboardDeadStockDismissal.objects.filter(product=product).first()
            if payload['action'] == 'dismiss':
                if dismissal is None:
                    dismissal = DashboardDeadStockDismissal.objects.create(
                        product=product, dismissed_by=request.user,
                        dismissed_at=at, expires_at=at + timedelta(days=DISMISSAL_DAYS),
                    )
                elif dismissal.expires_at <= at:
                    dismissal.dismissed_by = request.user
                    dismissal.dismissed_at = at
                    dismissal.expires_at = at + timedelta(days=DISMISSAL_DAYS)
                    dismissal.save(update_fields=['dismissed_by', 'dismissed_at', 'expires_at'])
            elif dismissal is not None and dismissal.expires_at > at:
                dismissal.expires_at = at
                dismissal.save(update_fields=['expires_at'])

            response = _batch(request, **filters, current_ids=payload['current_ids'], at=at)
            is_dismissed = dismissal is not None and dismissal.expires_at > at
            response.update({
                'product_id': product.pk, 'dismissed': is_dismissed,
                'expires_at': dismissal.expires_at.isoformat() if is_dismissed else None,
            })
            return JsonResponse(response)
