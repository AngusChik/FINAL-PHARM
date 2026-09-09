"""Read-only data for the Daily Report page.

Sales and ledger activity belong to the selected day. Inventory is deliberately
current: Product and ProductLot are live balances, not historical snapshots.
Money uses the same immutable transaction and correction rules as analytics.
"""

from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from django.db.models import Count, Exists, OuterRef, Q, Sum
from django.db.models.functions import Abs, TruncDate
from django.utils import timezone

from . import reporting
from .models import OrderDetail, Product, ProductLot, StockChange


ZERO = Decimal('0.00')
CENT = Decimal('0.01')
STOCK_LIST_LIMIT = 25
ACTIVITY_LIMIT = 30


def _sales_totals(rows):
    revenue = sum((row['revenue'] for row in rows), ZERO)
    cost = sum((row['cost'] for row in rows), ZERO)
    orders = len({row['order'].pk for row in rows if row['units'] > 0})
    return {
        'orders_today': orders,
        'revenue_today': revenue,
        'units_sold': sum(row['units'] for row in rows),
        'cost': cost,
        'profit': revenue - cost,
        'margin_pct': round((revenue - cost) / revenue * 100, 1) if revenue else None,
        'average_order': (revenue / orders).quantize(CENT) if orders else ZERO,
        # Non-restocked returns still consume inventory cost, even at zero
        # realized units. Missing snapshots must stay visible in that case.
        'missing_cost_units': sum(
            row['line'].realized_cost_quantity
            for row in rows if row['line'].cost_per_unit_at_sale is None
        ),
    }


def _comparison(current, baseline, day):
    delta = current - baseline
    return {
        'day': day,
        'revenue': baseline,
        'delta': delta,
        'pct': round(delta / baseline * 100, 1) if baseline else None,
        'direction': 'up' if delta > 0 else 'down' if delta < 0 else 'flat',
    }


def _product_sales(rows):
    products = {}
    for row in rows:
        line = row['line']
        # IDs disambiguate products sharing a name; deleted products retain
        # their immutable identity instead of collapsing into one null ID.
        key = (
            ('product', line.product_id) if line.product_id is not None
            else ('deleted', line.product_name, line.product_barcode)
        )
        product = products.setdefault(key, {
            'product_id': line.product_id,
            'can_open_product': bool(line.product and line.product.archived_at is None),
            'name': line.product_name,
            'barcode': line.product_barcode,
            'units': 0,
            'revenue': ZERO,
            'profit': ZERO,
        })
        product['units'] += row['units']
        product['revenue'] += row['revenue']
        product['profit'] += row['revenue'] - row['cost']
    return sorted(
        (item for item in products.values() if item['units'] or item['profit']),
        key=lambda item: (-item['revenue'], -item['units'], item['name'], item['barcode']),
    )


def _category_sales(rows, revenue):
    categories = {}
    for row in rows:
        product = row['line'].product
        name = product.category.name if product and product.category else 'Uncategorized'
        category = categories.setdefault(name, {
            'name': name, 'units': 0, 'revenue': ZERO, 'profit': ZERO,
        })
        category['units'] += row['units']
        category['revenue'] += row['revenue']
        category['profit'] += row['revenue'] - row['cost']
    result = []
    for item in categories.values():
        if not (item['units'] or item['profit']):
            continue
        item['share_pct'] = round(item['revenue'] / revenue * 100, 1) if revenue else ZERO
        result.append(item)
    return sorted(result, key=lambda item: (-item['revenue'], item['name']))


def _stock_lists(exclude_snacks):
    low_qs = reporting._low_stock_qs(exclude_snacks).select_related('category').order_by(
        'quantity_in_stock', 'name', 'product_id',
    )
    out_qs = reporting._drop_snacks(
        Product.objects.filter(status=True, quantity_in_stock=0), exclude_snacks,
    ).select_related('category').order_by('name', 'product_id')
    low = {'count': low_qs.count(), 'items': [{
        'product_id': product.pk,
        'name': product.name,
        'barcode': product.barcode or '',
        'quantity_in_stock': product.quantity_in_stock,
        'threshold': product.category.low_stock_threshold if product.category else reporting.LOW_STOCK_DEFAULT,
    } for product in low_qs[:STOCK_LIST_LIMIT]]}
    out = {'count': out_qs.count(), 'items': [{
        'product_id': product.pk,
        'name': product.name,
        'barcode': product.barcode or '',
        'quantity_in_stock': product.quantity_in_stock,
        'category_name': product.category.name if product.category else '',
    } for product in out_qs[:STOCK_LIST_LIMIT]]}
    return low, out


def _expiry_lists(inventory_day, exclude_snacks):
    """Each row is a stocked lot, or one legacy balance without active lots."""
    horizon = inventory_day + timedelta(days=7)
    products = reporting._drop_snacks(Product.objects.filter(status=True), exclude_snacks)
    lots = ProductLot.objects.filter(
        product_id__in=products.values('pk'),
        archived_at__isnull=True,
        quantity_on_hand__gt=0,
        expiry_date__lte=horizon,
    ).select_related('product')
    # Any active lot makes lots authoritative, even if it is depleted or has
    # no expiry. Do not revive a stale Product.expiry_date summary in that case.
    legacy = products.annotate(has_active_lots=Exists(
        ProductLot.objects.filter(product_id=OuterRef('pk'), archived_at__isnull=True),
    )).filter(
        has_active_lots=False, quantity_in_stock__gt=0, expiry_date__lte=horizon,
    )
    rows = [{
        'product_id': lot.product_id,
        'name': lot.product.name,
        'barcode': lot.product.barcode or '',
        'lot_number': lot.staff_name,
        'expiry_date': lot.expiry_date,
        'days_left': (lot.expiry_date - inventory_day).days,
        'quantity_in_stock': lot.quantity_on_hand,
        'source': 'lot',
    } for lot in lots]
    rows.extend({
        'product_id': product.pk,
        'name': product.name,
        'barcode': product.barcode or '',
        'lot_number': '',
        'expiry_date': product.expiry_date,
        'days_left': (product.expiry_date - inventory_day).days,
        'quantity_in_stock': product.quantity_in_stock,
        'source': 'legacy',
    } for product in legacy)
    rows.sort(key=lambda row: (row['expiry_date'], row['name'], row['lot_number'], row['product_id']))
    expired = [row for row in rows if row['days_left'] < 0]
    upcoming = [row for row in rows if row['days_left'] >= 0]
    return {'count': len(expired), 'items': expired}, {'count': len(upcoming), 'items': upcoming}


def _activity_row(change):
    timestamp = change.timestamp
    if timezone.is_aware(timestamp):
        timestamp = timezone.localtime(timestamp)
    return {
        'product_id': change.product_id,
        'can_open_product': bool(change.product and change.product.archived_at is None),
        'name': change.display_name,
        'barcode': change.display_barcode,
        'action': change.get_change_type_display(),
        'change_type': change.change_type,
        'qty': change.quantity,
        'user': change.user.get_username() if change.user else '',
        'note': change.staff_note,
        'time': timestamp.strftime('%H:%M'),
    }


def _daily_activity(day, exclude_snacks):
    base = reporting._drop_snacks(
        StockChange.objects.filter(timestamp__date=day), exclude_snacks, prefix='product__',
    )
    correction_filter = Q(change_type__in=reporting.STOCK_CORRECTION_TYPES)
    counts = base.aggregate(
        count=Count('pk'),
        checkin_count=Count('pk', filter=Q(change_type='checkin')),
        checkin_units=Sum(Abs('quantity'), filter=Q(change_type='checkin')),
        correction_count=Count('pk', filter=correction_filter),
        expired_count=Count('pk', filter=Q(change_type='expired')),
        expired_units=Sum(Abs('quantity'), filter=Q(change_type='expired')),
    )
    counts['checkin_units'] = counts['checkin_units'] or 0
    counts['expired_units'] = counts['expired_units'] or 0
    ordered = base.select_related('product', 'user').order_by('-timestamp', '-pk')
    activity = {**counts, 'items': [_activity_row(change) for change in ordered[:ACTIVITY_LIMIT]]}
    corrections = {
        'correction_count': counts['correction_count'],
        'expired_count': counts['expired_count'],
        'corrections': [_activity_row(change) for change in ordered.filter(correction_filter)[:ACTIVITY_LIMIT]],
        'expired_today': [_activity_row(change) for change in ordered.filter(change_type='expired')[:ACTIVITY_LIMIT]],
    }
    return activity, corrections


def build_daily_report(day=None, exclude_snacks=False):
    """Assemble selected-day operations and explicitly current inventory.

    The legacy daily_digest keys remain available for shared PDF consumers.
    Historical sales are correction-aware as of now, matching sales analytics;
    stored PDF archives remain the point-in-time record of an earlier report.
    """
    selected_day = day or date.today()
    inventory_day = date.today()
    start = selected_day - timedelta(days=7)
    lines = reporting.realized_sales_lines(reporting._drop_snacks(
        OrderDetail.objects.filter(
            order__submitted=True,
            order__order_date__date__range=(start, selected_day),
        ), exclude_snacks, prefix='product__',
    )).annotate(report_day=TruncDate('order__order_date')).select_related(
        'order', 'product__category',
    ).order_by('order_id', 'pk')
    # Include zero-revenue rows: a non-restocked return still carries cost.
    rows = reporting.settled_realized_sales_rows(
        lines, preserve_full_snapshot=not exclude_snacks,
    )
    by_day = defaultdict(list)
    for row in rows:
        by_day[row['line'].report_day].append(row)
    totals = {start + timedelta(days=offset): _sales_totals(by_day[start + timedelta(days=offset)])
              for offset in range(8)}
    sales = totals[selected_day]
    trend_days = [selected_day - timedelta(days=offset) for offset in reversed(range(7))]
    max_revenue = max((totals[trend_day]['revenue_today'] for trend_day in trend_days), default=ZERO)
    trend = [{
        'day': trend_day,
        'date': trend_day.isoformat(),
        'label': trend_day.strftime('%a %d'),
        'revenue': totals[trend_day]['revenue_today'],
        'orders': totals[trend_day]['orders_today'],
        'units': totals[trend_day]['units_sold'],
        'height_pct': round(totals[trend_day]['revenue_today'] / max_revenue * 100, 1) if max_revenue else ZERO,
    } for trend_day in trend_days]
    low, out = _stock_lists(exclude_snacks)
    expired, expiring = _expiry_lists(inventory_day, exclude_snacks)
    dead = reporting.dead_stock(inventory_day, exclude_snacks=exclude_snacks)
    dead['lookback_days'] = 69
    for item in dead['items']:
        item['retail_value'] = item['capital_tied']
    activity, corrections = _daily_activity(selected_day, exclude_snacks)
    movers = _product_sales([row for row in rows if row['line'].report_day >= trend_days[0]])
    movers.sort(key=lambda row: (-row['units'], -row['revenue'], row['name'], row['barcode']))
    return {
        'day': selected_day,
        'inventory_day': inventory_day,
        'exclude_snacks': exclude_snacks,
        'sales': sales,
        'comparisons': {
            name: _comparison(sales['revenue_today'], totals[comparison_day]['revenue_today'], comparison_day)
            for name, comparison_day in (
                ('previous_day', selected_day - timedelta(days=1)),
                ('previous_week', start),
            )
        },
        'trend': trend,
        'top_products': _product_sales(by_day[selected_day])[:8],
        'categories': _category_sales(by_day[selected_day], sales['revenue_today']),
        'stock_health': {
            'total_products': reporting._drop_snacks(Product.objects.filter(status=True), exclude_snacks).count(),
            'out_of_stock_count': out['count'],
            'low_stock_count': low['count'],
            'expiring_soon_count': expiring['count'],
            'expired_count': expired['count'],
        },
        'inventory': reporting.inventory_valuation(inventory_day, exclude_snacks),
        'top_movers': [{
            'product_id': row['product_id'],
            'product_name': row['name'],
            'product_barcode': row['barcode'],
            'total_qty': row['units'],
        } for row in movers if row['units'] > 0][:5],
        'low_stock': low,
        'out_of_stock': out,
        'expired_stock': expired,
        'expiring_week': expiring,
        'dead_stock': dead,
        'activity': activity,
        'corrections': corrections,
    }
