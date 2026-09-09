"""Centralized reporting / rollup helpers.

Single source of truth for the aggregate queries that the dashboard and the
daily end-of-day report need, so they aren't re-derived per view. Pure functions
that return plain dicts / lists (and a couple of querysets where the template
needs model objects).

Timezone note: the rest of the app filters on naive ``date.today()`` against
``timestamp__date`` / ``order_date__date``. We keep that convention here so the
numbers match the dashboard and the existing PDF views exactly — do NOT switch
to ``timezone.localdate()`` selectively or the rollups drift from everything else.
"""

import io
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from django.db.models import (
    Sum, F, Q, Count, Value, DecimalField, IntegerField, Max, Case, When,
    ExpressionWrapper, OuterRef, Subquery,
)
from django.db.models.functions import (
    Coalesce,
    Greatest,
    Round,
    TruncDate,
    TruncWeek,
)

from .models import (
    Order, Product, OrderDetail, StockChange, OrderingSheetEntry,
    TransactionCorrectionLine, RecentlyPurchasedProduct,
)
from .utils import (
    TAX_RATE,
    allocate_order_line_financials,
    calculate_order_financials_from_values,
    get_reorder_prediction,
)

LOW_STOCK_DEFAULT = 3
SALE_TYPES = ['checkout', 'checkout_unfulfilled']
STOCK_CORRECTION_TYPES = [
    'error_add',
    'error_subtract',
    'return',
    'return_no_restock',
    'void',
    'correction_undo',
    'restoration',
]

# Reports can optionally exclude this product category ("ignore snacks").
SNACKS_CATEGORY_NAME = 'Snacks'


def _resolve_day(day=None):
    return day or date.today()


def _drop_snacks(qs, exclude_snacks, prefix=''):
    """Exclude Snacks-category rows when the report opts in.

    prefix='' for Product querysets; 'product__' for OrderDetail/StockChange
    querysets (which reach the category through their product FK).
    """
    if not exclude_snacks:
        return qs
    return qs.exclude(**{f'{prefix}category__name__iexact': SNACKS_CATEGORY_NAME})


def _sale_revenue_expression():
    """Immutable pre-tax line revenue after any order-wide seniors discount."""
    return F('price') * F('realized_quantity') * Case(
        When(order__seniors_discount=True, then=Value(Decimal('0.90'))),
        default=Value(Decimal('1.00')),
        output_field=DecimalField(max_digits=4, decimal_places=2),
    )


REALIZED_MONEY_FIELD = DecimalField(max_digits=14, decimal_places=2)


def _active_corrected_quantity(*, restocked_only=False):
    """Correlated quantity removed from one sale line by active corrections.

    Corrections and their undo records are append-only. An undo therefore makes
    the associated correction inactive without changing either historical row.
    """
    corrections = TransactionCorrectionLine.objects.filter(
        order_detail_id=OuterRef('pk'),
        correction__undo__isnull=True,
    )
    if restocked_only:
        corrections = corrections.filter(
            disposition=TransactionCorrectionLine.DISPOSITION_RESTOCK,
        )
    return Subquery(
        corrections.order_by()
        .values('order_detail_id')
        .annotate(total=Sum('quantity'))
        .values('total')[:1],
        output_field=IntegerField(),
    )


def realized_sales_lines(queryset=None):
    """Annotate sale lines with quantities and money still realized.

    Revenue and units are reduced by every active return/void. Cost is reduced
    only when corrected stock was actually returned to inventory; damaged or
    non-restocked returns remain a real inventory cost. All calculations use
    immutable sale-time prices/costs and never rewrite the order snapshot.
    """
    queryset = queryset if queryset is not None else OrderDetail.objects.all()
    queryset = queryset.annotate(
        active_corrected_quantity=Coalesce(
            _active_corrected_quantity(),
            Value(0),
            output_field=IntegerField(),
        ),
        active_restocked_quantity=Coalesce(
            _active_corrected_quantity(restocked_only=True),
            Value(0),
            output_field=IntegerField(),
        ),
    ).annotate(
        realized_quantity=Greatest(
            F('quantity') - F('active_corrected_quantity'),
            Value(0),
            output_field=IntegerField(),
        ),
        realized_cost_quantity=Greatest(
            F('quantity') - F('active_restocked_quantity'),
            Value(0),
            output_field=IntegerField(),
        ),
    )
    return queryset.annotate(
        realized_revenue=ExpressionWrapper(
            _sale_revenue_expression(),
            output_field=REALIZED_MONEY_FIELD,
        ),
        realized_cost=Case(
            When(
                cost_per_unit_at_sale__isnull=False,
                then=ExpressionWrapper(
                    F('cost_per_unit_at_sale') * F('realized_cost_quantity'),
                    output_field=REALIZED_MONEY_FIELD,
                ),
            ),
            default=Value(Decimal('0.00')),
            output_field=REALIZED_MONEY_FIELD,
        ),
    )


def annotate_orders_with_realized_sales(queryset):
    """Add net pre-tax revenue and units to each order in ``queryset``."""
    lines = realized_sales_lines(
        OrderDetail.objects.filter(order_id=OuterRef('pk')),
    ).order_by().values('order_id').annotate(
        total_subtotal=Sum(
            ExpressionWrapper(
                F('price') * F('realized_quantity'),
                output_field=REALIZED_MONEY_FIELD,
            ),
            output_field=REALIZED_MONEY_FIELD,
        ),
        total_units=Sum('realized_quantity'),
        original_units=Sum('quantity'),
    )
    queryset = queryset.annotate(
        realized_subtotal=Coalesce(
            Subquery(
                lines.values('total_subtotal')[:1],
                output_field=REALIZED_MONEY_FIELD,
            ),
            Value(Decimal('0.00')),
            output_field=REALIZED_MONEY_FIELD,
        ),
        realized_units=Coalesce(
            Subquery(
                lines.values('total_units')[:1],
                output_field=IntegerField(),
            ),
            Value(0),
            output_field=IntegerField(),
        ),
        original_units=Coalesce(
            Subquery(
                lines.values('original_units')[:1],
                output_field=IntegerField(),
            ),
            Value(0),
            output_field=IntegerField(),
        ),
    )
    discounted_revenue = ExpressionWrapper(
        F('realized_subtotal') - Round(
            F('realized_subtotal') * Value(Decimal('0.10')),
            precision=2,
        ),
        output_field=REALIZED_MONEY_FIELD,
    )
    return queryset.annotate(
        realized_revenue=Case(
            When(
                Q(
                    financial_snapshot_source__gt='',
                    realized_units=F('original_units'),
                ),
                then=ExpressionWrapper(
                    F('subtotal') - F('discount_amount'),
                    output_field=REALIZED_MONEY_FIELD,
                ),
            ),
            When(seniors_discount=True, then=discounted_revenue),
            default=F('realized_subtotal'),
            output_field=REALIZED_MONEY_FIELD,
        ),
    )


def realized_order_financials(order, lines=None):
    """Return correction-aware order money without rewriting its snapshot.

    ``subtotal`` is the remaining gross line value, ``revenue`` is the
    remaining pre-tax value after the order's seniors discount, and ``total``
    adds tax only for remaining taxable units. Active correction lines reduce
    quantity; an append-only undo makes its correction inactive through
    ``realized_sales_lines``.
    """
    if lines is None:
        lines = realized_sales_lines(order.details.all()).order_by('pk')
    lines = list(lines)
    quantities = [
        max(0, int(getattr(line, 'realized_quantity', line.quantity)))
        for line in lines
    ]
    is_original_snapshot = bool(
        order.financial_snapshot_source
        and all(quantity == line.quantity for line, quantity in zip(lines, quantities))
    )
    if is_original_snapshot:
        # The captured order values are the amount that actually settled. This
        # also preserves legacy snapshots whose rounding policy may differ.
        subtotal = Decimal(order.subtotal)
        discount_amount = Decimal(order.discount_amount)
        tax = Decimal(order.tax)
        total = Decimal(order.total_price)
    else:
        values = calculate_order_financials_from_values(
            (
                (line.price, quantity, line.taxable_at_sale)
                for line, quantity in zip(lines, quantities)
            ),
            seniors_discount=order.seniors_discount,
            tax_rate=(
                order.tax_rate
                if order.financial_snapshot_source
                else TAX_RATE
            ),
        )
        subtotal = values['subtotal']
        discount_amount = values['discount_amount']
        tax = values['tax']
        total = values['total']
    revenue = subtotal - discount_amount
    return {
        'subtotal': subtotal,
        'discount_amount': discount_amount,
        'tax': tax,
        'total': total,
        'revenue': revenue,
        'units': sum(quantities),
    }


def settled_realized_sales_rows(lines, *, preserve_full_snapshot=True):
    """Return cent-settled revenue/cost rows for correction-aware analytics.

    ``realized_sales_lines`` is intentionally an efficient quantity/cost
    queryset. Seniors discount cents, however, must be settled once per order.
    This helper groups the selected rows, calculates each remaining basket, and
    deterministically allocates its discount so product/category rollups still
    add back to the exact order revenue. When ``preserve_full_snapshot`` is
    true, callers must supply the complete selected line set for each order;
    subset reports should pass false and intentionally recalculate that subset.
    """
    grouped = defaultdict(list)
    for line in lines:
        grouped[line.order_id].append(line)
    corrected_order_ids = set()
    if preserve_full_snapshot and grouped:
        corrected_order_ids = set(
            TransactionCorrectionLine.objects.filter(
                order_detail__order_id__in=grouped,
                correction__undo__isnull=True,
            ).values_list('order_detail__order_id', flat=True)
        )

    settled = []
    for order_lines in grouped.values():
        order_lines.sort(key=lambda line: line.pk)
        order = order_lines[0].order
        quantities = [
            max(0, int(getattr(line, 'realized_quantity', line.quantity)))
            for line in order_lines
        ]
        use_snapshot = bool(
            preserve_full_snapshot
            and order.financial_snapshot_source
            and order.pk not in corrected_order_ids
            and all(
                quantity == line.quantity
                for line, quantity in zip(order_lines, quantities)
            )
        )
        if use_snapshot:
            values = {
                'discount_amount': Decimal(order.discount_amount),
                'tax': Decimal(order.tax),
            }
        else:
            values = calculate_order_financials_from_values(
                (
                    (line.price, quantity, line.taxable_at_sale)
                    for line, quantity in zip(order_lines, quantities)
                ),
                seniors_discount=order.seniors_discount,
                tax_rate=(
                    order.tax_rate
                    if order.financial_snapshot_source
                    else TAX_RATE
                ),
            )
        allocations = allocate_order_line_financials(
            [line.price * quantity for line, quantity in zip(order_lines, quantities)],
            [line.taxable_at_sale is True for line in order_lines],
            values['discount_amount'],
            values['tax'],
        )
        for line, quantity, allocation in zip(
                order_lines, quantities, allocations):
            settled.append({
                'line': line,
                'order': order,
                'units': quantity,
                'revenue': allocation['net'],
                'cost': Decimal(getattr(line, 'realized_cost', 0) or 0),
            })
    return settled


def _low_stock_qs(exclude_snacks=False):
    """Active, in-stock products at or below their (category) low-stock threshold."""
    qs = (
        Product.objects.filter(status=True, quantity_in_stock__gt=0)
        .annotate(_threshold=Coalesce(F('category__low_stock_threshold'), Value(LOW_STOCK_DEFAULT)))
        .filter(quantity_in_stock__lte=F('_threshold'))
    )
    return _drop_snacks(qs, exclude_snacks)


# ── Individual metric groups ────────────────────────────────────────────────

def stock_health(day=None, exclude_snacks=False):
    today = _resolve_day(day)
    oos = _drop_snacks(Product.objects.filter(status=True, quantity_in_stock=0), exclude_snacks)
    expiring = _drop_snacks(Product.objects.filter(
        expiry_date__gte=today, expiry_date__lte=today + timedelta(days=7)
    ).exclude(expiry_date__isnull=True), exclude_snacks)
    total = _drop_snacks(Product.objects.filter(status=True), exclude_snacks)
    return {
        'out_of_stock_count': oos.count(),
        'low_stock_count': _low_stock_qs(exclude_snacks).count(),
        'expiring_soon_count': expiring.count(),
        'total_products': total.count(),
    }


def sales_summary(day=None, exclude_snacks=False):
    today = _resolve_day(day)
    if not exclude_snacks:
        orders = annotate_orders_with_realized_sales(
            Order.objects.filter(
                order_date__date=today,
                submitted=True,
            ),
        )
        realized_orders = orders.filter(realized_units__gt=0)
        return {
            'orders_today': realized_orders.count(),
            'revenue_today': realized_orders.aggregate(
                total=Sum('realized_revenue'),
            )['total'] or Decimal('0.00'),
            'units_sold': orders.aggregate(
                total=Sum('realized_units'),
            )['total'] or 0,
        }

    lines = realized_sales_lines(
        _drop_snacks(
            OrderDetail.objects.filter(
                order__order_date__date=today,
                order__submitted=True,
            ),
            exclude_snacks,
            prefix='product__',
        ),
    )
    settled_rows = settled_realized_sales_rows(
        lines.filter(realized_quantity__gt=0)
        .select_related('order', 'product__category')
        .order_by('order_id', 'pk'),
        preserve_full_snapshot=False,
    )
    order_ids = {
        row['order'].pk for row in settled_rows if row['units'] > 0
    }
    return {
        'orders_today': len(order_ids),
        'revenue_today': sum(
            (row['revenue'] for row in settled_rows), Decimal('0.00'),
        ),
        'units_sold': sum(row['units'] for row in settled_rows),
    }


def inventory_valuation(day=None, exclude_snacks=False):
    agg = _drop_snacks(Product.objects.filter(status=True), exclude_snacks).aggregate(
        total_units=Sum('quantity_in_stock'),
        total_retail=Sum(F('price') * F('quantity_in_stock')),
        total_cost=Sum(F('price_per_unit') * F('quantity_in_stock')),
    )
    total_retail = agg['total_retail'] or Decimal('0.00')
    total_cost = agg['total_cost'] or Decimal('0.00')
    return {
        'total_units': agg['total_units'] or 0,
        'total_retail': total_retail,
        'total_cost': total_cost,
        'gross_margin_pct': round(((total_retail - total_cost) / total_retail * 100), 1) if total_retail else 0,
    }


def top_movers(day=None, days=7, limit=5, exclude_snacks=False):
    today = _resolve_day(day)
    since = today - timedelta(days=days)
    qs = realized_sales_lines(
        _drop_snacks(
            OrderDetail.objects.filter(
                order__submitted=True,
                order__order_date__date__gte=since,
            ),
            exclude_snacks,
            prefix='product__',
        ),
    )
    return list(
        qs.filter(realized_quantity__gt=0)
        .values('product_name', 'product_barcode').annotate(
            total_qty=Sum('realized_quantity')
        ).order_by('-total_qty')[:limit]
    )


def expiry_buckets(day=None):
    today = _resolve_day(day)

    def _count(lo, hi):
        return Product.objects.filter(
            status=True, expiry_date__range=[today + timedelta(days=lo), today + timedelta(days=hi)]
        ).exclude(expiry_date__isnull=True).count()

    return {'exp_7d': _count(0, 7), 'exp_14d': _count(8, 14), 'exp_30d': _count(15, 30)}


def sales_chart(day=None, days=13):
    today = _resolve_day(day)
    start = today - timedelta(days=days)
    rows = list(
        annotate_orders_with_realized_sales(
            Order.objects.filter(
                submitted=True,
                order_date__date__gte=start,
            ),
        )
        .filter(realized_units__gt=0)
        .annotate(sale_date=TruncDate('order_date'))
        .values('sale_date')
        .annotate(
            daily_revenue=Sum(
                'realized_revenue', output_field=REALIZED_MONEY_FIELD,
            ),
            order_count=Count('pk', distinct=True),
            item_count=Sum('realized_units'),
        )
        .order_by('sale_date')
    )
    return [
        {
            'date': r['sale_date'].strftime('%b %d') if r['sale_date'] else '',
            'full_date': r['sale_date'].strftime('%Y-%m-%d') if r['sale_date'] else '',
            'day': r['sale_date'].strftime('%A') if r['sale_date'] else '',
            'revenue': float(r['daily_revenue'] or 0),
            'orders': r['order_count'],
            'items': r['item_count'],
        }
        for r in rows
    ]


def reorder_suggestions(day=None, limit=10):
    today = _resolve_day(day)
    products = list(
        _low_stock_qs().select_related('category').order_by('quantity_in_stock')[:limit]
    )
    pids = [p.product_id for p in products]
    recent_product_ids = set(RecentlyPurchasedProduct.objects.filter(
        product_id__in=pids, archived_at__isnull=True,
    ).values_list('product_id', flat=True))
    demand_map, weekly_map = {}, defaultdict(list)
    if pids:
        since = today - timedelta(days=60)
        demand_map = {
            r['product_id']: r['total']
            for r in StockChange.objects.filter(
                product_id__in=pids, timestamp__date__gte=since, change_type__in=SALE_TYPES,
            ).values('product_id').annotate(total=Sum('quantity'))
        }
        for r in StockChange.objects.filter(
            product_id__in=pids, timestamp__date__gte=since, change_type__in=SALE_TYPES,
        ).annotate(week=TruncWeek('timestamp')).values('product_id', 'week').annotate(
            total=Sum('quantity')
        ).order_by('product_id', 'week'):
            weekly_map[r['product_id']].append((r['week'], r['total']))

    suggestions = []
    for p in products:
        pred = get_reorder_prediction(
            p, demand_map.get(p.product_id, 0), weekly_demands=weekly_map.get(p.product_id, []),
        )
        suggestions.append({
            'product_id': p.product_id,
            'name': p.name,
            'barcode': p.barcode or '',
            'quantity_in_stock': p.quantity_in_stock,
            'threshold': p.category.low_stock_threshold if p.category else LOW_STOCK_DEFAULT,
            'suggested_qty': pred.get('suggested_qty', 0),
            'urgency': pred.get('urgency', 'ok'),
            'in_recently_purchased': p.product_id in recent_product_ids,
        })
    return suggestions


def dead_stock(day=None, lookback_days=69, limit=8, exclude_snacks=False):
    today = _resolve_day(day)
    cutoff = today - timedelta(days=lookback_days)
    recently_sold = set(
        StockChange.objects.filter(
            change_type='checkout', timestamp__date__gte=cutoff,
        ).values_list('product_id', flat=True).distinct()
    )
    base = _drop_snacks(
        Product.objects.filter(status=True, quantity_in_stock__gt=0).exclude(product_id__in=recently_sold),
        exclude_snacks,
    )
    prods = list(base.select_related('category').order_by('-quantity_in_stock')[:limit])
    # Last checkout per product in ONE query instead of one-per-item (N+1).
    last_sale_map = {}
    pids = [p.product_id for p in prods]
    if pids:
        for r in (StockChange.objects.filter(product_id__in=pids, change_type='checkout')
                  .values('product_id').annotate(last=Max('timestamp'))):
            last_sale_map[r['product_id']] = r['last']
    items = []
    for p in prods:
        last_sale = last_sale_map.get(p.product_id)
        items.append({
            'product_id': p.product_id,
            'name': p.name,
            'barcode': p.barcode or '',
            'quantity_in_stock': p.quantity_in_stock,
            'capital_tied': float(p.price * p.quantity_in_stock),
            'days_since_sale': (today - last_sale.date()).days if last_sale else 'Never',
            'category_name': p.category.name if p.category else '',
        })
    return {'items': items, 'count': base.count()}


def expiry_calendar(day=None, horizon_days=60):
    today = _resolve_day(day)
    rows = (
        Product.objects.filter(
            status=True, expiry_date__gte=today, expiry_date__lte=today + timedelta(days=horizon_days),
        ).exclude(expiry_date__isnull=True)
        .values('expiry_date').annotate(count=Count('product_id')).order_by('expiry_date')
    )
    return [{'date': r['expiry_date'].isoformat(), 'count': r['count']} for r in rows]


def recent_activity(limit=10):
    """Most recent stock changes — returned as a queryset (template needs model objects)."""
    return StockChange.objects.select_related('product').order_by('-timestamp')[:limit]


# ── Digest-only metric groups ───────────────────────────────────────────────

def low_stock_list(day=None, limit=None, exclude_snacks=False):
    qs = _low_stock_qs(exclude_snacks).select_related('category').order_by('quantity_in_stock')
    count = qs.count()
    rows = qs[:limit] if limit else qs
    items = [{
        'name': p.name, 'barcode': p.barcode or '',
        'quantity_in_stock': p.quantity_in_stock,
        'threshold': p.category.low_stock_threshold if p.category else LOW_STOCK_DEFAULT,
    } for p in rows]
    return {'count': count, 'items': items}


def out_of_stock_list(day=None, limit=None, exclude_snacks=False):
    qs = _drop_snacks(
        Product.objects.filter(status=True, quantity_in_stock=0), exclude_snacks
    ).select_related('category').order_by('name')
    count = qs.count()
    rows = qs[:limit] if limit else qs
    items = [{'name': p.name, 'barcode': p.barcode or '',
              'category_name': p.category.name if p.category else ''} for p in rows]
    return {'count': count, 'items': items}


def expiring_this_week(day=None, exclude_snacks=False):
    today = _resolve_day(day)
    qs = _drop_snacks(Product.objects.filter(
        status=True, expiry_date__gte=today, expiry_date__lte=today + timedelta(days=7),
    ).exclude(expiry_date__isnull=True), exclude_snacks).order_by('expiry_date')
    items = [{
        'name': p.name, 'barcode': p.barcode or '',
        'quantity_in_stock': p.quantity_in_stock,
        'expiry_date': p.expiry_date,
        'days_left': (p.expiry_date - today).days,
    } for p in qs]
    return {'count': len(items), 'items': items}


def stock_corrections(day=None, exclude_snacks=False):
    """Today's manual corrections and items marked expired today."""
    today = _resolve_day(day)
    base = _drop_snacks(
        StockChange.objects.select_related('product', 'user').filter(timestamp__date=today),
        exclude_snacks, prefix='product__',
    )

    def _row(sc):
        return {
            'time': sc.timestamp.strftime('%H:%M'),
            'name': sc.display_name,
            'barcode': sc.display_barcode,
            'action': sc.get_change_type_display(),
            'qty': sc.quantity,
            'user': sc.user.get_username() if sc.user else '',
            'note': sc.staff_note,
        }

    corrections = [_row(sc) for sc in base.filter(change_type__in=STOCK_CORRECTION_TYPES).order_by('-timestamp')]
    expired_today = [_row(sc) for sc in base.filter(change_type='expired').order_by('-timestamp')]
    return {
        'corrections': corrections, 'expired_today': expired_today,
        'correction_count': len(corrections), 'expired_count': len(expired_today),
    }


# ── Assemblers ──────────────────────────────────────────────────────────────

def dashboard_kpis(day=None):
    """All the numeric/dict context keys the dashboard home() view needs,
    emitted under their existing (legacy) template key names."""
    today = _resolve_day(day)
    health = stock_health(today)
    sales = sales_summary(today)
    inv = inventory_valuation(today)
    buckets = expiry_buckets(today)
    dead = dead_stock(today)
    chart = sales_chart(today)

    # Rolling 7-day revenue, derived from the (already computed) daily chart
    week_floor = (today - timedelta(days=6)).strftime('%Y-%m-%d')
    week_revenue = sum(d['revenue'] for d in chart if d['full_date'] >= week_floor)

    return {
        **health,
        'orders_today': sales['orders_today'],
        'revenue_today': sales['revenue_today'],
        'units_sold_today': sales['units_sold'],
        'week_revenue': week_revenue,
        'week_daily_avg': week_revenue / 7,
        'ordering_pending_count': OrderingSheetEntry.objects.filter(
            is_deleted=False, status=OrderingSheetEntry.STATUS_PENDING).count(),
        'total_units': inv['total_units'],
        'total_retail': inv['total_retail'],
        'total_cost': inv['total_cost'],
        'gross_margin_pct': inv['gross_margin_pct'],
        'best_sellers': top_movers(today),
        **buckets,
        'daily_chart_data': chart,
        'reorder_suggestions': reorder_suggestions(today),
        'dead_stock_items': dead['items'],
        'dead_stock_count': dead['count'],
        'expiry_calendar_json': expiry_calendar(today),
    }


def daily_digest(day=None, exclude_snacks=False):
    """Everything the end-of-day report needs, assembled from the helpers above."""
    today = _resolve_day(day)
    return {
        'day': today,
        'exclude_snacks': exclude_snacks,
        'sales': sales_summary(today, exclude_snacks),
        'stock_health': stock_health(today, exclude_snacks),
        'inventory': inventory_valuation(today, exclude_snacks),
        'top_movers': top_movers(today, exclude_snacks=exclude_snacks),
        'low_stock': low_stock_list(today, exclude_snacks=exclude_snacks),
        'out_of_stock': out_of_stock_list(today, exclude_snacks=exclude_snacks),
        'expiring_week': expiring_this_week(today, exclude_snacks),
        'dead_stock': dead_stock(today, exclude_snacks=exclude_snacks),
        'corrections': stock_corrections(today, exclude_snacks),
    }


# ── Archive (retained report snapshots) ─────────────────────────────────────

def prune_daily_report_archives(reference_date=None):
    """Compatibility hook: saved reports are retained without age-based deletion."""
    return 0

def archive_daily_report(day=None, digest=None, pdf=None):
    """Save a new snapshot only when the latest saved report content changes.

    Compare report data before rendering: PDF metadata changes on every
    generation, even when business content is unchanged. Comparing only the
    latest snapshot retains chronology when content changes A -> B -> A.
    """
    import hashlib
    import json
    from django.core.serializers.json import DjangoJSONEncoder
    from django.db import connection, transaction
    from .models import DailyReportArchive

    if digest is None:
        from .daily_reporting import build_daily_report
        digest = build_daily_report(day)
    d = digest['day']
    snapshot_json = json.dumps(
        {key: value for key, value in digest.items() if key != 'generated_at'},
        cls=DjangoJSONEncoder, sort_keys=True, separators=(',', ':'),
    )
    fingerprint = hashlib.sha256(snapshot_json.encode('utf-8')).hexdigest()
    with transaction.atomic():
        if connection.vendor == 'postgresql':
            # This transaction-scoped, per-day lock also protects the first
            # save, when no archive row exists yet to select_for_update.
            with connection.cursor() as cursor:
                cursor.execute('SELECT pg_advisory_xact_lock(%s, %s)', [17493, d.toordinal()])
        latest = DailyReportArchive.objects.select_for_update().defer('pdf', 'snapshot_data').filter(report_date=d).first()
        if latest and latest.content_sha256 == fingerprint:
            return latest
        s = digest['sales']
        return DailyReportArchive.objects.create(
            report_date=d,
            pdf=pdf if pdf is not None else build_daily_report_pdf(digest),
            summary=(
                f"${float(s['revenue_today']):,.2f} · {s['orders_today']} orders · "
                f"{s['units_sold']} units"
            ),
            snapshot_data=json.loads(snapshot_json),
            content_sha256=fingerprint,
        )


# ── PDF (shared by the view and the management command) ─────────────────────

def build_daily_report_pdf(digest):
    """Render either the legacy digest or the enhanced management report."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    day = digest['day']
    enhanced = 'inventory_day' in digest
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_w, page_h = letter
    margin = 40
    y = page_h - margin
    c.setTitle(f'Daily Report - {day:%Y-%m-%d}')

    def footer():
        c.setFont('Helvetica', 8)
        c.setFillColorRGB(0.39, 0.45, 0.55)
        c.drawString(margin, 24, f'Daily Report | {day:%b %d, %Y}')
        c.drawRightString(page_w - margin, 24, f'Page {c.getPageNumber()}')

    def new_page():
        nonlocal y
        footer()
        c.showPage()
        y = page_h - margin
        c.setFont('Helvetica-Bold', 10)
        c.setFillColorRGB(0.39, 0.45, 0.55)
        c.drawString(margin, y, f'Daily Report - {day:%b %d, %Y} (continued)')
        y -= 24

    def ensure_space(height):
        if y - height < margin + 10:
            new_page()

    def wrapped_lines(text, font, size, width):
        """Wrap complete text, including long unbroken product identifiers."""
        parts = []
        for paragraph in str(text).splitlines() or ['']:
            current = ''
            for word in paragraph.split():
                candidate = f'{current} {word}' if current else word
                if stringWidth(candidate, font, size) <= width:
                    current = candidate
                    continue
                if current:
                    parts.append(current)
                    current = ''
                while stringWidth(word, font, size) > width:
                    length = 1
                    while length < len(word) and stringWidth(word[:length + 1], font, size) <= width:
                        length += 1
                    parts.append(word[:length])
                    word = word[length:]
                current = word
            parts.append(current)
        return parts

    def heading(text):
        nonlocal y
        ensure_space(66)
        y -= 18
        c.setFillColorRGB(0.31, 0.27, 0.90)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(margin, y, text)
        y -= 4
        c.setStrokeColorRGB(0.89, 0.91, 0.94)
        c.line(margin, y, page_w - margin, y)
        y -= 14

    def line(text, bold=False, indent=0):
        nonlocal y
        font = 'Helvetica-Bold' if bold else 'Helvetica'
        parts = wrapped_lines(text, font, 9, page_w - 2 * margin - indent)
        # Keep normal rows together; exceptionally long notes may span pages.
        if len(parts) * 13 < page_h - 2 * margin - 40:
            ensure_space(len(parts) * 13)
        for part in parts:
            ensure_space(13)
            c.setFillColorRGB(0.06, 0.09, 0.16)
            c.setFont(font, 9)
            c.drawString(margin + indent, y, part)
            y -= 13

    def remaining(total, shown):
        if total > shown:
            line(f'Showing {shown} of {total}; {total - shown} more are available in the application.', indent=6)

    # Title
    c.setFillColorRGB(0.06, 0.09, 0.16)
    c.setFont('Helvetica-Bold', 18)
    c.drawString(margin, y, 'Daily Management Report' if enhanced else 'Daily End-of-Day Report')
    y -= 21
    line(day.strftime('%A, %B %d, %Y'))
    if digest.get('exclude_snacks'):
        line('Snacks category excluded')
    if enhanced:
        line('Sales and activity use the selected date. Sales include active corrections recorded since then.')
        line(f'Inventory reflects current balances as of {digest["inventory_day"]:%b %d, %Y}.')

    s, h, inv = digest['sales'], digest['stock_health'], digest['inventory']

    heading(f'Sales - {day:%b %d, %Y}')
    line(f"Net revenue: ${s['revenue_today']:,.2f}", bold=True)
    line('Before tax, after discounts and active returns or voids.')
    line(f"Realized orders: {s['orders_today']}    Units sold: {s['units_sold']}")
    if 'average_order' in s:
        line(f"Average order: ${s['average_order']:,.2f}    Recorded cost: ${s['cost']:,.2f}")
        missing_cost = s.get('missing_cost_units', 0)
        profit_label = 'Profit before missing costs' if missing_cost else 'Gross profit'
        margin_label = 'Margin before missing costs' if missing_cost else 'Sales margin'
        margin_text = f"{s['margin_pct']}%" if s['margin_pct'] is not None else 'N/A (no net sales)'
        line(f"{profit_label}: ${s['profit']:,.2f}    {margin_label}: {margin_text}")
        if missing_cost:
            line(f'Cost snapshots are missing for {missing_cost} units. Recorded cost is incomplete; '
                 'profit and margin may be overstated. Non-restocked returns retain inventory cost.')

    comparisons = digest.get('comparisons', {})
    if comparisons:
        heading('Revenue comparisons')
        for key, label in (('previous_day', 'Previous day'), ('previous_week', 'Same day last week')):
            comparison = comparisons.get(key)
            if comparison is None:
                continue
            change = f"{comparison['delta']:+,.2f}"
            percent = f"{comparison['pct']:+.1f}%" if comparison['pct'] is not None else 'no percentage baseline'
            line(f"{label} ({comparison['day']:%b %d}): ${comparison['revenue']:,.2f}; "
                 f"revenue change ${change} ({percent}).")

    trend = digest.get('trend', [])
    if trend:
        heading(f"Seven-day sales - {trend[0]['day']:%b %d} to {trend[-1]['day']:%b %d, %Y}")
        line(f"Net revenue: ${sum((row['revenue'] for row in trend), Decimal('0.00')):,.2f}    "
             f"Orders: {sum(row['orders'] for row in trend)}    Units: {sum(row['units'] for row in trend)}", bold=True)
        for row in trend:
            line(f"{row['day']:%a, %b %d}: ${row['revenue']:,.2f}    "
                 f"{row['orders']} orders    {row['units']} units", indent=6)

    if 'top_products' in digest:
        heading('Top products - selected date')
        for item in digest['top_products']:
            line(f"{item['name']} | {item.get('barcode') or 'No barcode'} | "
                 f"{item['units']} units | Revenue ${item['revenue']:,.2f}", indent=6)
        if not digest['top_products']:
            line('No realized product sales or retained return costs on this date.', indent=6)

    if 'categories' in digest:
        heading('Category sales - selected date')
        for item in digest['categories']:
            line(f"{item['name']} | {item['units']} units | Revenue ${item['revenue']:,.2f} | "
                 f"{item['share_pct']}% of revenue", indent=6)
        if not digest['categories']:
            line('No category sales on this date.', indent=6)

    heading('Top movers - seven days ending on the selected date' if enhanced else 'Top movers (last 7 days)')
    if digest['top_movers']:
        for m in digest['top_movers']:
            line(f"{m['total_qty']} units | {m['product_name']}", indent=6)
    else:
        line('No sales in the last 7 days.', indent=6)

    heading(f'Current inventory - {digest["inventory_day"]:%b %d, %Y}' if enhanced else 'Stock health')
    line(f"Out of stock: {h['out_of_stock_count']}    Low stock: {h['low_stock_count']}    "
         f"Active products: {h['total_products']}")
    line(f"Expiring within 7 days: {h['expiring_soon_count']}"
         + (f"    Expired stock: {h['expired_count']}" if 'expired_count' in h else ''))
    line(f"Inventory retail value: ${inv['total_retail']:,.2f}    "
         f"Stock valuation margin: {inv['gross_margin_pct']}%")
    if enhanced:
        line('Expiry counts represent stocked lots or legacy balances. Stock valuation margin is based '
             'on current retail prices and recorded product costs; it is not realized sales margin.')

    low = digest['low_stock']
    heading(f"Low stock ({low['count']})")
    for it in low['items'][:25]:
        line(f"{it['name']} | Stock {it['quantity_in_stock']} / threshold {it['threshold']}", indent=6)
    if not low['items']:
        line('No low-stock products.', indent=6)
    remaining(low['count'], len(low['items'][:25]))

    oos = digest['out_of_stock']
    heading(f"Out of stock ({oos['count']})")
    for it in oos['items'][:25]:
        line(it['name'], indent=6)
    if not oos['items']:
        line('No out-of-stock products.', indent=6)
    remaining(oos['count'], len(oos['items'][:25]))

    for key, label in (('expired_stock', 'Expired stock on hand'), ('expiring_week', 'Expiring within 7 days')):
        if key not in digest:
            continue
        expiry = digest[key]
        heading(f"{label} ({expiry['count']})")
        for it in expiry['items']:
            lot = f" | Lot {it['lot_number']}" if it.get('lot_number') else ''
            days_label = f"{abs(it['days_left'])} days overdue" if it['days_left'] < 0 else f"{it['days_left']} days left"
            line(f"{it['name']}{lot} | Expiry {it['expiry_date']:%b %d, %Y} | "
                 f"{days_label} | {it['quantity_in_stock']} units", indent=6)
        if not expiry['items']:
            line('No stocked items in this expiry range.', indent=6)
        remaining(expiry['count'], len(expiry['items']))

    dead = digest['dead_stock']
    heading(f"Dead stock ({dead['count']})")
    if 'lookback_days' in dead:
        line(f"No checkout recorded within {dead['lookback_days']} days. Values below use current retail prices.")
    for it in dead['items']:
        retail_value = it['retail_value'] if 'retail_value' in it else it['capital_tied']
        last_sale = 'Never' if it['days_since_sale'] == 'Never' else f"{it['days_since_sale']} days ago"
        line(f"{it['name']} | {it['quantity_in_stock']} units | Retail value ${retail_value:,.2f} | "
             f"Last sale: {last_sale}", indent=6)
    if not dead['items']:
        line('No dead stock detected.', indent=6)
    remaining(dead['count'], len(dead['items']))

    if 'activity' in digest:
        activity = digest['activity']
        heading(f'Daily activity - {day:%b %d, %Y}')
        line(f"Stock events: {activity['count']}    Check-ins: {activity['checkin_count']} "
             f"({activity['checkin_units']} units)")
        line(f"Corrections: {activity['correction_count']}    Expiry retirements: {activity['expired_count']} "
             f"({activity['expired_units']} units)")

    corr = digest['corrections']
    heading(f"Corrections ({corr['correction_count']}) and expiries ({corr['expired_count']}) - {day:%b %d}")
    for it in corr['corrections']:
        line(f"{it['time']} | {it['action']} | {it['qty']} units | {it['name']}", indent=6)
        if it.get('user') or it.get('note'):
            line(f"By {it.get('user') or 'Unknown'}: {it.get('note') or 'No note'}", indent=14)
    remaining(corr['correction_count'], len(corr['corrections']))
    for it in corr['expired_today']:
        line(f"{it['time']} | Expired | {it['qty']} units | {it['name']}", indent=6)
    remaining(corr['expired_count'], len(corr['expired_today']))
    if not corr['corrections'] and not corr['expired_today']:
        line('No corrections or expiries logged on this date.', indent=6)

    footer()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
