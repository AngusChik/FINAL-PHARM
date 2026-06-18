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

from django.db.models import Sum, F, Count, Value, DecimalField
from django.db.models.functions import TruncDate, TruncWeek, Coalesce

from .models import Product, Order, OrderDetail, StockChange
from .utils import get_reorder_prediction

LOW_STOCK_DEFAULT = 3
SALE_TYPES = ['checkout', 'checkout_unfulfilled']
STOCK_CORRECTION_TYPES = ['error_add', 'error_subtract']


def _resolve_day(day=None):
    return day or date.today()


def _low_stock_qs():
    """Active, in-stock products at or below their (category) low-stock threshold."""
    return (
        Product.objects.filter(status=True, quantity_in_stock__gt=0)
        .annotate(_threshold=Coalesce(F('category__low_stock_threshold'), Value(LOW_STOCK_DEFAULT)))
        .filter(quantity_in_stock__lte=F('_threshold'))
    )


# ── Individual metric groups ────────────────────────────────────────────────

def stock_health(day=None):
    today = _resolve_day(day)
    return {
        'out_of_stock_count': Product.objects.filter(status=True, quantity_in_stock=0).count(),
        'low_stock_count': _low_stock_qs().count(),
        'expiring_soon_count': Product.objects.filter(
            expiry_date__gte=today, expiry_date__lte=today + timedelta(days=7)
        ).exclude(expiry_date__isnull=True).count(),
        'total_products': Product.objects.filter(status=True).count(),
    }


def sales_summary(day=None):
    today = _resolve_day(day)
    lines = OrderDetail.objects.filter(order__order_date__date=today, order__submitted=True)
    return {
        'orders_today': Order.objects.filter(order_date__date=today, submitted=True).count(),
        'revenue_today': lines.aggregate(total=Sum(F('price') * F('quantity')))['total'] or Decimal('0.00'),
        'units_sold': lines.aggregate(total=Sum('quantity'))['total'] or 0,
    }


def inventory_valuation(day=None):
    agg = Product.objects.filter(status=True).aggregate(
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


def top_movers(day=None, days=7, limit=5):
    today = _resolve_day(day)
    since = today - timedelta(days=days)
    return list(
        OrderDetail.objects.filter(
            order__submitted=True, order__order_date__date__gte=since
        ).values('product_name', 'product_barcode').annotate(
            total_qty=Sum('quantity')
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
        OrderDetail.objects.filter(
            order__submitted=True, order__order_date__date__gte=start,
        )
        .annotate(sale_date=TruncDate('order__order_date'))
        .values('sale_date')
        .annotate(
            daily_revenue=Sum(F('price') * F('quantity'), output_field=DecimalField()),
            order_count=Count('order', distinct=True),
            item_count=Count('od_id'),
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
        })
    return suggestions


def dead_stock(day=None, lookback_days=69, limit=8):
    today = _resolve_day(day)
    cutoff = today - timedelta(days=lookback_days)
    recently_sold = set(
        StockChange.objects.filter(
            change_type='checkout', timestamp__date__gte=cutoff,
        ).values_list('product_id', flat=True).distinct()
    )
    base = Product.objects.filter(status=True, quantity_in_stock__gt=0).exclude(product_id__in=recently_sold)
    items = []
    for p in base.select_related('category').order_by('-quantity_in_stock')[:limit]:
        last_sale = (
            StockChange.objects.filter(product=p, change_type='checkout')
            .order_by('-timestamp').values_list('timestamp', flat=True).first()
        )
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

def low_stock_list(day=None, limit=None):
    qs = _low_stock_qs().select_related('category').order_by('quantity_in_stock')
    count = qs.count()
    rows = qs[:limit] if limit else qs
    items = [{
        'name': p.name, 'barcode': p.barcode or '',
        'quantity_in_stock': p.quantity_in_stock,
        'threshold': p.category.low_stock_threshold if p.category else LOW_STOCK_DEFAULT,
    } for p in rows]
    return {'count': count, 'items': items}


def out_of_stock_list(day=None, limit=None):
    qs = Product.objects.filter(status=True, quantity_in_stock=0).select_related('category').order_by('name')
    count = qs.count()
    rows = qs[:limit] if limit else qs
    items = [{'name': p.name, 'barcode': p.barcode or '',
              'category_name': p.category.name if p.category else ''} for p in rows]
    return {'count': count, 'items': items}


def expiring_this_week(day=None):
    today = _resolve_day(day)
    qs = Product.objects.filter(
        status=True, expiry_date__gte=today, expiry_date__lte=today + timedelta(days=7),
    ).exclude(expiry_date__isnull=True).order_by('expiry_date')
    items = [{
        'name': p.name, 'barcode': p.barcode or '',
        'quantity_in_stock': p.quantity_in_stock,
        'expiry_date': p.expiry_date,
        'days_left': (p.expiry_date - today).days,
    } for p in qs]
    return {'count': len(items), 'items': items}


def stock_corrections(day=None):
    """Today's manual corrections and items marked expired today."""
    today = _resolve_day(day)
    base = StockChange.objects.select_related('product', 'user').filter(timestamp__date=today)

    def _row(sc):
        return {
            'time': sc.timestamp.strftime('%H:%M'),
            'name': sc.display_name,
            'barcode': sc.display_barcode,
            'action': sc.get_change_type_display(),
            'qty': sc.quantity,
            'user': sc.user.get_username() if sc.user else '',
            'note': sc.note or '',
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
    return {
        **health,
        'orders_today': sales['orders_today'],
        'revenue_today': sales['revenue_today'],
        'total_units': inv['total_units'],
        'total_retail': inv['total_retail'],
        'total_cost': inv['total_cost'],
        'gross_margin_pct': inv['gross_margin_pct'],
        'best_sellers': top_movers(today),
        **buckets,
        'daily_chart_data': sales_chart(today),
        'reorder_suggestions': reorder_suggestions(today),
        'dead_stock_items': dead['items'],
        'dead_stock_count': dead['count'],
        'expiry_calendar_json': expiry_calendar(today),
    }


def daily_digest(day=None):
    """Everything the end-of-day report needs, assembled from the helpers above."""
    today = _resolve_day(day)
    return {
        'day': today,
        'sales': sales_summary(today),
        'stock_health': stock_health(today),
        'inventory': inventory_valuation(today),
        'top_movers': top_movers(today),
        'low_stock': low_stock_list(today),
        'out_of_stock': out_of_stock_list(today),
        'expiring_week': expiring_this_week(today),
        'dead_stock': dead_stock(today),
        'corrections': stock_corrections(today),
    }


# ── PDF (shared by the view and the management command) ─────────────────────

def build_daily_report_pdf(digest):
    """Render the daily digest to PDF bytes (reportlab canvas, mirrors the
    existing PDF views). Kept here so DailyReportPDFView and the
    send_daily_report command share one implementation."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    day = digest['day']
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_w, page_h = letter
    margin = 36
    y = page_h - margin

    def heading(text):
        nonlocal y
        if y < margin + 60:
            c.showPage(); y = page_h - margin
        y -= 22
        c.setFillColorRGB(0.31, 0.27, 0.90)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(margin, y, text)
        y -= 4
        c.setStrokeColorRGB(0.89, 0.91, 0.94)
        c.line(margin, y, page_w - margin, y)
        y -= 10

    def line(text, bold=False, indent=0):
        nonlocal y
        if y < margin + 20:
            c.showPage(); y = page_h - margin
        c.setFillColorRGB(0.06, 0.09, 0.16)
        c.setFont('Helvetica-Bold' if bold else 'Helvetica', 9)
        c.drawString(margin + indent, y, text[:115])
        y -= 14

    # Title
    c.setFillColorRGB(0.06, 0.09, 0.16)
    c.setFont('Helvetica-Bold', 18)
    c.drawString(margin, y, 'Daily End-of-Day Report')
    y -= 18
    c.setFont('Helvetica', 10)
    c.setFillColorRGB(0.39, 0.45, 0.55)
    c.drawString(margin, y, day.strftime('%A, %B %d, %Y'))
    y -= 16

    s, h, inv = digest['sales'], digest['stock_health'], digest['inventory']

    heading('Sales today')
    line(f"Revenue: ${float(s['revenue_today']):,.2f}", bold=True)
    line(f"Submitted orders: {s['orders_today']}    Units sold: {s['units_sold']}")

    heading('Stock health')
    line(f"Out of stock: {h['out_of_stock_count']}    Low stock: {h['low_stock_count']}    "
         f"Expiring ≤7d: {h['expiring_soon_count']}    Active products: {h['total_products']}")
    line(f"Inventory value (retail): ${float(inv['total_retail']):,.2f}    "
         f"Gross margin: {inv['gross_margin_pct']}%")

    heading('Top movers (last 7 days)')
    if digest['top_movers']:
        for m in digest['top_movers']:
            line(f"{m['total_qty']:>4}  ×  {m['product_name']}", indent=6)
    else:
        line('No sales in the last 7 days.', indent=6)

    low = digest['low_stock']
    heading(f"Low stock ({low['count']})")
    for it in low['items'][:25]:
        line(f"{it['quantity_in_stock']:>4} / {it['threshold']:<4}  {it['name']}", indent=6)
    if low['count'] > 25:
        line(f"... and {low['count'] - 25} more", indent=6)

    oos = digest['out_of_stock']
    heading(f"Out of stock ({oos['count']})")
    for it in oos['items'][:25]:
        line(f"- {it['name']}", indent=6)
    if oos['count'] > 25:
        line(f"... and {oos['count'] - 25} more", indent=6)

    exp = digest['expiring_week']
    heading(f"Expiring this week ({exp['count']})")
    for it in exp['items']:
        line(f"{it['days_left']:>3}d  {it['expiry_date'].strftime('%b %d')}  "
             f"qty {it['quantity_in_stock']}  {it['name']}", indent=6)

    dead = digest['dead_stock']
    heading(f"Dead stock ({dead['count']})")
    for it in dead['items']:
        line(f"qty {it['quantity_in_stock']:>4}  ${it['capital_tied']:,.2f} tied  "
             f"(last sold {it['days_since_sale']})  {it['name']}", indent=6)

    corr = digest['corrections']
    heading(f"Today's corrections ({corr['correction_count']}) & expiries ({corr['expired_count']})")
    for it in corr['corrections']:
        line(f"{it['time']}  {it['action']}  {it['qty']}  {it['name']}", indent=6)
    for it in corr['expired_today']:
        line(f"{it['time']}  Expired  {it['qty']}  {it['name']}", indent=6)
    if not corr['corrections'] and not corr['expired_today']:
        line('No corrections or expiries logged today.', indent=6)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()
