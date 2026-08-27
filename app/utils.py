from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import List
from datetime import datetime, timedelta, date, time
import math
from dateutil.relativedelta import relativedelta
from django.utils.timezone import make_aware
from django.db.models import Sum
from .models import StockChange, Product

TAX_RATE = Decimal("0.13")
MONEY_QUANTUM = Decimal("0.01")

STOCK_SIGN = {
    "checkin": +1,
    "error_add": +1,
    "restoration": +1,
    "return": +1,
    "checkout": -1,
    "expired": -1,
    "error_subtract": -1,
    "checkin_delete1": -1,
    "giveaway": -1,
    "deletion": -1,
}


def stock_change_delta(change_type, quantity, correction_disposition=None):
    """Return the signed on-hand movement represented by a ledger row.

    Transaction corrections need more context than a static sign map. A return
    always has a ``return`` row only when it was put back into stock, while a
    void and its undo can be either physical or financial-only depending on the
    correction line disposition. Keeping this rule in one helper prevents the
    Product Trend chart and end-of-day reconstruction from drifting apart as
    new ledger event types are added.
    """
    qty = abs(int(quantity or 0))
    if not qty:
        return 0

    if change_type == "void":
        return qty if correction_disposition == "restock" else 0
    if change_type == "correction_undo":
        return -qty if correction_disposition == "restock" else 0
    return STOCK_SIGN.get(change_type, 0) * qty

def get_stock_eod(product: Product, day: date) -> int:
    """
    Returns stock level at END OF ``day`` (EOD) from the append-only ledger.

    This intentionally stays uncached: Product Trend is used immediately after
    stock activity, and a one-hour cache could otherwise show a stale forecast
    or historical stock line after a sale, return, void, or check-in.
    """
    after_rows = (
        StockChange.objects
        .filter(product=product, timestamp__date__gt=day)
        .values("change_type", "correction_line__disposition")
        .annotate(total=Sum("quantity"))
    )

    net_after = 0
    for row in after_rows:
        net_after += stock_change_delta(
            row["change_type"],
            row["total"],
            row["correction_line__disposition"],
        )

    current_stock = int(product.quantity_in_stock or 0)
    return max(0, current_stock - net_after)

CLOSED_WEEKDAYS = {6}  # Sunday

def count_open_days(start_d: date, end_d: date, closed_weekdays=CLOSED_WEEKDAYS) -> int:
    if end_d < start_d: return 0
    n = 0
    d = start_d
    while d <= end_d:
        if d.weekday() not in closed_weekdays:
            n += 1
        d += timedelta(days=1)
    return n

def calculate_order_financials_from_values(
        line_values, seniors_discount=False, tax_rate=TAX_RATE):
    """Calculate settled order money from ``(price, quantity, taxable)`` rows.

    The seniors discount is deliberately rounded once at order level. Keeping
    this policy in one helper prevents detail pages, corrections, and reports
    from independently rounding discounted line values to different cents.
    """
    subtotal = Decimal("0.00")
    taxable_subtotal = Decimal("0.00")

    for price, quantity, taxable in line_values:
        line_total = Decimal(price) * int(quantity)
        subtotal += line_total
        if taxable is True:
            taxable_subtotal += line_total

    subtotal = subtotal.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    discount_amount = Decimal("0.00")
    taxable_base = taxable_subtotal
    if seniors_discount:
        discount_amount = (subtotal * Decimal("0.10")).quantize(
            MONEY_QUANTUM, rounding=ROUND_HALF_UP,
        )
        taxable_base = taxable_subtotal * Decimal("0.90")

    tax = (taxable_base * Decimal(tax_rate)).quantize(
        MONEY_QUANTUM, rounding=ROUND_HALF_UP,
    )
    total = (subtotal - discount_amount + tax).quantize(
        MONEY_QUANTUM, rounding=ROUND_HALF_UP,
    )
    return {
        "subtotal": subtotal,
        "discount_amount": discount_amount,
        "tax": tax,
        "total": total,
        "taxable_subtotal": taxable_subtotal.quantize(
            MONEY_QUANTUM, rounding=ROUND_HALF_UP,
        ),
    }


def calculate_order_financials(order_details, seniors_discount=False, tax_rate=TAX_RATE):
    """Calculate an order strictly from its immutable line snapshots."""
    return calculate_order_financials_from_values(
        (
            (detail.price, detail.quantity, detail.taxable_at_sale)
            for detail in order_details
        ),
        seniors_discount=seniors_discount,
        tax_rate=tax_rate,
    )


def allocate_currency(amount, weights):
    """Allocate a settled currency amount without creating or losing a cent.

    Values are apportioned using largest remainders. This keeps every returned
    value at two decimals and guarantees their sum is exactly ``amount``.
    """
    amount = Decimal(amount).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    weights = [max(Decimal("0.00"), Decimal(weight)) for weight in weights]
    if not weights:
        return []
    total_weight = sum(weights, Decimal("0.00"))
    if not amount or not total_weight:
        return [Decimal("0.00") for _ in weights]

    sign = Decimal("-1.00") if amount < 0 else Decimal("1.00")
    total_cents = int(
        ((abs(amount) / MONEY_QUANTUM)).to_integral_value(
            rounding=ROUND_HALF_UP,
        )
    )
    raw_cents = [Decimal(total_cents) * weight / total_weight for weight in weights]
    allocated_cents = [
        int(value.to_integral_value(rounding=ROUND_DOWN))
        for value in raw_cents
    ]
    cents_left = total_cents - sum(allocated_cents)
    remainder_order = sorted(
        range(len(weights)),
        key=lambda index: (
            raw_cents[index] - Decimal(allocated_cents[index]),
            -index,
        ),
        reverse=True,
    )
    for index in remainder_order[:cents_left]:
        allocated_cents[index] += 1

    return [
        (sign * Decimal(cents) * MONEY_QUANTUM).quantize(MONEY_QUANTUM)
        for cents in allocated_cents
    ]


def allocate_order_line_financials(
        line_totals, taxable_flags, discount_amount, tax_amount):
    """Reconcile line display values to settled order discount and tax totals."""
    line_totals = [Decimal(total) for total in line_totals]
    taxable_flags = list(taxable_flags)
    discount_shares = allocate_currency(discount_amount, line_totals)
    tax_weights = [
        total if taxable is True else Decimal("0.00")
        for total, taxable in zip(line_totals, taxable_flags)
    ]
    tax_shares = allocate_currency(tax_amount, tax_weights)
    return [
        {
            "discount": discount,
            "tax": tax,
            "net": (gross - discount).quantize(
                MONEY_QUANTUM, rounding=ROUND_HALF_UP,
            ),
            "total": (gross - discount + tax).quantize(
                MONEY_QUANTUM, rounding=ROUND_HALF_UP,
            ),
        }
        for gross, discount, tax in zip(
            line_totals, discount_shares, tax_shares,
        )
    ]


def recalculate_order_totals(order, snapshot_source=None):
    """Finalize and persist the order-level financial snapshot."""
    details = list(order.details.all())
    values = calculate_order_financials(
        details,
        seniors_discount=order.seniors_discount,
        tax_rate=TAX_RATE,
    )
    order.subtotal = values["subtotal"]
    order.discount_amount = values["discount_amount"]
    order.tax = values["tax"]
    order.tax_rate = TAX_RATE
    order.total_price = values["total"]
    if snapshot_source is not None:
        order.financial_snapshot_source = snapshot_source
    order.save(update_fields=[
        "subtotal", "discount_amount", "tax", "tax_rate", "total_price",
        "financial_snapshot_source",
    ])
    return values

# --- Data Models ---

class PurchaseRecord:
    def __init__(self, quantity: int, purchase_date: str):
        self.quantity = quantity
        self.purchase_date = datetime.strptime(purchase_date, "%Y-%m-%d")

class SaleRecord:
    def __init__(self, quantity: int, sale_date: str):
        self.quantity = quantity
        self.sale_date = datetime.strptime(sale_date, "%Y-%m-%d")

class ExpiryRecord:
    def __init__(self, quantity: int, expiry_date: str):
        self.quantity = quantity
        self.expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d")

def get_product_stock_records(product: Product, start_date: str, end_date: str):
    """
    Build structured records from StockChange between start_date and end_date.

    Sales are net of the transaction-correction ledger. Order returns and voids
    reduce demand, and undoing a void restores it. Corrections to terminal
    giveaways are deliberately excluded because a giveaway is a physical stock
    movement, not customer sales demand.

    Returns: purchases, sales, expiries, unfulfilled.
    """
    start = make_aware(datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), time.min))
    end = make_aware(datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), time.max))

    changes = list(
        StockChange.objects.filter(
            product=product,
            timestamp__range=(start, end),
        ).order_by("timestamp", "pk")
    )

    purchases = []
    sales = []
    unfulfilled = []
    expiries = []

    for change in changes:
        qty = abs(int(change.quantity or 0))
        record_date = change.timestamp.strftime("%Y-%m-%d")
        change_type = change.change_type

        if change_type in {"checkin", "error_add"}:
            purchases.append(PurchaseRecord(qty, record_date))
        elif change_type in {"error_subtract", "checkin_delete1"}:
            purchases.append(PurchaseRecord(-qty, record_date))
        elif change_type == "checkout":
            sales.append(SaleRecord(qty, record_date))
        elif change_type == "checkout_unfulfilled":
            unfulfilled.append(SaleRecord(qty, record_date))
        elif change_type == "expired":
            expiries.append(ExpiryRecord(qty, record_date))
        elif change.order_detail_id and change_type in {
            "return", "return_no_restock", "void",
        }:
            sales.append(SaleRecord(-qty, record_date))
        elif change.order_detail_id and change_type == "correction_undo":
            sales.append(SaleRecord(qty, record_date))

    return purchases, sales, expiries, unfulfilled

def weighted_avg_daily_demand(
    sales_records, missed_records, start_dt, end_dt, closed_weekdays, half_life_days=30
):
    """
    Exponential-decay weighted average of daily demand.
    Recent days carry more weight than older days (half-life = 30 days by default).

    Correction records can be negative. They reduce the weighted mean on the
    day they were entered, but the observed-demand series used for variability
    is floored at zero so a refund never becomes a physically negative demand
    day. Returns (weighted_avg, std_dev, daily_demands_dict).
    """
    daily = {}
    d = start_dt.date() if hasattr(start_dt, 'date') else start_dt
    end_d = end_dt.date() if hasattr(end_dt, 'date') else end_dt
    while d <= end_d:
        if d.weekday() not in closed_weekdays:
            daily[d] = 0
        d += timedelta(days=1)

    for s in sales_records:
        sd = s.sale_date.date() if hasattr(s.sale_date, 'date') else s.sale_date
        if sd in daily:
            daily[sd] += int(s.quantity)
    for u in missed_records:
        ud = u.sale_date.date() if hasattr(u.sale_date, 'date') else u.sale_date
        if ud in daily:
            daily[ud] += abs(int(u.quantity))

    if not daily:
        return 0.0, 0.0, daily

    sorted_dates = sorted(daily.keys())
    total_days = len(sorted_dates)
    last_date = sorted_dates[-1]

    weights = []
    demands = []
    for dt in sorted_dates:
        days_ago = (last_date - dt).days
        w = 2 ** (-days_ago / half_life_days)
        weights.append(w)
        demands.append(daily[dt])

    w_sum = sum(weights)
    if w_sum == 0:
        return 0.0, 0.0, daily

    w_avg = max(0.0, sum(d * w for d, w in zip(demands, weights)) / w_sum)

    # Variability is based on possible observed demand, never negative refunds.
    observed_demands = [max(0, demand) for demand in demands]
    variance = sum(
        w * (demand - w_avg) ** 2
        for demand, w in zip(observed_demands, weights)
    ) / w_sum
    std_dev = variance ** 0.5

    return w_avg, std_dev, daily


def compute_demand_trend(sales_records, missed_records, start_dt, end_dt, closed_weekdays):
    """
    Recency-weighted regression slope on calendar-aligned weekly demand.

    Partial first/last weeks are normalized by their number of open days, so a
    date-range boundary does not create a false rise or fall. Recent weeks carry
    more weight (eight-week half-life), and at least four weekly observations
    are required. Returns a slope in units per week.
    """
    start_d = start_dt.date() if hasattr(start_dt, 'date') else start_dt
    end_d = end_dt.date() if hasattr(end_dt, 'date') else end_dt

    # Build daily demand
    daily = {}
    d = start_d
    while d <= end_d:
        if d.weekday() not in closed_weekdays:
            daily[d] = 0
        d += timedelta(days=1)

    for s in sales_records:
        sd = s.sale_date.date() if hasattr(s.sale_date, 'date') else s.sale_date
        if sd in daily:
            daily[sd] += int(s.quantity)
    for u in missed_records:
        ud = u.sale_date.date() if hasattr(u.sale_date, 'date') else u.sale_date
        if ud in daily:
            daily[ud] += abs(int(u.quantity))

    if not daily:
        return 0.0

    # Calendar-aligned buckets with open-day normalization. Six is the normal
    # number of open days with the default Sunday closure, but this also adapts
    # if the configured closure set changes.
    normal_open_days = max(1, 7 - len(closed_weekdays))
    week_totals = {}
    week_open_days = {}
    for dt in sorted(daily):
        week_start = dt - timedelta(days=dt.weekday())
        week_totals[week_start] = week_totals.get(week_start, 0) + daily[dt]
        week_open_days[week_start] = week_open_days.get(week_start, 0) + 1

    weeks = [
        max(0.0, week_totals[week] / week_open_days[week] * normal_open_days)
        for week in sorted(week_totals)
        if week_open_days[week] > 0
    ]

    n = len(weeks)
    if n < 4:
        return 0.0

    weights = [2 ** (-(n - 1 - index) / 8.0) for index in range(n)]
    weight_sum = sum(weights)
    mean_x = sum(index * weight for index, weight in enumerate(weights)) / weight_sum
    mean_y = sum(value * weight for value, weight in zip(weeks, weights)) / weight_sum
    denominator = sum(
        weight * (index - mean_x) ** 2
        for index, weight in enumerate(weights)
    )
    if denominator == 0:
        return 0.0
    return sum(
        weight * (index - mean_x) * (value - mean_y)
        for index, (value, weight) in enumerate(zip(weeks, weights))
    ) / denominator


def expiring_stock_units(product, on_or_before, stock_level=None):
    """Return quantity-bearing stock due on/before a forecast boundary.

    Product lots are authoritative when available. Legacy products that have
    not adopted lot tracking fall back to their product-level expiry date.
    """
    total = 0
    has_dated_lot_history = False
    try:
        has_dated_lot_history = product.lots.filter(
            expiry_date__isnull=False,
        ).exists()
        total = int(
            product.lots.filter(
                archived_at__isnull=True,
                quantity_on_hand__gt=0,
                expiry_date__isnull=False,
                expiry_date__lte=on_or_before,
            ).aggregate(total=Sum("quantity_on_hand"))["total"]
            or 0
        )
    except (AttributeError, TypeError):
        total = 0

    if (
        not has_dated_lot_history
        and getattr(product, "expiry_date", None) is not None
    ):
        if product.expiry_date <= on_or_before:
            total = max(0, int(stock_level or product.quantity_in_stock or 0))

    if stock_level is not None:
        total = min(total, max(0, int(stock_level)))
    return max(0, total)


def recommend_inventory_action(
    product: Product,
    purchase_history,
    sale_history,
    expiry_history,
    unfulfilled_history,
    timeframe_start: str,
    timeframe_end: str,
    cost_per_unit: float,
    price_per_unit: float,
    granularity: str = "month",
    closed_weekdays=CLOSED_WEEKDAYS,
    service_level_z: float = 1.65,
    lead_time_days: int = 7,
) -> dict:
    start_dt = datetime.strptime(timeframe_start, "%Y-%m-%d")
    end_dt   = datetime.strptime(timeframe_end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("timeframe_end must be on or after timeframe_start")

    # Filter records. Sale history may contain negative correction records.
    sales = [s for s in sale_history if start_dt <= s.sale_date <= end_dt]
    missed = [u for u in unfulfilled_history if start_dt <= u.sale_date <= end_dt]
    purchases = [p for p in purchase_history if start_dt <= p.purchase_date <= end_dt]
    expiries = [e for e in expiry_history if start_dt <= e.expiry_date <= end_dt]

    total_sold = max(0, sum(int(s.quantity) for s in sales))
    total_missed = max(0, sum(abs(int(u.quantity)) for u in missed))
    net_received = sum(int(p.quantity) for p in purchases)
    gross_received = sum(max(0, int(p.quantity)) for p in purchases)
    total_expired = sum(abs(int(e.quantity)) for e in expiries)

    true_demand = total_sold + total_missed

    stock_eod_end = get_stock_eod(product, end_dt.date())
    opening_stock = get_stock_eod(product, start_dt.date() - timedelta(days=1))

    # Rates use stock made available for sale, not the closing balance. The old
    # denominator subtracted sales first and could produce misleading rates.
    available_for_sale = max(0, opening_stock + gross_received)
    sell_through_rate = (
        total_sold / available_for_sale * 100 if available_for_sale > 0 else 0
    )
    expiry_rate = (
        total_expired / available_for_sale * 100 if available_for_sale > 0 else 0
    )

    unit_margin = price_per_unit - cost_per_unit
    period_profit = (total_sold * unit_margin) - (total_expired * cost_per_unit)
    estimated_revenue_lost = total_missed * max(0.0, price_per_unit)
    estimated_gross_profit_lost = total_missed * max(0.0, unit_margin)

    open_days = count_open_days(start_dt.date(), end_dt.date(), closed_weekdays)
    flat_avg = (true_demand / open_days) if open_days > 0 else 0

    # Blend the stable whole-window rate with the recency-weighted rate. Short
    # windows cannot hand full control to a few recent transactions.
    w_avg, std_dev, daily_demands = weighted_avg_daily_demand(
        sales, missed, start_dt, end_dt, closed_weekdays
    )
    recency_share = min(0.75, open_days / 120.0) if open_days else 0.0
    avg_sales_per_day = (
        (flat_avg * (1.0 - recency_share)) + (w_avg * recency_share)
    )

    demand_days = sum(1 for demand in daily_demands.values() if demand > 0)
    coverage_score = min(1.0, open_days / 90.0) if open_days else 0.0
    if true_demand > 0:
        confidence_score = (
            0.65 * coverage_score + 0.35 * min(1.0, demand_days / 12.0)
        )
    else:
        confidence_score = coverage_score
    if confidence_score >= 0.75:
        forecast_confidence = "High"
        trend_cap = 0.40
    elif confidence_score >= 0.45:
        forecast_confidence = "Medium"
        trend_cap = 0.30
    else:
        forecast_confidence = "Low"
        trend_cap = 0.20

    # Forecast the next complete selected period. Monthly no longer means an
    # arbitrary 30 days; it follows the actual next calendar month.
    next_start = end_dt.date() + timedelta(days=1)
    if granularity == "day":
        next_end = next_start
    elif granularity == "week":
        next_end = next_start + timedelta(days=6)
    else:
        next_end = next_start + relativedelta(months=1) - timedelta(days=1)

    future_open_days = count_open_days(next_start, next_end, closed_weekdays)

    trend_slope = compute_demand_trend(sales, missed, start_dt, end_dt, closed_weekdays)
    trend_daily = trend_slope / 7.0  # convert weekly slope to daily

    trended_total = 0.0
    for day_i in range(1, future_open_days + 1):
        trended_total += max(0, avg_sales_per_day + trend_daily * day_i)

    flat_projection = avg_sales_per_day * future_open_days
    if flat_projection > 0:
        raw_trend_ratio = trended_total / flat_projection
        trend_ratio = max(
            1.0 - trend_cap,
            min(1.0 + trend_cap, raw_trend_ratio),
        )
        estimated_demand = int(math.ceil(flat_projection * trend_ratio))
    else:
        trend_ratio = 1.0
        estimated_demand = int(math.ceil(trended_total))

    if estimated_demand == 0 and avg_sales_per_day > 0:
        estimated_demand = 1

    # Safety stock covers supplier lead time rather than an entire monthly
    # horizon, keeping the buffer useful without systematically over-ordering.
    lead_time_end = next_start + timedelta(days=max(1, lead_time_days) - 1)
    lead_time_open_days = count_open_days(
        next_start, lead_time_end, closed_weekdays,
    )
    safety_stock = int(math.ceil(
        service_level_z * std_dev * math.sqrt(max(1, lead_time_open_days))
    )) if std_dev > 0 else 0

    # Quantity-bearing lots are authoritative. FEFO demand can consume units
    # due during the forecast; only the remainder is genuinely at risk and
    # should be removed from dependable stock for the reorder calculation.
    expiring_units = expiring_stock_units(product, next_end, stock_eod_end)
    expiry_units_at_risk = max(0, expiring_units - estimated_demand)
    usable_stock = max(0, stock_eod_end - expiry_units_at_risk)
    target_stock = estimated_demand + safety_stock
    needed = max(0, target_stock - usable_stock)

    product_expiry = getattr(product, "expiry_date", None)
    if product_expiry is not None:
        days_until_expiry = (product_expiry - next_start).days
    else:
        days_until_expiry = None

    if expiring_units > 0:
        expiry_risk_factor = 1.0
    elif days_until_expiry is not None and days_until_expiry < 30:
        expiry_risk_factor = 0.8
    elif days_until_expiry is not None and days_until_expiry < 90:
        expiry_risk_factor = 0.5
    else:
        expiry_risk_factor = 0.2

    # The previous optimizer charged the full leftover inventory cost to every
    # candidate order, including stock already owned. That could prefer zero or
    # oversized orders for the wrong reason. Use the standard order-up-to level
    # when unit economics are positive, then report projected gross profit.
    best_qty = needed if unit_margin > 0 else 0
    projected_sales = min(estimated_demand, usable_stock + best_qty)
    projected_profit = (
        projected_sales * unit_margin
        - expiry_units_at_risk * cost_per_unit * expiry_risk_factor
    )

    if avg_sales_per_day > 2.0:
        velocity_class = "fast"
    elif avg_sales_per_day >= 0.3:
        velocity_class = "moderate"
    elif avg_sales_per_day >= 0.01:
        velocity_class = "slow"
    else:
        velocity_class = "dead"

    recommendation = "Maintain current stock"

    if needed > 0 and unit_margin <= 0:
        recommendation = "Review price/cost before ordering"
    elif total_missed > 0 and usable_stock == 0 and best_qty > 0:
        recommendation = f"Immediate reorder: {best_qty} units"
    elif best_qty > 0:
        recommendation = f"Order {best_qty} units"
    elif velocity_class == "dead" and usable_stock > 0:
        recommendation = "Stop ordering (Dead Stock)"
    elif velocity_class == "slow" and usable_stock > max(3, estimated_demand * 3):
        recommendation = "Reduce stock levels"
    elif expiry_rate > (15 if velocity_class == "fast" else 30):
        recommendation = "High Expiry - Reduce Order Size"

    warnings = []
    if total_missed > 0:
        warnings.append(
            f"Missed {total_missed} sale unit(s) due to stockouts: "
            f"about ${estimated_revenue_lost:.2f} revenue and "
            f"${estimated_gross_profit_lost:.2f} gross profit opportunity lost."
        )

    expiry_warn_threshold = 15 if velocity_class == "fast" else (20 if velocity_class == "moderate" else 30)
    if expiry_rate > expiry_warn_threshold:
        warnings.append(f"Expiry rate is high (>{expiry_warn_threshold}%).")

    if velocity_class == "dead" and total_sold == 0:
        warnings.append("No sales in period (Potential Dead Stock).")

    if expiry_units_at_risk > 0:
        warnings.append(
            f"{expiry_units_at_risk} on-hand unit(s) may expire unsold by the forecast period end."
        )
    elif days_until_expiry is not None and 0 < days_until_expiry < 60:
        warnings.append(
            f"Stock approaches expiry in {days_until_expiry} days."
        )
    elif days_until_expiry is not None and days_until_expiry <= 0:
        warnings.append("Stock has passed expiry date.")

    if unit_margin <= 0:
        warnings.append("Sale price does not exceed unit cost; review pricing before reordering.")
    if forecast_confidence == "Low":
        warnings.append("Forecast confidence is low; use a longer history when possible.")

    trend_adjustment_pct = round((trend_ratio - 1.0) * 100, 1) if flat_projection > 0 else 0.0
    if trend_adjustment_pct >= 5:
        trend_label = "Rising"
    elif trend_adjustment_pct <= -5:
        trend_label = "Falling"
    else:
        trend_label = "Stable"

    return {
        "recommendation": recommendation,
        "suggested_order_quantity": best_qty,
        "expected_demand": estimated_demand,
        "projected_profit": round(projected_profit, 2),
        "actual_profit": round(period_profit, 2),
        "estimated_revenue_lost": round(estimated_revenue_lost, 2),
        "estimated_gross_profit_lost": round(estimated_gross_profit_lost, 2),
        "sell_through_rate": round(sell_through_rate, 1),
        "expiry_rate": round(expiry_rate, 1),
        "warnings": warnings,
        "velocity_class": velocity_class,
        "forecast_confidence": forecast_confidence,
        "confidence_score": round(confidence_score * 100),
        "trend_label": trend_label,
        "trend_adjustment_pct": trend_adjustment_pct,
        "safety_stock": safety_stock,
        "usable_stock": usable_stock,
        "expiring_stock_units": expiring_units,
        "expiry_units_at_risk": expiry_units_at_risk,
        "forecast_open_days": future_open_days,
        "debug": {
            "true_demand": true_demand,
            "missed": total_missed,
            "net_received": net_received,
            "gross_received": gross_received,
            "flat_avg_sales": round(flat_avg, 3),
            "weighted_avg_sales": round(w_avg, 3),
            "blended_avg_sales": round(avg_sales_per_day, 3),
            "demand_std_dev": round(std_dev, 3),
            "safety_stock": safety_stock,
            "lead_time_open_days": lead_time_open_days,
            "trend_slope_per_week": round(trend_slope, 3),
            "trend_adjustment_pct": trend_adjustment_pct,
            "expiry_risk_factor": expiry_risk_factor,
            "days_until_expiry": days_until_expiry,
            "expiring_stock_units": expiring_units,
            "expiry_units_at_risk": expiry_units_at_risk,
            "usable_stock": usable_stock,
        },
    }


def get_reorder_prediction(product, total_demand_60d: int,
                           weekly_demands=None,
                           monthly_demands=None,
                           days_lookback: int = 60,
                           lead_time_days: int = 7,
                           service_level_z: float = 1.65) -> dict:
    """
    Industry-standard reorder prediction aligned with recommend_inventory_action().

    Formulas used (matching the full recommendation engine):
      Safety stock  = Z × daily_σ × √(lead_time_days)        [Z=1.65 = 95% service level]
      ROP           = (adjusted_avg_daily × lead_time) + safety_stock
      Urgency       = triggered when current stock ≤ ROP (not just days-to-stockout)
      Suggested qty = ceil(adjusted_avg_daily × 30-day restock window)

    Inputs:
      weekly_demands:  list of (week_date, total_qty) sorted asc — last 60 days
                       → used for OLS trend (slope per week) + demand std_dev
      monthly_demands: list of (month_date, total_qty) — last 24 months
                       → used for month-of-year seasonal multiplier
    """
    from collections import defaultdict

    avg_daily = total_demand_60d / days_lookback if days_lookback > 0 else 0.0
    current_stock = max(product.quantity_in_stock or 0, 0)
    today = date.today()
    restock_target_days = 30

    # ── 1. Weekly stats: trend slope + demand std_dev ──────────────────────
    # Both come from the same weekly bucket data so we compute together.
    trend_slope_per_week = 0.0
    trend_label          = None
    daily_std_dev        = 0.0

    if weekly_demands and len(weekly_demands) >= 2:
        weekly_totals = [t for (_, t) in sorted(weekly_demands, key=lambda x: x[0])]
        n = len(weekly_totals)

        # Std dev of weekly demand → convert to daily (σ_daily ≈ σ_weekly / √7)
        w_mean        = sum(weekly_totals) / n
        w_variance    = sum((x - w_mean) ** 2 for x in weekly_totals) / n
        daily_std_dev = (w_variance ** 0.5) / (7 ** 0.5)

        # OLS trend slope — needs ≥ 3 weeks
        if n >= 3:
            sum_i  = n * (n - 1) // 2
            sum_y  = sum(weekly_totals)
            sum_iy = sum(i * y for i, y in enumerate(weekly_totals))
            sum_i2 = sum(i * i for i in range(n))
            denom  = n * sum_i2 - sum_i ** 2
            if denom:
                trend_slope_per_week = (n * sum_iy - sum_i * sum_y) / denom

            if trend_slope_per_week > 0.3:
                trend_label = 'rising'
            elif trend_slope_per_week < -0.3:
                trend_label = 'falling'
            else:
                trend_label = 'stable'

    # ── 2. Trend-adjusted demand (capped at ±40% of base — same cap as full engine) ──
    if avg_daily > 0 and trend_slope_per_week:
        raw_adj      = (trend_slope_per_week / 7) * (restock_target_days / 2)
        trend_adj    = max(-0.4 * avg_daily, min(0.4 * avg_daily, raw_adj))
    else:
        trend_adj    = 0.0
    trend_avg_daily = max(0.0, avg_daily + trend_adj)

    # ── 3. Seasonality: month-of-year multiplier ────────────────────────────
    seasonal_mult  = 1.0
    seasonal_label = None
    if monthly_demands and len(monthly_demands) >= 3:
        by_month = defaultdict(list)
        for month_date, total in monthly_demands:
            by_month[month_date.month].append(total)

        month_avgs          = {m: sum(v) / len(v) for m, v in by_month.items()}
        overall_monthly_avg = sum(month_avgs.values()) / len(month_avgs)
        coming_month        = ((today.replace(day=28) + timedelta(days=4)).replace(day=1)).month

        if overall_monthly_avg > 0 and coming_month in month_avgs:
            raw_mult      = month_avgs[coming_month] / overall_monthly_avg
            seasonal_mult = max(0.5, min(2.0, raw_mult))
            if seasonal_mult >= 1.2:
                seasonal_label = 'peak month'
            elif seasonal_mult <= 0.8:
                seasonal_label = 'slow month'

    # ── 4. Forward-looking demand (trend + seasonality combined) ────────────
    adjusted_avg_daily = trend_avg_daily * seasonal_mult
    effective_daily    = adjusted_avg_daily if adjusted_avg_daily > 0 else avg_daily

    # ── 5. Safety stock: Z × daily_σ × √(lead_time)  [matches full engine] ─
    safety_stock = (
        math.ceil(service_level_z * daily_std_dev * math.sqrt(lead_time_days))
        if daily_std_dev > 0 else 0
    )

    # ── 6. Reorder Point (ROP) ───────────────────────────────────────────────
    # ROP = demand consumed during lead time + safety stock buffer
    # Uses effective (trend+seasonal) demand for the lead-time window
    rop = math.ceil(effective_daily * lead_time_days) + safety_stock if effective_daily > 0 else safety_stock

    # ── 7. Days until stock hits ROP (the real order trigger) ───────────────
    # This replaces the old "days to stockout − lead_time" heuristic.
    if effective_daily > 0 and current_stock > rop:
        days_to_rop  = (current_stock - rop) / effective_daily
        reorder_date = today + timedelta(days=int(days_to_rop))
    elif effective_daily > 0:
        days_to_rop  = 0.0   # already at or below ROP
        reorder_date = today
    else:
        days_to_rop  = None
        reorder_date = None

    # ── 8. Velocity classification ──────────────────────────────────────────
    if avg_daily > 2.0:       velocity = 'fast'
    elif avg_daily >= 0.3:    velocity = 'moderate'
    elif avg_daily >= 0.01:   velocity = 'slow'
    else:                     velocity = 'dead'

    # ── 9. Urgency (ROP-based, not stockout-based) ──────────────────────────
    # critical → stock is already at or below ROP  (order now, lead time eats into safety stock)
    # warning  → stock will hit ROP within 7 days
    # ok       → plenty of time before ROP
    if velocity == 'dead' or effective_daily == 0:
        urgency = 'none'
    elif current_stock <= rop:
        urgency = 'critical'
    elif days_to_rop is not None and days_to_rop <= 7:
        urgency = 'warning'
    else:
        urgency = 'ok'

    # ── 10. Suggested order quantity (30-day adjusted coverage) ─────────────
    suggested_qty = max(1, math.ceil(effective_daily * restock_target_days)) if effective_daily > 0 else 0

    return {
        'avg_daily':      round(avg_daily, 2),
        'adjusted_daily': round(effective_daily, 2),
        'safety_stock':   safety_stock,
        'rop':            rop,
        'days_to_rop':    round(days_to_rop, 1) if days_to_rop is not None else None,
        'reorder_date':   reorder_date,
        'velocity':       velocity,
        'urgency':        urgency,
        'suggested_qty':  suggested_qty,
        'trend_label':    trend_label,       # 'rising' | 'falling' | 'stable' | None
        'trend_slope':    round(trend_slope_per_week, 2),
        'seasonal_mult':  round(seasonal_mult, 2),
        'seasonal_label': seasonal_label,    # 'peak month' | 'slow month' | None
    }
