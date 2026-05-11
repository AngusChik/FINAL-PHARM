from decimal import Decimal
from typing import List
from datetime import datetime, timedelta, date, time
import math
from django.utils.timezone import make_aware
from django.db.models import Sum
from .models import StockChange, Product
from django.core.cache import cache

TAX_RATE = Decimal("0.13")

STOCK_SIGN = {
    "checkin": +1,
    "error_add": +1,
    "checkout": -1,
    "expired": -1,
    "error_subtract": -1,
    "checkin_delete1": -1,
    # "checkout_unfulfilled" is excluded (0) as it doesn't change physical stock
}

def get_stock_eod(product: Product, day: date) -> int:
    """
    Returns stock level at END OF `day` (EOD).
    """
    cache_key = f"stock_eod:{product.product_id}:{day.isoformat()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    after_rows = (
        StockChange.objects
        .filter(product=product, timestamp__date__gt=day)
        .values("change_type")
        .annotate(total=Sum("quantity"))
    )

    net_after = 0
    for row in after_rows:
        change_type = row["change_type"]
        qty = abs(int(row["total"] or 0))
        sign = STOCK_SIGN.get(change_type, 0)
        net_after += sign * qty

    current_stock = int(product.quantity_in_stock or 0)
    eod_stock = current_stock - net_after
    result = max(0, eod_stock)
    
    cache.set(cache_key, result, 60 * 60)
    return result

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

def recalculate_order_totals(order):
    order_details = order.details.select_related('product').all()
    total_price_before_tax = Decimal("0.00")
    total_tax = Decimal("0.00")
    tax_rate = TAX_RATE

    for detail in order_details:
        # Use stored price on OrderDetail (survives product deletion)
        item_price = detail.price * detail.quantity
        total_price_before_tax += item_price
        # Only apply tax if the product still exists and is taxable
        if detail.product and getattr(detail.product, "taxable", False):
            total_tax += item_price * tax_rate

    total_price_after_tax = total_price_before_tax + total_tax
    order.total_price = total_price_after_tax
    order.save()
    return total_price_before_tax, total_price_after_tax

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
    Returns: purchases, sales, expiries, unfulfilled
    """
    start = make_aware(datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), time.min))
    end = make_aware(datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), time.max))

    changes = StockChange.objects.filter(product=product, timestamp__range=(start, end))

    purchases = []
    for c in changes.filter(change_type__in=["checkin", "error_add"]):
        purchases.append(PurchaseRecord(int(c.quantity), c.timestamp.strftime("%Y-%m-%d")))
    for c in changes.filter(change_type__in=["error_subtract", "checkin_delete1"]):
        purchases.append(PurchaseRecord(-abs(int(c.quantity)), c.timestamp.strftime("%Y-%m-%d")))

    sales = [SaleRecord(abs(int(c.quantity)), c.timestamp.strftime("%Y-%m-%d")) 
             for c in changes.filter(change_type="checkout")]
    
    # ✅ Unfulfilled orders (stockouts)
    unfulfilled = [SaleRecord(abs(int(c.quantity)), c.timestamp.strftime("%Y-%m-%d")) 
                   for c in changes.filter(change_type="checkout_unfulfilled")]

    expiries = [ExpiryRecord(abs(int(c.quantity)), c.timestamp.strftime("%Y-%m-%d")) 
                for c in changes.filter(change_type="expired")]

    return purchases, sales, expiries, unfulfilled

def weighted_avg_daily_demand(
    sales_records, missed_records, start_dt, end_dt, closed_weekdays, half_life_days=30
):
    """
    Exponential-decay weighted average of daily demand.
    Recent days carry more weight than older days (half-life = 30 days by default).
    Returns (weighted_avg, std_dev, daily_demands_dict).
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
            daily[sd] += abs(int(s.quantity))
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

    w_avg = sum(d * w for d, w in zip(demands, weights)) / w_sum

    # Weighted standard deviation
    variance = sum(w * (d - w_avg) ** 2 for d, w in zip(demands, weights)) / w_sum
    std_dev = variance ** 0.5

    return w_avg, std_dev, daily


def compute_demand_trend(sales_records, missed_records, start_dt, end_dt, closed_weekdays):
    """
    Simple linear regression slope on weekly demand buckets.
    Returns slope per week (0 if fewer than 3 weeks of data).
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
            daily[sd] += abs(int(s.quantity))
    for u in missed_records:
        ud = u.sale_date.date() if hasattr(u.sale_date, 'date') else u.sale_date
        if ud in daily:
            daily[ud] += abs(int(u.quantity))

    if not daily:
        return 0.0

    # Bucket into weeks
    sorted_dates = sorted(daily.keys())
    weeks = []
    week_total = 0
    week_start = sorted_dates[0]
    for dt in sorted_dates:
        if (dt - week_start).days >= 7:
            weeks.append(week_total)
            week_total = 0
            week_start = dt
        week_total += daily[dt]
    weeks.append(week_total)  # last partial week

    n = len(weeks)
    if n < 3:
        return 0.0

    # Ordinary least squares: slope = (N*sum(i*y) - sum(i)*sum(y)) / (N*sum(i^2) - sum(i)^2)
    sum_i = sum(range(n))
    sum_y = sum(weeks)
    sum_iy = sum(i * y for i, y in enumerate(weeks))
    sum_i2 = sum(i * i for i in range(n))

    denom = n * sum_i2 - sum_i * sum_i
    if denom == 0:
        return 0.0

    slope = (n * sum_iy - sum_i * sum_y) / denom
    return slope


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
) -> dict:

    start_dt = datetime.strptime(timeframe_start, "%Y-%m-%d")
    end_dt   = datetime.strptime(timeframe_end, "%Y-%m-%d")

    # Filter records
    sales = [s for s in sale_history if start_dt <= s.sale_date <= end_dt]
    missed = [u for u in unfulfilled_history if start_dt <= u.sale_date <= end_dt]
    purchases = [p for p in purchase_history if start_dt <= p.purchase_date <= end_dt]
    expiries = [e for e in expiry_history if start_dt <= e.expiry_date <= end_dt]

    total_sold = sum(int(s.quantity) for s in sales)
    total_missed = sum(int(u.quantity) for u in missed)
    total_bought = sum(int(p.quantity) for p in purchases)
    total_expired = sum(int(e.quantity) for e in expiries)

    true_demand = total_sold + total_missed

    stock_eod_end = get_stock_eod(product, end_dt.date())
    opening_stock = get_stock_eod(product, start_dt.date() - timedelta(days=1))

    net_available = max(0, opening_stock + total_bought - total_sold - total_expired)

    # Rates
    sell_through_rate = (total_sold / net_available * 100) if net_available > 0 else 0
    expiry_base = max(0, opening_stock + total_bought)
    expiry_rate = (total_expired / expiry_base * 100) if expiry_base > 0 else 0

    # Profit
    profit = (total_sold * price_per_unit) - (max(total_bought, 0) * cost_per_unit)

    open_days = count_open_days(start_dt.date(), end_dt.date(), closed_weekdays)
    flat_avg = (true_demand / open_days) if open_days > 0 else 0

    # --- Improvement 1 & 2: Recency-Weighted Demand + Variance ---
    w_avg, std_dev, daily_demands = weighted_avg_daily_demand(
        sales, missed, start_dt, end_dt, closed_weekdays
    )
    avg_sales_per_day = w_avg if w_avg > 0 else flat_avg

    # --- Next Period Length ---
    next_start = end_dt.date() + timedelta(days=1)
    if granularity == "day": next_end = next_start
    elif granularity == "week": next_end = next_start + timedelta(days=6)
    else: next_end = next_start + timedelta(days=29)

    future_open_days = count_open_days(next_start, next_end, closed_weekdays)

    # --- Improvement 4: Trend Detection ---
    trend_slope = compute_demand_trend(sales, missed, start_dt, end_dt, closed_weekdays)
    trend_daily = trend_slope / 7.0  # convert weekly slope to daily

    # Trended projection: sum demand for each future day with trend adjustment
    trended_total = 0.0
    for day_i in range(1, future_open_days + 1):
        trended_total += max(0, avg_sales_per_day + trend_daily * day_i)

    flat_projection = avg_sales_per_day * future_open_days
    # Cap trend adjustment at +/-40% of flat projection to guard against noise
    if flat_projection > 0:
        trend_ratio = trended_total / flat_projection
        trend_ratio = max(0.6, min(1.4, trend_ratio))
        estimated_demand = int(math.ceil(flat_projection * trend_ratio))
    else:
        estimated_demand = int(math.ceil(trended_total))

    # Ensure at least 1 unit if there's any demand signal
    if estimated_demand == 0 and avg_sales_per_day > 0:
        estimated_demand = 1

    # --- Improvement 2: Safety Stock ---
    safety_stock = int(math.ceil(service_level_z * std_dev * (future_open_days ** 0.5)))

    needed = max(0, estimated_demand + safety_stock - stock_eod_end)

    # --- Improvement 3: Expiry-Aware Holding Cost ---
    if product.expiry_date is not None:
        days_until_expiry = (product.expiry_date - next_end).days
    else:
        days_until_expiry = 180  # conservative default

    if days_until_expiry <= 0:
        expiry_risk_factor = 1.0
    elif days_until_expiry < 30:
        expiry_risk_factor = 0.8
    elif days_until_expiry < 90:
        expiry_risk_factor = 0.5
    else:
        expiry_risk_factor = 0.2

    # --- Improvement 5: Expanded Search Space ---
    max_test_qty = max(10, int(needed + 3 * std_dev) + 2)
    max_test_qty = min(max_test_qty, max(50, needed * 3))

    # --- Optimization Loop ---
    best_qty = 0
    best_profit = float("-inf")

    for qty in range(0, max_test_qty, 1):
        av = stock_eod_end + qty
        exp_sales = min(av, estimated_demand)
        leftover = max(0, av - estimated_demand)

        p = (exp_sales * price_per_unit) - (qty * cost_per_unit) - (leftover * cost_per_unit * expiry_risk_factor)

        if p > best_profit:
            best_profit = p
            best_qty = qty

    # Ensure we order at least enough for safety stock if profitable
    if best_qty < needed and best_profit >= 0 and needed > 0:
        # Check if ordering 'needed' is still profitable
        av = stock_eod_end + needed
        exp_sales = min(av, estimated_demand)
        leftover = max(0, av - estimated_demand)
        p_needed = (exp_sales * price_per_unit) - (needed * cost_per_unit) - (leftover * cost_per_unit * expiry_risk_factor)
        if p_needed >= 0:
            best_qty = needed
            best_profit = p_needed

    # --- Improvement 6: Velocity Classification ---
    if avg_sales_per_day > 2.0:
        velocity_class = "fast"
    elif avg_sales_per_day >= 0.3:
        velocity_class = "moderate"
    elif avg_sales_per_day >= 0.01:
        velocity_class = "slow"
    else:
        velocity_class = "dead"

    # --- Recommendation Logic (Velocity-Aware) ---
    recommendation = "Maintain current stock"

    if total_missed > 0 and stock_eod_end == 0:
        recommendation = "Immediate Reorder (Stockouts)"
    elif best_qty > 0:
        recommendation = f"Order {best_qty} units"
    elif velocity_class == "dead" and stock_eod_end > 0:
        recommendation = "Stop ordering (Dead Stock)"
    elif velocity_class == "slow" and stock_eod_end > max(3, estimated_demand * 3):
        recommendation = "Reduce stock levels"
    elif expiry_rate > (15 if velocity_class == "fast" else 30):
        recommendation = "High Expiry - Reduce Order Size"

    # --- Warnings (Velocity-Scaled) ---
    warnings = []
    if total_missed > 0:
        warnings.append(f"Missed {total_missed} sales due to stockouts.")

    expiry_warn_threshold = 15 if velocity_class == "fast" else (20 if velocity_class == "moderate" else 30)
    if expiry_rate > expiry_warn_threshold:
        warnings.append(f"Expiry rate is high (>{expiry_warn_threshold}%).")

    if velocity_class == "dead" and total_sold == 0:
        warnings.append("No sales in period (Potential Dead Stock).")

    if days_until_expiry < 60 and days_until_expiry > 0:
        warnings.append(f"Stock approaching expiry in {days_until_expiry} days — consider promotional pricing.")
    elif days_until_expiry <= 0 and product.expiry_date is not None:
        warnings.append("Stock has passed expiry date.")

    trend_adjustment_pct = round((trend_ratio - 1.0) * 100, 1) if flat_projection > 0 else 0.0

    return {
        "recommendation": recommendation,
        "suggested_order_quantity": best_qty,
        "expected_demand": estimated_demand,
        "projected_profit": round(best_profit, 2),
        "actual_profit": round(profit, 2),
        "sell_through_rate": round(sell_through_rate, 1),
        "expiry_rate": round(expiry_rate, 1),
        "warnings": warnings,
        "velocity_class": velocity_class,
        "debug": {
            "true_demand": true_demand,
            "missed": total_missed,
            "flat_avg_sales": round(flat_avg, 3),
            "weighted_avg_sales": round(w_avg, 3),
            "demand_std_dev": round(std_dev, 3),
            "safety_stock": safety_stock,
            "trend_slope_per_week": round(trend_slope, 3),
            "trend_adjustment_pct": trend_adjustment_pct,
            "expiry_risk_factor": expiry_risk_factor,
            "days_until_expiry": days_until_expiry,
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