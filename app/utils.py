from decimal import Decimal
from typing import List, Dict
from datetime import datetime, timedelta, date, time
import math
from collections import defaultdict
from django.utils.timezone import make_aware
from django.db.models import Sum
from .models import StockChange, Product

STOCK_SIGN = {
    "checkin": +1,
    "error_add": +1,
    "checkout": -1,
    "expired": -1,
    "error_subtract": -1,
    "checkin_delete1": -1,
}

def get_stock_eod(product: Product, day: date) -> int:
    """
    Stock level at END OF `day` (EOD), based on current Product.quantity_in_stock
    and rolling back all StockChange events AFTER `day`.
    """
    after_rows = (
        StockChange.objects
        .filter(product=product, timestamp__date__gt=day)
        .values("change_type")
        .annotate(total=Sum("quantity"))
    )

    net_after = 0
    for r in after_rows:
        ct = r["change_type"]
        qty = int(r["total"] or 0)
        net_after += STOCK_SIGN.get(ct, 0) * abs(qty)

    # If net_after is +5, that means stock increased by 5 after `day`,
    # so EOD(day) must have been current - 5.
    eod = int(product.quantity_in_stock) - net_after
    return max(0, eod)

CLOSED_WEEKDAYS = {6}  # Sunday (Mon=0 ... Sun=6)
  
def count_open_days(start_d: date, end_d: date, closed_weekdays=CLOSED_WEEKDAYS) -> int:
    """Counts days in [start_d, end_d] where weekday not in closed_weekdays."""
    if end_d < start_d:
        return 0
    n = 0
    d = start_d
    while d <= end_d:
        if d.weekday() not in closed_weekdays:
            n += 1
        d += timedelta(days=1)
    return n

def recalculate_order_totals(order):
    order_details = order.details.all()
    total_price_before_tax = Decimal("0.00")
    total_tax = Decimal("0.00")
    tax_rate = Decimal("0.13")  # 13% HST

    for detail in order_details:
        item_price = detail.product.price * detail.quantity
        total_price_before_tax += item_price

        if getattr(detail.product, "taxable", False):
            total_tax += item_price * tax_rate

    total_price_after_tax = total_price_before_tax + total_tax
    order.total_price = total_price_after_tax
    order.save()

    return total_price_before_tax, total_price_after_tax


# --- Data models used by the predictive algorithm ---------------------------


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


# --- Helper to convert StockChange rows into purchase/sale/expiry history ---


def get_product_stock_records(product: Product, start_date: str, end_date: str):
    """
    Build structured Purchase / Sale / Expiry records from StockChange
    between start_date and end_date (inclusive).
    """
    start = make_aware(
        datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), time.min)
    )
    end = make_aware(
        datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), time.max)
    )

    changes = StockChange.objects.filter(product=product, timestamp__range=(start, end))

    purchases: List[PurchaseRecord] = []

    # Normal checkins or manual add corrections
    for c in changes.filter(change_type__in=["checkin", "error_add"]):
        purchases.append(
            PurchaseRecord(
                quantity=int(c.quantity),
                purchase_date=c.timestamp.strftime("%Y-%m-%d"),
            )
        )

    # Manual subtractions / reversals are treated as negative purchases
    for c in changes.filter(change_type__in=["error_subtract", "checkin_delete1"]):
        purchases.append(
            PurchaseRecord(
                quantity=-abs(int(c.quantity)),
                purchase_date=c.timestamp.strftime("%Y-%m-%d"),
            )
        )

    sales: List[SaleRecord] = [
        SaleRecord(
            quantity=abs(int(c.quantity)),
            sale_date=c.timestamp.strftime("%Y-%m-%d"),
        )
        for c in changes.filter(change_type="checkout")
    ]

    expiries: List[ExpiryRecord] = [
        ExpiryRecord(
            quantity=abs(int(c.quantity)),
            expiry_date=c.timestamp.strftime("%Y-%m-%d"),
        )
        for c in changes.filter(change_type="expired")
    ]

    # DEBUG – you can comment this section out later
    print("=== STOCK RECORDS DEBUG ===")
    print("Product:", product.name, product.barcode)
    print("Timeframe:", start_date, "->", end_date)
    print("Purchases:", [(p.quantity, p.purchase_date.date()) for p in purchases])
    print("Sales:", [(s.quantity, s.sale_date.date()) for s in sales])
    print("Expiries:", [(e.quantity, e.expiry_date.date()) for e in expiries])

    return purchases, sales, expiries


# --- Main inventory recommendation algorithm --------------------------------


def recommend_inventory_action(
    product: Product,
    purchase_history,
    sale_history,
    expiry_history,
    timeframe_start: str,
    timeframe_end: str,
    cost_per_unit: float,
    price_per_unit: float,
    granularity: str = "month",
    closed_weekdays=CLOSED_WEEKDAYS,
) -> dict:

    start_dt = datetime.strptime(timeframe_start, "%Y-%m-%d")
    end_dt   = datetime.strptime(timeframe_end, "%Y-%m-%d")

    start_d = start_dt.date()
    end_d   = end_dt.date()

    # Filter records by timeframe (your record objects store datetimes)
    purchases = [p for p in purchase_history if start_dt <= p.purchase_date <= end_dt]
    sales     = [s for s in sale_history     if start_dt <= s.sale_date     <= end_dt]
    expiries  = [e for e in expiry_history   if start_dt <= e.expiry_date   <= end_dt]

    total_bought  = sum(int(p.quantity) for p in purchases)         # can include negatives
    total_sold    = sum(int(s.quantity) for s in sales)
    total_expired = sum(int(e.quantity) for e in expiries)

    # ✅ Opening stock fix (EOD of day before the range begins)
    opening_stock = get_stock_eod(product, start_d - timedelta(days=1))

    # ✅ On-hand at end of selected range (EOD end_d)
    stock_eod_end = get_stock_eod(product, end_d)

    # ✅ Net available should include OPENING stock, not just purchases-in-range
    net_available = max(0, opening_stock + total_bought - total_expired)

    # Sell-through should not be forced to 0 when purchases=0 but opening stock existed
    sell_through_rate = (total_sold / net_available * 100) if net_available > 0 else 0

    # Expiry rate: expiries relative to stock that could have expired (opening + bought)
    expiry_base = max(0, opening_stock + total_bought)
    expiry_rate = (total_expired / expiry_base * 100) if expiry_base > 0 else 0

    # Keep your "timeframe profit" behavior (matches chart cashflow style)
    profit = (total_sold * price_per_unit) - (max(total_bought, 0) * cost_per_unit)
    wastage_cost = total_expired * cost_per_unit

    # ✅ OPEN-day based average sales
    open_days_in_range = count_open_days(start_d, end_d, closed_weekdays=closed_weekdays)
    avg_sales_per_open_day = (total_sold / open_days_in_range) if open_days_in_range > 0 else 0

    # ✅ Determine "next period" open days based on granularity
    next_start = end_d + timedelta(days=1)
    if granularity == "day":
        next_end = next_start
    elif granularity == "week":
        next_end = next_start + timedelta(days=6)
    else:
        next_end = next_start + timedelta(days=29)

    open_days_in_next_period = count_open_days(next_start, next_end, closed_weekdays=closed_weekdays)

    # ✅ Demand estimate: use CEIL so small but real demand doesn't collapse to 0
    raw_demand = avg_sales_per_open_day * open_days_in_next_period
    estimated_demand = int(math.ceil(raw_demand)) if raw_demand > 0 else 0

    # ✅ Suggested order should consider on-hand stock you already have
    # Basic “needed units”
    needed = max(0, estimated_demand - stock_eod_end)

    # --- Optimization: test around "needed" (keeps your spirit, but respects on-hand) ---
    best_quantity = 0
    best_profit = float("-inf")

    # Search space (still safe if estimated_demand == 0)
    max_test_qty = max(10, needed * 3 + 1)

    for order_qty in range(0, max_test_qty, 5):
        available = stock_eod_end + order_qty
        expected_sales = min(available, estimated_demand)
        # treat leftover as potential waste/carry cost (simple model)
        leftover = max(0, available - estimated_demand)

        projected_profit = (
            (expected_sales * price_per_unit)
            - (order_qty * cost_per_unit)
            - (leftover * cost_per_unit)  # conservative: leftover "costs" you
        )

        if projected_profit > best_profit:
            best_profit = projected_profit
            best_quantity = order_qty

    # Decision logic (now based on fixed sell-through + expiry)
    if sell_through_rate < 40 and expiry_rate > 20:
        recommendation = "Stop or drastically reduce ordering"
    elif sell_through_rate > 80 and expiry_rate < 10:
        recommendation = "Increase ordering slightly"
    elif expiry_rate > 15:
        recommendation = "Reduce order quantity to cut wastage"
    elif profit < 0:
        recommendation = "Stop stocking — unprofitable"
    else:
        recommendation = "Maintain current order levels"

    warnings = []
    if expiry_rate > 20:
        warnings.append("High expiry rate detected.")
    if profit < 0:
        warnings.append("Negative profit in selected timeframe.")
    if sell_through_rate < 30:
        warnings.append("Very low sell-through rate — possible dead stock.")
    if open_days_in_range == 0:
        warnings.append("No open days in selected timeframe (closed days excluded).")
    if stock_eod_end > 0 and estimated_demand == 0:
        warnings.append("Demand estimate is 0 while stock is on-hand — consider shorter date range for clearer signal.")

    # ✅ DEBUG PRINTS (remove later)
    print("=== RECOMMENDATION DEBUG ===")
    print("Product:", product.name, product.barcode)
    print("Range:", start_d, "->", end_d, "| Granularity:", granularity)
    print("opening_stock (EOD day-before-start):", opening_stock)
    print("stock_eod_end:", stock_eod_end)
    print("total_bought / sold / expired:", total_bought, total_sold, total_expired)
    print("net_available:", net_available)
    print("open_days_in_range:", open_days_in_range, "avg_sales_per_open_day:", round(avg_sales_per_open_day, 4))
    print("open_days_in_next_period:", open_days_in_next_period, "estimated_demand:", estimated_demand)
    print("best_quantity:", best_quantity, "best_profit:", round(best_profit, 2))
    print("sell_through_rate:", round(sell_through_rate, 2), "expiry_rate:", round(expiry_rate, 2))
    print("============================")

    return {
        "recommendation": recommendation,
        "suggested_order_quantity": int(best_quantity),
        "expected_demand": int(estimated_demand),
        "projected_profit": round(best_profit, 2),
        "sell_through_rate": round(sell_through_rate, 2),
        "expiry_rate": round(expiry_rate, 2),
        "actual_profit": round(profit, 2),
        "wastage_cost": round(wastage_cost, 2),
        "warnings": warnings,
        "debug": {
            "opening_stock": opening_stock,
            "stock_eod_end": stock_eod_end,
            "total_bought": total_bought,
            "total_sold": total_sold,
            "total_expired": total_expired,
            "net_available": net_available,
            "open_days_in_range": open_days_in_range,
            "avg_sales_per_open_day": round(avg_sales_per_open_day, 4),
            "open_days_in_next_period": open_days_in_next_period,
            "raw_demand": round(raw_demand, 4),
        }
    }
