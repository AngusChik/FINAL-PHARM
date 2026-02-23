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

    # ✅ TRUE DEMAND = Sold + Missed (crucial for accurate reordering)
    true_demand = total_sold + total_missed

    stock_eod_end = get_stock_eod(product, end_dt.date())
    opening_stock = get_stock_eod(product, start_dt.date() - timedelta(days=1))
    
    # Net available stock during period
    net_available = max(0, opening_stock + total_bought - total_sold - total_expired)

    # Rates
    sell_through_rate = (total_sold / net_available * 100) if net_available > 0 else 0
    expiry_base = max(0, opening_stock + total_bought)
    expiry_rate = (total_expired / expiry_base * 100) if expiry_base > 0 else 0

    # Profit
    profit = (total_sold * price_per_unit) - (max(total_bought, 0) * cost_per_unit)
    
    # Demand Projection (Open Days)
    open_days = count_open_days(start_dt.date(), end_dt.date(), closed_weekdays)
    avg_sales_per_day = (true_demand / open_days) if open_days > 0 else 0

    # Next Period Length
    next_start = end_dt.date() + timedelta(days=1)
    if granularity == "day": next_end = next_start
    elif granularity == "week": next_end = next_start + timedelta(days=6)
    else: next_end = next_start + timedelta(days=29)
    
    future_open_days = count_open_days(next_start, next_end, closed_weekdays)
    
    # ✅ Use math.ceil so even 0.1 sales/day suggests keeping at least 1 unit
    estimated_demand = int(math.ceil(avg_sales_per_day * future_open_days))

    # Basic Need
    needed = max(0, estimated_demand - stock_eod_end)
    
    # --- Optimization for Low Volume ---
    best_qty = 0
    best_profit = float("-inf")
    
    # ✅ Search Space: Tighter. Stop ordering 5 if we only need 1.
    # Max test is roughly "needed" + a small buffer (e.g. 3 units), min 5 to check profitability
    max_test_qty = max(5, needed + 4) 

    # ✅ STEP BY 1: Essential for low-volume pharmacy items
    for qty in range(0, max_test_qty, 1): 
        av = stock_eod_end + qty
        exp_sales = min(av, estimated_demand)
        leftover = max(0, av - estimated_demand)
        
        # Simple profit model: Sales - Cost of Goods - (Holding Cost/Risk of Leftover)
        # We treat leftover cost aggressively to discourage overstocking slow meds
        p = (exp_sales * price_per_unit) - (qty * cost_per_unit) - (leftover * cost_per_unit * 0.5)
        
        if p > best_profit:
            best_profit = p
            best_qty = qty

    # --- Recommendation Logic (Tuned for Pharmacy) ---
    recommendation = "Maintain current stock"
    
    if total_missed > 0 and stock_eod_end == 0:
        recommendation = "Immediate Reorder (Stockouts)"
    elif best_qty > 0:
        recommendation = f"Order {best_qty} units"
    elif profit < 0 and total_sold == 0 and stock_eod_end > 0:
        recommendation = "Stop ordering (Dead Stock)"
    elif sell_through_rate < 10 and stock_eod_end > 5:
        # Only flag slow movers if we are sitting on > 5 units
        recommendation = "Reduce stock levels"
    elif expiry_rate > 25:
        recommendation = "High Expiry - Reduce Order Size"

    warnings = []
    if total_missed > 0: warnings.append(f"Missed {total_missed} sales due to stockouts.")
    if expiry_rate > 20: warnings.append("Expiry rate is high (>20%).")
    if sell_through_rate < 10 and total_sold == 0: warnings.append("No sales in period (Potential Dead Stock).")

    return {
        "recommendation": recommendation,
        "suggested_order_quantity": best_qty,
        "expected_demand": estimated_demand,
        "projected_profit": round(best_profit, 2),
        "actual_profit": round(profit, 2),
        "sell_through_rate": round(sell_through_rate, 1),
        "expiry_rate": round(expiry_rate, 1),
        "warnings": warnings,
        "debug": {"true_demand": true_demand, "missed": total_missed, "avg_sales": round(avg_sales_per_day, 3)}
    }