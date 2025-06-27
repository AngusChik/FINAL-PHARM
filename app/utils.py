from decimal import Decimal
from typing import List, Dict
from datetime import datetime, time
from .models import StockChange, Product
from django.utils.timezone import make_aware

def recalculate_order_totals(order):
    order_details = order.details.all()
    total_price_before_tax = Decimal('0.00')
    total_tax = Decimal('0.00')
    tax_rate = Decimal('0.13')  # Assuming 13% tax

    for detail in order_details:
        item_price = detail.product.price * detail.quantity
        total_price_before_tax += item_price

        # Apply tax if the product is taxable
        if detail.product.taxable:
            total_tax += item_price * tax_rate

    total_price_after_tax = total_price_before_tax + total_tax

    # Update the order with the calculated totals
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


# --- Main Predictive Algorithm ---

def recommend_inventory_action(
    purchase_history: List[PurchaseRecord],
    sale_history: List[SaleRecord],
    expiry_history: List[ExpiryRecord],
    timeframe_start: str,
    timeframe_end: str,
    cost_per_unit: float,
    price_per_unit: float,
    granularity: str = "month"
) -> Dict:

    start_date = datetime.strptime(timeframe_start, "%Y-%m-%d")
    end_date = datetime.strptime(timeframe_end, "%Y-%m-%d")

    # Filter records by timeframe
    purchases = [p for p in purchase_history if start_date <= p.purchase_date <= end_date]
    sales = [s for s in sale_history if start_date <= s.sale_date <= end_date]
    expiries = [e for e in expiry_history if start_date <= e.expiry_date <= end_date]

    total_bought = sum(p.quantity for p in purchases)
    total_sold = sum(s.quantity for s in sales)
    total_expired = sum(e.quantity for e in expiries)

    net_available = total_bought - total_expired
    sell_through_rate = (total_sold / net_available * 100) if net_available > 0 else 0
    expiry_rate = (total_expired / total_bought * 100) if total_bought > 0 else 0
    profit = (total_sold * price_per_unit) - (total_bought * cost_per_unit)
    wastage_cost = total_expired * cost_per_unit

    days = (end_date - start_date).days + 1

    # Calculate average sales per day
    if sales and days > 0:
        avg_sales_per_day = total_sold / days
    else:
        avg_sales_per_day = 0

     # Convert average daily sales to expected demand based on granularity
    if granularity == "day":
        estimated_demand = int(avg_sales_per_day)
    elif granularity == "week":
        estimated_demand = int(avg_sales_per_day * 7)
    else:  # month or default
        estimated_demand = int(avg_sales_per_day * 30)

    # Mathematical optimization: test multiple order quantities
    best_quantity = 0
    best_profit = float('-inf')

    for order_qty in range(0, max(10, estimated_demand * 2 + 1), 5):  # test every 5 units
        expected_sales = min(order_qty, estimated_demand)
        expected_expiry = max(0, order_qty - estimated_demand)
        projected_profit = (expected_sales * price_per_unit) - (order_qty * cost_per_unit) - (expected_expiry * cost_per_unit)

        if projected_profit > best_profit:
            best_profit = projected_profit
            best_quantity = order_qty

    # Decision logic
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

    # Optional warnings
    warnings = []
    if expiry_rate > 20:
        warnings.append("High expiry rate detected.")
    if profit < 0:
        warnings.append("Negative profit in selected timeframe.")
    if sell_through_rate < 30:
        warnings.append("Very low sell-through rate — possible dead stock.")

    return {
        "recommendation": recommendation,
        "suggested_order_quantity": best_quantity,
        "expected_demand": estimated_demand,
        "projected_profit": round(best_profit, 2),
        "sell_through_rate": round(sell_through_rate, 2),
        "expiry_rate": round(expiry_rate, 2),
        "actual_profit": round(profit, 2),
        "wastage_cost": round(wastage_cost, 2),
        "warnings": warnings
    }

def get_product_stock_records(product: Product, start_date: str, end_date: str):
    start = make_aware(datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), time.min))
    end = make_aware(datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), time.max))

    for sc in StockChange.objects.filter(product_id=9313):
        print(f"ID: {sc.id} | {sc.timestamp.isoformat()} | {sc.change_type} | Qty: {sc.quantity}")

    changes = StockChange.objects.filter(
        product=product,
        timestamp__range=(start, end)
    )

    purchases = []

    for c in changes.filter(change_type__in=["checkin", "error_add"]):
        purchases.append(PurchaseRecord(quantity=c.quantity, purchase_date=c.timestamp.strftime("%Y-%m-%d")))

    for c in changes.filter(change_type__in=["error_subtract", "checkin_delete1"]):
        # Subtract these quantities as negative purchases (stock removed)
        purchases.append(PurchaseRecord(quantity=-abs(c.quantity), purchase_date=c.timestamp.strftime("%Y-%m-%d")))


    sales = [
        SaleRecord(quantity=abs(c.quantity), sale_date=c.timestamp.strftime("%Y-%m-%d"))
        for c in changes.filter(change_type="checkout")
    ]

    expiries = [
        ExpiryRecord(quantity=abs(c.quantity), expiry_date=c.timestamp.strftime("%Y-%m-%d"))
        for c in changes.filter(change_type="expired")
    ]

    return purchases, sales, expiries