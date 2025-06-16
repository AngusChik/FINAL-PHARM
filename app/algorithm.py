from math import ceil
import numpy as np
from typing import List, Dict, Optional
from .models import StockForecastResult
from django.utils.timezone import now

def enhanced_stock_prediction(
    stock_bought: List[int],
    stock_sold: List[int], 
    stock_expired: List[int],
    initial_stock: int,
    price_per_unit: float,
    cost_per_unit: float,
    shelf_life_months: int = 3,
    service_level: float = 0.95,
    lead_time: int = 1,
    forecast_horizon: int = 3
) -> StockForecastResult:

    months = len(stock_bought)
    if months < 2:
        raise ValueError("Need at least 2 months of data for analysis")

    inventory_by_age = [0] * shelf_life_months
    monthly_metrics = []

    for i in range(months):
        expired_this_month = inventory_by_age[-1]
        inventory_by_age = [stock_bought[i]] + inventory_by_age[:-1]

        remaining_demand = stock_sold[i]
        units_sold = 0

        for age_bucket in reversed(range(shelf_life_months)):
            if remaining_demand <= 0:
                break
            sold_from_bucket = min(inventory_by_age[age_bucket], remaining_demand)
            inventory_by_age[age_bucket] -= sold_from_bucket
            units_sold += sold_from_bucket
            remaining_demand -= sold_from_bucket

        monthly_metrics.append({
            'demand': stock_sold[i],
            'sold': units_sold,
            'unmet_demand': remaining_demand,
            'expired': expired_this_month,
            'ending_inventory': sum(inventory_by_age),
            'service_level': units_sold / stock_sold[i] if stock_sold[i] > 0 else 1.0
        })

    def weighted_moving_average(data: List[float], weights: Optional[List[float]] = None) -> float:
        if not weights:
            weights = [i + 1 for i in range(len(data))]
        return sum(d * w for d, w in zip(data, weights)) / sum(weights)

    demands = [m['demand'] for m in monthly_metrics]
    recent_demands = demands[-min(6, len(demands)):]
    avg_demand = weighted_moving_average(recent_demands)
    demand_std = np.std(recent_demands) if len(recent_demands) > 1 else avg_demand * 0.2

    if len(demands) >= 3:
        x = np.arange(len(demands))
        z = np.polyfit(x, demands, 1)
        trend_slope = z[0]
        trend_adjusted_demand = max(0, avg_demand + trend_slope * forecast_horizon)
    else:
        trend_slope = 0
        trend_adjusted_demand = avg_demand

    z_scores = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
    z_score = z_scores.get(service_level, 1.645)
    safety_stock = z_score * demand_std * np.sqrt(lead_time)

    annual_demand = avg_demand * 12
    holding_cost_rate = 0.20
    ordering_cost = cost_per_unit * 0.1

    if annual_demand > 0 and holding_cost_rate > 0:
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / (cost_per_unit * holding_cost_rate))
    else:
        eoq = avg_demand * 2

    total_expired = sum(m['expired'] for m in monthly_metrics)
    total_purchased = sum(stock_bought) + initial_stock
    historical_wastage_rate = total_expired / total_purchased if total_purchased > 0 else 0
    shelf_life_factor = min(1.0, shelf_life_months * 30 / avg_demand)
    adjusted_wastage_rate = historical_wastage_rate * (2 - shelf_life_factor)

    current_inventory = sum(inventory_by_age)
    current_usable_inventory = sum(inventory_by_age[:-1])
    target_inventory = (trend_adjusted_demand * (lead_time + 1)) + safety_stock
    wastage_buffer = target_inventory * adjusted_wastage_rate
    total_needed = target_inventory + wastage_buffer
    optimal_order = max(0, total_needed - current_usable_inventory)

    min_order_size = max(avg_demand * 0.5, eoq * 0.3)
    if 0 < optimal_order < min_order_size:
        optimal_order = min_order_size

    avg_service_level = np.mean([m['service_level'] for m in monthly_metrics])
    total_unmet_demand = sum(m['unmet_demand'] for m in monthly_metrics)
    revenue_loss = total_unmet_demand * price_per_unit
    holding_cost = current_inventory * cost_per_unit * (holding_cost_rate / 12)
    wastage_cost = total_expired * cost_per_unit

    recommendations = generate_recommendations(avg_service_level, adjusted_wastage_rate, optimal_order, avg_demand)
    recs = (recommendations + [None] * 4)[:4]

    forecast = StockForecastResult.objects.create(
        optimal_stock_to_buy=ceil(optimal_order),
        forecasted_demand=round(trend_adjusted_demand, 2),
        safety_stock=ceil(safety_stock),
        current_inventory=current_inventory,
        target_inventory=ceil(target_inventory),
        eoq=ceil(eoq),

        historical_service_level=round(avg_service_level, 3),
        wastage_rate=round(adjusted_wastage_rate, 3),
        demand_std_dev=round(demand_std, 2),
        trend_slope=round(trend_slope, 4),
        revenue_loss=round(revenue_loss, 2),
        holding_cost=round(holding_cost, 2),
        wastage_cost=round(wastage_cost, 2),

        recommendation_1=recs[0],
        recommendation_2=recs[1],
        recommendation_3=recs[2],
        recommendation_4=recs[3]
    )

    return forecast

def generate_recommendations(service_level: float, wastage_rate: float, 
                              optimal_order: float, avg_demand: float) -> List[str]:
    recommendations = []
    if service_level < 0.90:
        recommendations.append("Service level is low - consider increasing safety stock")
    if wastage_rate > 0.15:
        recommendations.append("High wastage rate detected - review shelf life and ordering frequency")
    if optimal_order > avg_demand * 3:
        recommendations.append("Large order suggested - consider supply chain reliability")
    if optimal_order == 0:
        recommendations.append("Current inventory sufficient - monitor for demand changes")
    return recommendations
