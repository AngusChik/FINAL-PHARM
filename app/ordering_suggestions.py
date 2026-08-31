"""Read-only ordering suggestions for the Recently Purchased review board.

The supplier automation deliberately does not import this module.  Suggestions
are advisory: this service reads completed sales, missed customer demand,
quantity-bearing lots, and confirmed supplier orders, then returns plain
dictionaries for the UI.  It never creates or updates workflow state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Iterable, Sequence

from django.db.models import Min, Sum
from django.db.models.functions import TruncDate
from django.utils import timezone

from . import reporting
from .models import (
    OrderDetail,
    Product,
    ProductLot,
    RecentlyPurchasedProduct,
    StockChange,
    SupplierPurchaseOrder,
    SupplierPurchaseOrderLine,
)
from .utils import CLOSED_WEEKDAYS, count_open_days, stock_change_delta


HISTORY_WINDOWS = (
    (90, "Recent demand"),
    (180, "Steadier view"),
    (365, "Long-term view"),
)
LEAD_TIME_DAYS = 7
ORDER_CYCLE_DAYS = 30
RECENCY_HALF_LIFE_DAYS = 90
TSB_DEFAULT_ALPHA = 0.1
TSB_DEFAULT_BETA = 0.1
TSB_PARAMETER_GRID = (0.05, 0.1, 0.2)
SERVICE_LEVEL_Z = 1.65
MIN_ROLLING_ORIGINS = 12


@dataclass(frozen=True)
class DemandDay:
    day: date
    fulfilled: int = 0
    unfilled: int = 0
    observable: bool = True

    @property
    def demand(self) -> int:
        return max(0, int(self.fulfilled)) + max(0, int(self.unfilled))


def _open_dates(start: date, end: date) -> list[date]:
    """Return scheduled open dates, inclusive, using the existing Sunday rule."""
    if end < start:
        return []
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
        if (start + timedelta(days=offset)).weekday() not in CLOSED_WEEKDAYS
    ]


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(max(0.0, min(1.0, percentile)) * len(ordered)))
    return ordered[rank - 1]


def classify_demand_pattern(values: Sequence[int | float]) -> dict:
    """Classify demand using ADI/CV-squared boundaries.

    The internal four-way name selects the model.  The UI receives one of the
    deliberately plain labels Regular, Uneven, or Occasional.
    """
    cleaned = [max(0.0, float(value)) for value in values]
    positives = [value for value in cleaned if value > 0]
    if not cleaned or not positives:
        return {
            "internal": "none",
            "label": "No recorded demand",
            "average_interval": None,
            "squared_variation": None,
        }

    average_interval = len(cleaned) / len(positives)
    positive_mean = sum(positives) / len(positives)
    positive_variance = sum(
        (value - positive_mean) ** 2 for value in positives
    ) / len(positives)
    squared_variation = (
        positive_variance / (positive_mean ** 2) if positive_mean else 0.0
    )

    intermittent = average_interval >= 1.32
    variable = squared_variation >= 0.49
    if intermittent and variable:
        internal, label = "lumpy", "Occasional"
    elif intermittent:
        internal, label = "intermittent", "Occasional"
    elif variable:
        internal, label = "erratic", "Uneven"
    else:
        internal, label = "smooth", "Regular"
    return {
        "internal": internal,
        "label": label,
        "average_interval": round(average_interval, 3),
        "squared_variation": round(squared_variation, 3),
    }


def tsb_daily_rate(
    values: Sequence[int | float],
    *,
    alpha: float = TSB_DEFAULT_ALPHA,
    beta: float = TSB_DEFAULT_BETA,
) -> float:
    """Return the TSB occurrence-probability x demand-size forecast."""
    cleaned = [max(0.0, float(value)) for value in values]
    first_positive = next((i for i, value in enumerate(cleaned) if value > 0), None)
    if first_positive is None:
        return 0.0

    demand_size = cleaned[first_positive]
    occurrence_probability = 1.0 / (first_positive + 1)
    for value in cleaned[first_positive + 1:]:
        occurred = 1.0 if value > 0 else 0.0
        occurrence_probability += beta * (occurred - occurrence_probability)
        if occurred:
            demand_size += alpha * (value - demand_size)
    return max(0.0, demand_size * occurrence_probability)


def _select_tsb_parameters(values: Sequence[int | float]) -> tuple[float, float]:
    """Select smoothing values only when there are enough rolling origins."""
    cleaned = [max(0.0, float(value)) for value in values]
    usable_origins = [
        index for index in range(2, len(cleaned))
        if any(value > 0 for value in cleaned[:index])
    ]
    if len(usable_origins) < MIN_ROLLING_ORIGINS:
        return TSB_DEFAULT_ALPHA, TSB_DEFAULT_BETA

    candidates = [(TSB_DEFAULT_ALPHA, TSB_DEFAULT_BETA)] + [
        (alpha, beta)
        for alpha in TSB_PARAMETER_GRID
        for beta in TSB_PARAMETER_GRID
        if (alpha, beta) != (TSB_DEFAULT_ALPHA, TSB_DEFAULT_BETA)
    ]
    best = candidates[0]
    best_error = math.inf
    for alpha, beta in candidates:
        errors = [
            abs(
                cleaned[index]
                - tsb_daily_rate(cleaned[:index], alpha=alpha, beta=beta)
            )
            for index in usable_origins
        ]
        error = sum(errors) / len(errors)
        if error < best_error - 1e-12:
            best = alpha, beta
            best_error = error
    return best


def exponentially_weighted_daily_rate(
    points: Sequence[tuple[date, int | float]],
    *,
    half_life_days: int = RECENCY_HALF_LIFE_DAYS,
) -> float:
    if not points:
        return 0.0
    latest = max(day for day, _ in points)
    weighted_total = 0.0
    total_weight = 0.0
    for day, value in points:
        age = max(0, (latest - day).days)
        weight = 0.5 ** (age / max(1, half_life_days))
        weighted_total += max(0.0, float(value)) * weight
        total_weight += weight
    return weighted_total / total_weight if total_weight else 0.0


def adaptive_daily_forecast(
    points: Sequence[tuple[date, int | float]],
    *,
    optimize_tsb: bool = True,
) -> dict:
    """Choose a recency-weighted or intermittent-demand forecast."""
    ordered = sorted(points, key=lambda point: point[0])
    values = [max(0.0, float(value)) for _, value in ordered]
    pattern = classify_demand_pattern(values)
    if pattern["internal"] in {"intermittent", "lumpy"}:
        alpha, beta = (
            _select_tsb_parameters(values)
            if optimize_tsb else (TSB_DEFAULT_ALPHA, TSB_DEFAULT_BETA)
        )
        rate = tsb_daily_rate(values, alpha=alpha, beta=beta)
        model = "intermittent"
    else:
        alpha = beta = None
        rate = exponentially_weighted_daily_rate(ordered)
        model = "recency_weighted"
    return {
        "daily_rate": max(0.0, rate),
        "model": model,
        "pattern": pattern,
        "alpha": alpha,
        "beta": beta,
    }


def calculate_window_metrics(
    daily_history: Sequence[DemandDay],
    *,
    end_day: date,
    horizon_days: int,
    label: str,
    future_open_days: int,
    recorded_start: date | None = None,
) -> dict:
    """Summarize one historical window without inventing unavailable zeroes."""
    requested_start = end_day - timedelta(days=horizon_days - 1)
    in_window = [
        record for record in daily_history
        if requested_start <= record.day <= end_day
    ]
    observable = [record for record in in_window if record.observable]
    first_recorded_day = recorded_start or min(
        (record.day for record in in_window), default=None,
    )
    if recorded_start is None and first_recorded_day is not None:
        # Closed leading days are still part of trustworthy calendar coverage.
        # The demand series intentionally contains open days only, so infer a
        # requested-start Sunday when its first Monday is present.
        first_scheduled_open = next(iter(_open_dates(requested_start, end_day)), None)
        if first_recorded_day == first_scheduled_open:
            first_recorded_day = requested_start
    if first_recorded_day is not None:
        first_recorded_day = max(requested_start, first_recorded_day)
    history_days = (
        (end_day - first_recorded_day).days + 1
        if first_recorded_day is not None and first_recorded_day <= end_day else 0
    )
    fulfilled = sum(record.fulfilled for record in observable)
    unfilled = sum(record.unfilled for record in observable)
    total = fulfilled + unfilled
    positive_dates = [record.day for record in observable if record.demand > 0]
    request_intervals = [
        (current - previous).days
        for previous, current in zip(positive_dates, positive_dates[1:])
    ]
    average_interval = (
        sum(request_intervals) / len(request_intervals)
        if request_intervals else None
    )
    observable_days = len(observable)
    monthly_units = (
        total / observable_days * future_open_days if observable_days else 0.0
    )
    coverage_label = f"{history_days} of {horizon_days} days available"
    return {
        "days": horizon_days,
        "label": label,
        "fulfilled_units": int(fulfilled),
        "unfilled_units": int(unfilled),
        "total_demand": int(total),
        "history_days": history_days,
        "requested_days": horizon_days,
        "coverage_label": coverage_label,
        "observable_open_days": observable_days,
        "scheduled_open_days": (
            count_open_days(first_recorded_day, end_day)
            if first_recorded_day is not None else 0
        ),
        "active_demand_days": len(positive_dates),
        "active_demand_weeks": len(
            {(day.isocalendar().year, day.isocalendar().week) for day in positive_dates}
        ),
        "average_request_interval_days": (
            round(average_interval, 1) if average_interval is not None else None
        ),
        "monthly_units": round(monthly_units, 1),
        "monthly_units_label": f"approximately {round(monthly_units):.0f} units needed monthly",
    }


def classify_momentum(current: dict, previous: dict, *, future_open_days: int) -> str:
    """Return rising/falling only for material, statistically credible movement."""
    current_days = current.get("observable_open_days", 0)
    previous_days = previous.get("observable_open_days", 0)
    if current_days < 30 or previous_days < 30:
        return "insufficient"

    current_total = max(0, current.get("total_demand", 0))
    previous_total = max(0, previous.get("total_demand", 0))
    current_month = current_total / current_days * future_open_days
    previous_month = previous_total / previous_days * future_open_days
    difference = current_month - previous_month
    relative = abs(difference) / max(previous_month, 1.0)
    standard_error = future_open_days * math.sqrt(
        current_total / (current_days ** 2)
        + previous_total / (previous_days ** 2)
    )
    if (
        abs(difference) < 2
        or relative < 0.25
        or abs(difference) <= 1.96 * standard_error
    ):
        return "stable"
    return "rising" if difference > 0 else "falling"


def calculate_safety_stock(
    points: Sequence[tuple[date, int | float]],
    *,
    expected_lead_time_demand: float,
) -> tuple[int, int]:
    """Use rolling seven-day positive errors, with a conservative fallback.

    Only origins whose whole following seven-day open period is observable are
    compared.  This prevents censored stockout gaps from masquerading as zero
    demand in backtesting.
    """
    ordered = sorted(points, key=lambda point: point[0])
    point_map = {day: max(0.0, float(value)) for day, value in ordered}
    errors: list[float] = []
    if ordered:
        for index in range(2, len(ordered)):
            origin = ordered[index - 1][0]
            future_dates = _open_dates(origin + timedelta(days=1), origin + timedelta(days=7))
            if not future_dates or any(day not in point_map for day in future_dates):
                continue
            train = ordered[:index]
            forecast_rate = adaptive_daily_forecast(
                train, optimize_tsb=False,
            )["daily_rate"]
            forecast = forecast_rate * len(future_dates)
            actual = sum(point_map[day] for day in future_dates)
            errors.append(max(0.0, actual - forecast))

    if len(errors) >= MIN_ROLLING_ORIGINS:
        # A one-sided safety buffer is based on occasions where the forecast
        # was actually too low.  Including the many zero-error origins of an
        # intermittent item can otherwise make one real spike disappear at
        # the 95th percentile.
        positive_errors = [error for error in errors if error > 1e-12]
        empirical = _nearest_rank_percentile(positive_errors, 0.95)
        return math.ceil(empirical), len(errors)
    fallback = SERVICE_LEVEL_Z * math.sqrt(max(0.0, expected_lead_time_demand))
    return math.ceil(fallback), len(errors)


def calculate_dependable_stock(
    lots: Sequence[dict],
    *,
    as_of: date,
    daily_rate: float,
) -> dict:
    """Apply FEFO and retain only dated stock likely to sell before expiry."""
    positive_lots = [lot for lot in lots if int(lot.get("quantity_on_hand") or 0) > 0]
    positive_lots.sort(key=lambda lot: (lot.get("expiry_date") is None, lot.get("expiry_date") or date.max))
    stock_on_hand = sum(int(lot["quantity_on_hand"]) for lot in positive_lots)
    dependable = 0
    assigned_before_expiry = 0
    expired_units = 0

    for lot in positive_lots:
        quantity = int(lot["quantity_on_hand"])
        expiry = lot.get("expiry_date")
        if expiry is None:
            dependable += quantity
            continue
        if expiry < as_of:
            expired_units += quantity
            continue
        cumulative_capacity = math.ceil(
            max(0.0, daily_rate) * count_open_days(as_of, expiry)
        )
        usable = min(quantity, max(0, cumulative_capacity - assigned_before_expiry))
        dependable += usable
        assigned_before_expiry += usable

    return {
        "stock_on_hand": stock_on_hand,
        "dependable_stock": dependable,
        "expiry_units_at_risk": max(0, stock_on_hand - dependable),
        "already_expired_units": expired_units,
    }


def simulate_scheduled_coverage(
    *,
    dependable_stock: int,
    incoming_schedule: Sequence[dict],
    daily_rate: float,
    as_of: date,
    safety_stock: int = 0,
    horizon_days: int = 365,
) -> dict:
    """Simulate when stock is available instead of treating receipts as on hand.

    Reliable supplier quantities enter inventory at the start of their expected
    date.  Unknown and overdue receipts are omitted by the caller.  The
    theoretical balance is allowed to go negative so the maximum units needed
    to avoid a pre-delivery gap remains visible.
    """
    schedule = defaultdict(int)
    for receipt in incoming_schedule:
        expected = receipt.get("expected_date")
        quantity = max(0, int(receipt.get("quantity") or 0))
        if isinstance(expected, date) and expected >= as_of and quantity:
            schedule[expected] += quantity

    rate = max(0.0, float(daily_rate))
    balance = float(max(0, int(dependable_stock or 0)))
    first_gap = None
    covered_open_days = 0
    min_lead_balance = balance
    lead_balance = balance
    two_week_balance = balance
    lead_end = as_of + timedelta(days=LEAD_TIME_DAYS - 1)
    two_week_end = as_of + timedelta(days=13)

    for offset in range(max(1, horizon_days)):
        day = as_of + timedelta(days=offset)
        balance += schedule.get(day, 0)
        available_before_demand = balance
        if day.weekday() not in CLOSED_WEEKDAYS and rate > 0:
            if first_gap is None:
                if available_before_demand + 1e-12 >= rate:
                    covered_open_days += 1
                else:
                    first_gap = day
            balance -= rate
        if day <= lead_end:
            min_lead_balance = min(min_lead_balance, balance)
            lead_balance = balance
        if day <= two_week_end:
            two_week_balance = balance

    lead_gap_units = math.ceil(max(
        0.0,
        -min_lead_balance,
        float(safety_stock) - lead_balance,
    ))
    return {
        "coverage_days": covered_open_days,
        "first_gap_date": first_gap,
        "lead_gap_units": lead_gap_units,
        "lead_balance": round(lead_balance, 3),
        "two_week_balance": round(two_week_balance, 3),
        "next_incoming_date": min(schedule, default=None),
    }


def describe_incoming(incoming: dict) -> str:
    """Return a plain timing note and explain quantities excluded from math."""
    confirmed = int(incoming.get("confirmed") or 0)
    if not confirmed:
        return "No confirmed incoming stock"

    schedule = incoming.get("schedule") or {}
    uncertain = int(incoming.get("uncertain") or 0)
    overdue = int(incoming.get("overdue") or 0)
    pieces = [f"{confirmed} confirmed"]
    if schedule:
        next_date = min(schedule)
        next_quantity = int(schedule[next_date])
        date_label = f"{next_date.strftime('%b')} {next_date.day}"
        pieces.append(f"next {next_quantity} due {date_label}")
    excluded = uncertain + overdue
    if excluded:
        reasons = []
        if uncertain:
            reasons.append(f"{uncertain} with no delivery date")
        if overdue:
            reasons.append(f"{overdue} overdue")
        pieces.append(f"{' and '.join(reasons)} not counted yet")
    return "; ".join(pieces)


def sort_suggestions(suggestions: Sequence[dict]) -> list[dict]:
    """Put actionable cards first while keeping product order deterministic."""
    priority = {
        "order_now": 0,
        "order_soon": 1,
        "needs_attention": 2,
        "wait_for_now": 3,
    }
    return sorted(
        suggestions,
        key=lambda suggestion: (
            priority.get(suggestion.get("classification"), 99),
            str(suggestion.get("name") or "").casefold(),
            int(suggestion.get("product_id") or 0),
            int(suggestion.get("recent_purchase_id") or 0),
        ),
    )


def _history_start_for_product(product: Product, ledger_start: date, end_day: date) -> date:
    created = product.created_at
    if isinstance(created, datetime):
        created = timezone.localtime(created).date() if timezone.is_aware(created) else created.date()
    if not isinstance(created, date):
        created = ledger_start
    return max(ledger_start, created, end_day - timedelta(days=364))


def _as_recent_rows(recent_products: Iterable[RecentlyPurchasedProduct]) -> list:
    if hasattr(recent_products, "select_related"):
        recent_products = recent_products.select_related(
            "product", "product__category",
        )
    rows = list(recent_products)
    return [row for row in rows if getattr(row, "product_id", None)]


def _load_realized_demand(product_ids: Sequence[int], start: date, end: date) -> dict:
    demand = defaultdict(int)
    if end < start:
        return demand
    queryset = reporting.realized_sales_lines(
        OrderDetail.objects.filter(
            product_id__in=product_ids,
            order__submitted=True,
            order__order_date__date__range=(start, end),
        ),
    )
    rows = (
        queryset.annotate(day=TruncDate("order__order_date"))
        .values("product_id", "day")
        .annotate(total=Sum("realized_quantity"))
    )
    for row in rows:
        demand[(row["product_id"], row["day"])] += max(0, int(row["total"] or 0))
    return demand


def _load_unfilled_demand(product_ids: Sequence[int], start: date, end: date) -> dict:
    demand = defaultdict(int)
    if end < start:
        return demand
    rows = (
        StockChange.objects.filter(
            product_id__in=product_ids,
            change_type="checkout_unfulfilled",
            timestamp__date__range=(start, end),
        )
        .annotate(day=TruncDate("timestamp"))
        .values("product_id", "day")
        .annotate(total=Sum("quantity"))
    )
    for row in rows:
        demand[(row["product_id"], row["day"])] += max(0, int(row["total"] or 0))
    return demand


def _load_ledger_movements(product_ids: Sequence[int], start: date, end: date) -> tuple[dict, dict]:
    movements = defaultdict(int)
    positive_inflow = defaultdict(int)
    if end < start:
        return movements, positive_inflow
    rows = (
        StockChange.objects.filter(
            product_id__in=product_ids,
            timestamp__date__range=(start, end),
        )
        .annotate(day=TruncDate("timestamp"))
        .values("product_id", "day", "change_type", "correction_line__disposition")
        .annotate(total=Sum("quantity"))
    )
    for row in rows:
        key = row["product_id"], row["day"]
        delta = stock_change_delta(
            row["change_type"],
            row["total"],
            row["correction_line__disposition"],
        )
        movements[key] += delta
        positive_inflow[key] += max(0, delta)
    return movements, positive_inflow


def _build_daily_history(
    product: Product,
    *,
    start: date,
    end: date,
    as_of: date,
    current_stock: int,
    fulfilled: dict,
    unfilled: dict,
    movements: dict,
    positive_inflow: dict,
) -> list[DemandDay]:
    if end < start:
        return []
    product_id = product.pk
    # Current lots are the authoritative stock position.  Remove today's net
    # movement to obtain yesterday EOD, then walk the append-only ledger back.
    eod_stock = current_stock - movements.get((product_id, as_of), 0)
    result_descending: list[DemandDay] = []
    for day in (end - timedelta(days=offset) for offset in range((end - start).days + 1)):
        net_movement = movements.get((product_id, day), 0)
        start_stock = eod_stock - net_movement
        fulfilled_units = fulfilled.get((product_id, day), 0)
        unfilled_units = unfilled.get((product_id, day), 0)
        is_open = day.weekday() not in CLOSED_WEEKDAYS
        observable = is_open and (
            start_stock > 0
            or eod_stock > 0
            or positive_inflow.get((product_id, day), 0) > 0
            or fulfilled_units > 0
            or unfilled_units > 0
        )
        if is_open:
            result_descending.append(DemandDay(
                day=day,
                fulfilled=fulfilled_units,
                unfilled=unfilled_units,
                observable=observable,
            ))
        eod_stock = start_stock
    return list(reversed(result_descending))


def _window_before(
    daily_history: Sequence[DemandDay],
    *,
    end_day: date,
    days: int,
    future_open_days: int,
) -> dict:
    return calculate_window_metrics(
        daily_history,
        end_day=end_day,
        horizon_days=days,
        label="Comparison period",
        future_open_days=future_open_days,
    )


def _history_agreement(windows: Sequence[dict]) -> tuple[bool, bool]:
    usable = [
        window["monthly_units"] for window in windows
        if window["observable_open_days"] >= 12
        and window["history_days"] >= min(30, window["days"])
    ]
    if len(usable) < 2:
        return False, False
    spread = max(usable) - min(usable)
    average = sum(usable) / len(usable)
    material = (
        spread > 2.0
        or spread / max(average, 1.0) > 0.35
    )
    return not material, material


def build_ordering_suggestions(
    recent_products: Iterable[RecentlyPurchasedProduct],
    *,
    as_of: date | None = None,
) -> dict:
    """Return advisory suggestions for a filtered Recently Purchased iterable."""
    as_of = as_of or timezone.localdate()
    generated_at = timezone.now()
    rows = _as_recent_rows(recent_products)
    if not rows:
        return {
            "suggestions": [],
            "summary": {
                "total": 0,
                "order_now": 0,
                "order_soon": 0,
                "wait_for_now": 0,
                "needs_attention": 0,
            },
            "generated_at": generated_at,
        }

    products_by_id = {row.product_id: row.product for row in rows}
    product_ids = list(products_by_id)
    historical_end = as_of - timedelta(days=1)
    earliest_ledger_timestamp = StockChange.objects.aggregate(
        earliest=Min("timestamp"),
    )["earliest"]
    if earliest_ledger_timestamp:
        ledger_start = (
            timezone.localtime(earliest_ledger_timestamp).date()
            if timezone.is_aware(earliest_ledger_timestamp)
            else earliest_ledger_timestamp.date()
        )
    else:
        ledger_start = historical_end
    load_start = max(ledger_start, historical_end - timedelta(days=364))

    fulfilled = _load_realized_demand(product_ids, load_start, historical_end)
    unfilled = _load_unfilled_demand(product_ids, load_start, historical_end)
    today_unfilled = _load_unfilled_demand(product_ids, as_of, as_of)
    movements, positive_inflow = _load_ledger_movements(
        product_ids, load_start, as_of,
    )

    lots_by_product = defaultdict(list)
    for lot in ProductLot.objects.filter(
        product_id__in=product_ids,
        archived_at__isnull=True,
        quantity_on_hand__gt=0,
    ).values("product_id", "expiry_date", "quantity_on_hand"):
        lots_by_product[lot["product_id"]].append(lot)

    incoming_by_product = defaultdict(lambda: {
        "confirmed": 0,
        "timely": 0,
        "uncertain": 0,
        "overdue": 0,
        "schedule": defaultdict(int),
    })
    incoming_rows = SupplierPurchaseOrderLine.objects.filter(
        product_id__in=product_ids,
        purchase_order__archived_at__isnull=True,
        purchase_order__status__in=(
            SupplierPurchaseOrder.STATUS_SUBMITTED,
            SupplierPurchaseOrder.STATUS_PARTIAL,
        ),
    ).values(
        "product_id",
        "quantity_ordered",
        "quantity_received",
        "purchase_order__expected_date",
    )
    for incoming in incoming_rows:
        remaining = max(
            0,
            int(incoming["quantity_ordered"] or 0)
            - int(incoming["quantity_received"] or 0),
        )
        if not remaining:
            continue
        bucket = incoming_by_product[incoming["product_id"]]
        bucket["confirmed"] += remaining
        expected = incoming["purchase_order__expected_date"]
        if expected is None:
            bucket["uncertain"] += remaining
        elif expected < as_of:
            bucket["overdue"] += remaining
        else:
            bucket["schedule"][expected] += remaining
            if expected <= as_of + timedelta(days=ORDER_CYCLE_DAYS - 1):
                bucket["timely"] += remaining

    future_open_days = count_open_days(
        as_of, as_of + timedelta(days=ORDER_CYCLE_DAYS - 1),
    )
    lead_open_days = count_open_days(
        as_of, as_of + timedelta(days=LEAD_TIME_DAYS - 1),
    )
    suggestions = []
    for row in rows:
        product = products_by_id[row.product_id]
        product_lots = lots_by_product[product.pk]
        lot_stock = sum(int(lot["quantity_on_hand"]) for lot in product_lots)
        history_start = _history_start_for_product(product, load_start, historical_end)
        daily_history = _build_daily_history(
            product,
            start=history_start,
            end=historical_end,
            as_of=as_of,
            current_stock=lot_stock,
            fulfilled=fulfilled,
            unfilled=unfilled,
            movements=movements,
            positive_inflow=positive_inflow,
        )
        windows = [
            calculate_window_metrics(
                daily_history,
                end_day=historical_end,
                horizon_days=days,
                label=label,
                future_open_days=future_open_days,
                recorded_start=history_start,
            )
            for days, label in HISTORY_WINDOWS
        ]
        observed_points = [
            (record.day, record.demand)
            for record in daily_history if record.observable
        ]
        forecast = adaptive_daily_forecast(observed_points)
        daily_rate = forecast["daily_rate"]
        forecast_30 = daily_rate * future_open_days
        expected_lead = daily_rate * lead_open_days
        safety_stock, rolling_origins = calculate_safety_stock(
            observed_points,
            expected_lead_time_demand=expected_lead,
        )
        reorder_point = math.ceil(expected_lead) + safety_stock
        stock = calculate_dependable_stock(
            product_lots,
            as_of=as_of,
            daily_rate=daily_rate,
        )
        incoming = incoming_by_product[product.pk]
        inventory_position = stock["dependable_stock"] + incoming["timely"]
        suggested_quantity = max(
            0,
            math.ceil(forecast_30 + safety_stock - inventory_position),
        )
        incoming_schedule = [
            {"expected_date": expected, "quantity": quantity}
            for expected, quantity in sorted(incoming["schedule"].items())
        ]
        scheduled_coverage = simulate_scheduled_coverage(
            dependable_stock=stock["dependable_stock"],
            incoming_schedule=incoming_schedule,
            daily_rate=daily_rate,
            as_of=as_of,
            safety_stock=safety_stock,
        )
        suggested_quantity = max(
            suggested_quantity,
            scheduled_coverage["lead_gap_units"],
        )
        today_missed = today_unfilled.get((product.pk, as_of), 0)
        live_unavailable_demand = today_missed > 0 and stock["stock_on_hand"] <= 0
        if live_unavailable_demand:
            # Today's request is not inserted into the completed-history model,
            # but it is a real live need.  Never tell a user to "Order 0" when
            # customers have already asked for unavailable units today.
            suggested_quantity = max(suggested_quantity, int(today_missed), 1)

        recent_start = historical_end - timedelta(days=89)
        previous_90 = _window_before(
            daily_history,
            end_day=recent_start - timedelta(days=1),
            days=90,
            future_open_days=future_open_days,
        )
        momentum = classify_momentum(
            windows[0], previous_90, future_open_days=future_open_days,
        )
        previous_185_end = historical_end - timedelta(days=180)
        previous_185 = _window_before(
            daily_history,
            end_day=previous_185_end,
            days=185,
            future_open_days=future_open_days,
        )
        longer_movement = (
            classify_momentum(
                windows[1], previous_185, future_open_days=future_open_days,
            )
            if windows[-1]["history_days"] >= 365
            else "insufficient"
        )
        agrees, material_disagreement = _history_agreement(windows)
        individual_actions = {
            math.ceil(window["monthly_units"] + safety_stock - inventory_position) > 0
            for window in windows
            if window["observable_open_days"] >= 12
        }
        action_conflict = material_disagreement and len(individual_actions) > 1

        long_window = windows[-1]
        observation_fraction = (
            long_window["observable_open_days"] / long_window["scheduled_open_days"]
            if long_window["scheduled_open_days"] else 0.0
        )
        positive_days = long_window["active_demand_days"]
        if (
            long_window["history_days"] >= 300
            and positive_days >= 12
            and observation_fraction >= 0.8
            and agrees
            and rolling_origins >= MIN_ROLLING_ORIGINS
        ):
            confidence = "high"
        elif (
            long_window["history_days"] >= 90
            and positive_days >= 4
            and observation_fraction >= 0.5
            and not action_conflict
        ):
            confidence = "medium"
        else:
            confidence = "low"

        if action_conflict:
            history_signal = "History is mixed"
        elif long_window["history_days"] < 365:
            history_signal = "Limited history"
        elif (
            momentum in {"rising", "falling"}
            or longer_movement in {"rising", "falling"}
        ):
            history_signal = "Recent change"
        elif agrees:
            history_signal = "History agrees"
        else:
            history_signal = "Limited history"

        integrity_mismatch = int(product.quantity_in_stock or 0) != lot_stock
        insufficient_history = (
            long_window["history_days"] < 30
            or long_window["observable_open_days"] < 12
        )
        serious_availability_gap = (
            long_window["scheduled_open_days"] >= 30
            and observation_fraction < 0.5
            and long_window["total_demand"] > 0
        )
        actionable_missing_barcode = suggested_quantity > 0 and not (product.barcode or "").strip()
        expiry_conflict = stock["expiry_units_at_risk"] > 0 and suggested_quantity > 0
        incoming_uncertain = incoming["uncertain"] > 0 or incoming["overdue"] > 0

        review_reason = None
        if not product.status:
            review_reason = "This product is inactive and should be reviewed before ordering."
        elif integrity_mismatch:
            review_reason = "The product stock and lot totals do not match."
        elif insufficient_history and not live_unavailable_demand:
            review_reason = "Not enough reliable demand history is available yet."
        elif serious_availability_gap and not live_unavailable_demand:
            review_reason = "The product was unavailable for too much of the recorded period."
        elif actionable_missing_barcode:
            review_reason = "A barcode is needed before this item can be ordered safely."
        elif expiry_conflict:
            review_reason = "Some stock may expire before it can be used."
        elif incoming_uncertain:
            review_reason = "A confirmed supplier order has an uncertain or overdue delivery date."
        elif action_conflict and not live_unavailable_demand:
            review_reason = "Recent and longer demand history point to different actions."

        if review_reason:
            classification = "needs_attention"
            action_label = "Review first"
            reason = review_reason
        elif live_unavailable_demand:
            classification = "order_now"
            action_label = f"Order {suggested_quantity} now"
            reason = "Customers asked for this today while it was unavailable."
        elif suggested_quantity > 0 and (
            scheduled_coverage["lead_gap_units"] > 0
            or scheduled_coverage["lead_balance"] <= safety_stock
        ):
            classification = "order_now"
            action_label = f"Order {suggested_quantity} now"
            if scheduled_coverage["next_incoming_date"]:
                next_date = scheduled_coverage["next_incoming_date"]
                reason = (
                    "Available stock may run out before the next confirmed "
                    f"delivery on {next_date.strftime('%b')} {next_date.day}."
                )
            else:
                reason = "Available stock may not cover the supplier lead time."
        elif (
            suggested_quantity > 0
            and (
                (
                    scheduled_coverage["first_gap_date"] is not None
                    and scheduled_coverage["first_gap_date"]
                    <= as_of + timedelta(days=13)
                )
                or scheduled_coverage["two_week_balance"] <= reorder_point
            )
        ):
            classification = "order_soon"
            action_label = f"Order {suggested_quantity} soon"
            reason = "Stock is expected to reach its safe reorder level within two weeks."
        else:
            classification = "wait_for_now"
            action_label = "Wait for now"
            if observed_points and not any(value > 0 for _, value in observed_points):
                reason = "No customer demand was recorded during the reliable history available."
            else:
                reason = "Current and incoming stock cover the expected need."

        coverage_days = (
            scheduled_coverage["coverage_days"] if daily_rate > 0 else None
        )
        if coverage_days is None:
            coverage_label = "No recorded demand"
        elif scheduled_coverage["first_gap_date"] is None:
            coverage_label = f"more than {coverage_days} open days"
        else:
            coverage_label = f"about {coverage_days} open days before a gap"
        suggestions.append({
            "recent_purchase_id": row.pk,
            "product_id": product.pk,
            "name": product.name,
            "brand": product.brand,
            "barcode": product.barcode or "",
            "item_number": product.item_number or "",
            "category": product.category.name if product.category_id else "Uncategorized",
            "classification": classification,
            "action_label": action_label,
            "suggested_quantity": suggested_quantity,
            "reason": reason,
            "stock_on_hand": stock["stock_on_hand"],
            "dependable_stock": stock["dependable_stock"],
            "expiry_units_at_risk": stock["expiry_units_at_risk"],
            "confirmed_incoming": incoming["confirmed"],
            "timely_incoming": incoming["timely"],
            "incoming_note": describe_incoming(incoming),
            "next_incoming_date": scheduled_coverage["next_incoming_date"],
            "coverage_days": coverage_days,
            "coverage_label": coverage_label,
            "confidence": confidence,
            "confidence_label": confidence.title(),
            "history_signal": history_signal,
            "demand_pattern": forecast["pattern"]["label"],
            "forecast_30_days": round(forecast_30, 1),
            "safety_stock": safety_stock,
            "today_missed_demand": int(today_missed),
            "momentum": momentum,
            "longer_movement": longer_movement,
            "windows": windows,
        })

    suggestions = sort_suggestions(suggestions)
    summary = {
        "total": len(suggestions),
        "order_now": 0,
        "order_soon": 0,
        "wait_for_now": 0,
        "needs_attention": 0,
    }
    for suggestion in suggestions:
        summary[suggestion["classification"]] += 1
    return {
        "suggestions": suggestions,
        "summary": summary,
        "generated_at": generated_at,
    }
