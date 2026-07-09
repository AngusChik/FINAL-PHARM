"""Order-list building for the McKesson PharmaClik re-order flow.

Shared between the web app (preview endpoint in views.py) and the
standalone automation script (mckesson_order.py) so both always agree on
what would be ordered.
"""

from datetime import timedelta

from django.db.models import Sum
from django.db.models.functions import TruncWeek
from django.utils.timezone import now

from app.models import RecentlyPurchasedProduct, StockChange
from app.reporting import SALE_TYPES
from app.utils import get_reorder_prediction


def predicted_quantities(products):
    """suggested_qty per product_id, using the app's reorder-prediction formula.

    Mirrors reporting.reorder_suggestions(): 60 days of sales from
    StockChange, bucketed weekly for the trend/std-dev inputs to
    get_reorder_prediction().
    """
    pids = [p.product_id for p in products]
    since = now().date() - timedelta(days=60)
    demand_map = {
        r["product_id"]: r["total"]
        for r in StockChange.objects.filter(
            product_id__in=pids, timestamp__date__gte=since, change_type__in=SALE_TYPES,
        ).values("product_id").annotate(total=Sum("quantity"))
    }
    weekly_map = {}
    for r in StockChange.objects.filter(
        product_id__in=pids, timestamp__date__gte=since, change_type__in=SALE_TYPES,
    ).annotate(week=TruncWeek("timestamp")).values("product_id", "week").annotate(
        total=Sum("quantity")
    ).order_by("product_id", "week"):
        weekly_map.setdefault(r["product_id"], []).append((r["week"], r["total"]))

    return {
        p.product_id: get_reorder_prediction(
            p, demand_map.get(p.product_id, 0),
            weekly_demands=weekly_map.get(p.product_id, []),
        ).get("suggested_qty", 0)
        for p in products
    }


def collect_order_items(days=None, limit=None, qty_mode="predicted",
                        exclude_category_ids=None):
    """Build [{barcode, name, quantity, product_id}, ...] plus pre-skipped rows.

    qty_mode "predicted": app's reorder-prediction suggested_qty (falls back
    to units sold when the prediction is 0). "sold": units sold since the
    Recently Purchased list was last cleared.
    """
    exclude_category_ids = set(exclude_category_ids or [])
    qs = RecentlyPurchasedProduct.objects.select_related("product")
    if days:
        qs = qs.filter(order_date__gte=now() - timedelta(days=days))
    qs = qs.order_by("-order_date")

    rows = list(qs)
    predictions = {}
    if qty_mode == "predicted":
        predictions = predicted_quantities([rp.product for rp in rows])

    items, skipped = [], []
    seen = {}
    for rp in rows:
        p = rp.product
        if p.category_id and p.category_id in exclude_category_ids:
            skipped.append({"name": p.name, "barcode": p.barcode or "",
                            "quantity": rp.quantity, "reason": "excluded category",
                            "product_id": p.product_id})
            continue
        barcode = (p.barcode or "").strip()
        if not barcode:
            skipped.append({"name": p.name, "barcode": "", "quantity": rp.quantity,
                            "reason": "no barcode on product",
                            "product_id": p.product_id})
            continue
        if barcode in seen:
            # predicted qty is per-product, not per-row — only sum in sold mode
            if qty_mode == "sold":
                seen[barcode]["quantity"] += rp.quantity
            continue
        qty = rp.quantity
        if qty_mode == "predicted":
            qty = predictions.get(p.product_id, 0) or rp.quantity
        entry = {"barcode": barcode, "name": p.name, "quantity": qty,
                 "product_id": p.product_id}
        seen[barcode] = entry
        items.append(entry)

    if limit:
        items = items[:limit]
    return items, skipped
