"""Shared Inventory quick filters and counts for the page and CSV export."""

from datetime import date, timedelta

from django.db.models import Count, Exists, F, OuterRef, Q, Value
from django.db.models.functions import Coalesce

from .models import ProductLot


STOCK_FILTER_OPTIONS = (
    ('all', 'All products', 'Every product in the selected departments and search.'),
    (
        'attention', 'Needs attention',
        'Expired or soon-expiring stock, unassigned lots, or active products with low or no stock.',
    ),
    ('unassigned', 'Unassigned', 'Stock held in an active UNASSIGNED lot.'),
    ('expired', 'Expired', 'In-stock products with a lot that expired before today.'),
    (
        'expiring_soon', 'Expiring soon',
        'In-stock products with a lot expiring today or within the next 30 days.',
    ),
    (
        'low_stock', 'Low stock',
        'In-stock products at or below their department threshold; 3 units when no department is set.',
    ),
    ('out_of_stock', 'Out of stock', 'Products with no stock on hand.'),
    ('inactive', 'Inactive', 'Products marked inactive.'),
)


def normalize_stock_filter(value):
    return value if value in {option[0] for option in STOCK_FILTER_OPTIONS} else 'all'


def annotate_inventory_filters(products):
    """Use existing lots without multiplying product rows or inventory totals.

    As on the Expired Stock page, a product-level expiry is a legacy fallback
    only when the product has no active, quantity-bearing dated lot.
    """
    today = date.today()
    active_stock_lots = ProductLot.objects.filter(
        product_id=OuterRef('pk'),
        archived_at__isnull=True,
        quantity_on_hand__gt=0,
    )
    dated_lots = active_stock_lots.filter(expiry_date__isnull=False)
    return products.annotate(
        stock_threshold=Coalesce(F('category__low_stock_threshold'), Value(3)),
        _inventory_unassigned=Exists(
            active_stock_lots.filter(lot_number=ProductLot.UNASSIGNED),
        ),
        _inventory_has_dated_lot=Exists(dated_lots),
        _inventory_expired_lot=Exists(dated_lots.filter(expiry_date__lt=today)),
        _inventory_soon_lot=Exists(dated_lots.filter(
            expiry_date__gte=today,
            expiry_date__lte=today + timedelta(days=30),
        )),
    )


def _stock_filter_conditions():
    today = date.today()
    unassigned = Q(_inventory_unassigned=True)
    expired = Q(quantity_in_stock__gt=0) & (
        Q(_inventory_expired_lot=True)
        | Q(_inventory_has_dated_lot=False, expiry_date__lt=today)
    )
    soon = Q(quantity_in_stock__gt=0) & (
        Q(_inventory_soon_lot=True)
        | Q(
            _inventory_has_dated_lot=False,
            expiry_date__gte=today,
            expiry_date__lte=today + timedelta(days=30),
        )
    )
    low = Q(quantity_in_stock__gt=0, quantity_in_stock__lte=F('stock_threshold'))
    out = Q(quantity_in_stock=0)
    return {
        'all': Q(),
        'attention': unassigned | expired | soon | (Q(status=True) & (low | out)),
        'unassigned': unassigned,
        'expired': expired,
        'expiring_soon': soon,
        'low_stock': low,
        'out_of_stock': out,
        'inactive': Q(status=False),
    }


def apply_stock_filter(products, stock_filter):
    """Filter an annotated queryset; unknown values leave all products visible."""
    return products.filter(_stock_filter_conditions()[normalize_stock_filter(stock_filter)])


def inventory_filter_options(products):
    """Count each option within the current department/search selection."""
    conditions = _stock_filter_conditions()
    counts = products.aggregate(**{
        value: Count('pk', filter=conditions[value])
        for value, _label, _description in STOCK_FILTER_OPTIONS
    })
    return [
        {'value': value, 'label': label, 'description': description, 'count': counts[value]}
        for value, label, description in STOCK_FILTER_OPTIONS
    ]
