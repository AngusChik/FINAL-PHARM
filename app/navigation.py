from urllib.parse import urlsplit

from django.urls import Resolver404, resolve, reverse
from django.utils.http import url_has_allowed_host_and_scheme


PRODUCT_RETURN_LABELS = {
    'inventory_display': 'Back to Inventory',
    'product_search': 'Back to Product Search',
    'product_trend': 'Back to Product Trend',
    'checkin_dashboard': 'Back to Check-in',
    'checkin_session': 'Back to Check-in',
    'checkin_session_detail': 'Back to Check-in',
    'low_stock': 'Back to Recently Purchased',
    'expired_products': 'Back to Expired Stock',
    'expiring_soon': 'Back to Expiring Soon',
    'out_of_stock': 'Back to Out of Stock',
    'low_stock_trend': 'Back to Low Stock',
}


def safe_local_return_url(request, raw, fallback_name='inventory_display'):
    """Keep form return navigation on this site and out of a self-loop."""
    fallback = reverse(fallback_name)
    if not raw or not url_has_allowed_host_and_scheme(
        raw,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return fallback

    path = urlsplit(raw).path
    if not path.startswith('/'):
        return fallback
    if path.rstrip('/') == request.path.rstrip('/'):
        return fallback
    return raw


def product_return_label(return_url):
    """Describe a known product origin without discarding unknown safe URLs."""
    try:
        route_name = resolve(urlsplit(return_url).path).url_name
    except Resolver404:
        route_name = None
    return PRODUCT_RETURN_LABELS.get(route_name, 'Back to Previous Page')
