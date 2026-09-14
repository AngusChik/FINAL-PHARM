from urllib.parse import urlsplit

from django.urls import Resolver404, resolve, reverse
from django.utils.http import url_has_allowed_host_and_scheme


PRODUCT_RETURN_LABELS = {
    'inventory_display': 'Back to Inventory',
    'product_search': 'Back to Product Search',
    'product_details': 'Back to Product Details',
    'prescription_drug_detail': 'Back to Prescription Drug Details',
    'checkin_dashboard': 'Back to Check-in',
    'checkin_session': 'Back to Check-in',
    'checkin_session_detail': 'Back to Check-in',
    'low_stock': 'Back to Recently Purchased',
    'expired_products': 'Back to Expired Stock',
    'expiring_soon': 'Back to Expiring Soon',
    'out_of_stock': 'Back to Out of Stock',
    'low_stock_trend': 'Back to Low Stock',
    'daily_report': 'Back to Daily Report',
    'history': 'Back to History',
    'history_detail': 'Back to History Record',
}


# Header return links may point only at real, browser-rendered workflow pages.
# This is deliberately an allowlist rather than a collection of URL-pattern
# guesses: adding a new API, export, PDF, or write action must never make that
# endpoint a valid return destination by accident. Active carts/sessions remain
# valid *destinations* even though their own templates do not show a return link.
PAGE_RETURN_DESTINATIONS = {
    'dashboard': 'Dashboard',
    'inventory_display': 'Inventory',
    'new_product': 'Add Product',
    'edit_product': 'Edit Product',
    'product_details': 'Product Details',
    'expired_products': 'Expired Stock',
    'expired_log': 'Expired Log',
    'business_loss': 'Business Loss',
    'expiring_soon': 'Expiring Soon',
    'out_of_stock': 'Out of Stock',
    'low_stock_trend': 'Low Stock Alert',
    'low_stock': 'Recently Purchased',
    'checkin_dashboard': 'Check-in',
    'checkin_session': 'Check-in Session',
    'checkin_session_detail': 'Check-in Session',
    'checkin_reconcile': 'Reconciliation',
    'create_order': 'Purchase',
    'order_view': 'Transactions',
    'order_detail': 'Transaction',
    'order_correction': 'Transaction Correction',
    'sales_analytics': 'Sales Analytics',
    'daily_report': 'Daily Report',
    'checkout': 'Checkout',
    'checkout_cart': 'Checkout Cart',
    'giveaway_detail': 'Transaction',
    'giveaway_correction': 'Transaction Correction',
    'delivery': 'Delivery',
    'item_list': 'Special Orders',
    'label_printing': 'Labels',
    'ordering_sheet': 'Ordering',
    'supplier_purchase_orders': 'Supplier Orders',
    'prescription_drugs': 'Prescription Drugs',
    'add_prescription_drug': 'Add Prescription Drug',
    'prescription_drug_history': 'Prescription Drug History',
    'prescription_drug_detail': 'Prescription Drug Details',
    'edit_prescription_drug': 'Edit Prescription Drug',
    'add_prescription_drug_package': 'Add Pack Size',
    'edit_prescription_drug_package': 'Edit Pack Size',
    'add_prescription_drug_supplier': 'Add Drug Supplier',
    'edit_prescription_drug_supplier': 'Edit Drug Supplier',
    'activity_log': 'History',
    'history': 'History',
    'history_detail': 'History Record',
    'active_sessions': 'Active Sessions',
    'archive_recovery': 'Recovery',
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


def safe_product_details_return_url(request, raw):
    """Preserve the supported list or report that opened a product record."""
    inventory_url = reverse('inventory_display')
    candidate = safe_local_return_url(
        request,
        raw,
        fallback_name='inventory_display',
    )
    allowed_paths = {
        inventory_url.rstrip('/'),
        reverse('expired_products').rstrip('/'),
        reverse('daily_report').rstrip('/'),
    }
    if urlsplit(candidate).path.rstrip('/') not in allowed_paths:
        try:
            destination = resolve(urlsplit(candidate).path)
        except Resolver404:
            return inventory_url
        if destination.url_name not in {'prescription_drug_detail', 'history', 'history_detail'}:
            return inventory_url
    return candidate


def product_return_label(return_url):
    """Describe a known product origin without discarding unknown safe URLs."""
    try:
        route_name = resolve(urlsplit(return_url).path).url_name
    except Resolver404:
        route_name = None
    return PRODUCT_RETURN_LABELS.get(route_name, 'Back to Previous Page')


def page_return_candidate(request, raw):
    """Validate and describe a possible header-return destination.

    A candidate retains the caller's exact safe URL (including its query
    string) while resolving its path against the explicit page allowlist.  The
    current-path flag lets the context processor distinguish an in-page GET or
    validation rerender from a genuinely new navigation without ever emitting
    a self-referencing link.
    """
    if not raw or not url_has_allowed_host_and_scheme(
        raw,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return None

    path = urlsplit(raw).path
    if not path.startswith('/'):
        return None

    try:
        route_name = resolve(path).url_name
    except Resolver404:
        return None

    destination = PAGE_RETURN_DESTINATIONS.get(route_name)
    if destination is None:
        return None

    return {
        'url': raw,
        'destination': destination,
        'label': f'Back to {destination}',
        'is_current': path.rstrip('/') == request.path.rstrip('/'),
    }
