from datetime import date
from django.conf import settings
from django.urls import NoReverseMatch, reverse

from app.mixins import PASSKEY_SESSION_KEY, has_admin_access, passkey_unlocked
from app.navigation import product_return_label, safe_local_return_url
from app.models import (
    DeliveryCheckIn,
    Order,
    Product,
    RecentlyPurchasedProduct,
    UserTablePreference,
)
from app.page_lock import GUARDED_PAGE_NAMES


WORKFLOW_GUIDES = {
    'products': {
        'title': 'Product records',
        'summary': 'Find products, review stock and lots, or maintain the product record.',
        'steps': [
            'Search by product name, SKU, or barcode.',
            'Use Check In for stock changes; use Edit for product details and lot allocation.',
            'Move obsolete products to Recovery instead of erasing their history.',
        ],
        'tip': 'Adding, fully editing, or removing a product requires staff access or the admin passkey.',
    },
    'checkin': {
        'title': 'Check-in workflow',
        'summary': 'Record stock only when it physically arrives and keep lot information attached.',
        'steps': [
            'Open or start a check-in session.',
            'Scan the barcode, confirm quantity, then enter the lot number and expiry when available.',
            'Review the session totals and end the session when receiving is complete.',
        ],
        'tip': 'Inline product editing is available to every signed-in user. Lot totals must continue to match stock.',
    },
    'stock_exceptions': {
        'title': 'Stock exceptions',
        'summary': 'Work through expired, expiring, out-of-stock, low-stock, and recently purchased items.',
        'steps': [
            'Choose the exception list that matches the task.',
            'Search or filter to narrow the worklist.',
            'Use the page action to record the real inventory outcome.',
        ],
        'tip': 'Expired-stock pages are available to every signed-in user; protected bulk actions show a lock.',
    },
    'sales': {
        'title': 'Purchase and transaction workflow',
        'summary': 'Build the current sale, review completed transactions, and record corrections without rewriting history.',
        'steps': [
            'Scan or search products into the current order.',
            'Review quantities, discounts, tax, and fulfillment before submitting.',
            'Use Transactions for details, reports, receipts, returns, voids, and corrections.',
        ],
        'tip': 'Returns and voids preserve the original sale. Recording a correction requires staff access or the admin passkey.',
    },
    'checkout': {
        'title': 'PU checkout workflow',
        'summary': 'Use a terminal-specific draft to record no-sale stock removals safely.',
        'steps': [
            'Start a new checkout or resume the draft assigned to this computer.',
            'Scan products and verify the supplied quantity.',
            'Submit only after the basket matches what physically left the pharmacy.',
        ],
        'tip': 'Drafts are stored in the database and can be resumed. Unfulfilled units do not reduce inventory.',
    },
    'ordering': {
        'title': 'Ordering workflow',
        'summary': 'Capture requests, then progress them through ordering, receiving, readiness, and pickup.',
        'steps': [
            'Add the drug or OTC request with urgency and initials.',
            'Staff records supplier, quantity ordered, expected date, and receiving progress.',
            'Move the request to Ready, Contacted, and Picked Up as work is completed.',
        ],
        'tip': 'Regular users can edit their own pending requests. Shared lifecycle progress requires staff access or the admin passkey.',
    },
    'supplier_orders': {
        'title': 'Supplier order tracking',
        'summary': 'Track confirmed supplier orders separately from physical inventory receiving.',
        'steps': [
            'Create a tracker from a saved plan or record the supplier confirmation directly.',
            'Update expected dates, notes, and received quantities as the supplier order progresses.',
            'Use Check-in when stock physically arrives; tracking alone never increases inventory.',
        ],
        'tip': 'Supplier-order tracking is protected by staff access or the admin passkey.',
    },
    'delivery': {
        'title': 'Delivery workflow',
        'summary': 'See who is currently on site first, then switch to checked-out history or both tables.',
        'steps': [
            'Check a person in and confirm their identifying details.',
            'Use the view toggle to focus on current, completed, or all records.',
            'Check out the record when the visit is complete.',
        ],
        'tip': 'Normal delivery actions are available to signed-in users. Destructive history controls require elevated access.',
    },
    'labels': {
        'title': 'Label workflow',
        'summary': 'Build a print queue from products, categories, scans, or custom labels.',
        'steps': [
            'Search, scan, or select the labels to add.',
            'Review copies and per-label overrides in the queue.',
            'Generate the PDF, then use Print History when a batch must be reproduced.',
        ],
        'tip': 'Custom labels and print history are stored in the database for the signed-in user.',
    },
    'management': {
        'title': 'Management tools',
        'summary': 'Review activity, active sessions, and records available for recovery.',
        'steps': [
            'Use Activity Log to investigate who changed stock or workflow data.',
            'Use Active Sessions to review computers currently connected.',
            'Use Recovery to search archived records and restore the correct item.',
        ],
        'tip': 'Management pages require staff access or an active admin-passkey unlock.',
    },
}


WORKFLOW_PAGE_GROUPS = {
    'products': {'inventory_display', 'new_product', 'edit_product', 'product_search', 'product_trend'},
    'checkin': {
        'checkin_dashboard', 'checkin_start', 'checkin_session', 'checkin_session_detail',
        'checkin_reconcile', 'checkin_edit_product', 'checkin_session_adjust',
        'checkin_session_remove_line',
    },
    'stock_exceptions': {
        'expired_products', 'expiring_soon', 'out_of_stock', 'low_stock_trend', 'low_stock',
    },
    'sales': {
        'create_order', 'submit_order', 'order_success', 'order_view', 'order_detail',
        'order_correction', 'sales_analytics', 'daily_report',
    },
    'checkout': {
        'checkout', 'checkout_cart', 'checkout_continue', 'checkout_success',
        'giveaway_detail', 'giveaway_correction',
    },
    'ordering': {'ordering_sheet'},
    'supplier_orders': {'supplier_purchase_orders'},
    'delivery': {'delivery', 'item_list'},
    'labels': {
        'label_printing', 'label_sessions', 'label_session_detail',
        'label_session_regenerate',
    },
    'management': {'activity_log', 'active_sessions', 'archive_recovery'},
}


WORKFLOW_PARENT_ROUTES = {
    # Product record sub-pages return to the main inventory page.
    'product_search': ('inventory_display', 'Back to Inventory'),
    'product_trend': ('inventory_display', 'Back to Inventory'),
    # Check-in detail pages return to the session list.
    'checkin_session': ('checkin_dashboard', 'Back to Check-in'),
    'checkin_session_detail': ('checkin_dashboard', 'Back to Check-in'),
    # Stock-exception worklists are reached from Inventory.
    'expired_products': ('inventory_display', 'Back to Inventory'),
    'expiring_soon': ('inventory_display', 'Back to Inventory'),
    'out_of_stock': ('inventory_display', 'Back to Inventory'),
    'low_stock_trend': ('inventory_display', 'Back to Inventory'),
    'low_stock': ('inventory_display', 'Back to Inventory'),
    # Purchasing detail/report pages return to their major workflow page.
    'order_view': ('create_order', 'Back to Purchase'),
    'order_detail': ('order_view', 'Back to Transactions'),
    'order_success': ('create_order', 'Back to Purchase'),
    'sales_analytics': ('create_order', 'Back to Purchase'),
    'daily_report': ('create_order', 'Back to Purchase'),
    # Checkout and fulfillment detail pages.
    'checkout_cart': ('checkout', 'Back to Checkout'),
    'checkout_success': ('checkout', 'Back to Checkout'),
    'giveaway_detail': ('order_view', 'Back to Transactions'),
    'supplier_purchase_orders': ('ordering_sheet', 'Back to Ordering'),
}


def _product_form_parent(request):
    """Preserve a safe product-form origin while keeping the label meaningful."""
    raw = request.GET.get('next')
    if request.method == 'POST':
        raw = request.POST.get('next') or raw
    return_url = safe_local_return_url(request, raw)
    return {
        'url': return_url,
        'label': product_return_label(return_url),
    }


def _workflow_parent(request, page_key, resolver):
    """Return the major parent page shown beside Dashboard in workflow headers."""
    if page_key in {'new_product', 'edit_product'}:
        return _product_form_parent(request)

    kwargs = resolver.kwargs if resolver else {}
    dynamic_routes = {
        'checkin_reconcile': (
            'checkin_session', 'session_id', 'Back to Session',
        ),
        'checkin_edit_product': (
            'checkin_session', 'session_id', 'Back to Session',
        ),
        'order_correction': (
            'order_detail', 'order_id', 'Back to Transaction',
        ),
        'giveaway_correction': (
            'giveaway_detail', 'checkout_id', 'Back to Transaction',
        ),
    }
    dynamic = dynamic_routes.get(page_key)
    if dynamic:
        route_name, kwarg_name, label = dynamic
        object_id = kwargs.get(kwarg_name)
        if object_id is not None:
            try:
                return {
                    'url': reverse(route_name, kwargs={kwarg_name: object_id}),
                    'label': label,
                }
            except NoReverseMatch:
                return None

    route = WORKFLOW_PARENT_ROUTES.get(page_key)
    if not route:
        return None
    route_name, label = route
    return {'url': reverse(route_name), 'label': label}


def page_lock(request):
    """Expose the current page's lock key so base.html can run the heartbeat
    only on guarded pages."""
    rm = getattr(request, 'resolver_match', None)
    key = ''
    if (request.method == 'GET' and rm and rm.url_name in GUARDED_PAGE_NAMES
            and request.user.is_authenticated):
        key = request.path
    return {'page_lock_key': key}


def nav_badges(request):
    if not request.user.is_authenticated:
        return {}

    today = date.today()
    return {
        "nav_expired_count": Product.objects.filter(expiry_date__lt=today).exclude(expiry_date__isnull=True).count(),
        "nav_recent_count": RecentlyPurchasedProduct.objects.count(),
        "nav_transaction_count": Order.objects.filter(submitted=True).count(),
        "nav_delivery_count": DeliveryCheckIn.objects.filter(
            checked_out_at__isnull=True, archived_at__isnull=True,
        ).count(),
    }


def ui_context(request):
    """Global usability context: access cues, page help, and table preferences."""
    resolver = getattr(request, 'resolver_match', None)
    page_key = resolver.url_name if resolver and resolver.url_name else 'unknown'
    workflow_help = None
    for group, page_names in WORKFLOW_PAGE_GROUPS.items():
        if page_key in page_names or (
            group == 'labels' and page_key.startswith('label_session')
        ):
            workflow_help = WORKFLOW_GUIDES[group]
            break
    workflow_parent = _workflow_parent(request, page_key, resolver)

    if not request.user.is_authenticated:
        return {
            'can_administer': False,
            'ui_access': {'role_label': 'Signed out', 'source': 'none'},
            'workflow_help': workflow_help,
            'workflow_parent': workflow_parent,
            'ui_table_preferences': {},
        }

    elevated = passkey_unlocked(request) and not request.user.is_staff
    expires_at = None
    if elevated:
        unlocked_at = request.session.get(PASSKEY_SESSION_KEY)
        ttl = getattr(settings, 'ADMIN_PASSKEY_TTL', 0)
        if unlocked_at and ttl:
            expires_at = int(float(unlocked_at) + ttl)

    preferences = {
        preference.table_key: {
            'density': preference.density,
            'page_size': preference.page_size,
            'hidden_columns': preference.hidden_columns,
        }
        for preference in UserTablePreference.objects.filter(
            user=request.user,
            page_key=page_key,
        )
    }
    can_admin = has_admin_access(request)
    return {
        'can_administer': can_admin,
        'ui_access': {
            'role_label': 'Staff admin' if request.user.is_staff else 'PU user',
            'source': 'staff' if request.user.is_staff else ('passkey' if elevated else 'user'),
            'can_administer': can_admin,
            'passkey_expires_at': expires_at,
        },
        'workflow_help': workflow_help,
        'workflow_parent': workflow_parent,
        'ui_table_preferences': preferences,
    }
