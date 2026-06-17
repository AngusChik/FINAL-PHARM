"""Per-page presence lock: only one computer (browser session) may hold a
guarded work page at a time. Shared by the middleware, the heartbeat endpoints
and the template context processor."""

from django.utils.timezone import now

# Seconds a holder stays valid without a heartbeat. The client pings every ~10s,
# so 25s tolerates a missed ping before the lock is considered free.
PAGE_LOCK_TTL = 25

# Key work pages limited to one computer at a time (matched by URL name).
# Maps the URL name to a short human label used in the nav presence bubble.
GUARDED_PAGE_LABELS = {
    # NOTE: The following pages are intentionally NOT guarded:
    #  - Checkout (checkout / checkout_cart): handled by CheckoutView's own
    #    per-draft ownership guard; the PU checkout dashboard lets multiple
    #    computers manage separate sessions concurrently.
    #  - Transactions (order_view / order_detail): viewing and editing orders
    #    is allowed from multiple computers at once.
    #  - Stock (inventory_display) and Low stock (low_stock) reports: read
    #    reports that don't need single-computer access.
    #  - Check-in DASHBOARD (checkin_dashboard): multiple computers may view it;
    #    only the individual check-in SESSION pages below are one-computer.
    # Purchase (POS)
    'create_order': 'Purchase',
    # Check-in — individual sessions only (URL-keyed by id, so locked per session)
    'checkin_session': 'Check-in',
    'checkin_session_detail': 'Check-in',
}
GUARDED_PAGE_NAMES = set(GUARDED_PAGE_LABELS)

# Seconds a computer's "current screen" stays shown in the nav presence bubble
# without a fresh heartbeat. The client beats every ~10s.
PRESENCE_TTL = 30

# Friendly labels for the live "who's on which screen" nav bubble. Unlike the
# guard list above, this covers ALL the main pages — presence is just awareness,
# not a lock. Unmapped pages fall back to a prettified URL name.
PAGE_LABELS = {
    'dashboard': 'Dashboard',
    'create_order': 'Purchase',
    'checkout': 'Checkout',
    'checkout_cart': 'Checkout cart',
    'order_view': 'Transactions',
    'order_detail': 'Transaction',
    'inventory_display': 'Stock',
    'low_stock': 'Low stock',
    'checkin_dashboard': 'Check-in',
    'checkin_session': 'Check-in session',
    'checkin_session_detail': 'Check-in detail',
    'expired_products': 'Expired',
    'label_printing': 'Labels',
    'delivery': 'Delivery',
}


def path_label(path):
    """Friendly label for any URL path, for the nav presence bubble."""
    if not path:
        return '—'
    try:
        from django.urls import resolve
        url_name = resolve(path).url_name
    except Exception:
        return path
    if not url_name:
        return path
    return PAGE_LABELS.get(url_name) or url_name.replace('_', ' ').title()


def client_ip(request):
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', '') or ''


def simplify_ua(ua):
    ua = ua or ''
    browser = 'Unknown browser'
    for token, label in (('Edg', 'Edge'), ('OPR', 'Opera'), ('Chrome', 'Chrome'),
                         ('Firefox', 'Firefox'), ('Safari', 'Safari')):
        if token in ua:
            browser = label
            break
    os_name = ''
    for token, label in (('Windows', 'Windows'), ('Mac OS', 'macOS'), ('Android', 'Android'),
                         ('iPhone', 'iPhone'), ('iPad', 'iPad'), ('Linux', 'Linux')):
        if token in ua:
            os_name = label
            break
    return f"{browser}{(' on ' + os_name) if os_name else ''}"


def is_fresh(presence):
    return (now() - presence.last_seen).total_seconds() <= PAGE_LOCK_TTL


def holder_info(presence):
    return {
        'ip': presence.ip_address or '—',
        'browser': simplify_ua(presence.user_agent),
        'user': presence.user.get_username() if presence.user else '',
    }


def page_label(path):
    """Resolve a stored page path back to a short human label for the nav bubble."""
    try:
        from django.urls import resolve
        url_name = resolve(path).url_name
        return GUARDED_PAGE_LABELS.get(url_name, url_name or path)
    except Exception:
        return path


def presence_defaults(request):
    return {
        'session_key': request.session.session_key,
        'user': request.user if request.user.is_authenticated else None,
        'ip_address': client_ip(request),
        'user_agent': request.META.get('HTTP_USER_AGENT', '')[:300],
    }
