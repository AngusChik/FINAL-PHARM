"""Per-page presence lock: only one computer (browser session) may hold a
guarded work page at a time. Shared by the middleware, the heartbeat endpoints
and the template context processor."""

from datetime import timedelta

from django.utils.timezone import now

# Seconds a holder stays valid without a heartbeat. The client pings every ~10s,
# so 25s tolerates a missed ping before the lock is considered free.
PAGE_LOCK_TTL = 25
CHECKIN_REVIEW_AFTER = timedelta(hours=24)

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

# Mutating endpoints that operate on a guarded work page. The lock must protect
# the write itself, not just the GET that rendered the form; otherwise a stale
# tab can continue changing data after another terminal takes over.
PURCHASE_MUTATION_NAMES = {
    'create_order', 'submit_order', 'delete_order_item', 'add_product_by_id',
}
CHECKIN_SESSION_MUTATION_NAMES = {
    'checkin_session', 'checkin_end', 'checkin_reconcile',
    'checkin_session_adjust', 'checkin_session_remove_line', 'add_quantity',
    'delete_one', 'set_quantity', 'checkin_edit_product', 'checkin_add_by_id',
    'checkin_reassign_lot',
}


def guarded_page_path(request):
    """Return the canonical guarded page affected by this request, if any."""
    match = request.resolver_match
    if not match or not match.url_name:
        return None
    if match.url_name in GUARDED_PAGE_NAMES:
        return request.path

    from django.urls import reverse

    if match.url_name in PURCHASE_MUTATION_NAMES:
        return reverse('create_order')
    if match.url_name in CHECKIN_SESSION_MUTATION_NAMES:
        session_id = match.kwargs.get('session_id')
        if session_id is not None:
            return reverse('checkin_session', kwargs={'session_id': session_id})
    return None


def checkin_session_last_activity(session):
    """Latest durable activity used to distinguish active from abandoned work."""
    latest_change = getattr(session, 'last_stock_change', None)
    if latest_change is None:
        latest_change = session.stock_changes.order_by('-timestamp').values_list(
            'timestamp', flat=True,
        ).first()
    candidates = [session.started_at, session.reopened_at, latest_change]
    return max(value for value in candidates if value is not None)


def checkin_session_needs_review(session):
    return bool(
        session.ended_at is None
        and now() - checkin_session_last_activity(session) > CHECKIN_REVIEW_AFTER
    )

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
    'ordering_sheet': 'Ordering',
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
        'user': session_identity(presence.session_key, presence.user),
    }


def session_identity(session_key, fallback_user=None):
    """Resolve a tracked browser session to its PU1..PU6/admin identity."""
    from app.models import UserSession

    tracked = (
        UserSession.objects.select_related('user')
        .filter(session_key=session_key)
        .first()
    )
    if tracked:
        return tracked.identity_label
    return fallback_user.get_username() if fallback_user else ''


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
