from datetime import date
from app.models import Product, Order, RecentlyPurchasedProduct, DeliveryCheckIn
from app.page_lock import GUARDED_PAGE_NAMES


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
        "nav_delivery_count": DeliveryCheckIn.objects.filter(checked_out_at__isnull=True).count(),
    }
