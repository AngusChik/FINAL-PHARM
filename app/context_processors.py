from datetime import date
from app.models import Product, Order, RecentlyPurchasedProduct, DeliveryCheckIn


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
