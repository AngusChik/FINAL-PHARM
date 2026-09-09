"""Add a reorder candidate to the shared ordering list without creating a sale."""

import json

from django.db import transaction
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect

from .mixins import has_admin_access
from .models import Product, RecentlyPurchasedProduct, UserAction


@method_decorator(never_cache, name='dispatch')
@method_decorator(csrf_protect, name='dispatch')
class DashboardReorderAddAPIView(View):
    http_method_names = ['post', 'options']

    def post(self, request):
        if not request.user.is_authenticated:
            return JsonResponse({'ok': False, 'error': 'Please sign in again.'}, status=401)
        if not has_admin_access(request):
            return JsonResponse({
                'ok': False,
                'error': 'Unlock admin access to add products to Recently Purchased.',
            }, status=403)
        try:
            payload = json.loads(request.body)
        except (ValueError, UnicodeDecodeError):
            return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)
        if not isinstance(payload, dict):
            return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)
        product_id = payload.get('product_id')
        quantity = payload.get('quantity')
        if type(product_id) is not int or not 1 <= product_id <= 2147483647:
            return JsonResponse({'ok': False, 'error': 'Choose a valid product.'}, status=400)
        if type(quantity) is not int or not 1 <= quantity <= 9999:
            return JsonResponse({
                'ok': False, 'error': 'Enter a whole quantity from 1 to 9999.',
            }, status=400)

        with transaction.atomic():
            # Match checkout's product lock, including the first insertion when
            # there is no list row to lock. Concurrent clicks become a no-op.
            product = Product.objects.select_for_update().filter(
                pk=product_id, status=True,
            ).first()
            if product is None:
                return JsonResponse({
                    'ok': False, 'error': 'This product is no longer available.',
                }, status=404)
            recent = RecentlyPurchasedProduct.objects.filter(
                product=product, archived_at__isnull=True,
            ).first()
            already_added = recent is not None
            if recent is None:
                recent = RecentlyPurchasedProduct.objects.create(
                    product=product, quantity=0, manual_order_quantity=quantity,
                )
                UserAction.objects.create(
                    user=request.user,
                    action='add_recently_purchased',
                    target=f'Product #{product.pk}',
                    detail=f'Added to ordering list: {product.name}. Manual order quantity: {quantity}.',
                )

        return JsonResponse({
            'ok': True,
            'product_id': product.pk,
            'already_added': already_added,
            'recent_id': recent.pk,
            'quantity': recent.manual_order_quantity or recent.quantity,
            'message': (
                'Already in Recently Purchased.' if already_added
                else 'Added to Recently Purchased for ordering.'
            ),
        })
