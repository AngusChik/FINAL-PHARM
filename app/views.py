from decimal import Decimal
import os
import csv
from collections import defaultdict
from itertools import product
from urllib import request
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.views import View
from django.contrib import messages  # ✅ CORRECT IMPORT
from django.db.models import Sum
from django.db import transaction
from urllib.parse import urlencode
from django.conf import settings
from django.core.paginator import Paginator
from functools import lru_cache
from pathlib import Path
from django.core.cache import cache
from app.mixins import AdminRequiredMixin
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView
from django.contrib.auth.forms import UserCreationForm
from datetime import date, datetime, timedelta
from django.utils.timezone import now
from django.db import IntegrityError
from .utils import recalculate_order_totals, get_product_stock_records, recommend_inventory_action
from .forms import EditProductForm, OrderDetailForm, BarcodeForm, ItemForm, AddProductForm
from .models import Item, Product, Category, Order, OrderDetail, RecentlyPurchasedProduct, StockChange
from dateutil.relativedelta import relativedelta
from django.db.models import Sum, Q
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth, TruncDate
from decimal import Decimal


# ✅ Add message level configuration
MESSAGE_TAGS = {
    messages.SUCCESS: 'success',
    messages.ERROR: 'danger',
    messages.WARNING: 'warning',
    messages.INFO: 'info',
}

# Path to Master.csv in project root
BASE_DIR = Path(settings.BASE_DIR)
MASTER_CSV_PATH = (BASE_DIR / "master.csv")  # or "Master.csv" if that's the exact name


def _normalize_barcode(value: str) -> str:
    """Keep only digits and strip leading zeros for comparison."""
    if value is None:
        return ""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits.lstrip("0")

def find_product_by_barcode(barcode: str, for_update: bool = False):
    """
    Look up Product by barcode, tolerant of leading zeros.
    1) Try exact match (fast, index-friendly)
    2) If not found, strip leading zeros and match '^0*<digits>$'
    """
    raw = (barcode or "").strip()
    if not raw:
        return None

    normalized = _normalize_barcode(raw)
    qs = Product.objects.all()
    if for_update:
        qs = qs.select_for_update()

    # Exact match first
    product = qs.filter(barcode__iexact=raw).first()
    if product or not normalized:
        return product

    # Leading-zeros-tolerant match (e.g. '0523...' vs '523...')
    return qs.filter(barcode__regex=rf"^0*{normalized}$").first()


@lru_cache(maxsize=1)
def _load_master_catalog():
    """
    Load Master.csv once per process and cache rows as a list of dicts
    with stripped keys/values.
    """
    if not MASTER_CSV_PATH.exists():
        print(f"[MASTER CSV] File not found at {MASTER_CSV_PATH}")
        return []

    rows = []
    with MASTER_CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = { (k or "").strip(): (v or "").strip() for k, v in raw_row.items() }
            rows.append(row)

    print(f"[MASTER CSV] Loaded {len(rows)} rows from {MASTER_CSV_PATH}")
    return rows


def get_master_catalog_entry(barcode: str):
    """
    Find a row in Master.csv whose 'GTIN/UPC (unit)' matches the scanned barcode.
    Comparison is done on digits only, ignoring leading zeros.
    """
    if not barcode:
        return None

    target = _normalize_barcode(barcode)
    if not target:
        return None

    for row in _load_master_catalog():
        candidate = row.get("GTIN/UPC (unit)") or row.get("GTIN/UPC") or row.get("UPC")
        if _normalize_barcode(candidate) == target:
            return row

    return None

def _clean_price(value: str) -> str:
    """Turn things like '$6.4399 ' into '6.44' for the form."""
    if not value:
        return ""
    text = str(value).replace("$", "").strip()
    try:
        return f"{Decimal(text):.2f}"
    except Exception:
        return text  # fall back to raw string if parsing fails


# Utility generator for inclusive date ranges
def _daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

class ProductTrendView(AdminRequiredMixin, View):
    template_name = "product_trend.html"

    def get(self, request):
        query = request.GET.get("q", "").strip()
        chart_type = request.GET.get("type", "bar")
        granularity = request.GET.get("granularity", "month")

        # --- Date handling (fallback to last 12 months if empty) ---
        try:
            end_date = datetime.strptime(request.GET.get("end", ""), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            end_date = date.today()

        try:
            start_date = datetime.strptime(request.GET.get("start", ""), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            start_date = end_date - timedelta(days=365)

        # All products for client-side autocomplete
        all_products = list(
            Product.objects.values(
                "product_id", "name", "barcode", "item_number", "price", "quantity_in_stock"
            )
        )

        context = {
            "query": query,
            "chart_type": chart_type,
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity,
            "all_products": all_products,
            "search_results": None,
        }

        product = None

        if query:
            # Prefer barcode exact match first
            product = find_product_by_barcode(query)

            # If no exact barcode match, show fuzzy search results (by name / barcode)
            search_results = Product.objects.filter(
                name__icontains=query
            ) | Product.objects.filter(barcode__icontains=query)

            context["search_results"] = search_results.distinct()

            if product:
                (
                    sold,
                    restocked,
                    labels,
                    cumulative_stock,
                    expired,
                    stock_bought_errors,
                ) = self._grouped_totals(product, start_date, end_date, granularity)

                historical_stock_levels = self._calculate_historical_stock_levels(
                    product, start_date, end_date, granularity
                )

                # DEBUG – keep for now; comment out later if noisy
                print("=== PRODUCT TREND DEBUG ===")
                print("Product:", product.name, product.barcode)
                print("Period labels:", labels)
                print("Sold:", sold)
                print("Restocked:", restocked)
                print("Expired:", expired)
                print("Historical stock levels:", historical_stock_levels)
                print("Stock bought errors:", stock_bought_errors)

                if product.price_per_unit is None:
                    # No cost basis — cannot compute recommendation, just show chart
                    context.update(
                        {
                            "product": product,
                            "sold": sold,
                            "restocked": restocked,
                            "periods": labels,
                            "cumulative_stock": cumulative_stock,
                            "expired": expired,
                            "stock_bought_errors": stock_bought_errors,
                            "price_per_unit_missing_message": "Adjust cost per unit through Edit Product to enable recommendations.",
                            "current_stock": product.quantity_in_stock,
                            "historical_stock_levels": historical_stock_levels,
                        }
                    )
                else:
                    purchases, sales, expiries = get_product_stock_records(
                        product, str(start_date), str(end_date)
                    )

                    recommendation_data = recommend_inventory_action(
                        product=product,  # ✅ REQUIRED NOW
                        purchase_history=purchases,
                        sale_history=sales,
                        expiry_history=expiries,
                        timeframe_start=str(start_date),
                        timeframe_end=str(end_date),
                        cost_per_unit=float(product.price_per_unit),
                        price_per_unit=float(product.price),
                        granularity=granularity,
                    )

                    context.update(
                        {
                            "product": product,
                            "sold": sold,
                            "restocked": restocked,
                            "periods": labels,
                            "cumulative_stock": cumulative_stock,
                            "expired": expired,
                            "stock_bought_errors": stock_bought_errors,
                            "recommendation_data": recommendation_data,
                            "granularity": granularity,
                            "total_price": product.price
                            * recommendation_data["suggested_order_quantity"],
                            "current_stock": product.quantity_in_stock,
                            "historical_stock_levels": historical_stock_levels,
                        }
                    )
            else:
                messages.error(
                    request, f"No product found with barcode or name '{query}'."
                )
                context["product"] = None

        return render(request, self.template_name, context)

    def _grouped_totals(self, product, start_date, end_date, granularity):
        """
        Returns:
          sold, restocked, periods_labels, cumulative_stock, expired, stock_bought_errors
        """

        trunc_map = {
            "day": TruncDay("timestamp"),
            "week": TruncWeek("timestamp"),
            "month": TruncMonth("timestamp"),
        }
        trunc = trunc_map.get(granularity, TruncMonth("timestamp"))

        qs = (
            StockChange.objects.filter(
                product=product,
                timestamp__date__gte=start_date,
                timestamp__date__lte=end_date,
            )
            .annotate(period=trunc)
            .values("period", "change_type")
            .annotate(total=Sum("quantity"))
            .order_by("period")
        )

        # Build labels over the whole range
        periods = []
        current = start_date
        while current <= end_date:
            if granularity == "day":
                label = current.strftime("%Y-%m-%d")
                current += timedelta(days=1)
            elif granularity == "week":
                week_start = current - timedelta(days=current.weekday())
                label = f"Week of {week_start.strftime('%Y-%m-%d')}"
                current += timedelta(weeks=1)
            else:  # month
                label = current.strftime("%b %Y")
                # jump to first day of next month
                current = (current + timedelta(days=32)).replace(day=1)

            periods.append(label)

        sold = [0] * len(periods)
        restocked = [0] * len(periods)
        total_stock_changes = [0] * len(periods)
        expired = [0] * len(periods)
        stock_bought_errors = [False] * len(periods)

        label_to_index = {label: i for i, label in enumerate(periods)}

        for row in qs:
            period_date = row["period"].date()
            if granularity == "day":
                label = period_date.strftime("%Y-%m-%d")
            elif granularity == "week":
                label = f"Week of {(period_date - timedelta(days=period_date.weekday())).strftime('%Y-%m-%d')}"
            else:
                label = period_date.strftime("%b %Y")

            idx = label_to_index.get(label)
            if idx is None:
                continue

            change_type = row["change_type"]
            qty = row["total"] or 0

            if change_type == "checkout":
                sold[idx] += abs(qty)
                total_stock_changes[idx] -= abs(qty)
            elif change_type == "checkin":
                restocked[idx] += qty
                total_stock_changes[idx] += qty
            elif change_type == "error_add":
                restocked[idx] += qty
                total_stock_changes[idx] += qty
            elif change_type == "error_subtract":
                restocked[idx] -= abs(qty)
                total_stock_changes[idx] -= abs(qty)
            elif change_type == "checkin_delete1":
                restocked[idx] -= abs(qty)
                total_stock_changes[idx] -= abs(qty)
            elif change_type == "expired":
                expired[idx] += abs(qty)
                total_stock_changes[idx] -= abs(qty)

        # Clip negatives in restocked and mark errors
        for i in range(len(restocked)):
            if restocked[i] < 0:
                stock_bought_errors[i] = True
                restocked[i] = 0

        # Cumulative stock (relative, starting at 0)
        cumulative_stock = []
        running = 0
        for delta in total_stock_changes:
            running = max(0, running + delta)
            cumulative_stock.append(running)

        return sold, restocked, periods, cumulative_stock, expired, stock_bought_errors

    def _calculate_historical_stock_levels(self, product, start_date, end_date, granularity):
        """
        True stock level at end of each period label, based on:
        - All StockChange rows in [start_date, end_date]
        - Current quantity_in_stock and all changes after end_date
        """

        # 1) Build labels identical to _grouped_totals
        periods = []
        current = start_date
        while current <= end_date:
            if granularity == "day":
                periods.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            elif granularity == "week":
                week_start = current - timedelta(days=current.weekday())
                periods.append(f"Week of {week_start.strftime('%Y-%m-%d')}")
                current += timedelta(weeks=1)
            else:
                periods.append(current.strftime("%b %Y"))
                current = (current + timedelta(days=32)).replace(day=1)

        sign = {
            "checkin": +1,
            "error_add": +1,
            "checkout": -1,
            "expired": -1,
            "error_subtract": -1,
            "checkin_delete1": -1,
        }

        # 2) Daily deltas in the range
        daily_rows = (
            StockChange.objects.filter(
                product=product,
                timestamp__date__gte=start_date,
                timestamp__date__lte=end_date,
            )
            .annotate(day=TruncDate("timestamp"))
            .values("day", "change_type")
            .annotate(total=Sum("quantity"))
            .order_by("day")
        )

        daily_delta = defaultdict(int)
        for r in daily_rows:
            ct = r["change_type"]
            qty = r["total"] or 0
            daily_delta[r["day"]] += sign.get(ct, 0) * abs(qty)

        # 3) Compute net changes AFTER end_date, so we can back-solve stock_at_end_date
        after_rows = (
            StockChange.objects.filter(product=product, timestamp__date__gt=end_date)
            .values("change_type")
            .annotate(total=Sum("quantity"))
        )

        net_after_end = 0
        for r in after_rows:
            net_after_end += sign.get(r["change_type"], 0) * abs(r["total"] or 0)

        stock_at_end_date = product.quantity_in_stock - net_after_end

        # 4) Back to start_date: remove all changes within [start_date, end_date]
        net_in_range = sum(daily_delta[d] for d in _daterange(start_date, end_date))
        running = stock_at_end_date - net_in_range

        # 5) Build daily EOD stock levels, then bucket
        buckets = defaultdict(list)  # label -> [daily EOD stock values]

        for d in _daterange(start_date, end_date):
            running = max(0, running + daily_delta[d])
            if granularity == "day":
                label = d.strftime("%Y-%m-%d")
            elif granularity == "week":
                ws = d - timedelta(days=d.weekday())
                label = f"Week of {ws.strftime('%Y-%m-%d')}"
            else:
                label = d.strftime("%b %Y")
            buckets[label].append(running)

        # 6) For each label, take the last daily value (end-of-period stock)
        out = []
        last_known = 0

        for label in periods:
            vals = buckets.get(label, [])
            if not vals:
                out.append(last_known)
            else:
                value = vals[-1]
                last_known = value
                out.append(value)

        return out


# Home view
@login_required
def home(request):
   if not request.user.is_authenticated:
       return redirect('login')  # Redirect to login page
   return render(request, 'home.html')

def signup(request):
   if request.method == 'POST':
       form = UserCreationForm(request.POST)
       if form.is_valid():
           form.save()
           messages.success(request, "Your account has been created successfully! You can now log in.")
           return redirect('login')
   else:
       form = UserCreationForm()
   return render(request, 'signup.html', {'form': form})
 
class CustomLoginView(LoginView):
    def get(self, request, *args, **kwargs):
        # Redirect authenticated users to the appropriate page
        if request.user.is_authenticated:
            if request.user.is_staff:  # Redirect admins
                return redirect('inventory_display')  # Example: Admin page
            return redirect('checkin')  # Example: Regular user page
        return super().get(request, *args, **kwargs)


    def get_success_url(self):
        """
        Redirect users based on their role after a successful login.
        """
        if self.request.user.is_staff:
            return reverse('inventory_display')  # Admin-specific page
        return reverse('checkin')  # Regular user page
   
# Display all orders - Transaction page.
class OrderView(AdminRequiredMixin, View):
    template_name = 'order_view.html'

    def get(self, request):
        orders = Order.objects.all().order_by('-order_id')
        current_order_id = request.session.get('order_id')  # Get current active order
        return render(request, self.template_name, {
            'orders': orders,
            'current_order_id': current_order_id  # Pass it to template
        })

   
class OrderDetailView(View):
    template_name = 'order_detail.html'

    def get(self, request, order_id):
        # Get the order and its details
        order = get_object_or_404(Order, order_id=order_id)
        order_details = order.details.all()  # Assuming 'details' is the related name for the OrderDetail model

        # Calculate total price per item (quantity × price)
        order_details_with_total = [
            {
                'detail': detail,
                'total_price': detail.product.price * detail.quantity
            }
            for detail in order_details
        ]

        # Calculate order total before tax and after tax
        total_price_before_tax = sum(item['total_price'] for item in order_details_with_total)
        total_price_after_tax = total_price_before_tax * Decimal('1.13')  # Assuming 13% tax

        return render(request, self.template_name, {
            'order': order,
            'order_details_with_total': order_details_with_total,
            'total_price_before_tax': total_price_before_tax,
            'total_price_after_tax': total_price_after_tax,
        })
        
# change
class AddProductByIdView(LoginRequiredMixin, View):
    def post(self, request, product_id):
        order = CreateOrderView().get_order(request)
        requested_quantity = int(request.POST.get("quantity", 1))

        try:
            with transaction.atomic():
                product = Product.objects.select_for_update().get(product_id=product_id)

                if product.expiry_date and product.expiry_date < now().date():
                    messages.error(
                        request,
                        f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                        extra_tags="order",
                    )
                    return redirect("create_order")

                order_detail, _ = OrderDetail.objects.select_for_update().get_or_create(
                    order=order,
                    product=product,
                    defaults={"quantity": 0, "price": product.price},
                )

                current_qty = int(order_detail.quantity or 0)
                desired_qty = current_qty + requested_quantity

                stock = int(product.quantity_in_stock or 0)
                capped_qty = min(desired_qty, stock)

                order_detail.quantity = capped_qty
                order_detail.price = product.price
                order_detail.save(update_fields=["quantity", "price"])

            recalculate_order_totals(order)

            if stock <= 0:
                messages.warning(
                    request,
                    f"'{product.name}' is OUT OF STOCK (0). Add accepted — quantity stays 0.",
                    extra_tags="order",
                )
            elif capped_qty < desired_qty:
                messages.warning(
                    request,
                    f"'{product.name}' capped at {stock} (in stock).",
                    extra_tags="order",
                )
            else:
                messages.success(
                    request,
                    f"Added {requested_quantity} unit(s) of '{product.name}'. (Now {capped_qty}/{stock})",
                    extra_tags="order",
                )

            return redirect("create_order")

        except Product.DoesNotExist:
            messages.error(request, "Product not found.", extra_tags="order")
            return redirect("create_order")

    

class CreateOrderView(LoginRequiredMixin, View):
    template_name = 'order_form.html'

    def get_order(self, request):
        order_id = request.session.get('order_id')
        
        if order_id:
            try:
                # Retrieve only non-submitted orders
                return Order.objects.get(order_id=order_id, submitted=False)
            except Order.DoesNotExist:
                del request.session['order_id']  # Clear invalid session data

        # Create a new order if none is found or session is invalid
        return self._create_new_order(request)

    def _create_new_order(self, request):
            last_order = Order.objects.order_by('-order_id').first()
            next_order_id = 1 if not last_order else last_order.order_id + 1
            order = Order.objects.create(order_id=next_order_id, total_price=Decimal('0.00'))
            request.session['order_id'] = order.order_id
            return order

    def get(self, request, *args, **kwargs):
        order = self.get_order(request)
        form = BarcodeForm()

        # change
        name_query = request.GET.get('name_query', '')
        search_results = []
        all_products = [
            {
                'id': p['product_id'],
                'name': p['name'],
                'price': str(p['price']),
                'quantity_in_stock': p['quantity_in_stock'],
                'item_number': p['item_number']  # ✅ Add this line
            } for p in Product.objects.values('product_id', 'name', 'price', 'quantity_in_stock', 'item_number')
        ]


        # change
        if name_query:
            search_results = Product.objects.filter(name__icontains=name_query).order_by('name')

        # Order details and totals
        order_details = order.details.all().order_by('-order_date')
        total_price_before_tax = sum(detail.product.price * detail.quantity for detail in order_details)
        total_price_after_tax = total_price_before_tax * Decimal('1.13')
        
        # Change name_query, search results
        return render(request, self.template_name, {
            'order': order,
            'form': form,
            'order_details': order_details,
            'total_price_before_tax': total_price_before_tax,
            'total_price_after_tax': total_price_after_tax,
            'search_results': search_results,
            'name_query': name_query,
            'all_products': all_products,

        })

    def post(self, request, *args, **kwargs):
        order = self.get_order(request)
        form = BarcodeForm(request.POST)

        if not form.is_valid():
            return self._render_order_page(request, order, form)

        barcode = (form.cleaned_data.get("barcode") or "").strip()
        requested_quantity = int(form.cleaned_data.get("quantity") or 1)

        try:
            with transaction.atomic():
                product = find_product_by_barcode(barcode, for_update=True)
                if not product:
                    raise Product.DoesNotExist


                # expired guard
                if product.expiry_date and product.expiry_date < now().date():
                    messages.error(
                        request,
                        f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                        extra_tags="order",
                    )
                    return redirect("create_order")

                # lock/get order line (create a row even if qty becomes 0)
                order_detail, _ = OrderDetail.objects.select_for_update().get_or_create(
                    order=order,
                    product=product,
                    defaults={"quantity": 0, "price": product.price},
                )

                current_qty = int(order_detail.quantity or 0)
                desired_qty = current_qty + requested_quantity

                stock = int(product.quantity_in_stock or 0)

                # ✅ HARD CAP to available stock
                capped_qty = min(desired_qty, stock)

                order_detail.quantity = capped_qty
                order_detail.price = product.price
                order_detail.save(update_fields=["quantity", "price"])

            recalculate_order_totals(order)

            # messages
            if stock <= 0:
                messages.warning(
                    request,
                    f"'{product.name}' is OUT OF STOCK (0). Scan accepted — quantity stays 0.",
                    extra_tags="order",
                )
            elif capped_qty < desired_qty:
                messages.warning(
                    request,
                    f"'{product.name}' capped at {stock} (in stock).",
                    extra_tags="order",
                )
            else:
                messages.success(
                    request,
                    f"Added {requested_quantity} unit(s) of '{product.name}'. (Now {capped_qty}/{stock})",
                    extra_tags="order",
                )

            return redirect("create_order")

        except Product.DoesNotExist:
            messages.error(request, f"No product found with barcode '{barcode}'.", extra_tags="order")
            return self._render_order_page(request, order, form)


    def _render_order_page(self, request, order, form):
        order_details = order.details.all().order_by('-order_date')
        total_price_before_tax = sum(detail.product.price * detail.quantity for detail in order_details)
        total_price_after_tax = total_price_before_tax * Decimal('1.13')

        # Reconstruct the product list for autocomplete
        all_products = [
            {
                'id': p['product_id'],
                'name': p['name'],
                'price': str(p['price']),
                'quantity_in_stock': p['quantity_in_stock']
            } for p in Product.objects.values('product_id', 'name', 'price', 'quantity_in_stock','item_number')
        ]

        return render(request, self.template_name, {
            'order': order,
            'form': form,
            'order_details': order_details,
            'total_price_before_tax': total_price_before_tax,
            'total_price_after_tax': total_price_after_tax,
            'all_products': all_products,
            'name_query': '',  # Clear or maintain the previous query if needed
            'search_results': [],  # Optional: keep previous manual search results if needed
        })


class SubmitOrderView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        if 'order_id' not in request.session:
            messages.error(request, "No active order found.")
            return redirect('create_order')

        order = get_object_or_404(Order, order_id=request.session['order_id'])

        if not order.details.exists():
            messages.error(request, "Cannot submit an empty order.")
            return redirect('create_order')

        unfulfilled_lines = []  # collect shortages for a warning

        with transaction.atomic():
            details = order.details.select_for_update().select_related("product")

            # Lock products
            product_ids = [d.product_id for d in details]
            products_by_id = {
                p.product_id: p
                for p in Product.objects.select_for_update().filter(product_id__in=product_ids)
            }

            for d in details:
                p = products_by_id[d.product_id]
                requested = int(d.quantity or 0)
                available = int(p.quantity_in_stock or 0)

                fulfilled = min(requested, available)
                missing = requested - fulfilled

                # ✅ Fulfill what we can (never negative stock)
                if fulfilled > 0:
                    p.quantity_in_stock = available - fulfilled
                    p.save(update_fields=["quantity_in_stock"])

                    record_stock_change(
                        product=p,
                        qty=fulfilled,
                        change_type="checkout",
                        note=f"Order {order.order_id} submission (fulfilled)"
                    )

                # ✅ Audit the unfulfilled amount without touching stock
                if missing > 0:
                    unfulfilled_lines.append(f"{p.name} (missing {missing})")

                    # This creates StockChange row but does NOT change counters
                    record_stock_change(
                        product=p,
                        qty=missing,
                        change_type="checkout_unfulfilled",
                        note=f"Order {order.order_id} submission (out of stock / unfulfilled)"
                    )

                # RecentlyPurchasedProduct: log what was actually fulfilled (optional)
                if fulfilled > 0:
                    rp, created = RecentlyPurchasedProduct.objects.get_or_create(product=p)
                    rp.quantity = (rp.quantity or 0) + fulfilled
                    rp.save(update_fields=["quantity"])

            order.submitted = True
            order.save(update_fields=["submitted"])
            del request.session['order_id']

        if unfulfilled_lines:
            messages.warning(
                request,
                "Order submitted, but some items could not be fulfilled due to low/zero stock: "
                + ", ".join(unfulfilled_lines),
                extra_tags="order"
            )
        else:
            messages.success(request, "Order submitted successfully.", extra_tags="order")

        return redirect('create_order')



# deletes item from the purchase order
@login_required
def delete_order_item(request, item_id):
    with transaction.atomic():
        # ✅ ADD select_for_update() HERE
        order_detail = OrderDetail.objects.select_for_update().get(od_id=item_id)
        order = order_detail.order
        product = Product.objects.select_for_update().get(product_id=order_detail.product.product_id)

        if order_detail.quantity > 1:
            order_detail.quantity -= 1
            order_detail.save()

        else:
            product.quantity_in_stock += order_detail.quantity
            product.save()
            order_detail.delete()

        recalculate_order_totals(order)

    messages.success(request, f"1 unit of {product.name} removed from the order.")
    return redirect('create_order')
    

# View for order success page
"""
class OrderSuccessView(View):
   template_name = 'order_success.html'
 
   def get(self, request, *args, **kwargs):
       return render(request, self.template_name)
"""
#Change - Function to annotate changes

def record_stock_change(
    product: Product,
    qty: int,
    change_type: str,
    note: str = ""
) -> None:
    """
    Creates a StockChange row *and* updates per-product counters.

    • Positive qty  -> stock added
    • Negative qty  -> stock removed
    """
    with transaction.atomic():

        # 1) persist the audit trail
        StockChange.objects.create(
            product=product,
            change_type=change_type,
            quantity=qty,
            note=note or None,
        )

        # 2) update running totals on Product
        if change_type == "checkin":
            product.stock_bought += abs(qty)  
        elif change_type == "checkout":
            product.stock_sold += abs(qty) 
        elif change_type == "expired":
            product.stock_expired += abs(qty)
        elif change_type == "error_subtract":
            product.stock_bought -= abs(qty) 
        elif change_type == "error_add":
            product.stock_bought += abs(qty) 
        elif change_type == "checkin_delete1":
            product.stock_bought -= abs(qty) 

        # -- optional: keep other change types (return/adjustment) out of the counters,
        #             or handle them however you prefer.

        product.save(
            update_fields=["stock_bought", "stock_sold", "stock_expired"]
        )


# DELETES ON ITEM ON CHECKIN BUTTON -- CHANGE - there is a bug with the checkout technically, not a checkout but a misclick 
def delete_one(request, product_id):
    if request.method != "POST":
        return redirect("checkin")

    with transaction.atomic():
        product = get_object_or_404(Product.objects.select_for_update(), pk=product_id)

        if product.quantity_in_stock <= 0:
            messages.error(
                request,
                f"Cannot subtract. {product.name} is already out of stock.",
                extra_tags="checkin",
            )
        else:
            product.quantity_in_stock -= 1
            product.save(update_fields=["quantity_in_stock"])
            record_stock_change(
                product,
                qty=1,
                change_type="checkin_delete1",
                note="1 unit removed due to UI misclick during check-in"
            )
            messages.success(
                request,
                f"Adjusted: 1 unit removed from {product.name}'s stock.",
                extra_tags="checkin",
            )
    return redirect(f"{reverse('checkin')}?barcode={product.barcode}")


#add1 checkin
def AddQuantityView(request, product_id):
    if request.method != "POST":
        return redirect("checkin")

    try:
        quantity_to_add = int(request.POST.get("quantity_to_add", 1))
    except ValueError:
        messages.error(request, "Please enter a valid quantity.", extra_tags="checkin")
        return redirect("inventory_display")

    if quantity_to_add < 1:
        messages.error(request, "Quantity to add must be at least 1.", extra_tags="checkin")
        return redirect("inventory_display")

    with transaction.atomic():
        product = get_object_or_404(Product.objects.select_for_update(), product_id=product_id)
        product.quantity_in_stock += quantity_to_add
        product.save(update_fields=["quantity_in_stock"])
        record_stock_change(product, qty=quantity_to_add, change_type="checkin", note="Manual add via UI")

    messages.success(
        request,
        f"{quantity_to_add} unit(s) of {product.name} added to stock.",
        extra_tags="checkin",
    )

    # Redirect to GET checkin with barcode query param to avoid double-post on refresh
    return redirect(f"{reverse('checkin')}?barcode={product.barcode}")


#add products without barcode
class AddProductByIdCheckinView(LoginRequiredMixin, View):
    def post(self, request, product_id):
        quantity = int(request.POST.get("quantity", 1))

        with transaction.atomic():
            try:
                product = Product.objects.select_for_update().get(product_id=product_id)
            except Product.DoesNotExist:
                messages.error(request, "Product not found.", extra_tags="checkin")
                return redirect("checkin")

            product.quantity_in_stock += quantity
            product.save(update_fields=["quantity_in_stock"])
            record_stock_change(product, qty=quantity, change_type="checkin", note="Add via ID button")

        messages.success(
            request,
            f"{quantity} unit(s) of '{product.name}' successfully added to stock.",
            extra_tags="checkin",
        )
        return render(request, "checkin.html", {
            "product": product,
            "edit_form": EditProductForm(instance=product),
            "all_products": list(Product.objects.values("product_id","name","price","quantity_in_stock","item_number")),
            "categories": Category.objects.all(),
            "search_results": [],
        })


#checkin views
class CheckinProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"

    def get(self, request):
        barcode = (request.GET.get("barcode") or "").strip()
        product = find_product_by_barcode(barcode) if barcode else None


        query = (request.GET.get("name_query") or "").strip()
        search_results = Product.objects.filter(name__icontains=query) if query else []

        edit_form = EditProductForm(instance=product) if product else None

        return render(request, self.template_name, {
            "search_results": search_results,
            "all_products": list(
                Product.objects.values(
                    "product_id", "name", "price", "quantity_in_stock", "item_number"
                )
            ),
            "product": product,
            "edit_form": edit_form,
            "categories": Category.objects.all(),
        })

    def post(self, request):
        barcode = (request.POST.get("barcode") or "").strip()
        if not barcode:
            messages.error(
                request,
                "❌ No barcode provided. Please scan a barcode.",
                extra_tags="checkin error"
            )
            return self._render_no_product(request)

        # Try to find product in *store* catalogue first
        with transaction.atomic():
            product = find_product_by_barcode(barcode, for_update=True)

            if product:
                product.quantity_in_stock += 1
                product.save(update_fields=["quantity_in_stock"])
                record_stock_change(product, qty=1, change_type="checkin", note="Barcode scan")

                messages.success(
                    request,
                    f"✅ 1 unit of {product.name} added to stock.",
                    extra_tags="checkin success"
                )
                return redirect(f"{reverse('checkin')}?barcode={product.barcode}")


        # ─────────────────────────────────────────────
        # Not in store → try MASTER.csv
        # ─────────────────────────────────────────────
        master_row = get_master_catalog_entry(barcode)

        params = {
            "barcode": barcode,  # always pass scanned barcode
        }

        if master_row:
            # Map real Master.csv headers → your form fields
            params.update({
                # from 'ITEM DESC' column
                "name": master_row.get("ITEM DESCRIPTION", ""),

                # McKesson item number
                "item_number": master_row.get("DIN", ""),

                # optional: pack/size description
                "unit_size": master_row.get("PRODUCT FORMAT", ""),

                # price – try SUGGESTED first, then cost
                "price_per_unit": _clean_price(
                    master_row.get("COST")
                ),
                "UPC": master_row.get("GTIN/UPC (unit)","")
            })

            messages.info(
                request,
                f"Scanned barcode {barcode} is not in store catalogue. "
                "Details pulled from master catalogue – please review and save.",
                extra_tags="checkin",
            )
        else:
            messages.warning(
                request,
                f"Scanned barcode {barcode} is not in the store or master catalogue. "
                "Please add the product details.",
                extra_tags="checkin",
            )

        # Redirect to AddProductView with query parameters
        add_url = reverse("new_product")  # URL name for AddProductView
        return redirect(f"{add_url}?{urlencode(params)}")


    # helper
    def _render_no_product(self, request):
        return render(
            request,
            self.template_name,
            {
                "all_products": list(
                    Product.objects.values(
                        "product_id", "name", "price",
                        "quantity_in_stock", "item_number"
                    )
                ),
                "categories": Category.objects.all(),
            },
        )

    
class CheckinEditProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"

    def post(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        old_quantity = product.quantity_in_stock

        form = EditProductForm(request.POST, instance=product)

        if form.is_valid():
            updated = form.save(commit=False)
            new_quantity = updated.quantity_in_stock

            if new_quantity != old_quantity:
                change = "error_add" if new_quantity > old_quantity else "error_subtract"
                record_stock_change(
                    product=product,
                    qty=abs(new_quantity - old_quantity),
                    change_type=change,
                    note="Product updated via check-in inline edit"
                )

            updated.save()

            messages.success(
                request,
                f"✅ Updated {updated.name}.",
                extra_tags="checkin success"
            )
            return redirect(f"{reverse('checkin')}?barcode={updated.barcode}")

        # invalid -> re-render same page with errors + current product
        messages.error(
            request,
            "❌ Could not update product. Please review the highlighted fields.",
            extra_tags="checkin error"
        )

        return render(request, self.template_name, {
            "search_results": [],
            "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number")),
            "product": product,
            "edit_form": form,                   # ✅ show errors inline
            "categories": Category.objects.all(),
        })


# Edit product.
class EditProductView(View):
    template_name = 'edit_product.html'

    def get(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        form = EditProductForm(instance=product)

        # Use GET param or fallback to referring page
        next_url = request.GET.get('next') or request.META.get('HTTP_REFERER', '/inventory_display')
        return render(request, self.template_name, {
            'form': form,
            'next': next_url,
            'product': product
        })

    def post(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        old_quantity = product.quantity_in_stock  # Store old quantity for stock change
        form = EditProductForm(request.POST, instance=product)
        next_url = request.POST.get('next', '/inventory_display')

        if form.is_valid():
            updated_product = form.save(commit=False)  # Save without committing to DB yet
            new_quantity = updated_product.quantity_in_stock  # Get updated quantity
            
            if new_quantity - old_quantity != 0:
                # Record stock change only if quantity has changed
                change = "error_add" if new_quantity > old_quantity else "error_subtract"
                record_stock_change(
                    product=product,
                    qty=abs(new_quantity - old_quantity),
                    change_type=change,
                    note="Product updated via edit form"
                )
            
            form.save()  # Now save the updated product
            messages.success(request, "Product updated successfully.")
            return redirect(next_url)
        else:
            messages.error(request, "Failed to update the product.")
            return render(request, self.template_name, {
                'form': form,
                'next': next_url,
                'product': product
            })


# Add a new product
class AddProductView(LoginRequiredMixin, View):
    template_name = 'new_product.html'

    def get(self, request):
        next_url = request.GET.get('next', '')
        categories = Category.objects.all()

        initial_data = {
            'name':        request.GET.get('name', ''),
            'brand':       request.GET.get('brand', ''),
            'item_number': request.GET.get('item_number', ''),
            'barcode':     request.GET.get('barcode', ''),
            'price_per_unit':       request.GET.get('price_per_unit', ''),
        }
        form = AddProductForm(initial=initial_data)

        return render(request, self.template_name, {
            'categories': categories,
            'form': form,
            'next': next_url
        })


    def post(self, request):
        form = AddProductForm(request.POST)
        next_url = request.POST.get('next', '')

        if form.is_valid():
            raw_barcode = (form.cleaned_data.get('barcode') or '').strip()
            barcode = raw_barcode or None
            item_number = (form.cleaned_data.get('item_number') or '').strip()

            # Pre-check duplicates (case-insensitive + leading-zero tolerant)
            if barcode:
                normalized = _normalize_barcode(raw_barcode)
                if Product.objects.filter(barcode__regex=rf"^0*{normalized}$").exists():
                    form.add_error(
                        "barcode",
                        f"Barcode '{raw_barcode}' already exists (with or without leading zeros)."
                    )

            if item_number and Product.objects.filter(item_number__iexact=item_number).exists():
                form.add_error("item_number", f"Item number '{item_number}' already exists.")


            if form.errors:
                return render(request, self.template_name, {
                    'categories': Category.objects.all(),
                    'form': form,
                    'next': next_url
                })

            try:
                with transaction.atomic():
                    product = form.save(commit=False)
                    product.barcode = barcode   # <-- actually apply your normalized value
                    product.save()
                    record_stock_change(
                        product=product,
                        qty=int(product.quantity_in_stock or 0),
                        change_type="checkin",
                        note="New product added via form"
                    )

                messages.success(request, "Product added successfully.", extra_tags='new_product')
                return redirect(next_url) if next_url else redirect('checkin')

            except IntegrityError as e:
                print("INTEGRITY ERROR >>>", repr(str(e)))  # <-- THIS is the truth from the DB

                msg = str(e).lower()
                if "barcode" in msg:
                    form.add_error("barcode", "That barcode already exists (or a blank barcode already exists).")
                elif "item_number" in msg or "item number" in msg:
                    form.add_error("item_number", "That item number already exists (or blank item numbers conflict).")
                else:
                    form.add_error(None, f"Database error: {str(e)}")

        return render(request, self.template_name, {
            'categories': Category.objects.all(),
            'form': form,
            'next': next_url
        })


# Display inventory
class InventoryView(LoginRequiredMixin, View):
    template_name = 'inventory_display.html'

    def get(self, request):
        # Get filter parameters from the request
        selected_category_id = request.GET.get('category_id', '')
        barcode_query = (request.GET.get("barcode_query") or "").strip()
        name_query = request.GET.get('name_query', '')
        sort_column = request.GET.get('sort', 'name')  # Default sorting column is 'name'
        sort_direction = request.GET.get('direction', 'asc')  # Default sorting direction is ascending

        # Query products based on filters
        products = Product.objects.all()
        if selected_category_id:
            products = products.filter(category_id=selected_category_id)

        if barcode_query:
            product = find_product_by_barcode(barcode_query)
            if product: 
                products = products.filter(product_id=product.product_id)
            else:
                products = products.none()  # No matching product found
        if name_query:
            products = products.filter(name__icontains=name_query)

        # Apply sorting dynamically
        valid_sort_columns = ['name', 'quantity_in_stock', 'price', 'expiry_date']

        if sort_column in valid_sort_columns:
            sort_prefix = '-' if sort_direction == 'desc' else ''
            products = products.order_by(f'{sort_prefix}{sort_column}')
        else:
            # Fallback to default sort
            products = products.order_by('name')

        # Paginate the filtered products
        paginator = Paginator(products, 100)  # Show 100 items per page
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        # Pass all query parameters and the paginator to the template
        return render(request, self.template_name, {
            'page_obj': page_obj,
            'categories': Category.objects.all(),
            'selected_category_id': selected_category_id, 
            'barcode_query': barcode_query,
            'name_query': name_query,
            'sort_column': sort_column,
            'sort_direction': sort_direction,
        })

# Change 
class ExpiredProductView(LoginRequiredMixin, View):
    template_name = 'expired_products.html'

    def get(self, request):
        date_filter = request.GET.get('date_filter', '')
        name_query = request.GET.get('name_query', '').strip()
        pid = request.GET.get("pid", None)

        products = self._filter_products(date_filter, name_query)
        product = Product.objects.filter(pk=pid).first() if pid else None

        return render(request, self.template_name, {
            "products": products,
            "product": product,
            "date_filter": date_filter,
            "name_query": name_query,
            "all_products": list(Product.objects.values("product_id", "name")),
        })

    def post(self, request):
        barcode      = request.POST.get("barcode", "").strip()
        date_filter  = request.POST.get("date_filter", "")
        name_query   = request.POST.get("name_query", "").strip()
        product = find_product_by_barcode(barcode)
        products     = self._filter_products(date_filter, name_query)

        if product and request.POST.get("retire_expired") == "1":
            try:
                qty = int(request.POST.get("retire_quantity"))
            except (ValueError, TypeError):
                qty = 0

            if qty > 0 and qty <= product.quantity_in_stock:
                # Update stock
                product.quantity_in_stock -= qty
                product.save(update_fields=["quantity_in_stock", "stock_expired"])

                # Log the change
                record_stock_change(product, qty=qty, change_type="expired", note="Marked as expired from expired product view")
                messages.success(request, f"{qty} units of '{product.name}' marked as expired.")
            else:
                messages.error(request, "Invalid quantity to retire.")

        return render(request, self.template_name, {
            "products": products,
            "product": product,
            "date_filter": date_filter,
            "name_query": name_query,
            "all_products": list(Product.objects.values("product_id", "name")),
        })


    def _filter_products(self, date_filter, name_query):
        today = date.today()
        if date_filter == "1_week":
            end = today + timedelta(weeks=1)
            qs  = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "3_months":
            end = today + relativedelta(months=3)
            qs  = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        else:
            qs  = Product.objects.filter(expiry_date__lt=today)

        if name_query:
            qs = qs.filter(name__icontains=name_query)
        return qs.exclude(expiry_date__isnull=True).order_by("expiry_date")
    
     
# View for displaying low-stock items
class LowStockView(AdminRequiredMixin, View):
    template_name = 'low_stock.html'
    threshold = 2

    def get(self, request):
        low_stock_products = Product.objects.filter(quantity_in_stock__lt=self.threshold).order_by('name')
        recently_purchased = RecentlyPurchasedProduct.objects.all().order_by('-order_date')

        paginator_low_stock = Paginator(low_stock_products, 100)
        page_number_low_stock = request.GET.get('page')
        page_obj_low_stock = paginator_low_stock.get_page(page_number_low_stock)

        paginator_recent = Paginator(recently_purchased, 80)
        page_number_recent = request.GET.get('page_recent')
        page_obj_recent = paginator_recent.get_page(page_number_recent)

        return render(request, self.template_name, {
            'page_obj_low_stock': page_obj_low_stock,
            'page_obj_recent': page_obj_recent,
            'threshold': self.threshold,
        })


# Delete a recently purchased product
class DeleteRecentlyPurchasedProductView(LoginRequiredMixin, View):
   def post(self, request, id):  # Use 'id' to match the model's primary key field name
       try:
           recently_purchased = RecentlyPurchasedProduct.objects.get(id=id)
           product_name = recently_purchased.product.name  # Capture the name before deletion
           recently_purchased.delete()
           messages.success(request, f"{product_name} has been deleted from the recently purchased list.")
       except RecentlyPurchasedProduct.DoesNotExist:
           messages.error(request, "The selected product does not exist in the recently purchased list.")
       return redirect('low_stock')


class DeleteAllRecentlyPurchasedView(LoginRequiredMixin, View):
   def post(self, request):
       # Delete all recently purchased products
       RecentlyPurchasedProduct.objects.all().delete()
       messages.success(request, "All recently purchased products have been deleted.")
       return redirect('low_stock')

# Delete an item
@login_required
def delete_item(request, product_id):
    product = get_object_or_404(Product, product_id=product_id)
    product.delete()
    messages.success(request, f"Product '{product.name}' has been deleted.")

    # Redirect back to inventory page with query parameters
    page = request.POST.get('page', 1)
    category_id = request.POST.get('category_id', '')
    barcode_query = request.POST.get('barcode_query', '')
    name_query = request.POST.get('name_query', '')

    redirect_url = f"{reverse('inventory_display')}?page={page}"
    if category_id:
        redirect_url += f"&category_id={category_id}"
    if barcode_query:
        redirect_url += f"&barcode_query={barcode_query}"
    if name_query:
        redirect_url += f"&name_query={name_query}"

    return redirect(redirect_url)

# Delete all orders
class DeleteAllOrdersView(LoginRequiredMixin, View):
   def post(self, request, *args, **kwargs):
       Order.objects.all().delete()
       request.session['next_order_id'] = 1
       messages.success(request, "All orders have been deleted successfully.")
       return redirect('order_view')




# Item list view
class ItemListView(LoginRequiredMixin,View):
   template_name = 'item_list.html'
   form_class = ItemForm

   def get(self, request):
       form = self.form_class()
       items = Item.objects.all()
       return render(request, self.template_name, {'form': form, 'items': items})

   def post(self, request):
       if 'delete' in request.POST:
           item_id = request.POST.get('item_id')
           item = get_object_or_404(Item, id=item_id)
           item.delete()
           messages.success(request, f"Item '{item.item_name}' has been deleted.")
           return redirect('item_list')
       elif 'update_checked' in request.POST:
           item_id = request.POST.get('item_id')
           is_checked = request.POST.get('is_checked') == 'on'
           item = get_object_or_404(Item, id=item_id)
           item.is_checked = is_checked
           item.save()
           return redirect('item_list')
       else:
           form = self.form_class(request.POST)
           if form.is_valid():
               form.save()
               return redirect('item_list')


       items = Item.objects.all()
       return render(request, self.template_name, {'form': form, 'items': items})

@login_required
def update_product_settings(request, product_id):
    if request.method == 'POST':
        product = get_object_or_404(Product, product_id=product_id)
        expiry_input = request.POST.get('expiry_date')
        taxable_input = request.POST.get('taxable')
        category_id = request.POST.get('category')

        try:
            from django.utils.dateparse import parse_date

            if expiry_input:
                product.expiry_date =  parse_date(expiry_input)

            if taxable_input in ['True', 'False']:
                product.taxable = (taxable_input == 'True')

            if category_id:
                try:
                    new_category = Category.objects.get(pk=category_id)
                    product.category = new_category
                except Category.DoesNotExist:
                    pass  # silently ignore bad input

            product.save()
            messages.success(request, "Product settings updated.", extra_tags='checkin')
        except Exception as e:
            messages.error(request, "Failed to update product settings.", extra_tags='checkin')

    return redirect(request.META.get('HTTP_REFERER', 'checkin'))


