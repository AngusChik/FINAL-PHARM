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
from django.utils.dateparse import parse_date
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
from django.http import HttpResponse
from django.db.models import Sum, Q
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth, TruncDate
from decimal import Decimal
from django.db import connection
import io
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter, portrait
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.barcode import code128

# --- Constants from your script ---
LABEL_WIDTH = 2.00 * inch
LABEL_HEIGHT = 1.25 * inch
LEFT_MARGIN, RIGHT_MARGIN = 0.25 * inch, 0.25 * inch
TOP_MARGIN, BOTTOM_MARGIN = 0.50 * inch, 0.50 * inch
COLUMNS, ROWS = 4, 8
LABELS_PER_PAGE = COLUMNS * ROWS
LEFT_PADDING, RIGHT_PADDING = 6, 6
TOP_PADDING, BOTTOM_PADDING = 4, 4

class LabelPrintingView(LoginRequiredMixin, View):
    template_name = "label_printing.html"

    def get(self, request):
        # 1. Manual Session Queue
        session_queue = request.session.get("label_queue", [])
        
        # 2. Permanent "Print Label" Category Items
        category_items = Product.objects.filter(
            category__name__icontains="Print Label", 
            status=True
        )

        # Search functionality
        query = request.GET.get("q", "").strip()
        search_results = []
        if query:
            search_results = Product.objects.filter(
                Q(name__icontains=query) | Q(barcode__icontains=query)
            ).distinct()[:10]

        return render(request, self.template_name, {
            "session_queue": session_queue,
            "category_items": category_items,
            "query": query,
            "search_results": search_results,
        })

    def post(self, request):
        # Handle adding items to the queue
        if "add_product" in request.POST:
            product_id = request.POST.get("product_id")
            product = get_object_or_404(Product, pk=product_id)
            queue = request.session.get("label_queue", [])
            
            queue.append({
                "name": product.name,
                "brand": product.brand or "",
                "item_number": product.item_number or "",
                "barcode": product.barcode or "",
                "price": str(product.price)
            })
            request.session["label_queue"] = queue
            messages.success(request, f"Added {product.name} to label queue.")
        
        elif "clear_queue" in request.POST:
            request.session["label_queue"] = []
            messages.info(request, "Label queue cleared.")
            
        elif "remove_index" in request.POST:
            idx = int(request.POST.get("remove_index"))
            queue = request.session.get("label_queue", [])
            if 0 <= idx < len(queue):
                queue.pop(idx)
            request.session["label_queue"] = queue

        return redirect("label_printing")

class GenerateLabelPDFView(LoginRequiredMixin, View):
    def get(self, request):
        # 1. Get items from the temporary session queue
        session_queue = request.session.get("label_queue", [])
        
        # 2. Get permanent items from the "Print Label" category
        category_items = Product.objects.filter(
            category__name__icontains="Print Label", 
            status=True
        )

        # 3. Merge both into a single list for the PDF engine
        final_queue = []
        
        # Add Category items first
        for p in category_items:
            final_queue.append({
                "name": p.name,
                "brand": p.brand or "",
                "item_number": p.item_number or "",
                "barcode": p.barcode or "",
                "price": str(p.price)
            })
            
        # Append session items
        final_queue.extend(session_queue)

        if not final_queue:
            messages.error(request, "No labels to print.")
            return redirect("label_printing")

        # Start PDF Generation
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=portrait(letter))
        PAGE_WIDTH, PAGE_HEIGHT = portrait(letter)

        def wrap_text(text, font_name, font_size, max_width):
            words = text.split()
            lines, current = [], ""
            for w in words:
                test = (current + " " + w) if current else w
                if stringWidth(test, font_name, font_size) <= max_width:
                    current = test
                else:
                    if current: lines.append(current)
                    current = w
            if current: lines.append(current)
            return lines

        def draw_label(c, x, y, data):
            name = data.get("name", "")
            brand = data.get("brand", "")
            item_num = data.get("item_number", "")
            bc_val = data.get("barcode", "")
            price = f"${float(data.get('price', 0)):.2f}"

            c.setFont("Helvetica-Bold", 10)
            max_w = LABEL_WIDTH - LEFT_PADDING - RIGHT_PADDING
            lines = wrap_text(name, "Helvetica-Bold", 10, max_w)[:4]
            for i, line in enumerate(lines):
                c.drawCentredString(x + LABEL_WIDTH/2, y + LABEL_HEIGHT - 10 - (i*11), line)

            base_y = y + BOTTOM_PADDING
            body_x = x + LEFT_PADDING

            if bc_val:
                try:
                    barcode = code128.Code128(bc_val, barHeight=16, barWidth=0.9, humanReadable=False)
                    barcode.drawOn(c, body_x, base_y + 20)
                    c.setFont("Helvetica", 6)
                    c.drawString(body_x, base_y + 14, bc_val)
                except: pass

            if item_num:
                c.setFont("Helvetica", 6)
                c.drawString(body_x, base_y + 8, f"Item #: {item_num}")
            
            if brand:
                c.setFont("Helvetica", 6)
                c.drawString(body_x, base_y + 2, brand[:25])

            c.setFont("Helvetica-Bold", 17)
            c.drawRightString(x + LABEL_WIDTH - RIGHT_PADDING, base_y + 4, price)

        # Layout Calculations
        usable_w = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
        usable_h = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
        h_gutter = (usable_w - (COLUMNS * LABEL_WIDTH)) / (COLUMNS - 1) if COLUMNS > 1 else 0
        v_gutter = (usable_h - (ROWS * LABEL_HEIGHT)) / (ROWS - 1) if ROWS > 1 else 0

        # LOOP THROUGH final_queue
        for count, item in enumerate(final_queue):
            col = count % COLUMNS
            row_num = (count // COLUMNS) % ROWS
            x = LEFT_MARGIN + col * (LABEL_WIDTH + h_gutter)
            y_top = PAGE_HEIGHT - TOP_MARGIN - row_num * (LABEL_HEIGHT + v_gutter)
            y = y_top - LABEL_HEIGHT
            
            draw_label(c, x, y, item)
            
            # Use final_queue here for page break logic
            if (count + 1) % LABELS_PER_PAGE == 0 and (count + 1) < len(final_queue):
                c.showPage()

        c.save()
        buffer.seek(0)
        return HttpResponse(buffer, content_type='application/pdf')

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


# Utility generator
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

        try:
            end_date = datetime.strptime(request.GET.get("end", ""), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            end_date = date.today()

        try:
            start_date = datetime.strptime(request.GET.get("start", ""), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            start_date = end_date - timedelta(days=365)

        all_products = list(Product.objects.values("product_id", "name", "barcode", "item_number", "price", "quantity_in_stock"))

        # --- Overview stats (always computed, no product needed) ---
        top_sellers    = list(Product.objects.filter(stock_sold__gt=0).order_by("-stock_sold")[:5])
        out_of_stock_count = Product.objects.filter(status=True, quantity_in_stock=0).count()
        low_stock_count    = Product.objects.filter(status=True, quantity_in_stock__gt=0, quantity_in_stock__lte=3).count()

        context = {
            "query": query,
            "chart_type": chart_type,
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity,
            "all_products": all_products,
            "search_results": None,
            "top_sellers": top_sellers,
            "out_of_stock_count": out_of_stock_count,
            "low_stock_count": low_stock_count,
        }

        if query:
            product = find_product_by_barcode(query)
            search_results = Product.objects.filter(name__icontains=query) | Product.objects.filter(barcode__icontains=query)
            context["search_results"] = search_results.distinct()

            if product:
                # 1. Get Grouped Data for Charts (Including Missed Sales)
                (sold, restocked, labels, cumulative_stock, expired, 
                 stock_bought_errors, missed_sales) = self._grouped_totals(product, start_date, end_date, granularity)

                # 2. Get Historical Levels (Fixed AttributeError)
                historical_stock_levels = self._calculate_historical_stock_levels(product, start_date, end_date, granularity)

                context_data = {
                    "product": product,
                    "sold": sold,
                    "restocked": restocked,
                    "missed_sales": missed_sales,
                    "periods": labels,
                    "cumulative_stock": cumulative_stock,
                    "expired": expired,
                    "stock_bought_errors": stock_bought_errors,
                    "current_stock": product.quantity_in_stock,
                    "historical_stock_levels": historical_stock_levels,
                    "recent_changes": StockChange.objects.filter(
                        product=product
                    ).order_by("-timestamp")[:20],
                }

                if product.price_per_unit is None:
                    context_data["price_per_unit_missing_message"] = "Adjust cost per unit to enable recommendations."
                else:
                    # 3. Get Full History for Algorithm
                    purchases, sales, expiries, unfulfilled = get_product_stock_records(
                        product, str(start_date), str(end_date)
                    )

                    recommendation_data = recommend_inventory_action(
                        product=product,
                        purchase_history=purchases,
                        sale_history=sales,
                        expiry_history=expiries,
                        unfulfilled_history=unfulfilled, # ✅ Pass unfulfilled orders
                        timeframe_start=str(start_date),
                        timeframe_end=str(end_date),
                        cost_per_unit=float(product.price_per_unit),
                        price_per_unit=float(product.price),
                        granularity=granularity,
                    )
                    
                    context_data["recommendation_data"] = recommendation_data
                    context_data["total_price"] = product.price * recommendation_data["suggested_order_quantity"]

                context.update(context_data)
            else:
                messages.error(request, f"No product found with barcode or name '{query}'.")

        return render(request, self.template_name, context)

    def _grouped_totals(self, product, start_date, end_date, granularity):
        """
        Returns chart data arrays: sold, restocked, periods, stock, expired, errors, missed_sales
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
            else:
                label = current.strftime("%b %Y")
                current = (current + timedelta(days=32)).replace(day=1)
            periods.append(label)

        length = len(periods)
        sold = [0] * length
        restocked = [0] * length
        expired = [0] * length
        missed_sales = [0] * length
        stock_bought_errors = [False] * length
        total_stock_changes = [0] * length

        label_to_index = {label: i for i, label in enumerate(periods)}

        for row in qs:
            period_date = row["period"].date()
            if granularity == "day": label = period_date.strftime("%Y-%m-%d")
            elif granularity == "week": 
                ws = period_date - timedelta(days=period_date.weekday())
                label = f"Week of {ws.strftime('%Y-%m-%d')}"
            else: label = period_date.strftime("%b %Y")

            idx = label_to_index.get(label)
            if idx is None: continue

            ctype = row["change_type"]
            qty = row["total"] or 0

            if ctype == "checkout":
                sold[idx] += abs(qty)
                total_stock_changes[idx] -= abs(qty)
            elif ctype == "checkout_unfulfilled":
                missed_sales[idx] += abs(qty) # Track separately, no physical stock change
            elif ctype == "checkin" or ctype == "error_add":
                restocked[idx] += qty
                total_stock_changes[idx] += qty
            elif ctype == "error_subtract" or ctype == "checkin_delete1":
                restocked[idx] -= abs(qty)
                total_stock_changes[idx] -= abs(qty)
            elif ctype == "expired":
                expired[idx] += abs(qty)
                total_stock_changes[idx] -= abs(qty)

        for i in range(length):
            if restocked[i] < 0:
                stock_bought_errors[i] = True
                restocked[i] = 0

        cumulative_stock = []
        running = 0
        for delta in total_stock_changes:
            running = max(0, running + delta)
            cumulative_stock.append(running)

        return sold, restocked, periods, cumulative_stock, expired, stock_bought_errors, missed_sales

    def _calculate_historical_stock_levels(self, product, start_date, end_date, granularity):
        """
        True stock level at end of each period label.
        """
        # 1) Build labels
        periods = []
        current = start_date
        while current <= end_date:
            if granularity == "day":
                periods.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            elif granularity == "week":
                ws = current - timedelta(days=current.weekday())
                periods.append(f"Week of {ws.strftime('%Y-%m-%d')}")
                current += timedelta(weeks=1)
            else:
                periods.append(current.strftime("%b %Y"))
                current = (current + timedelta(days=32)).replace(day=1)

        sign = {
            "checkin": +1, "error_add": +1,
            "checkout": -1, "expired": -1,
            "error_subtract": -1, "checkin_delete1": -1,
        }

        # 2) Daily deltas
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
            daily_delta[r["day"]] += sign.get(r["change_type"], 0) * abs(r["total"] or 0)

        # 3) Back-calculate stock from today
        after_rows = (
            StockChange.objects.filter(product=product, timestamp__date__gt=end_date)
            .values("change_type")
            .annotate(total=Sum("quantity"))
        )
        net_after_end = 0
        for r in after_rows:
            net_after_end += sign.get(r["change_type"], 0) * abs(r["total"] or 0)

        stock_at_end_date = product.quantity_in_stock - net_after_end

        # 4) Calculate starting stock
        net_in_range = sum(daily_delta[d] for d in _daterange(start_date, end_date))
        running = stock_at_end_date - net_in_range

        # 5) Bucket into periods
        buckets = defaultdict(list)
        for d in _daterange(start_date, end_date):
            running = max(0, running + daily_delta[d])
            if granularity == "day": label = d.strftime("%Y-%m-%d")
            elif granularity == "week": 
                ws = d - timedelta(days=d.weekday())
                label = f"Week of {ws.strftime('%Y-%m-%d')}"
            else: label = d.strftime("%b %Y")
            buckets[label].append(running)

        out = []
        last_known = 0
        for label in periods:
            vals = buckets.get(label, [])
            if not vals: out.append(last_known)
            else:
                last_known = vals[-1]
                out.append(last_known)
        return out

def get_cart(request):
    return request.session.setdefault("cart", {})

class OutOfStockView(AdminRequiredMixin, View):
    template_name = "out_of_stock.html"

    def get(self, request):
        products = Product.objects.filter(
            status=True, quantity_in_stock=0
        ).order_by("name")
        return render(request, self.template_name, {"products": products})


class LowStockTrendView(AdminRequiredMixin, View):
    template_name = "low_stock_trend.html"

    def get(self, request):
        products = Product.objects.filter(
            status=True, quantity_in_stock__gt=0, quantity_in_stock__lte=3
        ).order_by("quantity_in_stock", "name")
        return render(request, self.template_name, {"products": products})


def save_cart(request, cart):
    request.session["cart"] = cart
    request.session.modified = True

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
                return redirect('create_order')  # Example: Admin page
            return redirect('checkin')  # Example: Regular user page
        return super().get(request, *args, **kwargs)


    def get_success_url(self):
        """
        Redirect users based on their role after a successful login.
        """
        if self.request.user.is_staff:
            return reverse('create_order')  # Admin-specific page
        return reverse('checkin')  # Regular user page
   
# Display all orders - Transaction page.
class OrderView(AdminRequiredMixin, View):
    template_name = 'order_view.html'

    def get(self, request):
        orders = Order.objects.all().order_by('-order_id')
        current_order_id = request.session.get('order_id')  # Get current active order

        # Log total_price for debugging
        for order in orders:
            print(f"Order {order.order_id} total_price: {order.total_price}")  # <-- debug log

        return render(request, self.template_name, {
            'orders': orders,
            'current_order_id': current_order_id
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
        inventory_mode = request.POST.get("inventory_mode") == "true"
        # ✅ Validate quantity input
        try:
            requested_quantity = int(request.POST.get("quantity", 1))
            if requested_quantity < 0:
                messages.error(request, "Quantity cannot be negative.", extra_tags="order")
                return redirect("create_order")
        except (ValueError, TypeError):
            messages.error(request, "Invalid quantity value.", extra_tags="order")
            return redirect("create_order")

        try:
            requested_quantity = int(request.POST.get("quantity", 1))

            # ✅ FIXED: Add transaction and select_for_update
            with transaction.atomic():
                # ✅ CRITICAL FIX: Lock the row to prevent race conditions
                product = Product.objects.select_for_update().get(product_id=product_id)
                
                if inventory_mode:
                    product.status = True
                    product.save(update_fields=['status'])

                # Expiry guard (read-only check)
                if product.expiry_date and product.expiry_date < now().date():
                    messages.error(
                        request,
                        f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                        extra_tags="order",
                    )
                    return redirect("create_order")

                # ─── SESSION CART (safe - no DB changes) ───
                cart = request.session.setdefault("cart", {})
                pid = str(product.product_id)

                cart.setdefault(pid, {
                    "quantity": 0,
                    "price": str(product.price),
                    "name": product.name,
                })

                current_qty = cart[pid]["quantity"]
                desired_qty = current_qty + requested_quantity

                stock = int(product.quantity_in_stock or 0)
                capped_qty = min(desired_qty, stock)

                cart[pid]["quantity"] = capped_qty
                request.session.modified = True
                # ─────────────────────────────────────────────

            # ✅ Messages AFTER transaction (lock released)
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

            return redirect(f"{reverse('create_order')}?inventory_mode={str(inventory_mode).lower()}")
        except Product.DoesNotExist:
            messages.error(request, "Product not found.", extra_tags="order")
            return redirect("create_order")
    

class CreateOrderView(LoginRequiredMixin, View):
    template_name = "order_form.html"

    def get_order(self, request):
        order_id = request.session.get("order_id")

        if order_id:
            try:
                return Order.objects.get(order_id=order_id, submitted=False)
            except Order.DoesNotExist:
                request.session.pop("order_id", None)

        order = Order.objects.create(total_price=Decimal("0.00"))
        request.session["order_id"] = order.order_id
        return order


    def get(self, request, *args, **kwargs):
        form = BarcodeForm()
        order = self.get_order(request)

        cart = request.session.get("cart", {})

        # 🔁 Rehydrate products for template
        product_ids = [int(pid) for pid in cart.keys()]
        products = Product.objects.filter(product_id__in=product_ids)
        
        products_by_id = {p.product_id: p for p in products}

        order_items = []
        total_price_before_tax = Decimal("0.00")
        cart_modified = False

        for pid_str, line in list(cart.items()):
            pid = int(pid_str)
            product = products_by_id.get(pid)
            
            # ✅ Check if product was deleted
            if not product:
                messages.warning(
                    request,
                    f"Removed '{line.get('name', 'Unknown product')}' from cart - product no longer exists.",
                    extra_tags="order"
                )
                del cart[pid_str]
                cart_modified = True
                continue

            # If status is still False here it means something unusual happened
            # (e.g. product deactivated by another user between the post and get).
            # Warn but keep it in the cart — don't silently eject it.
            if not product.status:
                messages.warning(
                    request,
                    f"⚠️ '{product.name}' in cart is currently inactive.",
                    extra_tags="order"
                )
            
            # ⚠️ CHANGED: Just warn about expired, don't auto-remove
            # Users may have overridden this too
            if product.expiry_date and product.expiry_date < now().date():
                # Don't remove, just show info message
                messages.info(
                    request,
                    f"Note: '{product.name}' in cart is expired (Expiry: {product.expiry_date}). "
                    f"It was added with override.",
                    extra_tags="order"
                )
                # Don't use 'continue' - let it stay in cart
            
            # ✅ Validate quantity doesn't exceed current stock
            qty = int(line["quantity"])
            if qty > product.quantity_in_stock:
                old_qty = qty
                qty = product.quantity_in_stock
                cart[pid_str]["quantity"] = qty
                cart_modified = True
                messages.warning(
                    request,
                    f"Reduced '{product.name}' quantity from {old_qty} to {qty} (current stock).",
                    extra_tags="order"
                )

            subtotal = product.price * qty
            total_price_before_tax += subtotal

            order_items.append({
                "product": product,
                "quantity": qty,
                "subtotal": subtotal,
            })

        # ✅ Save cart changes if any validation occurred
        if cart_modified:
            request.session["cart"] = cart
            request.session.modified = True

        total_price_after_tax = total_price_before_tax * Decimal("1.13")

        # Search
        name_query = request.GET.get("name_query", "")
        search_results = (
            Product.objects.filter(name__icontains=name_query).order_by("name")
            if name_query else []
        )

        # Autocomplete payload
        all_products = list(Product.objects.values(
            "product_id",
            "name",
            "price",
            "quantity_in_stock",
            "item_number",
            "barcode",
            "expiry_date",
            "status",  # ✅ Make sure this is included
        ))

        return render(request, self.template_name, {
            "order": order,
            "form": form,
            "order_items": order_items,
            "total_price_before_tax": total_price_before_tax,
            "total_price_after_tax": total_price_after_tax,
            "name_query": name_query,
            "search_results": search_results,
            "all_products": all_products,
        })

    # ─────────────────────────────
    # POST — SCAN BARCODE (SESSION)
    # ─────────────────────────────
    def post(self, request, *args, **kwargs):
        form = BarcodeForm(request.POST)

        if not form.is_valid():
            return redirect("create_order")

        barcode            = form.cleaned_data["barcode"].strip()
        requested_quantity = int(form.cleaned_data.get("quantity") or 1)
        override_expiry    = request.POST.get("override_expiry")   == "1"
        override_inactive  = request.POST.get("override_inactive") == "1"

        product = find_product_by_barcode(barcode)

        if not product:
            messages.error(request, f"No product found with barcode '{barcode}'.", extra_tags="order")
            return redirect("create_order")

        with transaction.atomic():
            product = Product.objects.select_for_update().get(pk=product.pk)

            # ── 1. Inactive guard ──────────────────────────────────────────
            if not product.status:
                if override_inactive:
                    # Activate in DB NOW — so get() won't eject it from cart
                    product.status = True
                    product.save(update_fields=["status"])
                else:
                    messages.error(
                        request,
                        f"Cannot add '{product.name}' — product is inactive.",
                        extra_tags="order",
                    )
                    return redirect("create_order")

            # ── 2. Expiry guard ────────────────────────────────────────────
            if product.expiry_date and product.expiry_date < now().date():
                if not override_expiry:
                    messages.error(
                        request,
                        f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                        extra_tags="order",
                    )
                    return redirect("create_order")

            # ── 3. Add to session cart ─────────────────────────────────────
            cart = request.session.setdefault("cart", {})
            pid  = str(product.product_id)

            cart.setdefault(pid, {
                "name":     product.name,
                "price":    str(product.price),
                "quantity": 0,
            })

            current_qty = cart[pid]["quantity"]
            desired_qty = current_qty + requested_quantity
            stock       = int(product.quantity_in_stock or 0)
            capped_qty  = min(desired_qty, stock)

            cart[pid]["quantity"] = capped_qty
            request.session.modified = True

        # ── Messages (outside transaction) ────────────────────────────────
        override_notes = []
        if override_inactive: override_notes.append("product activated")
        if override_expiry:   override_notes.append("expired override")

        if stock <= 0:
            messages.warning(request,
                f"'{product.name}' is OUT OF STOCK (0). Scan accepted — quantity stays 0.",
                extra_tags="order")
        elif capped_qty < desired_qty:
            messages.warning(request,
                f"'{product.name}' capped at {stock} (in stock).",
                extra_tags="order")
        elif override_notes:
            messages.warning(request,
                f"⚠️ Added '{product.name}' ({', '.join(override_notes)}).",
                extra_tags="order")
        else:
            messages.success(request,
                f"Added {requested_quantity} unit(s) of '{product.name}'. (Now {capped_qty}/{stock})",
                extra_tags="order")

        return redirect("create_order")



class SubmitOrderView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        cart = request.session.get("cart")

        if not cart:
            messages.error(request, "Cannot submit an empty order.", extra_tags="order")
            return redirect("create_order")

        order = get_object_or_404(
            Order,
            order_id=request.session.get("order_id"),
            submitted=False
        )

        unfulfilled_lines = []

        with transaction.atomic():
            # 🔒 Lock all products in cart
            product_ids = [int(pid) for pid in cart.keys()]
            products = (
                Product.objects
                .select_for_update()
                .filter(product_id__in=product_ids)
            )

            products_by_id = {p.product_id: p for p in products}

            for pid_str, line in cart.items():
                pid = int(pid_str)
                requested = int(line["quantity"])
                product = products_by_id.get(pid)

                if not product:
                    continue

                available = int(product.quantity_in_stock or 0)
                fulfilled = min(requested, available)
                missing = requested - fulfilled

                # ✅ Create order line with what was requested
                OrderDetail.objects.create(
                    order=order,
                    product=product,
                    quantity=fulfilled,
                    price=product.price,
                )

                # ✅ Decrement stock ONLY here
                if fulfilled > 0:
                    product.quantity_in_stock = available - fulfilled
                    product.save(update_fields=["quantity_in_stock"])

                    record_stock_change(
                        product=product,
                        qty=fulfilled,
                        change_type="checkout",
                        note=f"Order {order.order_id} submission"
                    )

                if missing > 0:
                    unfulfilled_lines.append(f"{product.name} (missing {missing})")

                    record_stock_change(
                        product=product,
                        qty=missing,
                        change_type="checkout_unfulfilled",
                        note=f"Order {order.order_id} submission (unfulfilled)"
                    )

                # Optional analytics
                if fulfilled > 0:
                    rp, _ = RecentlyPurchasedProduct.objects.get_or_create(product=product)
                    rp.quantity = (rp.quantity or 0) + fulfilled
                    rp.save(update_fields=["quantity"])

            # ✅ Finalize order
            order.submitted = True
            order.save(update_fields=["submitted"])

            # ✅ Clear session state
            request.session.pop("cart", None)
            request.session.pop("order_id", None)
            request.session.modified = True

        if unfulfilled_lines:
            messages.warning(
                request,
                "Order submitted, but some items were not fulfilled: "
                + ", ".join(unfulfilled_lines),
                extra_tags="order",
            )
        else:
            messages.success(
                request,
                "Order submitted successfully.",
                extra_tags="order",
            )

        return redirect("create_order")


# deletes item from the purchase order
@login_required
def delete_order_item(request, product_id):  # Changed product_id to item_id
    cart = request.session.get("cart", {})
    pid = str(product_id)  # Use item_id here as well

    if pid not in cart:
        messages.warning(request, "Item not found in cart.")
        return redirect("create_order")

    if cart[pid]["quantity"] > 1:
        cart[pid]["quantity"] -= 1
    else:
        del cart[pid]

    request.session.modified = True
    messages.success(request, "1 unit removed from the order.")
    return redirect("create_order")


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
    Creates a StockChange row and updates per-product counters.
    
    ✅ FIXED: Now handles all change types including unfulfilled orders
    """
    with transaction.atomic():
        # 1) Persist the audit trail
        StockChange.objects.create(
            product=product,
            change_type=change_type,
            quantity=qty,
            note=note or None,
        )

        # 2) Update running totals on Product
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
        
        # ✅ FIXED: Add unfulfilled tracking
        # Note: You'll need to add a new field to Product model:
        # stock_unfulfilled = models.IntegerField(default=0)
        elif change_type == "checkout_unfulfilled":
            # Track missed sales separately
            if hasattr(product, 'stock_unfulfilled'):
                product.stock_unfulfilled = (product.stock_unfulfilled or 0) + abs(qty)
        
        # ✅ FIXED: Add deletion tracking
        elif change_type == "deletion":
            # Stock was lost due to product deletion
            # Track in expired as "waste"
            product.stock_expired += abs(qty)

        product.save(
            update_fields=["stock_bought", "stock_sold", "stock_expired"]
        )



# DELETES ONE ITEM ON CHECKIN BUTTON
@login_required
def delete_one(request, product_id):
    """
    Subtract 1 unit from product stock (with inventory mode support).
    """
    if request.method != "POST":
        return redirect("checkin")

    # Capture inventory mode
    inventory_mode = request.POST.get("inventory_mode") == "true"

    with transaction.atomic():
        product = get_object_or_404(
            Product.objects.select_for_update(), 
            pk=product_id
        )

        if product.quantity_in_stock <= 0:
            messages.error(
                request,
                f"Cannot subtract. {product.name} is already out of stock.",
                extra_tags="checkin error",
            )
        else:
            product.quantity_in_stock -= 1
            
            update_fields = ["quantity_in_stock"]

            # Inventory Mode logic
            if inventory_mode:
                product.status = True
                update_fields.append("status")
                messages.success(
                    request, 
                    f"Adjusted (-1) & Activated {product.name}.", 
                    extra_tags="checkin success"
                )
            else:
                messages.success(
                    request,
                    f"Adjusted: 1 unit removed from {product.name}'s stock.",
                    extra_tags="checkin success",
                )

            product.save(update_fields=update_fields)
            
            record_stock_change(
                product,
                qty=1,
                change_type="checkin_delete1",
                note="1 unit removed via UI"
            )

    # Redirect preserving inventory_mode
    return redirect(
        f"{reverse('checkin')}?barcode={product.barcode}&inventory_mode={str(inventory_mode).lower()}"
    )


#add1 checkin
@login_required
def AddQuantityView(request, product_id):
    """
    Add quantity to product stock (with inventory mode support).
    """
    if request.method != "POST":
        return redirect("checkin")

    # Capture inventory mode
    inventory_mode = request.POST.get("inventory_mode") == "true"

    # ✅ FIXED: Validate quantity input
    try:
        quantity_to_add = int(request.POST.get("amount", 1))
        if quantity_to_add <= 0:
            messages.error(
                request,
                "Quantity must be greater than 0.",
                extra_tags="checkin error"
            )
            return redirect("checkin")
        if quantity_to_add > 1000:  # Sanity check
            messages.error(
                request,
                "Quantity too large. Maximum 1000 units per operation.",
                extra_tags="checkin error"
            )
            return redirect("checkin")
    except (ValueError, TypeError):
        messages.error(
            request,
            "Invalid quantity value.",
            extra_tags="checkin error"
        )
        return redirect("checkin")

    with transaction.atomic():
        product = get_object_or_404(
            Product.objects.select_for_update(), 
            product_id=product_id
        )
        
        product.quantity_in_stock += quantity_to_add
        
        update_fields = ["quantity_in_stock"]

        # Inventory Mode logic
        if inventory_mode:
            product.status = True
            update_fields.append("status")
            messages.success(
                request, 
                f"Added (+{quantity_to_add}) & Activated {product.name}.", 
                extra_tags="checkin success"
            )
        else:
            messages.success(
                request,
                f"{quantity_to_add} unit(s) of {product.name} added to stock.",
                extra_tags="checkin success",
            )

        product.save(update_fields=update_fields)
        
        record_stock_change(
            product, 
            qty=quantity_to_add, 
            change_type="checkin", 
            note="Manual add via UI"
        )

    # Redirect preserving inventory_mode
    return redirect(
        f"{reverse('checkin')}?barcode={product.barcode}&inventory_mode={str(inventory_mode).lower()}"
    )

# add products without barcode (triggered via Search/Autocomplete)
class AddProductByIdCheckinView(LoginRequiredMixin, View):
    def post(self, request, product_id):
        # 1. Capture Inputs
        quantity = int(request.POST.get("quantity", 1))
        # ✅ Capture inventory_mode state from the hidden input or form data
        inventory_mode = request.POST.get("inventory_mode") == "true"

        with transaction.atomic():
            try:
                # Use select_for_update to lock the row
                product = Product.objects.select_for_update().get(product_id=product_id)
                
                # 2. Update Quantity
                product.quantity_in_stock += quantity
                update_fields = ["quantity_in_stock"]

                # ✅ 3. APPLY INVENTORY MODE LOGIC
                if inventory_mode:
                    product.status = True  # Force product to be Active
                    update_fields.append("status")
                    msg_text = f"✅ {product.name}: Counted (+{quantity}) & Activated."
                else:
                    msg_text = f"✅ {quantity} unit(s) of '{product.name}' added to inventory."

                product.save(update_fields=update_fields)
                
                # Record the audit trail
                record_stock_change(
                    product, 
                    qty=quantity, 
                    change_type="checkin", 
                    note="Add via search ID" + (" (Inventory Mode)" if inventory_mode else "")
                )
                
            except Product.DoesNotExist:
                messages.error(request, "Product not found.", extra_tags="checkin error")
                return redirect("checkin")

        messages.success(request, msg_text, extra_tags="checkin success")
        
        # 4. Re-fetch all data to keep the 'Inventory Mode' UI active
        all_products = list(Product.objects.values(
            "product_id", "name", "price", "quantity_in_stock", 
            "item_number", "barcode", "status", "taxable"
        ))

        # 5. Return the same template with context
        return render(request, "checkin.html", {
            "product": product, 
            "inventory_mode": inventory_mode, # ✅ CRITICAL: Keeps the UI toggle ON
            "edit_form": EditProductForm(instance=product),
            "all_products": all_products,
            "categories": Category.objects.all(),
            "search_results": [],
        })

#checkin views
class CheckinProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"

    def get(self, request):
        barcode = (request.GET.get("barcode") or "").strip()
        
        # Check if we are in inventory mode
        inventory_mode = request.GET.get("inventory_mode") == "true"

        product = None
        if barcode:
            product = find_product_by_barcode(barcode)

        query = (request.GET.get("name_query") or "").strip()
        search_results = []
        if query:
            # ✅ FIXED: Search by name, barcode, AND item_number
            search_results = Product.objects.filter(
                Q(name__icontains=query) | 
                Q(barcode__icontains=query) |
                Q(item_number__icontains=query)
            ).distinct()[:20]  # Limit results

        edit_form = EditProductForm(instance=product) if product else None

        return render(request, self.template_name, {
            "search_results": search_results,
            "inventory_mode": inventory_mode, 
            "all_products": list(
                Product.objects.values(
                    "product_id", "name", "price", "quantity_in_stock", 
                    "item_number", "barcode"  # ✅ Add barcode to autocomplete data
                )
            ),
            "product": product,
            "edit_form": edit_form,
            "categories": Category.objects.all(),
        })

    def post(self, request):
        barcode = (request.POST.get("barcode") or "").strip()
        
        # 1. Capture the toggle state from the form
        inventory_mode = request.POST.get("inventory_mode") == "true"

        if not barcode:
            messages.error(
                request,
                "❌ No barcode provided. Please scan a barcode.",
                extra_tags="checkin error"
            )
            return self._render_no_product(request, inventory_mode)
            
        # Try to find product in *store* catalogue first
        with transaction.atomic():
            product = find_product_by_barcode(barcode, for_update=True)

            if product:
                # Standard Check-in logic: Add 1 to stock
                product.quantity_in_stock += 1
                
                # ✅ INVENTORY MODE LOGIC
                if inventory_mode:
                    product.status = True # Force Active
                    note = "Inventory Count (Activated)"
                    msg_tag = "checkin success"
                    msg_text = f"✅ {product.name}: Counted (+1) & Activated."
                else:
                    note = "Barcode scan"
                    if not product.status:
                        msg_tag = "checkin warning"
                        msg_text = f"{product.name} is INACTIVE. Stock updated, but item is not sellable."
                    else:
                        msg_tag = "checkin success"
                        msg_text = f"✅ 1 unit of {product.name} added to stock."

                product.save(update_fields=["quantity_in_stock", "status"])
                record_stock_change(product, qty=1, change_type="checkin", note=note)

                messages.add_message(request, messages.INFO, msg_text, extra_tags=msg_tag)
                
                # Redirect WITH the inventory_mode param so the toggle stays on
                return redirect(f"{reverse('checkin')}?barcode={product.barcode}&inventory_mode={str(inventory_mode).lower()}")

        # ─────────────────────────────────────────────
        # Not in store → try MASTER.csv
        # ─────────────────────────────────────────────
        master_row = get_master_catalog_entry(barcode)
        
        # Pass params to Add Product form
        params = {
            "barcode": barcode,
            "next": f"/checkin/?inventory_mode={str(inventory_mode).lower()}" # Return to checkin with mode on
        }

        if master_row:
            params.update({
                "name": master_row.get("ITEM DESCRIPTION", ""),
                "item_number": master_row.get("DIN", ""),
                "unit_size": master_row.get("PRODUCT FORMAT", ""),
                "price_per_unit": _clean_price(master_row.get("COST")),
                "UPC": master_row.get("GTIN/UPC (unit)",""),
                # If in inventory mode, pre-set status to True in the new form
                "status": "on" if inventory_mode else None 
            })
            messages.info(request, "Details pulled from master catalogue.", extra_tags="checkin")
        else:
            messages.warning(request, "Barcode not found. Please add manually.", extra_tags="checkin")

        add_url = reverse("new_product")
        return redirect(f"{add_url}?{urlencode(params)}")

    def _render_no_product(self, request, inventory_mode=False):
        return render(
            request,
            self.template_name,
            {
                "inventory_mode": inventory_mode,
                "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number")),
                "categories": Category.objects.all(),
            },
        )

    
class CheckinEditProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"
    
    def post(self, request, product_id):
        # ✅ Capture inventory_mode from the form
        inventory_mode = request.POST.get("inventory_mode") == "true"
        
        # ✅ ADD TRANSACTION
        with transaction.atomic():
            product = Product.objects.select_for_update().get(product_id=product_id)
            old_quantity = product.quantity_in_stock

            # ✅ 1. Normalize the Date Format
            post_data = request.POST.copy()
            raw_date = post_data.get('expiry_date', '').strip().rstrip('-')

            if raw_date:
                try:
                    clean_date = datetime.strptime(raw_date, '%d-%m-%Y').date()
                    post_data['expiry_date'] = clean_date.strftime('%Y-%m-%d')
                except ValueError:
                    pass

            # ✅ 2. Initialize form with the normalized post_data
            form = EditProductForm(post_data, instance=product)
            
            if form.is_valid():
                updated = form.save(commit=False)
                new_quantity = updated.quantity_in_stock

                # Stock change tracking logic
                if new_quantity != old_quantity:
                    change = "error_add" if new_quantity > old_quantity else "error_subtract"
                    record_stock_change(
                        product=updated,
                        qty=abs(new_quantity - old_quantity),
                        change_type=change,
                        note="Product updated via check-in inline edit"
                    )

                updated.save()
                form.save_m2m()

                messages.success(
                    request,
                    f"✅ Updated {updated.name}.",
                    extra_tags="checkin success"
                )
                # ✅ Preserve inventory_mode in redirect URL
                return redirect(
                    f"{reverse('checkin')}?barcode={updated.barcode}&inventory_mode={str(inventory_mode).lower()}"
                )

        # ✅ 3. Failure State: Re-render with errors (outside transaction)
        messages.error(
            request,
            "❌ Could not update product. Please review the highlighted fields.",
            extra_tags="checkin error"
        )

        return render(request, self.template_name, {
            "search_results": [],
            "inventory_mode": inventory_mode, # Ensure toggle doesn't reset on error
            "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number", "barcode")),
            "product": product,
            "edit_form": form,
            "categories": Category.objects.all(),
        })


class RevertPrintLabelCategoryView(LoginRequiredMixin, View):
    def post(self, request):
        # Target all products currently in the Print Label category
        print_label_products = Product.objects.filter(
            category__name__icontains="Print Label"
        )
        
        reverted_count = 0
        with transaction.atomic():
            for p in print_label_products:
                if p.previous_category:
                    p.category = p.previous_category
                    p.previous_category = None # Clear the memory
                    p.save(update_fields=['category', 'previous_category'])
                    reverted_count += 1
        
        if reverted_count > 0:
            messages.success(request, f"Reverted {reverted_count} products to their original categories.")
        else:
            messages.info(request, "No products had a stored category to revert to.")
            
        return redirect('label_printing')


# Edit product.
class EditProductView(LoginRequiredMixin, View):
    template_name = 'edit_product.html'

    def get(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        form = EditProductForm(instance=product)

        next_url = request.GET.get('next') or request.META.get(
            'HTTP_REFERER', '/inventory_display'
        )

        return render(request, self.template_name, {
            'form': form,
            'next': next_url,
            'product': product
        })


    def post(self, request, product_id):
            product = get_object_or_404(Product, product_id=product_id)

            old_category = product.category
            old_quantity = product.quantity_in_stock
            
            # 1. Create a mutable copy of the POST data to fix the date string
            post_data = request.POST.copy()
            date_str = post_data.get('expiry_date', '').strip().rstrip('-')
            
            # 2. Manual normalization: If user typed DD-MM-YYYY, convert to YYYY-MM-DD
            # so Django's internal validation doesn't reject it immediately.
            if date_str:
                try:
                    # Try parsing the format you are sending from the frontend
                    valid_date = datetime.strptime(date_str, '%d-%m-%Y').date()
                    post_data['expiry_date'] = valid_date.strftime('%Y-%m-%d')
                except ValueError:
                    # If it's already YYYY-MM-DD or invalid, let the form handle it
                    pass

            # 3. Use the fixed post_data
            form = EditProductForm(post_data, instance=product)
            next_url = request.POST.get('next', '/inventory_display')

            if not form.is_valid():
                # DEBUG: Uncomment the line below to see exact errors in your terminal
                # print(form.errors) 
                messages.error(request, "Failed to update. Please check the date format (DD-MM-YYYY).")
                return render(request, self.template_name, {
                    'form': form,
                    'next': next_url,
                    'product': product
                })

            updated_product = form.save(commit=False)
            new_category = updated_product.category

            # --- CATEGORY MEMORY LOGIC ---
            # Checks if moving INTO the Print Label category
            if new_category and "PRINT LABEL" in new_category.name.upper():
                if old_category and "PRINT LABEL" not in old_category.name.upper():
                    updated_product.previous_category = old_category
            # Clear memory if moving to a standard category
            elif new_category and "PRINT LABEL" not in new_category.name.upper():
                updated_product.previous_category = None

            # --- STOCK CHANGE TRACKING ---
            delta = updated_product.quantity_in_stock - old_quantity
            if delta != 0:
                record_stock_change(
                    product=updated_product,
                    qty=abs(delta),
                    change_type="error_add" if delta > 0 else "error_subtract",
                    note="Product updated via edit form"
                )

            updated_product.save()
            form.save_m2m() 

            messages.success(request, f"Product '{updated_product.name}' updated successfully.")
            return redirect(next_url)

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
        # 1. Normalize the date string before validation
        post_data = request.POST.copy()
        date_str = post_data.get('expiry_date', '').strip().rstrip('-')
        
        if date_str:
            try:
                # Parse the user-friendly DD-MM-YYYY format to standard YYYY-MM-DD
                valid_date = datetime.strptime(date_str, '%d-%m-%Y').date()
                post_data['expiry_date'] = valid_date.strftime('%Y-%m-%d')
            except ValueError:
                # If parsing fails, leave it as-is and let the form validator handle the error
                pass

        # 2. Initialize the form with mutated data
        form = AddProductForm(post_data)
        next_url = request.POST.get('next', '') or 'checkin'

        # 3. Core Validation Check
        # Django's is_valid() catches missing required fields and incorrect types
        if form.is_valid():
            raw_barcode = (form.cleaned_data.get('barcode') or '').strip()
            barcode = raw_barcode or None
            item_number = (form.cleaned_data.get('item_number') or '').strip()

            # 4. Custom Business Logic Validation (Duplicates)
            if barcode:
                normalized = _normalize_barcode(raw_barcode)
                if Product.objects.filter(barcode__regex=rf"^0*{normalized}$").exists():
                    form.add_error("barcode", f"Barcode '{raw_barcode}' already exists.")

            if item_number and Product.objects.filter(item_number__iexact=item_number).exists():
                form.add_error("item_number", f"Item number '{item_number}' already exists.")

            # If custom checks added errors, return the form immediately
            if form.errors:
                return render(request, self.template_name, {
                    'categories': Category.objects.all(),
                    'form': form,
                    'next': next_url
                })

            # 5. Atomic Save and Exception Handling
            try:
                with transaction.atomic():
                    product = form.save(commit=False)
                    product.barcode = barcode
                    product.previous_category = None 
                    product.save()
                    
                    # Safety check for stock recording
                    stock_qty = product.quantity_in_stock if product.quantity_in_stock is not None else 0
                    record_stock_change(
                        product=product,
                        qty=int(stock_qty),
                        change_type="checkin",
                        note="New product added via form"
                    )

                messages.success(request, f"Product '{product.name}' added successfully.", extra_tags='new_product')
                return redirect(next_url)

            except IntegrityError as e:
                # Catch database-level unique constraint violations
                msg = str(e).lower()
                if "barcode" in msg:
                    form.add_error("barcode", "A product with this barcode already exists.")
                elif "item_number" in msg:
                    form.add_error("item_number", "A product with this item number already exists.")
                else:
                    form.add_error(None, f"Database error: {str(e)}")
            except Exception as e:
                # Catch-all for unexpected crashes to prevent a 500 error page
                form.add_error(None, f"An unexpected error occurred: {str(e)}")

        # 6. Fallback: Re-render with errors
        # This reaches if form.is_valid() was False or an exception occurred
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

# ✅ Update the valid columns list
        valid_sort_columns = [
            'barcode', 
            'status', 
            'item_number', 
            'name', 
            'quantity_in_stock', 
            'price', 
            'expiry_date'
        ]

        if sort_column in valid_sort_columns:
            sort_prefix = '-' if sort_direction == 'desc' else ''
            products = products.order_by(f'{sort_prefix}{sort_column}')
        else:
            # Fallback to default sort if column is invalid or reset
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
        barcode = request.POST.get("barcode", "").strip()
        date_filter = request.POST.get("date_filter", "")
        name_query = request.POST.get("name_query", "").strip()
        
        products = self._filter_products(date_filter, name_query)
        product = None

        if barcode:
            product = find_product_by_barcode(barcode)

        if product and request.POST.get("retire_expired") == "1":
            # ✅ Validate quantity
            try:
                qty = int(request.POST.get("retire_quantity", 0))
            except (ValueError, TypeError):
                qty = 0

            if qty <= 0:
                messages.error(request, "Quantity must be greater than 0.")
            elif qty > product.quantity_in_stock:
                messages.error(
                    request, 
                    f"Cannot retire {qty} units. Only {product.quantity_in_stock} in stock."
                )
            else:
                # ✅ FIXED: Wrap in transaction with row locking
                with transaction.atomic():
                    # ✅ Lock the product row
                    product = Product.objects.select_for_update().get(pk=product.pk)
                    
                    # Double-check stock hasn't changed
                    if qty > product.quantity_in_stock:
                        messages.error(
                            request,
                            "Stock level changed. Please try again."
                        )
                    else:
                        # Update stock
                        product.quantity_in_stock -= qty
                        product.save(update_fields=["quantity_in_stock"])

                        # Log the change
                        record_stock_change(
                            product, 
                            qty=qty, 
                            change_type="expired", 
                            note="Marked as expired from expired product view"
                        )
                        
                        messages.success(
                            request, 
                            f"{qty} units of '{product.name}' marked as expired."
                        )

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
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "3_months":
            end = today + relativedelta(months=3)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        else:
            qs = Product.objects.filter(expiry_date__lt=today)

        if name_query:
            qs = qs.filter(name__icontains=name_query)
        return qs.exclude(expiry_date__isnull=True).order_by("expiry_date")
    
     
# View for displaying low-stock items
class LowStockView(AdminRequiredMixin, View):
    template_name = 'low_stock.html'
    threshold = 2

    def get(self, request):
        low_stock_products = Product.objects.filter(quantity_in_stock__lt=self.threshold,status=True).order_by('name')
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

class ExportRecentlyPurchasedCSVView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        # Create the HttpResponse object with the appropriate CSV header.
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="recently_purchased_{now().strftime("%Y%m%d_%H%M")}.csv"'

        writer = csv.writer(response)
        # Write Header Row
        writer.writerow(['Product Name', 'Barcode', 'Item Number', 'Brand', 'Units Bought', 'Current Stock Level'])

        # Fetch all items
        items = RecentlyPurchasedProduct.objects.all().select_related('product')

        for item in items:
            writer.writerow([
                item.product.name if item.product else "N/A",
                item.product.barcode if item.product else "N/A",
                item.product.item_number if item.product else "N/A",
                item.product.brand if item.product else "N/A",
                item.quantity,
                item.product.quantity_in_stock if item.product else "N/A",
            ])

        return response


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
    """
    Delete a product and record any remaining stock in the audit trail.
    """
    if request.method != 'POST':
        messages.error(request, "Invalid request method.")
        return redirect('inventory_display')
    
    with transaction.atomic():
        product = get_object_or_404(Product.objects.select_for_update(), product_id=product_id)
        product_name = product.name
        remaining_stock = product.quantity_in_stock
        
        # ✅ FIXED: Record stock loss if any inventory remains
        if remaining_stock > 0:
            record_stock_change(
                product=product,
                qty=remaining_stock,
                change_type="deletion",  # ⚠️ Add 'deletion' to StockChange choices!
                note=f"Product deleted with {remaining_stock} units in stock"
            )
        
        # Delete the product
        product.delete()
    
    messages.success(request, f"Product '{product_name}' has been deleted.")

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
        # Delete all orders
        Order.objects.all().delete()

        # Reset the auto-increment sequence for order_id
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_get_serial_sequence('app_order', 'order_id');")
            sequence_name = cursor.fetchone()[0]
            if sequence_name:
                cursor.execute(f"ALTER SEQUENCE {sequence_name} RESTART WITH 1;")

        # ✅ FIXED: Clear session references to deleted orders
        if 'order_id' in request.session:
            request.session.pop('order_id')
        if 'cart' in request.session:
            request.session.pop('cart')
        request.session.modified = True

        messages.success(
            request, 
            "All orders have been deleted successfully and order IDs reset. Cart cleared."
        )
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
    if request.method != 'POST':
        return redirect('create_order')

    product = get_object_or_404(Product, product_id=product_id)

    expiry_input = request.POST.get('expiry_date', '').strip()
    taxable_input = request.POST.get('taxable')
    category_id = request.POST.get('category')

    # ─── Expiry date ─────────────────────────────
    if expiry_input:
        parsed_date = parse_date(expiry_input)
        if parsed_date:
            product.expiry_date = parsed_date
        else:
            messages.error(request, "Invalid expiry date format.")
            return redirect('create_order')
    else:
        product.expiry_date = None
    
    # ─── Taxable flag ────────────────────────────
    product.taxable = taxable_input == 'on'

    # ─── Category ────────────────────────────────
    if category_id:
        try:
            product.category = Category.objects.get(id=category_id)
        except Category.DoesNotExist:
            messages.error(request, "Selected category does not exist.")
            return redirect('create_order')
    else:
        product.category = None

    product.save()

    messages.success(request, f"Settings updated for {product.name}.")
    return redirect('create_order')