from decimal import Decimal
import os
import csv
import io
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from datetime import date, datetime, timedelta
import queue
from urllib import request
from dateutil.relativedelta import relativedelta
from urllib.parse import urlencode
from django.shortcuts import render, redirect, get_object_or_404
from django.template.loader import render_to_string
from django.urls import reverse
from django.views import View
from django.contrib import messages
from django.db import transaction, connection, IntegrityError
from django.db.models import Sum, Q, F, Avg, Count, Value, DecimalField, Case, When
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth, TruncDate, Coalesce
from django.conf import settings
from django.core.paginator import Paginator
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.utils.dateparse import parse_date
from django.utils.timezone import now
from django.utils.timesince import timesince
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView
from django.contrib.auth.forms import UserCreationForm
from app.mixins import AdminRequiredMixin
from .utils import recalculate_order_totals, get_product_stock_records, recommend_inventory_action, get_reorder_prediction, TAX_RATE
from .forms import EditProductForm, OrderDetailForm, BarcodeForm, ItemForm, AddProductForm
from .models import Item, Product, Category, Order, OrderDetail, RecentlyPurchasedProduct, StockChange, DeliveryCheckIn, LoginAudit, UserAction
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

def _get_client_ip(request):
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


class LabelPrintingView(LoginRequiredMixin, View):
    template_name = "label_printing.html"

    @staticmethod
    def _format_product(product):
        return {
            "product_id": product.product_id, # FIX 2: Added product_id to prevent URL crash
            "name": product.name,
            "brand": product.brand or "",
            "item_number": product.item_number or "",
            "barcode": product.barcode or "",
            "price": str(product.price),
            "qty": 1,
        }

    @staticmethod
    def _save_history(request, queue):
        history = request.session.get("label_queue_history", [])
        history.append(list(queue))
        if len(history) > 5:
            history.pop(0)
        request.session["label_queue_history"] = history

    def get(self, request):
        # Intercept AJAX requests from the Javascript Modal
        if request.headers.get('Accept') == 'application/json' and 'category_id' in request.GET:
            cat_id = request.GET.get('category_id')
            products = Product.objects.filter(category_id=cat_id, status=True).values(
                'product_id', 'name', 'barcode', 'price'
            ).order_by('name')
            return JsonResponse({'products': list(products)})

        session_queue = request.session.get("label_queue", [])
        category_items = Product.objects.filter(category__name__icontains="Print Label", status=True)
        query = request.GET.get("q", "").strip()
        search_results = Product.objects.filter(Q(name__icontains=query) | Q(barcode__icontains=query)).distinct()[:10] if query else []

        # Check if there is history to enable the Undo button
        history = request.session.get("label_queue_history", [])
        can_undo = len(history) > 0

        # FIX 1: Fetch all active products as a list of dictionaries for the JS Search
        all_products = list(Product.objects.filter(status=True).values(
            'product_id', 'name', 'barcode', 'item_number', 'price', 'quantity_in_stock'
        ))

        # Reversed for display: newest item at the top. Each entry is
        # (original_index, item) so the remove/qty forms use the correct index.
        session_queue_display = list(reversed(list(enumerate(session_queue))))

        return render(request, self.template_name, {
            "session_queue": session_queue,
            "session_queue_display": session_queue_display,
            "category_items": category_items,
            "query": query,
            "search_results": search_results,
            "categories": Category.objects.all().order_by('name'),
            "can_undo": can_undo,
            "all_products": all_products, # FIX 1: Send the list to the template
        })

    def post(self, request):
        queue = request.session.get("label_queue", [])
        history = request.session.get("label_queue_history", [])

        # 1. Handle the Undo Action first
        if "undo_action" in request.POST:
            if history:
                queue = history.pop()
                request.session["label_queue"] = queue
                request.session["label_queue_history"] = history
                messages.success(request, "Last action undone.")
            else:
                messages.warning(request, "Nothing to undo.")
            return redirect("label_printing")

        # 2. Snapshot logic for modifying actions
        mutating_actions = [
            "add_product", "add_category", "add_low_stock", 
            "quick_scan", "clear_queue", "remove_index", "add_selected_products"
        ]
        
        if any(action in request.POST for action in mutating_actions):
            history.append(list(queue))
            if len(history) > 5:
                history.pop(0)
            request.session["label_queue_history"] = history

        # --- Local format_product helper ---
        def format_product(p):
            return {
                "product_id": p.product_id, # FIX 2: Added product_id here too
                "name": p.name,
                "brand": p.brand or "",
                "item_number": p.item_number or "",
                "barcode": p.barcode or "",
                "price": str(p.price),
                "qty": 1
            }

        if "add_product" in request.POST:
            product = get_object_or_404(Product, pk=request.POST.get("product_id"))
            queue.append(format_product(product))
            messages.success(request, f"Added {product.name} to label queue.")
            
        elif "add_selected_products" in request.POST:
            product_ids = request.POST.getlist("selected_products")
            if product_ids:
                products = Product.objects.filter(product_id__in=product_ids, status=True)
                queue.extend([format_product(p) for p in products])
                messages.success(request, f"Added {products.count()} selected items to print queue.")

        elif "add_category" in request.POST:
            cat_id = request.POST.get("category_id")
            if cat_id:
                products = Product.objects.filter(category_id=cat_id, status=True)
                queue.extend([format_product(p) for p in products])
                messages.success(request, f"Added {products.count()} items from category.")
                
        elif "quick_scan" in request.POST:
            barcode = request.POST.get("barcode", "").strip()
            # Assuming find_product_by_barcode is imported or defined elsewhere in your file
            product = find_product_by_barcode(barcode) 
            if product:
                queue.append(format_product(product))
                messages.success(request, f"Scanned and added: {product.name}")
            else:
                history.pop() 
                request.session["label_queue_history"] = history
                messages.error(request, f"Barcode '{barcode}' not found.")

        elif "clear_queue" in request.POST:
            queue = []
            messages.info(request, "Label queue cleared.")

        elif "remove_index" in request.POST:
            idx = int(request.POST.get("remove_index"))
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

        # 3. Merge both into a single list, expanding qty for duplicates
        merged = []
        for p in category_items:
            merged.append({
                "name": p.name, "brand": p.brand or "",
                "item_number": p.item_number or "",
                "barcode": p.barcode or "", "price": str(p.price),
            })
        merged.extend(session_queue)

        # Expand qty: repeat each item qty times
        final_queue = []
        for item in merged:
            qty = max(1, int(item.get("qty", 1)))
            for _ in range(qty):
                final_queue.append(item)

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
        category_filter = request.GET.get('category', '')
        include_inactive = request.GET.get('include_inactive', '') == '1'

        # Base query: out-of-stock products
        if include_inactive:
            products_qs = Product.objects.filter(quantity_in_stock=0)
        else:
            products_qs = Product.objects.filter(status=True, quantity_in_stock=0)

        # Category filter
        if category_filter:
            products_qs = products_qs.filter(category_id=category_filter)

        products = list(
            products_qs.select_related('category').order_by("-stock_unfulfilled", "name")
        )

        total_missed = 0
        total_revenue_lost = Decimal('0.00')
        for p in products:
            p.revenue_lost = p.stock_unfulfilled * p.price
            total_missed += p.stock_unfulfilled
            total_revenue_lost += p.revenue_lost

        return render(request, self.template_name, {
            "products": products,
            "total_missed": total_missed,
            "total_revenue_lost": total_revenue_lost,
            "product_count": len(products),
            "categories": Category.objects.all().order_by('name'),
            "category_filter": category_filter,
            "include_inactive": include_inactive,
        })


class LowStockTrendView(AdminRequiredMixin, View):
    template_name = "low_stock_trend.html"

    def get(self, request):
        category_filter = request.GET.get('category', '')
        include_inactive = request.GET.get('include_inactive', '') == '1'

        # Base query: low stock products (1-3 units)
        if include_inactive:
            products_qs = Product.objects.filter(quantity_in_stock__gt=0, quantity_in_stock__lte=3)
        else:
            products_qs = Product.objects.filter(status=True, quantity_in_stock__gt=0, quantity_in_stock__lte=3)

        # Category filter
        if category_filter:
            products_qs = products_qs.filter(category_id=category_filter)

        products = list(
            products_qs.select_related('category').order_by("quantity_in_stock", "name")
        )

        today = date.today()
        critical_count = 0
        high_priority_count = 0

        for p in products:
            days_active = max((today - p.created_at.date()).days, 1) if p.created_at else 1
            avg_daily = p.stock_sold / days_active if p.stock_sold else 0
            if avg_daily > 0:
                p.days_remaining = round(p.quantity_in_stock / avg_daily, 1)
            else:
                p.days_remaining = None  # No sales history
            p.avg_daily_sales = round(avg_daily, 2)

            if p.days_remaining is not None and p.days_remaining < 3:
                p.priority = 'HIGH'
                high_priority_count += 1
            elif p.days_remaining is not None and p.days_remaining < 7:
                p.priority = 'MEDIUM'
            else:
                p.priority = 'LOW'

            if p.quantity_in_stock == 1:
                critical_count += 1

        # Sort by days_remaining (None last)
        products.sort(key=lambda p: (p.days_remaining is None, p.days_remaining or 9999))

        return render(request, self.template_name, {
            "products": products,
            "product_count": len(products),
            "critical_count": critical_count,
            "high_priority_count": high_priority_count,
            "categories": Category.objects.all().order_by('name'),
            "category_filter": category_filter,
            "include_inactive": include_inactive,
        })


def save_cart(request, cart):
    request.session["cart"] = cart
    request.session.modified = True

# Home view
@login_required
def home(request):
    today = date.today()
    week_ago = today - timedelta(days=7)

    # Stock health
    out_of_stock_count = Product.objects.filter(status=True, quantity_in_stock=0).count()
    low_stock_count = Product.objects.filter(status=True, quantity_in_stock__gt=0, quantity_in_stock__lte=3).count()
    expiring_soon_count = Product.objects.filter(
        expiry_date__gte=today, expiry_date__lte=today + timedelta(days=7)
    ).exclude(expiry_date__isnull=True).count()
    total_products = Product.objects.filter(status=True).count()

    # Today's orders & revenue
    orders_today = Order.objects.filter(order_date__date=today, submitted=True).count()
    revenue_today = OrderDetail.objects.filter(
        order__order_date__date=today, order__submitted=True
    ).aggregate(total=Sum(F('price') * F('quantity')))['total'] or Decimal('0.00')

    # Inventory value + cost
    inv_agg = Product.objects.filter(status=True).aggregate(
        total_units=Sum('quantity_in_stock'),
        total_retail=Sum(F('price') * F('quantity_in_stock')),
        total_cost=Sum(F('price_per_unit') * F('quantity_in_stock')),
    )
    total_retail = inv_agg['total_retail'] or Decimal('0.00')
    total_cost = inv_agg['total_cost'] or Decimal('0.00')
    gross_margin_pct = round(((total_retail - total_cost) / total_retail * 100), 1) if total_retail else 0

    # Top 5 best sellers (last 7 days)
    best_sellers = list(
        OrderDetail.objects.filter(
            order__submitted=True, order__order_date__date__gte=week_ago
        ).values('product_name', 'product_barcode').annotate(
            total_qty=Sum('quantity')
        ).order_by('-total_qty')[:5]
    )

    # Expiry countdown buckets
    exp_7d = Product.objects.filter(status=True, expiry_date__range=[today, today + timedelta(days=7)]).exclude(expiry_date__isnull=True).count()
    exp_14d = Product.objects.filter(status=True, expiry_date__range=[today + timedelta(days=8), today + timedelta(days=14)]).exclude(expiry_date__isnull=True).count()
    exp_30d = Product.objects.filter(status=True, expiry_date__range=[today + timedelta(days=15), today + timedelta(days=30)]).exclude(expiry_date__isnull=True).count()

    # Recent activity feed (last 10 stock changes)
    recent_activity = StockChange.objects.select_related('product').order_by('-timestamp')[:10]

    return render(request, 'home.html', {
        'out_of_stock_count': out_of_stock_count,
        'low_stock_count': low_stock_count,
        'expiring_soon_count': expiring_soon_count,
        'total_products': total_products,
        'orders_today': orders_today,
        'revenue_today': revenue_today,
        'total_units': inv_agg['total_units'] or 0,
        'total_retail': total_retail,
        'total_cost': total_cost,
        'gross_margin_pct': gross_margin_pct,
        'best_sellers': best_sellers,
        'exp_7d': exp_7d,
        'exp_14d': exp_14d,
        'exp_30d': exp_30d,
        'recent_activity': recent_activity,
    })

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
        if request.user.is_authenticated:
            if request.user.is_staff:
                return redirect('create_order')
            return redirect('checkin')
        return super().get(request, *args, **kwargs)

    def form_valid(self, form):
        response = super().form_valid(form)
        LoginAudit.objects.create(
            user=self.request.user,
            username=self.request.user.username,
            ip_address=_get_client_ip(self.request),
            success=True,
        )
        return response

    def form_invalid(self, form):
        username = self.request.POST.get('username', '')
        LoginAudit.objects.create(
            user=None,
            username=username,
            ip_address=_get_client_ip(self.request),
            success=False,
        )
        return super().form_invalid(form)

    def get_success_url(self):
        if self.request.user.is_staff:
            return reverse('create_order')
        return reverse('checkin')
   
# Display all orders - Transaction page.
class OrderView(AdminRequiredMixin, View):
    template_name = 'order_view.html'

    def get(self, request):
        today = date.today()
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')
        status_filter = request.GET.get('status', '')

        orders = Order.objects.annotate(
            calc_total=Sum(F('details__price') * F('details__quantity'))
        ).order_by('-order_id')

        # Apply filters
        if date_from:
            parsed = parse_date(date_from)
            if parsed:
                orders = orders.filter(order_date__date__gte=parsed)
        if date_to:
            parsed = parse_date(date_to)
            if parsed:
                orders = orders.filter(order_date__date__lte=parsed)
        if status_filter == 'completed':
            orders = orders.filter(submitted=True)
        elif status_filter == 'pending':
            orders = orders.filter(submitted=False)

        # Aggregates for KPI strip
        submitted_orders = orders.filter(submitted=True)
        agg = submitted_orders.aggregate(
            total_revenue=Sum('calc_total'),
            avg_order=Avg('calc_total'),
        )
        orders_today_count = submitted_orders.filter(order_date__date=today).count()

        # Daily sales data for chart
        daily_sales_qs = OrderDetail.objects.filter(order__submitted=True)
        if date_from:
            parsed_from = parse_date(date_from)
            if parsed_from:
                daily_sales_qs = daily_sales_qs.filter(order__order_date__date__gte=parsed_from)
        if date_to:
            parsed_to = parse_date(date_to)
            if parsed_to:
                daily_sales_qs = daily_sales_qs.filter(order__order_date__date__lte=parsed_to)

        daily_sales = list(
            daily_sales_qs
            .annotate(sale_date=TruncDate('order__order_date'))
            .values('sale_date')
            .annotate(
                daily_revenue=Sum(F('price') * F('quantity'), output_field=DecimalField()),
                order_count=Count('order', distinct=True),
                item_count=Count('od_id'),
            )
            .order_by('sale_date')
        )
        daily_chart_data = [
            {
                'date': d['sale_date'].strftime('%b %d') if d['sale_date'] else '',
                'full_date': d['sale_date'].strftime('%Y-%m-%d') if d['sale_date'] else '',
                'day': d['sale_date'].strftime('%A') if d['sale_date'] else '',
                'revenue': float(d['daily_revenue'] or 0),
                'orders': d['order_count'],
                'items': d['item_count'],
            }
            for d in daily_sales
        ]

        # Pagination
        paginator = Paginator(orders, 50)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        current_order_id = request.session.get('order_id')

        return render(request, self.template_name, {
            'page_obj': page_obj,
            'current_order_id': current_order_id,
            'total_orders': paginator.count,
            'total_revenue': agg['total_revenue'] or Decimal('0.00'),
            'avg_order': agg['avg_order'] or Decimal('0.00'),
            'orders_today': orders_today_count,
            'daily_chart_data': daily_chart_data,
            'date_from': date_from,
            'date_to': date_to,
            'status_filter': status_filter,
            'today': today,
        })
   
class OrderDetailView(View):
    template_name = 'order_detail.html'

    def get(self, request, order_id):
        order = get_object_or_404(Order, order_id=order_id)
        order_details = order.details.select_related('product', 'product__category').all()

        total_items = 0
        total_units = 0
        total_price_before_tax = Decimal("0.00")
        total_tax = Decimal("0.00")
        total_cost = Decimal("0.00")
        taxable_subtotal = Decimal("0.00")
        nontaxable_subtotal = Decimal("0.00")

        order_details_with_total = []
        for detail in order_details:
            # Use stored price (always available), product fields only if product still exists
            line_total = detail.price * detail.quantity
            product = detail.product  # May be None if product was deleted

            is_taxable = getattr(product, "taxable", False) if product else False
            item_tax = (line_total * TAX_RATE) if is_taxable else Decimal("0.00")

            if product and product.price_per_unit:
                cost = product.price_per_unit * detail.quantity
                profit = line_total - cost
            else:
                cost = Decimal("0.00")
                profit = None

            order_details_with_total.append({
                'detail': detail,
                'total_price': line_total,
                'is_taxable': is_taxable,
                'item_tax': item_tax,
                'line_with_tax': line_total + item_tax,
                'cost': cost,
                'profit': profit,
                'product_deleted': product is None,
            })

            total_items += 1
            total_units += detail.quantity
            total_price_before_tax += line_total
            total_tax += item_tax
            total_cost += cost
            if is_taxable:
                taxable_subtotal += line_total
            else:
                nontaxable_subtotal += line_total

        total_price_after_tax = total_price_before_tax + total_tax
        total_profit = total_price_before_tax - total_cost if total_cost > 0 else None
        margin_pct = ((total_profit / total_price_before_tax) * 100) if total_profit and total_price_before_tax > 0 else None

        # Navigation: previous and next order IDs
        prev_order = Order.objects.filter(order_id__lt=order_id).order_by('-order_id').values_list('order_id', flat=True).first()
        next_order = Order.objects.filter(order_id__gt=order_id).order_by('order_id').values_list('order_id', flat=True).first()

        return render(request, self.template_name, {
            'order': order,
            'order_details_with_total': order_details_with_total,
            'total_price_before_tax': total_price_before_tax,
            'total_price_after_tax': total_price_after_tax,
            'total_tax': total_tax,
            'total_items': total_items,
            'total_units': total_units,
            'taxable_subtotal': taxable_subtotal,
            'nontaxable_subtotal': nontaxable_subtotal,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'margin_pct': margin_pct,
            'prev_order': prev_order,
            'next_order': next_order,
        })


class OrderReceiptPDFView(LoginRequiredMixin, View):
    """Generate a professional PDF transaction report for a single order."""

    def get(self, request, order_id):
        from reportlab.lib.colors import HexColor

        order = get_object_or_404(Order, order_id=order_id)
        details = order.details.select_related('product', 'product__category').all()

        # ── Compute line items & totals (same logic as OrderDetailView) ──
        items = []
        total_units = 0
        subtotal = Decimal("0.00")
        total_tax = Decimal("0.00")

        for d in details:
            line_total = d.price * d.quantity
            product = d.product
            is_taxable = getattr(product, "taxable", False) if product else False
            item_tax = (line_total * TAX_RATE) if is_taxable else Decimal("0.00")
            items.append({
                'name': d.display_name,
                'barcode': d.display_barcode,
                'qty': d.quantity,
                'price': d.price,
                'line_total': line_total,
                'taxable': is_taxable,
                'item_tax': item_tax,
            })
            total_units += d.quantity
            subtotal += line_total
            total_tax += item_tax

        grand_total = subtotal + total_tax

        # ── PDF setup ──
        buffer = io.BytesIO()
        PAGE_W, PAGE_H = letter
        c = canvas.Canvas(buffer, pagesize=letter)
        MARGIN = 54  # 0.75 inch

        # Colours
        brand_color = HexColor("#4f46e5")
        dark = HexColor("#1e293b")
        muted = HexColor("#64748b")
        line_color = HexColor("#e2e8f0")
        row_alt = HexColor("#f8fafc")
        white = HexColor("#ffffff")

        y = PAGE_H - MARGIN  # cursor starts at top

        # ── Helper: horizontal rule ──
        def hr(y_pos, color=line_color):
            c.setStrokeColor(color)
            c.setLineWidth(0.5)
            c.line(MARGIN, y_pos, PAGE_W - MARGIN, y_pos)
            return y_pos

        # ────────────────────────────────────────
        # HEADER
        # ────────────────────────────────────────
        c.setFillColor(brand_color)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(MARGIN, y, "MPCP")
        c.setFillColor(muted)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN, y - 16, "Meadowvale Professional Center Pharmacy")

        # Right side – report title + order info
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 14)
        c.drawRightString(PAGE_W - MARGIN, y, "TRANSACTION REPORT")
        c.setFont("Helvetica", 10)
        c.setFillColor(muted)
        c.drawRightString(PAGE_W - MARGIN, y - 18, f"Order #{order.order_id}")
        c.drawRightString(PAGE_W - MARGIN, y - 32, order.order_date.strftime("%B %d, %Y  %I:%M %p"))
        status_text = "Completed" if order.submitted else "Pending"
        c.drawRightString(PAGE_W - MARGIN, y - 46, f"Status: {status_text}")

        y -= 62
        hr(y)
        y -= 18

        # ────────────────────────────────────────
        # ITEMS TABLE
        # ────────────────────────────────────────
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(MARGIN, y, "Order Items")
        y -= 20

        # Column positions
        col_num = MARGIN
        col_product = MARGIN + 28
        col_qty = 340
        col_price = 405
        col_tax = 468
        col_total = PAGE_W - MARGIN

        # Table header
        row_h = 18
        c.setFillColor(HexColor("#f1f5f9"))
        c.rect(MARGIN, y - 4, PAGE_W - 2 * MARGIN, row_h, fill=1, stroke=0)
        c.setFillColor(muted)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(col_num, y + 2, "#")
        c.drawString(col_product, y + 2, "PRODUCT")
        c.drawRightString(col_qty, y + 2, "QTY")
        c.drawRightString(col_price, y + 2, "PRICE")
        c.drawCentredString(col_tax + 5, y + 2, "TAX")
        c.drawRightString(col_total, y + 2, "TOTAL")
        y -= row_h + 4

        # Table rows
        c.setFont("Helvetica", 9)
        for idx, item in enumerate(items, 1):
            # Page break check
            if y < MARGIN + 120:
                c.showPage()
                y = PAGE_H - MARGIN

            # Alternating row background
            if idx % 2 == 0:
                c.setFillColor(row_alt)
                c.rect(MARGIN, y - 4, PAGE_W - 2 * MARGIN, row_h, fill=1, stroke=0)

            c.setFillColor(dark)
            c.setFont("Helvetica", 9)
            c.drawString(col_num, y + 2, str(idx))

            # Truncate long product names
            name = item['name']
            max_name_w = col_qty - col_product - 20
            if stringWidth(name, "Helvetica", 9) > max_name_w:
                while stringWidth(name + "...", "Helvetica", 9) > max_name_w and len(name) > 1:
                    name = name[:-1]
                name += "..."
            c.drawString(col_product, y + 2, name)

            c.drawRightString(col_qty, y + 2, str(item['qty']))
            c.drawRightString(col_price, y + 2, f"${item['price']:.2f}")

            c.setFont("Helvetica", 7)
            tax_label = "TAX" if item['taxable'] else "--"
            c.setFillColor(muted)
            c.drawCentredString(col_tax + 5, y + 2, tax_label)

            c.setFillColor(dark)
            c.setFont("Helvetica-Bold", 9)
            c.drawRightString(col_total, y + 2, f"${item['line_total']:.2f}")

            y -= row_h

        # Table bottom rule
        hr(y)
        y -= 24

        # ────────────────────────────────────────
        # FINANCIAL SUMMARY
        # ────────────────────────────────────────
        summary_x_label = PAGE_W - MARGIN - 170
        summary_x_val = PAGE_W - MARGIN

        def summary_row(label, value, bold=False, color=dark, size=10):
            nonlocal y
            font = "Helvetica-Bold" if bold else "Helvetica"
            c.setFont(font, size)
            c.setFillColor(muted)
            c.drawString(summary_x_label, y, label)
            c.setFillColor(color)
            c.drawRightString(summary_x_val, y, value)
            y -= 18

        summary_row("Subtotal", f"${subtotal:.2f}")
        summary_row("Tax (13%)", f"${total_tax:.2f}")

        # Divider before grand total
        c.setStrokeColor(dark)
        c.setLineWidth(1)
        c.line(summary_x_label, y + 8, summary_x_val, y + 8)
        y -= 4

        summary_row("TOTAL", f"${grand_total:.2f}", bold=True, color=brand_color, size=14)

        y -= 12

        # ────────────────────────────────────────
        # FOOTER
        # ────────────────────────────────────────
        hr(y)
        y -= 16

        c.setFont("Helvetica", 8)
        c.setFillColor(muted)
        c.drawString(MARGIN, y, f"Items: {len(items)}  |  Units: {total_units}")
        c.drawRightString(PAGE_W - MARGIN, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        y -= 24

        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(brand_color)
        c.drawCentredString(PAGE_W / 2, y, "MPCP  ·  Meadowvale Professional Center Pharmacy")
        c.setFont("Helvetica", 8)
        c.setFillColor(muted)
        c.drawCentredString(PAGE_W / 2, y - 14, "Thank you for your business")

        c.save()
        buffer.seek(0)

        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="MPCP-Order-{order_id}.pdf"'
        return response


class SalesAnalyticsView(AdminRequiredMixin, View):
    template_name = 'sales_analytics.html'

    def get(self, request):
        # ── Date range + granularity ───────────────────────────────────────
        today = date.today()
        default_start = (today - relativedelta(months=12)).replace(day=1)
        try:
            start_date = parse_date(request.GET.get('start', '')) or default_start
        except (ValueError, TypeError):
            start_date = default_start
        try:
            end_date = parse_date(request.GET.get('end', '')) or today
        except (ValueError, TypeError):
            end_date = today
        gran = request.GET.get('gran', 'month')
        if gran not in ('day', 'week', 'month'):
            gran = 'month'

        # ── Base queryset (submitted orders only) ─────────────────────────
        base_qs = OrderDetail.objects.filter(
            order__submitted=True,
            order__order_date__date__range=[start_date, end_date],
        )

        # Cost expression: price_per_unit × qty when set, else 0
        cost_expr = Case(
            When(
                product__price_per_unit__isnull=False,
                then=F('product__price_per_unit') * F('quantity'),
            ),
            default=Value(Decimal('0')),
            output_field=DecimalField(max_digits=12, decimal_places=2),
        )

        # ── KPI aggregates ─────────────────────────────────────────────────
        kpi_raw = base_qs.aggregate(
            total_revenue=Sum(F('price') * F('quantity'), output_field=DecimalField(max_digits=12, decimal_places=2)),
            total_orders=Count('order', distinct=True),
            total_items=Sum('quantity'),
            total_cost=Sum(cost_expr),
        )
        total_revenue = float(kpi_raw['total_revenue'] or 0)
        total_cost    = float(kpi_raw['total_cost']    or 0)
        total_profit  = total_revenue - total_cost
        total_orders  = kpi_raw['total_orders'] or 0
        total_items   = kpi_raw['total_items']  or 0
        avg_order     = total_revenue / total_orders if total_orders else 0
        margin_pct    = (total_profit / total_revenue * 100) if total_revenue else 0
        has_cost_data = total_cost > 0

        # ── Revenue series ─────────────────────────────────────────────────
        trunc_map = {'day': TruncDay, 'week': TruncWeek, 'month': TruncMonth}
        TruncFn = trunc_map.get(gran, TruncMonth)
        label_fmt = {'day': '%d %b %Y', 'week': '%d %b %Y', 'month': '%b %Y'}[gran]

        revenue_series = [
            {
                'label':   r['period'].strftime(label_fmt),
                'revenue': float(r['revenue'] or 0),
                'cost':    float(r['cost']    or 0),
                'profit':  float(r['revenue'] or 0) - float(r['cost'] or 0),
                'orders':  r['orders'],
            }
            for r in (
                base_qs
                .annotate(period=TruncFn('order__order_date'))
                .values('period')
                .annotate(
                    revenue=Sum(F('price') * F('quantity'), output_field=DecimalField(max_digits=12, decimal_places=2)),
                    cost=Sum(cost_expr),
                    orders=Count('order', distinct=True),
                )
                .order_by('period')
            )
        ]

        # ── Top 15 products by revenue ─────────────────────────────────────
        top_products = [
            {
                'name':    p['product_name'],
                'revenue': float(p['revenue'] or 0),
                'units':   p['units'],
                'cost':    float(p['cost']    or 0),
                'profit':  float(p['revenue'] or 0) - float(p['cost'] or 0),
            }
            for p in (
                base_qs
                .values('product_name')
                .annotate(
                    revenue=Sum(F('price') * F('quantity'), output_field=DecimalField(max_digits=12, decimal_places=2)),
                    units=Sum('quantity'),
                    cost=Sum(cost_expr),
                )
                .order_by('-revenue')[:15]
            )
        ]

        # ── Category sales + margins ───────────────────────────────────────
        category_sales = [
            {
                'name':    c['cat_name'],
                'revenue': float(c['revenue'] or 0),
                'units':   c['units'],
                'cost':    float(c['cost']    or 0),
                'profit':  float(c['revenue'] or 0) - float(c['cost'] or 0),
            }
            for c in (
                base_qs
                .values(cat_name=Coalesce('product__category__name', Value('Uncategorised')))
                .annotate(
                    revenue=Sum(F('price') * F('quantity'), output_field=DecimalField(max_digits=12, decimal_places=2)),
                    units=Sum('quantity'),
                    cost=Sum(cost_expr),
                )
                .order_by('-revenue')
            )
        ]

        # ── Top 5 products within each category ───────────────────────────
        top_by_cat = {}
        for row in (
            base_qs
            .values(
                cat_name=Coalesce('product__category__name', Value('Uncategorised')),
                prod=F('product_name'),
            )
            .annotate(
                revenue=Sum(F('price') * F('quantity'), output_field=DecimalField(max_digits=12, decimal_places=2)),
                units=Sum('quantity'),
                cost=Sum(cost_expr),
            )
            .order_by('cat_name', '-revenue')
        ):
            cat = row['cat_name']
            if cat not in top_by_cat:
                top_by_cat[cat] = []
            if len(top_by_cat[cat]) < 5:
                top_by_cat[cat].append({
                    'name':    row['prod'],
                    'revenue': float(row['revenue'] or 0),
                    'units':   row['units'],
                    'cost':    float(row['cost']    or 0),
                    'profit':  float(row['revenue'] or 0) - float(row['cost'] or 0),
                })

        return render(request, self.template_name, {
            'kpi': {
                'revenue':    round(total_revenue, 2),
                'orders':     total_orders,
                'avg_order':  round(avg_order, 2),
                'profit':     round(total_profit, 2),
                'items':      total_items,
                'margin_pct': round(margin_pct, 1),
            },
            'revenue_series': revenue_series,
            'top_products':   top_products,
            'category_sales': category_sales,
            'top_by_cat':     top_by_cat,
            'has_cost_data':  has_cost_data,
            'start_date':     start_date.isoformat(),
            'end_date':       end_date.isoformat(),
            'gran':           gran,
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

        order = Order.objects.create(total_price=Decimal("0.00"), user=request.user)
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
            
            qty = int(line["quantity"])
            if qty > product.quantity_in_stock:
                messages.info(
                    request,
                    f"'{product.name}' quantity ({qty}) exceeds stock ({product.quantity_in_stock}).",
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

        total_price_after_tax = total_price_before_tax * (1 + TAX_RATE)
        tax_amount = total_price_after_tax - total_price_before_tax

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
            "tax_amount": tax_amount,
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

            cart[pid]["quantity"] = desired_qty
            request.session.modified = True

        # ── Messages (outside transaction) ────────────────────────────────
        override_notes = []
        if override_inactive: override_notes.append("product activated")
        if override_expiry:   override_notes.append("expired override")

        if stock <= 0:
            messages.info(request,
                f"Added '{product.name}' (0 in stock).",
                extra_tags="order")
        elif desired_qty > stock:
            messages.warning(request,
                f"'{product.name}' quantity ({desired_qty}) exceeds stock ({stock}).",
                extra_tags="order")
        elif override_notes:
            messages.warning(request,
                f"⚠️ Added '{product.name}' ({', '.join(override_notes)}).",
                extra_tags="order")
        else:
            messages.success(request,
                f"Added {requested_quantity} unit(s) of '{product.name}'. (Now {desired_qty}/{stock})",
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

                # ✅ Create order line with full requested quantity
                # Store product name/barcode so order history survives product deletion
                OrderDetail.objects.create(
                    order=order,
                    product=product,
                    product_name=product.name,
                    product_barcode=product.barcode or "",
                    quantity=requested,
                    price=product.price,
                )

                # ✅ Decrement stock (floor at 0 — never go negative)
                if requested > 0:
                    deduct = min(requested, available)
                    if deduct > 0:
                        product.quantity_in_stock = available - deduct
                        product.save(update_fields=["quantity_in_stock"])

                    record_stock_change(
                        product=product,
                        qty=requested,
                        change_type="checkout",
                        note=f"Order {order.order_id} submission",
                        user=request.user,
                    )

                if requested > available:
                    unfulfilled_lines.append(f"{product.name} (short {requested - available})")

                # Optional analytics
                if requested > 0:
                    rp, _ = RecentlyPurchasedProduct.objects.get_or_create(product=product)
                    rp.quantity = (rp.quantity or 0) + requested
                    rp.save(update_fields=["quantity"])

            # ✅ Finalize order
            order.submitted = True
            order.save(update_fields=["submitted"])

            UserAction.objects.create(
                user=request.user, action='submit_order',
                target=f'Order #{order.order_id}',
            )

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

        return redirect("order_success", order_id=order.order_id)


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
class OrderSuccessView(LoginRequiredMixin, View):
    template_name = 'order_success.html'

    def get(self, request, order_id):
        order = get_object_or_404(Order, order_id=order_id)
        details = order.details.select_related('product').all()
        items = []
        subtotal = Decimal('0.00')
        for d in details:
            line = d.price * d.quantity
            subtotal += line
            items.append({'name': d.display_name, 'qty': d.quantity, 'price': d.price, 'total': line})

        total_tax = Decimal('0.00')
        for d in details:
            if d.product and getattr(d.product, 'taxable', False):
                total_tax += d.price * d.quantity * TAX_RATE

        return render(request, self.template_name, {
            'order': order,
            'items': items,
            'subtotal': subtotal,
            'total_tax': total_tax,
            'grand_total': subtotal + total_tax,
            'item_count': details.count(),
        })
#Change - Function to annotate changes

def record_stock_change(
    product: Product,
    qty: int,
    change_type: str,
    note: str = "",
    user=None,
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
            user=user,
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
            update_fields=["stock_bought", "stock_sold", "stock_expired", "stock_unfulfilled"]
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
                note="1 unit removed via UI",
                user=request.user,
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
            note="Manual add via UI",
            user=request.user,
        )

    # Redirect preserving inventory_mode
    return redirect(
        f"{reverse('checkin')}?barcode={product.barcode}&inventory_mode={str(inventory_mode).lower()}"
    )

# add products without barcode (triggered via Search/Autocomplete)
class AddProductByIdCheckinView(LoginRequiredMixin, View):
    def post(self, request, product_id):
        inventory_mode = request.POST.get("inventory_mode") == "true"

        try:
            product = Product.objects.get(product_id=product_id)
        except Product.DoesNotExist:
            messages.error(request, "Product not found.", extra_tags="checkin error")
            return redirect("checkin")

        params = {'barcode': product.barcode}
        if inventory_mode:
            params['inventory_mode'] = 'true'
        return redirect(f"{reverse('checkin')}?{urlencode(params)}")

#checkin views
class CheckinProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"

    def get(self, request):
        # ── AJAX Recent Scans API ──
        if request.GET.get('format') == 'recent_scans' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            scans_qs = StockChange.objects.filter(
                change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract']
            ).select_related('product', 'product__category').order_by('-timestamp')[:25]
            today = date.today()
            today_scans = StockChange.objects.filter(
                change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract'],
                timestamp__date=today
            )
            entries = []
            for sc in scans_qs:
                entries.append({
                    'time': sc.timestamp.strftime('%b %d %H:%M'),
                    'time_ago': timesince(sc.timestamp),
                    'name': sc.product.name if sc.product else 'Deleted',
                    'barcode': sc.product.barcode if sc.product else '',
                    'qty': sc.quantity,
                    'positive': sc.quantity > 0,
                    'stock': sc.product.quantity_in_stock if sc.product else 0,
                    'action': sc.get_change_type_display(),
                })
            return JsonResponse({
                'entries': entries,
                'scanned_today': today_scans.filter(change_type='checkin').count(),
                'products_updated': today_scans.values('product').distinct().count(),
            })

        # ── AJAX Stock Log API ──
        if request.GET.get('format') == 'json' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            log_qs = StockChange.objects.select_related('product').order_by('-timestamp')
            log_product = request.GET.get('log_product', '').strip()
            log_type = request.GET.get('log_type', '')
            log_date_from = request.GET.get('log_date_from', '')
            log_date_to = request.GET.get('log_date_to', '')
            if log_product:
                log_qs = log_qs.filter(Q(product__name__icontains=log_product) | Q(product__barcode__icontains=log_product))
            if log_type:
                log_qs = log_qs.filter(change_type=log_type)
            if log_date_from:
                parsed = parse_date(log_date_from)
                if parsed:
                    log_qs = log_qs.filter(timestamp__date__gte=parsed)
            if log_date_to:
                parsed = parse_date(log_date_to)
                if parsed:
                    log_qs = log_qs.filter(timestamp__date__lte=parsed)
            # CSV export
            if request.GET.get('export') == 'csv':
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="stock_log_{now().strftime("%Y%m%d_%H%M")}.csv"'
                writer = csv.writer(response)
                writer.writerow(['Timestamp', 'Product', 'Barcode', 'Action', 'Quantity', 'Note'])
                for sc in log_qs[:2000]:
                    writer.writerow([sc.timestamp.strftime('%Y-%m-%d %H:%M'), sc.product.name if sc.product else 'Deleted', sc.product.barcode if sc.product else '', sc.get_change_type_display(), sc.quantity, sc.note or ''])
                return response
            # Paginate
            paginator = Paginator(log_qs, 50)
            page = paginator.get_page(request.GET.get('log_page', 1))
            today = date.today()
            today_all = StockChange.objects.filter(timestamp__date=today)
            entries = []
            for sc in page:
                positive = sc.change_type in ('checkin', 'error_add', 'return')
                badge_cls = 'checkin' if sc.change_type == 'checkin' else 'checkout' if sc.change_type == 'checkout' else 'expired' if sc.change_type == 'expired' else 'error' if sc.change_type in ('error_add', 'error_subtract') else 'other'
                entries.append({
                    'time': sc.timestamp.strftime('%b %d %H:%M'),
                    'name': sc.product.name if sc.product else 'Deleted',
                    'barcode': sc.product.barcode if sc.product else '',
                    'action': sc.get_change_type_display(),
                    'badge_cls': badge_cls,
                    'qty': sc.quantity,
                    'positive': positive,
                    'note': sc.note or '—',
                })
            return JsonResponse({
                'entries': entries,
                'page': page.number,
                'num_pages': paginator.num_pages,
                'has_prev': page.has_previous(),
                'has_next': page.has_next(),
                'kpi': {
                    'checkins': today_all.filter(change_type='checkin').count(),
                    'sales': today_all.filter(change_type='checkout').count(),
                    'adjustments': today_all.filter(change_type__in=['error_add', 'error_subtract']).count(),
                },
            })

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

        # Last checkin timestamp for this product
        last_checkin = None
        if product:
            last_checkin = StockChange.objects.filter(
                product=product, change_type='checkin'
            ).order_by('-timestamp').first()

        # Recent scan history (last 25 check-in actions)
        recent_scans = StockChange.objects.filter(
            change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract']
        ).select_related('product', 'product__category').order_by('-timestamp')[:25]

        # Today's check-in stats
        today = date.today()
        today_scans = StockChange.objects.filter(
            change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract'],
            timestamp__date=today
        )
        scanned_today_count = today_scans.filter(change_type='checkin').count()
        products_updated_today = today_scans.values('product').distinct().count()

        # ── Stock Movement Log (merged from StockLogView) ──
        log_qs = StockChange.objects.select_related('product').order_by('-timestamp')
        log_product = request.GET.get('log_product', '').strip()
        log_type = request.GET.get('log_type', '')
        log_date_from = request.GET.get('log_date_from', '')
        log_date_to = request.GET.get('log_date_to', '')

        if log_product:
            log_qs = log_qs.filter(Q(product__name__icontains=log_product) | Q(product__barcode__icontains=log_product))
        if log_type:
            log_qs = log_qs.filter(change_type=log_type)
        if log_date_from:
            parsed = parse_date(log_date_from)
            if parsed:
                log_qs = log_qs.filter(timestamp__date__gte=parsed)
        if log_date_to:
            parsed = parse_date(log_date_to)
            if parsed:
                log_qs = log_qs.filter(timestamp__date__lte=parsed)

        # CSV export
        if request.GET.get('export') == 'csv':
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="stock_log_{now().strftime("%Y%m%d_%H%M")}.csv"'
            writer = csv.writer(response)
            writer.writerow(['Timestamp', 'Product', 'Barcode', 'Action', 'Quantity', 'Note'])
            for sc in log_qs[:2000]:
                writer.writerow([
                    sc.timestamp.strftime('%Y-%m-%d %H:%M'),
                    sc.product.name if sc.product else 'Deleted',
                    sc.product.barcode if sc.product else '',
                    sc.get_change_type_display(),
                    sc.quantity,
                    sc.note or '',
                ])
            return response

        log_paginator = Paginator(log_qs, 50)
        stock_log_page = log_paginator.get_page(request.GET.get('log_page', 1))
        change_types = StockChange._meta.get_field('change_type').choices

        today_all = StockChange.objects.filter(timestamp__date=today)
        checkins_today = today_all.filter(change_type='checkin').count()
        sales_today = today_all.filter(change_type='checkout').count()
        adjustments_today = today_all.filter(change_type__in=['error_add', 'error_subtract']).count()

        return render(request, self.template_name, {
            "search_results": search_results,
            "inventory_mode": inventory_mode,
            "all_products": list(
                Product.objects.values(
                    "product_id", "name", "price", "quantity_in_stock",
                    "item_number", "barcode"
                )
            ),
            "product": product,
            "edit_form": edit_form,
            "categories": Category.objects.all(),
            "recent_scans": recent_scans,
            "scanned_today_count": scanned_today_count,
            "products_updated_today": products_updated_today,
            "last_checkin": last_checkin,
            # Stock log context
            "stock_log_page": stock_log_page,
            "log_product": log_product,
            "log_type_filter": log_type,
            "log_date_from": log_date_from,
            "log_date_to": log_date_to,
            "change_types": change_types,
            "log_checkins_today": checkins_today,
            "log_sales_today": sales_today,
            "log_adjustments_today": adjustments_today,
        })

    def post(self, request):
        # ── Cycle Count Submit (from slider panel) ──
        if 'cc_submit' in request.POST:
            counted = 0
            discrepancies = 0
            net_adjustment = 0
            details = []
            with transaction.atomic():
                for key, value in request.POST.items():
                    if not key.startswith('count_'):
                        continue
                    try:
                        product_id = int(key.replace('count_', ''))
                        actual_count = int(value)
                    except (ValueError, TypeError):
                        continue
                    product = Product.objects.select_for_update().filter(pk=product_id).first()
                    if not product:
                        continue
                    counted += 1
                    diff = actual_count - product.quantity_in_stock
                    if diff != 0:
                        discrepancies += 1
                        net_adjustment += diff
                        old_qty = product.quantity_in_stock
                        product.quantity_in_stock = actual_count
                        product.save(update_fields=['quantity_in_stock'])
                        change_type = 'error_add' if diff > 0 else 'error_subtract'
                        record_stock_change(
                            product, qty=abs(diff), change_type=change_type,
                            note=f"Cycle count: {old_qty} → {actual_count}",
                            user=request.user,
                        )
                        details.append({'name': product.name, 'old': old_qty, 'new': actual_count, 'diff': diff})
            if counted > 0:
                msg = f"Cycle count complete: {counted} counted, {discrepancies} discrepancies"
                if discrepancies > 0:
                    msg += f", net adjustment: {net_adjustment:+d}"
                messages.success(request, msg, extra_tags="checkin success")
            else:
                messages.warning(request, "No products were counted.", extra_tags="checkin warning")
            params = {}
            cc_cat = request.POST.get('cc_category', '')
            if cc_cat:
                params['cc_category'] = cc_cat
            inv_mode = request.POST.get('inventory_mode', 'false')
            if inv_mode == 'true':
                params['inventory_mode'] = 'true'
            url = reverse('checkin')
            if params:
                url += '?' + urlencode(params)
            return redirect(url)

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
        product = find_product_by_barcode(barcode)

        if product:
            params = {'barcode': product.barcode}
            if inventory_mode:
                params['inventory_mode'] = 'true'
            return redirect(f"{reverse('checkin')}?{urlencode(params)}")

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
                        note="Product updated via check-in inline edit",
                        user=request.user,
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
                    note="Product updated via edit form",
                    user=request.user,
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
                        note="New product added via form",
                        user=request.user,
                    )

                UserAction.objects.create(
                    user=request.user, action='add_product',
                    target=product.name,
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
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

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

        # For AJAX live search return all matching rows; otherwise paginate normally
        page_size = 10000 if is_ajax else 100
        paginator = Paginator(products, page_size)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        # AJAX early return — just the table rows as an HTML fragment
        if is_ajax:
            rows_html = render_to_string(
                'partials/inv_rows.html',
                {'page_obj': page_obj},
                request=request,
            )
            return JsonResponse({'html': rows_html, 'count': paginator.count})

        # Aggregate stats for the filtered queryset
        stats = products.aggregate(
            total_units=Sum('quantity_in_stock'),
            total_retail=Sum(F('price') * F('quantity_in_stock')),
            total_cost=Sum(F('price_per_unit') * F('quantity_in_stock')),
        )

        # Pass all query parameters and the paginator to the template
        return render(request, self.template_name, {
            'page_obj': page_obj,
            'categories': Category.objects.all(),
            'selected_category_id': selected_category_id,
            'barcode_query': barcode_query,
            'name_query': name_query,
            'sort_column': sort_column,
            'sort_direction': sort_direction,
            'total_products': paginator.count,
            'total_units': stats['total_units'] or 0,
            'total_retail': stats['total_retail'] or Decimal('0.00'),
            'total_cost': stats['total_cost'] or Decimal('0.00'),
        })

class ExportInventoryCSVView(LoginRequiredMixin, View):
    def get(self, request):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="inventory_{now().strftime("%Y%m%d_%H%M")}.csv"'

        writer = csv.writer(response)
        writer.writerow(['Name', 'Barcode', 'SKU', 'Category', 'Price', 'Cost', 'Qty In Stock', 'Status', 'Expiry Date'])

        products = Product.objects.select_related('category').all()

        # Apply same filters as inventory page
        category_id = request.GET.get('category_id', '')
        name_query = request.GET.get('name_query', '')
        if category_id:
            products = products.filter(category_id=category_id)
        if name_query:
            products = products.filter(name__icontains=name_query)

        products = products.order_by('name')

        for p in products:
            writer.writerow([
                p.name,
                p.barcode or '',
                p.item_number or '',
                p.category.name if p.category else '',
                p.price,
                p.price_per_unit or '',
                p.quantity_in_stock,
                'Active' if p.status else 'Inactive',
                p.expiry_date.strftime('%Y-%m-%d') if p.expiry_date else '',
            ])

        return response


class ExportTransactionsCSVView(AdminRequiredMixin, View):
    def get(self, request):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="transactions_{now().strftime("%Y%m%d_%H%M")}.csv"'

        writer = csv.writer(response)
        writer.writerow(['Order ID', 'Date', 'Status', 'Product Name', 'Barcode', 'Quantity', 'Unit Price', 'Line Total'])

        details = OrderDetail.objects.select_related('order', 'product').order_by('-order__order_date')

        # Apply same filters as OrderView
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')
        status_filter = request.GET.get('status', '')

        if date_from:
            parsed = parse_date(date_from)
            if parsed:
                details = details.filter(order__order_date__date__gte=parsed)
        if date_to:
            parsed = parse_date(date_to)
            if parsed:
                details = details.filter(order__order_date__date__lte=parsed)
        if status_filter == 'completed':
            details = details.filter(order__submitted=True)
        elif status_filter == 'pending':
            details = details.filter(order__submitted=False)
        else:
            details = details.filter(order__submitted=True)  # Default: completed only

        for d in details:
            product_name = d.product.name if d.product else d.product_name
            barcode = d.product.barcode if d.product else d.product_barcode
            line_total = d.price * d.quantity
            writer.writerow([
                d.order.order_id,
                d.order.order_date.strftime('%Y-%m-%d %H:%M'),
                'Completed' if d.order.submitted else 'Pending',
                product_name,
                barcode or '',
                d.quantity,
                f'{d.price:.2f}',
                f'{line_total:.2f}',
            ])

        return response


# ========== NEW FEATURE VIEWS ==========

class GlobalSearchAPIView(LoginRequiredMixin, View):
    """AJAX endpoint for global nav search."""
    def get(self, request):
        q = request.GET.get('q', '').strip()
        if len(q) < 2:
            return JsonResponse({'results': []})
        products = Product.objects.filter(
            Q(name__icontains=q) | Q(barcode__icontains=q) | Q(item_number__icontains=q)
        ).values('product_id', 'name', 'barcode', 'quantity_in_stock', 'status')[:6]
        return JsonResponse({'results': list(products)})


class AlertBannerAPIView(LoginRequiredMixin, View):
    """Returns urgent alerts for the site-wide banner."""
    def get(self, request):
        today = date.today()
        alerts = []
        expiring = Product.objects.filter(
            status=True, expiry_date__range=[today, today + timedelta(days=7)]
        ).exclude(expiry_date__isnull=True).count()
        if expiring:
            alerts.append({'type': 'warning', 'text': f'{expiring} expiring this week', 'url': '/expired-products/?date_filter=1_week'})
        return JsonResponse({'alerts': alerts})


class StockLogView(AdminRequiredMixin, View):
    """Full audit trail of all stock movements."""
    template_name = 'stock_log.html'

    def get(self, request):
        qs = StockChange.objects.select_related('product').order_by('-timestamp')

        # Filters
        product_query = request.GET.get('product', '').strip()
        change_type = request.GET.get('type', '')
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')

        if product_query:
            qs = qs.filter(Q(product__name__icontains=product_query) | Q(product__barcode__icontains=product_query))
        if change_type:
            qs = qs.filter(change_type=change_type)
        if date_from:
            parsed = parse_date(date_from)
            if parsed:
                qs = qs.filter(timestamp__date__gte=parsed)
        if date_to:
            parsed = parse_date(date_to)
            if parsed:
                qs = qs.filter(timestamp__date__lte=parsed)

        # CSV export
        if request.GET.get('export') == 'csv':
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="stock_log_{now().strftime("%Y%m%d_%H%M")}.csv"'
            writer = csv.writer(response)
            writer.writerow(['Timestamp', 'Product', 'Barcode', 'Action', 'Quantity', 'Note'])
            for sc in qs[:2000]:
                writer.writerow([
                    sc.timestamp.strftime('%Y-%m-%d %H:%M'),
                    sc.product.name if sc.product else 'Deleted',
                    sc.product.barcode if sc.product else '',
                    sc.get_change_type_display(),
                    sc.quantity,
                    sc.note or '',
                ])
            return response

        # Today's stats
        today = date.today()
        today_changes = StockChange.objects.filter(timestamp__date=today)
        checkins_today = today_changes.filter(change_type='checkin').count()
        sales_today = today_changes.filter(change_type='checkout').count()
        adjustments_today = today_changes.filter(change_type__in=['error_add', 'error_subtract']).count()

        # Pagination
        paginator = Paginator(qs, 50)
        page_obj = paginator.get_page(request.GET.get('page', 1))

        # Change type choices for filter dropdown
        change_types = StockChange._meta.get_field('change_type').choices

        return render(request, self.template_name, {
            'page_obj': page_obj,
            'product_query': product_query,
            'change_type_filter': change_type,
            'date_from': date_from,
            'date_to': date_to,
            'change_types': change_types,
            'checkins_today': checkins_today,
            'sales_today': sales_today,
            'adjustments_today': adjustments_today,
        })


class CycleCountView(AdminRequiredMixin, View):
    """Inventory reconciliation — physical count vs system count."""
    template_name = 'cycle_count.html'

    def get(self, request):
        category_id = request.GET.get('category', '')
        search_query = request.GET.get('search', '').strip()
        qs = Product.objects.filter(status=True).select_related('category').order_by('name')
        if category_id:
            qs = qs.filter(category_id=category_id)
        if search_query:
            qs = qs.filter(Q(name__icontains=search_query) | Q(barcode__icontains=search_query))

        return render(request, self.template_name, {
            'products': qs,
            'categories': Category.objects.all().order_by('name'),
            'selected_category': category_id,
            'search_query': search_query,
            'summary': None,
        })

    def post(self, request):
        counted = 0
        discrepancies = 0
        net_adjustment = 0
        details = []

        with transaction.atomic():
            for key, value in request.POST.items():
                if not key.startswith('count_'):
                    continue
                try:
                    product_id = int(key.replace('count_', ''))
                    actual_count = int(value)
                except (ValueError, TypeError):
                    continue

                product = Product.objects.select_for_update().filter(pk=product_id).first()
                if not product:
                    continue

                counted += 1
                diff = actual_count - product.quantity_in_stock

                if diff != 0:
                    discrepancies += 1
                    net_adjustment += diff
                    old_qty = product.quantity_in_stock
                    product.quantity_in_stock = actual_count
                    product.save(update_fields=['quantity_in_stock'])

                    change_type = 'error_add' if diff > 0 else 'error_subtract'
                    record_stock_change(
                        product, qty=abs(diff), change_type=change_type,
                        note=f"Cycle count: {old_qty} → {actual_count}",
                        user=request.user,
                    )
                    details.append({
                        'name': product.name,
                        'old': old_qty,
                        'new': actual_count,
                        'diff': diff,
                    })

        summary = {
            'counted': counted,
            'discrepancies': discrepancies,
            'net_adjustment': net_adjustment,
            'details': details,
        }

        category_id = request.POST.get('category', '')
        qs = Product.objects.filter(status=True).select_related('category').order_by('name')
        if category_id:
            qs = qs.filter(category_id=category_id)

        return render(request, self.template_name, {
            'products': qs,
            'categories': Category.objects.all().order_by('name'),
            'selected_category': category_id,
            'summary': summary,
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

        # Aggregate stats
        exp_agg = products.aggregate(
            total_units=Sum('quantity_in_stock'),
            value_at_risk=Sum(F('price') * F('quantity_in_stock')),
            total_expired_units=Sum('stock_expired'),
        )

        return render(request, self.template_name, {
            "products": products,
            "product": product,
            "date_filter": date_filter,
            "name_query": name_query,
            "all_products": list(Product.objects.values("product_id", "name", "barcode", "item_number", "price", "quantity_in_stock")),
            "product_count": products.count(),
            "total_units_on_shelf": exp_agg['total_units'] or 0,
            "value_at_risk": exp_agg['value_at_risk'] or Decimal('0.00'),
            "total_expired_units": exp_agg['total_expired_units'] or 0,
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
                            note="Marked as expired from expired product view",
                            user=request.user,
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
            "all_products": list(Product.objects.values("product_id", "name", "barcode", "item_number", "price", "quantity_in_stock")),
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
        return qs.exclude(expiry_date__isnull=True).select_related('category').order_by("expiry_date")
    
     
# View for displaying low-stock items
class LowStockView(AdminRequiredMixin, View):
    template_name = 'low_stock.html'
    threshold = 2

    def get(self, request):
        low_stock_products = Product.objects.filter(
            quantity_in_stock__lt=self.threshold, status=True
        ).order_by('name')

        q = request.GET.get('q', '').strip()
        category_filter = request.GET.get('category', '').strip()
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        active_categories = list(
            Category.objects
            .filter(product__recentlypurchasedproduct__isnull=False)
            .distinct().order_by('name')
            .values_list('id', 'name')
        )

        recently_purchased = (
            RecentlyPurchasedProduct.objects
            .all()
            .order_by('-order_date')
            .select_related('product', 'product__category')
        )
        if q:
            recently_purchased = recently_purchased.filter(
                Q(product__name__icontains=q) |
                Q(product__barcode__icontains=q) |
                Q(product__brand__icontains=q)
            )
        if category_filter:
            recently_purchased = recently_purchased.filter(product__category_id=category_filter)

        paginator_low_stock = Paginator(low_stock_products, 100)
        page_obj_low_stock = paginator_low_stock.get_page(request.GET.get('page'))

        # For AJAX live search, return all matching rows (no pagination cap)
        recent_page_size = 10000 if is_ajax else 80
        paginator_recent = Paginator(recently_purchased, recent_page_size)
        page_obj_recent = paginator_recent.get_page(request.GET.get('page_recent'))

        # ── Reorder predictions: 3 batch queries, no per-row DB hits ──────────
        today = date.today()
        page_product_ids = [
            item.product_id for item in page_obj_recent.object_list if item.product_id
        ]

        # Q1 — 60-day totals (base daily avg)
        demand_map = {
            row['product_id']: row['total']
            for row in StockChange.objects
            .filter(
                product_id__in=page_product_ids,
                timestamp__date__gte=today - timedelta(days=60),
                change_type__in=['checkout', 'checkout_unfulfilled'],
            )
            .values('product_id')
            .annotate(total=Sum('quantity'))
        }

        # Q2 — weekly totals for last 60 days (trend: linear regression)
        weekly_map = defaultdict(list)
        for row in (
            StockChange.objects
            .filter(
                product_id__in=page_product_ids,
                timestamp__date__gte=today - timedelta(days=60),
                change_type__in=['checkout', 'checkout_unfulfilled'],
            )
            .annotate(week=TruncWeek('timestamp'))
            .values('product_id', 'week')
            .annotate(total=Sum('quantity'))
            .order_by('product_id', 'week')
        ):
            weekly_map[row['product_id']].append((row['week'], row['total']))

        # Q3 — monthly totals for last 24 months (seasonality: month-of-year multiplier)
        monthly_map = defaultdict(list)
        for row in (
            StockChange.objects
            .filter(
                product_id__in=page_product_ids,
                timestamp__date__gte=today - timedelta(days=730),
                change_type__in=['checkout', 'checkout_unfulfilled'],
            )
            .annotate(month=TruncMonth('timestamp'))
            .values('product_id', 'month')
            .annotate(total=Sum('quantity'))
            .order_by('product_id', 'month')
        ):
            monthly_map[row['product_id']].append((row['month'], row['total']))

        # Q4 — weekly restock totals for last 60 days (chart: restocked line)
        restock_weekly_map = defaultdict(list)
        for row in (
            StockChange.objects
            .filter(
                product_id__in=page_product_ids,
                timestamp__date__gte=today - timedelta(days=60),
                change_type__in=['checkin', 'error_add'],
            )
            .annotate(week=TruncWeek('timestamp'))
            .values('product_id', 'week')
            .annotate(total=Sum('quantity'))
            .order_by('product_id', 'week')
        ):
            restock_weekly_map[row['product_id']].append((row['week'], row['total']))

        for item in page_obj_recent.object_list:
            item.reorder = (
                get_reorder_prediction(
                    item.product,
                    demand_map.get(item.product_id, 0),
                    weekly_demands=weekly_map.get(item.product_id, []),
                    monthly_demands=monthly_map.get(item.product_id, []),
                )
                if item.product_id else None
            )
            wk = weekly_map.get(item.product_id, [])
            restock_wk = restock_weekly_map.get(item.product_id, [])
            item.chart_json = json.dumps({
                'sold': [{'week': d.strftime('%Y-%m-%d'), 'qty': t} for d, t in wk],
                'restocked': [{'week': d.strftime('%Y-%m-%d'), 'qty': t} for d, t in restock_wk],
            })
        # ────────────────────────────────────────────────────────────────────

        if is_ajax:
            rows_html = render_to_string(
                'partials/rp_rows.html',
                {'page_obj_recent': page_obj_recent, 'q': q},
                request=request,
            )
            return JsonResponse({
                'html': rows_html,
                'count': page_obj_recent.paginator.count,
                'q': q,
                'category': category_filter,
                'categories': active_categories,
            })

        return render(request, self.template_name, {
            'page_obj_low_stock': page_obj_low_stock,
            'page_obj_recent':    page_obj_recent,
            'threshold':          self.threshold,
            'q':                  q,
            'active_categories':  active_categories,
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


class ActivityLogView(AdminRequiredMixin, View):
    template_name = 'activity_log.html'

    STOCK_TYPE_MAP = {
        'checkin': ['checkin'],
        'checkout': ['checkout'],
        'expired': ['expired'],
        'adjustment': ['error_add', 'error_subtract'],
        'deletion': ['deletion'],
        'checkin_delete1': ['checkin_delete1'],
    }
    ACTION_TYPE_MAP = {
        'delete_product': ['delete_product'],
        'delete_order': ['delete_order', 'delete_all_orders'],
        'delete_recently_purchased': ['delete_recently_purchased', 'delete_all_recently_purchased', 'bulk_delete_recently_purchased'],
        'submit_order': ['submit_order'],
        'add_product': ['add_product'],
    }
    LOGIN_TYPES = ('', 'all_logins', 'login', 'login_success', 'login_failed')
    STOCK_TYPES = ('', 'all_stock')
    ACTION_TYPES = ('', 'all_actions')

    def _build_events(self, event_type, user_filter, parsed_from, parsed_to):
        events = []
        include_logins = event_type in self.LOGIN_TYPES or event_type in ('login_success', 'login_failed')
        include_stock = event_type in self.STOCK_TYPES or event_type in self.STOCK_TYPE_MAP
        include_actions = event_type in self.ACTION_TYPES or event_type in self.ACTION_TYPE_MAP

        # Login events
        if include_logins:
            login_qs = LoginAudit.objects.select_related('user').all()
            if user_filter:
                login_qs = login_qs.filter(username__icontains=user_filter)
            if parsed_from:
                login_qs = login_qs.filter(timestamp__date__gte=parsed_from)
            if parsed_to:
                login_qs = login_qs.filter(timestamp__date__lte=parsed_to)
            if event_type == 'login_success':
                login_qs = login_qs.filter(success=True)
            elif event_type == 'login_failed':
                login_qs = login_qs.filter(success=False)
            for la in login_qs[:500]:
                events.append({
                    'timestamp': la.timestamp,
                    'category': 'Login',
                    'user': la.username,
                    'action': 'Login Success' if la.success else 'Login Failed',
                    'detail': f'IP: {la.ip_address or "unknown"}',
                    'badge': 'success' if la.success else 'failed',
                    'link': '',
                })

        # Stock change events
        if include_stock:
            stock_qs = StockChange.objects.select_related('product', 'user').all()
            if user_filter:
                stock_qs = stock_qs.filter(Q(user__username__icontains=user_filter))
            if parsed_from:
                stock_qs = stock_qs.filter(timestamp__date__gte=parsed_from)
            if parsed_to:
                stock_qs = stock_qs.filter(timestamp__date__lte=parsed_to)
            if event_type in self.STOCK_TYPE_MAP:
                stock_qs = stock_qs.filter(change_type__in=self.STOCK_TYPE_MAP[event_type])
            for sc in stock_qs[:500]:
                product_name = sc.product.name if sc.product else 'Deleted product'
                user_display = sc.user.username if sc.user else '—'
                if sc.change_type in ('checkin', 'error_add'):
                    badge = 'checkin'
                elif sc.change_type == 'checkout':
                    badge = 'checkout'
                elif sc.change_type == 'expired':
                    badge = 'expired'
                elif sc.change_type == 'deletion':
                    badge = 'deletion'
                else:
                    badge = 'other'
                # Build link to product on checkin page
                link = ''
                if sc.product and sc.product.barcode:
                    link = f"{reverse('checkin')}?barcode={sc.product.barcode}"
                events.append({
                    'timestamp': sc.timestamp,
                    'category': 'Stock',
                    'user': user_display,
                    'action': sc.get_change_type_display(),
                    'detail': f'{product_name} (qty: {sc.quantity})',
                    'badge': badge,
                    'link': link,
                })

        # User action events
        if include_actions:
            action_qs = UserAction.objects.select_related('user').all()
            if user_filter:
                action_qs = action_qs.filter(user__username__icontains=user_filter)
            if parsed_from:
                action_qs = action_qs.filter(timestamp__date__gte=parsed_from)
            if parsed_to:
                action_qs = action_qs.filter(timestamp__date__lte=parsed_to)
            if event_type in self.ACTION_TYPE_MAP:
                action_qs = action_qs.filter(action__in=self.ACTION_TYPE_MAP[event_type])
            for ua in action_qs[:500]:
                user_display = ua.user.username if ua.user else '—'
                if 'delete' in ua.action:
                    badge = 'deletion'
                elif ua.action == 'submit_order':
                    badge = 'checkout'
                elif ua.action == 'add_product':
                    badge = 'checkin'
                else:
                    badge = 'other'
                # Build link based on action type
                link = ''
                if ua.action == 'submit_order':
                    # Target is like "Order #54" — extract the ID
                    m = re.search(r'#(\d+)', ua.target)
                    if m:
                        link = reverse('order_detail', args=[int(m.group(1))])
                elif ua.action == 'add_product':
                    # Link to product on checkin by name lookup
                    try:
                        prod = Product.objects.filter(name=ua.target).first()
                        if prod and prod.barcode:
                            link = f"{reverse('checkin')}?barcode={prod.barcode}"
                    except Exception:
                        pass
                events.append({
                    'timestamp': ua.timestamp,
                    'category': 'Action',
                    'user': user_display,
                    'action': ua.get_action_display(),
                    'detail': ua.target,
                    'badge': badge,
                    'link': link,
                })

        events.sort(key=lambda e: e['timestamp'], reverse=True)
        return events

    def _filter_label(self, event_type):
        labels = {
            '': 'All Events', 'all_logins': 'All Logins', 'login_success': 'Login Success',
            'login_failed': 'Login Failed', 'all_stock': 'All Stock Changes',
            'checkin': 'Check-in', 'checkout': 'Checkout (Sale)', 'expired': 'Expired',
            'adjustment': 'Manual Adjustment', 'checkin_delete1': 'Stock Removed (UI)',
            'deletion': 'Product Deletion', 'all_actions': 'All Actions',
            'delete_product': 'Delete Product', 'delete_order': 'Delete Order',
            'delete_recently_purchased': 'Delete Recently Purchased',
            'submit_order': 'Submit Order', 'add_product': 'New Product',
        }
        return labels.get(event_type, 'All Events')

    def get(self, request):
        from django.contrib.auth import get_user_model
        User = get_user_model()

        user_filter = request.GET.get('user', '')
        event_type = request.GET.get('type', '')
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')

        parsed_from = parse_date(date_from) if date_from else None
        parsed_to = parse_date(date_to) if date_to else None

        events = self._build_events(event_type, user_filter, parsed_from, parsed_to)

        # PDF export
        if request.GET.get('export') == 'pdf':
            return self._render_pdf(events, event_type, user_filter, date_from, date_to)

        paginator = Paginator(events, 50)
        page_obj = paginator.get_page(request.GET.get('page', 1))

        today = date.today()
        logins_today = LoginAudit.objects.filter(timestamp__date=today, success=True).count()
        failed_today = LoginAudit.objects.filter(timestamp__date=today, success=False).count()
        actions_today = StockChange.objects.filter(timestamp__date=today).count() + UserAction.objects.filter(timestamp__date=today).count()

        users = User.objects.filter(is_active=True).order_by('username').values_list('username', flat=True)

        return render(request, self.template_name, {
            'page_obj': page_obj,
            'user_filter': user_filter,
            'event_type': event_type,
            'date_from': date_from,
            'date_to': date_to,
            'logins_today': logins_today,
            'failed_today': failed_today,
            'actions_today': actions_today,
            'users': list(users),
        })

    def _render_pdf(self, events, event_type, user_filter, date_from, date_to):
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas as pdf_canvas

        buffer = io.BytesIO()
        page_w, page_h = landscape(letter)
        c = pdf_canvas.Canvas(buffer, pagesize=landscape(letter))

        margin = 0.4 * inch
        usable_w = page_w - 2 * margin
        col_widths = [usable_w * 0.14, usable_w * 0.09, usable_w * 0.10, usable_w * 0.18, usable_w * 0.49]
        headers = ['Time', 'Category', 'User', 'Action', 'Details']
        row_height = 14
        header_height = 18
        font_size = 7.5
        header_font_size = 8

        filter_label = self._filter_label(event_type)
        date_range = ''
        if date_from and date_to:
            date_range = f' | {date_from} to {date_to}'
        elif date_from:
            date_range = f' | From {date_from}'
        elif date_to:
            date_range = f' | To {date_to}'
        user_label = f' | User: {user_filter}' if user_filter else ''
        subtitle = f'Filter: {filter_label}{user_label}{date_range} | {len(events)} events'

        def draw_page_header(c, page_num):
            c.setFont('Helvetica-Bold', 12)
            c.drawString(margin, page_h - margin, 'Activity Log')
            c.setFont('Helvetica', 8)
            c.drawString(margin, page_h - margin - 14, subtitle)
            c.drawRightString(page_w - margin, page_h - margin, f'Page {page_num}')
            c.setFont('Helvetica', 6.5)
            c.drawRightString(page_w - margin, page_h - margin - 14,
                              f'Generated {date.today().strftime("%b %d, %Y")}')

        def draw_table_header(c, y):
            c.setFillColor(colors.Color(0.95, 0.96, 0.98))
            c.rect(margin, y - header_height + 4, usable_w, header_height, fill=1, stroke=0)
            c.setFillColor(colors.Color(0.3, 0.3, 0.4))
            c.setFont('Helvetica-Bold', header_font_size)
            x = margin + 4
            for i, hdr in enumerate(headers):
                c.drawString(x, y - header_height + 8, hdr.upper())
                x += col_widths[i]
            return y - header_height

        def truncate(text, font, size, max_w):
            from reportlab.pdfbase.pdfmetrics import stringWidth as sw
            if sw(text, font, size) <= max_w:
                return text
            while len(text) > 1 and sw(text + '...', font, size) > max_w:
                text = text[:-1]
            return text + '...'

        page_num = 1
        draw_page_header(c, page_num)
        y = page_h - margin - 32
        y = draw_table_header(c, y)
        bottom = margin + 20

        for idx, ev in enumerate(events):
            if y - row_height < bottom:
                c.setFont('Helvetica', 6.5)
                c.setFillColor(colors.Color(0.6, 0.6, 0.6))
                c.drawCentredString(page_w / 2, margin + 4, f'Page {page_num} of Activity Log')
                c.showPage()
                page_num += 1
                draw_page_header(c, page_num)
                y = page_h - margin - 32
                y = draw_table_header(c, y)

            if idx % 2 == 0:
                c.setFillColor(colors.Color(0.98, 0.98, 1.0))
                c.rect(margin, y - row_height + 4, usable_w, row_height, fill=1, stroke=0)

            c.setFillColor(colors.Color(0.2, 0.2, 0.3))
            c.setFont('Helvetica', font_size)
            x = margin + 4
            row_data = [
                ev['timestamp'].strftime('%b %d, %Y %H:%M'),
                ev['category'],
                ev['user'],
                ev['action'],
                ev['detail'],
            ]
            for i, cell in enumerate(row_data):
                text = truncate(str(cell), 'Helvetica', font_size, col_widths[i] - 8)
                c.drawString(x, y - row_height + 8, text)
                x += col_widths[i]
            y -= row_height

        c.setFont('Helvetica', 6.5)
        c.setFillColor(colors.Color(0.6, 0.6, 0.6))
        c.drawCentredString(page_w / 2, margin + 4, f'Page {page_num} of Activity Log')
        c.save()
        buffer.seek(0)

        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="activity_log_{date.today().strftime("%Y%m%d")}.pdf"'
        return response


# Delete a recently purchased product
class DeleteRecentlyPurchasedProductView(LoginRequiredMixin, View):
   def post(self, request, id):
       is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
       try:
           recently_purchased = RecentlyPurchasedProduct.objects.get(id=id)
           product_name = recently_purchased.product.name if recently_purchased.product else "Unknown"
           recently_purchased.delete()
           UserAction.objects.create(
               user=request.user, action='delete_recently_purchased',
               target=product_name,
           )
           if is_ajax:
               return JsonResponse({'success': True, 'name': product_name})
           messages.success(request, f"{product_name} has been deleted from the recently purchased list.")
       except RecentlyPurchasedProduct.DoesNotExist:
           if is_ajax:
               return JsonResponse({'success': False, 'error': 'Item not found'}, status=404)
           messages.error(request, "The selected product does not exist in the recently purchased list.")
       page_recent = request.POST.get('page_recent', '1')
       return redirect(f"{reverse('low_stock')}?page_recent={page_recent}")


class DeleteAllRecentlyPurchasedView(LoginRequiredMixin, View):
   def post(self, request):
       is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
       count = RecentlyPurchasedProduct.objects.count()
       RecentlyPurchasedProduct.objects.all().delete()
       UserAction.objects.create(
           user=request.user, action='delete_all_recently_purchased',
           target=f'{count} items',
       )
       if is_ajax:
           return JsonResponse({'success': True})
       messages.success(request, "All recently purchased products have been deleted.")
       return redirect('low_stock')


class BulkDeleteRecentlyPurchasedView(LoginRequiredMixin, View):
   def post(self, request):
       try:
           data = json.loads(request.body)
           ids = data.get('ids', [])
           if not ids:
               return JsonResponse({'success': False, 'error': 'No IDs provided'}, status=400)
           deleted_count, _ = RecentlyPurchasedProduct.objects.filter(id__in=ids).delete()
           UserAction.objects.create(
               user=request.user, action='bulk_delete_recently_purchased',
               target=f'{deleted_count} items',
           )
           return JsonResponse({'success': True, 'deleted_count': deleted_count})
       except (json.JSONDecodeError, Exception) as e:
           return JsonResponse({'success': False, 'error': str(e)}, status=400)


class DeleteByCategoryRecentlyPurchasedView(LoginRequiredMixin, View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            category_id = data.get('category_id')
            if not category_id:
                return JsonResponse({'success': False, 'error': 'No category ID'}, status=400)
            deleted_count, _ = RecentlyPurchasedProduct.objects.filter(
                product__category_id=category_id
            ).delete()
            UserAction.objects.create(
                user=request.user, action='bulk_delete_recently_purchased',
                target=f'{deleted_count} items (by category)',
            )
            return JsonResponse({'success': True, 'deleted_count': deleted_count})
        except (json.JSONDecodeError, Exception) as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)


class DeleteOlderThanRecentlyPurchasedView(LoginRequiredMixin, View):
    ALLOWED_DAYS = {30, 60, 90}

    def post(self, request):
        try:
            data = json.loads(request.body)
            days = data.get('days')
            if days not in self.ALLOWED_DAYS:
                return JsonResponse({'success': False, 'error': 'Invalid days value'}, status=400)
            cutoff = now() - timedelta(days=days)
            deleted_count, _ = RecentlyPurchasedProduct.objects.filter(
                order_date__lt=cutoff
            ).delete()
            UserAction.objects.create(
                user=request.user, action='bulk_delete_recently_purchased',
                target=f'{deleted_count} items (older than {days} days)',
            )
            return JsonResponse({'success': True, 'deleted_count': deleted_count})
        except (json.JSONDecodeError, Exception) as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)


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
                change_type="deletion",
                note=f"Product deleted with {remaining_stock} units in stock",
                user=request.user,
            )
        
        # Delete the product
        product.delete()

    UserAction.objects.create(
        user=request.user, action='delete_product',
        target=product_name,
        detail=f"Had {remaining_stock} units remaining",
    )
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
        order_count = Order.objects.count()
        Order.objects.all().delete()
        UserAction.objects.create(
            user=request.user, action='delete_all_orders',
            target=f'{order_count} orders',
        )

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


# Delete a single order
class DeleteOrderView(AdminRequiredMixin, View):
    def post(self, request, order_id):
        order = get_object_or_404(Order, order_id=order_id)

        # If this is the current in-progress order, clear session state
        if request.session.get('order_id') == order_id:
            request.session.pop('order_id', None)
            request.session.pop('cart', None)
            request.session.modified = True

        order.delete()
        UserAction.objects.create(
            user=request.user, action='delete_order',
            target=f'Order #{order_id}',
        )
        messages.success(request, f"Order #{order_id} has been deleted.")
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

class DeliveryView(LoginRequiredMixin, View):
    template_name = 'delivery.html'

    def get(self, request):
        active_records = DeliveryCheckIn.objects.filter(checked_out_at__isnull=True).order_by('-checked_in_at')
        history_records = DeliveryCheckIn.objects.filter(checked_out_at__isnull=False).order_by('-checked_out_at')[:50]

        return render(request, self.template_name, {
            'active_records': active_records,
            'history_records': history_records,
        })

    def post(self, request):
        action = request.POST.get('action')

        if action == 'checkin':
            barcode = _normalize_barcode(request.POST.get('barcode', ''))
            first_name = request.POST.get('first_name', '').strip()
            last_name = request.POST.get('last_name', '').strip()

            if not barcode or not first_name or not last_name:
                messages.error(request, "Barcode, first name, and last name are all required.")
                return redirect('delivery')

            already = DeliveryCheckIn.objects.filter(barcode=barcode, checked_out_at__isnull=True).first()
            if already:
                messages.error(request, f"{already.first_name} {already.last_name} is already checked in with that barcode.")
                return redirect('delivery')

            DeliveryCheckIn.objects.create(
                barcode=barcode,
                first_name=first_name,
                last_name=last_name,
            )
            messages.success(request, f"{first_name} {last_name} checked in.")
            return redirect('delivery')

        elif action == 'checkout':
            record_id = request.POST.get('record_id', '').strip()
            barcode_raw = request.POST.get('barcode', '').strip()

            if record_id:
                record = DeliveryCheckIn.objects.filter(pk=record_id, checked_out_at__isnull=True).first()
            else:
                barcode = _normalize_barcode(barcode_raw)
                record = DeliveryCheckIn.objects.filter(
                    barcode=barcode, checked_out_at__isnull=True
                ).order_by('-checked_in_at').first()

            if record:
                record.checked_out_at = now()
                record.save()
                return JsonResponse({
                    'status': 'ok',
                    'name': f"{record.first_name} {record.last_name}",
                    'record_id': record.pk,
                    'barcode': record.barcode,
                    'checked_in_at': record.checked_in_at.strftime('%d %b %Y, %H:%M'),
                    'checked_out_at': record.checked_out_at.strftime('%d %b %Y, %H:%M'),
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'No active check-in found for this barcode.'})

        elif action == 'undo_checkout':
            record_id = request.POST.get('record_id', '').strip()
            record = DeliveryCheckIn.objects.filter(pk=record_id, checked_out_at__isnull=False).first()
            if record:
                record.checked_out_at = None
                record.save()
                return JsonResponse({
                    'status': 'ok',
                    'record_id': record.pk,
                    'name': f"{record.first_name} {record.last_name}",
                    'barcode': record.barcode,
                    'checked_in_at': record.checked_in_at.strftime('%d %b %Y, %H:%M'),
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'Record not found or already active.'})

        elif action == 'clear_history':
            DeliveryCheckIn.objects.filter(checked_out_at__isnull=False).delete()
            messages.success(request, "Checkout history cleared.")
            return redirect('delivery')

        return redirect('delivery')


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