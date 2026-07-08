from decimal import Decimal, ROUND_HALF_UP
import hmac
import time
import os
import csv
import io
import base64
import json
import re
import qrcode
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
from django.utils.timezone import now, localtime
from django.utils.timesince import timesince
from django.utils.http import url_has_allowed_host_and_scheme
from django.contrib.auth.decorators import login_required, user_passes_test
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView
from django.contrib.sessions.models import Session as DjangoSession
from django.contrib.auth.forms import UserCreationForm
from app.mixins import (
    AdminRequiredMixin, UserRequiredMixin,
    has_admin_access, passkey_unlocked, PASSKEY_SESSION_KEY,
)
from .utils import recalculate_order_totals, get_product_stock_records, recommend_inventory_action, get_reorder_prediction, TAX_RATE
from .forms import EditProductForm, OrderDetailForm, BarcodeForm, ItemForm, AddProductForm, OrderingSheetForm, OTCOrderingForm
from .models import Item, Product, Category, Order, OrderDetail, RecentlyPurchasedProduct, StockChange, CheckinSession, DeliveryCheckIn, LoginAudit, UserAction, LabelQueueItem, LabelSession, LabelSessionItem, ProductExpiryDate, UserSession, CheckoutOrder, CheckoutOrderItem, PagePresence, OrderingSheetEntry, InventoryCountLine, DailyReportArchive
from .page_lock import is_fresh, holder_info, presence_defaults, simplify_ua, page_label, path_label, PRESENCE_TTL
from . import session_limits
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


def _label_wrap_text(text, font_name, font_size, max_width):
    def _break_long_word(word):
        """Split a single word that itself overflows max_width into char chunks
        so unbroken strings (e.g. a long SKU or run-on name) still wrap."""
        if stringWidth(word, font_name, font_size) <= max_width:
            return [word]
        chunks, current = [], ""
        for ch in word:
            if current and stringWidth(current + ch, font_name, font_size) > max_width:
                chunks.append(current)
                current = ch
            else:
                current += ch
        if current:
            chunks.append(current)
        return chunks

    lines, current = [], ""
    for w in text.split():
        for piece in _break_long_word(w):
            test = (current + " " + piece) if current else piece
            if stringWidth(test, font_name, font_size) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = piece
    if current:
        lines.append(current)
    return lines


def _draw_label(c, x, y, data):
    name = data.get("name", "")
    brand = data.get("brand", "")
    item_num = data.get("item_number", "")
    bc_val = data.get("barcode", "")
    price = f"${float(data.get('price', 0)):.2f}"

    c.setFont("Helvetica-Bold", 10)
    max_w = LABEL_WIDTH - LEFT_PADDING - RIGHT_PADDING
    lines = _label_wrap_text(name, "Helvetica-Bold", 10, max_w)[:4]
    for i, line in enumerate(lines):
        c.drawCentredString(x + LABEL_WIDTH / 2, y + LABEL_HEIGHT - 10 - (i * 11), line)

    base_y = y + BOTTOM_PADDING
    body_x = x + LEFT_PADDING

    if bc_val:
        try:
            barcode = code128.Code128(bc_val, barHeight=16, barWidth=0.9, humanReadable=False)
            barcode.drawOn(c, body_x, base_y + 20)
            c.setFont("Helvetica", 6)
            c.drawString(body_x, base_y + 14, bc_val)
        except Exception:
            pass

    if item_num:
        c.setFont("Helvetica", 6)
        c.drawString(body_x, base_y + 8, f"Item #: {item_num}")

    if brand:
        c.setFont("Helvetica", 6)
        c.drawString(body_x, base_y + 2, brand[:25])

    c.setFont("Helvetica-Bold", 17)
    c.drawRightString(x + LABEL_WIDTH - RIGHT_PADDING, base_y + 4, price)


def _truncate_to_width(text, font_name, font_size, max_width):
    """Trim text with an ellipsis so it fits within max_width."""
    text = str(text or "")
    if stringWidth(text, font_name, font_size) <= max_width:
        return text
    ell = "…"
    trimmed = text
    while trimmed and stringWidth(trimmed + ell, font_name, font_size) > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + ell) if trimmed else text[:1]


def _fit_font_size(text, font, max_size, min_size, max_width):
    """Largest font size (down to min_size, 0.5pt steps) at which text fits."""
    size = max_size
    while size > min_size and stringWidth(text, font, size) > max_width:
        size -= 0.5
    return size


def _draw_custom_label(c, x, y, label):
    """Draw a custom label: an item name centered at the top plus up to five
    text/price section lines beneath it.

    Accepts the current shape {"title": str, "lines": [{"text", "price"}]} and
    the legacy shapes (plain list of products, or lines keyed "name") so old
    queued labels and the legacy direct-print route keep working. Text is
    shrunk to fit its section, then ellipsised as a last resort.
    """
    if isinstance(label, dict):
        title = str(label.get("title", "") or "").strip()
        raw_lines = label.get("lines", []) or []
    else:  # legacy: plain list of {"name","price"} products
        title = ""
        raw_lines = label or []

    lines = []
    for p in raw_lines:
        text = str(p.get("text", p.get("name", "")) or "").strip()
        if text:
            lines.append({"text": text, "price": p.get("price", 0)})
    lines = lines[:5]

    if not title and not lines:
        return

    h_pad = 11
    pad_top, pad_bottom = 7, 7
    body_x = x + h_pad
    right_x = x + LABEL_WIDTH - h_pad
    inner_w = LABEL_WIDTH - 2 * h_pad
    center_x = x + LABEL_WIDTH / 2
    font = "Helvetica-Bold"

    region_top = y + LABEL_HEIGHT - pad_top
    region_bottom = y + pad_bottom

    # ── Title: centered across the top, sized to fit the label width ──
    if title:
        if lines:
            title_h = 18
        else:
            title_h = LABEL_HEIGHT - pad_top - pad_bottom  # title-only label
        t_size = _fit_font_size(title, font, 13 if lines else 16, 7, inner_w)
        if stringWidth(title, font, t_size) > inner_w:
            ell_w = stringWidth("…", font, t_size)
            title = _truncate_to_width(title, font, t_size, inner_w - ell_w) + "…"
        c.setFont(font, t_size)
        c.drawCentredString(center_x, region_top - title_h / 2 - t_size * 0.34, title)

        if lines:
            sep_y = region_top - title_h
            c.setLineWidth(0.5)
            c.setStrokeGray(0.55)
            c.line(body_x, sep_y, right_x, sep_y)
            c.setStrokeGray(0)
            region_top = sep_y - 1

    # ── Section lines: one band each, text left / price right ──
    n = len(lines)
    if n == 0:
        return
    region_h = region_top - region_bottom
    band_h = region_h / n

    # Font ceiling scales with how much room each band has.
    max_size = max(6.5, min(11.5, band_h * 0.55))

    for i, p in enumerate(lines):
        band_top = region_top - i * band_h
        band_center = band_top - band_h / 2

        price_val = p.get("price", 0)
        try:
            price = f"${float(price_val or 0):.2f}"
        except (TypeError, ValueError):
            price = "$0.00"
        price_size = max_size
        price_w = stringWidth(price, font, price_size)
        max_text_w = inner_w - price_w - 8

        text = p["text"]
        t_size = _fit_font_size(text, font, max_size, 6.5, max_text_w)
        if stringWidth(text, font, t_size) > max_text_w:
            ell_w = stringWidth("…", font, t_size)
            text = _truncate_to_width(text, font, t_size, max_text_w - ell_w) + "…"

        c.setFont(font, t_size)
        c.drawString(body_x, band_center - t_size * 0.34, text)
        c.setFont(font, price_size)
        c.drawRightString(right_x, band_center - price_size * 0.34, price)

        if i < n - 1:
            sep_y = band_top - band_h
            c.setLineWidth(0.3)
            c.setStrokeGray(0.78)
            c.line(body_x, sep_y, right_x, sep_y)
            c.setStrokeGray(0)


def render_labels_pdf_response(final_queue, draw_fn=_draw_label):
    """Render a list of label items into a 4x8 PDF sheet using the given draw function."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=portrait(letter))
    PAGE_WIDTH, PAGE_HEIGHT = portrait(letter)

    usable_w = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
    usable_h = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    h_gutter = (usable_w - (COLUMNS * LABEL_WIDTH)) / (COLUMNS - 1) if COLUMNS > 1 else 0
    v_gutter = (usable_h - (ROWS * LABEL_HEIGHT)) / (ROWS - 1) if ROWS > 1 else 0

    for count, item in enumerate(final_queue):
        col = count % COLUMNS
        row_num = (count // COLUMNS) % ROWS
        x = LEFT_MARGIN + col * (LABEL_WIDTH + h_gutter)
        y_top = PAGE_HEIGHT - TOP_MARGIN - row_num * (LABEL_HEIGHT + v_gutter)
        y = y_top - LABEL_HEIGHT

        draw_fn(c, x, y, item)

        if (count + 1) % LABELS_PER_PAGE == 0 and (count + 1) < len(final_queue):
            c.showPage()

    c.save()
    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf')


def _draw_label_or_custom(c, x, y, item):
    """Draw a product label, or a title+lines custom label when item['custom']."""
    if isinstance(item, dict) and item.get('custom'):
        _draw_custom_label(c, x, y, item)
    else:
        _draw_label(c, x, y, item)


def _session_custom_labels(request):
    """The current session's queued custom labels (free-form name/price labels)."""
    return request.session.get('custom_labels', [])


def _build_preview_labels(category_items, queue_items, custom_labels):
    """Flat list of labels for the live sheet preview (products + custom)."""
    labels = []
    for p in category_items:
        labels.append({'name': p.name, 'barcode': p.barcode or '', 'price': str(p.price),
                       'brand': p.brand or '', 'item_number': p.item_number or '', 'qty': 1})
    for qi in queue_items:
        p = qi.product
        labels.append({'name': p.name, 'barcode': p.barcode or '', 'price': str(p.price),
                       'brand': p.brand or '', 'item_number': p.item_number or '', 'qty': qi.qty})
    for cl in custom_labels:
        labels.append({'custom': True, 'title': cl.get('title', ''),
                       'lines': cl.get('lines', []),
                       'qty': max(1, int(cl.get('copies', 1)))})
    return labels


def _get_client_ip(request):
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


class LabelPrintingView(LoginRequiredMixin, View):
    template_name = "label_printing.html"

    def _get_queue(self, request):
        return LabelQueueItem.objects.filter(user=request.user).select_related('product')

    def get(self, request):
        if request.headers.get('Accept') == 'application/json' and 'category_id' in request.GET:
            cat_id = request.GET.get('category_id')
            products = Product.objects.filter(category_id=cat_id, status=True).values(
                'product_id', 'name', 'barcode', 'price'
            ).order_by('name')
            return JsonResponse({'products': list(products)})

        queue_items = self._get_queue(request)
        category_items = Product.objects.filter(category__name__icontains="Print Label", status=True)

        all_products = list(Product.objects.filter(status=True).values(
            'product_id', 'name', 'barcode', 'item_number', 'price', 'quantity_in_stock'
        ))

        custom_labels = _session_custom_labels(request)

        return render(request, self.template_name, {
            "queue_items": queue_items,
            "category_items": category_items,
            "category_items_count": category_items.count(),
            "categories": Category.objects.all().order_by('name'),
            "all_products": all_products,
            "custom_labels": list(enumerate(custom_labels)),
            "custom_labels_count": sum(max(1, int(cl.get('copies', 1))) for cl in custom_labels),
            "preview_labels": _build_preview_labels(category_items, queue_items, custom_labels),
        })

    def post(self, request):
        if "add_product" in request.POST:
            product = get_object_or_404(Product, pk=request.POST.get("product_id"))
            LabelQueueItem.objects.create(product=product, user=request.user)
            messages.success(request, f"Added {product.name} to label queue.")

        elif "add_selected_products" in request.POST:
            product_ids = request.POST.getlist("selected_products")
            if product_ids:
                products = Product.objects.filter(product_id__in=product_ids, status=True)
                LabelQueueItem.objects.bulk_create([
                    LabelQueueItem(product=p, user=request.user) for p in products
                ])
                messages.success(request, f"Added {products.count()} selected items to print queue.")
            else:
                messages.warning(request, "Select at least one product to add.")

        elif "add_category" in request.POST:
            cat_id = request.POST.get("category_id")
            if not cat_id:
                messages.warning(request, "Select a category first.")
            else:
                products = Product.objects.filter(category_id=cat_id, status=True)
                if products.exists():
                    LabelQueueItem.objects.bulk_create([
                        LabelQueueItem(product=p, user=request.user) for p in products
                    ])
                    messages.success(request, f"Added {products.count()} items from category.")
                else:
                    messages.warning(request, "That category has no active products to add.")

        elif "quick_scan" in request.POST:
            barcode = request.POST.get("barcode", "").strip()
            if not barcode:
                messages.warning(request, "Scan or type a barcode first.")
            else:
                product = find_product_by_barcode(barcode)
                if product:
                    LabelQueueItem.objects.create(product=product, user=request.user)
                    messages.success(request, f"Scanned and added: {product.name}")
                else:
                    messages.error(request, f"Barcode '{barcode}' not found.")

        elif "add_custom_label" in request.POST:
            title = (request.POST.get("custom_title", "") or "").strip()[:80]
            lines = []
            for i in range(5):
                text = (request.POST.get(f"line_text_{i}", "") or "").strip()
                if not text:
                    continue
                try:
                    price = float(request.POST.get(f"line_price_{i}", 0) or 0)
                except (ValueError, TypeError):
                    price = 0.0
                lines.append({"text": text[:120], "price": price})
            if not title:
                messages.error(request, "Enter the item name for the top of the label.")
            else:
                try:
                    copies = max(1, min(99, int(request.POST.get("copies", 1) or 1)))
                except (ValueError, TypeError):
                    copies = 1
                custom = _session_custom_labels(request)
                custom.append({"title": title, "lines": lines, "copies": copies})
                request.session["custom_labels"] = custom
                request.session.modified = True
                plural = "s" if len(lines) != 1 else ""
                messages.success(
                    request,
                    f"Added custom label '{title}' ({len(lines)} line{plural} × {copies})."
                )

        elif "remove_custom_label" in request.POST:
            try:
                idx = int(request.POST.get("remove_custom_label"))
            except (ValueError, TypeError):
                idx = -1
            custom = _session_custom_labels(request)
            if 0 <= idx < len(custom):
                custom.pop(idx)
                request.session["custom_labels"] = custom
                request.session.modified = True
                messages.success(request, "Custom label removed.")
            else:
                messages.warning(request, "That custom label was already removed.")

        elif "clear_queue" in request.POST:
            self._get_queue(request).delete()
            request.session["custom_labels"] = []
            request.session.modified = True
            UserAction.objects.create(user=request.user, action='clear_label_queue',
                target='Label queue cleared')
            messages.info(request, "Label queue cleared.")

        elif "remove_item" in request.POST:
            item_id = request.POST.get("remove_item")
            deleted, _ = self._get_queue(request).filter(pk=item_id).delete()
            if deleted:
                messages.success(request, "Label removed from queue.")
            else:
                messages.warning(request, "That label was already removed.")

        elif "update_qty" in request.POST:
            item_id = request.POST.get("item_id")
            try:
                qty = max(1, int(request.POST.get("qty", 1)))
            except (ValueError, TypeError):
                qty = 1
            updated = self._get_queue(request).filter(pk=item_id).update(qty=qty)
            if updated:
                messages.success(request, f"Label quantity set to {qty}.")
            else:
                messages.warning(request, "Label not found — it may have been removed.")

        return redirect("label_printing")
    
class GenerateLabelPDFView(LoginRequiredMixin, View):
    def get(self, request):
        category_items = Product.objects.filter(
            category__name__icontains="Print Label",
            status=True
        )
        queue_items = LabelQueueItem.objects.filter(user=request.user).select_related('product')

        merged = []
        for p in category_items:
            merged.append({
                "name": p.name, "brand": p.brand or "",
                "item_number": p.item_number or "",
                "barcode": p.barcode or "", "price": str(p.price),
            })
        for qi in queue_items:
            p = qi.product
            merged.append({
                "name": p.name, "brand": p.brand or "",
                "item_number": p.item_number or "",
                "barcode": p.barcode or "", "price": str(p.price),
                "qty": qi.qty,
            })

        # Expand qty: repeat each item qty times
        final_queue = []
        for item in merged:
            qty = max(1, int(item.get("qty", 1)))
            for _ in range(qty):
                final_queue.append(item)

        # Custom labels (centered title + up to 5 text/price lines) added via
        # the "Add Label" button — expand by copies and mark them so the sheet
        # draws them with the custom layout.
        for cl in _session_custom_labels(request):
            label = {"custom": True, "title": cl.get("title", ""), "lines": cl.get("lines", [])}
            for _ in range(max(1, int(cl.get("copies", 1)))):
                final_queue.append(label)

        if not final_queue:
            messages.error(request, "No labels to print.")
            return redirect("label_printing")

        # ── Save session snapshot ──
        session_obj = LabelSession.objects.create(
            user=request.user,
            label_count=len(final_queue),
        )
        snapshot_items = []
        for p in category_items:
            snapshot_items.append(LabelSessionItem(
                session=session_obj, product=p,
                product_name=p.name, product_barcode=p.barcode or '',
                product_price=p.price, product_brand=p.brand or '',
                product_item_number=p.item_number or '', qty=1,
            ))
        for qi in queue_items:
            p = qi.product
            snapshot_items.append(LabelSessionItem(
                session=session_obj, product=p,
                product_name=p.name, product_barcode=p.barcode or '',
                product_price=p.price, product_brand=p.brand or '',
                product_item_number=p.item_number or '', qty=qi.qty,
            ))
        LabelSessionItem.objects.bulk_create(snapshot_items)
        UserAction.objects.create(user=request.user, action='print_labels',
            target=f'Session #{session_obj.pk}', detail=f'{len(final_queue)} labels printed')

        return render_labels_pdf_response(final_queue, draw_fn=_draw_label_or_custom)


class CustomLabelPDFView(LoginRequiredMixin, View):
    """Generate a special label holding up to 3 manually-entered name/price products."""
    MAX_PRODUCTS = 3

    def post(self, request):
        products = []
        for i in range(self.MAX_PRODUCTS):
            name = (request.POST.get(f"name_{i}", "") or "").strip()
            if not name:
                continue
            try:
                price = float(request.POST.get(f"price_{i}", 0) or 0)
            except (ValueError, TypeError):
                price = 0.0
            products.append({"name": name, "price": price})

        if not products:
            messages.error(request, "Enter at least one product name to print a custom label.")
            return redirect("label_printing")

        try:
            copies = max(1, min(99, int(request.POST.get("copies", 1) or 1)))
        except (ValueError, TypeError):
            copies = 1

        # Each label holds all the products; repeat the whole label `copies` times.
        final_queue = [products for _ in range(copies)]

        UserAction.objects.create(
            user=request.user, action='print_custom_labels',
            target='Custom labels',
            detail=f'{len(products)} products × {copies} copies',
        )

        return render_labels_pdf_response(final_queue, draw_fn=_draw_custom_label)

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


_NO_BARCODE_ALIASES = {"nb", "no barcode", "n/a", "0"}

def _is_no_barcode(value: str) -> bool:
    """Return True if the value represents a 'no barcode' entry."""
    cleaned = (value or "").strip().lower()
    if cleaned in _NO_BARCODE_ALIASES:
        return True
    # Treat any string of only zeros ("00", "000", …) as no barcode
    if cleaned and all(ch == '0' for ch in cleaned):
        return True
    return False

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


def barcode_search_q(query, field='barcode'):
    """Q object matching `query` against a barcode as a partial OR a
    leading-zero-tolerant exact match, so scanned barcodes match regardless of
    leading zeros — consistent with find_product_by_barcode. `field` supports
    related lookups (e.g. 'product__barcode'). Safe for non-barcode (name)
    queries: the exact clause is only added when the query contains digits.
    """
    q = Q(**{f'{field}__icontains': query})
    normalized = _normalize_barcode(query)
    if normalized:
        q |= Q(**{f'{field}__regex': rf'^0*{normalized}$'})
    return q


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


@lru_cache(maxsize=1)
def _master_unit_gtin_index():
    """Map normalized unit GTIN -> list of catalog rows (built once per process)."""
    index = defaultdict(list)
    for row in _load_master_catalog():
        key = _normalize_barcode(
            row.get("GTIN/UPC (unit)") or row.get("GTIN/UPC") or row.get("UPC")
        )
        if key:  # skip blank / all-zero placeholder GTINs
            index[key].append(row)
    return index


def get_master_catalog_entry(barcode: str):
    """Return the catalog row for a scanned barcode, but ONLY on a single
    unambiguous match (digits only, leading zeros ignored).

    A missing barcode, no match, or an ambiguous match (the same GTIN shared by
    more than one catalog row) all return None — better to fill the form in
    manually than to risk pre-filling the wrong product.
    """
    if not barcode:
        return None

    target = _normalize_barcode(barcode)
    if not target:
        return None

    matches = _master_unit_gtin_index().get(target, [])
    return matches[0] if len(matches) == 1 else None

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
        low_stock_count    = Product.objects.filter(
            status=True, quantity_in_stock__gt=0,
        ).annotate(
            _threshold=Coalesce(F('category__low_stock_threshold'), Value(3))
        ).filter(quantity_in_stock__lte=F('_threshold')).count()

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
            search_results = Product.objects.filter(Q(name__icontains=query) | barcode_search_q(query))
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
            elif ctype == "giveaway":
                # Free giveaway via PU terminal — physically removes stock,
                # but is not a sale, so only the on-hand running total moves.
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
            "giveaway": -1,  # terminal giveaway removes stock (giveaway_unfulfilled → 0 via .get)
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
        search_q = request.GET.get('q', '').strip()

        if include_inactive:
            products_qs = Product.objects.filter(quantity_in_stock=0)
        else:
            products_qs = Product.objects.filter(status=True, quantity_in_stock=0)

        if category_filter:
            products_qs = products_qs.filter(category_id=category_filter)

        if search_q:
            products_qs = products_qs.filter(
                Q(name__icontains=search_q) | barcode_search_q(search_q)
            )

        products = list(
            products_qs.select_related('category').order_by("-stock_unfulfilled", "name")
        )

        thirty_days_ago = date.today() - timedelta(days=30)
        recent_unfulfilled = dict(
            StockChange.objects.filter(
                product__in=[p.product_id for p in products],
                change_type='checkout_unfulfilled',
                timestamp__gte=thirty_days_ago,
            ).values_list('product_id').annotate(missed=Sum('quantity'))
        )

        total_missed = 0
        total_revenue_lost = Decimal('0.00')
        for p in products:
            p.missed_30d = recent_unfulfilled.get(p.product_id, 0)
            p.revenue_lost_30d = p.missed_30d * p.price
            total_missed += p.missed_30d
            total_revenue_lost += p.revenue_lost_30d

        paginator = Paginator(products, 50)
        page_obj = paginator.get_page(request.GET.get('page'))

        self._attach_reorder_predictions(page_obj)

        return render(request, self.template_name, {
            "products": page_obj,
            "total_missed": total_missed,
            "total_revenue_lost": total_revenue_lost,
            "product_count": len(products),
            "page_obj": page_obj,
            "categories": Category.objects.all().order_by('name'),
            "category_filter": category_filter,
            "include_inactive": include_inactive,
            "search_q": search_q,
        })

    @staticmethod
    def _attach_reorder_predictions(page_obj):
        today = date.today()
        pids = [p.product_id for p in page_obj]
        if not pids:
            return

        demand_map = {
            r['product_id']: r['total']
            for r in StockChange.objects.filter(
                product_id__in=pids,
                timestamp__date__gte=today - timedelta(days=60),
                change_type__in=['checkout', 'checkout_unfulfilled'],
            ).values('product_id').annotate(total=Sum('quantity'))
        }

        weekly_map = defaultdict(list)
        for r in StockChange.objects.filter(
            product_id__in=pids,
            timestamp__date__gte=today - timedelta(days=60),
            change_type__in=['checkout', 'checkout_unfulfilled'],
        ).annotate(week=TruncWeek('timestamp')).values('product_id', 'week').annotate(total=Sum('quantity')).order_by('product_id', 'week'):
            weekly_map[r['product_id']].append((r['week'], r['total']))

        monthly_map = defaultdict(list)
        for r in StockChange.objects.filter(
            product_id__in=pids,
            timestamp__date__gte=today - timedelta(days=730),
            change_type__in=['checkout', 'checkout_unfulfilled'],
        ).annotate(month=TruncMonth('timestamp')).values('product_id', 'month').annotate(total=Sum('quantity')).order_by('product_id', 'month'):
            monthly_map[r['product_id']].append((r['month'], r['total']))

        for p in page_obj:
            p.reorder = get_reorder_prediction(
                p, demand_map.get(p.product_id, 0),
                weekly_demands=weekly_map.get(p.product_id, []),
                monthly_demands=monthly_map.get(p.product_id, []),
            )


class ExpiringSoonView(AdminRequiredMixin, View):
    """Products whose earliest expiry date falls within the next N days.

    Complements ExpiredProductView (already past date) by giving staff time
    to discount, return, or rotate stock before it becomes waste.
    """
    template_name = "expiring_soon.html"
    WINDOWS = (30, 60, 90)

    def get(self, request):
        try:
            days = int(request.GET.get('days', '30'))
        except (TypeError, ValueError):
            days = 30
        if days not in self.WINDOWS:
            days = 30

        category_filter = request.GET.get('category', '')
        search_q = request.GET.get('q', '').strip()

        today = date.today()
        cutoff = today + timedelta(days=days)

        products_qs = Product.objects.filter(
            status=True,
            quantity_in_stock__gt=0,
            expiry_date__gte=today,
            expiry_date__lte=cutoff,
        )
        if category_filter:
            products_qs = products_qs.filter(category_id=category_filter)
        if search_q:
            products_qs = products_qs.filter(
                Q(name__icontains=search_q) | barcode_search_q(search_q)
            )

        products = list(
            products_qs.select_related('category').order_by('expiry_date', 'name')
        )

        total_units = 0
        total_value = Decimal('0.00')
        urgent_count = 0
        for p in products:
            p.days_left = (p.expiry_date - today).days
            p.value_at_risk = (p.price or Decimal('0.00')) * p.quantity_in_stock
            total_units += p.quantity_in_stock
            total_value += p.value_at_risk
            if p.days_left <= 7:
                urgent_count += 1

        paginator = Paginator(products, 50)
        page_obj = paginator.get_page(request.GET.get('page'))

        return render(request, self.template_name, {
            "products": page_obj,
            "page_obj": page_obj,
            "product_count": len(products),
            "urgent_count": urgent_count,
            "total_units": total_units,
            "total_value": total_value,
            "days": days,
            "windows": self.WINDOWS,
            "categories": Category.objects.all().order_by('name'),
            "category_filter": category_filter,
            "search_q": search_q,
        })


class LowStockTrendView(AdminRequiredMixin, View):
    template_name = "low_stock_trend.html"

    def get(self, request):
        category_filter = request.GET.get('category', '')
        include_inactive = request.GET.get('include_inactive', '') == '1'
        search_q = request.GET.get('q', '').strip()

        base_qs = Product.objects.filter(quantity_in_stock__gt=0)
        if not include_inactive:
            base_qs = base_qs.filter(status=True)

        products_qs = base_qs.annotate(
            _threshold=Coalesce(F('category__low_stock_threshold'), Value(3))
        ).filter(quantity_in_stock__lte=F('_threshold'))

        if category_filter:
            products_qs = products_qs.filter(category_id=category_filter)

        if search_q:
            products_qs = products_qs.filter(
                Q(name__icontains=search_q) | barcode_search_q(search_q)
            )

        products = list(
            products_qs.select_related('category').order_by("quantity_in_stock", "name")
        )

        today = date.today()
        thirty_days_ago = today - timedelta(days=30)

        recent_sales = dict(
            StockChange.objects.filter(
                product__in=[p.product_id for p in products],
                change_type='checkout',
                timestamp__gte=thirty_days_ago,
            ).values_list('product_id').annotate(total=Sum('quantity'))
        )

        critical_count = 0
        high_priority_count = 0

        for p in products:
            sold_30d = recent_sales.get(p.product_id, 0)
            avg_daily = sold_30d / 30
            if avg_daily > 0:
                p.days_remaining = round(p.quantity_in_stock / avg_daily, 1)
            else:
                p.days_remaining = None
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

        products.sort(key=lambda p: (p.days_remaining is None, p.days_remaining or 9999))

        paginator = Paginator(products, 50)
        page_obj = paginator.get_page(request.GET.get('page'))

        OutOfStockView._attach_reorder_predictions(page_obj)

        return render(request, self.template_name, {
            "products": page_obj,
            "product_count": len(products),
            "critical_count": critical_count,
            "high_priority_count": high_priority_count,
            "page_obj": page_obj,
            "categories": Category.objects.all().order_by('name'),
            "category_filter": category_filter,
            "include_inactive": include_inactive,
            "search_q": search_q,
        })


def save_cart(request, cart):
    request.session["cart"] = cart
    request.session.modified = True
    # Mirror the live cart onto the unsubmitted Order so it survives logout/login.
    oid = request.session.get("order_id")
    if oid:
        Order.objects.filter(order_id=oid, submitted=False, is_deleted=False).update(draft_cart=cart)

# Home view
@login_required
def home(request):
    from app import reporting

    # Today's scan activity (kept inline — dashboard-specific, not a report rollup)
    today_scans = StockChange.objects.filter(
        change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract'],
        timestamp__date=date.today(),
    )

    # "Connect Phone" QR — points a phone at this server's LAN address so it can
    # open the (mobile-responsive) app. The /connect-phone/ landing tags the
    # phone's session for a 2-hour login (see connect_phone / CustomLoginView).
    connect_base = _lan_base_url(request)
    connect_phone_url = connect_base + reverse('connect_phone')

    return render(request, 'home.html', {
        # Centralized rollups (stock health, sales, inventory value, best sellers,
        # expiry buckets, sales chart, reorder suggestions, dead stock, expiry calendar)
        **reporting.dashboard_kpis(),
        'recent_activity': reporting.recent_activity(),
        'categories': Category.objects.all().order_by('name'),
        'all_products': list(
            Product.objects.values(
                'product_id', 'name', 'price', 'quantity_in_stock',
                'item_number', 'barcode'
            )
        ),
        'change_types': StockChange._meta.get_field('change_type').choices,
        'scanned_today_count': today_scans.filter(change_type='checkin').count(),
        'products_updated_today': today_scans.values('product').distinct().count(),
        # Powers the shared Expired Log pull-out tab (partials/_expired_log_slider.html).
        'expired_logs': (StockChange.objects.filter(change_type='expired')
                         .select_related('product', 'user').order_by('-timestamp')[:50]),
        'connect_phone_url': connect_phone_url,
        'connect_phone_qr': _qr_data_uri(connect_phone_url),
        'connect_phone_base': connect_base,
    })


@login_required
def stock_log_api(request):
    """Canonical stock-movement log feed shared by the dashboard, the check-in
    dashboard and the check-in session page. Params: log_product, log_type,
    log_date_from, log_date_to, log_page, export=csv. Returns entries + today KPIs."""
    try:
        log_qs = StockChange.objects.select_related('product').order_by('-timestamp')
        log_product = request.GET.get('log_product', '').strip()
        log_type = request.GET.get('log_type', '')
        log_date_from = request.GET.get('log_date_from', '')
        log_date_to = request.GET.get('log_date_to', '')
        if log_product:
            log_qs = log_qs.filter(Q(product__name__icontains=log_product) | barcode_search_q(log_product, 'product__barcode'))
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
                writer.writerow([sc.timestamp.strftime('%Y-%m-%d %H:%M'), sc.display_name, sc.display_barcode, sc.get_change_type_display(), sc.quantity, sc.note or ''])
            return response
        # Paginate
        paginator = Paginator(log_qs, 50)
        page = paginator.get_page(request.GET.get('log_page', 1))
        today = date.today()
        today_all = StockChange.objects.filter(timestamp__date=today)
        entries = []
        for sc in page:
            try:
                positive = sc.change_type in ('checkin', 'error_add')
                badge_cls = 'checkin' if sc.change_type == 'checkin' else 'checkout' if sc.change_type == 'checkout' else 'expired' if sc.change_type == 'expired' else 'error' if sc.change_type in ('error_add', 'error_subtract') else 'other'
                entries.append({
                    'time': sc.timestamp.strftime('%b %d %H:%M'),
                    'name': sc.display_name,
                    'barcode': sc.display_barcode,
                    'action': sc.get_change_type_display(),
                    'badge_cls': badge_cls,
                    'qty': sc.quantity,
                    'positive': positive,
                    'note': sc.note or '—',
                })
            except Exception:
                continue
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
    except Exception as e:
        return JsonResponse({'error': str(e), 'entries': [], 'page': 1, 'num_pages': 1, 'has_prev': False, 'has_next': False, 'kpi': {'checkins': 0, 'sales': 0, 'adjustments': 0}})


class DailyReportView(AdminRequiredMixin, View):
    """On-screen end-of-day digest (sales, stock, expiry, dead stock, corrections)."""
    template_name = 'daily_report.html'

    def get(self, request):
        from app import reporting
        ignore_snacks = request.GET.get('ignore_snacks') == '1'
        digest = reporting.daily_digest(exclude_snacks=ignore_snacks)
        # Always archive the canonical FULL report (best-effort — never block the
        # page; the snacks toggle only affects what's shown/downloaded, not the
        # stored daily snapshot).
        try:
            reporting.archive_daily_report()
        except Exception:
            pass
        report_archives = DailyReportArchive.objects.all()  # newest first (Meta ordering)
        return render(request, self.template_name, {
            'digest': digest, 'today': digest['day'],
            'report_archives': report_archives,
            'ignore_snacks': ignore_snacks,
        })


class DailyReportPDFView(AdminRequiredMixin, View):
    """Downloadable PDF of today's end-of-day digest."""

    def get(self, request):
        from app import reporting
        ignore_snacks = request.GET.get('ignore_snacks') == '1'
        digest = reporting.daily_digest(exclude_snacks=ignore_snacks)
        pdf = reporting.build_daily_report_pdf(digest)
        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = f'inline; filename="daily_report_{digest["day"].strftime("%Y%m%d")}.pdf"'
        return response


class DailyReportArchivePDFView(AdminRequiredMixin, View):
    """Serve a stored (archived) daily-report PDF for viewing / printing."""

    def get(self, request, pk):
        archive = get_object_or_404(DailyReportArchive, pk=pk)
        response = HttpResponse(bytes(archive.pdf), content_type='application/pdf')
        disp = 'attachment' if request.GET.get('download') else 'inline'
        response['Content-Disposition'] = f'{disp}; filename="daily_report_{archive.report_date:%Y%m%d}.pdf"'
        return response


class DailyReportArchiveDeleteView(AdminRequiredMixin, View):
    """Delete one stored daily-report snapshot."""

    def post(self, request, pk):
        archive = DailyReportArchive.objects.filter(pk=pk).first()
        if archive:
            day = archive.report_date
            archive.delete()
            messages.success(request, f"Deleted saved report for {day:%b %d, %Y}.")
        return redirect('daily_report')


@login_required
def signup(request):
   # Admin function — staff, or a PU session unlocked with the passkey.
   if not has_admin_access(request):
       return redirect(f"{reverse('passkey_unlock')}?{urlencode({'next': request.get_full_path()})}")
   if request.method == 'POST':
       form = UserCreationForm(request.POST)
       if form.is_valid():
           new_user = form.save()
           UserAction.objects.create(user=new_user, action='create_account',
               target=new_user.username)
           messages.success(request, f"Account '{new_user.username}' has been created successfully!")
           return redirect('signup')
   else:
       form = UserCreationForm()
   return render(request, 'signup.html', {'form': form})


class PasskeyUnlockView(LoginRequiredMixin, View):
    """
    Lets a logged-in regular user (PU) unlock admin-only functions for their
    session by entering the admin passkey. Staff users are already unlocked and
    are bounced straight to their destination.
    """
    template_name = 'passkey_unlock.html'

    def _safe_next(self, request, raw):
        if raw and url_has_allowed_host_and_scheme(
            raw, allowed_hosts={request.get_host()}, require_https=request.is_secure()
        ):
            return raw
        return reverse('dashboard')

    def get(self, request):
        nxt = self._safe_next(request, request.GET.get('next'))
        if has_admin_access(request):
            return redirect(nxt)
        return render(request, self.template_name, {'next': nxt})

    # Failed-attempt throttle: django-axes only rate-limits the login page,
    # so the passkey form needs its own guard against brute-forcing.
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_SECONDS = 300

    def post(self, request):
        nxt = self._safe_next(request, request.POST.get('next'))
        now = time.time()
        locked_until = request.session.get('passkey_locked_until', 0)
        if now < locked_until:
            wait_min = max(1, int(locked_until - now + 59) // 60)
            messages.error(
                request,
                f"Too many incorrect attempts. Try again in {wait_min} minute(s)."
            )
            return render(request, self.template_name, {'next': nxt})
        entered = request.POST.get('passkey', '')
        expected = getattr(settings, 'ADMIN_PASSKEY', '') or ''
        if expected and hmac.compare_digest(str(entered), str(expected)):
            request.session.pop('passkey_failed_attempts', None)
            request.session.pop('passkey_locked_until', None)
            request.session[PASSKEY_SESSION_KEY] = time.time()
            UserAction.objects.create(
                user=request.user, action='passkey_unlock', target='admin access'
            )
            messages.success(request, "Admin access unlocked for this session.")
            return redirect(nxt)
        fails = request.session.get('passkey_failed_attempts', 0) + 1
        if fails >= self.MAX_FAILED_ATTEMPTS:
            request.session['passkey_locked_until'] = now + self.LOCKOUT_SECONDS
            request.session['passkey_failed_attempts'] = 0
            UserAction.objects.create(
                user=request.user, action='passkey_lockout', target='admin access',
                detail=f"{fails} failed passkey attempts",
            )
            messages.error(
                request,
                "Too many incorrect attempts. Passkey entry locked for 5 minutes."
            )
        else:
            request.session['passkey_failed_attempts'] = fails
            messages.error(request, "Incorrect passkey.")
        return render(request, self.template_name, {'next': nxt})


def _lan_base_url(request):
    """Base URL a phone on the same network should use to reach this server.

    The dashboard is often open on the shop computer at localhost, but the QR a
    phone scans must point at the machine's LAN IP. Prefer the configured LAN
    host (DJANGO_ALLOWED_HOSTS, set by configure_ip.py); fall back to whatever
    host the request came in on.
    """
    port = request.get_port() or '8000'
    lan_ip = next(
        (h.strip() for h in settings.ALLOWED_HOSTS
         if h.strip() and h.strip() not in ('localhost', '127.0.0.1', '0.0.0.0', '*')),
        None,
    )
    host = f"{lan_ip}:{port}" if lan_ip else request.get_host()
    return f"{request.scheme}://{host}"


def _qr_data_uri(text):
    """Render `text` as a QR code PNG data URI (self-contained — no network)."""
    img = qrcode.make(text, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def connect_phone(request):
    """Landing page hit by scanning the dashboard "Connect Phone" QR.

    Flags the (phone's) session so the upcoming login is treated as a phone —
    a 2-hour expiry (settings.PHONE_SESSION_AGE) and a phone-tagged UserSession
    — then sends the phone to the login page. The admin (GINA) account is never
    connected this way: the flag is ignored for staff at login time, and an
    already-signed-in admin scanning it just gets a note.
    """
    if request.user.is_authenticated:
        if request.user.is_staff:
            messages.info(
                request,
                'Phone connect is for staff (PU) accounts — the admin account '
                'stays on its main computer.'
            )
        else:
            # Already signed in as PU on this phone: convert the session in place.
            request.session.set_expiry(settings.PHONE_SESSION_AGE)
            UserSession.objects.filter(
                session_key=request.session.session_key
            ).update(device_type=UserSession.DEVICE_PHONE)
            messages.success(
                request,
                'Phone connected — you will stay signed in for 2 hours.'
            )
        return redirect('dashboard')

    request.session['connect_phone'] = True
    return redirect('login')


class CustomLoginView(LoginView):
    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            if request.user.is_staff:
                return redirect('create_order')
            return redirect('dashboard')
        return super().get(request, *args, **kwargs)

    def form_valid(self, form):
        user = form.get_user()
        ip = _get_client_ip(self.request)

        # The whole login runs in one transaction guarded by a global advisory
        # lock, so the active-count check and the new-session insert can't be
        # raced by a simultaneous login on another computer (see session_limits).
        with transaction.atomic():
            session_limits.take_global_lock()
            session_limits.prune_stale()          # reclaim dead computers' slots
            session_limits.drop_computer(user, ip)  # free this computer's own old slot

            if user.is_staff:
                # Admin (GINA) is a singleton AND never locked out: kick the
                # admin's other sessions, and if the cap is full make room.
                session_limits.evict_for_user(user)
                if session_limits.active_count() >= session_limits.global_max():
                    session_limits.evict_stalest()
            else:
                # Regular (PU): hard global cap — block the 6th computer.
                if session_limits.active_count() >= session_limits.global_max():
                    messages.error(
                        self.request,
                        f'Maximum {session_limits.global_max()} computers are already '
                        f'signed in. Ask someone to log out, or wait a few minutes.'
                    )
                    LoginAudit.objects.create(
                        user=user,
                        username=user.username,
                        ip_address=ip,
                        success=False,
                    )
                    return render(self.request, self.get_template_names()[0], {
                        'form': form,
                    })

            # A phone connecting via the dashboard QR flags its pre-login
            # session (see connect_phone). Honour it for PU accounts only —
            # the admin account is never a "phone".
            wants_phone = bool(self.request.session.get('connect_phone')) and not user.is_staff

            # Log in (mints the session_key) and register THIS session inside
            # the lock so the count and the insert are one atomic unit.
            response = super().form_valid(form)
            self.request.session.pop('connect_phone', None)
            if wants_phone:
                # Shorter 2-hour session for a phone (vs an 8-hour shift on a computer).
                self.request.session.set_expiry(settings.PHONE_SESSION_AGE)
            UserSession.objects.create(
                user=self.request.user,
                session_key=self.request.session.session_key,
                ip_address=ip,
                user_agent=self.request.META.get('HTTP_USER_AGENT', '')[:300],
                device_type=(UserSession.DEVICE_PHONE if wants_phone
                             else UserSession.DEVICE_COMPUTER),
            )
            LoginAudit.objects.create(
                user=self.request.user,
                username=self.request.user.username,
                ip_address=ip,
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
        return reverse('dashboard')

# Display all orders - Transaction page.
class OrderView(LoginRequiredMixin, View):
    template_name = 'order_view.html'

    def get(self, request):
        today = date.today()
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')
        status_filter = request.GET.get('status', '')
        source_filter = request.GET.get('source', '')  # '', 'all', 'pos', 'giveaway'

        # Pre-tax order total, with the 10% seniors discount applied when set.
        orders = (
            Order.objects
            .annotate(gross_total=Sum(F('details__price') * F('details__quantity')))
            .annotate(calc_total=Case(
                When(seniors_discount=True, then=F('gross_total') * Value(Decimal('0.90'))),
                default=F('gross_total'),
                output_field=DecimalField(max_digits=12, decimal_places=2),
            ))
            .order_by('-order_id')
        )

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

        current_order_id = request.session.get('order_id')

        # ── Unified transaction list: POS orders + terminal giveaways ──
        rows = []

        # "deleted" status shows only soft-deleted POS orders (the recycle bin);
        # every other status shows the live list and hides soft-deleted ones.
        show_deleted = status_filter == 'deleted'
        if source_filter in ('', 'all', 'pos'):
            # Soft-deleted orders stay in the aggregates/chart above (they still
            # count for reports) but are hidden from the visible transaction list.
            for o in orders.filter(is_deleted=show_deleted):
                rows.append({
                    'source': 'pos',
                    'id': o.order_id,
                    'date': o.order_date,
                    'total': o.calc_total or Decimal('0.00'),
                    'seniors_discount': o.seniors_discount,
                    'submitted': o.submitted,
                    'is_current': o.order_id == current_order_id,
                    'is_deleted': o.is_deleted,
                    'detail_url': reverse('order_detail', args=[o.order_id]),
                    'pdf_url': reverse('order_pdf', args=[o.order_id]),
                    'delete_url': None if o.is_deleted else reverse('delete_order', args=[o.order_id]),
                    'restore_url': reverse('restore_order', args=[o.order_id]) if o.is_deleted else None,
                })

        # Giveaways aren't soft-deletable, so they're excluded from the deleted
        # view; they're also excluded when filtering to "pending".
        if source_filter in ('', 'all', 'giveaway') and status_filter not in ('pending', 'deleted'):
            giveaways = CheckoutOrder.objects.filter(status=CheckoutOrder.STATUS_SUBMITTED)
            if date_from:
                parsed = parse_date(date_from)
                if parsed:
                    giveaways = giveaways.filter(submitted_at__date__gte=parsed)
            if date_to:
                parsed = parse_date(date_to)
                if parsed:
                    giveaways = giveaways.filter(submitted_at__date__lte=parsed)
            for g in giveaways:
                rows.append({
                    'source': 'giveaway',
                    'id': g.pk,
                    'date': g.submitted_at,
                    'total': g.total_price or Decimal('0.00'),
                    'submitted': True,
                    'is_current': False,
                    'detail_url': reverse('giveaway_detail', args=[g.pk]),
                    'pdf_url': None,
                    'delete_url': None,
                })

        # Newest first (date fields are populated for all rows here)
        rows.sort(key=lambda r: r['date'], reverse=True)

        # Pagination over the combined list
        paginator = Paginator(rows, 50)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, self.template_name, {
            'page_obj': page_obj,
            'current_order_id': current_order_id,
            'total_orders': submitted_orders.count(),
            'total_revenue': agg['total_revenue'] or Decimal('0.00'),
            'avg_order': agg['avg_order'] or Decimal('0.00'),
            'orders_today': orders_today_count,
            'daily_chart_data': daily_chart_data,
            'date_from': date_from,
            'date_to': date_to,
            'status_filter': status_filter,
            'source_filter': source_filter,
            'today': today,
        })


def build_order_transaction_context(order):
    order_details = order.details.select_related('product', 'product__category').all()

    total_items = 0
    total_units = 0
    total_price_before_tax = Decimal("0.00")
    total_tax = Decimal("0.00")
    total_cost = Decimal("0.00")
    taxable_subtotal = Decimal("0.00")
    nontaxable_subtotal = Decimal("0.00")

    # Local calendar date the order was placed — used to flag items that were
    # already past their expiry date when the sale happened.
    order_date_local = localtime(order.order_date).date() if order.order_date else None

    order_details_with_total = []
    expired_sold_count = 0
    for detail in order_details:
        line_total = detail.price * detail.quantity
        product = detail.product

        is_taxable = getattr(product, "taxable", False) if product else False
        item_tax = (line_total * TAX_RATE) if is_taxable else Decimal("0.00")

        if product and product.price_per_unit is not None:
            cost = product.price_per_unit * detail.quantity
            profit = line_total - cost
        else:
            cost = Decimal("0.00")
            profit = None

        # "Expired when sold": the earliest expiry had already passed on the order
        # date. Prefer the expiry snapshot captured at submit time (exact); fall back
        # to the product's current earliest expiry for older, un-snapshotted lines.
        expiry_date = detail.expiry_at_sale
        if expiry_date is None and product is not None:
            expiry_date = product.expiry_date
        expired_at_sale = bool(expiry_date and order_date_local and expiry_date < order_date_local)
        if expired_at_sale:
            expired_sold_count += 1

        order_details_with_total.append({
            'detail': detail,
            'total_price': line_total,
            'is_taxable': is_taxable,
            'item_tax': item_tax,
            'line_with_tax': line_total + item_tax,
            'cost': cost,
            'profit': profit,
            'product_deleted': product is None,
            'expired_at_sale': expired_at_sale,
            'expiry_date': expiry_date,
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

    # Seniors discount: 10% off the pre-tax subtotal. The discount is uniform, so
    # the taxable base drops 10% too — i.e. tax is reduced proportionally.
    seniors_discount = order.seniors_discount
    seniors_discount_amount = Decimal("0.00")
    if seniors_discount:
        seniors_discount_amount = (total_price_before_tax * Decimal("0.10")).quantize(Decimal("0.01"))
        total_tax = (total_tax * Decimal("0.90")).quantize(Decimal("0.01"))

    total_price_after_tax = (total_price_before_tax - seniors_discount_amount) + total_tax
    total_profit = (total_price_before_tax - seniors_discount_amount - total_cost) if total_cost > 0 else None
    margin_pct = (
        (total_profit / total_price_before_tax) * 100
        if total_profit is not None and total_price_before_tax > 0
        else None
    )

    return {
        'order': order,
        'order_details_with_total': order_details_with_total,
        'total_price_before_tax': total_price_before_tax,
        'total_price_after_tax': total_price_after_tax,
        'total_tax': total_tax,
        'seniors_discount': seniors_discount,
        'seniors_discount_amount': seniors_discount_amount,
        'total_items': total_items,
        'total_units': total_units,
        'taxable_subtotal': taxable_subtotal,
        'nontaxable_subtotal': nontaxable_subtotal,
        'total_cost': total_cost,
        'total_profit': total_profit,
        'margin_pct': margin_pct,
        'expired_sold_count': expired_sold_count,
        'any_expired_sold': expired_sold_count > 0,
    }


class OrderDetailView(LoginRequiredMixin, View):
    template_name = 'order_detail.html'

    def get(self, request, order_id):
        order = get_object_or_404(Order, order_id=order_id)
        context = build_order_transaction_context(order)

        # Navigation: previous and next order IDs
        prev_order = Order.objects.filter(order_id__lt=order_id).order_by('-order_id').values_list('order_id', flat=True).first()
        next_order = Order.objects.filter(order_id__gt=order_id).order_by('order_id').values_list('order_id', flat=True).first()

        context.update({
            'prev_order': prev_order,
            'next_order': next_order,
        })

        return render(request, self.template_name, context)


class OrderPDFView(LoginRequiredMixin, View):
    """Generate a professional PDF transaction report for a single order."""

    def get(self, request, order_id):
        from reportlab.lib.colors import HexColor

        order = get_object_or_404(Order, order_id=order_id)
        ctx = build_order_transaction_context(order)

        # Build flat items list for PDF
        items = []
        for item in ctx['order_details_with_total']:
            d = item['detail']
            items.append({
                'name': d.display_name,
                'barcode': d.display_barcode,
                'qty': d.quantity,
                'price': d.price,
                'line_total': item['total_price'],
                'taxable': item['is_taxable'],
            })

        subtotal = ctx['total_price_before_tax']
        total_tax = ctx['total_tax']
        grand_total = ctx['total_price_after_tax']
        total_items = ctx['total_items']
        total_units = ctx['total_units']
        seniors_discount = ctx['seniors_discount']
        seniors_discount_amount = ctx['seniors_discount_amount']

        # ── PDF setup ──
        buffer = io.BytesIO()
        PAGE_W, PAGE_H = letter
        c = canvas.Canvas(buffer, pagesize=letter)
        M = 50  # margin

        # Colours
        BRAND = HexColor("#4f46e5")
        DARK = HexColor("#1e293b")
        MUTED = HexColor("#64748b")
        LIGHT = HexColor("#f1f5f9")
        LINE = HexColor("#e2e8f0")
        ALT = HexColor("#f8fafc")
        WHITE = HexColor("#ffffff")
        SUCCESS = HexColor("#059669")

        content_w = PAGE_W - 2 * M
        page_num = [1]

        # ── Reusable helpers ──
        def hr(yy, color=LINE, width=0.5):
            c.setStrokeColor(color)
            c.setLineWidth(width)
            c.line(M, yy, PAGE_W - M, yy)

        def draw_footer(yy):
            hr(yy, LINE, 0.5)
            yy -= 14
            c.setFont("Helvetica", 7)
            c.setFillColor(MUTED)
            c.drawString(M, yy, f"Generated: {datetime.now().strftime('%B %d, %Y  %I:%M %p')}")
            c.drawRightString(PAGE_W - M, yy, f"Page {page_num[0]}")
            yy -= 16
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(BRAND)
            c.drawCentredString(PAGE_W / 2, yy, "MPCP  ·  Meadowvale Professional Center Pharmacy")
            c.setFont("Helvetica", 7)
            c.setFillColor(MUTED)
            c.drawCentredString(PAGE_W / 2, yy - 12, "Thank you for your business")

        def new_page():
            draw_footer(M + 36)
            c.showPage()
            page_num[0] += 1
            return PAGE_H - M

        def check_space(yy, needed):
            if yy < M + 60 + needed:
                return new_page()
            return yy

        # ════════════════════════════════════════
        # PAGE 1 — HEADER
        # ════════════════════════════════════════
        y = PAGE_H - M

        # Brand bar
        c.setFillColor(BRAND)
        c.rect(M, y - 6, content_w, 36, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(M + 12, y + 4, "MPCP")
        c.setFont("Helvetica", 8)
        c.drawString(M + 70, y + 6, "Meadowvale Professional Center Pharmacy")
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(PAGE_W - M - 12, y + 4, "TRANSACTION REPORT")
        y -= 42

        # ── Order # headline ──
        y -= 8
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(M, y, f"Order #{order.order_id}")
        y -= 28

        # ── Info box ──
        box_h = 58
        c.setFillColor(LIGHT)
        c.roundRect(M, y - box_h, content_w, box_h, 6, fill=1, stroke=0)
        c.setStrokeColor(BRAND)
        c.setLineWidth(2)
        c.line(M, y - box_h, M, y)  # left accent

        info_y = y - 16
        col1_x = M + 14
        col2_x = M + content_w / 2

        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawString(col1_x, info_y, "DATE")
        c.drawString(col2_x, info_y, "STATUS")
        info_y -= 14
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(DARK)
        c.drawString(col1_x, info_y, order.order_date.strftime("%B %d, %Y  %I:%M %p"))
        status_text = "Completed" if order.submitted else "Pending"
        status_color = SUCCESS if order.submitted else HexColor("#d97706")
        c.setFillColor(status_color)
        c.drawString(col2_x, info_y, status_text)
        info_y -= 18
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawString(col1_x, info_y, f"{total_items} product{'s' if total_items != 1 else ''}  ·  {total_units} unit{'s' if total_units != 1 else ''}")

        y -= box_h + 20

        # ── Section: ORDER CONTENTS ──
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(M, y, "ORDER CONTENTS")
        y -= 6
        hr(y, DARK, 1)
        y -= 18

        # ── Table header ──
        row_h = 20
        col_num = M
        col_prod = M + 24
        col_qty = M + content_w * 0.58
        col_price = M + content_w * 0.73
        col_total = PAGE_W - M

        c.setFillColor(LIGHT)
        c.rect(M, y - 5, content_w, row_h, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(col_num + 4, y + 2, "#")
        c.drawString(col_prod, y + 2, "PRODUCT")
        c.drawRightString(col_qty, y + 2, "QTY")
        c.drawRightString(col_price, y + 2, "PRICE")
        c.drawRightString(col_total, y + 2, "TOTAL")
        y -= row_h + 2

        # ── Table rows ──
        for idx, item in enumerate(items, 1):
            # Each item needs ~32px (row + barcode line)
            y = check_space(y, 34)

            # Alternating background
            if idx % 2 == 0:
                c.setFillColor(ALT)
                c.rect(M, y - 5, content_w, row_h, fill=1, stroke=0)

            # Row number
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 8)
            c.drawString(col_num + 4, y + 2, str(idx))

            # Product name (truncate if needed)
            c.setFillColor(DARK)
            c.setFont("Helvetica-Bold", 9)
            name = item['name']
            max_w = col_qty - col_prod - 30
            if stringWidth(name, "Helvetica-Bold", 9) > max_w:
                while stringWidth(name + "...", "Helvetica-Bold", 9) > max_w and len(name) > 1:
                    name = name[:-1]
                name += "..."
            c.drawString(col_prod, y + 2, name)

            # Qty, Price, Total
            c.setFont("Helvetica", 9)
            c.setFillColor(DARK)
            c.drawRightString(col_qty, y + 2, str(item['qty']))
            c.drawRightString(col_price, y + 2, f"${item['price']:.2f}")

            # Total with tax marker
            c.setFont("Helvetica-Bold", 9)
            total_str = f"${item['line_total']:.2f}"
            if item['taxable']:
                total_str += " T"
            c.drawRightString(col_total, y + 2, total_str)

            y -= row_h

            # Barcode line (smaller, muted, indented under product name)
            barcode = item.get('barcode', '')
            if barcode:
                c.setFont("Helvetica", 6.5)
                c.setFillColor(MUTED)
                c.drawString(col_prod, y + 4, f"Barcode: {barcode}")
                y -= 12

        # Bottom line of table
        y -= 2
        hr(y, DARK, 1)
        y -= 22

        # ── Financial summary ──
        y = check_space(y, 80)
        sum_lbl = PAGE_W - M - 170
        sum_val = PAGE_W - M

        def draw_summary_line(label, value, bold=False, color=DARK, size=10):
            nonlocal y
            font = "Helvetica-Bold" if bold else "Helvetica"
            c.setFont(font, size)
            c.setFillColor(MUTED)
            c.drawString(sum_lbl, y, label)
            c.setFillColor(color)
            c.drawRightString(sum_val, y, value)
            y -= 18

        draw_summary_line("Subtotal", f"${subtotal:.2f}")
        if seniors_discount:
            draw_summary_line("Seniors Discount (-10%)", f"-${seniors_discount_amount:.2f}", color=SUCCESS)
        draw_summary_line("Tax (13%)", f"${total_tax:.2f}")

        # Divider
        c.setStrokeColor(DARK)
        c.setLineWidth(1.5)
        c.line(sum_lbl, y + 8, sum_val, y + 8)
        y -= 6

        draw_summary_line("TOTAL", f"${grand_total:.2f}", bold=True, color=BRAND, size=14)

        # ── Footer ──
        draw_footer(M + 36)

        c.save()
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="MPCP-Order-{order_id}.pdf"'
        return response


class ExportAllOrdersPDFView(AdminRequiredMixin, View):
    """Generate a multi-order PDF report for all (filtered) transactions."""

    def get(self, request):
        from reportlab.lib.colors import HexColor

        # ── Apply same filters as ExportTransactionsCSVView ──
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')
        status_filter = request.GET.get('status', '')

        orders_qs = Order.objects.prefetch_related(
            'details', 'details__product', 'details__product__category'
        ).order_by('-order_date')

        if date_from:
            parsed = parse_date(date_from)
            if parsed:
                orders_qs = orders_qs.filter(order_date__date__gte=parsed)
        if date_to:
            parsed = parse_date(date_to)
            if parsed:
                orders_qs = orders_qs.filter(order_date__date__lte=parsed)
        if status_filter == 'completed':
            orders_qs = orders_qs.filter(submitted=True)
        elif status_filter == 'pending':
            orders_qs = orders_qs.filter(submitted=False)
        else:
            orders_qs = orders_qs.filter(submitted=True)

        orders = list(orders_qs)

        if not orders:
            messages.info(request, "No orders match the current filters.")
            return redirect('order_view')

        # ── PDF setup ──
        buffer = io.BytesIO()
        PAGE_W, PAGE_H = letter
        c = canvas.Canvas(buffer, pagesize=letter)
        M = 50
        content_w = PAGE_W - 2 * M

        BRAND = HexColor("#4f46e5")
        DARK = HexColor("#1e293b")
        MUTED = HexColor("#64748b")
        LIGHT = HexColor("#f1f5f9")
        LINE = HexColor("#e2e8f0")
        WHITE = HexColor("#ffffff")
        SUCCESS = HexColor("#059669")

        page_num = [1]
        generated = datetime.now().strftime('%B %d, %Y  %I:%M %p')

        def draw_page_footer():
            c.setStrokeColor(LINE)
            c.setLineWidth(0.5)
            c.line(M, M + 30, PAGE_W - M, M + 30)
            c.setFont("Helvetica", 7)
            c.setFillColor(MUTED)
            c.drawString(M, M + 18, f"Generated: {generated}")
            c.drawRightString(PAGE_W - M, M + 18, f"Page {page_num[0]}")
            c.setFont("Helvetica-Bold", 7)
            c.setFillColor(BRAND)
            c.drawCentredString(PAGE_W / 2, M + 6, "MPCP  ·  Meadowvale Professional Center Pharmacy")

        def new_page():
            draw_page_footer()
            c.showPage()
            page_num[0] += 1
            # Mini header on continuation pages
            yy = PAGE_H - M
            c.setFillColor(BRAND)
            c.rect(M, yy - 6, content_w, 24, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(M + 8, yy, "MPCP")
            c.setFont("Helvetica", 7)
            c.drawString(M + 50, yy + 1, "Transaction Report")
            c.drawRightString(PAGE_W - M - 8, yy, f"Page {page_num[0]}")
            return yy - 36

        def check_space(yy, needed):
            if yy < M + 50 + needed:
                return new_page()
            return yy

        # ════════════════════════════════════════
        # COVER / HEADER
        # ════════════════════════════════════════
        y = PAGE_H - M

        # Brand bar
        c.setFillColor(BRAND)
        c.rect(M, y - 6, content_w, 36, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(M + 12, y + 4, "MPCP")
        c.setFont("Helvetica", 8)
        c.drawString(M + 70, y + 6, "Meadowvale Professional Center Pharmacy")
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(PAGE_W - M - 12, y + 4, "TRANSACTION REPORT")
        y -= 48

        # Summary stats
        grand_revenue = sum(
            sum(d.price * d.quantity for d in o.details.all()) for o in orders
        )
        date_range_str = ""
        if date_from:
            date_range_str += f"From: {date_from}"
        if date_to:
            date_range_str += f"  To: {date_to}"
        if not date_range_str:
            date_range_str = "All dates"

        c.setFillColor(LIGHT)
        c.roundRect(M, y - 50, content_w, 50, 6, fill=1, stroke=0)
        c.setStrokeColor(BRAND)
        c.setLineWidth(2)
        c.line(M, y - 50, M, y)

        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(DARK)
        c.drawString(M + 14, y - 22, f"{len(orders)} Orders")
        c.setFont("Helvetica", 10)
        c.setFillColor(MUTED)
        c.drawString(M + 14, y - 38, date_range_str)

        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(BRAND)
        c.drawRightString(PAGE_W - M - 14, y - 22, f"${grand_revenue:.2f}")
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawRightString(PAGE_W - M - 14, y - 36, "total revenue (before tax)")

        y -= 68

        # ════════════════════════════════════════
        # ORDER BLOCKS
        # ════════════════════════════════════════
        for order in orders:
            details = order.details.all()
            if not details.exists():
                continue

            # Calculate how much space this order needs
            detail_count = details.count()
            needed = 60 + (detail_count * 16) + 60  # header + rows + summary
            y = check_space(y, min(needed, 200))  # at least start if it's huge

            # ── Order header bar ──
            c.setFillColor(DARK)
            c.rect(M, y - 4, content_w, 22, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(M + 8, y + 2, f"Order #{order.order_id}")
            c.setFont("Helvetica", 8)
            c.drawString(M + 100, y + 3, order.order_date.strftime("%b %d, %Y  %I:%M %p"))
            status = "Completed" if order.submitted else "Pending"
            c.drawRightString(PAGE_W - M - 8, y + 3, status)
            y -= 30

            # ── Column headers ──
            col_prod = M + 6
            col_qty = M + content_w * 0.58
            col_price = M + content_w * 0.75
            col_total = PAGE_W - M - 6

            c.setFont("Helvetica-Bold", 7)
            c.setFillColor(MUTED)
            c.drawString(col_prod, y + 2, "PRODUCT")
            c.drawRightString(col_qty, y + 2, "QTY")
            c.drawRightString(col_price, y + 2, "PRICE")
            c.drawRightString(col_total, y + 2, "TOTAL")
            y -= 14

            # ── Item rows ──
            order_subtotal = Decimal("0.00")
            order_tax = Decimal("0.00")
            for d in details:
                y = check_space(y, 16)
                line_total = d.price * d.quantity
                order_subtotal += line_total
                product = d.product
                is_taxable = getattr(product, "taxable", False) if product else False
                if is_taxable:
                    order_tax += line_total * TAX_RATE

                c.setFont("Helvetica", 8)
                c.setFillColor(DARK)
                name = d.display_name
                max_w = col_qty - col_prod - 20
                if stringWidth(name, "Helvetica", 8) > max_w:
                    while stringWidth(name + "...", "Helvetica", 8) > max_w and len(name) > 1:
                        name = name[:-1]
                    name += "..."
                c.drawString(col_prod, y + 2, name)
                c.drawRightString(col_qty, y + 2, str(d.quantity))
                c.drawRightString(col_price, y + 2, f"${d.price:.2f}")
                c.setFont("Helvetica-Bold", 8)
                c.drawRightString(col_total, y + 2, f"${line_total:.2f}")
                y -= 16

            # ── Order totals ──
            y -= 2
            c.setStrokeColor(LINE)
            c.setLineWidth(0.5)
            c.line(col_price - 20, y + 8, col_total, y + 8)

            # Seniors discount (10% off pre-tax) — reduces the taxable base too.
            seniors_amt = Decimal("0.00")
            if order.seniors_discount:
                seniors_amt = (order_subtotal * Decimal("0.10")).quantize(Decimal("0.01"))
                order_tax = (order_tax * Decimal("0.90")).quantize(Decimal("0.01"))
            order_grand = (order_subtotal - seniors_amt) + order_tax

            c.setFont("Helvetica", 8)
            c.setFillColor(MUTED)
            c.drawString(col_price - 20, y - 2, "Subtotal:")
            c.setFillColor(DARK)
            c.drawRightString(col_total, y - 2, f"${order_subtotal:.2f}")
            y -= 14

            if order.seniors_discount:
                c.setFillColor(MUTED)
                c.drawString(col_price - 20, y - 2, "Seniors Discount (-10%):")
                c.setFillColor(SUCCESS)
                c.drawRightString(col_total, y - 2, f"-${seniors_amt:.2f}")
                y -= 14

            c.setFillColor(MUTED)
            c.drawString(col_price - 20, y - 2, "Tax:")
            c.setFillColor(DARK)
            c.drawRightString(col_total, y - 2, f"${order_tax:.2f}")
            y -= 14

            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(BRAND)
            c.drawString(col_price - 20, y - 2, "TOTAL:")
            c.drawRightString(col_total, y - 2, f"${order_grand:.2f}")
            y -= 22

            # Divider between orders
            c.setStrokeColor(LINE)
            c.setLineWidth(0.5)
            c.line(M, y, PAGE_W - M, y)
            y -= 16

        # Final page footer
        draw_page_footer()

        c.save()
        buffer.seek(0)
        filename = f"MPCP-Transactions-{datetime.now().strftime('%Y%m%d')}.pdf"
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
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
        # "Ignore snacks" toggle — excludes the Snacks category from every series
        # below (they all derive from base_qs).
        ignore_snacks = request.GET.get('ignore_snacks') == '1'
        if ignore_snacks:
            from app import reporting
            base_qs = base_qs.exclude(product__category__name__iexact=reporting.SNACKS_CATEGORY_NAME)

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
            'ignore_snacks':  ignore_snacks,
        })


# change
class AddProductByIdView(AdminRequiredMixin, View):
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
    

class CreateOrderView(AdminRequiredMixin, View):
    template_name = "order_form.html"

    def get_order(self, request, create_if_missing=False):
        """The current draft order, or None.

        Lazy: an Order row is NOT created just by opening the purchase page — it
        is created only once there's something to put in it (an item was scanned)
        or when a caller explicitly needs one. This stops empty $0 drafts from
        piling up. Resume only picks an in-progress draft (one with a saved cart),
        so a blank order is never resurrected.
        """
        oid = request.session.get("order_id")
        order = Order.objects.filter(order_id=oid, submitted=False, is_deleted=False).first() if oid else None

        if order is None:
            order = (
                Order.objects.filter(user=request.user, submitted=False, is_deleted=False)
                .exclude(draft_cart={})
                .order_by("-order_date").first()
            )
            if order is not None:
                request.session["order_id"] = order.order_id

        session_cart = request.session.get("cart") or {}

        if order is None and (create_if_missing or session_cart):
            order = Order.objects.create(
                total_price=Decimal("0.00"), user=request.user, draft_cart=dict(session_cart)
            )
            request.session["order_id"] = order.order_id

        if order is None:
            return None

        # Sync the durable draft <-> the live session cart.
        if not session_cart and order.draft_cart:
            # Fresh session (e.g. just logged in) — reload the saved cart.
            request.session["cart"] = dict(order.draft_cart)
            request.session.modified = True
        elif session_cart and session_cart != order.draft_cart:
            # Keep the durable copy in step with the live cart.
            order.draft_cart = session_cart
            order.save(update_fields=["draft_cart"])
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
            save_cart(request, cart)

        # Seniors discount: 10% off the pre-tax subtotal, then tax the reduced base.
        seniors_discount = bool(order and order.seniors_discount)
        seniors_discount_amount = Decimal("0.00")
        taxable_base = total_price_before_tax
        if seniors_discount:
            seniors_discount_amount = (total_price_before_tax * Decimal("0.10")).quantize(Decimal("0.01"))
            taxable_base = total_price_before_tax - seniors_discount_amount

        total_price_after_tax = taxable_base * (1 + TAX_RATE)
        tax_amount = total_price_after_tax - taxable_base

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
            "seniors_discount": seniors_discount,
            "seniors_discount_amount": seniors_discount_amount,
            "name_query": name_query,
            "search_results": search_results,
            "all_products": all_products,
            "change_types": StockChange._meta.get_field('change_type').choices,
        })

    # ─────────────────────────────
    # POST — SCAN BARCODE (SESSION)
    # ─────────────────────────────
    def post(self, request, *args, **kwargs):
        # Toggle the seniors discount (10% off pre-tax) on the current draft order.
        if request.POST.get("action") == "toggle_seniors_discount":
            order = self.get_order(request)
            if order is not None:
                order.seniors_discount = not order.seniors_discount
                order.save(update_fields=["seniors_discount"])
            return redirect("create_order")

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
            save_cart(request, cart)

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



class SubmitOrderView(AdminRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        cart = request.session.get("cart")

        if not cart:
            messages.error(request, "Cannot submit an empty order.", extra_tags="order")
            return redirect("create_order")

        order = get_object_or_404(
            Order,
            order_id=request.session.get("order_id"),
            submitted=False,
            is_deleted=False,
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
                    expiry_at_sale=product.expiry_date,
                )

                # ✅ Decrement stock (floor at 0 — never go negative)
                if requested > 0:
                    deduct = min(requested, available)
                    if deduct > 0:
                        product.quantity_in_stock = available - deduct
                        product.save(update_fields=["quantity_in_stock"])

                        # Record only what was actually sold from stock
                        record_stock_change(
                            product=product,
                            qty=deduct,
                            change_type="checkout",
                            note=f"Order {order.order_id} submission",
                            user=request.user,
                        )

                    # Record the unfulfilled portion as a missed sale (stockout)
                    shortfall = requested - deduct
                    if shortfall > 0:
                        record_stock_change(
                            product=product,
                            qty=shortfall,
                            change_type="checkout_unfulfilled",
                            note=f"Order {order.order_id} — short {shortfall} (stockout)",
                            user=request.user,
                        )

                if requested > available:
                    unfulfilled_lines.append(f"{product.name} (short {requested - available})")

                # Optional analytics
                if requested > 0:
                    rp, _ = RecentlyPurchasedProduct.objects.get_or_create(product=product)
                    rp.quantity = (rp.quantity or 0) + requested
                    rp.save(update_fields=["quantity"])

            # ✅ Finalize order — clear the durable draft so it carries no stale cart
            order.submitted = True
            order.draft_cart = {}
            order.save(update_fields=["submitted", "draft_cart"])

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
    if not has_admin_access(request):
        return redirect(f"{reverse('passkey_unlock')}?{urlencode({'next': request.get_full_path()})}")
    cart = request.session.get("cart", {})
    pid = str(product_id)  # Use item_id here as well

    if pid not in cart:
        messages.warning(request, "Item not found in cart.")
        return redirect("create_order")

    if cart[pid]["quantity"] > 1:
        cart[pid]["quantity"] -= 1
    else:
        del cart[pid]

    save_cart(request, cart)

    # Last item removed → drop the now-empty draft so no empty order is left open.
    if not cart:
        oid = request.session.pop("order_id", None)
        request.session.modified = True
        if oid:
            Order.objects.filter(order_id=oid, submitted=False, is_deleted=False).delete()

    messages.success(request, "1 unit removed from the order.")
    return redirect("create_order")


# View for order success page
class OrderSuccessView(AdminRequiredMixin, View):
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

        # Seniors discount: 10% off pre-tax subtotal; tax drops proportionally.
        seniors_discount = order.seniors_discount
        seniors_discount_amount = Decimal('0.00')
        if seniors_discount:
            seniors_discount_amount = (subtotal * Decimal('0.10')).quantize(Decimal('0.01'))
            total_tax = (total_tax * Decimal('0.90')).quantize(Decimal('0.01'))

        return render(request, self.template_name, {
            'order': order,
            'items': items,
            'subtotal': subtotal,
            'total_tax': total_tax,
            'grand_total': (subtotal - seniors_discount_amount) + total_tax,
            'seniors_discount': seniors_discount,
            'seniors_discount_amount': seniors_discount_amount,
            'item_count': details.count(),
        })


# ══════════════════════════════════════════════════════════════════════════
# PU CHECKOUT — durable, per-user checkout (separate from admin Orders)
# ══════════════════════════════════════════════════════════════════════════

def get_current_checkout(request):
    """The draft checkout this browser session (terminal) is currently working on.

    Multiple drafts per user are allowed — each terminal tracks its own active
    session via request.session['checkout_id']. Returns None if this terminal
    has no current draft (caller should send the user to the chooser).
    """
    if not request.session.session_key:
        request.session.save()
    cid = request.session.get('checkout_id')
    if cid:
        # Shared checkout: any account can resume the draft its terminal points to.
        return CheckoutOrder.objects.filter(
            pk=cid, status=CheckoutOrder.STATUS_DRAFT
        ).first()
    return None


def other_live_sessions(request):
    """Other still-registered sessions for this user (other computers signed in)."""
    return list(
        UserSession.objects.filter(user=request.user)
        .exclude(session_key=request.session.session_key)
        .order_by('-last_activity')
    )


class CheckoutChooserView(UserRequiredMixin, View):
    """Modal chooser shown when a PU user clicks Checkout: active sessions,
    history, Start New, and Continue."""
    template_name = "checkout_chooser.html"

    def get(self, request, *args, **kwargs):
        if not request.session.session_key:
            request.session.save()
        my_key = request.session.session_key
        # Shared checkout dashboard: show every account's drafts, not just this user's.
        active_sessions = list(
            CheckoutOrder.objects.filter(
                status=CheckoutOrder.STATUS_DRAFT
            ).select_related('user').order_by('-updated_at')
        )
        # Which computer currently holds each draft: a draft's active_session_key
        # is "live" only if that session is still signed in (has a UserSession).
        live = {
            us.session_key: us
            for us in UserSession.objects.all()
        }
        for s in active_sessions:
            key = s.active_session_key
            holder = live.get(key) if key else None
            if key and key == my_key:
                s.holder_state = 'this'          # this computer is on it
                s.holder_label = ''
                s.holder_browser = ''
            elif holder:
                s.holder_state = 'other'         # another live computer is on it
                s.holder_label = holder.ip_address or 'another computer'
                s.holder_browser = simplify_ua(holder.user_agent)
            else:
                s.holder_state = 'idle'          # not currently held
                s.holder_label = ''
                s.holder_browser = ''
        # ── Active purchases (in-progress order drafts) ──────────────────────
        # A Purchase is a recorded sale (separate from a no-charge Checkout). The
        # purchase page is one-computer-locked via PagePresence, so surface any
        # in-progress purchase here too — and which computer is currently on it.
        from django.contrib.sessions.models import Session
        purchase_path = reverse('create_order')
        ph = PagePresence.objects.filter(page=purchase_path).first()
        purchase_holder = ph if (ph and is_fresh(ph)) else None
        held_order_id = None
        if purchase_holder:
            sess = Session.objects.filter(session_key=purchase_holder.session_key).first()
            if sess:
                held_order_id = sess.get_decoded().get('order_id')

        active_purchases = list(
            Order.objects.filter(submitted=False, is_deleted=False)
            .exclude(draft_cart={})
            .select_related('user').order_by('-order_date')
        )
        for o in active_purchases:
            o.item_count = sum(
                int(v.get('quantity', 0)) if isinstance(v, dict) else int(v or 0)
                for v in (o.draft_cart or {}).values()
            )
            o.is_mine = (o.user_id == request.user.id)
            if purchase_holder and held_order_id == o.order_id:
                if purchase_holder.session_key == my_key:
                    o.holder_state = 'this'           # open on this computer
                    o.holder_label = ''
                    o.holder_browser = ''
                else:
                    o.holder_state = 'other'          # open on another live computer
                    o.holder_label = purchase_holder.ip_address or 'another computer'
                    o.holder_browser = simplify_ua(purchase_holder.user_agent)
            else:
                o.holder_state = 'idle'               # a saved draft, not currently open
                o.holder_label = ''
                o.holder_browser = ''
            # Resume only the current user's own draft, and never one another
            # computer is actively holding.
            o.can_continue = o.is_mine and o.holder_state != 'other'

        history_qs = CheckoutOrder.objects.filter(
            status=CheckoutOrder.STATUS_SUBMITTED
        ).select_related('user').order_by('-submitted_at')
        history_count = history_qs.count()
        history = list(history_qs[:50])
        return render(request, self.template_name, {
            'active_sessions': active_sessions,
            'active_purchases': active_purchases,
            'history': history,
            'history_count': history_count,
            'current_id': request.session.get('checkout_id'),
        })


class CheckoutContinueView(UserRequiredMixin, View):
    """Make an existing draft the current session for this terminal, then open the cart."""
    def post(self, request, checkout_id):
        co = get_object_or_404(
            CheckoutOrder, pk=checkout_id,
            status=CheckoutOrder.STATUS_DRAFT,
        )
        if not request.session.session_key:
            request.session.save()
        co.active_session_key = request.session.session_key or ''
        co.save(update_fields=['active_session_key', 'updated_at'])
        request.session['checkout_id'] = co.pk
        return redirect('checkout_cart')


class CheckoutView(UserRequiredMixin, View):
    template_name = "checkout.html"

    def get(self, request, *args, **kwargs):
        checkout = get_current_checkout(request)
        if not checkout:
            return redirect('checkout')  # no current session → chooser
        session_key = request.session.session_key
        others = other_live_sessions(request)

        has_items = checkout.items.exists()

        # Concurrency guard: only warn when a DIFFERENT, still-live session owns a
        # non-empty draft. Otherwise auto-resume (claim ownership) so the checkout
        # survives session expiry without losing items.
        show_conflict = bool(
            has_items
            and checkout.active_session_key
            and checkout.active_session_key != session_key
            and UserSession.objects.filter(session_key=checkout.active_session_key).exists()
        )
        if not show_conflict and checkout.active_session_key != session_key:
            checkout.active_session_key = session_key
            checkout.save(update_fields=["active_session_key", "updated_at"])

        order_items = []
        subtotal = Decimal("0.00")
        tax_total = Decimal("0.00")
        for item in checkout.items.select_related("product").all():
            product = item.product
            qty = item.quantity
            line = item.price * qty
            subtotal += line
            if item.taxable:
                tax_total += line * TAX_RATE

            # Validation hints (suppressed while the conflict modal is up)
            if not show_conflict:
                if product is None:
                    messages.info(
                        request,
                        f"Note: '{item.product_name}' is no longer in the catalog.",
                        extra_tags="order",
                    )
                else:
                    if not product.status:
                        messages.warning(
                            request, f"⚠️ '{product.name}' is currently inactive.",
                            extra_tags="order",
                        )
                    if product.expiry_date and product.expiry_date < now().date():
                        messages.info(
                            request,
                            f"Note: '{product.name}' is expired (Expiry: {product.expiry_date}).",
                            extra_tags="order",
                        )
                    if qty > (product.quantity_in_stock or 0):
                        messages.info(
                            request,
                            f"'{product.name}' quantity ({qty}) exceeds stock ({product.quantity_in_stock}).",
                            extra_tags="order",
                        )

            order_items.append({
                "item": item,
                "product": product,
                "quantity": qty,
                "subtotal": line,
            })

        total_after_tax = subtotal + tax_total

        name_query = request.GET.get("name_query", "")
        search_results = (
            Product.objects.filter(name__icontains=name_query).order_by("name")
            if name_query else []
        )
        all_products = list(Product.objects.values(
            "product_id", "name", "price", "quantity_in_stock",
            "item_number", "barcode", "expiry_date", "status",
        ))

        return render(request, self.template_name, {
            "checkout": checkout,
            "form": BarcodeForm(),
            "order_items": order_items,
            "total_price_before_tax": subtotal,
            "tax_amount": tax_total,
            "total_price_after_tax": total_after_tax,
            "name_query": name_query,
            "search_results": search_results,
            "all_products": all_products,
            "other_sessions": others,
            "show_active_conflict": show_conflict,
        })

    # POST — scan barcode → add to the DB-backed checkout
    def post(self, request, *args, **kwargs):
        form = BarcodeForm(request.POST)
        if not form.is_valid():
            messages.error(request, "Enter a valid barcode and quantity.", extra_tags="order")
            return redirect("checkout_cart")

        barcode = form.cleaned_data["barcode"].strip()
        requested_quantity = int(form.cleaned_data.get("quantity") or 1)
        override_expiry = request.POST.get("override_expiry") == "1"
        override_inactive = request.POST.get("override_inactive") == "1"

        product = find_product_by_barcode(barcode)
        if not product:
            messages.error(request, f"No product found with barcode '{barcode}'.", extra_tags="order")
            return redirect("checkout_cart")

        checkout = get_current_checkout(request)
        if not checkout:
            messages.warning(request, "No active checkout session — start or resume one first.")
            return redirect("checkout")
        session_key = request.session.session_key

        with transaction.atomic():
            product = Product.objects.select_for_update().get(pk=product.pk)

            if not product.status:
                if override_inactive:
                    product.status = True
                    product.save(update_fields=["status"])
                else:
                    messages.error(request, f"Cannot add '{product.name}' — product is inactive.", extra_tags="order")
                    return redirect("checkout_cart")

            if product.expiry_date and product.expiry_date < now().date():
                if not override_expiry:
                    messages.error(
                        request,
                        f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                        extra_tags="order",
                    )
                    return redirect("checkout_cart")

            item, _ = CheckoutOrderItem.objects.get_or_create(
                checkout=checkout, product=product,
                defaults={
                    "product_name": product.name,
                    "product_barcode": product.barcode or "",
                    "price": product.price,
                    "taxable": product.taxable,
                    "quantity": 0,
                },
            )
            CheckoutOrderItem.objects.filter(pk=item.pk).update(quantity=F("quantity") + requested_quantity)
            CheckoutOrder.objects.filter(pk=checkout.pk).update(
                active_session_key=session_key, updated_at=now()
            )
            stock = int(product.quantity_in_stock or 0)
            item.refresh_from_db()
            desired_qty = item.quantity

        override_notes = []
        if override_inactive: override_notes.append("product activated")
        if override_expiry: override_notes.append("expired override")

        if stock <= 0:
            messages.info(request, f"Added '{product.name}' (0 in stock).", extra_tags="order")
        elif desired_qty > stock:
            messages.warning(request, f"'{product.name}' quantity ({desired_qty}) exceeds stock ({stock}).", extra_tags="order")
        elif override_notes:
            messages.warning(request, f"⚠️ Added '{product.name}' ({', '.join(override_notes)}).", extra_tags="order")
        else:
            messages.success(request, f"Added {requested_quantity} unit(s) of '{product.name}'. (Now {desired_qty}/{stock})", extra_tags="order")

        return redirect("checkout_cart")


class CheckoutAddView(UserRequiredMixin, View):
    """Add a product to the checkout by id (search / inventory path), capped at stock."""
    def post(self, request, product_id):
        try:
            requested_quantity = int(request.POST.get("quantity", 1))
            if requested_quantity < 0:
                messages.error(request, "Quantity cannot be negative.", extra_tags="order")
                return redirect("checkout_cart")
        except (ValueError, TypeError):
            messages.error(request, "Invalid quantity value.", extra_tags="order")
            return redirect("checkout_cart")

        checkout = get_current_checkout(request)
        if not checkout:
            messages.warning(request, "No active checkout session — start or resume one first.")
            return redirect("checkout")
        session_key = request.session.session_key

        try:
            with transaction.atomic():
                product = Product.objects.select_for_update().get(product_id=product_id)

                if product.expiry_date and product.expiry_date < now().date():
                    messages.error(
                        request,
                        f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                        extra_tags="order",
                    )
                    return redirect("checkout_cart")

                stock = int(product.quantity_in_stock or 0)
                item, _ = CheckoutOrderItem.objects.get_or_create(
                    checkout=checkout, product=product,
                    defaults={
                        "product_name": product.name,
                        "product_barcode": product.barcode or "",
                        "price": product.price,
                        "taxable": product.taxable,
                        "quantity": 0,
                    },
                )
                desired_qty = item.quantity + requested_quantity
                capped_qty = min(desired_qty, stock)
                CheckoutOrderItem.objects.filter(pk=item.pk).update(quantity=capped_qty)
                CheckoutOrder.objects.filter(pk=checkout.pk).update(
                    active_session_key=session_key, updated_at=now()
                )
        except Product.DoesNotExist:
            messages.error(request, "Product not found.", extra_tags="order")
            return redirect("checkout_cart")

        if stock <= 0:
            messages.warning(request, f"'{product.name}' is OUT OF STOCK (0). Add accepted — quantity stays 0.", extra_tags="order")
        elif capped_qty < desired_qty:
            messages.warning(request, f"'{product.name}' capped at {stock} (in stock).", extra_tags="order")
        else:
            messages.success(request, f"Added {requested_quantity} unit(s) of '{product.name}'. (Now {capped_qty}/{stock})", extra_tags="order")
        return redirect("checkout_cart")


@user_passes_test(lambda u: u.is_authenticated)
def checkout_delete_item(request, item_id):
    checkout = get_current_checkout(request)
    if not checkout:
        messages.warning(request, "No active checkout.", extra_tags="order")
        return redirect("checkout")

    item = checkout.items.filter(pk=item_id).first()
    if not item:
        messages.warning(request, "Item not found in checkout.", extra_tags="order")
        return redirect("checkout_cart")

    if item.quantity > 1:
        CheckoutOrderItem.objects.filter(pk=item.pk).update(quantity=F("quantity") - 1)
    else:
        item.delete()

    messages.success(request, "1 unit removed from the order.", extra_tags="order")
    return redirect("checkout_cart")


class CheckoutNewView(UserRequiredMixin, View):
    """Start a brand-new checkout session for this terminal (used by the chooser
    and the concurrency modal)."""
    def post(self, request, *args, **kwargs):
        if not request.session.session_key:
            request.session.save()
        checkout = CheckoutOrder.objects.create(
            user=request.user,
            status=CheckoutOrder.STATUS_DRAFT,
            active_session_key=request.session.session_key or "",
        )
        request.session['checkout_id'] = checkout.pk
        UserAction.objects.create(
            user=request.user, action='checkout_new',
            target=f'Checkout #{checkout.pk}',
        )
        messages.success(request, "Started a new checkout.", extra_tags="order")
        return redirect("checkout_cart")


class CheckoutSubmitView(UserRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        checkout = get_current_checkout(request)
        if not checkout:
            return redirect("checkout")
        if not checkout.items.exists():
            messages.error(request, "Cannot submit an empty checkout.", extra_tags="order")
            return redirect("checkout_cart")

        unfulfilled_lines = []

        with transaction.atomic():
            checkout = CheckoutOrder.objects.select_for_update().get(pk=checkout.pk)
            # Idempotent: if already submitted (double-submit), go to success
            if checkout.status != CheckoutOrder.STATUS_DRAFT:
                return redirect("checkout_success", checkout_id=checkout.pk)

            items = list(checkout.items.select_related("product").all())
            product_ids = [it.product_id for it in items if it.product_id]
            locked = {
                p.product_id: p
                for p in Product.objects.select_for_update().filter(product_id__in=product_ids)
            }

            subtotal = Decimal("0.00")
            tax_total = Decimal("0.00")
            for it in items:
                requested = int(it.quantity)
                line = it.price * requested
                subtotal += line
                if it.taxable:
                    tax_total += line * TAX_RATE

                product = locked.get(it.product_id)
                if not product:
                    continue  # deleted product — keep the line, no stock effect

                available = int(product.quantity_in_stock or 0)
                if requested > 0:
                    deduct = min(requested, available)
                    if deduct > 0:
                        product.quantity_in_stock = available - deduct
                        product.save(update_fields=["quantity_in_stock"])
                        record_stock_change(
                            product=product, qty=deduct, change_type="giveaway",
                            note=f"PU Checkout {checkout.pk}", user=request.user,
                        )
                    shortfall = requested - deduct
                    if shortfall > 0:
                        record_stock_change(
                            product=product, qty=shortfall, change_type="giveaway_unfulfilled",
                            note=f"PU Checkout {checkout.pk} — short {shortfall} (stockout)",
                            user=request.user,
                        )
                        unfulfilled_lines.append(f"{product.name} (short {shortfall})")

                    # Giveaways are NOT sales demand, so they do not feed
                    # RecentlyPurchasedProduct (reorder velocity).

            checkout.subtotal = subtotal
            checkout.tax = tax_total
            checkout.total_price = subtotal + tax_total
            checkout.status = CheckoutOrder.STATUS_SUBMITTED
            checkout.submitted_at = now()
            checkout.active_session_key = ""
            checkout.save(update_fields=[
                "subtotal", "tax", "total_price", "status",
                "submitted_at", "active_session_key", "updated_at",
            ])

            UserAction.objects.create(
                user=request.user, action='checkout_submit',
                target=f'Checkout #{checkout.pk}',
                detail=f'{len(items)} line(s), total {checkout.total_price}',
            )

        if unfulfilled_lines:
            messages.warning(
                request,
                "Checkout submitted, but some items were not fulfilled: " + ", ".join(unfulfilled_lines),
                extra_tags="order",
            )
        else:
            messages.success(request, "Checkout submitted successfully.", extra_tags="order")

        # This terminal's current session is done — clear it so the next visit
        # to Checkout shows the chooser.
        if request.session.get('checkout_id') == checkout.pk:
            request.session.pop('checkout_id', None)

        return redirect("checkout_success", checkout_id=checkout.pk)


class CheckoutSuccessView(UserRequiredMixin, View):
    template_name = 'checkout_success.html'

    def get(self, request, checkout_id):
        checkout = get_object_or_404(CheckoutOrder, pk=checkout_id)
        items = checkout.items.all()
        return render(request, self.template_name, {
            'checkout': checkout,
            'items': items,
            'item_count': sum(i.quantity for i in items),
        })


class CheckoutHistoryDeleteView(UserRequiredMixin, View):
    """Delete one submitted checkout from this user's history.

    Removes the record only — the stock already moved when it was submitted and
    the Stock Log audit (StockChange rows) is preserved.
    """
    def post(self, request, checkout_id):
        # History is shared; only staff may remove entries.
        if not request.user.is_staff:
            messages.error(request, "Only staff can delete shared checkout history.", extra_tags="order")
            return redirect('checkout')
        co = CheckoutOrder.objects.filter(
            pk=checkout_id, status=CheckoutOrder.STATUS_SUBMITTED
        ).first()
        if co:
            co.delete()
            messages.success(request, f"Removed checkout #{checkout_id} from history.", extra_tags="order")
        else:
            messages.warning(request, "Checkout not found in history.", extra_tags="order")
        return redirect('checkout')


class CheckoutSessionDeleteView(UserRequiredMixin, View):
    """Delete an in-progress (draft) checkout session and its items.

    Drafts have not moved any stock yet, so deleting one just discards the
    in-progress cart — nothing in the Stock Log is affected.
    """
    def post(self, request, checkout_id):
        co = CheckoutOrder.objects.filter(
            pk=checkout_id, status=CheckoutOrder.STATUS_DRAFT
        ).first()
        if co:
            # If this browser was holding that draft, clear the reference.
            if request.session.get('checkout_id') == co.pk:
                request.session.pop('checkout_id', None)
                request.session.modified = True
            co.delete()  # cascades CheckoutOrderItem rows
            messages.success(request, f"Deleted checkout session #{checkout_id}.", extra_tags="order")
        else:
            messages.warning(request, "Active session not found.", extra_tags="order")
        return redirect('checkout')


class CheckoutHistoryClearView(UserRequiredMixin, View):
    """Clear all submitted checkouts from this user's history (records only)."""
    def post(self, request):
        # History is shared; only staff may clear it.
        if not request.user.is_staff:
            messages.error(request, "Only staff can clear shared checkout history.", extra_tags="order")
            return redirect('checkout')
        qs = CheckoutOrder.objects.filter(
            status=CheckoutOrder.STATUS_SUBMITTED
        )
        count = qs.count()
        qs.delete()
        messages.success(request, f"Cleared {count} checkout(s) from history.", extra_tags="order")
        return redirect('checkout')


class GiveawayDetailView(AdminRequiredMixin, View):
    """Admin-readable detail for one terminal giveaway (a submitted CheckoutOrder)."""
    template_name = 'giveaway_detail.html'

    def get(self, request, checkout_id):
        checkout = get_object_or_404(
            CheckoutOrder, pk=checkout_id, status=CheckoutOrder.STATUS_SUBMITTED
        )
        items = checkout.items.select_related('product').all()
        return render(request, self.template_name, {
            'checkout': checkout,
            'items': items,
            'item_count': sum(i.quantity for i in items),
        })


# ── Page presence (one-computer-per-page lock) heartbeat endpoints ──

@login_required
@require_POST
def presence_ping(request):
    """Heartbeat: refresh/claim this page's lock, or report it's held by another."""
    key = request.POST.get('page', '')
    if not key:
        return JsonResponse({'status': 'idle'})
    if not request.session.session_key:
        request.session.save()
    my = request.session.session_key
    holder = PagePresence.objects.filter(page=key).first()
    if holder and holder.session_key != my and is_fresh(holder):
        return JsonResponse({'status': 'blocked', 'holder': holder_info(holder)})
    PagePresence.objects.update_or_create(page=key, defaults=presence_defaults(request))
    return JsonResponse({'status': 'held'})


@login_required
@require_POST
def presence_takeover(request):
    """Force this computer to become the holder of the page (kicks the other)."""
    key = request.POST.get('page', '')
    if not key:
        return JsonResponse({'status': 'idle'})
    if not request.session.session_key:
        request.session.save()
    PagePresence.objects.update_or_create(page=key, defaults=presence_defaults(request))
    return JsonResponse({'status': 'held'})


@login_required
@require_POST
def presence_heartbeat(request):
    """Global heartbeat from every signed-in computer: record the screen it's on.

    Decoupled from the one-computer page lock — this drives the live nav bubble
    that shows which computer is on which screen, on ALL pages.
    """
    if not request.session.session_key:
        request.session.save()
    path = (request.POST.get('page', '') or '')[:200]
    UserSession.objects.filter(session_key=request.session.session_key).update(
        current_path=path, last_activity=now()
    )
    return JsonResponse({'ok': True})


@login_required
def presence_active(request):
    """Which OTHER computers are signed in and on which screen, for the nav bubble."""
    if not request.session.session_key:
        request.session.save()
    my = request.session.session_key
    cutoff = now() - timedelta(seconds=PRESENCE_TTL)
    rows = (
        UserSession.objects
        .filter(last_activity__gte=cutoff)
        .exclude(session_key=my)
        .select_related('user')
        .order_by('-last_activity')
    )
    pages = []
    for us in rows:
        pages.append({
            'page': path_label(us.current_path),
            'ip': us.ip_address or '—',
            'browser': simplify_ua(us.user_agent),
            'user': us.user.get_username() if us.user else '',
        })
    return JsonResponse({'count': len(pages), 'pages': pages})


@csrf_exempt
@login_required
@require_POST
def presence_release(request):
    """Release this page's lock (sent via sendBeacon on page unload)."""
    key = request.POST.get('page', '')
    if key and request.session.session_key:
        PagePresence.objects.filter(page=key, session_key=request.session.session_key).delete()
    return JsonResponse({'ok': True})


class ActiveSessionsView(AdminRequiredMixin, View):
    """Admin oversight page: who is currently signed in, from which computer,
    and which screen they're on.

    Reads the live UserSession heartbeat data (every signed-in computer pings
    presence_heartbeat every ~10s, refreshing last_activity + current_path).
    Three display states:
      - online: heartbeat fresher than PRESENCE_TTL (30s) — green, live.
      - idle:   between 30s and SESSION_ACTIVE_WINDOW — still holds a slot.
      - stale:  older than the window — no longer counts; will be cleared at the
                next login or by the prune_sessions command.
    "Active slots N / GLOBAL_MAX_SESSIONS" reflects the same cap the login
    enforces. This view is READ-ONLY: it never deletes rows (it auto-refreshes
    via GET), so pruning is left to login / the scheduled command.

    Supports ?format=json so the page can auto-refresh without a full reload.
    """
    template_name = 'active_sessions.html'

    def _rows(self, request):
        my = request.session.session_key
        live_cutoff = now() - timedelta(seconds=PRESENCE_TTL)
        active_cutoff = session_limits.active_cutoff()
        sessions = (
            UserSession.objects
            .select_related('user')
            .order_by('-last_activity')
        )
        rows = []
        for us in sessions:
            online = us.last_activity >= live_cutoff
            counts = us.last_activity >= active_cutoff
            if online:
                status, label = 'online', 'Online'
            elif counts:
                status, label = 'idle', 'Idle'
            else:
                status, label = 'stale', 'Disconnected'
            rows.append({
                'id': us.pk,
                'username': us.user.get_username() if us.user else '—',
                'role': 'Admin' if (us.user and us.user.is_staff) else 'Regular',
                'ip': us.ip_address or '—',
                'browser': simplify_ua(us.user_agent),
                'device': us.get_device_type_display(),
                'is_phone': us.device_type == UserSession.DEVICE_PHONE,
                'screen': path_label(us.current_path),
                'online': online,
                'status': status,
                'status_label': label,
                'counts': counts,
                'last_active': 'Active now' if online else (timesince(us.last_activity) + ' ago'),
                'since': localtime(us.created_at).strftime('%b %d, %I:%M %p'),
                'is_me': us.session_key == my,
            })
        return rows

    def get(self, request):
        rows = self._rows(request)
        payload = {
            'as_of': localtime(now()).strftime('%I:%M:%S %p'),
            'online_count': sum(1 for r in rows if r['online']),
            'active_slots': sum(1 for r in rows if r['counts']),
            'max_slots': session_limits.global_max(),
            'total': len(rows),
            'rows': rows,
        }
        if request.GET.get('format') == 'json':
            return JsonResponse(payload)
        return render(request, self.template_name, payload)

    def post(self, request):
        """Admin "boot": end another computer's session so it bounces to login.

        Deleting the Django session row deauthenticates that browser — its next
        request (a navigation or the ~10s presence heartbeat) is redirected to
        the login screen. We also drop the UserSession (frees a slot, clears it
        from this monitor) and any page-presence locks it held.
        """
        # Booting is GINA-only. AdminRequiredMixin also lets a passkey-unlocked PU
        # onto this page, so guard the action itself against non-staff accounts.
        if not request.user.is_staff:
            return JsonResponse(
                {'ok': False, 'error': 'Only the GINA account can log other users off.'},
                status=403,
            )

        if request.POST.get('action') != 'boot':
            return JsonResponse({'ok': False, 'error': 'Unknown action.'}, status=400)

        target = (UserSession.objects.select_related('user')
                  .filter(pk=request.POST.get('session_id')).first())
        if not target:
            return JsonResponse({'ok': False, 'error': 'That session is no longer active.'}, status=404)
        if target.session_key == request.session.session_key:
            return JsonResponse({'ok': False, 'error': "You can't log yourself off here."}, status=400)

        username = target.user.get_username() if target.user else '—'
        DjangoSession.objects.filter(session_key=target.session_key).delete()
        PagePresence.objects.filter(session_key=target.session_key).delete()
        target.delete()
        UserAction.objects.create(user=request.user, action='boot_session', target=username)
        return JsonResponse({'ok': True, 'username': username})


def _parse_expiry_date(raw):
    raw = raw.strip().rstrip('-')
    if not raw:
        return None
    for fmt in ('%d-%m-%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(raw, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _save_expiry_dates(product, primary_date, extra_date_strings):
    product.expiry_dates.all().delete()
    dates = []
    if primary_date:
        dates.append(primary_date)
    for raw in extra_date_strings:
        parsed = _parse_expiry_date(raw)
        if parsed:
            dates.append(parsed)
    for d in dates:
        ProductExpiryDate.objects.create(product=product, expiry_date=d)
    product.refresh_earliest_expiry()


#Change - Function to annotate changes

def record_stock_change(
    product: Product,
    qty: int,
    change_type: str,
    note: str = "",
    user=None,
    session=None,
) -> None:
    """
    Creates a StockChange row and updates per-product counters.
    """
    with transaction.atomic():
        # 1) Persist the audit trail (snapshot product identity so the row stays
        #    readable if the product is later deleted → product FK becomes NULL).
        StockChange.objects.create(
            product=product,
            product_name=product.name,
            product_barcode=product.barcode or "",
            change_type=change_type,
            quantity=qty,
            note=note or None,
            user=user,
            session=session,
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
        
        # Product deletion loss — tracked separately from genuine expiry so it
        # does not inflate stock_expired (shrinkage/discontinuation, not expiry).
        elif change_type == "deletion":
            product.stock_deleted = (product.stock_deleted or 0) + abs(qty)

        # Giveaway (PU terminal) — physically removes stock, but tracked
        # separately from sales so it never inflates stock_sold / sales demand.
        elif change_type == "giveaway":
            product.stock_giveaway = (product.stock_giveaway or 0) + abs(qty)

        elif change_type == "giveaway_unfulfilled":
            # No physical stock change and not a sale — audit row only.
            pass

        product.save(
            update_fields=[
                "stock_bought", "stock_sold", "stock_expired",
                "stock_unfulfilled", "stock_giveaway", "stock_deleted",
            ]
        )



def _adjust_inventory_count(session, product, delta):
    """Adjust the per-session inventory count tally for a product (count buffer).

    Used by the inventory-count scan/＋/－ paths instead of mutating live stock.
    Auto-adds the product to scope (snapshotting expected qty) if it wasn't one of
    the selected categories. Counts floor at 0. Returns (line, created).
    """
    line, created = InventoryCountLine.objects.get_or_create(
        session=session, product=product,
        defaults={
            'product_name': product.name,
            'product_barcode': product.barcode or "",
            'expected_qty': product.quantity_in_stock,
            'counted_qty': 0,
        },
    )
    line.counted_qty = max(0, line.counted_qty + delta)
    line.save(update_fields=['counted_qty', 'updated_at'])
    return line, created


# DELETES ONE ITEM ON CHECKIN BUTTON
@login_required
def delete_one(request, session_id, product_id):
    """
    Subtract 1 unit from product stock (with inventory mode support).
    """
    session = get_object_or_404(CheckinSession, pk=session_id)
    if not session.is_active:
        messages.error(request, "This session has ended.", extra_tags="checkin error")
        return redirect("checkin_dashboard")

    if request.method != "POST":
        return redirect("checkin_session", session_id=session.pk)

    inventory_mode = session.inventory_mode

    with transaction.atomic():
        product = get_object_or_404(
            Product.objects.select_for_update(),
            pk=product_id
        )

        if inventory_mode:
            # Count buffer: decrement the tally (floor 0), never live stock.
            line, _ = _adjust_inventory_count(session, product, -1)
            if not product.status:
                product.status = True
                product.save(update_fields=["status"])
            messages.success(
                request,
                f"Count −1 {product.name} (count {line.counted_qty} · system {product.quantity_in_stock}).",
                extra_tags="checkin success",
            )
        elif product.quantity_in_stock <= 0:
            messages.error(request, f"Cannot subtract. {product.name} is already out of stock.", extra_tags="checkin error")
        else:
            product.quantity_in_stock -= 1
            product.save(update_fields=["quantity_in_stock"])
            messages.success(request, f"Adjusted: 1 unit removed from {product.name}'s stock.", extra_tags="checkin success")
            record_stock_change(product, qty=1, change_type="checkin_delete1", note="1 unit removed via UI", user=request.user, session=session)

    return redirect(f"{reverse('checkin_session', kwargs={'session_id': session.pk})}?barcode={product.barcode}")


#add1 checkin
@login_required
def AddQuantityView(request, session_id, product_id):
    """
    Add quantity to product stock (with inventory mode support).
    """
    session = get_object_or_404(CheckinSession, pk=session_id)
    if not session.is_active:
        messages.error(request, "This session has ended.", extra_tags="checkin error")
        return redirect("checkin_dashboard")

    session_url = reverse('checkin_session', kwargs={'session_id': session.pk})

    if request.method != "POST":
        return redirect("checkin_session", session_id=session.pk)

    inventory_mode = session.inventory_mode

    try:
        quantity_to_add = int(request.POST.get("amount", 1))
        if quantity_to_add <= 0:
            messages.error(request, "Quantity must be greater than 0.", extra_tags="checkin error")
            return redirect("checkin_session", session_id=session.pk)
        if quantity_to_add > 1000:
            messages.error(request, "Quantity too large. Maximum 1000 units per operation.", extra_tags="checkin error")
            return redirect("checkin_session", session_id=session.pk)
    except (ValueError, TypeError):
        messages.error(request, "Invalid quantity value.", extra_tags="checkin error")
        return redirect("checkin_session", session_id=session.pk)

    with transaction.atomic():
        product = get_object_or_404(Product.objects.select_for_update(), product_id=product_id)

        if inventory_mode:
            # Count buffer: add to the tally, never live stock.
            line, _ = _adjust_inventory_count(session, product, quantity_to_add)
            if not product.status:
                product.status = True
                product.save(update_fields=["status"])
            messages.success(
                request,
                f"Count +{quantity_to_add} {product.name} (count {line.counted_qty} · system {product.quantity_in_stock}).",
                extra_tags="checkin success",
            )
        else:
            product.quantity_in_stock += quantity_to_add
            product.save(update_fields=["quantity_in_stock"])
            messages.success(request, f"{quantity_to_add} unit(s) of {product.name} added to stock.", extra_tags="checkin success")
            record_stock_change(product, qty=quantity_to_add, change_type="checkin", note="Manual add via UI", user=request.user, session=session)

    return redirect(f"{session_url}?barcode={product.barcode}")

# add products without barcode (triggered via Search/Autocomplete)
class AddProductByIdCheckinView(LoginRequiredMixin, View):
    def post(self, request, session_id, product_id):
        session = get_object_or_404(CheckinSession, pk=session_id)
        if not session.is_active:
            messages.error(request, "This session has ended.", extra_tags="checkin error")
            return redirect("checkin_dashboard")

        try:
            product = Product.objects.get(product_id=product_id)
        except Product.DoesNotExist:
            messages.error(request, "Product not found.", extra_tags="checkin error")
            return redirect("checkin_session", session_id=session.pk)

        return redirect(f"{reverse('checkin_session', kwargs={'session_id': session.pk})}?barcode={product.barcode}")

# ── Checkin Session Dashboard & Lifecycle Views ──

class CheckinDashboardView(LoginRequiredMixin, View):
    template_name = "checkin_dashboard.html"

    @staticmethod
    def _session_presence(request, active_sessions):
        """Map {session_pk: {ip, browser}} for active sessions whose individual
        check-in page is currently held by ANOTHER computer (fresh page lock)."""
        if not request.session.session_key:
            request.session.save()
        my = request.session.session_key
        path_to_pk = {
            reverse('checkin_session', kwargs={'session_id': s.pk}): s.pk
            for s in active_sessions
        }
        result = {}
        if path_to_pk:
            rows = PagePresence.objects.filter(page__in=path_to_pk.keys()).exclude(session_key=my)
            for p in rows:
                if is_fresh(p):
                    result[path_to_pk[p.page]] = {
                        'ip': p.ip_address or 'another computer',
                        'browser': simplify_ua(p.user_agent),
                    }
        return result

    def get(self, request):
        # ── AJAX presence API: which active sessions another computer is on ──
        if request.GET.get('format') == 'presence' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            active = list(CheckinSession.objects.filter(ended_at__isnull=True).only('id'))
            return JsonResponse({'in_use': self._session_presence(request, active)})

        # ── AJAX Recent Scans API ──
        if request.GET.get('format') == 'recent_scans' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            try:
                scans_qs = StockChange.objects.filter(
                    change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract']
                ).select_related('product').order_by('-timestamp')[:25]
                today = date.today()
                today_scans = StockChange.objects.filter(
                    change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract'],
                    timestamp__date=today
                )
                entries = []
                for sc in scans_qs:
                    try:
                        entries.append({
                            'time': sc.timestamp.strftime('%b %d %H:%M'),
                            'time_ago': timesince(sc.timestamp),
                            'name': sc.display_name,
                            'barcode': sc.display_barcode,
                            'qty': sc.quantity,
                            'positive': sc.change_type in ('checkin', 'error_add'),
                            'stock': sc.product.quantity_in_stock if sc.product else 0,
                            'action': sc.get_change_type_display(),
                        })
                    except Exception:
                        continue
                return JsonResponse({
                    'entries': entries,
                    'scanned_today': today_scans.filter(change_type='checkin').count(),
                    'products_updated': today_scans.values('product').distinct().count(),
                })
            except Exception as e:
                return JsonResponse({'error': str(e), 'entries': [], 'scanned_today': 0, 'products_updated': 0})

        # ── AJAX Stock Log API → canonical shared endpoint ──
        if request.GET.get('format') == 'json' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return stock_log_api(request)

        # All active sessions (could be multiple via reopen)
        active_sessions = list(
            CheckinSession.objects.filter(ended_at__isnull=True)
            .select_related('user')
            .order_by('-started_at')
        )

        # Flag sessions another computer is currently working on (live page lock)
        presence = self._session_presence(request, active_sessions)
        for s in active_sessions:
            info = presence.get(s.pk)
            s.in_use = bool(info)
            s.in_use_by = ' · '.join(filter(None, [info['ip'], info['browser']])) if info else ''

        # Session history (all sessions, most recent first)
        sessions_qs = CheckinSession.objects.select_related('user').all()
        paginator = Paginator(sessions_qs, 15)
        page = paginator.get_page(request.GET.get('page', 1))

        change_types = StockChange._meta.get_field('change_type').choices

        # Data for the Inventory Count Mode start modal (category → product picker).
        inv_categories = (
            Category.objects.annotate(product_count=Count('product')).order_by('name')
        )
        inv_products_json = list(
            Product.objects.values('product_id', 'name', 'barcode', 'category_id', 'quantity_in_stock')
        )

        return render(request, self.template_name, {
            "active_sessions": active_sessions,
            "sessions_page": page,
            "change_types": change_types,
            "inv_categories": inv_categories,
            "inv_products_json": inv_products_json,
        })


class StartCheckinSessionView(LoginRequiredMixin, View):
    def post(self, request):
        scanned_by = request.POST.get("scanned_by", "").strip()
        if not scanned_by:
            messages.error(request, "Please enter your name to start a session.", extra_tags="checkin error")
            return redirect("checkin_dashboard")

        note = request.POST.get("note", "").strip()
        inventory_mode = request.POST.get("inventory_mode") == "on"
        try:
            session = CheckinSession.objects.create(
                user=request.user, scanned_by=scanned_by,
                note=note, inventory_mode=inventory_mode,
            )
        except Exception:
            # Fallback if DB schema is behind (missing columns)
            session = CheckinSession.objects.create(user=request.user, note=f"{scanned_by} | {note}".strip(" |"))

        detail = f'Scanned by: {scanned_by}'

        # Inventory Count Mode: build the count scope from the products chosen in
        # the start modal. Snapshot expected qty; DO NOT touch live stock.
        scope_count = 0
        if inventory_mode:
            raw_ids = request.POST.get("count_product_ids", "")
            id_list = [int(x) for x in raw_ids.split(",") if x.strip().isdigit()]
            if id_list:
                products = Product.objects.filter(product_id__in=id_list)
                lines = [
                    InventoryCountLine(
                        session=session, product=p,
                        product_name=p.name, product_barcode=p.barcode or "",
                        expected_qty=p.quantity_in_stock, counted_qty=0,
                    )
                    for p in products
                ]
                InventoryCountLine.objects.bulk_create(lines, ignore_conflicts=True)
                scope_count = len(lines)
                detail += f' — inventory count, {scope_count} products in scope'

        UserAction.objects.create(user=request.user, action='start_session',
            target=f'Session #{session.pk}', detail=detail)
        return redirect("checkin_session", session_id=session.pk)


class EndCheckinSessionView(LoginRequiredMixin, View):
    def post(self, request, session_id):
        session = get_object_or_404(CheckinSession, pk=session_id)
        # Inventory-count sessions must go through reconcile (apply counts +
        # variance) rather than ending directly.
        if session.is_active and session.inventory_mode:
            return redirect("checkin_reconcile", session_id=session.pk)
        if session.is_active:
            session.ended_at = now()
            session.save(update_fields=["ended_at"])
            UserAction.objects.create(user=request.user, action='end_session',
                target=f'Session #{session.pk}', detail=f'{session.items_scanned} items scanned')
            messages.success(request, f"Session ended. {session.items_scanned} items were scanned.", extra_tags="checkin success")
        return redirect("checkin_dashboard")


class CheckinReconcileView(LoginRequiredMixin, View):
    """Review + apply an Inventory Count Mode session.

    GET shows expected vs counted vs variance for every count line (unscanned
    in-scope rows highlighted). POST applies the counts: the physical count is
    the source of truth — set quantity_in_stock = counted (unscanned in-scope
    → 0), record the delta vs live stock as a StockChange, then end the session.
    """
    template_name = "checkin_reconcile.html"

    def _load(self, session_id):
        session = get_object_or_404(CheckinSession, pk=session_id)
        lines = list(session.count_lines.select_related('product').all())
        return session, lines

    def get(self, request, session_id):
        session, lines = self._load(session_id)
        if not session.inventory_mode:
            return redirect("checkin_session_detail", session_id=session.pk)
        if not session.is_active:
            return redirect("checkin_session_detail", session_id=session.pk)

        discrepancies = sum(1 for l in lines if l.variance != 0)
        net = sum(l.variance for l in lines)
        zero_rows = sum(1 for l in lines if l.counted_qty == 0)
        return render(request, self.template_name, {
            "session": session,
            "lines": lines,
            "products_counted": len(lines),
            "discrepancies": discrepancies,
            "net_adjustment": net,
            "zero_rows": zero_rows,
        })

    def post(self, request, session_id):
        session, lines = self._load(session_id)
        if not session.inventory_mode or not session.is_active:
            return redirect("checkin_dashboard")

        applied = 0
        discrepancies = 0
        net = 0
        with transaction.atomic():
            for line in lines:
                if not line.product_id:
                    continue
                product = Product.objects.select_for_update().filter(pk=line.product_id).first()
                if not product:
                    continue
                old = product.quantity_in_stock
                new = line.counted_qty
                diff = new - old
                if diff != 0:
                    product.quantity_in_stock = new
                    product.save(update_fields=["quantity_in_stock"])
                    record_stock_change(
                        product, qty=abs(diff),
                        change_type='error_add' if diff > 0 else 'error_subtract',
                        note=f"Inventory count: {old} → {new}",
                        user=request.user, session=session,
                    )
                    discrepancies += 1
                    net += diff
                applied += 1

            session.ended_at = now()
            session.save(update_fields=["ended_at"])

        UserAction.objects.create(
            user=request.user, action='cycle_count',
            target=f'Session #{session.pk}: {applied} products counted',
            detail=f'{discrepancies} discrepancies, net adjustment: {net:+d}',
        )
        msg = f"Inventory count applied: {applied} products counted, {discrepancies} discrepancies"
        if discrepancies:
            msg += f", net {net:+d}"
        messages.success(request, msg, extra_tags="checkin success")
        return redirect("checkin_dashboard")


class CheckinSessionDetailView(LoginRequiredMixin, View):
    template_name = "checkin_session_detail.html"

    def get(self, request, session_id):
        session = get_object_or_404(CheckinSession.objects.select_related('user'), pk=session_id)
        changes = session.stock_changes.select_related('product').order_by('-timestamp')
        products_touched = changes.values('product').distinct().count()

        # Net stock delta per product for this session
        net_totals = {}
        positive_types = {'checkin', 'error_add'}
        for c in changes:
            pid = c.product_id
            if pid not in net_totals:
                net_totals[pid] = {"name": c.product.name if c.product else "Deleted", "net": 0}
            if c.change_type in positive_types:
                net_totals[pid]["net"] += c.quantity
            else:
                net_totals[pid]["net"] -= c.quantity

        # Session lifecycle events (reopens, etc.)
        session_events = UserAction.objects.filter(
            action='reopen_session',
            target=f'Session #{session.pk}'
        ).select_related('user').order_by('-timestamp')

        return render(request, self.template_name, {
            "session": session,
            "changes": changes,
            "products_touched": products_touched,
            "can_edit": has_admin_access(request),
            "net_totals": net_totals,
            "session_events": session_events,
        })


class ReopenCheckinSessionView(LoginRequiredMixin, View):
    """Reopen a completed session so lines can be edited (admin or passkey-unlocked)."""

    def post(self, request, session_id):
        if not has_admin_access(request):
            return redirect(f"{reverse('passkey_unlock')}?{urlencode({'next': request.get_full_path()})}")
        session = get_object_or_404(CheckinSession, pk=session_id)
        if session.is_active:
            messages.info(request, "Session is already active.", extra_tags="checkin info")
        else:
            session.ended_at = None
            session.reopened_at = now()
            session.save(update_fields=["ended_at", "reopened_at"])
            UserAction.objects.create(user=request.user, action='reopen_session',
                target=f'Session #{session.pk}')
            messages.success(request, "Session reopened for editing.", extra_tags="checkin success")
        return redirect("checkin_session_detail", session_id=session.pk)


class SessionAdjustLineView(LoginRequiredMixin, View):
    """Adjust the quantity on a stock-change line within a session (admin or passkey-unlocked)."""

    def post(self, request, session_id, change_id):
        if not has_admin_access(request):
            return JsonResponse({"error": "Passkey required"}, status=403)
        session = get_object_or_404(CheckinSession, pk=session_id)
        change = get_object_or_404(StockChange, pk=change_id, session=session)

        try:
            new_qty = int(request.POST.get("new_qty", 0))
            if new_qty < 1 or new_qty > 10000:
                raise ValueError
        except (ValueError, TypeError):
            messages.error(request, "Invalid quantity.", extra_tags="checkin error")
            return redirect("checkin_session_detail", session_id=session.pk)

        old_qty = change.quantity
        diff = new_qty - old_qty
        if diff == 0:
            return redirect("checkin_session_detail", session_id=session.pk)

        with transaction.atomic():
            product = Product.objects.select_for_update().get(pk=change.product_id)

            # Determine stock direction of the original change
            positive_types = {'checkin', 'error_add'}
            original_was_add = change.change_type in positive_types

            # Update on-hand stock: if original was an add, more qty = more stock
            if original_was_add:
                product.quantity_in_stock += diff
            else:
                product.quantity_in_stock -= diff

            product.quantity_in_stock = max(product.quantity_in_stock, 0)
            product.save(update_fields=["quantity_in_stock"])

            # Update the original change row
            change.quantity = new_qty
            change.save(update_fields=["quantity"])

            # Record corrective audit entry
            if diff > 0:
                corr_type = "error_add" if original_was_add else "error_subtract"
            else:
                corr_type = "error_subtract" if original_was_add else "error_add"

            record_stock_change(
                product=product,
                qty=abs(diff),
                change_type=corr_type,
                note=f"Session #{session.pk} line adjusted: {old_qty} → {new_qty}",
                user=request.user,
                session=session,
            )

        UserAction.objects.create(user=request.user, action='adjust_session_line',
            target=f'Session #{session.pk}', detail=f'{product.name}: {old_qty} → {new_qty}')
        messages.success(
            request,
            f"Adjusted {product.name}: {old_qty} → {new_qty}.",
            extra_tags="checkin success",
        )
        return redirect("checkin_session_detail", session_id=session.pk)


class SessionRemoveLineView(LoginRequiredMixin, View):
    """Reverse a stock-change line and remove it from the session (admin or passkey-unlocked)."""

    def post(self, request, session_id, change_id):
        if not has_admin_access(request):
            return JsonResponse({"error": "Passkey required"}, status=403)
        session = get_object_or_404(CheckinSession, pk=session_id)
        change = get_object_or_404(StockChange, pk=change_id, session=session)

        with transaction.atomic():
            product = Product.objects.select_for_update().get(pk=change.product_id)

            positive_types = {'checkin', 'error_add'}
            original_was_add = change.change_type in positive_types

            # Reverse the stock effect
            if original_was_add:
                product.quantity_in_stock -= change.quantity
                corr_type = "error_subtract"
            else:
                product.quantity_in_stock += change.quantity
                corr_type = "error_add"

            product.quantity_in_stock = max(product.quantity_in_stock, 0)
            product.save(update_fields=["quantity_in_stock"])

            # Record corrective audit entry
            record_stock_change(
                product=product,
                qty=change.quantity,
                change_type=corr_type,
                note=f"Session #{session.pk} line removed (was {change.get_change_type_display()} x{change.quantity})",
                user=request.user,
                session=session,
            )

            # Delete the original change
            prod_name = product.name
            change_qty = change.quantity
            change.delete()

        UserAction.objects.create(user=request.user, action='remove_session_line',
            target=f'Session #{session.pk}', detail=f'{prod_name} x{change_qty} removed')
        messages.success(
            request,
            f"Removed {prod_name} line and reversed stock.",
            extra_tags="checkin success",
        )
        return redirect("checkin_session_detail", session_id=session.pk)


class DeleteCheckinSessionView(LoginRequiredMixin, View):
    def post(self, request, session_id):
        session = get_object_or_404(CheckinSession, pk=session_id)
        # Unlink stock changes (keep the audit trail, just detach from session)
        session.stock_changes.update(session=None)
        session.delete()
        UserAction.objects.create(user=request.user, action='delete_session',
            target=f'Session #{session_id}')
        messages.success(request, "Session deleted.", extra_tags="checkin success")
        return redirect("checkin_dashboard")


class ClearCheckinHistoryView(LoginRequiredMixin, View):
    def post(self, request):
        # Only clear completed sessions, not active ones
        completed = CheckinSession.objects.filter(ended_at__isnull=False)
        # Unlink stock changes first
        StockChange.objects.filter(session__in=completed).update(session=None)
        count = completed.count()
        completed.delete()
        UserAction.objects.create(user=request.user, action='clear_session_history',
            target=f'{count} sessions cleared')
        messages.success(request, f"Cleared {count} completed session(s).", extra_tags="checkin success")
        return redirect("checkin_dashboard")


class CheckinAllSessionsPDFView(LoginRequiredMixin, View):
    """Generate a PDF with each session and its indented stock change contents."""

    def get(self, request):
        from reportlab.lib.colors import HexColor

        sessions = CheckinSession.objects.filter(
            ended_at__isnull=False
        ).select_related('user').prefetch_related(
            'stock_changes__product'
        ).order_by('-started_at')

        buffer = io.BytesIO()
        PAGE_W, PAGE_H = letter
        c = canvas.Canvas(buffer, pagesize=letter)
        MARGIN = 54
        INDENT = MARGIN + 24

        brand = HexColor("#4f46e5")
        dark = HexColor("#1e293b")
        muted = HexColor("#64748b")
        line_clr = HexColor("#e2e8f0")
        row_alt = HexColor("#f8fafc")
        session_bg = HexColor("#f1f5f9")
        green = HexColor("#059669")
        red = HexColor("#dc2626")
        row_h = 14

        def hr(y_pos, color=line_clr, left=MARGIN):
            c.setStrokeColor(color)
            c.setLineWidth(0.5)
            c.line(left, y_pos, PAGE_W - MARGIN, y_pos)

        page_num = 1

        def draw_footer():
            c.setFillColor(muted)
            c.setFont("Helvetica", 7)
            c.drawString(MARGIN, 30, f"MPCP  |  All Check-in Sessions  |  Generated {now().strftime('%b %d, %Y %H:%M')}")
            c.drawRightString(PAGE_W - MARGIN, 30, f"Page {page_num}  |  Meadowvale Professional Center Pharmacy")

        def check_page(y_pos, needed=40):
            nonlocal page_num
            if y_pos < MARGIN + needed:
                draw_footer()
                c.showPage()
                page_num += 1
                return PAGE_H - MARGIN
            return y_pos

        # ── Header ──
        y = PAGE_H - MARGIN

        c.setFillColor(brand)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(MARGIN, y, "MPCP")
        c.setFillColor(muted)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN, y - 16, "Meadowvale Professional Center Pharmacy")

        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 14)
        c.drawRightString(PAGE_W - MARGIN, y, "CHECK-IN SESSIONS")
        c.setFillColor(muted)
        c.setFont("Helvetica", 10)
        c.drawRightString(PAGE_W - MARGIN, y - 18, f"{sessions.count()} completed session(s)")
        c.drawRightString(PAGE_W - MARGIN, y - 32, now().strftime("%B %d, %Y  %I:%M %p"))

        y -= 62
        hr(y)
        y -= 12

        total_items = 0
        total_actions = 0

        for idx, s in enumerate(sessions, 1):
            changes = s.stock_changes.select_related('product').order_by('timestamp')
            action_count = changes.count()
            item_count = s.items_scanned
            total_items += item_count
            total_actions += action_count

            # Duration
            dur = s.duration
            total_sec = int(dur.total_seconds())
            if total_sec < 60:
                dur_str = f"{total_sec}s"
            elif total_sec < 3600:
                dur_str = f"{total_sec // 60}m"
            else:
                dur_str = f"{total_sec // 3600}h {(total_sec % 3600) // 60}m"

            user_name = s.scanned_by or (s.user.username if s.user else 'Unknown')

            # ── Session header bar ──
            y = check_page(y, 60)

            # Background bar
            c.setFillColor(session_bg)
            c.rect(MARGIN, y - 5, PAGE_W - 2 * MARGIN, 20, fill=1, stroke=0)

            c.setFillColor(dark)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(MARGIN + 6, y + 1, f"Session #{s.pk}")

            c.setFillColor(muted)
            c.setFont("Helvetica", 8)
            c.drawString(MARGIN + 80, y + 1, f"{user_name}  |  {s.started_at.strftime('%b %d, %Y %H:%M')}  |  {dur_str}  |  {item_count} items")

            if s.note:
                c.setFillColor(brand)
                c.drawRightString(PAGE_W - MARGIN - 6, y + 1, s.note[:30])

            if s.inventory_mode:
                c.setFillColor(green)
                c.setFont("Helvetica-Bold", 7)
                c.drawString(MARGIN + 80 + stringWidth(f"{user_name}  |  {s.started_at.strftime('%b %d, %Y %H:%M')}  |  {dur_str}  |  {item_count} items", "Helvetica", 8) + 8, y + 1, "INV")

            y -= 22

            # ── Content rows (indented) ──
            if action_count == 0:
                c.setFillColor(muted)
                c.setFont("Helvetica-Oblique", 7.5)
                c.drawString(INDENT, y + 1, "No stock changes recorded")
                y -= row_h
            else:
                # Column headers for contents
                c.setFillColor(muted)
                c.setFont("Helvetica-Bold", 6.5)
                c.drawString(INDENT, y + 1, "TIME")
                c.drawString(INDENT + 50, y + 1, "PRODUCT")
                c.drawString(INDENT + 210, y + 1, "BARCODE")
                c.drawString(INDENT + 300, y + 1, "ACTION")
                c.drawRightString(INDENT + 400, y + 1, "QTY")
                c.drawRightString(PAGE_W - MARGIN - 6, y + 1, "NOTE")
                y -= row_h

                for ci, sc in enumerate(changes):
                    y = check_page(y)

                    if ci % 2 == 1:
                        c.setFillColor(row_alt)
                        c.rect(INDENT - 4, y - 3, PAGE_W - MARGIN - INDENT + 4, row_h, fill=1, stroke=0)

                    is_add = sc.change_type in ('checkin', 'error_add')
                    qty_str = f"+{sc.quantity}" if is_add else f"-{sc.quantity}"

                    c.setFont("Helvetica", 7)
                    c.setFillColor(muted)
                    c.drawString(INDENT, y + 1, sc.timestamp.strftime('%H:%M'))
                    c.setFillColor(dark)
                    c.drawString(INDENT + 50, y + 1, (sc.display_name)[:26])
                    c.setFillColor(muted)
                    c.drawString(INDENT + 210, y + 1, (sc.display_barcode or '-')[:14])
                    c.setFillColor(dark)
                    c.drawString(INDENT + 300, y + 1, sc.get_change_type_display()[:16])
                    c.setFillColor(green if is_add else red)
                    c.setFont("Helvetica-Bold", 7)
                    c.drawRightString(INDENT + 400, y + 1, qty_str)
                    c.setFont("Helvetica", 7)
                    c.setFillColor(muted)
                    c.drawRightString(PAGE_W - MARGIN - 6, y + 1, (sc.note or '-')[:18])

                    y -= row_h

            # Divider between sessions
            y -= 4
            hr(y, line_clr, MARGIN)
            y -= 12

        # ── Grand totals ──
        y = check_page(y)
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN, y, f"Total: {sessions.count()} session(s)  |  {total_items} items scanned  |  {total_actions} stock actions")

        draw_footer()
        c.save()
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="all_checkin_sessions.pdf"'
        return response


class CheckinSessionPDFView(LoginRequiredMixin, View):
    def get(self, request, session_id):
        from reportlab.lib.colors import HexColor

        session = get_object_or_404(CheckinSession.objects.select_related('user'), pk=session_id)
        changes = session.stock_changes.select_related('product').order_by('-timestamp')
        products_touched = changes.values('product').distinct().count()

        buffer = io.BytesIO()
        PAGE_W, PAGE_H = letter
        c = canvas.Canvas(buffer, pagesize=letter)
        MARGIN = 54

        # ── Brand colours ──
        brand = HexColor("#4f46e5")
        dark = HexColor("#1e293b")
        muted = HexColor("#64748b")
        line_clr = HexColor("#e2e8f0")
        row_alt = HexColor("#f8fafc")
        green = HexColor("#059669")
        red = HexColor("#dc2626")
        white = HexColor("#ffffff")

        def hr(y_pos, color=line_clr):
            c.setStrokeColor(color)
            c.setLineWidth(0.5)
            c.line(MARGIN, y_pos, PAGE_W - MARGIN, y_pos)
            return y_pos

        def draw_footer():
            c.setFillColor(muted)
            c.setFont("Helvetica", 7)
            c.drawString(MARGIN, 30, f"MPCP  |  Check-in Session #{session.pk}  |  Generated {now().strftime('%b %d, %Y %H:%M')}")
            c.drawRightString(PAGE_W - MARGIN, 30, "Meadowvale Professional Center Pharmacy")

        # ────────────────────────────────────────
        # HEADER
        # ────────────────────────────────────────
        y = PAGE_H - MARGIN

        c.setFillColor(brand)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(MARGIN, y, "MPCP")
        c.setFillColor(muted)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN, y - 16, "Meadowvale Professional Center Pharmacy")

        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 14)
        c.drawRightString(PAGE_W - MARGIN, y, "CHECK-IN REPORT")
        c.setFont("Helvetica", 10)
        c.setFillColor(muted)
        c.drawRightString(PAGE_W - MARGIN, y - 18, f"Session #{session.pk}")
        c.drawRightString(PAGE_W - MARGIN, y - 32, session.started_at.strftime("%B %d, %Y  %I:%M %p"))
        status_text = "In Progress" if session.is_active else "Completed"
        c.drawRightString(PAGE_W - MARGIN, y - 46, f"Status: {status_text}")

        y -= 62
        hr(y)
        y -= 22

        # ────────────────────────────────────────
        # SESSION DETAILS
        # ────────────────────────────────────────
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(MARGIN, y, "Session Details")
        y -= 18

        details = [
            ("Scanned By", session.scanned_by or (session.user.username if session.user else "Unknown")),
            ("Started", session.started_at.strftime("%b %d, %Y %H:%M")),
        ]
        if session.ended_at:
            details.append(("Ended", session.ended_at.strftime("%b %d, %Y %H:%M")))
        if session.note:
            details.append(("Label", session.note[:60]))
        details.append(("Total Actions", str(changes.count())))
        details.append(("Products Touched", str(products_touched)))

        c.setFont("Helvetica", 9)
        for label, value in details:
            c.setFillColor(muted)
            c.drawString(MARGIN, y, f"{label}:")
            c.setFillColor(dark)
            c.drawString(MARGIN + 110, y, value)
            y -= 15

        y -= 8
        hr(y)
        y -= 22

        # ────────────────────────────────────────
        # STOCK CHANGES TABLE
        # ────────────────────────────────────────
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(MARGIN, y, "Stock Changes")
        y -= 20

        if not changes.exists():
            c.setFillColor(muted)
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(MARGIN, y, "No stock changes were recorded in this session.")
            draw_footer()
            c.save()
            buffer.seek(0)
            response = HttpResponse(buffer, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="checkin_session_{session.pk}.pdf"'
            return response

        # Column layout
        col_num = MARGIN
        col_time = MARGIN + 22
        col_product = MARGIN + 80
        col_barcode = 290
        col_action = 380
        col_qty = 470
        col_note = PAGE_W - MARGIN
        row_h = 16

        def draw_table_header(y_pos):
            c.setFillColor(HexColor("#f1f5f9"))
            c.rect(MARGIN, y_pos - 4, PAGE_W - 2 * MARGIN, row_h + 2, fill=1, stroke=0)
            c.setFillColor(muted)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(col_num, y_pos + 1, "#")
            c.drawString(col_time, y_pos + 1, "TIME")
            c.drawString(col_product, y_pos + 1, "PRODUCT")
            c.drawString(col_barcode, y_pos + 1, "BARCODE")
            c.drawString(col_action, y_pos + 1, "ACTION")
            c.drawRightString(col_qty, y_pos + 1, "QTY")
            c.drawRightString(col_note, y_pos + 1, "NOTE")
            return y_pos - row_h - 4

        y = draw_table_header(y)

        c.setFont("Helvetica", 8)
        for idx, sc in enumerate(changes, 1):
            if y < MARGIN + 50:
                draw_footer()
                c.showPage()
                y = PAGE_H - MARGIN
                y = draw_table_header(y)
                c.setFont("Helvetica", 8)

            # Alternating row background
            if idx % 2 == 0:
                c.setFillColor(row_alt)
                c.rect(MARGIN, y - 3, PAGE_W - 2 * MARGIN, row_h, fill=1, stroke=0)

            time_str = sc.timestamp.strftime('%H:%M:%S')
            name = (sc.display_name)[:30]
            barcode = (sc.display_barcode or '-')[:15]
            action = sc.get_change_type_display()[:18]
            is_add = sc.change_type in ('checkin', 'error_add')
            qty_str = f"+{sc.quantity}" if is_add else f"-{sc.quantity}"
            note = (sc.note or '-')[:22]

            c.setFillColor(dark)
            c.drawString(col_num, y + 1, str(idx))
            c.setFillColor(muted)
            c.drawString(col_time, y + 1, time_str)
            c.setFillColor(dark)
            c.drawString(col_product, y + 1, name)
            c.setFillColor(muted)
            c.drawString(col_barcode, y + 1, barcode)
            c.setFillColor(dark)
            c.drawString(col_action, y + 1, action)
            c.setFillColor(green if is_add else red)
            c.setFont("Helvetica-Bold", 8)
            c.drawRightString(col_qty, y + 1, qty_str)
            c.setFont("Helvetica", 8)
            c.setFillColor(muted)
            c.drawRightString(col_note, y + 1, note)

            y -= row_h

        # Summary line
        y -= 8
        hr(y)
        y -= 18
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN, y, f"Total: {changes.count()} action(s) across {products_touched} product(s)")

        draw_footer()
        c.save()
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="checkin_session_{session.pk}.pdf"'
        return response


#checkin views
class CheckinProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"

    def get(self, request, session_id):
        session = get_object_or_404(CheckinSession, pk=session_id)
        if not session.is_active:
            return redirect("checkin_session_detail", session_id=session.pk)

        # ── AJAX Recent Scans API ──
        if request.GET.get('format') == 'recent_scans' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            try:
                scans_qs = StockChange.objects.filter(
                    change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract']
                ).select_related('product').order_by('-timestamp')[:25]
                today = date.today()
                today_scans = StockChange.objects.filter(
                    change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract'],
                    timestamp__date=today
                )
                entries = []
                for sc in scans_qs:
                    try:
                        entries.append({
                            'time': sc.timestamp.strftime('%b %d %H:%M'),
                            'time_ago': timesince(sc.timestamp),
                            'name': sc.display_name,
                            'barcode': sc.display_barcode,
                            'qty': sc.quantity,
                            'positive': sc.change_type in ('checkin', 'error_add'),
                            'stock': sc.product.quantity_in_stock if sc.product else 0,
                            'action': sc.get_change_type_display(),
                        })
                    except Exception:
                        continue
                return JsonResponse({
                    'entries': entries,
                    'scanned_today': today_scans.filter(change_type='checkin').count(),
                    'products_updated': today_scans.values('product').distinct().count(),
                })
            except Exception as e:
                return JsonResponse({'error': str(e), 'entries': [], 'scanned_today': 0, 'products_updated': 0})

        # ── AJAX Stock Log API → canonical shared endpoint ──
        if request.GET.get('format') == 'json' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return stock_log_api(request)

        barcode = (request.GET.get("barcode") or "").strip()
        product_id = (request.GET.get("product_id") or "").strip()
        inventory_mode = session.inventory_mode

        product = None
        # Prefer product_id (always present, works for barcode-less items)
        if product_id:
            product = Product.objects.filter(product_id=product_id).first()
        if product is None and barcode:
            product = find_product_by_barcode(barcode)

        query = (request.GET.get("name_query") or "").strip()
        search_results = []
        if query:
            # ✅ FIXED: Search by name, barcode, AND item_number
            search_results = Product.objects.filter(
                Q(name__icontains=query) |
                barcode_search_q(query) |
                Q(item_number__icontains=query)
            ).distinct()[:20]  # Limit results

        edit_form = EditProductForm(instance=product) if product else None

        # Last checkin timestamp for this product
        last_checkin = None
        if product:
            last_checkin = StockChange.objects.filter(
                product=product, change_type='checkin'
            ).order_by('-timestamp').first()

        # Per-product history: last 10 changes + 90-day daily movement chart
        product_history = []
        history_chart = []
        if product:
            product_history = list(
                StockChange.objects.filter(product=product)
                .select_related('user')
                .order_by('-timestamp')[:10]
            )
            in_types = {'checkin', 'error_add'}
            out_types = {'checkout', 'expired', 'error_subtract',
                         'checkin_delete1', 'giveaway', 'deletion'}
            daily = (
                StockChange.objects.filter(
                    product=product,
                    timestamp__date__gte=date.today() - timedelta(days=90),
                )
                .annotate(day=TruncDate('timestamp'))
                .values('day', 'change_type')
                .annotate(total=Sum('quantity'))
                .order_by('day')
            )
            by_day = {}
            for r in daily:
                rec = by_day.setdefault(
                    r['day'].isoformat(),
                    {'label': r['day'].strftime('%d %b'), 'in': 0, 'out': 0},
                )
                qty = abs(int(r['total'] or 0))
                if r['change_type'] in in_types:
                    rec['in'] += qty
                elif r['change_type'] in out_types:
                    rec['out'] += qty
            history_chart = [by_day[k] for k in sorted(by_day)]

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
            log_qs = log_qs.filter(Q(product__name__icontains=log_product) | barcode_search_q(log_product, 'product__barcode'))
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
                    sc.display_name,
                    sc.display_barcode,
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

        # Inventory Count Mode: the count tally for the whole session (progress
        # panel) and for the currently displayed product (card shows counted).
        count_lines = []
        count_line = None
        if inventory_mode:
            count_lines = list(
                session.count_lines.select_related('product').all()
            )
            if product:
                count_line = next((cl for cl in count_lines if cl.product_id == product.product_id), None)

        return render(request, self.template_name, {
            "session": session,
            "count_lines": count_lines,
            "count_line": count_line,
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
            "extra_dates": product.expiry_dates.all() if product else [],
            "categories": Category.objects.all(),
            "recent_scans": recent_scans,
            "scanned_today_count": scanned_today_count,
            "products_updated_today": products_updated_today,
            "last_checkin": last_checkin,
            "product_history": product_history,
            "history_chart": history_chart,
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

    def post(self, request, session_id):
        session = get_object_or_404(CheckinSession, pk=session_id)
        if not session.is_active:
            messages.error(request, "This session has ended.", extra_tags="checkin error")
            return redirect("checkin_dashboard")

        session_url = reverse('checkin_session', kwargs={'session_id': session.pk})

        barcode = (request.POST.get("barcode") or "").strip()
        inventory_mode = session.inventory_mode

        if not barcode:
            messages.error(request, "No barcode provided. Please scan a barcode.", extra_tags="checkin error")
            return self._render_no_product(request, inventory_mode, session)

        product = find_product_by_barcode(barcode)

        if product:
            # If same product is already displayed, add +1 to stock
            current_barcode = (request.POST.get("current_barcode") or "").strip()
            current_product = find_product_by_barcode(current_barcode) if current_barcode else None

            if current_product and current_product.pk == product.pk:
                with transaction.atomic():
                    product = Product.objects.select_for_update().get(pk=product.pk)

                    if inventory_mode:
                        # Count buffer: re-scanning the displayed product tallies +1
                        # into the count, never live stock.
                        line, created = _adjust_inventory_count(session, product, 1)
                        if not product.status:
                            product.status = True
                            product.save(update_fields=["status"])
                        if created:
                            messages.info(
                                request,
                                f"{product.name} added to this count (was not in the selected categories).",
                                extra_tags="checkin",
                            )
                        messages.success(
                            request,
                            f"Count +1 {product.name} (count {line.counted_qty} · system {product.quantity_in_stock})",
                            extra_tags="checkin success",
                        )
                    else:
                        product.quantity_in_stock += 1
                        product.save(update_fields=["quantity_in_stock"])
                        record_stock_change(
                            product, qty=1, change_type="checkin",
                            note="Barcode scan (+1)", user=request.user, session=session,
                        )
                        messages.success(request, f"+1 {product.name} (now {product.quantity_in_stock})", extra_tags="checkin success")

            return redirect(f"{session_url}?barcode={product.barcode}")

        # Not in store → try MASTER.csv
        master_row = get_master_catalog_entry(barcode)

        params = {
            "barcode": barcode,
            "next": session_url,
        }

        if master_row:
            # Many non-drug catalog rows have DIN = 0/blank; fall back to the
            # trimmed scanned barcode so the Item #/SKU isn't pre-filled as "0".
            din = (master_row.get("DIN", "") or "").strip()
            params.update({
                "name": master_row.get("ITEM DESCRIPTION", ""),
                "item_number": din if din and din != "0" else barcode,
                "unit_size": master_row.get("PRODUCT FORMAT", ""),
                "price_per_unit": _clean_price(master_row.get("COST")),
                "UPC": master_row.get("GTIN/UPC (unit)",""),
                "status": "on" if inventory_mode else None
            })
            # Suggested retail (informational tooltip on the form). Many drug rows
            # store "#VALUE!" here, so only pass a clean positive number.
            suggested = _clean_price(master_row.get("SUGGESTED RETAIL"))
            try:
                if Decimal(suggested) > 0:
                    params["suggested_retail"] = suggested
            except Exception:
                pass
            messages.info(request, "Details pulled from master catalogue.", extra_tags="checkin")
        else:
            messages.warning(request, "Barcode not found. Please add manually.", extra_tags="checkin")

        add_url = reverse("new_product")
        return redirect(f"{add_url}?{urlencode(params)}")

    def _render_no_product(self, request, inventory_mode=False, session=None):
        return render(
            request,
            self.template_name,
            {
                "session": session,
                "inventory_mode": inventory_mode,
                "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number", "barcode")),
                "categories": Category.objects.all(),
                "change_types": StockChange._meta.get_field('change_type').choices,
            },
        )

    
class CheckinEditProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"

    def post(self, request, session_id, product_id):
        session = get_object_or_404(CheckinSession, pk=session_id)
        if not session.is_active:
            messages.error(request, "This session has ended.", extra_tags="checkin error")
            return redirect("checkin_dashboard")

        session_url = reverse('checkin_session', kwargs={'session_id': session.pk})
        inventory_mode = session.inventory_mode

        with transaction.atomic():
            product = Product.objects.select_for_update().get(product_id=product_id)
            old_quantity = product.quantity_in_stock

            post_data = request.POST.copy()
            raw_date = post_data.get('expiry_date', '').strip().rstrip('-')
            if raw_date:
                try:
                    clean_date = datetime.strptime(raw_date, '%d-%m-%Y').date()
                    post_data['expiry_date'] = clean_date.strftime('%Y-%m-%d')
                except ValueError:
                    pass

            form = EditProductForm(post_data, instance=product)

            if form.is_valid():
                updated = form.save(commit=False)
                new_quantity = updated.quantity_in_stock

                if new_quantity != old_quantity:
                    change = "error_add" if new_quantity > old_quantity else "error_subtract"
                    record_stock_change(
                        product=updated, qty=abs(new_quantity - old_quantity),
                        change_type=change, note="Product updated via check-in inline edit",
                        user=request.user, session=session,
                    )

                updated.save()
                form.save_m2m()
                _save_expiry_dates(updated, updated.expiry_date, request.POST.getlist('extra_expiry_dates'))
                UserAction.objects.create(user=request.user, action='edit_product',
                    target=updated.name, detail=f'Edited via check-in inline (Session #{session.pk})')
                messages.success(request, f"Updated {updated.name}.", extra_tags="checkin success")
                return redirect(f"{session_url}?barcode={updated.barcode}")

        messages.error(request, "Could not update product. Please review the highlighted fields.", extra_tags="checkin error")
        return render(request, self.template_name, {
            "session": session,
            "search_results": [],
            "inventory_mode": inventory_mode,
            "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number", "barcode")),
            "product": product,
            "edit_form": form,
            "categories": Category.objects.all(),
            "change_types": StockChange._meta.get_field('change_type').choices,
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
            UserAction.objects.create(user=request.user, action='revert_label_category',
                target=f'{reverted_count} products reverted')
            messages.success(request, f"Reverted {reverted_count} products to their original categories.")
        else:
            messages.info(request, "No products had a stored category to revert to.")

        return redirect('label_printing')


# ── Label Session History API ──────────────────────────────────
class LabelSessionListView(LoginRequiredMixin, View):
    """GET → JSON list of user's label sessions (most recent first)."""
    def get(self, request):
        sessions = LabelSession.objects.filter(user=request.user).order_by('-created_at')[:50]
        data = []
        for s in sessions:
            data.append({
                'id': s.pk,
                'created_at': s.created_at.strftime('%b %d, %Y %I:%M %p'),
                'label_count': s.label_count,
                'note': s.note,
            })
        return JsonResponse({'sessions': data})


class LabelSessionDetailView(LoginRequiredMixin, View):
    """GET → JSON detail of a single session with all its items."""
    def get(self, request, session_id):
        session_obj = get_object_or_404(LabelSession, pk=session_id, user=request.user)
        items = session_obj.items.all()
        data = {
            'id': session_obj.pk,
            'created_at': session_obj.created_at.strftime('%b %d, %Y %I:%M %p'),
            'label_count': session_obj.label_count,
            'note': session_obj.note,
            'items': [{
                'product_name': i.product_name,
                'product_barcode': i.product_barcode,
                'product_price': str(i.product_price),
                'product_brand': i.product_brand,
                'product_item_number': i.product_item_number,
                'qty': i.qty,
                'product_exists': i.product_id is not None,
            } for i in items],
        }
        return JsonResponse(data)


class LabelSessionDeleteView(LoginRequiredMixin, View):
    """POST → delete a session."""
    def post(self, request, session_id):
        session_obj = get_object_or_404(LabelSession, pk=session_id, user=request.user)
        session_obj.delete()
        UserAction.objects.create(user=request.user, action='delete_label_session',
            target=f'Label Session #{session_id}')
        return JsonResponse({'ok': True})


class LabelSessionRegenerateView(LoginRequiredMixin, View):
    """POST → reload session items back into the current label queue."""
    def post(self, request, session_id):
        session_obj = get_object_or_404(LabelSession, pk=session_id, user=request.user)
        items = list(session_obj.items.filter(product__isnull=False).select_related('product'))
        if not items:
            return JsonResponse({'ok': False, 'error': 'No active products in this session.'}, status=400)

        # Clear current queue and reload from snapshot
        LabelQueueItem.objects.filter(user=request.user).delete()
        LabelQueueItem.objects.bulk_create([
            LabelQueueItem(product=i.product, user=request.user, qty=i.qty)
            for i in items
        ])
        UserAction.objects.create(user=request.user, action='regenerate_label_session',
            target=f'Label Session #{session_id}', detail=f'{len(items)} items loaded')
        return JsonResponse({'ok': True, 'loaded': len(items)})


class LabelSessionAddToQueueView(LoginRequiredMixin, View):
    """POST → append session items to the current queue (without clearing it)."""
    def post(self, request, session_id):
        session_obj = get_object_or_404(LabelSession, pk=session_id, user=request.user)
        items = list(session_obj.items.filter(product__isnull=False).select_related('product'))
        if not items:
            return JsonResponse({'ok': False, 'error': 'No active products in this session.'}, status=400)

        LabelQueueItem.objects.bulk_create([
            LabelQueueItem(product=i.product, user=request.user, qty=i.qty)
            for i in items
        ])
        return JsonResponse({'ok': True, 'added': len(items)})


class LabelSessionClearAllView(LoginRequiredMixin, View):
    """POST → delete all sessions for this user."""
    def post(self, request):
        deleted_count, _ = LabelSession.objects.filter(user=request.user).delete()
        UserAction.objects.create(user=request.user, action='clear_all_label_sessions',
            target=f'{deleted_count} label sessions cleared')
        return JsonResponse({'ok': True, 'deleted': deleted_count})


# Edit product.
class EditProductView(LoginRequiredMixin, View):
    template_name = 'edit_product.html'

    def get(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        form = EditProductForm(instance=product)
        extra_dates = product.expiry_dates.all()

        next_url = request.GET.get('next') or request.META.get(
            'HTTP_REFERER', '/inventory_display'
        )

        return render(request, self.template_name, {
            'form': form,
            'next': next_url,
            'product': product,
            'extra_dates': extra_dates,
        })


    def post(self, request, product_id):
            product = get_object_or_404(Product, product_id=product_id)

            old_category = product.category

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

            with transaction.atomic():
                # Lock the row and read the authoritative pre-edit stock UNDER the
                # lock, so the delta calc, audit row, and save are one race-free unit.
                old_quantity = (
                    Product.objects.select_for_update()
                    .values_list("quantity_in_stock", flat=True)
                    .get(product_id=product_id)
                )

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

                _save_expiry_dates(updated_product, updated_product.expiry_date, request.POST.getlist('extra_expiry_dates'))

            UserAction.objects.create(user=request.user, action='edit_product',
                target=updated_product.name, detail='Edited via product form')
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

        # Catalog suggested retail + implied markup over wholesale cost — shown as
        # an informational hover tooltip next to the Retail Selling Price field.
        # The raw catalogue value is snapped to the nearest price ending in .99
        # (e.g. 12.34 → 11.99, 12.60 → 12.99) so the suggestion matches shelf
        # pricing conventions while staying closest to the catalogue's markup.
        suggested_retail = request.GET.get('suggested_retail', '').strip()
        wholesale_cost = (request.GET.get('price_per_unit', '') or '').strip()
        suggested_markup = None
        if suggested_retail:
            try:
                raw = Decimal(suggested_retail)
                if raw > 0:
                    snapped = (raw + Decimal('0.01')).quantize(
                        Decimal('1'), rounding=ROUND_HALF_UP
                    ) - Decimal('0.01')
                    suggested_retail = f"{max(snapped, Decimal('0.99')):.2f}"
            except Exception:
                pass
        if suggested_retail and wholesale_cost:
            try:
                retail, cost = Decimal(suggested_retail), Decimal(wholesale_cost)
                if cost > 0:
                    suggested_markup = round((retail - cost) / cost * 100)
            except Exception:
                pass

        return render(request, self.template_name, {
            'categories': categories,
            'form': form,
            'next': next_url,
            'suggested_retail': suggested_retail,
            'suggested_markup': suggested_markup,
            'wholesale_cost': wholesale_cost,
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
                    
                    _save_expiry_dates(product, product.expiry_date, request.POST.getlist('extra_expiry_dates'))

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
        products = Product.objects.select_related('category').prefetch_related('expiry_dates').annotate(
            stock_threshold=Coalesce(F('category__low_stock_threshold'), Value(3))
        )
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
        page_size = 200 if is_ajax else 100
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

        products = Product.objects.select_related('category').prefetch_related('expiry_dates').all()

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
                '; '.join(d.expiry_date.strftime('%Y-%m-%d') for d in p.expiry_dates.all()) or (p.expiry_date.strftime('%Y-%m-%d') if p.expiry_date else ''),
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

        # barcode_search_q makes the barcode match leading-zero-tolerant, so a
        # scanned '066259042505' still finds a product stored as '66259042505'.
        filters = Q(name__icontains=q) | barcode_search_q(q) | Q(item_number__icontains=q)

        products = Product.objects.filter(filters).values(
            'product_id', 'name', 'barcode', 'quantity_in_stock', 'price', 'status'
        )[:8]
        return JsonResponse({'results': list(products)})


class ProductDetailAPIView(LoginRequiredMixin, View):
    """AJAX endpoint returning product info + 6-month sales chart data."""
    def get(self, request):
        pid = request.GET.get('id', '').strip()
        if not pid:
            return JsonResponse({'error': 'Missing id'}, status=400)
        try:
            product = Product.objects.select_related('category').prefetch_related('expiry_dates').get(product_id=pid)
        except Product.DoesNotExist:
            return JsonResponse({'error': 'Not found'}, status=404)

        margin = None
        if product.price_per_unit and product.price:
            margin = round(float((product.price - product.price_per_unit) / product.price * 100), 1)

        try:
            end_date = datetime.strptime(request.GET.get('end', ''), '%Y-%m-%d').date()
        except (TypeError, ValueError):
            end_date = date.today()
        try:
            start_date = datetime.strptime(request.GET.get('start', ''), '%Y-%m-%d').date()
        except (TypeError, ValueError):
            start_date = end_date - timedelta(days=180)

        periods, sold, restocked, expired = self._chart_data(product, start_date, end_date)

        recent_sales = OrderDetail.objects.filter(
            product=product,
            order__submitted=True,
            order__order_date__date__gte=end_date - timedelta(days=30),
        ).aggregate(total=Sum('quantity'))['total'] or 0

        # Net units restocked in the last 30 days, mirroring the chart's
        # "Restocked" series (check-ins add, correction-removals subtract).
        recent_bought = 0
        for sc in (StockChange.objects
                   .filter(product=product,
                           timestamp__date__gte=end_date - timedelta(days=30),
                           timestamp__date__lte=end_date,
                           change_type__in=['checkin', 'error_add', 'error_subtract', 'checkin_delete1'])
                   .values('change_type', 'quantity')):
            qty = abs(sc['quantity'] or 0)
            if sc['change_type'] in ('checkin', 'error_add'):
                recent_bought += qty
            else:
                recent_bought -= qty

        info = {
            'product_id': product.product_id,
            'name': product.name,
            'barcode': product.barcode or '',
            'brand': product.brand or '',
            'item_number': product.item_number or '',
            'category': product.category.name if product.category else '',
            'unit_size': product.unit_size or '',
            'price': float(product.price),
            'price_per_unit': float(product.price_per_unit) if product.price_per_unit else None,
            'margin': margin,
            'quantity_in_stock': product.quantity_in_stock,
            'stock_sold': product.stock_sold,
            'stock_bought': product.stock_bought,
            'stock_expired': product.stock_expired,
            'stock_unfulfilled': product.stock_unfulfilled,
            'expiry_date': product.expiry_date.isoformat() if product.expiry_date else None,
            'expiry_dates': [d.expiry_date.isoformat() for d in product.expiry_dates.all()],
            'taxable': product.taxable,
            'status': product.status,
            'recent_sales_30d': recent_sales,
            'recent_bought_30d': recent_bought,
            'chart': {
                'periods': periods,
                'sold': sold,
                'restocked': restocked,
                'expired': expired,
            },
        }
        return JsonResponse(info)

    def _chart_data(self, product, start_date, end_date):
        qs = (
            StockChange.objects.filter(
                product=product,
                timestamp__date__gte=start_date,
                timestamp__date__lte=end_date,
            )
            .annotate(period=TruncMonth('timestamp'))
            .values('period', 'change_type')
            .annotate(total=Sum('quantity'))
            .order_by('period')
        )

        periods = []
        current = start_date.replace(day=1)
        while current <= end_date:
            periods.append(current.strftime('%b %Y'))
            current = (current + timedelta(days=32)).replace(day=1)

        length = len(periods)
        sold = [0] * length
        restocked = [0] * length
        expired = [0] * length
        label_to_idx = {label: i for i, label in enumerate(periods)}

        for row in qs:
            label = row['period'].date().strftime('%b %Y')
            idx = label_to_idx.get(label)
            if idx is None:
                continue
            ctype = row['change_type']
            qty = row['total'] or 0
            if ctype == 'checkout':
                sold[idx] += abs(qty)
            elif ctype in ('checkin', 'error_add'):
                restocked[idx] += qty
            elif ctype in ('error_subtract', 'checkin_delete1'):
                restocked[idx] -= abs(qty)
            elif ctype == 'expired':
                expired[idx] += abs(qty)

        return periods, sold, restocked, expired


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
            qs = qs.filter(Q(product__name__icontains=product_query) | barcode_search_q(product_query, 'product__barcode'))
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
                    sc.display_name,
                    sc.display_barcode,
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


# Change
class ExpiredProductView(LoginRequiredMixin, View):
    template_name = 'expired_products.html'

    def get(self, request):
        date_filter = request.GET.get('date_filter', '')
        name_query = request.GET.get('name_query', '').strip()
        sort = request.GET.get('sort', 'expiry_date')
        pid = request.GET.get("pid", None)
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')

        products = self._filter_products(date_filter, name_query, sort, date_from=date_from, date_to=date_to)
        product = (Product.objects.filter(pk=pid).select_related('category').prefetch_related('expiry_dates').first()
                   if pid else None)

        # Per-product expiry breakdown for the log-mode detail card.
        product_extra = self._product_expiry_summary(product) if product else None

        # Aggregate stats
        exp_agg = products.aggregate(
            total_units=Sum('quantity_in_stock'),
            value_at_risk=Sum(F('price') * F('quantity_in_stock')),
            total_expired_units=Sum('stock_expired'),
        )

        # Recent expired log entries
        expired_logs = (
            StockChange.objects.filter(change_type="expired")
            .select_related("product", "user")
            .order_by("-timestamp")[:50]
        )

        return render(request, self.template_name, {
            "products": products,
            "product": product,
            "product_extra": product_extra,
            "date_filter": date_filter,
            "name_query": name_query,
            "sort": sort,
            "date_from": date_from,
            "date_to": date_to,
            "all_products": list(Product.objects.values("product_id", "name", "barcode", "item_number", "price", "quantity_in_stock")),
            "product_count": products.count(),
            "total_units_on_shelf": exp_agg['total_units'] or 0,
            "value_at_risk": exp_agg['value_at_risk'] or Decimal('0.00'),
            "total_expired_units": exp_agg['total_expired_units'] or 0,
            "expired_logs": expired_logs,
        })

    @staticmethod
    def _product_expiry_summary(product):
        """Expiry breakdown for the loaded product: per-lot status + value at risk.

        Each lot is tagged 'expired' (past), 'soon' (≤30 days) or 'ok'. The
        overall status mirrors the earliest (most urgent) lot. `days` is signed:
        negative = days since expiry, positive = days until expiry.
        """
        today = date.today()
        lots = list(product.expiry_dates.order_by('expiry_date').values_list('expiry_date', flat=True))
        if not lots and product.expiry_date:
            lots = [product.expiry_date]

        def classify(d):
            delta = (d - today).days
            if delta < 0:
                return 'expired', delta
            if delta <= 30:
                return 'soon', delta
            return 'ok', delta

        lot_rows = []
        for d in lots:
            status, delta = classify(d)
            lot_rows.append({'date': d, 'days': delta, 'days_abs': abs(delta), 'status': status})

        value = (product.price or Decimal('0.00')) * product.quantity_in_stock
        return {
            'lots': lot_rows,
            'status': lot_rows[0]['status'] if lot_rows else 'none',
            'days': lot_rows[0]['days'] if lot_rows else None,
            'days_abs': lot_rows[0]['days_abs'] if lot_rows else None,
            'value': value,
        }

    def post(self, request):
        barcode = request.POST.get("barcode", "").strip()
        product = None

        if not barcode:
            messages.warning(request, "Scan or type a barcode first.")
        else:
            product = find_product_by_barcode(barcode)
            if not product:
                messages.error(request, f"No product found with barcode '{barcode}'.")

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
                        UserAction.objects.create(user=request.user, action='retire_expired',
                            target=product.name, detail=f'{qty} units retired')

                        messages.success(
                            request,
                            f"{qty} unit{'s' if qty != 1 else ''} of '{product.name}' marked as "
                            f"expired — {product.quantity_in_stock} left in stock."
                        )
                        # Tell staff what to physically do with the retired units.
                        messages.info(
                            request,
                            "Next: pull the expired units off the shelf, bag them and mark "
                            "them with today's date, then place them in the expired-returns "
                            "bin. Do not sell or restock them."
                        )
                        # Guard against mis-scans: flag when the product isn't
                        # actually past its earliest expiry date yet.
                        if product.expiry_date and product.expiry_date >= date.today():
                            messages.warning(
                                request,
                                f"Heads up: this product's earliest expiry is "
                                f"{product.expiry_date.strftime('%d %b %Y')} — it is not "
                                "expired yet. Undo via Check-in if this was a mis-scan."
                            )

        # Post/Redirect/Get: bounce back to the GET handler so the page is
        # rebuilt with the full context — including a fresh `expired_logs`
        # query, so the pull-out Expired Log reflects what was just retired.
        # Also avoids re-submitting the retire on refresh. Messages survive
        # the redirect via the messages framework.
        redirect_url = f"{reverse('expired_products')}?mode=log"
        if product:
            redirect_url += f"&pid={product.pk}"
        return redirect(redirect_url)

    ALLOWED_SORTS = {"expiry_date", "-expiry_date", "name", "-name", "barcode", "-barcode", "category__name", "-category__name"}

    def _filter_products(self, date_filter, name_query, sort="expiry_date", date_from=None, date_to=None):
        today = date.today()
        if date_filter == "custom" and (date_from or date_to):
            try:
                qs = Product.objects.all()
                if date_from:
                    from_dt = date.fromisoformat(date_from)
                    qs = qs.filter(expiry_date__gte=from_dt)
                if date_to:
                    to_dt = date.fromisoformat(date_to)
                    qs = qs.filter(expiry_date__lte=to_dt)
            except (ValueError, TypeError):
                qs = Product.objects.filter(expiry_date__lt=today)
        elif date_filter == "1_week":
            end = today + timedelta(weeks=1)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "2_weeks":
            end = today + timedelta(weeks=2)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "1_month":
            end = today + relativedelta(months=1)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "2_months":
            end = today + relativedelta(months=2)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "3_months":
            end = today + relativedelta(months=3)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        else:
            qs = Product.objects.filter(expiry_date__lt=today)

        if name_query:
            qs = qs.filter(name__icontains=name_query)

        order_field = sort if sort in self.ALLOWED_SORTS else "expiry_date"
        return qs.exclude(expiry_date__isnull=True).select_related('category').prefetch_related('expiry_dates').order_by(order_field)


class ExpiredProductPDFView(LoginRequiredMixin, View):
    """Generate a PDF report for expired / expiring products."""

    FILTER_TITLES = {
        "": "Expired Products",
        "1_week": "Expiring in 1 Week",
        "2_weeks": "Expiring in 2 Weeks",
        "1_month": "Expiring in 1 Month",
        "2_months": "Expiring in 2 Months",
        "3_months": "Expiring in 3 Months",
    }

    ALLOWED_SORTS = {"expiry_date", "-expiry_date", "name", "-name", "barcode", "-barcode", "category__name", "-category__name"}

    def get(self, request):
        date_filter = request.GET.get("date_filter", "")
        sort = request.GET.get("sort", "expiry_date")
        today = date.today()

        # ── Filter queryset (same logic as ExpiredProductView) ──
        if date_filter == "1_week":
            end = today + timedelta(weeks=1)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "2_weeks":
            end = today + timedelta(weeks=2)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "1_month":
            end = today + relativedelta(months=1)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "2_months":
            end = today + relativedelta(months=2)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "3_months":
            end = today + relativedelta(months=3)
            qs = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        else:
            qs = Product.objects.filter(expiry_date__lt=today)

        order_field = sort if sort in self.ALLOWED_SORTS else "expiry_date"
        products = qs.exclude(expiry_date__isnull=True).select_related("category").order_by(order_field)

        # ── Aggregate KPIs ──
        agg = products.aggregate(
            total_units=Sum("quantity_in_stock"),
            value_at_risk=Sum(F("price") * F("quantity_in_stock")),
        )
        product_count = products.count()
        total_units = agg["total_units"] or 0
        value_at_risk = float(agg["value_at_risk"] or 0)

        # ── Build PDF ──
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        page_w, page_h = letter
        margin = 36

        report_title = self.FILTER_TITLES.get(date_filter, "Expired Products")

        # --- Title ---
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, page_h - margin, f"{report_title} Report")

        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.39, 0.45, 0.55)
        c.drawString(margin, page_h - margin - 18, f"Generated on {today.strftime('%B %d, %Y')}")

        # --- KPI summary line ---
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.39, 0.45, 0.55)
        kpi_line = f"{product_count} products  ·  {total_units} units on shelf  ·  ${value_at_risk:,.2f} value at risk"
        c.drawString(margin, page_h - margin - 36, kpi_line)

        # --- Table header ---
        usable = page_w - 2 * margin
        table_top = page_h - margin - 56
        cols = [
            ("Expiry Date", margin, 85),
            ("Name", margin + 85, 200),
            ("Category", margin + 285, 95),
            ("Barcode", margin + 380, 90),
            ("Price", margin + 470, 45),
            ("Qty", margin + 515, usable - 515 + margin),
        ]

        row_h = 18
        c.setFillColorRGB(0.95, 0.96, 0.98)
        c.rect(margin, table_top - row_h, page_w - 2 * margin, row_h, stroke=0, fill=1)

        c.setFillColorRGB(0.39, 0.45, 0.55)
        c.setFont("Helvetica-Bold", 7)
        for col_name, col_x, col_w in cols:
            c.drawString(col_x + 4, table_top - row_h + 6, col_name.upper())

        # --- Table rows ---
        y = table_top - row_h
        c.setFont("Helvetica", 8)

        for p in products:
            y -= row_h
            if y < margin + 20:
                # New page
                c.showPage()
                c.setFont("Helvetica", 8)
                y = page_h - margin

            # Alternating row bg
            c.setFillColorRGB(0.06, 0.09, 0.16)

            # Separator line
            c.setStrokeColorRGB(0.89, 0.91, 0.94)
            c.line(margin, y, page_w - margin, y)

            expiry_str = p.expiry_date.strftime("%b %d, %Y") if p.expiry_date else "N/A"
            cat_name = p.category.name if p.category else "--"
            name_display = p.name[:38] + "..." if len(p.name) > 38 else p.name

            row_data = [
                (expiry_str, cols[0][1]),
                (name_display, cols[1][1]),
                (cat_name, cols[2][1]),
                (str(p.barcode or ""), cols[3][1]),
                (f"${p.price:.2f}", cols[4][1]),
                (str(p.quantity_in_stock), cols[5][1]),
            ]

            for val, col_x in row_data:
                c.drawString(col_x + 4, y + 5, val)

        c.save()
        buffer.seek(0)

        filename = f"{report_title.lower().replace(' ', '_')}_report_{today.strftime('%Y%m%d')}.pdf"
        response = HttpResponse(buffer, content_type="application/pdf")
        response["Content-Disposition"] = f'inline; filename="{filename}"'
        return response


class ExpiredLogPDFView(LoginRequiredMixin, View):
    """Generate a PDF of expired stock log entries, optionally filtered by date range."""

    def _fmt_date(self, d):
        """Format a date string (YYYY-MM-DD) to readable form."""
        try:
            return date.fromisoformat(d).strftime("%b %d, %Y")
        except (ValueError, TypeError):
            return d

    def get(self, request):
        date_from = request.GET.get("from", "").strip()
        date_to = request.GET.get("to", "").strip()
        today = date.today()

        qs = StockChange.objects.filter(change_type="expired").select_related("product", "user").order_by("-timestamp")

        if date_from:
            try:
                qs = qs.filter(timestamp__date__gte=date.fromisoformat(date_from))
            except ValueError:
                pass
        if date_to:
            try:
                qs = qs.filter(timestamp__date__lte=date.fromisoformat(date_to))
            except ValueError:
                pass

        logs = list(qs[:200])

        # Build date range label
        if date_from and date_to:
            date_range = f"{self._fmt_date(date_from)} — {self._fmt_date(date_to)}"
        elif date_from:
            date_range = f"{self._fmt_date(date_from)} — Present"
        elif date_to:
            date_range = f"Up to {self._fmt_date(date_to)}"
        else:
            date_range = f"All records up to {today.strftime('%b %d, %Y')}"

        total_qty = sum(abs(l.quantity) for l in logs)
        total_value = sum(abs(l.quantity) * float(l.product.price) for l in logs if l.product)

        # PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        page_w, page_h = letter
        margin = 36

        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, page_h - margin, "Expired Stock Log")

        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.39, 0.45, 0.55)
        c.drawString(margin, page_h - margin - 18, f"Generated on {today.strftime('%B %d, %Y')}")

        c.setFont("Helvetica", 9)
        c.drawString(margin, page_h - margin - 32, f"Date range: {date_range}")

        c.drawString(margin, page_h - margin - 46, f"{len(logs)} entries  ·  {total_qty} total units  ·  ${total_value:,.2f} total value")

        # Table header
        usable = page_w - 2 * margin
        table_top = page_h - margin - 66
        cols = [
            ("Date", margin, 100),
            ("Product", margin + 100, 185),
            ("Qty", margin + 285, 35),
            ("Price", margin + 320, 50),
            ("Value", margin + 370, 55),
            ("User", margin + 425, 60),
            ("Note", margin + 485, usable - 485 + margin),
        ]

        row_h = 17
        c.setFillColorRGB(0.95, 0.96, 0.98)
        c.rect(margin, table_top - row_h, page_w - 2 * margin, row_h, stroke=0, fill=1)

        c.setFillColorRGB(0.39, 0.45, 0.55)
        c.setFont("Helvetica-Bold", 7)
        for col_name, col_x, col_w in cols:
            c.drawString(col_x + 4, table_top - row_h + 5, col_name.upper())

        y = table_top - row_h
        c.setFont("Helvetica", 8)

        for log in logs:
            y -= row_h
            if y < margin + 20:
                c.showPage()
                c.setFont("Helvetica", 8)
                y = page_h - margin

            c.setFillColorRGB(0.06, 0.09, 0.16)
            c.setStrokeColorRGB(0.89, 0.91, 0.94)
            c.line(margin, y, page_w - margin, y)

            ts = log.timestamp.strftime("%b %d, %Y %H:%M") if log.timestamp else ""
            product_name = log.product.name if log.product else "Deleted"
            name_display = product_name[:35] + "..." if len(product_name) > 35 else product_name
            qty = abs(log.quantity)
            price = float(log.product.price) if log.product else 0
            line_value = qty * price
            user_name = log.user.username if log.user else "—"
            note = (log.note or "—")[:20]

            row_data = [
                (ts, cols[0][1]),
                (name_display, cols[1][1]),
                (f"-{qty}", cols[2][1]),
                (f"${price:.2f}", cols[3][1]),
                (f"${line_value:.2f}", cols[4][1]),
                (user_name, cols[5][1]),
                (note, cols[6][1]),
            ]
            for val, col_x in row_data:
                c.drawString(col_x + 4, y + 4, val)

        c.save()
        buffer.seek(0)

        filename = f"expired_log_{today.strftime('%Y%m%d')}.pdf"
        response = HttpResponse(buffer, content_type="application/pdf")
        response["Content-Disposition"] = f'inline; filename="{filename}"'
        return response


# View for displaying low-stock items
class LowStockView(AdminRequiredMixin, View):
    template_name = 'low_stock.html'

    # Keys match the table's <th data-sort> column numbering in low_stock.html
    SORT_FIELDS = {
        '1': 'product__brand',
        '2': 'product__name',
        '3': 'product__barcode',
        '4': 'product__item_number',
        '5': 'quantity',
        '6': 'product__quantity_in_stock',
    }

    def get(self, request):
        low_stock_products = Product.objects.filter(
            status=True
        ).annotate(
            _threshold=Coalesce(F('category__low_stock_threshold'), Value(3))
        ).filter(quantity_in_stock__lte=F('_threshold')).order_by('name')

        q = request.GET.get('q', '').strip()
        category_filter = request.GET.get('category', '').strip()
        sort_col = request.GET.get('sort', '').strip()
        sort_dir = request.GET.get('dir', 'asc').strip()
        hide_snacks = request.GET.get('hide_snacks', '').strip()
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        active_categories = list(
            Category.objects
            .filter(product__recentlypurchasedproduct__isnull=False)
            .distinct().order_by('name')
            .values_list('id', 'name')
        )

        # Build sort order
        order_field = self.SORT_FIELDS.get(sort_col)
        if order_field:
            if sort_dir == 'desc':
                order_field = '-' + order_field
            ordering = [order_field, '-order_date']
        else:
            ordering = ['-order_date']

        recently_purchased = (
            RecentlyPurchasedProduct.objects
            .all()
            .order_by(*ordering)
            .select_related('product', 'product__category')
        )

        if hide_snacks == '1':
            recently_purchased = recently_purchased.exclude(
                product__category__name__iexact='Snacks'
            )

        if q:
            recently_purchased = recently_purchased.filter(
                Q(product__name__icontains=q) |
                barcode_search_q(q, 'product__barcode') |
                Q(product__brand__icontains=q)
            )
        if category_filter:
            cat_ids = [c.strip() for c in category_filter.split(',') if c.strip()]
            if len(cat_ids) == 1:
                recently_purchased = recently_purchased.filter(product__category_id=cat_ids[0])
            elif cat_ids:
                recently_purchased = recently_purchased.filter(product__category_id__in=cat_ids)

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

        # Q4 — units bought (ordered) in the last 60 days, per product. Mirrors
        # how RecentlyPurchasedProduct.quantity is accumulated (sum of submitted
        # order line quantities), windowed to 60 days for the "Bought" column.
        bought_map = {
            row['product_id']: row['total']
            for row in OrderDetail.objects
            .filter(
                product_id__in=page_product_ids,
                order__submitted=True,
                order__order_date__date__gte=today - timedelta(days=60),
            )
            .values('product_id')
            .annotate(total=Sum('quantity'))
        }

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
            item.bought_60d = bought_map.get(item.product_id, 0)
        # The per-item movement chart is loaded on demand (with a range filter)
        # from RecentlyPurchasedChartAPIView when a row is expanded.
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
                'sort': sort_col,
                'dir': sort_dir,
                'hide_snacks': hide_snacks,
            })

        return render(request, self.template_name, {
            'page_obj_low_stock': page_obj_low_stock,
            'page_obj_recent':    page_obj_recent,
            'q':                  q,
            'active_categories':  active_categories,
            'sort':               sort_col,
            'dir':                sort_dir,
            'hide_snacks':        hide_snacks,
        })


class RecentlyPurchasedChartAPIView(AdminRequiredMixin, View):
    """Movement chart data (sold vs restocked) for one product over a range.

    Powers the per-item dropdown chart on the Recently Purchased page. Short
    ranges bucket by week; "all time" buckets by month so the payload stays
    small over long histories.
    """
    RANGE_DAYS = {'1m': 30, '3m': 90, '6m': 180}

    def get(self, request):
        product = Product.objects.filter(pk=request.GET.get('product_id')).first()
        if not product:
            return JsonResponse({'error': 'Product not found'}, status=404)

        rng = request.GET.get('range', '3m')
        today = date.today()
        if rng == 'all':
            trunc, start = TruncMonth('timestamp'), None
        else:
            trunc = TruncWeek('timestamp')
            start = today - timedelta(days=self.RANGE_DAYS.get(rng, 90))

        def series(change_types):
            qs = StockChange.objects.filter(product=product, change_type__in=change_types)
            if start:
                qs = qs.filter(timestamp__date__gte=start)
            rows = (qs.annotate(bucket=trunc)
                      .values('bucket')
                      .annotate(total=Sum('quantity'))
                      .order_by('bucket'))
            return [{'week': r['bucket'].strftime('%Y-%m-%d'), 'qty': r['total'] or 0}
                    for r in rows if r['bucket']]

        return JsonResponse({
            'range': rng,
            'bucket': 'month' if rng == 'all' else 'week',
            'sold': series(['checkout', 'checkout_unfulfilled']),
            'restocked': series(['checkin', 'error_add']),
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
        'checkout_unfulfilled': ['checkout_unfulfilled'],
    }
    ACTION_TYPE_MAP = {
        'delete_product': ['delete_product'],
        'delete_order': ['delete_order', 'delete_all_orders'],
        'delete_recently_purchased': ['delete_recently_purchased', 'delete_all_recently_purchased', 'bulk_delete_recently_purchased'],
        'submit_order': ['submit_order'],
        'add_product': ['add_product'],
        'edit_product': ['edit_product', 'update_product_settings'],
        'session_ops': ['start_session', 'end_session', 'reopen_session', 'adjust_session_line', 'remove_session_line', 'delete_session', 'clear_session_history'],
        'delivery_ops': ['delivery_checkin', 'delivery_checkout', 'delivery_undo_checkout', 'delivery_clear_history', 'delivery_delete_record'],
        'item_list_ops': ['delete_item_list', 'add_item_list'],
        'revert_label_category': ['revert_label_category'],
        'create_account': ['create_account'],
        'clear_label_queue': ['clear_label_queue'],
        'label_session_ops': ['print_labels', 'delete_label_session', 'regenerate_label_session', 'clear_all_label_sessions'],
        'cycle_count': ['cycle_count'],
        'retire_expired': ['retire_expired'],
    }
    SESSION_ACTIONS = {'start_session', 'end_session', 'reopen_session', 'adjust_session_line', 'remove_session_line', 'delete_session', 'clear_session_history'}
    DELIVERY_ACTIONS = {'delivery_checkin', 'delivery_checkout', 'delivery_undo_checkout', 'delivery_clear_history', 'delivery_delete_record'}
    LOGIN_TYPES = ('', 'all_logins', 'login', 'login_success', 'login_failed')
    STOCK_TYPES = ('', 'all_stock')
    ACTION_TYPES = ('', 'all_actions')

    def _build_events(self, event_type, user_filter, parsed_from, parsed_to):
        events = []
        include_logins = event_type in self.LOGIN_TYPES or event_type in ('login_success', 'login_failed')
        include_stock = event_type in self.STOCK_TYPES or event_type in self.STOCK_TYPE_MAP
        include_actions = event_type in self.ACTION_TYPES or event_type in self.ACTION_TYPE_MAP or event_type in ('all_sessions', 'all_delivery')

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
                product_name = sc.display_name
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
                    link = f"{reverse('checkin_dashboard')}?barcode={sc.product.barcode}"
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
            elif event_type == 'all_sessions':
                action_qs = action_qs.filter(action__in=self.SESSION_ACTIONS)
            elif event_type == 'all_delivery':
                action_qs = action_qs.filter(action__in=self.DELIVERY_ACTIONS)
            for ua in action_qs[:500]:
                user_display = ua.user.username if ua.user else '—'
                # Badge logic
                if 'delete' in ua.action or 'clear' in ua.action or 'remove' in ua.action:
                    badge = 'deletion'
                elif ua.action == 'submit_order':
                    badge = 'checkout'
                elif ua.action in ('add_product', 'create_account'):
                    badge = 'checkin'
                elif ua.action in self.SESSION_ACTIONS:
                    badge = 'session'
                elif ua.action in self.DELIVERY_ACTIONS:
                    badge = 'delivery'
                elif ua.action in ('edit_product', 'update_product_settings', 'revert_label_category'):
                    badge = 'other'
                else:
                    badge = 'other'
                # Category label
                if ua.action in self.SESSION_ACTIONS:
                    category = 'Session'
                elif ua.action in self.DELIVERY_ACTIONS:
                    category = 'Delivery'
                else:
                    category = 'Action'
                # Build link based on action type
                link = ''
                if ua.action == 'submit_order':
                    m = re.search(r'#(\d+)', ua.target)
                    if m:
                        link = reverse('order_detail', args=[int(m.group(1))])
                elif ua.action in ('add_product', 'edit_product', 'update_product_settings'):
                    try:
                        prod = Product.objects.filter(name=ua.target).first()
                        if prod and prod.barcode:
                            link = f"{reverse('checkin_dashboard')}?barcode={prod.barcode}"
                    except Exception:
                        pass
                elif ua.action in ('start_session', 'end_session', 'reopen_session', 'adjust_session_line', 'remove_session_line'):
                    m = re.search(r'#(\d+)', ua.target)
                    if m:
                        try:
                            link = reverse('checkin_session_detail', args=[int(m.group(1))])
                        except Exception:
                            pass
                elif ua.action in self.DELIVERY_ACTIONS and ua.action != 'delivery_clear_history':
                    link = reverse('delivery')
                events.append({
                    'timestamp': ua.timestamp,
                    'category': category,
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
            'checkout_unfulfilled': 'Unfulfilled Sale',
            'delete_product': 'Delete Product', 'delete_order': 'Delete Order',
            'delete_recently_purchased': 'Delete Recently Purchased',
            'submit_order': 'Submit Order', 'add_product': 'New Product',
            'edit_product': 'Edit Product', 'session_ops': 'All Session Operations',
            'delivery_ops': 'All Delivery Operations',
            'all_sessions': 'All Sessions', 'all_delivery': 'All Delivery',
            'revert_label_category': 'Revert Label Category',
            'create_account': 'New Account', 'clear_label_queue': 'Clear Label Queue',
            'item_list_ops': 'Item List Operations', 'all_item_list': 'All Item List',
            'label_session_ops': 'Label Session Operations',
            'cycle_count': 'Cycle Count', 'retire_expired': 'Retired Expired',
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
        # Soft delete: hide all currently-visible orders from the list while
        # preserving their data (OrderDetail, StockChange ledger, counters) so
        # reports and reorder predictions keep working. IDs are NOT reset because
        # the rows still exist.
        order_count = Order.objects.filter(is_deleted=False).update(
            is_deleted=True, deleted_at=now(), deleted_by=request.user,
        )
        UserAction.objects.create(
            user=request.user, action='delete_all_orders',
            target=f'{order_count} orders',
        )

        # Clear session references to the in-progress order/cart.
        if 'order_id' in request.session:
            request.session.pop('order_id')
        if 'cart' in request.session:
            request.session.pop('cart')
        request.session.modified = True

        messages.success(
            request,
            f"{order_count} order(s) removed from the list. History is preserved for reports."
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

        # Soft delete: hide from the order list but keep the data so reports and
        # reorder predictions are unaffected. Stock, ledger, and counters are
        # intentionally left untouched.
        order.is_deleted = True
        order.deleted_at = now()
        order.deleted_by = request.user
        order.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])
        UserAction.objects.create(
            user=request.user, action='delete_order',
            target=f'Order #{order_id}',
        )
        messages.success(request, f"Order #{order_id} has been removed from the list.")
        return redirect('order_view')


# Restore a soft-deleted order back to the list
class RestoreOrderView(AdminRequiredMixin, View):
    def post(self, request, order_id):
        order = get_object_or_404(Order, order_id=order_id)
        order.is_deleted = False
        order.deleted_at = None
        order.deleted_by = None
        order.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])
        UserAction.objects.create(
            user=request.user, action='restore_order',
            target=f'Order #{order_id}',
        )
        messages.success(request, f"Order #{order_id} has been restored.")
        return redirect(f"{reverse('order_view')}?status=deleted")


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
           item_name = item.item_name
           item.delete()
           UserAction.objects.create(user=request.user, action='delete_item_list',
               target=item_name, detail=f'{item.first_name} {item.last_name}')
           messages.success(request, f"Item '{item_name}' has been deleted.")
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
               new_item = form.save()
               UserAction.objects.create(user=request.user, action='add_item_list',
                   target=new_item.item_name, detail=f'{new_item.first_name} {new_item.last_name}')
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
            raw_barcode = request.POST.get('barcode', '').strip()
            first_name = request.POST.get('first_name', '').strip()
            last_name = request.POST.get('last_name', '').strip()
            comment = request.POST.get('comment', '').strip()

            no_barcode = _is_no_barcode(raw_barcode)
            barcode = 'NB' if no_barcode else _normalize_barcode(raw_barcode)

            if not barcode or not first_name or not last_name:
                messages.error(request, "Barcode, first name, and last name are all required.")
                return redirect('delivery')

            # Skip duplicate check for no-barcode entries
            if not no_barcode:
                already = DeliveryCheckIn.objects.filter(barcode=barcode, checked_out_at__isnull=True).first()
                if already:
                    messages.error(request, f"{already.first_name} {already.last_name} is already checked in with that barcode.")
                    return redirect('delivery')

            DeliveryCheckIn.objects.create(
                barcode=barcode,
                first_name=first_name,
                last_name=last_name,
                comment=comment,
            )
            UserAction.objects.create(user=request.user, action='delivery_checkin',
                target=f'{first_name} {last_name}', detail=f'Barcode: {barcode}')
            messages.success(request, f"{first_name} {last_name} checked in.")
            return redirect('delivery')

        elif action == 'checkout':
            record_id = request.POST.get('record_id', '').strip()
            barcode_raw = request.POST.get('barcode', '').strip()

            if record_id:
                record = DeliveryCheckIn.objects.filter(pk=record_id, checked_out_at__isnull=True).first()
            elif _is_no_barcode(barcode_raw):
                return JsonResponse({
                    'status': 'error',
                    'message': 'No-barcode deliveries must be checked out from the table.',
                })
            else:
                barcode = _normalize_barcode(barcode_raw)
                record = DeliveryCheckIn.objects.filter(
                    barcode=barcode, checked_out_at__isnull=True
                ).order_by('-checked_in_at').first()

            if record:
                record.checked_out_at = now()
                record.save()
                UserAction.objects.create(user=request.user, action='delivery_checkout',
                    target=f'{record.first_name} {record.last_name}', detail=f'Barcode: {record.barcode}')
                return JsonResponse({
                    'status': 'ok',
                    'name': f"{record.first_name} {record.last_name}",
                    'record_id': record.pk,
                    'barcode': record.barcode,
                    'comment': record.comment,
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
                UserAction.objects.create(user=request.user, action='delivery_undo_checkout',
                    target=f'{record.first_name} {record.last_name}', detail=f'Barcode: {record.barcode}')
                return JsonResponse({
                    'status': 'ok',
                    'record_id': record.pk,
                    'name': f"{record.first_name} {record.last_name}",
                    'barcode': record.barcode,
                    'comment': record.comment,
                    'checked_in_at': record.checked_in_at.strftime('%d %b %Y, %H:%M'),
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'Record not found or already active.'})

        elif action == 'delete_record':
            record_id = request.POST.get('record_id', '').strip()
            record = DeliveryCheckIn.objects.filter(pk=record_id).first()
            if record:
                name = f"{record.first_name} {record.last_name}"
                barcode = record.barcode
                record.delete()
                UserAction.objects.create(user=request.user, action='delivery_delete_record',
                    target=name, detail=f'Barcode: {barcode}')
                return JsonResponse({'status': 'ok', 'record_id': int(record_id), 'name': name})
            else:
                return JsonResponse({'status': 'error', 'message': 'Record not found.'})

        elif action == 'clear_history':
            del_count = DeliveryCheckIn.objects.filter(checked_out_at__isnull=False).count()
            DeliveryCheckIn.objects.filter(checked_out_at__isnull=False).delete()
            UserAction.objects.create(user=request.user, action='delivery_clear_history',
                target=f'{del_count} records cleared')
            messages.success(request, "Checkout history cleared.")
            return redirect('delivery')

        return redirect('delivery')


class OrderingSheetView(LoginRequiredMixin, View):
    """Daily ordering sheet.

    Any logged-in user can add a row. Only GINA (request.user.is_staff — the sole
    staff account) may change a row's Status or delete it after submission; that's
    enforced here as well as hidden in the template (defense in depth).
    """
    template_name = 'ordering_sheet.html'
    embed_template_name = 'ordering_sheet_embed.html'

    @staticmethod
    def _is_embed(request):
        # The dashboard opens this page in an iframe modal with ?embed=1 — render
        # without the nav chrome and keep the flag across post→redirect.
        return request.GET.get('embed') == '1'

    def _redirect(self, request):
        if self._is_embed(request):
            return redirect(f"{reverse('ordering_sheet')}?embed=1")
        return redirect('ordering_sheet')

    def get(self, request):
        # Drugs render first, then OTC products. Within each group, high
        # urgency floats to the top, then newest first.
        type_rank = Case(
            When(entry_type=OrderingSheetEntry.ENTRY_DRUG, then=Value(0)),
            default=Value(1),
        )
        urgency_rank = Case(
            When(urgency=OrderingSheetEntry.URGENCY_HIGH, then=Value(0)),
            When(urgency=OrderingSheetEntry.URGENCY_MEDIUM, then=Value(1)),
            default=Value(2),
        )
        entries = (OrderingSheetEntry.objects
                   .filter(is_deleted=False)
                   .annotate(type_rank=type_rank, urgency_rank=urgency_rank)
                   .order_by('type_rank', 'urgency_rank', '-created_at'))

        # (value, label) pairs GINA can pick from the inline status dropdown.
        status_labels = dict(OrderingSheetEntry.STATUS_CHOICES)
        gina_status_options = [(v, status_labels[v]) for v in OrderingSheetEntry.GINA_STATUS_CHOICES]

        embed = self._is_embed(request)
        template = self.embed_template_name if embed else self.template_name
        response = render(request, template, {
            'form': OrderingSheetForm(),
            'otc_form': OTCOrderingForm(),
            'entries': entries,
            'gina_status_options': gina_status_options,
            'embed': embed,
        })
        if embed:
            # Project default is X-Frame-Options: DENY. Allow this page to load
            # inside the dashboard's same-origin iframe modal. Setting the header
            # here pre-empts XFrameOptionsMiddleware (it won't overwrite it).
            response['X-Frame-Options'] = 'SAMEORIGIN'
        return response

    def post(self, request):
        action = request.POST.get('action')

        if action == 'add':
            form = OrderingSheetForm(request.POST)
            if form.is_valid():
                entry = form.save(commit=False)
                entry.entry_type = OrderingSheetEntry.ENTRY_DRUG
                entry.created_by = request.user
                entry.status = OrderingSheetEntry.STATUS_PENDING
                entry.save()
                messages.success(request, f"Added “{entry.name}” to the ordering sheet.")
            else:
                first_error = next(iter(form.errors.values()))[0]
                messages.error(request, f"Could not add entry: {first_error}")
            return self._redirect(request)

        elif action == 'add_otc':
            form = OTCOrderingForm(request.POST)
            if form.is_valid():
                entry = form.save(commit=False)
                entry.entry_type = OrderingSheetEntry.ENTRY_OTC
                entry.reasoning = ''
                entry.urgency = OrderingSheetEntry.URGENCY_NA
                entry.created_by = request.user
                entry.status = OrderingSheetEntry.STATUS_PENDING
                entry.save()
                messages.success(request, f"Added OTC product “{entry.name}” to the ordering sheet.")
            else:
                first_error = next(iter(form.errors.values()))[0]
                messages.error(request, f"Could not add OTC product: {first_error}")
            return self._redirect(request)

        elif action == 'update_status':
            # GINA only.
            if not request.user.is_staff:
                messages.error(request, "Only the GINA account can change an order's status.")
                return self._redirect(request)

            entry = OrderingSheetEntry.objects.filter(pk=request.POST.get('entry_id'), is_deleted=False).first()
            new_status = request.POST.get('status', '')
            if not entry:
                messages.error(request, "Ordering-sheet entry not found.")
            elif new_status not in OrderingSheetEntry.GINA_STATUS_CHOICES:
                messages.error(request, "Invalid status.")
            else:
                entry.status = new_status
                entry.status_updated_by = request.user
                entry.status_updated_at = now()
                entry.save(update_fields=['status', 'status_updated_by', 'status_updated_at'])
                UserAction.objects.create(user=request.user, action='ordering_status_update',
                    target=entry.name, detail=f'Status → {entry.get_status_display()}')
                messages.success(request, f"“{entry.name}” marked {entry.get_status_display()}.")
            return self._redirect(request)

        elif action == 'update_note':
            # GINA only — add/edit a free-text comment on any row (decoupled from status).
            if not request.user.is_staff:
                messages.error(request, "Only the GINA account can edit notes.")
                return self._redirect(request)
            entry = OrderingSheetEntry.objects.filter(pk=request.POST.get('entry_id'), is_deleted=False).first()
            if entry:
                entry.order_note = request.POST.get('order_note', '').strip()[:255]
                entry.save(update_fields=['order_note'])
                messages.success(request, "Note saved.")
            else:
                messages.error(request, "Ordering-sheet entry not found.")
            return self._redirect(request)

        elif action == 'delete':
            # GINA only — soft delete.
            if not request.user.is_staff:
                messages.error(request, "Only the GINA account can remove ordering-sheet entries.")
                return self._redirect(request)

            entry = OrderingSheetEntry.objects.filter(pk=request.POST.get('entry_id'), is_deleted=False).first()
            if entry:
                entry.is_deleted = True
                entry.deleted_at = now()
                entry.deleted_by = request.user
                entry.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])
                UserAction.objects.create(user=request.user, action='ordering_delete',
                    target=entry.name)
                messages.success(request, f"Removed “{entry.name}” from the ordering sheet.")
            else:
                messages.error(request, "Ordering-sheet entry not found.")
            return self._redirect(request)

        return self._redirect(request)


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

    UserAction.objects.create(user=request.user, action='update_product_settings',
        target=product.name, detail='Expiry/taxable/category updated')
    messages.success(request, f"Settings updated for {product.name}.")
    return redirect('create_order')
