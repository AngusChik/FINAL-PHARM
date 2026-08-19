from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import hmac
import time
import os
import csv
import io
import base64
import json
import re
import subprocess
import qrcode
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from datetime import date, datetime, timedelta
import queue
from urllib import request
from dateutil.relativedelta import relativedelta
from urllib.parse import urlencode, urlsplit
from django.shortcuts import render, redirect, get_object_or_404
from django.template.loader import render_to_string
from django.urls import reverse
from django.views import View
from django.contrib import messages
from django.db import transaction, connection, IntegrityError
from django.db.models import (
    Sum, Q, F, Avg, Count, Value, DecimalField, CharField, Case, When,
    DurationField, ExpressionWrapper, Exists, OuterRef, Max, Prefetch,
)
from django.db.models.functions import Cast, NullIf, TruncDay, TruncWeek, TruncMonth, TruncDate, Coalesce
from django.conf import settings
from django.core.paginator import Paginator
from django.core.cache import cache
from django.core.exceptions import ValidationError
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
from .utils import (
    TAX_RATE,
    allocate_order_line_financials,
    calculate_order_financials_from_values,
    get_product_stock_records,
    get_reorder_prediction,
    recalculate_order_totals,
    recommend_inventory_action,
)
from .forms import EditProductForm, OrderDetailForm, BarcodeForm, ItemForm, AddProductForm, OrderingSheetForm, OTCOrderingForm
from .models import (
    Item, Product, Category, Order, OrderDetail, RecentlyPurchasedProduct,
    StockChange, CheckinSession, DeliveryCheckIn, LoginAudit, UserAction,
    LabelQueueItem, CustomLabelQueueItem, LabelPrintOverride, LabelSession,
    LabelSessionItem, ProductExpiryDate, UserSession, CheckoutOrder,
    CheckoutOrderItem, PagePresence, OrderingSheetEntry, InventoryCountLine,
    DailyReportArchive, DashboardTask, SupplierOrderPlan,
    SupplierOrderPlanItem, SupplierOrderRun, SupplierOrderRunItem,
    ProductLot, ProductLotMovement, CheckinReceivingDraft, TransactionCorrection,
    TransactionCorrectionLine, TransactionCorrectionUndo, SupplierPurchaseOrder,
    SupplierPurchaseOrderLine, OrderingSheetStatusEvent,
    UserTablePreference, InventoryAuditRun, ScheduledJobRun,
    normalize_barcode_key,
)
from .inventory_services import (
    add_stock_to_lot, remove_stock_from_lots, restore_stock_to_original_lots,
    remove_stock_from_recorded_lots, ensure_lot_balance, lot_balance_issue,
)
from .page_lock import (
    PRESENCE_TTL, checkin_session_last_activity,
    checkin_session_needs_review, holder_info, is_fresh, page_label,
    path_label, presence_defaults, simplify_ua,
)
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

TABLE_PAGE_SIZES = {25, 50, 100, 200}


def preferred_table_page_size(request, default=50, table_key='main'):
    """Return this user's saved row count for the current page's primary table."""
    if not request.user.is_authenticated:
        return default
    resolver = getattr(request, 'resolver_match', None)
    page_key = resolver.url_name if resolver and resolver.url_name else 'unknown'
    saved = UserTablePreference.objects.filter(
        user=request.user,
        page_key=page_key,
        table_key=table_key,
    ).values_list('page_size', flat=True).first()
    return saved if saved in TABLE_PAGE_SIZES else default


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


def _wrap_text_to_width(text, font, size, max_width):
    """Greedy word-wrap to fit max_width. Long words are hard-broken so a
    single very long token still wraps (mirrors the HTML preview's
    overflow-wrap:anywhere). Returns a list of display lines."""
    lines, cur = [], ""
    for word in str(text or "").split():
        # Hard-break a word wider than a whole line.
        while stringWidth(word, font, size) > max_width and len(word) > 1:
            i = 1
            while i < len(word) and stringWidth(word[:i + 1], font, size) <= max_width:
                i += 1
            if cur:
                lines.append(cur); cur = ""
            lines.append(word[:i]); word = word[i:]
        trial = word if not cur else cur + " " + word
        if stringWidth(trial, font, size) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


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

    # ── Title: centered, WORD-WRAPPED to match the on-screen preview ──
    # Explicit newlines start new paragraphs; each paragraph then wraps to the
    # label width. The font shrinks until the whole wrapped block fits the
    # available height (so long titles look the same in the PDF as the preview).
    title_paras = [p.strip() for p in title.split("\n") if p.strip()] if title else []
    if title_paras:
        max_size = 13 if lines else 16
        min_size = 6
        avail_h = (LABEL_HEIGHT - pad_top - pad_bottom) * (0.6 if lines else 1.0)

        t_size = max_size
        wrapped = []
        while True:
            wrapped = []
            for para in title_paras:
                wrapped.extend(_wrap_text_to_width(para, font, t_size, inner_w))
            line_h = t_size * 1.15
            if len(wrapped) * line_h <= avail_h or t_size <= min_size:
                break
            t_size -= 0.5
        line_h = t_size * 1.15

        # Absolute last resort: if it still overflows at the smallest size,
        # keep as many lines as fit and ellipsise the last one.
        max_lines = max(1, int(avail_h // line_h))
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            ell_w = stringWidth("…", font, t_size)
            wrapped[-1] = _truncate_to_width(wrapped[-1], font, t_size, inner_w - ell_w) + "…"
        block_h = len(wrapped) * line_h

        c.setFont(font, t_size)
        if lines:
            baseline = region_top - t_size            # top-aligned block
        else:
            mid = (region_top + region_bottom) / 2    # title-only: centre it
            baseline = mid + block_h / 2 - t_size * 0.9
        for wl in wrapped:
            c.drawCentredString(center_x, baseline, wl)
            baseline -= line_h

        if lines:
            sep_y = region_top - block_h - 2
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


def _custom_label_dict(item):
    """Serializable representation shared by the page, preview, and PDF."""
    return {
        'id': item.pk,
        'title': item.title,
        'lines': item.lines if isinstance(item.lines, list) else [],
        'copies': max(1, int(item.copies or 1)),
    }


def _migrate_legacy_custom_labels(request):
    """Move custom labels from the old browser-session queue into the database.

    This compatibility bridge runs once per surviving session after deployment,
    so labels created before the migration are not silently lost.
    """
    legacy = request.session.pop('custom_labels', None)
    if legacy is None:
        return

    rows = []
    if isinstance(legacy, list):
        for label in legacy:
            if not isinstance(label, dict):
                continue
            title = str(label.get('title', '') or '').strip()[:200]
            if not title:
                continue
            lines = label.get('lines', [])
            if not isinstance(lines, list):
                lines = []
            try:
                copies = max(1, min(99, int(label.get('copies', 1) or 1)))
            except (ValueError, TypeError):
                copies = 1
            rows.append(CustomLabelQueueItem(
                user=request.user, title=title, lines=lines[:5], copies=copies,
            ))
    if rows:
        CustomLabelQueueItem.objects.bulk_create(rows)
    request.session.modified = True


def _custom_label_queue(request):
    """Return the current user's durable custom-label queue as dictionaries."""
    _migrate_legacy_custom_labels(request)
    return [
        _custom_label_dict(item)
        for item in CustomLabelQueueItem.objects.filter(user=request.user)
    ]


def _parse_custom_label_post(request):
    """Parse the custom-label form fields into (title, lines, copies). Shared by
    the add and edit actions so both behave identically."""
    raw_title = (request.POST.get("custom_title", "") or "").replace("\r\n", "\n").replace("\r", "\n")
    title_lines = [ln.strip() for ln in raw_title.split("\n")]
    title_lines = [ln for ln in title_lines if ln][:6]
    title = "\n".join(title_lines)[:200]
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
    try:
        copies = max(1, min(99, int(request.POST.get("copies", 1) or 1)))
    except (ValueError, TypeError):
        copies = 1
    return title, lines, copies


def _label_overrides(request):
    """Per-label print-time overrides (name / price / barcode) that change what
    a product label PRINTS without touching the product. Keyed by row id —
    'p<product_id>' for permanent category items, 'q<queue_pk>' for queued
    items — and stored in the session like custom labels (cleared with the
    queue). Each value is a dict with any of 'name' / 'price' / 'barcode'.
    """
    legacy = request.session.pop('label_overrides', None)
    if isinstance(legacy, dict):
        for key, value in legacy.items():
            if not isinstance(value, dict):
                continue
            target = _label_override_target(request.user, key)
            if not target:
                continue
            raw_price = value.get('price')
            try:
                price = Decimal(str(raw_price)) if raw_price not in (None, '') else None
            except (InvalidOperation, ValueError, TypeError):
                price = None
            LabelPrintOverride.objects.update_or_create(
                user=request.user, **target,
                defaults={
                    'name': str(value.get('name') or '')[:200],
                    'price': price,
                    'barcode': str(value.get('barcode') or '')[:64],
                    'barcode_overridden': 'barcode' in value,
                },
            )
        request.session.modified = True

    overrides = {}
    for override in LabelPrintOverride.objects.filter(user=request.user):
        key = (f"p{override.product_id}" if override.product_id
               else f"q{override.queue_item_id}")
        value = {
            'name': override.name,
            'price': str(override.price) if override.price is not None else '',
        }
        if override.barcode_overridden:
            value['barcode'] = override.barcode
        overrides[key] = value
    return overrides


def _label_override_target(user, key):
    """Resolve a UI override key to safe model fields owned by this user."""
    key = str(key or '').strip()
    try:
        target_id = int(key[1:])
    except (ValueError, TypeError):
        return None
    if key.startswith('p') and Product.objects.filter(pk=target_id).exists():
        return {'product_id': target_id, 'queue_item': None}
    if key.startswith('q') and LabelQueueItem.objects.filter(
            pk=target_id, user=user).exists():
        return {'product': None, 'queue_item_id': target_id}
    return None


def _effective_label(product, key, overrides, qty=1):
    """Label fields for a product row with any per-label override applied.
    An override value wins only when non-empty (blank falls back to the
    product), except barcode which may be deliberately blanked."""
    ov = overrides.get(key, {}) or {}
    barcode = ov['barcode'] if 'barcode' in ov else (product.barcode or '')
    return {
        'name': (ov.get('name') or product.name),
        'brand': product.brand or '',
        'item_number': product.item_number or '',
        'barcode': barcode,
        'price': (ov.get('price') or str(product.price)),
        'qty': qty,
        'key': key,
        'overridden': bool(ov),
    }


def _build_preview_labels(category_items, queue_items, custom_labels, overrides=None):
    """Flat list of labels for the live sheet preview (products + custom),
    with per-label overrides applied to the product labels."""
    overrides = overrides or {}
    labels = []
    for p in category_items:
        eff = _effective_label(p, f"p{p.product_id}", overrides, qty=1)
        labels.append({'name': eff['name'], 'barcode': eff['barcode'], 'price': eff['price'],
                       'brand': eff['brand'], 'item_number': eff['item_number'], 'qty': 1})
    for qi in queue_items:
        eff = _effective_label(qi.product, f"q{qi.pk}", overrides, qty=qi.qty)
        labels.append({'name': eff['name'], 'barcode': eff['barcode'], 'price': eff['price'],
                       'brand': eff['brand'], 'item_number': eff['item_number'], 'qty': qi.qty})
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


@login_required
def label_queue_add(request):
    """AJAX: add one product to the current user's print-label queue without
    leaving the calling page (used by the check-in Quick Actions)."""
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': 'POST required'}, status=405)
    product = Product.objects.filter(product_id=request.POST.get('product_id')).first()
    if not product:
        return JsonResponse({'ok': False, 'error': 'Product not found'}, status=404)
    LabelQueueItem.objects.create(product=product, user=request.user)
    count = LabelQueueItem.objects.filter(user=request.user).count()
    return JsonResponse({'ok': True, 'name': product.name, 'queue_count': count})


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

        queue_items = list(self._get_queue(request))
        category_items = list(
            Product.objects.filter(category__name__icontains="Print Label", status=True)
        )

        all_products = list(Product.objects.filter(status=True).values(
            'product_id', 'name', 'barcode', 'item_number', 'price', 'quantity_in_stock'
        ))

        custom_labels = _custom_label_queue(request)
        overrides = _label_overrides(request)

        # Effective (override-aware) rows for the queue table + edit modals.
        perm_rows = []
        for p in category_items:
            eff = _effective_label(p, f"p{p.product_id}", overrides)
            eff['product_id'] = p.product_id
            perm_rows.append(eff)
        queue_rows = []
        for qi in queue_items:
            eff = _effective_label(qi.product, f"q{qi.pk}", overrides, qty=qi.qty)
            eff['pk'] = qi.pk
            eff['product_id'] = qi.product.product_id
            queue_rows.append(eff)

        return render(request, self.template_name, {
            "queue_items": queue_items,
            "category_items": category_items,
            "perm_rows": perm_rows,
            "queue_rows": queue_rows,
            "category_items_count": len(category_items),
            "queue_items_count": len(queue_items),
            "categories": Category.objects.all().order_by('name'),
            "all_products": all_products,
            "custom_labels": custom_labels,
            "custom_labels_raw": custom_labels,
            "custom_labels_count": sum(max(1, int(cl.get('copies', 1))) for cl in custom_labels),
            "preview_labels": _build_preview_labels(category_items, queue_items, custom_labels, overrides),
        })

    def post(self, request):
        _migrate_legacy_custom_labels(request)
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
            title, lines, copies = _parse_custom_label_post(request)
            if not title:
                messages.error(request, "Enter the item name for the top of the label.")
            else:
                CustomLabelQueueItem.objects.create(
                    user=request.user, title=title, lines=lines, copies=copies,
                )
                plural = "s" if len(lines) != 1 else ""
                messages.success(
                    request,
                    f"Added custom label '{title}' ({len(lines)} line{plural} × {copies})."
                )

        elif "edit_custom_label" in request.POST:
            # Update an existing custom label in place (same fields as add).
            try:
                custom_id = int(request.POST.get("edit_custom_label"))
            except (ValueError, TypeError):
                custom_id = -1
            title, lines, copies = _parse_custom_label_post(request)
            custom = CustomLabelQueueItem.objects.filter(
                pk=custom_id, user=request.user,
            ).first()
            if custom is None:
                messages.warning(request, "That custom label was removed — nothing to update.")
            elif not title:
                messages.error(request, "Enter the item name for the top of the label.")
            else:
                custom.title = title
                custom.lines = lines
                custom.copies = copies
                custom.save(update_fields=['title', 'lines', 'copies'])
                messages.success(request, f"Updated custom label '{title}'.")

        elif "save_label_override" in request.POST:
            # Per-label override of what a PRODUCT label prints (name/price/
            # barcode) without changing the product. Key = p<product_id> or
            # q<queue_pk>.
            key = (request.POST.get("override_key") or "").strip()
            target = _label_override_target(request.user, key)
            if not target:
                messages.warning(request, "Could not identify which label to edit.")
            elif request.POST.get("reset_override"):
                LabelPrintOverride.objects.filter(user=request.user, **target).delete()
                messages.success(request, "Label reset to the product's details.")
            else:
                # Price must stay numeric — the PDF/preview format it as a float.
                raw_price = (request.POST.get("override_price", "") or "").strip()
                try:
                    price_str = f"{float(raw_price):.2f}" if raw_price else ""
                except (ValueError, TypeError):
                    price_str = ""
                LabelPrintOverride.objects.update_or_create(
                    user=request.user, **target,
                    defaults={
                        "name": (request.POST.get("override_name", "") or "").strip()[:200],
                        "price": Decimal(price_str) if price_str else None,
                        "barcode": (request.POST.get("override_barcode", "") or "").strip()[:64],
                        "barcode_overridden": True,
                    },
                )
                messages.success(request, "Label updated for this print.")

        elif "remove_custom_label" in request.POST:
            try:
                custom_id = int(request.POST.get("remove_custom_label"))
            except (ValueError, TypeError):
                custom_id = -1
            deleted, _ = CustomLabelQueueItem.objects.filter(
                pk=custom_id, user=request.user,
            ).delete()
            if deleted:
                messages.success(request, "Custom label removed.")
            else:
                messages.warning(request, "That custom label was already removed.")

        elif "clear_queue" in request.POST:
            self._get_queue(request).delete()
            CustomLabelQueueItem.objects.filter(user=request.user).delete()
            LabelPrintOverride.objects.filter(user=request.user).delete()
            request.session.pop("custom_labels", None)
            request.session.pop("label_overrides", None)
            request.session.modified = True
            UserAction.objects.create(user=request.user, action='clear_label_queue',
                target='Label queue cleared')
            messages.info(request, "Label queue cleared.")

        elif "remove_item" in request.POST:
            item_id = request.POST.get("remove_item")
            deleted, _ = self._get_queue(request).filter(pk=item_id).delete()
            if deleted:
                # Queue-target overrides cascade with their queue row.
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
        overrides = _label_overrides(request)
        custom_labels = _custom_label_queue(request)
        category_items = list(Product.objects.filter(
            category__name__icontains="Print Label", status=True))
        queue_items = list(
            LabelQueueItem.objects.filter(user=request.user).select_related('product'))

        # Effective (override-aware) label fields, reused for the PDF + snapshot.
        perm_eff = [_effective_label(p, f"p{p.product_id}", overrides) for p in category_items]
        queue_eff = [_effective_label(qi.product, f"q{qi.pk}", overrides, qty=qi.qty)
                     for qi in queue_items]

        merged = []
        for eff in perm_eff:
            merged.append({
                "name": eff["name"], "brand": eff["brand"],
                "item_number": eff["item_number"],
                "barcode": eff["barcode"], "price": eff["price"],
            })
        for eff in queue_eff:
            merged.append({
                "name": eff["name"], "brand": eff["brand"],
                "item_number": eff["item_number"],
                "barcode": eff["barcode"], "price": eff["price"],
                "qty": eff["qty"],
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
        for cl in custom_labels:
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
        def _price_dec(s, fallback):
            try:
                return Decimal(str(s))
            except (InvalidOperation, ValueError, TypeError):
                return fallback

        snapshot_items = []
        for p, eff in zip(category_items, perm_eff):
            snapshot_items.append(LabelSessionItem(
                session=session_obj, product=p,
                product_name=eff['name'], product_barcode=eff['barcode'],
                product_price=_price_dec(eff['price'], p.price), product_brand=eff['brand'],
                product_item_number=eff['item_number'], qty=1,
            ))
        for qi, eff in zip(queue_items, queue_eff):
            snapshot_items.append(LabelSessionItem(
                session=session_obj, product=qi.product,
                product_name=eff['name'], product_barcode=eff['barcode'],
                product_price=_price_dec(eff['price'], qi.product.price), product_brand=eff['brand'],
                product_item_number=eff['item_number'], qty=eff['qty'],
            ))
        for cl in custom_labels:
            snapshot_items.append(LabelSessionItem(
                session=session_obj, product=None,
                product_name=cl.get('title', ''), product_barcode='',
                product_price=Decimal('0.00'), product_brand='',
                product_item_number='', qty=max(1, int(cl.get('copies', 1))),
                is_custom=True, custom_lines=cl.get('lines', []),
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
    """Return the same durable key used by Product.normalized_barcode."""
    return normalize_barcode_key(value) or ""

def find_product_by_barcode(barcode: str, for_update: bool = False):
    """
    Look up Product by barcode using the stored normalized unique key.
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

    return qs.filter(normalized_barcode=normalized).first()


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
        normalized_field = (
            f'{field[:-len("barcode")]}normalized_barcode'
            if field.endswith('barcode') else field
        )
        q |= Q(**{normalized_field: normalized})
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
                timestamp__date__gte=thirty_days_ago,
            ).values_list('product_id').annotate(missed=Sum('quantity'))
        )

        total_missed = 0
        total_revenue_lost = Decimal('0.00')
        for p in products:
            p.missed_30d = recent_unfulfilled.get(p.product_id, 0)
            p.revenue_lost_30d = p.missed_30d * p.price
            total_missed += p.missed_30d
            total_revenue_lost += p.revenue_lost_30d

        paginator = Paginator(products, preferred_table_page_size(request, 50))
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

        paginator = Paginator(products, preferred_table_page_size(request, 50))
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
                timestamp__date__gte=thirty_days_ago,
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

        paginator = Paginator(products, preferred_table_page_size(request, 50))
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


def get_active_purchase_order(request, create_if_missing=False):
    """Return this user's current purchase draft.

    The session stores only the draft's identifier. The cart itself is owned by
    Order.draft_cart, which is the sole authoritative copy.
    """
    order_id = request.session.get("order_id")
    order = None
    if order_id:
        order = Order.objects.filter(
            order_id=order_id,
            user=request.user,
            submitted=False,
            is_deleted=False,
        ).first()

    if order is None:
        order = (
            Order.objects.filter(
                user=request.user, submitted=False, is_deleted=False,
            )
            .exclude(draft_cart={})
            .order_by("-order_date")
            .first()
        )

    # One-time compatibility import for a browser that still has a cart from a
    # pre-database release. Never overwrite an existing database cart.
    legacy_cart = request.session.pop("cart", None)
    if legacy_cart is not None:
        request.session.modified = True
    if not isinstance(legacy_cart, dict):
        legacy_cart = {}

    if order is None and (create_if_missing or legacy_cart):
        order = Order.objects.create(
            total_price=Decimal("0.00"),
            user=request.user,
            draft_cart=dict(legacy_cart),
        )
    elif order is not None and legacy_cart and not order.draft_cart:
        order.draft_cart = dict(legacy_cart)
        order.save(update_fields=["draft_cart"])

    if order is None:
        request.session.pop("order_id", None)
        return None

    if request.session.get("order_id") != order.order_id:
        request.session["order_id"] = order.order_id
        request.session.modified = True

    if order.draft_cart and order.draft_expires_at is None:
        timestamp = now()
        order.draft_expires_at = timestamp + timedelta(minutes=10)
        order.last_timer_reset_at = timestamp
        order.save(update_fields=["draft_expires_at", "last_timer_reset_at"])
    return order


def save_cart(request, cart, order=None):
    """Persist a purchase cart only in the database and return its draft."""
    order = order or get_active_purchase_order(request, create_if_missing=bool(cart))
    request.session.pop("cart", None)
    request.session.modified = True
    if order is None:
        return None

    order.draft_cart = dict(cart)
    update_fields = ["draft_cart"]
    if order.draft_cart and order.draft_expires_at is None:
        timestamp = now()
        order.draft_expires_at = timestamp + timedelta(minutes=10)
        order.last_timer_reset_at = timestamp
        update_fields.extend(["draft_expires_at", "last_timer_reset_at"])
    order.save(update_fields=update_fields)
    request.session["order_id"] = order.order_id
    return order

# Dashboard expand pop-outs — full detailed lists, fetched on click so they
# don't slow the dashboard's initial load.
@login_required
def dashboard_expand(request):
    from app import reporting
    section = request.GET.get('section')
    if section == 'reorder':
        return JsonResponse({'ok': True, 'items': reporting.reorder_suggestions(limit=300)})
    if section == 'deadstock':
        d = reporting.dead_stock(limit=300)
        return JsonResponse({'ok': True, 'items': d['items'], 'count': d['count']})
    return JsonResponse({'ok': False, 'error': 'Unknown section.'}, status=400)


# Home view
@login_required
def home(request):
    from app import reporting
    from app.scheduled_jobs import store_hours_payload

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
        'store_hours_json': store_hours_payload(),
    })


class DashboardTasksAPIView(LoginRequiredMixin, View):
    """Shared dashboard task list backed by soft-archived database rows."""

    @staticmethod
    def _serialize(task):
        return {
            'id': task.pk,
            'text': task.text,
            'done': task.completed,
            'user': task.created_by_name or (
                task.created_by.get_short_name() or task.created_by.username
                if task.created_by else ''
            ),
            'created_at': task.created_at.isoformat(),
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
        }

    def get(self, request):
        tasks = DashboardTask.objects.filter(archived_at__isnull=True).select_related('created_by')
        return JsonResponse({'ok': True, 'items': [self._serialize(task) for task in tasks]})

    def post(self, request):
        try:
            data = json.loads(request.body or '{}')
        except ValueError:
            return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)
        action = data.get('action')
        display_name = request.user.get_short_name() or request.user.username

        if action == 'add':
            text = str(data.get('text') or '').strip()[:200]
            if not text:
                return JsonResponse({'ok': False, 'error': 'Type a note first.'}, status=400)
            task = DashboardTask.objects.create(
                text=text, created_by=request.user, created_by_name=display_name,
            )
            return JsonResponse({'ok': True, 'item': self._serialize(task)})

        if action == 'toggle':
            task = DashboardTask.objects.filter(
                pk=data.get('id'), archived_at__isnull=True,
            ).first()
            if not task:
                return JsonResponse({'ok': False, 'error': 'Task not found.'}, status=404)
            task.completed = not task.completed
            task.completed_by = request.user if task.completed else None
            task.completed_at = now() if task.completed else None
            task.save(update_fields=['completed', 'completed_by', 'completed_at', 'updated_at'])
            return JsonResponse({'ok': True, 'item': self._serialize(task)})

        if action == 'delete':
            task = DashboardTask.objects.filter(
                pk=data.get('id'), archived_at__isnull=True,
            ).first()
            if not task:
                return JsonResponse({'ok': False, 'error': 'Task not found.'}, status=404)
            task.archived_at = now()
            task.archived_by = request.user
            task.save(update_fields=['archived_at', 'archived_by', 'updated_at'])
            return JsonResponse({'ok': True})

        if action == 'clear_completed':
            count = DashboardTask.objects.filter(
                completed=True, archived_at__isnull=True,
            ).update(archived_at=now(), archived_by=request.user)
            return JsonResponse({'ok': True, 'archived': count})

        if action == 'import_legacy':
            items = data.get('items') if isinstance(data.get('items'), list) else []
            rows = []
            timestamp = now()
            existing = set(DashboardTask.objects.values_list(
                'text', 'completed', 'created_by_name',
            ))
            for item in items[:200]:
                if not isinstance(item, dict):
                    continue
                text = str(item.get('text') or '').strip()[:200]
                if not text:
                    continue
                completed = bool(item.get('done'))
                creator_name = str(item.get('user') or display_name)[:150]
                key = (text, completed, creator_name)
                if key in existing:
                    continue
                existing.add(key)
                rows.append(DashboardTask(
                    text=text,
                    created_by=request.user,
                    created_by_name=creator_name,
                    completed=completed,
                    completed_by=request.user if completed else None,
                    completed_at=timestamp if completed else None,
                ))
            if rows:
                DashboardTask.objects.bulk_create(rows)
            return JsonResponse({'ok': True, 'imported': len(rows)})

        return JsonResponse({'ok': False, 'error': 'Unknown task action.'}, status=400)


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

    def _safe_return(self, request, raw):
        """Keep the passkey cancel action on-site and out of a prompt loop."""
        fallback = reverse('dashboard')
        if not raw or not url_has_allowed_host_and_scheme(
            raw, allowed_hosts={request.get_host()}, require_https=request.is_secure()
        ):
            return fallback
        if urlsplit(raw).path.rstrip('/') == reverse('passkey_unlock').rstrip('/'):
            return fallback
        return raw

    def _template_context(self, request, nxt, return_to):
        return {
            'next': nxt,
            'return_to': self._safe_return(request, return_to),
        }

    def get(self, request):
        nxt = self._safe_next(request, request.GET.get('next'))
        if has_admin_access(request):
            return redirect(nxt)
        return_to = request.GET.get('return_to') or request.META.get('HTTP_REFERER')
        return render(
            request,
            self.template_name,
            self._template_context(request, nxt, return_to),
        )

    # Failed-attempt throttle: django-axes only rate-limits the login page,
    # so the passkey form needs its own guard against brute-forcing.
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_SECONDS = 300

    def post(self, request):
        nxt = self._safe_next(request, request.POST.get('next'))
        return_to = request.POST.get('return_to')
        now = time.time()
        locked_until = request.session.get('passkey_locked_until', 0)
        if now < locked_until:
            wait_min = max(1, int(locked_until - now + 59) // 60)
            messages.error(
                request,
                f"Too many incorrect attempts. Try again in {wait_min} minute(s)."
            )
            return render(
                request,
                self.template_name,
                self._template_context(request, nxt, return_to),
            )
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
        return render(
            request,
            self.template_name,
            self._template_context(request, nxt, return_to),
        )


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
        from app import reporting

        today = date.today()
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')
        status_filter = request.GET.get('status', '')
        source_filter = request.GET.get('source', '')  # '', 'all', 'pos', 'giveaway'

        # Preserve every submitted order while calculating the amount that is
        # still realized after active returns/voids. Undo records make their
        # correction inactive; the immutable order snapshot is never rewritten.
        orders = reporting.annotate_orders_with_realized_sales(
            Order.objects.all(),
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

        # KPI/chart scope follows the visible transaction filters. Giveaways are
        # explicitly no-sale records, so a giveaway-only view has no POS orders
        # or revenue. Deleted sales stay excluded unless that view is selected.
        show_deleted = status_filter == 'deleted'
        include_pos_metrics = (
            source_filter in ('', 'all', 'pos')
            and status_filter != 'pending'
        )
        if include_pos_metrics:
            submitted_orders = orders.filter(
                submitted=True,
                is_deleted=show_deleted,
            )
            realized_orders = submitted_orders.filter(realized_units__gt=0)
            agg = realized_orders.aggregate(
                total_revenue=Sum('realized_revenue'),
                avg_order=Avg('realized_revenue'),
            )
            orders_today_count = realized_orders.filter(
                order_date__date=today,
            ).count()

            daily_sales = list(
                realized_orders
                .annotate(sale_date=TruncDate('order_date'))
                .values('sale_date')
                .annotate(
                    daily_revenue=Sum(
                        'realized_revenue',
                        output_field=reporting.REALIZED_MONEY_FIELD,
                    ),
                    order_count=Count('pk', distinct=True),
                    item_count=Sum('realized_units'),
                )
                .order_by('sale_date')
            )
        else:
            realized_orders = Order.objects.none()
            agg = {'total_revenue': None, 'avg_order': None}
            orders_today_count = 0
            daily_sales = []
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
        if source_filter in ('', 'all', 'pos'):
            # The same deleted/live scope is used by the rows and KPI/chart above.
            for o in orders.filter(is_deleted=show_deleted):
                rows.append({
                    'source': 'pos',
                    'id': o.order_id,
                    'date': o.order_date,
                    'total': o.realized_revenue or Decimal('0.00'),
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
        paginator = Paginator(rows, preferred_table_page_size(request, 50))
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        transaction_export_query = urlencode({
            key: value for key, value in {
                'date_from': date_from,
                'date_to': date_to,
                'status': status_filter,
                'source': source_filter,
            }.items() if value
        })

        return render(request, self.template_name, {
            'page_obj': page_obj,
            'current_order_id': current_order_id,
            'total_orders': realized_orders.count(),
            'total_revenue': agg['total_revenue'] or Decimal('0.00'),
            'avg_order': agg['avg_order'] or Decimal('0.00'),
            'orders_today': orders_today_count,
            'daily_chart_data': daily_chart_data,
            'date_from': date_from,
            'date_to': date_to,
            'status_filter': status_filter,
            'source_filter': source_filter,
            'transaction_export_query': transaction_export_query,
            'metric_scope_label': (
                'PU no-sale records · POS excluded'
                if source_filter == 'giveaway'
                else 'pending sales · not submitted'
                if status_filter == 'pending'
                else 'deleted POS sales'
                if show_deleted
                else 'POS sales in view'
            ),
            'today': today,
        })


def build_order_transaction_context(order):
    order_details = list(
        order.details.select_related('product', 'product__category').all()
    )
    order_details.sort(key=lambda detail: getattr(detail, 'pk', 0) or 0)
    correction_manager = getattr(order, 'corrections', None)
    corrections = (
        correction_manager.prefetch_related('lines').select_related(
            'created_by', 'undo__created_by',
        )
        if correction_manager is not None
        else ()
    )
    correction_total = (
        correction_manager.filter(undo__isnull=True).aggregate(
            total=Sum('adjustment_amount'),
        )['total']
        or Decimal('0.00')
        if correction_manager is not None
        else Decimal('0.00')
    )

    total_items = 0
    total_units = 0
    total_price_before_tax = Decimal("0.00")
    total_tax = Decimal("0.00")
    total_cost = Decimal("0.00")
    taxable_subtotal = Decimal("0.00")
    nontaxable_subtotal = Decimal("0.00")
    missing_cost_count = 0
    tax_rate = order.tax_rate if order.financial_snapshot_source else TAX_RATE

    # Local calendar date the order was placed — used to flag items that were
    # already past their expiry date when the sale happened.
    order_date_local = localtime(order.order_date).date() if order.order_date else None

    order_details_with_total = []
    expired_sold_count = 0
    for detail in order_details:
        line_total = detail.price * detail.quantity
        product = detail.product

        is_taxable = detail.taxable_at_sale is True
        if detail.cost_per_unit_at_sale is not None:
            cost = detail.cost_per_unit_at_sale * detail.quantity
            profit = line_total - cost
        else:
            cost = Decimal("0.00")
            profit = None
            missing_cost_count += 1

        # "Expired when sold": the earliest expiry had already passed on the order
        # date. Prefer the expiry snapshot captured at submit time (exact); fall back
        # to the product's current earliest expiry for older, un-snapshotted lines.
        expiry_date = detail.expiry_at_sale
        if expiry_date is None and product is not None:
            expiry_date = product.expiry_date
        expired_at_sale = bool(expiry_date and order_date_local and expiry_date < order_date_local)
        if expired_at_sale:
            expired_sold_count += 1

        correction_lines = getattr(detail, 'correction_lines', None)
        corrected_qty = (
            correction_lines.filter(
                correction__undo__isnull=True,
            ).aggregate(total=Sum('quantity'))['total'] or 0
            if correction_lines is not None
            else 0
        )
        order_details_with_total.append({
            'detail': detail,
            'total_price': line_total,
            'is_taxable': is_taxable,
            'item_tax': Decimal('0.00'),
            'line_with_tax': line_total,
            'cost': cost,
            'profit': profit,
            'product_deleted': product is None or bool(
                getattr(product, 'archived_at', None)
            ),
            'expired_at_sale': expired_at_sale,
            'expiry_date': expiry_date,
            'corrected_qty': corrected_qty,
            'remaining_correctable_qty': max(0, detail.quantity - corrected_qty),
        })

        total_items += 1
        total_units += detail.quantity
        total_price_before_tax += line_total
        total_cost += cost
        if is_taxable:
            taxable_subtotal += line_total
        else:
            nontaxable_subtotal += line_total

    seniors_discount = order.seniors_discount
    if order.financial_snapshot_source:
        total_price_before_tax = order.subtotal
        seniors_discount_amount = order.discount_amount
        total_tax = order.tax
        total_price_after_tax = order.total_price
    else:
        values = calculate_order_financials_from_values(
            (
                (detail.price, detail.quantity, detail.taxable_at_sale)
                for detail in order_details
            ),
            seniors_discount=seniors_discount,
            tax_rate=tax_rate,
        )
        total_price_before_tax = values['subtotal']
        seniors_discount_amount = values['discount_amount']
        total_tax = values['tax']
        total_price_after_tax = values['total']

    line_allocations = allocate_order_line_financials(
        [row['total_price'] for row in order_details_with_total],
        [row['is_taxable'] for row in order_details_with_total],
        seniors_discount_amount,
        total_tax,
    )
    for row, allocation in zip(order_details_with_total, line_allocations):
        row['discount_share'] = allocation['discount']
        row['net_line_total'] = allocation['net']
        row['item_tax'] = allocation['tax']
        row['line_with_tax'] = allocation['total']
        if row['profit'] is not None:
            row['profit'] = allocation['net'] - row['cost']

    has_active_corrections = any(
        row['corrected_qty'] for row in order_details_with_total
    )
    if has_active_corrections:
        realized_total = calculate_order_financials_from_values(
            (
                (
                    row['detail'].price,
                    max(0, row['detail'].quantity - row['corrected_qty']),
                    row['detail'].taxable_at_sale,
                )
                for row in order_details_with_total
            ),
            seniors_discount=seniors_discount,
            tax_rate=tax_rate,
        )['total']
    else:
        realized_total = total_price_after_tax

    has_complete_cost_data = bool(order_details) and missing_cost_count == 0
    total_profit = (
        total_price_before_tax - seniors_discount_amount - total_cost
        if has_complete_cost_data else None
    )
    net_revenue = total_price_before_tax - seniors_discount_amount
    margin_pct = (
        (total_profit / net_revenue) * 100
        if total_profit is not None and net_revenue > 0
        else None
    )

    return {
        'order': order,
        'order_details_with_total': order_details_with_total,
        'total_price_before_tax': total_price_before_tax,
        'total_price_after_tax': total_price_after_tax,
        'correction_total': correction_total,
        'net_total_after_corrections': realized_total,
        'total_tax': total_tax,
        'seniors_discount': seniors_discount,
        'seniors_discount_amount': seniors_discount_amount,
        'total_items': total_items,
        'total_units': total_units,
        'taxable_subtotal': taxable_subtotal,
        'nontaxable_subtotal': nontaxable_subtotal,
        'total_cost': total_cost,
        'total_profit': total_profit,
        'net_revenue': net_revenue,
        'has_complete_cost_data': has_complete_cost_data,
        'margin_pct': margin_pct,
        'financial_snapshot_source': order.financial_snapshot_source,
        'expired_sold_count': expired_sold_count,
        'any_expired_sold': expired_sold_count > 0,
        'corrections': corrections,
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


class TransactionCorrectionView(AdminRequiredMixin, View):
    """Record a return, void, or correction without rewriting the original sale."""

    template_name = 'transaction_correction.html'

    def _source(self, order_id=None, checkout_id=None, for_update=False):
        if order_id is not None:
            qs = Order.objects
            if for_update:
                qs = qs.select_for_update()
            transaction_obj = get_object_or_404(qs, order_id=order_id, submitted=True)
            lines = list(transaction_obj.details.select_related('product').all())
            return 'order', transaction_obj, lines
        qs = CheckoutOrder.objects
        if for_update:
            qs = qs.select_for_update()
        transaction_obj = get_object_or_404(
            qs, pk=checkout_id, status=CheckoutOrder.STATUS_SUBMITTED,
        )
        lines = list(transaction_obj.items.select_related('product').all())
        return 'checkout', transaction_obj, lines

    @staticmethod
    def _corrected_quantity(kind, line):
        return line.correction_lines.filter(
            correction__undo__isnull=True,
        ).aggregate(total=Sum('quantity'))['total'] or 0

    @staticmethod
    def _fulfilled_quantity(kind, line):
        """Units physically supplied by the original transaction.

        New and backfilled transactions have direct stock-ledger links. If an
        older row has no linked ledger at all, fall back to its stored quantity
        so it remains correctable rather than becoming stranded.
        """
        fulfilled_type = 'checkout' if kind == 'order' else 'giveaway'
        unfulfilled_type = (
            'checkout_unfulfilled' if kind == 'order'
            else 'giveaway_unfulfilled'
        )
        history = line.stock_changes.filter(
            change_type__in=[fulfilled_type, unfulfilled_type],
        )
        if not history.exists():
            return line.quantity
        return history.filter(change_type=fulfilled_type).aggregate(
            total=Sum('quantity'),
        )['total'] or 0

    @staticmethod
    def _transaction_adjustment(kind, source, lines, quantities, requested_lines):
        """Return the settled before/after delta for one correction.

        Seniors discounts and tax are order-level amounts. Calculating each
        corrected line independently can create or lose a cent, so corrections
        are priced as the difference between the complete remaining baskets.
        """
        after_quantities = dict(quantities)
        for line, quantity, _disposition in requested_lines:
            after_quantities[line.pk] = max(
                0, after_quantities.get(line.pk, 0) - quantity,
            )

        if kind == 'order':
            seniors_discount = source.seniors_discount
            tax_rate = (
                source.tax_rate
                if source.financial_snapshot_source
                else TAX_RATE
            )
            taxable = lambda line: line.taxable_at_sale
        else:
            seniors_discount = False
            tax_rate = TAX_RATE
            taxable = lambda line: bool(line.taxable)

        is_original_state = all(
            quantities.get(line.pk, 0) == line.quantity
            for line in lines
        )
        if is_original_state:
            before_total = Decimal(source.total_price)
        else:
            before_total = calculate_order_financials_from_values(
                (
                    (line.price, quantities.get(line.pk, 0), taxable(line))
                    for line in lines
                ),
                seniors_discount=seniors_discount,
                tax_rate=tax_rate,
            )['total']
        after_total = calculate_order_financials_from_values(
            (
                (line.price, after_quantities.get(line.pk, 0), taxable(line))
                for line in lines
            ),
            seniors_discount=seniors_discount,
            tax_rate=tax_rate,
        )['total']
        return max(
            Decimal('0.00'), before_total - after_total,
        ).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def get(self, request, order_id=None, checkout_id=None):
        kind, source, lines = self._source(order_id, checkout_id)
        line_rows = []
        for line in lines:
            corrected = self._corrected_quantity(kind, line)
            fulfilled = self._fulfilled_quantity(kind, line)
            line_rows.append({
                'line': line,
                'name': line.product_name,
                'barcode': line.product_barcode,
                'original_qty': line.quantity,
                'fulfilled_qty': fulfilled,
                'corrected_qty': corrected,
                'remaining_qty': max(0, fulfilled - corrected),
            })
        return render(request, self.template_name, {
            'kind': kind,
            'source': source,
            'line_rows': line_rows,
            'dispositions': TransactionCorrectionLine.DISPOSITION_CHOICES,
        })

    def post(self, request, order_id=None, checkout_id=None):
        correction_type = request.POST.get('correction_type', '')
        reason = request.POST.get('reason', '').strip()
        note = request.POST.get('note', '').strip()
        valid_types = dict(TransactionCorrection.TYPE_CHOICES)
        if correction_type not in valid_types or not reason:
            messages.error(request, 'Choose a correction type and enter a reason.')
            return self.get(request, order_id, checkout_id)

        redirect_name = 'order_detail' if order_id is not None else 'giveaway_detail'
        redirect_id = order_id if order_id is not None else checkout_id
        with transaction.atomic():
            kind, source, lines = self._source(
                order_id, checkout_id, for_update=True,
            )
            requested_lines = []
            remaining_quantities = {}
            for line in lines:
                corrected = self._corrected_quantity(kind, line)
                fulfilled = self._fulfilled_quantity(kind, line)
                remaining = max(0, fulfilled - corrected)
                remaining_quantities[line.pk] = remaining
                if correction_type == TransactionCorrection.TYPE_VOID:
                    quantity = remaining
                else:
                    try:
                        quantity = int(request.POST.get(f'qty_{line.pk}', 0) or 0)
                    except (TypeError, ValueError):
                        quantity = -1
                if quantity < 0 or quantity > remaining:
                    messages.error(
                        request,
                        f'{line.product_name}: choose between 0 and {remaining} units.',
                    )
                    return self.get(request, order_id, checkout_id)
                if quantity == 0:
                    continue
                disposition = request.POST.get(
                    f'disposition_{line.pk}',
                    TransactionCorrectionLine.DISPOSITION_RESTOCK,
                )
                if disposition not in dict(TransactionCorrectionLine.DISPOSITION_CHOICES):
                    disposition = TransactionCorrectionLine.DISPOSITION_NO_RESTOCK
                requested_lines.append((line, quantity, disposition))

            if not requested_lines:
                messages.error(request, 'Select at least one unit to correct.')
                return self.get(request, order_id, checkout_id)

            adjustment = self._transaction_adjustment(
                kind, source, lines, remaining_quantities, requested_lines,
            )

            correction = TransactionCorrection.objects.create(
                correction_type=correction_type,
                order=source if kind == 'order' else None,
                checkout=source if kind == 'checkout' else None,
                reason=reason[:255], note=note,
                adjustment_amount=adjustment,
                created_by=request.user,
            )
            for source_line, quantity, disposition in requested_lines:
                correction_line = TransactionCorrectionLine.objects.create(
                    correction=correction,
                    order_detail=source_line if kind == 'order' else None,
                    checkout_item=source_line if kind == 'checkout' else None,
                    product=source_line.product,
                    product_name=source_line.product_name,
                    product_barcode=source_line.product_barcode,
                    quantity=quantity,
                    unit_price=source_line.price,
                    disposition=disposition,
                )
                if not source_line.product_id:
                    continue
                product = Product.objects.select_for_update().get(pk=source_line.product_id)
                should_restock = disposition == TransactionCorrectionLine.DISPOSITION_RESTOCK
                if should_restock:
                    product.quantity_in_stock += quantity
                    product.save(update_fields=['quantity_in_stock'])
                if correction_type == TransactionCorrection.TYPE_VOID:
                    change_type = 'void'
                elif should_restock:
                    change_type = 'return'
                else:
                    change_type = 'return_no_restock'
                stock_change = record_stock_change(
                    product=product, qty=quantity, change_type=change_type,
                    note=f'{correction.get_correction_type_display()}: {reason}',
                    user=request.user,
                    order_detail=source_line if kind == 'order' else None,
                    checkout_item=source_line if kind == 'checkout' else None,
                    correction_line=correction_line,
                )
                if should_restock:
                    source_changes = StockChange.objects.filter(
                        **(
                            {'order_detail': source_line}
                            if kind == 'order' else {'checkout_item': source_line}
                        ),
                        change_type__in=['checkout', 'giveaway'],
                    )
                    restore_stock_to_original_lots(
                        product, quantity, source_changes, stock_change,
                    )

            UserAction.objects.create(
                user=request.user, action='transaction_correction',
                target=correction.transaction_label,
                detail=f'{correction.get_correction_type_display()}: {reason}',
            )

        messages.success(
            request,
            f'{valid_types[correction_type]} recorded. The original transaction was preserved.',
        )
        return redirect(redirect_name, redirect_id)


class TransactionCorrectionUndoView(AdminRequiredMixin, View):
    """Reverse an accidental void while preserving both audit records."""

    @staticmethod
    def _redirect_target(correction):
        if correction.order_id:
            return 'order_detail', correction.order_id
        return 'giveaway_detail', correction.checkout_id

    def post(self, request, correction_id):
        correction = get_object_or_404(
            TransactionCorrection.objects.select_related('order', 'checkout'),
            pk=correction_id,
        )
        redirect_name, redirect_id = self._redirect_target(correction)

        try:
            with transaction.atomic():
                correction = get_object_or_404(
                    TransactionCorrection.objects.select_for_update(),
                    pk=correction_id,
                )
                redirect_name, redirect_id = self._redirect_target(correction)
                if correction.correction_type != TransactionCorrection.TYPE_VOID:
                    messages.error(request, 'Only a transaction void can be undone.')
                    return redirect(redirect_name, redirect_id)
                if TransactionCorrectionUndo.objects.filter(
                    correction=correction,
                ).exists():
                    messages.info(request, 'This void has already been undone.')
                    return redirect(redirect_name, redirect_id)

                # TransactionCorrectionView locks the parent transaction before
                # writing. Take the same lock so a new correction cannot race an
                # undo and claim the same units.
                if correction.order_id:
                    Order.objects.select_for_update().get(pk=correction.order_id)
                else:
                    CheckoutOrder.objects.select_for_update().get(
                        pk=correction.checkout_id,
                    )

                lines = list(
                    correction.lines.select_related(
                        'product', 'order_detail', 'checkout_item',
                    )
                )
                for line in lines:
                    if not line.product_id:
                        # The original void skipped inventory work for a product
                        # that no longer existed; there is nothing physical to
                        # reverse for this line.
                        continue
                    void_changes = line.stock_changes.filter(change_type='void')
                    if not void_changes.exists():
                        continue
                    product = Product.all_objects.select_for_update().get(
                        pk=line.product_id,
                    )
                    returned_to_stock = (
                        line.disposition
                        == TransactionCorrectionLine.DISPOSITION_RESTOCK
                    )
                    if returned_to_stock:
                        if product.quantity_in_stock < line.quantity:
                            raise ValidationError(
                                f'Undo unavailable: {product.name} now has only '
                                f'{product.quantity_in_stock} unit(s), but the '
                                f'void returned {line.quantity}. Use an inventory '
                                'correction if those units have already been used.'
                            )
                        product.quantity_in_stock -= line.quantity
                        product.save(update_fields=['quantity_in_stock'])

                    undo_change = record_stock_change(
                        product=product,
                        qty=-line.quantity,
                        change_type='correction_undo',
                        note=f'Undo void: {correction.reason}',
                        user=request.user,
                        order_detail=line.order_detail,
                        checkout_item=line.checkout_item,
                        correction_line=line,
                    )
                    if returned_to_stock:
                        remove_stock_from_recorded_lots(
                            product, line.quantity, void_changes, undo_change,
                        )

                TransactionCorrectionUndo.objects.create(
                    correction=correction,
                    created_by=request.user,
                )
                UserAction.objects.create(
                    user=request.user,
                    action='transaction_correction_undo',
                    target=correction.transaction_label,
                    detail=f'Undid void: {correction.reason}',
                )
        except ValidationError as exc:
            messages.error(request, exc.messages[0])
            return redirect(redirect_name, redirect_id)

        messages.success(
            request,
            'Void undone. Inventory and transaction totals were restored, and '
            'the void remains visible in correction history.',
        )
        return redirect(redirect_name, redirect_id)


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


def _filtered_transaction_export_rows(request):
    """Return the same combined POS/no-sale rows shown by ``OrderView``."""
    from app import reporting

    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    status_filter = request.GET.get('status', '')
    source_filter = request.GET.get('source', '')
    transactions = []

    if source_filter in ('', 'all', 'pos'):
        realized_details = reporting.realized_sales_lines(
            OrderDetail.objects.select_related('product', 'product__category'),
        ).order_by('pk')
        orders = reporting.annotate_orders_with_realized_sales(
            Order.objects.all(),
        ).prefetch_related(
            Prefetch(
                'details', queryset=realized_details,
                to_attr='realized_details',
            ),
        ).order_by('-order_date')
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
        orders = orders.filter(is_deleted=(status_filter == 'deleted'))
        for order in orders:
            transactions.append({
                'source': 'pos',
                'object': order,
                'date': order.order_date,
                'financials': reporting.realized_order_financials(
                    order, order.realized_details,
                ),
            })

    if source_filter in ('', 'all', 'giveaway') and status_filter not in ('pending', 'deleted'):
        checkouts = CheckoutOrder.objects.filter(
            status=CheckoutOrder.STATUS_SUBMITTED,
        ).prefetch_related('items', 'items__product')
        if date_from:
            parsed = parse_date(date_from)
            if parsed:
                checkouts = checkouts.filter(submitted_at__date__gte=parsed)
        if date_to:
            parsed = parse_date(date_to)
            if parsed:
                checkouts = checkouts.filter(submitted_at__date__lte=parsed)
        transactions.extend({
            'source': 'giveaway',
            'object': checkout,
            'date': checkout.submitted_at or checkout.created_at,
        } for checkout in checkouts)

    transactions.sort(key=lambda row: row['date'], reverse=True)
    return transactions, {
        'date_from': date_from,
        'date_to': date_to,
        'status': status_filter,
        'source': source_filter,
    }


class ExportAllOrdersPDFView(LoginRequiredMixin, View):
    """Generate a multi-order PDF report for all (filtered) transactions."""

    def get(self, request):
        from reportlab.lib.colors import HexColor

        transactions, export_filters = _filtered_transaction_export_rows(request)
        date_from = export_filters['date_from']
        date_to = export_filters['date_to']

        if not transactions:
            messages.info(request, "No transactions match the current filters.")
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
            (
                transaction['financials']['revenue']
                for transaction in transactions
                if transaction['source'] == 'pos'
            ),
            Decimal('0.00'),
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
        c.drawString(M + 14, y - 22, f"{len(transactions)} Transactions")
        c.setFont("Helvetica", 10)
        c.setFillColor(MUTED)
        c.drawString(M + 14, y - 38, date_range_str)

        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(BRAND)
        c.drawRightString(PAGE_W - M - 14, y - 22, f"${grand_revenue:.2f}")
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawRightString(PAGE_W - M - 14, y - 36, "POS revenue (before tax)")

        y -= 68

        # ════════════════════════════════════════
        # ORDER BLOCKS
        # ════════════════════════════════════════
        for transaction_row in transactions:
            source = transaction_row['source']
            order = transaction_row['object']
            is_giveaway = source == 'giveaway'
            details = list(order.items.all()) if is_giveaway else order.realized_details
            if not details:
                continue

            # Calculate how much space this order needs
            detail_count = len(details)
            needed = 60 + (detail_count * 16) + 60  # header + rows + summary
            y = check_space(y, min(needed, 200))  # at least start if it's huge

            # ── Order header bar ──
            c.setFillColor(DARK)
            c.rect(M, y - 4, content_w, 22, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 10)
            transaction_label = (
                f"No-sale #{order.pk}" if is_giveaway
                else f"Order #{order.order_id}"
            )
            c.drawString(M + 8, y + 2, transaction_label)
            c.setFont("Helvetica", 8)
            transaction_date = transaction_row['date']
            c.drawString(M + 100, y + 3, transaction_date.strftime("%b %d, %Y  %I:%M %p"))
            status = "No sale" if is_giveaway else ("Completed" if order.submitted else "Pending")
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
                realized_quantity = d.quantity if is_giveaway else d.realized_quantity
                line_total = d.price * realized_quantity
                order_subtotal += line_total

                c.setFont("Helvetica", 8)
                c.setFillColor(DARK)
                name = d.product_name if is_giveaway else d.display_name
                max_w = col_qty - col_prod - 20
                if stringWidth(name, "Helvetica", 8) > max_w:
                    while stringWidth(name + "...", "Helvetica", 8) > max_w and len(name) > 1:
                        name = name[:-1]
                    name += "..."
                c.drawString(col_prod, y + 2, name)
                c.drawRightString(col_qty, y + 2, str(realized_quantity))
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
            has_seniors_discount = False if is_giveaway else order.seniors_discount
            if is_giveaway:
                order_subtotal = order.subtotal
                order_tax = order.tax
                order_grand = order.total_price
            else:
                financials = transaction_row['financials']
                order_subtotal = financials['subtotal']
                seniors_amt = financials['discount_amount']
                order_tax = financials['tax']
                order_grand = financials['total']

            c.setFont("Helvetica", 8)
            c.setFillColor(MUTED)
            c.drawString(col_price - 20, y - 2, "Subtotal:")
            c.setFillColor(DARK)
            c.drawRightString(col_total, y - 2, f"${order_subtotal:.2f}")
            y -= 14

            if has_seniors_discount:
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
        from app import reporting

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
        base_qs = reporting.realized_sales_lines(
            OrderDetail.objects.filter(
                order__submitted=True,
                order__order_date__date__range=[start_date, end_date],
            ),
        )
        # "Ignore snacks" toggle — excludes the Snacks category from every series
        # below (they all derive from base_qs).
        ignore_snacks = request.GET.get('ignore_snacks') == '1'
        if ignore_snacks:
            base_qs = base_qs.exclude(product__category__name__iexact=reporting.SNACKS_CATEGORY_NAME)
        # Keep cost-only lines when corrected stock was not returned. Fully
        # restocked zero rows have neither realized units nor realized cost.
        financial_qs = base_qs.filter(
            Q(realized_quantity__gt=0) | Q(realized_cost__gt=0),
        )

        # ── KPI aggregates ─────────────────────────────────────────────────
        financial_lines = list(
            financial_qs.select_related('order', 'product__category')
            .order_by('order_id', 'pk')
        )
        settled_rows = reporting.settled_realized_sales_rows(
            financial_lines,
            preserve_full_snapshot=not ignore_snacks,
        )
        total_revenue_value = sum(
            (row['revenue'] for row in settled_rows), Decimal('0.00'),
        )
        total_cost_value = sum(
            (row['cost'] for row in settled_rows), Decimal('0.00'),
        )
        realized_order_ids = {
            row['order'].pk for row in settled_rows if row['units'] > 0
        }
        total_items = sum(row['units'] for row in settled_rows)
        total_revenue = float(total_revenue_value)
        total_cost = float(total_cost_value)
        total_profit = total_revenue - total_cost
        total_orders = len(realized_order_ids)
        avg_order = total_revenue / total_orders if total_orders else 0
        margin_pct = (total_profit / total_revenue * 100) if total_revenue else 0
        has_cost_data = total_cost > 0

        # ── Revenue series ─────────────────────────────────────────────────
        def period_for(order_date):
            local_day = localtime(order_date).date()
            if gran == 'day':
                return local_day
            if gran == 'week':
                return local_day - timedelta(days=local_day.weekday())
            return local_day.replace(day=1)

        label_fmt = {
            'day': '%d %b %Y',
            'week': '%d %b %Y',
            'month': '%b %Y',
        }[gran]
        period_totals = defaultdict(
            lambda: {
                'revenue': Decimal('0.00'),
                'cost': Decimal('0.00'),
                'orders': set(),
            },
        )
        product_totals = defaultdict(
            lambda: {
                'revenue': Decimal('0.00'),
                'cost': Decimal('0.00'),
                'units': 0,
            },
        )
        category_totals = defaultdict(
            lambda: {
                'revenue': Decimal('0.00'),
                'cost': Decimal('0.00'),
                'units': 0,
            },
        )
        category_product_totals = defaultdict(
            lambda: {
                'revenue': Decimal('0.00'),
                'cost': Decimal('0.00'),
                'units': 0,
            },
        )

        for row in settled_rows:
            line = row['line']
            order = row['order']
            period = period_for(order.order_date)
            period_totals[period]['revenue'] += row['revenue']
            period_totals[period]['cost'] += row['cost']
            if row['units'] > 0:
                period_totals[period]['orders'].add(order.pk)

            product_name = line.product_name
            category_name = (
                line.product.category.name
                if line.product_id and line.product.category_id
                else 'Uncategorised'
            )
            for totals in (
                product_totals[product_name],
                category_totals[category_name],
                category_product_totals[(category_name, product_name)],
            ):
                totals['revenue'] += row['revenue']
                totals['cost'] += row['cost']
                totals['units'] += row['units']

        revenue_series = []
        for period, totals in sorted(period_totals.items()):
            revenue = float(totals['revenue'])
            cost = float(totals['cost'])
            revenue_series.append({
                'label': period.strftime(label_fmt),
                'revenue': revenue,
                'cost': cost,
                'profit': revenue - cost,
                'orders': len(totals['orders']),
            })

        # ── Top 15 products by revenue ─────────────────────────────────────
        def totals_row(name, totals):
            revenue = float(totals['revenue'])
            cost = float(totals['cost'])
            return {
                'name': name,
                'revenue': revenue,
                'units': totals['units'],
                'cost': cost,
                'profit': revenue - cost,
            }

        top_products = [
            totals_row(name, totals)
            for name, totals in sorted(
                product_totals.items(),
                key=lambda item: (-item[1]['revenue'], item[0]),
            )[:15]
        ]

        # ── Category sales + margins ───────────────────────────────────────
        category_sales = [
            totals_row(name, totals)
            for name, totals in sorted(
                category_totals.items(),
                key=lambda item: (-item[1]['revenue'], item[0]),
            )
        ]

        # ── Top 5 products within each category ───────────────────────────
        top_by_cat = {}
        for (category_name, product_name), totals in sorted(
                category_product_totals.items(),
                key=lambda item: (
                    item[0][0], -item[1]['revenue'], item[0][1],
                )):
            rows = top_by_cat.setdefault(category_name, [])
            if len(rows) < 5:
                rows.append(totals_row(product_name, totals))

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
class AddProductByIdView(UserRequiredMixin, View):
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
            order = get_active_purchase_order(request)

            # ✅ FIXED: Add transaction and select_for_update
            with transaction.atomic():
                # Use the same lock order as submission: draft first, products
                # second. This avoids an add/auto-submit deadlock.
                if order is None:
                    order = Order.objects.create(user=request.user, draft_cart={})
                else:
                    order = (
                        Order.objects.select_for_update().filter(
                            pk=order.pk, user=request.user, submitted=False, is_deleted=False,
                        ).first()
                        or Order.objects.create(user=request.user, draft_cart={})
                    )
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
                    transaction.set_rollback(True)
                    return redirect("create_order")

                cart = dict(order.draft_cart or {})
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
                save_cart(request, cart, order=order)

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
    

class CreateOrderView(UserRequiredMixin, View):
    template_name = "order_form.html"
    AUTO_SUBMIT_SECONDS = 10 * 60

    def get_order(self, request, create_if_missing=False):
        """The current database-backed draft order, or None."""
        return get_active_purchase_order(request, create_if_missing=create_if_missing)


    def get(self, request, *args, **kwargs):
        form = BarcodeForm()
        order = self.get_order(request)

        cart = dict(order.draft_cart or {}) if order else {}

        # 🔁 Rehydrate products for template
        product_ids = [int(pid) for pid in cart.keys()]
        products = Product.objects.filter(product_id__in=product_ids)
        
        products_by_id = {p.product_id: p for p in products}

        order_items = []
        total_price_before_tax = Decimal("0.00")
        taxable_subtotal = Decimal("0.00")
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
            if product.taxable:
                taxable_subtotal += subtotal

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
        taxable_base = taxable_subtotal
        if seniors_discount:
            seniors_discount_amount = (total_price_before_tax * Decimal("0.10")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )
            taxable_base = taxable_subtotal * Decimal("0.90")

        tax_amount = (taxable_base * TAX_RATE).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        total_price_after_tax = total_price_before_tax - seniors_discount_amount + tax_amount

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
        if request.POST.get("action") == "reset_order_timer":
            order = self.get_order(request)
            if order is None or not order.draft_cart:
                return JsonResponse({'ok': False, 'error': 'There is no active order.'}, status=400)
            timestamp = now()
            order.draft_expires_at = timestamp + timedelta(seconds=self.AUTO_SUBMIT_SECONDS)
            order.last_timer_reset_at = timestamp
            order.timer_reset_count = F('timer_reset_count') + 1
            order.save(update_fields=[
                'draft_expires_at', 'last_timer_reset_at', 'timer_reset_count',
            ])
            order.refresh_from_db(fields=['draft_expires_at', 'timer_reset_count'])
            return JsonResponse({
                'ok': True,
                'expires_at': order.draft_expires_at.isoformat(),
                'expires_at_ms': int(order.draft_expires_at.timestamp() * 1000),
                'reset_count': order.timer_reset_count,
            })

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

        order = get_active_purchase_order(request)
        with transaction.atomic():
            if order is None:
                order = Order.objects.create(user=request.user, draft_cart={})
            else:
                order = (
                    Order.objects.select_for_update().filter(
                        pk=order.pk, user=request.user, submitted=False, is_deleted=False,
                    ).first()
                    or Order.objects.create(user=request.user, draft_cart={})
                )
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
                    transaction.set_rollback(True)
                    return redirect("create_order")

            # ── 2. Expiry guard ────────────────────────────────────────────
            if product.expiry_date and product.expiry_date < now().date():
                if not override_expiry:
                    messages.error(
                        request,
                        f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                        extra_tags="order",
                    )
                    transaction.set_rollback(True)
                    return redirect("create_order")

            # ── 3. Add to the authoritative database cart ──────────────────
            cart = dict(order.draft_cart or {})
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
            save_cart(request, cart, order=order)

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



class SubmitOrderView(UserRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        current_order = get_active_purchase_order(request)
        if current_order is None or not current_order.draft_cart:
            messages.error(request, "Cannot submit an empty order.", extra_tags="order")
            return redirect("create_order")

        unfulfilled_lines = []

        with transaction.atomic():
            order = get_object_or_404(
                Order.objects.select_for_update(),
                order_id=current_order.order_id,
                user=request.user,
                submitted=False,
                is_deleted=False,
            )
            cart = dict(order.draft_cart or {})
            if not cart:
                messages.error(request, "Cannot submit an empty order.", extra_tags="order")
                return redirect("create_order")

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

                available = max(0, int(product.quantity_in_stock or 0))
                fulfilled = min(requested, available) if requested > 0 else 0
                shortfall = max(0, requested - fulfilled)
                order_detail = None

                # A completed order line represents units actually supplied and
                # billed. Keep the requested-but-unavailable units exclusively
                # in the stockout ledger below.
                if fulfilled > 0:
                    order_detail = OrderDetail.objects.create(
                        order=order,
                        product=product,
                        product_name=product.name,
                        product_barcode=product.barcode or "",
                        quantity=fulfilled,
                        price=product.price,
                        taxable_at_sale=product.taxable,
                        cost_per_unit_at_sale=product.price_per_unit,
                        expiry_at_sale=product.expiry_date,
                    )

                    product.quantity_in_stock = available - fulfilled
                    product.save(update_fields=["quantity_in_stock"])

                    stock_change = record_stock_change(
                        product=product,
                        qty=fulfilled,
                        change_type="checkout",
                        note=f"Order {order.order_id} submission",
                        user=request.user,
                        order_detail=order_detail,
                    )
                    remove_stock_from_lots(product, fulfilled, stock_change)

                if shortfall > 0:
                    record_stock_change(
                        product=product,
                        qty=shortfall,
                        change_type="checkout_unfulfilled",
                        note=f"Order {order.order_id} — short {shortfall} (stockout)",
                        user=request.user,
                        order_detail=order_detail,
                    )
                    unfulfilled_lines.append(f"{product.name} (short {shortfall})")

                # Reordering analytics must reflect units that actually left
                # inventory, not requested units that could not be supplied.
                if fulfilled > 0:
                    rp = RecentlyPurchasedProduct.objects.filter(
                        product=product, archived_at__isnull=True,
                    ).first()
                    if rp is None:
                        rp = RecentlyPurchasedProduct.objects.create(product=product)
                    rp.quantity = (rp.quantity or 0) + fulfilled
                    rp.save(update_fields=["quantity"])

            # Finalize the authoritative financial snapshot before closing the
            # draft. All future transaction views read these stored values.
            recalculate_order_totals(
                order, snapshot_source=Order.SNAPSHOT_CAPTURED,
            )
            order.submitted = True
            order.draft_cart = {}
            order.save(update_fields=["submitted", "draft_cart"])

            UserAction.objects.create(
                user=request.user, action='submit_order',
                target=f'Order #{order.order_id}',
                detail=f'Total {order.total_price}',
            )

        # The browser keeps only a draft identifier, never cart contents.
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
    order = get_active_purchase_order(request)
    pid = str(product_id)  # Use item_id here as well

    if order is None:
        messages.warning(request, "Item not found in cart.")
        return redirect("create_order")

    with transaction.atomic():
        order = Order.objects.select_for_update().filter(
            pk=order.pk, user=request.user, submitted=False, is_deleted=False,
        ).first()
        cart = dict(order.draft_cart or {}) if order else {}
        if pid not in cart:
            messages.warning(request, "Item not found in cart.")
            return redirect("create_order")

        if cart[pid]["quantity"] > 1:
            cart[pid]["quantity"] -= 1
            save_cart(request, cart, order=order)
        else:
            del cart[pid]
            if cart:
                save_cart(request, cart, order=order)
            else:
                # Last item removed: delete the empty draft in the same atomic
                # operation so it cannot be resumed between update and delete.
                order.delete()
                request.session.pop("order_id", None)
                request.session.pop("cart", None)
                request.session.modified = True

    messages.success(request, "1 unit removed from the order.")
    return redirect("create_order")


# View for order success page
class OrderSuccessView(UserRequiredMixin, View):
    template_name = 'order_success.html'

    def get(self, request, order_id):
        order = get_object_or_404(Order, order_id=order_id)
        ctx = build_order_transaction_context(order)
        items = [
            {
                'name': row['detail'].display_name,
                'qty': row['detail'].quantity,
                'price': row['detail'].price,
                'total': row['total_price'],
            }
            for row in ctx['order_details_with_total']
        ]

        return render(request, self.template_name, {
            'order': order,
            'items': items,
            'subtotal': ctx['total_price_before_tax'],
            'total_tax': ctx['total_tax'],
            'grand_total': ctx['total_price_after_tax'],
            'seniors_discount': ctx['seniors_discount'],
            'seniors_discount_amount': ctx['seniors_discount_amount'],
            'item_count': ctx['total_items'],
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


def checkout_held_by_other(request, checkout):
    """Whether another still-registered browser owns this checkout draft."""
    if not request.session.session_key:
        request.session.save()
    holder_key = checkout.active_session_key
    return bool(
        holder_key
        and holder_key != request.session.session_key
        and UserSession.objects.filter(session_key=holder_key).exists()
    )


def reject_checkout_write_if_held(request, checkout):
    if not checkout_held_by_other(request, checkout):
        return None
    messages.warning(
        request,
        "That checkout is active on another computer. Return to Checkout and "
        "wait for it to be released.",
        extra_tags="order",
    )
    return redirect('checkout')


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
            s.is_mine = s.user_id == request.user.id
            s.can_continue = (
                s.holder_state != 'other'
                and (s.is_mine or has_admin_access(request))
            )
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
            # Purchase is a globally guarded work page. If another computer is
            # using it, no saved purchase cart can be resumed until that lock is
            # released, even when that computer currently has a different cart.
            o.purchase_blocked = bool(
                purchase_holder and purchase_holder.session_key != my_key
            )
            # Purchase drafts retain their original cashier for auditability.
            # Only that user can resume the cart.
            o.can_continue = o.is_mine and not o.purchase_blocked

        history_qs = CheckoutOrder.objects.filter(
            status=CheckoutOrder.STATUS_SUBMITTED,
            hidden_from_history=False,
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
        if not request.session.session_key:
            request.session.save()
        with transaction.atomic():
            co = get_object_or_404(
                CheckoutOrder.objects.select_for_update(), pk=checkout_id,
                status=CheckoutOrder.STATUS_DRAFT,
            )
            if co.user_id != request.user.id and not has_admin_access(request):
                messages.error(request, "That checkout belongs to another user.", extra_tags="order")
                return redirect('checkout')
            blocked = reject_checkout_write_if_held(request, co)
            if blocked:
                return blocked
            co.active_session_key = request.session.session_key or ''
            co.save(update_fields=['active_session_key', 'updated_at'])
        request.session['checkout_id'] = co.pk
        return redirect('checkout_cart')


class PurchaseContinueView(UserRequiredMixin, View):
    """Resume the exact saved purchase cart selected on the Checkout dashboard."""

    def post(self, request, order_id):
        if not request.session.session_key:
            request.session.save()
        my_key = request.session.session_key
        purchase_path = reverse('create_order')
        with transaction.atomic():
            order = get_object_or_404(
                Order.objects.select_for_update(),
                pk=order_id,
                submitted=False,
                is_deleted=False,
            )
            if not order.draft_cart:
                messages.error(request, "That purchase cart is empty.", extra_tags="order")
                return redirect('checkout')
            if order.user_id != request.user.id:
                messages.error(
                    request,
                    "That purchase cart belongs to another user.",
                    extra_tags="order",
                )
                return redirect('checkout')

            holder = PagePresence.objects.filter(page=purchase_path).first()
            if holder and holder.session_key != my_key and is_fresh(holder):
                who = holder.ip_address or 'another computer'
                messages.warning(
                    request,
                    f"Purchase is currently in use on {who}. Try again when it is available.",
                    extra_tags="order",
                )
                return redirect('checkout')

            # Point this browser at the selected database cart. Explicitly
            # reopening an old cart starts a fresh review window so an expired
            # client timer cannot submit it immediately on page load.
            timestamp = now()
            order.draft_expires_at = timestamp + timedelta(
                seconds=CreateOrderView.AUTO_SUBMIT_SECONDS,
            )
            order.last_timer_reset_at = timestamp
            order.timer_reset_count = F('timer_reset_count') + 1
            order.save(update_fields=[
                'draft_expires_at', 'last_timer_reset_at', 'timer_reset_count',
            ])

        request.session['order_id'] = order.pk
        request.session.pop('cart', None)
        request.session.modified = True
        messages.success(
            request,
            f"Purchase cart #{order.pk} reopened. Review it before submitting.",
            extra_tags="order",
        )
        return redirect('create_order')


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
            checkout = CheckoutOrder.objects.select_for_update().get(pk=checkout.pk)
            blocked = reject_checkout_write_if_held(request, checkout)
            if blocked:
                return blocked
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
                checkout = CheckoutOrder.objects.select_for_update().get(pk=checkout.pk)
                blocked = reject_checkout_write_if_held(request, checkout)
                if blocked:
                    return blocked
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

    with transaction.atomic():
        checkout = CheckoutOrder.objects.select_for_update().get(pk=checkout.pk)
        blocked = reject_checkout_write_if_held(request, checkout)
        if blocked:
            return blocked
        item = checkout.items.select_for_update().filter(pk=item_id).first()
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
            blocked = reject_checkout_write_if_held(request, checkout)
            if blocked:
                return blocked

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
                        stock_change = record_stock_change(
                            product=product, qty=deduct, change_type="giveaway",
                            note=f"PU Checkout {checkout.pk}", user=request.user,
                            checkout_item=it,
                        )
                        remove_stock_from_lots(product, deduct, stock_change)
                    shortfall = requested - deduct
                    if shortfall > 0:
                        record_stock_change(
                            product=product, qty=shortfall, change_type="giveaway_unfulfilled",
                            note=f"PU Checkout {checkout.pk} — short {shortfall} (stockout)",
                            user=request.user,
                            checkout_item=it,
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


class CheckoutHistoryDeleteView(AdminRequiredMixin, View):
    """Hide one submitted checkout from the chooser's History panel.

    Non-destructive: the record is kept, so it still shows on the Transactions
    page (as a giveaway row), in reports, and in the Stock Log.
    """
    def post(self, request, checkout_id):
        co = CheckoutOrder.objects.filter(
            pk=checkout_id, status=CheckoutOrder.STATUS_SUBMITTED
        ).first()
        if co:
            co.hidden_from_history = True
            co.save(update_fields=['hidden_from_history'])
            messages.success(
                request,
                f"Removed checkout #{checkout_id} from history. It remains on the Transactions page.",
                extra_tags="order",
            )
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
        if co and co.user_id != request.user.id and not has_admin_access(request):
            messages.error(request, "Admin passkey required to delete another user's draft.", extra_tags="order")
            return redirect('checkout')
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
    """Hide all submitted checkouts from the chooser's History panel.

    Non-destructive: the records are kept, so they still show on the
    Transactions page, in reports, and in the Stock Log.
    """
    def post(self, request):
        if not has_admin_access(request):
            messages.error(request, "Admin passkey required to clear shared checkout history.", extra_tags="order")
            return redirect('checkout')
        qs = CheckoutOrder.objects.filter(
            status=CheckoutOrder.STATUS_SUBMITTED,
            hidden_from_history=False,
        )
        count = qs.update(hidden_from_history=True)
        messages.success(
            request,
            f"Cleared {count} checkout(s) from history. They remain on the Transactions page.",
            extra_tags="order",
        )
        return redirect('checkout')


class GiveawayDetailView(LoginRequiredMixin, View):
    """Admin-readable detail for one terminal giveaway (a submitted CheckoutOrder)."""
    template_name = 'giveaway_detail.html'

    def get(self, request, checkout_id):
        checkout = get_object_or_404(
            CheckoutOrder, pk=checkout_id, status=CheckoutOrder.STATUS_SUBMITTED
        )
        items = checkout.items.select_related('product').all()
        item_rows = []
        for item in items:
            corrected = item.correction_lines.filter(
                correction__undo__isnull=True,
            ).aggregate(total=Sum('quantity'))['total'] or 0
            item_rows.append({
                'item': item,
                'corrected_qty': corrected,
                'remaining_qty': max(0, item.quantity - corrected),
            })
        return render(request, self.template_name, {
            'checkout': checkout,
            'items': items,
            'item_rows': item_rows,
            'item_count': sum(i.quantity for i in items),
            'corrections': checkout.corrections.prefetch_related('lines').select_related(
                'created_by', 'undo__created_by',
            ),
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
    raw = (raw or '').strip().rstrip('-')
    if not raw:
        return None
    for fmt in ('%d-%m-%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(raw, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _normalize_expiry_post(post_data, instance=None):
    """Normalize the primary expiry_date in a mutable POST copy so the form always
    receives a value the DateField accepts (ISO). A malformed/partial date can never
    block an otherwise-valid product edit: it falls back to the instance's current
    date (or blank). Mirrors the leniency of _parse_expiry_date for extra dates.
    """
    raw = post_data.get('expiry_date', '').strip().rstrip('-')
    if not raw:
        return post_data
    parsed = _parse_expiry_date(raw)
    if parsed:
        post_data['expiry_date'] = parsed.strftime('%Y-%m-%d')
    else:
        existing = getattr(instance, 'expiry_date', None) if instance else None
        post_data['expiry_date'] = existing.strftime('%Y-%m-%d') if existing else ''
    return post_data


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


def _submitted_lot_rows(post_data):
    """Parse the repeatable lot editor. None means no lot UI was submitted."""
    if 'lot_number' not in post_data and 'lot_quantity' not in post_data:
        return None
    numbers = post_data.getlist('lot_number')
    expiries = post_data.getlist('lot_expiry')
    quantities = post_data.getlist('lot_quantity')
    if not any(value.strip() for value in numbers + expiries + quantities):
        return None
    size = max(len(numbers), len(expiries), len(quantities), 1)
    combined = {}
    for index in range(size):
        number = (
            numbers[index].strip().upper()
            if index < len(numbers) and numbers[index].strip()
            else ProductLot.UNASSIGNED
        )
        expiry_raw = expiries[index].strip() if index < len(expiries) else ''
        expiry = _parse_expiry_date(expiry_raw)
        if expiry_raw and not expiry:
            raise ValidationError(f'Lot row {index + 1} has an invalid expiry date.')
        raw_qty = quantities[index].strip() if index < len(quantities) else '0'
        try:
            quantity = int(raw_qty or 0)
        except (TypeError, ValueError):
            raise ValidationError(f'Lot row {index + 1} needs a whole-number quantity.')
        if quantity < 0:
            raise ValidationError(f'Lot row {index + 1} cannot have negative stock.')
        key = (number, expiry)
        combined[key] = combined.get(key, 0) + quantity
    return [
        {'lot_number': key[0], 'expiry_date': key[1], 'quantity': quantity}
        for key, quantity in combined.items()
    ]


def _validate_lot_rows(form, post_data):
    try:
        rows = _submitted_lot_rows(post_data)
    except ValidationError as exc:
        form.add_error('quantity_in_stock', exc.message)
        return None, False
    if rows is None:
        return None, True
    expected = int(form.cleaned_data.get('quantity_in_stock') or 0)
    total = sum(row['quantity'] for row in rows)
    if total != expected:
        form.add_error(
            'quantity_in_stock',
            f'Lot quantities total {total}, but Units in Stock is {expected}. '
            'Adjust the lot rows so the totals match.',
        )
        return rows, False
    return rows, True


def _save_product_lots(product, rows, user=None, initial_stock_change=None):
    """Replace active lot quantities without deleting historical movement rows."""
    if rows is None:
        ensure_lot_balance(product)
        return
    existing = {
        (lot.lot_number, lot.expiry_date): lot
        for lot in ProductLot.objects.select_for_update().filter(
            product=product,
        )
    }
    submitted_keys = set()
    for row in rows:
        key = (row['lot_number'], row['expiry_date'])
        submitted_keys.add(key)
        lot = existing.get(key)
        if lot:
            lot.quantity_on_hand = row['quantity']
            lot.archived_at = None
            lot.archived_by = None
            lot.save(update_fields=[
                'quantity_on_hand', 'archived_at', 'archived_by', 'updated_at',
            ])
        else:
            lot = ProductLot.objects.create(
                product=product,
                lot_number=row['lot_number'],
                expiry_date=row['expiry_date'],
                quantity_on_hand=row['quantity'],
            )
        if initial_stock_change and row['quantity'] > 0:
            ProductLotMovement.objects.create(
                stock_change=initial_stock_change,
                lot=lot,
                lot_number=lot.lot_number,
                expiry_date=lot.expiry_date,
                quantity=row['quantity'],
                direction=ProductLotMovement.DIRECTION_IN,
            )
    for key, lot in existing.items():
        if key not in submitted_keys and lot.archived_at is None:
            lot.archived_at = now()
            lot.archived_by = user
            lot.save(update_fields=['archived_at', 'archived_by', 'updated_at'])

    dated = sorted({
        row['expiry_date'] for row in rows
        if row['quantity'] > 0 and row['expiry_date']
    })
    product.expiry_dates.all().delete()
    ProductExpiryDate.objects.bulk_create([
        ProductExpiryDate(product=product, expiry_date=value) for value in dated
    ])
    earliest = dated[0] if dated else None
    if product.expiry_date != earliest:
        product.expiry_date = earliest
        product.save(update_fields=['expiry_date'])


def _lot_rows_for_template(product=None, post_data=None):
    if post_data is not None and (
        'lot_number' in post_data or 'lot_quantity' in post_data
    ):
        try:
            return _submitted_lot_rows(post_data) or []
        except ValidationError:
            numbers = post_data.getlist('lot_number')
            expiries = post_data.getlist('lot_expiry')
            quantities = post_data.getlist('lot_quantity')
            return [
                {
                    'lot_number': numbers[i] if i < len(numbers) else '',
                    'expiry_date_raw': expiries[i] if i < len(expiries) else '',
                    'quantity': quantities[i] if i < len(quantities) else '',
                }
                for i in range(max(len(numbers), len(expiries), len(quantities), 1))
            ]
    if not product:
        return []
    return [
        {
            'lot_number': lot.lot_number,
            'expiry_date': lot.expiry_date,
            'quantity': lot.quantity_on_hand,
        }
        for lot in product.lots.filter(archived_at__isnull=True)
        .order_by(F('expiry_date').asc(nulls_last=True), 'lot_number')
    ]


def _saved_receiving_lots(product=None):
    """Active, usable lot/expiry pairs offered for quick check-in reuse."""
    if not product:
        return []
    return list(
        product.lots.filter(archived_at__isnull=True)
        .exclude(lot_number=ProductLot.UNASSIGNED)
        .filter(Q(expiry_date__isnull=True) | Q(expiry_date__gte=date.today()))
        .order_by('-updated_at', 'lot_number', 'pk')
    )


def _selected_receiving_lot_id(lots, raw_id):
    try:
        requested_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    return requested_id if any(lot.pk == requested_id for lot in lots) else None


def _receiving_lot_details(post_data, product):
    """Resolve a saved lot safely, or parse a manually entered lot/expiry pair."""
    saved_lot_id = str(post_data.get('existing_lot_id') or '').strip()
    if saved_lot_id:
        try:
            saved_lot_id = int(saved_lot_id)
        except (TypeError, ValueError):
            saved_lot_id = None
        saved_lot = None
        if saved_lot_id:
            saved_lot = (
                ProductLot.objects.select_for_update()
                .filter(
                    pk=saved_lot_id,
                    product=product,
                    archived_at__isnull=True,
                )
                .exclude(lot_number=ProductLot.UNASSIGNED)
                .filter(Q(expiry_date__isnull=True) | Q(expiry_date__gte=date.today()))
                .first()
            )
        if not saved_lot:
            raise ValidationError(
                'That saved lot is no longer available for this product. '
                'Choose another saved lot or enter a new one.'
            )
        return saved_lot.lot_number, saved_lot.expiry_date, saved_lot

    lot_number = str(post_data.get('lot_number') or '').strip()
    lot_expiry_raw = str(post_data.get('lot_expiry') or '').strip()
    lot_expiry = _parse_expiry_date(lot_expiry_raw)
    if lot_expiry_raw and lot_expiry is None:
        raise ValidationError('Enter the lot expiry as DD-MM-YYYY.')
    if lot_expiry and lot_expiry < date.today():
        raise ValidationError('The lot expiry cannot be in the past.')
    return lot_number, lot_expiry, None


def _checkin_product_url(session, product, receiving_lot_id=None):
    params = {'product_id': product.product_id}
    if receiving_lot_id:
        params['receiving_lot_id'] = receiving_lot_id
    session_url = reverse('checkin_session', kwargs={'session_id': session.pk})
    return f'{session_url}?{urlencode(params)}'


def _receiving_draft_context(session, product, saved_lots=None, preferred_lot_id=None):
    """Return restored receiving controls without changing quantity-bearing lots."""
    if not session or not product or session.inventory_mode:
        return {
            'receiving_draft': None,
            'receiving_draft_revision': 0,
            'selected_receiving_lot_id': None,
            'receiving_draft_lot_number': '',
            'receiving_draft_lot_expiry': '',
        }
    saved_lots = saved_lots if saved_lots is not None else _saved_receiving_lots(product)
    draft = (
        CheckinReceivingDraft.objects.select_related('existing_lot')
        .filter(session=session, product=product)
        .first()
    )
    requested_lot_id = preferred_lot_id
    if not requested_lot_id and draft:
        requested_lot_id = draft.existing_lot_id
    selected_lot_id = _selected_receiving_lot_id(saved_lots, requested_lot_id)
    # A valid saved selection fills the readonly fields from its option data in
    # the browser. Only a manually typed draft needs explicit input values.
    typed_number = draft.lot_number if draft and not selected_lot_id else ''
    typed_expiry = (
        draft.lot_expiry.strftime('%d-%m-%Y')
        if draft and draft.lot_expiry and not selected_lot_id else ''
    )
    return {
        'receiving_draft': draft,
        'receiving_draft_revision': draft.revision if draft else 0,
        'selected_receiving_lot_id': selected_lot_id,
        'receiving_draft_lot_number': typed_number,
        'receiving_draft_lot_expiry': typed_expiry,
    }


def _remember_receiving_lot_draft(session, product, lot):
    """Keep the received lot selected for the next + or barcode scan."""
    if not session or not product or not lot or session.inventory_mode:
        return None
    if lot.lot_number == ProductLot.UNASSIGNED:
        # Preserve the revision row as a tombstone. Deleting it would reset the
        # revision to zero and allow an old browser tab to overwrite a newer
        # receiving choice after a clear-and-recreate sequence.
        draft = (
            CheckinReceivingDraft.objects.select_for_update()
            .filter(session=session, product=product)
            .first()
        )
        if not draft:
            return None
        draft.existing_lot = None
        draft.lot_number = ''
        draft.lot_expiry = None
        draft.revision += 1
        draft.save(update_fields=[
            'existing_lot', 'lot_number', 'lot_expiry', 'revision', 'updated_at',
        ])
        return draft
    draft = (
        CheckinReceivingDraft.objects.select_for_update()
        .filter(session=session, product=product)
        .first()
    )
    if draft:
        draft.existing_lot = lot
        draft.lot_number = lot.lot_number
        draft.lot_expiry = lot.expiry_date
        draft.revision += 1
        draft.save(update_fields=[
            'existing_lot', 'lot_number', 'lot_expiry', 'revision', 'updated_at',
        ])
    else:
        draft = CheckinReceivingDraft.objects.create(
            session=session,
            product=product,
            existing_lot=lot,
            lot_number=lot.lot_number,
            lot_expiry=lot.expiry_date,
            revision=1,
        )
    return draft


def _serialize_receiving_draft(draft):
    if not draft:
        return None
    return {
        'existing_lot_id': draft.existing_lot_id,
        'lot_number': draft.lot_number,
        'lot_expiry': draft.lot_expiry.strftime('%d-%m-%Y') if draft.lot_expiry else '',
        'revision': draft.revision,
    }


@login_required
@require_POST
def save_checkin_receiving_draft(request, session_id, product_id):
    """Autosave receiving metadata without modifying Product or ProductLot stock."""
    with transaction.atomic():
        session = get_object_or_404(CheckinSession, pk=session_id)
        if not session.is_active:
            return JsonResponse(
                {'ok': False, 'error': 'This check-in session has ended.'},
                status=409,
            )
        if session.inventory_mode:
            return JsonResponse(
                {'ok': False, 'error': 'Receiving lots are not used during an inventory count.'},
                status=400,
            )
        # Match stock-update lock order: Product -> selected ProductLot -> draft.
        # The product lock also serializes concurrent first-draft creation.
        product = get_object_or_404(
            Product.objects.select_for_update(), product_id=product_id,
        )
        try:
            lot_number, lot_expiry, saved_lot = _receiving_lot_details(
                request.POST, product,
            )
        except ValidationError as exc:
            field = 'existing_lot_id' if request.POST.get('existing_lot_id') else 'lot_expiry'
            return JsonResponse(
                {'ok': False, 'error': exc.messages[0], 'field': field},
                status=400,
            )

        draft = (
            CheckinReceivingDraft.objects.select_for_update()
            .filter(session=session, product=product)
            .first()
        )
        try:
            expected_revision = int(request.POST.get('revision', 0) or 0)
        except (TypeError, ValueError):
            expected_revision = -1
        actual_revision = draft.revision if draft else 0
        if expected_revision != actual_revision:
            return JsonResponse({
                'ok': False,
                'conflict': True,
                'error': 'Receiving details changed in another action. The latest saved choice was kept.',
                'draft': _serialize_receiving_draft(draft),
            }, status=409)

        lot_number = (lot_number or '').strip().upper()
        if len(lot_number) > 64:
            return JsonResponse(
                {'ok': False, 'error': 'Lot number must be 64 characters or fewer.', 'field': 'lot_number'},
                status=400,
            )

        if not saved_lot and not lot_number and lot_expiry is None:
            if draft:
                # Keep a blank revision tombstone so stale clients cannot pass
                # the compare-and-swap check after a later draft is created.
                draft.existing_lot = None
                draft.lot_number = ''
                draft.lot_expiry = None
                draft.revision += 1
                draft.save(update_fields=[
                    'existing_lot', 'lot_number', 'lot_expiry', 'revision', 'updated_at',
                ])
            return JsonResponse({
                'ok': True,
                'cleared': True,
                'draft': _serialize_receiving_draft(draft),
            })

        if draft:
            draft.existing_lot = saved_lot
            draft.lot_number = lot_number
            draft.lot_expiry = lot_expiry
            draft.revision += 1
            draft.save(update_fields=[
                'existing_lot', 'lot_number', 'lot_expiry', 'revision', 'updated_at',
            ])
        else:
            draft = CheckinReceivingDraft.objects.create(
                session=session,
                product=product,
                existing_lot=saved_lot,
                lot_number=lot_number,
                lot_expiry=lot_expiry,
                revision=1,
            )

    return JsonResponse({
        'ok': True,
        'cleared': False,
        'draft': _serialize_receiving_draft(draft),
    })


#Change - Function to annotate changes

def record_stock_change(
    product: Product,
    qty: int,
    change_type: str,
    note: str = "",
    user=None,
    session=None,
    order_detail=None,
    checkout_item=None,
    correction_line=None,
) -> StockChange:
    """
    Creates a StockChange row and updates per-product counters.
    """
    with transaction.atomic():
        # 1) Persist the audit trail (snapshot product identity so the row stays
        #    readable if the product is later deleted → product FK becomes NULL).
        change = StockChange.objects.create(
            product=product,
            product_name=product.name,
            product_barcode=product.barcode or "",
            change_type=change_type,
            quantity=qty,
            note=note or None,
            user=user,
            session=session,
            order_detail=order_detail,
            checkout_item=checkout_item,
            correction_line=correction_line,
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

        elif change_type == "restoration":
            product.stock_deleted = max(
                0, (product.stock_deleted or 0) - abs(qty),
            )

        # Giveaway (PU terminal) — physically removes stock, but tracked
        # separately from sales so it never inflates stock_sold / sales demand.
        elif change_type == "giveaway":
            product.stock_giveaway = (product.stock_giveaway or 0) + abs(qty)

        elif change_type == "giveaway_unfulfilled":
            # No physical stock change and not a sale — audit row only.
            pass

        elif change_type in {"return", "return_no_restock", "void"}:
            if order_detail is not None:
                product.stock_sold = max(0, (product.stock_sold or 0) - abs(qty))
            elif checkout_item is not None:
                product.stock_giveaway = max(0, (product.stock_giveaway or 0) - abs(qty))

        elif change_type == "correction_undo":
            if order_detail is not None:
                product.stock_sold = (product.stock_sold or 0) + abs(qty)
            elif checkout_item is not None:
                product.stock_giveaway = (product.stock_giveaway or 0) + abs(qty)

        product.save(
            update_fields=[
                "stock_bought", "stock_sold", "stock_expired",
                "stock_unfulfilled", "stock_giveaway", "stock_deleted",
            ]
        )
        return change



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
            stock_change = record_stock_change(product, qty=1, change_type="checkin_delete1", note="1 unit removed via UI", user=request.user, session=session)
            remove_stock_from_lots(product, 1, stock_change)

    return redirect(f"{reverse('checkin_session', kwargs={'session_id': session.pk})}?product_id={product.product_id}")


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
        receiving_lot_id = None

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
            try:
                lot_number, lot_expiry, saved_lot = _receiving_lot_details(
                    request.POST, product,
                )
            except ValidationError as exc:
                messages.error(request, exc.messages[0], extra_tags='checkin error')
                return redirect(_checkin_product_url(session, product))
            product.quantity_in_stock += quantity_to_add
            product.save(update_fields=["quantity_in_stock"])
            change_note = 'Manual add via UI'
            if saved_lot:
                change_note += f' using saved lot {saved_lot.lot_number}'
            stock_change = record_stock_change(product, qty=quantity_to_add, change_type="checkin", note=change_note, user=request.user, session=session)
            lot = add_stock_to_lot(
                product, quantity_to_add, stock_change,
                lot_number=lot_number, expiry_date=lot_expiry, session=session,
            )
            _remember_receiving_lot_draft(session, product, lot)
            receiving_lot_id = lot.pk
            if lot_number:
                messages.success(
                    request,
                    f"Added {quantity_to_add} {product.name} to "
                    f"{'saved ' if saved_lot else ''}lot {lot.lot_number}.",
                    extra_tags="checkin success",
                )
            else:
                messages.warning(
                    request,
                    f"Added {quantity_to_add} {product.name}; no lot number was entered, so it is tracked as UNASSIGNED.",
                    extra_tags="checkin",
                )

    return redirect(_checkin_product_url(session, product, receiving_lot_id))


@login_required
def set_quantity(request, session_id, product_id):
    """Set a product's stock to an EXACT value — the check-in page lets you
    double-click the Units-in-Stock number and type the new total (e.g. a
    500-box delivery) instead of pressing ＋ repeatedly. The difference is
    recorded as a manual stock change. In inventory-count mode it sets the
    counted tally for this session instead of live stock.
    """
    session = get_object_or_404(CheckinSession, pk=session_id)
    if not session.is_active:
        messages.error(request, "This session has ended.", extra_tags="checkin error")
        return redirect("checkin_dashboard")

    session_url = reverse('checkin_session', kwargs={'session_id': session.pk})
    if request.method != "POST":
        return redirect("checkin_session", session_id=session.pk)

    def back():
        return redirect(f"{session_url}?product_id={product_id}")

    try:
        new_qty = int((request.POST.get("quantity", "") or "").strip())
    except (ValueError, TypeError):
        messages.error(request, "Enter a whole number for the quantity.", extra_tags="checkin error")
        return back()
    if new_qty < 0:
        messages.error(request, "Quantity can't be negative.", extra_tags="checkin error")
        return back()
    if new_qty > 100000:
        messages.error(request, "That quantity is too large (max 100000).", extra_tags="checkin error")
        return back()

    with transaction.atomic():
        product = get_object_or_404(Product.objects.select_for_update(), product_id=product_id)
        receiving_lot_id = None

        if session.inventory_mode:
            # Inventory-count mode: the visible number is the counted tally — set
            # THAT to new_qty (never live stock), via a delta on the count line.
            line, _ = _adjust_inventory_count(session, product, 0)  # ensure the line exists
            _adjust_inventory_count(session, product, new_qty - line.counted_qty)
            if not product.status:
                product.status = True
                product.save(update_fields=["status"])
            messages.success(
                request,
                f"Count set to {new_qty} for {product.name} (system {product.quantity_in_stock}).",
                extra_tags="checkin success",
            )
        else:
            old = product.quantity_in_stock
            delta = new_qty - old
            if delta == 0:
                messages.success(request, f"{product.name} is already at {new_qty} in stock.", extra_tags="checkin success")
            else:
                lot_number = lot_expiry = saved_lot = None
                if delta > 0:
                    try:
                        lot_number, lot_expiry, saved_lot = _receiving_lot_details(
                            request.POST, product,
                        )
                    except ValidationError as exc:
                        messages.error(request, exc.messages[0], extra_tags='checkin error')
                        return back()
                product.quantity_in_stock = new_qty
                product.save(update_fields=["quantity_in_stock"])
                change_type = "error_add" if delta > 0 else "error_subtract"
                change_note = f"Stock set to {new_qty} via check-in (was {old})"
                if saved_lot:
                    change_note += f' using saved lot {saved_lot.lot_number}'
                stock_change = record_stock_change(
                    product, qty=abs(delta), change_type=change_type,
                    note=change_note,
                    user=request.user, session=session,
                )
                if delta > 0:
                    lot = add_stock_to_lot(
                        product, delta, stock_change,
                        lot_number=lot_number, expiry_date=lot_expiry, session=session,
                    )
                    _remember_receiving_lot_draft(session, product, lot)
                    receiving_lot_id = lot.pk
                else:
                    remove_stock_from_lots(product, abs(delta), stock_change)
                sign = "+" if delta > 0 else ""
                lot_message = ''
                if delta > 0 and receiving_lot_id:
                    lot_message = (
                        f" into {'saved ' if saved_lot else ''}lot {lot.lot_number}"
                    )
                messages.success(
                    request,
                    f"{product.name} stock set to {new_qty} ({sign}{delta}){lot_message}.",
                    extra_tags="checkin success",
                )

    return redirect(_checkin_product_url(session, product, receiving_lot_id))

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

        return redirect(f"{reverse('checkin_session', kwargs={'session_id': session.pk})}?product_id={product.product_id}")

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
            .annotate(last_stock_change=Max('stock_changes__timestamp'))
            .select_related('user')
            .order_by('-started_at')
        )

        # Flag sessions another computer is currently working on (live page lock)
        presence = self._session_presence(request, active_sessions)
        for s in active_sessions:
            info = presence.get(s.pk)
            s.in_use = bool(info)
            s.in_use_by = ' · '.join(filter(None, [info['ip'], info['browser']])) if info else ''
            s.needs_review = checkin_session_needs_review(s)
            s.last_activity_at = checkin_session_last_activity(s)

        # Session history (all sessions, most recent first)
        sessions_qs = CheckinSession.objects.select_related('user').all()
        paginator = Paginator(sessions_qs, preferred_table_page_size(request, 25))
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


class CheckinReconcileView(AdminRequiredMixin, View):
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
                    stock_change = record_stock_change(
                        product, qty=abs(diff),
                        change_type='error_add' if diff > 0 else 'error_subtract',
                        note=f"Inventory count: {old} → {new}",
                        user=request.user, session=session,
                    )
                    if diff > 0:
                        add_stock_to_lot(product, diff, stock_change, session=session)
                    else:
                        remove_stock_from_lots(product, abs(diff), stock_change)
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
        session = get_object_or_404(CheckinSession, pk=session_id)
        if session.is_active:
            if checkin_session_needs_review(session):
                session.reopened_at = now()
                session.save(update_fields=['reopened_at'])
                UserAction.objects.create(
                    user=request.user,
                    action='reopen_session',
                    target=f'Session #{session.pk}',
                    detail='Confirmed resume after 24+ hours without activity',
                )
                messages.success(
                    request,
                    "Old session reviewed and resumed. Confirm its contents before scanning.",
                    extra_tags="checkin success",
                )
                return redirect('checkin_session', session_id=session.pk)
            messages.info(request, "Session is already active.", extra_tags="checkin info")
        else:
            if not has_admin_access(request):
                return redirect(f"{reverse('passkey_unlock')}?{urlencode({'next': request.get_full_path()})}")
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

            stock_delta = diff if original_was_add else -diff
            if stock_delta < 0 and abs(stock_delta) > product.quantity_in_stock:
                messages.error(
                    request,
                    f"Cannot reverse {abs(stock_delta)} units from {product.name}; "
                    f"only {product.quantity_in_stock} remain in stock.",
                    extra_tags="checkin error",
                )
                return redirect("checkin_session_detail", session_id=session.pk)

            # Update on-hand stock: if original was an add, more qty = more stock
            product.quantity_in_stock += stock_delta
            product.save(update_fields=["quantity_in_stock"])

            # Update the original change row
            change.quantity = new_qty
            change.save(update_fields=["quantity"])

            # Record corrective audit entry
            if diff > 0:
                corr_type = "error_add" if original_was_add else "error_subtract"
            else:
                corr_type = "error_subtract" if original_was_add else "error_add"

            stock_change_correction = record_stock_change(
                product=product,
                qty=abs(diff),
                change_type=corr_type,
                note=f"Session #{session.pk} line adjusted: {old_qty} → {new_qty}",
                user=request.user,
                session=session,
            )
            stock_increased = (
                (original_was_add and diff > 0)
                or (not original_was_add and diff < 0)
            )
            if stock_increased:
                add_stock_to_lot(
                    product, abs(diff), stock_change_correction, session=session,
                )
            else:
                remove_stock_from_lots(product, abs(diff), stock_change_correction)

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

            if original_was_add and change.quantity > product.quantity_in_stock:
                messages.error(
                    request,
                    f"Cannot remove this line: it added {change.quantity} units, "
                    f"but only {product.quantity_in_stock} remain in stock.",
                    extra_tags="checkin error",
                )
                return redirect("checkin_session_detail", session_id=session.pk)

            # Reverse the stock effect
            if original_was_add:
                product.quantity_in_stock -= change.quantity
                corr_type = "error_subtract"
            else:
                product.quantity_in_stock += change.quantity
                corr_type = "error_add"

            product.save(update_fields=["quantity_in_stock"])

            # Record corrective audit entry
            stock_change_correction = record_stock_change(
                product=product,
                qty=change.quantity,
                change_type=corr_type,
                note=f"Session #{session.pk} line removed (was {change.get_change_type_display()} x{change.quantity})",
                user=request.user,
                session=session,
            )
            if original_was_add:
                remove_stock_from_lots(product, change.quantity, stock_change_correction)
            else:
                add_stock_to_lot(
                    product, change.quantity, stock_change_correction, session=session,
                )

            # Preserve the original ledger row while removing it from the
            # session's editable list. The corrective row records the reversal.
            prod_name = product.name
            change_qty = change.quantity
            change.session = None
            change.note = (
                (change.note or '') + f' | Removed from Session #{session.pk}'
            ).strip(' |')
            change.save(update_fields=['session', 'note'])

        UserAction.objects.create(user=request.user, action='remove_session_line',
            target=f'Session #{session.pk}', detail=f'{prod_name} x{change_qty} removed')
        messages.success(
            request,
            f"Removed {prod_name} line and reversed stock.",
            extra_tags="checkin success",
        )
        return redirect("checkin_session_detail", session_id=session.pk)


class DeleteCheckinSessionView(AdminRequiredMixin, View):
    def post(self, request, session_id):
        session = get_object_or_404(CheckinSession, pk=session_id)
        # Unlink stock changes (keep the audit trail, just detach from session)
        session.stock_changes.update(session=None)
        session.delete()
        UserAction.objects.create(user=request.user, action='delete_session',
            target=f'Session #{session_id}')
        messages.success(request, "Session deleted.", extra_tags="checkin success")
        return redirect("checkin_dashboard")


class ClearCheckinHistoryView(AdminRequiredMixin, View):
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


def _restock_recommendation(pred, sold_60d):
    """Turn a reorder prediction into a plain 'keep restocking / buy less' call
    for the check-in Restock Trend card."""
    trend = pred.get('trend_label')
    velocity = pred.get('velocity')
    urgency = pred.get('urgency')
    sug = pred.get('suggested_qty', 0)
    pd = round(pred.get('adjusted_daily') or pred.get('avg_daily') or 0, 2)

    if sold_60d <= 0 or velocity == 'dead':
        tone, headline = 'dead', 'Hold off — no recent sales'
        detail = 'No sales in the last 60 days. Restocking ties up capital — buy little or none.'
    elif trend is None:
        tone, headline = 'new', 'Not enough history yet'
        detail = f'Only a little sales data so far (~{pd:g}/day). Restock cautiously.'
    elif trend == 'rising':
        tone, headline = 'up', 'Keep restocking — demand is rising'
        detail = f'Sales are trending up (~{pd:g}/day). Suggested order: {sug}.'
    elif trend == 'falling':
        tone, headline = 'down', 'Buy less — demand is falling'
        detail = f'Sales are trending down (~{pd:g}/day). Order lightly (suggested {sug}).'
    else:
        tone, headline = 'steady', 'Restock as usual — steady demand'
        detail = f'Demand is steady (~{pd:g}/day). Suggested order: {sug}.'

    if urgency in ('critical', 'warning') and tone in ('up', 'steady', 'new'):
        detail += ' Stock is low — order soon.'
    return {
        'tone': tone, 'headline': headline, 'detail': detail,
        'sold_60d': sold_60d, 'per_day': pd, 'suggested_qty': sug,
        'velocity': velocity or 'dead', 'trend': trend, 'urgency': urgency,
    }


#checkin views
class CheckinProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"
    SESSION_HISTORY_LIMIT = 50

    @classmethod
    def _session_history_context(cls, session):
        """Build the compact activity feed shown beside the active workflow."""
        if session.inventory_mode:
            history_qs = session.count_lines.filter(counted_qty__gt=0)
            summary = history_qs.aggregate(
                action_count=Count('pk'),
                net=Coalesce(Sum('counted_qty'), 0),
            )
            history = list(
                history_qs.select_related('product')
                .order_by('-updated_at')[:cls.SESSION_HISTORY_LIMIT]
            )
            return {
                'session_history': history,
                'session_history_is_count': True,
                'session_history_action_count': summary['action_count'],
                'session_history_product_count': summary['action_count'],
                'session_history_net': summary['net'],
                'session_history_has_more': summary['action_count'] > len(history),
                'session_history_limit': cls.SESSION_HISTORY_LIMIT,
            }

        positive_types = ['checkin', 'error_add']
        negative_types = ['checkin_delete1', 'error_subtract']
        history_qs = session.stock_changes.filter(
            change_type__in=positive_types + negative_types,
        )
        summary = history_qs.aggregate(
            action_count=Count('pk'),
            product_count=Count(
                Coalesce(
                    Cast('product_id', output_field=CharField()),
                    NullIf('product_barcode', Value('')),
                    'product_name',
                ),
                distinct=True,
            ),
            positive=Coalesce(Sum('quantity', filter=Q(change_type__in=positive_types)), 0),
            negative=Coalesce(Sum('quantity', filter=Q(change_type__in=negative_types)), 0),
        )
        history = list(
            history_qs.select_related('product')
            .order_by('-timestamp')[:cls.SESSION_HISTORY_LIMIT]
        )
        return {
            'session_history': history,
            'session_history_is_count': False,
            'session_history_action_count': summary['action_count'],
            'session_history_product_count': summary['product_count'],
            'session_history_net': summary['positive'] - summary['negative'],
            'session_history_has_more': summary['action_count'] > len(history),
            'session_history_limit': cls.SESSION_HISTORY_LIMIT,
        }

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

        # Per-product 90-day daily movement chart.
        history_chart = []
        restock = None
        if product:
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

            # Restock trend: rising demand (keep restocking) vs falling (buy less)
            from app.utils import get_reorder_prediction
            from app.reporting import SALE_TYPES
            from django.db.models.functions import TruncWeek
            since60 = date.today() - timedelta(days=60)
            demand_60 = StockChange.objects.filter(
                product=product, timestamp__date__gte=since60, change_type__in=SALE_TYPES,
            ).aggregate(t=Sum('quantity'))['t'] or 0
            weekly = [(r['week'], r['total']) for r in (
                StockChange.objects.filter(
                    product=product, timestamp__date__gte=since60, change_type__in=SALE_TYPES,
                ).annotate(week=TruncWeek('timestamp')).values('week')
                .annotate(total=Sum('quantity')).order_by('week'))]
            pred = get_reorder_prediction(product, int(demand_60), weekly_demands=weekly)
            restock = _restock_recommendation(pred, int(demand_60))

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

        saved_receiving_lots = _saved_receiving_lots(product)
        receiving_draft_context = _receiving_draft_context(
            session,
            product,
            saved_receiving_lots,
            preferred_lot_id=request.GET.get('receiving_lot_id'),
        )

        context = {
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
            "product_lots": _lot_rows_for_template(product),
            "saved_receiving_lots": saved_receiving_lots,
            "edit_form": edit_form,
            "extra_dates": product.expiry_dates.all() if product else [],
            "categories": Category.objects.all(),
            "recent_scans": recent_scans,
            "scanned_today_count": scanned_today_count,
            "products_updated_today": products_updated_today,
            "last_checkin": last_checkin,
            "history_chart": history_chart,
            "restock": restock,
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
        }
        context.update(receiving_draft_context)
        context.update(self._session_history_context(session))
        return render(request, self.template_name, context)

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
            receiving_lot_id = None

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
                        try:
                            lot_number, lot_expiry, saved_lot = _receiving_lot_details(
                                request.POST, product,
                            )
                        except ValidationError as exc:
                            messages.error(request, exc.messages[0], extra_tags='checkin error')
                            return redirect(_checkin_product_url(session, product))
                        product.quantity_in_stock += 1
                        product.save(update_fields=["quantity_in_stock"])
                        change_note = 'Barcode scan (+1)'
                        if saved_lot:
                            change_note += f' using saved lot {saved_lot.lot_number}'
                        stock_change = record_stock_change(
                            product, qty=1, change_type="checkin",
                            note=change_note, user=request.user, session=session,
                        )
                        lot = add_stock_to_lot(
                            product, 1, stock_change,
                            lot_number=lot_number, expiry_date=lot_expiry,
                            session=session,
                        )
                        _remember_receiving_lot_draft(session, product, lot)
                        receiving_lot_id = lot.pk
                        messages.success(
                            request,
                            f"+1 {product.name} (now {product.quantity_in_stock}); "
                            f"tracked in {'saved ' if saved_lot else ''}lot {lot.lot_number}.",
                            extra_tags="checkin success",
                        )

            return redirect(_checkin_product_url(session, product, receiving_lot_id))

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
        context = {
            "session": session,
            "inventory_mode": inventory_mode,
            "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number", "barcode")),
            "categories": Category.objects.all(),
            "change_types": StockChange._meta.get_field('change_type').choices,
        }
        if session:
            context.update(self._session_history_context(session))
        return render(request, self.template_name, context)

    
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

            post_data = _normalize_expiry_post(request.POST.copy(), product)
            # Stock is controlled by the dedicated + / - / exact-quantity tools.
            # Never let a stale hidden value in the inline details form overwrite
            # or double-record a stock adjustment made while editing.
            post_data["quantity_in_stock"] = str(product.quantity_in_stock)

            form = EditProductForm(post_data, instance=product)

            if form.is_valid():
                lot_rows, lots_valid = _validate_lot_rows(form, request.POST)
            else:
                lot_rows, lots_valid = None, False

            if form.is_valid() and lots_valid:
                updated = form.save(commit=False)
                updated.save()
                form.save_m2m()
                _save_expiry_dates(updated, updated.expiry_date, request.POST.getlist('extra_expiry_dates'))
                _save_product_lots(updated, lot_rows, request.user)
                UserAction.objects.create(user=request.user, action='edit_product',
                    target=updated.name, detail=f'Edited via check-in inline (Session #{session.pk})')
                messages.success(request, f"Updated {updated.name}.", extra_tags="checkin success")
                # Redirect by product_id (always present) so the just-edited
                # product stays shown — barcode may be blank or have just been
                # changed in the edit.
                return redirect(f"{session_url}?product_id={updated.product_id}")

        messages.error(request, "Could not update product. Please review the highlighted fields.", extra_tags="checkin error")
        return render(request, self.template_name, {
            "session": session,
            "search_results": [],
            "inventory_mode": inventory_mode,
            "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number", "barcode")),
            "product": product,
            "product_lots": _lot_rows_for_template(product, request.POST),
            "saved_receiving_lots": _saved_receiving_lots(product),
            **_receiving_draft_context(session, product),
            "edit_form": form,
            "extra_dates": product.expiry_dates.all(),
            "categories": Category.objects.all(),
            "change_types": StockChange._meta.get_field('change_type').choices,
        })


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
                'is_custom': i.is_custom,
                'custom_lines': i.custom_lines if isinstance(i.custom_lines, list) else [],
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


def _restore_label_session_queue(user, items, replace=False):
    """Restore product and custom snapshot rows into a user's current queue."""
    product_rows = [
        LabelQueueItem(product=item.product, user=user, qty=item.qty)
        for item in items
        if not item.is_custom and item.product_id is not None
    ]
    custom_rows = [
        CustomLabelQueueItem(
            user=user,
            title=item.product_name[:200],
            lines=item.custom_lines if isinstance(item.custom_lines, list) else [],
            copies=max(1, int(item.qty or 1)),
        )
        for item in items
        if item.is_custom and item.product_name
    ]

    if not product_rows and not custom_rows:
        return 0

    with transaction.atomic():
        if replace:
            LabelQueueItem.objects.filter(user=user).delete()
            CustomLabelQueueItem.objects.filter(user=user).delete()
        LabelQueueItem.objects.bulk_create(product_rows)
        CustomLabelQueueItem.objects.bulk_create(custom_rows)
    return len(product_rows) + len(custom_rows)


class LabelSessionRegenerateView(LoginRequiredMixin, View):
    """POST → reload session items back into the current label queue."""
    def post(self, request, session_id):
        session_obj = get_object_or_404(LabelSession, pk=session_id, user=request.user)
        items = list(session_obj.items.select_related('product'))
        loaded = _restore_label_session_queue(request.user, items, replace=True)
        if not loaded:
            return JsonResponse({'ok': False, 'error': 'No restorable labels in this session.'}, status=400)

        UserAction.objects.create(user=request.user, action='regenerate_label_session',
            target=f'Label Session #{session_id}', detail=f'{loaded} items loaded')
        return JsonResponse({'ok': True, 'loaded': loaded})


class LabelSessionAddToQueueView(LoginRequiredMixin, View):
    """POST → append session items to the current queue (without clearing it)."""
    def post(self, request, session_id):
        session_obj = get_object_or_404(LabelSession, pk=session_id, user=request.user)
        items = list(session_obj.items.select_related('product'))
        added = _restore_label_session_queue(request.user, items)
        if not added:
            return JsonResponse({'ok': False, 'error': 'No restorable labels in this session.'}, status=400)
        return JsonResponse({'ok': True, 'added': added})


class LabelSessionClearAllView(LoginRequiredMixin, View):
    """POST → delete all sessions for this user."""
    def post(self, request):
        deleted_count, _ = LabelSession.objects.filter(user=request.user).delete()
        UserAction.objects.create(user=request.user, action='clear_all_label_sessions',
            target=f'{deleted_count} label sessions cleared')
        return JsonResponse({'ok': True, 'deleted': deleted_count})


# Edit product.
class EditProductView(AdminRequiredMixin, View):
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
            'lot_rows': _lot_rows_for_template(product),
        })


    def post(self, request, product_id):
            product = get_object_or_404(Product, product_id=product_id)

            # Normalize the primary expiry date to ISO so the form accepts it; a
            # malformed/partial date falls back to the product's current date and
            # can never block a category/barcode edit.
            post_data = _normalize_expiry_post(request.POST.copy(), product)

            form = EditProductForm(post_data, instance=product)
            next_url = request.POST.get('next', '/inventory_display')

            lot_rows = None
            lots_valid = False
            if form.is_valid():
                lot_rows, lots_valid = _validate_lot_rows(form, request.POST)

            if not form.is_valid() or not lots_valid:
                # Surface the real validation errors instead of a hard-coded (and
                # often wrong) "date format" message.
                error_bits = []
                for field_name, errs in form.errors.items():
                    label = 'Form' if field_name == '__all__' else field_name.replace('_', ' ').title()
                    error_bits.append(f"{label}: {'; '.join(errs)}")
                messages.error(request, "Could not update product — " + " | ".join(error_bits))
                return render(request, self.template_name, {
                    'form': form,
                    'next': next_url,
                    'product': product,
                    'extra_dates': product.expiry_dates.all(),
                    'lot_rows': _lot_rows_for_template(product, request.POST),
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
                updated_product.save()
                form.save_m2m()
                # --- STOCK CHANGE TRACKING ---
                delta = updated_product.quantity_in_stock - old_quantity
                if delta != 0:
                    stock_change = record_stock_change(
                        product=updated_product,
                        qty=abs(delta),
                        change_type="error_add" if delta > 0 else "error_subtract",
                        note="Product updated via edit form",
                        user=request.user,
                    )
                    if lot_rows is None:
                        if delta > 0:
                            add_stock_to_lot(updated_product, delta, stock_change)
                        else:
                            remove_stock_from_lots(updated_product, abs(delta), stock_change)

                _save_expiry_dates(updated_product, updated_product.expiry_date, request.POST.getlist('extra_expiry_dates'))
                _save_product_lots(updated_product, lot_rows, request.user)

            UserAction.objects.create(user=request.user, action='edit_product',
                target=updated_product.name, detail='Edited via product form')
            messages.success(request, f"Product '{updated_product.name}' updated successfully.")
            return redirect(next_url)

# Add a new product
class AddProductView(AdminRequiredMixin, View):
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
            'lot_rows': [],
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

            # Barcodes identify scan targets and remain unique. Item numbers are
            # supplier/catalog references and are intentionally allowed to repeat.
            if barcode:
                normalized = _normalize_barcode(raw_barcode)
                duplicate = Product.all_objects.filter(
                    normalized_barcode=normalized,
                ).first()
                if duplicate:
                    if duplicate.archived_at:
                        form.add_error(
                            "barcode",
                            f"Barcode '{raw_barcode}' belongs to archived product "
                            f"'{duplicate.name}'. Restore it from Recovery instead.",
                        )
                    else:
                        form.add_error(
                            "barcode", f"Barcode '{raw_barcode}' already exists."
                        )

            lot_rows, lots_valid = _validate_lot_rows(form, request.POST)

            # If custom checks added errors, return the form immediately
            if form.errors or not lots_valid:
                return render(request, self.template_name, {
                    'categories': Category.objects.all(),
                    'form': form,
                    'next': next_url,
                    'lot_rows': _lot_rows_for_template(post_data=request.POST),
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
                    initial_change = None
                    if stock_qty:
                        initial_change = record_stock_change(
                            product=product,
                            qty=int(stock_qty),
                            change_type="checkin",
                            note="New product added via form",
                            user=request.user,
                        )
                    if lot_rows is None:
                        if stock_qty:
                            add_stock_to_lot(product, int(stock_qty), initial_change)
                        else:
                            _save_product_lots(product, None, request.user)
                    else:
                        _save_product_lots(
                            product, lot_rows, request.user,
                            initial_stock_change=initial_change,
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
            'next': next_url,
            'lot_rows': _lot_rows_for_template(post_data=request.POST),
        })

# Display inventory
def _category_selection_is_a_subset(category_ids):
    """Return True only when named categories narrow the full inventory."""
    return bool(category_ids) and Category.objects.exclude(pk__in=category_ids).exists()


class InventoryView(LoginRequiredMixin, View):
    template_name = 'inventory_display.html'

    def get(self, request):
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        # Get filter parameters from the request. Category is multi-select:
        # any number of category_id params (category_id=3&category_id=7).
        selected_category_ids = [c for c in request.GET.getlist('category_id') if c.strip().isdigit()]
        # `q` is the single name / SKU / barcode field. Keep accepting the two
        # legacy parameters so old bookmarks and links continue to work.
        search_query = (
            request.GET.get('q')
            or request.GET.get('barcode_query')
            or request.GET.get('name_query')
            or ''
        ).strip()
        sort_column = request.GET.get('sort', 'name')  # Default sorting column is 'name'
        sort_direction = request.GET.get('direction', 'asc')  # Default sorting direction is ascending

        # Reusable query fragment so every link/action keeps the selected
        # categories (e.g. "&category_id=3&category_id=7"). Ids are numeric.
        category_qs = ''.join('&category_id=' + c for c in selected_category_ids)

        # Query products based on filters
        products = Product.objects.select_related('category').prefetch_related('expiry_dates', 'lots').annotate(
            stock_threshold=Coalesce(F('category__low_stock_threshold'), Value(3))
        )
        if _category_selection_is_a_subset(selected_category_ids):
            products = products.filter(category_id__in=selected_category_ids)

        if search_query:
            products = products.filter(
                Q(name__icontains=search_query)
                | Q(item_number__icontains=search_query)
                | barcode_search_q(search_query)
            )

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

        # Paginate consistently for both full loads and AJAX so the live
        # search and the floating pager always agree.
        paginator = Paginator(products, preferred_table_page_size(request, 100))
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        # AJAX early return — table rows + the re-rendered pager
        if is_ajax:
            pager_ctx = {
                'page_obj': page_obj,
                'sort_column': sort_column,
                'sort_direction': sort_direction,
                'category_qs': category_qs,
                'search_query': search_query,
            }
            rows_html = render_to_string('partials/inv_rows.html', {'page_obj': page_obj}, request=request)
            pager_html = render_to_string('partials/inv_pager.html', pager_ctx, request=request)
            return JsonResponse({
                'html': rows_html,
                'pager': pager_html,
                'count': paginator.count,
                'num_pages': paginator.num_pages,
            })

        # Aggregate stats for the filtered queryset
        stats = products.aggregate(
            total_units=Sum('quantity_in_stock'),
            total_retail=Sum(F('price') * F('quantity_in_stock')),
            total_cost=Sum(F('price_per_unit') * F('quantity_in_stock')),
        )

        # Pass all query parameters and the paginator to the template
        from app.inventory_audit import serialize_audit_run
        latest_audit = (
            InventoryAuditRun.objects.select_related('created_by')
            .prefetch_related('issues')
            .first()
        )
        return render(request, self.template_name, {
            'page_obj': page_obj,
            'categories': Category.objects.all(),
            'selected_category_ids': selected_category_ids,
            'category_qs': category_qs,
            'search_query': search_query,
            'sort_column': sort_column,
            'sort_direction': sort_direction,
            'total_products': paginator.count,
            'total_units': stats['total_units'] or 0,
            'total_retail': stats['total_retail'] or Decimal('0.00'),
            'total_cost': stats['total_cost'] or Decimal('0.00'),
            'latest_inventory_audit': serialize_audit_run(latest_audit),
        })


class InventoryAuditAPIView(LoginRequiredMixin, View):
    """Read saved audit results or run a protected audit without a page reload."""

    @staticmethod
    def _history():
        return [
            {
                'id': run.pk,
                'status': run.status,
                'status_label': run.get_status_display(),
                'summary': run.summary,
                'issue_count': run.issue_count,
                'repaired_count': run.repaired_count,
                'started_at': run.started_at.isoformat(),
                'created_by': (
                    run.created_by.get_short_name() or run.created_by.get_username()
                    if run.created_by else 'System'
                ),
            }
            for run in InventoryAuditRun.objects.select_related('created_by')[:10]
        ]

    def get(self, request):
        from app.inventory_audit import serialize_audit_run

        run_id = request.GET.get('run_id')
        queryset = InventoryAuditRun.objects.select_related('created_by').prefetch_related('issues')
        if run_id:
            try:
                run_id = int(run_id)
            except (TypeError, ValueError):
                return JsonResponse({'ok': False, 'error': 'Invalid audit id.'}, status=400)
        run = queryset.filter(pk=run_id).first() if run_id else queryset.first()
        return JsonResponse({
            'ok': True,
            'run': serialize_audit_run(run),
            'history': self._history(),
            'can_repair': has_admin_access(request),
        })

    def post(self, request):
        if not has_admin_access(request):
            unlock_url = reverse('passkey_unlock') + '?' + urlencode({
                'next': reverse('inventory_display'),
            })
            return JsonResponse({
                'ok': False,
                'requires_admin': True,
                'unlock_url': unlock_url,
                'error': 'Admin passkey required to run or repair an inventory audit.',
            }, status=403)

        try:
            data = json.loads(request.body or '{}')
        except ValueError:
            return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)
        action = data.get('action', 'run')
        if action not in {'run', 'repair'}:
            return JsonResponse({'ok': False, 'error': 'Unknown audit action.'}, status=400)

        from app.inventory_audit import run_inventory_audit, serialize_audit_run

        run = run_inventory_audit(
            created_by=request.user,
            repair_unassigned=action == 'repair',
        )
        payload = {
            'ok': run.status != InventoryAuditRun.STATUS_ERROR,
            'run': serialize_audit_run(run),
            'history': self._history(),
            'can_repair': True,
        }
        return JsonResponse(
            payload,
            status=500 if run.status == InventoryAuditRun.STATUS_ERROR else 200,
        )

class ExportInventoryCSVView(LoginRequiredMixin, View):
    def get(self, request):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="inventory_{now().strftime("%Y%m%d_%H%M")}.csv"'

        writer = csv.writer(response)
        writer.writerow(['Name', 'Barcode', 'SKU', 'Category', 'Price', 'Cost', 'Qty In Stock', 'Lot Numbers', 'Status', 'Expiry Date'])

        products = Product.objects.select_related('category').prefetch_related('expiry_dates', 'lots').all()

        # Apply same filters as inventory page (multi-select categories)
        category_ids = [c for c in request.GET.getlist('category_id') if c.strip().isdigit()]
        search_query = (
            request.GET.get('q')
            or request.GET.get('barcode_query')
            or request.GET.get('name_query')
            or ''
        ).strip()
        if _category_selection_is_a_subset(category_ids):
            products = products.filter(category_id__in=category_ids)
        if search_query:
            products = products.filter(
                Q(name__icontains=search_query)
                | Q(item_number__icontains=search_query)
                | barcode_search_q(search_query)
            )

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
                '; '.join(
                    f'{lot.lot_number}:{lot.quantity_on_hand}'
                    for lot in p.lots.all()
                    if lot.archived_at is None and lot.quantity_on_hand
                ),
                'Active' if p.status else 'Inactive',
                '; '.join(d.expiry_date.strftime('%Y-%m-%d') for d in p.expiry_dates.all()) or (p.expiry_date.strftime('%Y-%m-%d') if p.expiry_date else ''),
            ])

        return response


class ExportTransactionsCSVView(LoginRequiredMixin, View):
    def get(self, request):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="transactions_{now().strftime("%Y%m%d_%H%M")}.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Order ID', 'Date', 'Status', 'Product Name at Sale', 'Barcode at Sale',
            'Quantity', 'Unit Price at Sale', 'Line Total', 'Taxable at Sale',
            'Unit Cost at Sale', 'Order Subtotal', 'Order Discount', 'Order Tax',
            'Order Total', 'Financial Snapshot Source', 'Source',
        ])

        transactions, _ = _filtered_transaction_export_rows(request)
        for transaction_row in transactions:
            source = transaction_row['source']
            transaction_record = transaction_row['object']
            transaction_date = transaction_row['date'].strftime('%Y-%m-%d %H:%M')

            if source == 'giveaway':
                for item in transaction_record.items.all():
                    writer.writerow([
                        transaction_record.pk,
                        transaction_date,
                        'No sale',
                        item.product_name,
                        item.product_barcode or '',
                        item.quantity,
                        f'{item.price:.2f}',
                        f'{item.line_total:.2f}',
                        'Yes' if item.taxable else 'No',
                        '',
                        f'{transaction_record.subtotal:.2f}',
                        '0.00',
                        f'{transaction_record.tax:.2f}',
                        f'{transaction_record.total_price:.2f}',
                        'pu_checkout',
                        'PU No-Sale',
                    ])
                continue

            status = (
                'Deleted' if transaction_record.is_deleted
                else ('Completed' if transaction_record.submitted else 'Pending')
            )
            financials = transaction_row['financials']
            for detail in transaction_record.realized_details:
                line_total = detail.price * detail.realized_quantity
                writer.writerow([
                    transaction_record.order_id,
                    transaction_date,
                    status,
                    detail.product_name,
                    detail.product_barcode or '',
                    detail.realized_quantity,
                    f'{detail.price:.2f}',
                    f'{line_total:.2f}',
                    'Yes' if detail.taxable_at_sale is True else ('No' if detail.taxable_at_sale is False else 'Unknown'),
                    f'{detail.cost_per_unit_at_sale:.2f}' if detail.cost_per_unit_at_sale is not None else '',
                    f'{financials["subtotal"]:.2f}',
                    f'{financials["discount_amount"]:.2f}',
                    f'{financials["tax"]:.2f}',
                    f'{financials["total"]:.2f}',
                    transaction_record.financial_snapshot_source,
                    'POS',
                ])

        return response


# ========== NEW FEATURE VIEWS ==========

class TablePreferenceAPIView(LoginRequiredMixin, View):
    """Persist a user's density, page size, and visible columns for one table."""

    @staticmethod
    def _keys(request, payload=None):
        payload = payload or {}
        page_key = str(payload.get('page_key') or request.GET.get('page_key') or '').strip()
        table_key = str(payload.get('table_key') or request.GET.get('table_key') or 'main').strip()
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,100}', page_key):
            raise ValidationError('Invalid page preference key.')
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,100}', table_key):
            raise ValidationError('Invalid table preference key.')
        return page_key, table_key

    @staticmethod
    def _serialize(preference):
        return {
            'page_key': preference.page_key,
            'table_key': preference.table_key,
            'density': preference.density,
            'page_size': preference.page_size,
            'hidden_columns': preference.hidden_columns,
        }

    def get(self, request):
        try:
            page_key, table_key = self._keys(request)
        except ValidationError as exc:
            return JsonResponse({'ok': False, 'error': exc.message}, status=400)
        preference = UserTablePreference.objects.filter(
            user=request.user,
            page_key=page_key,
            table_key=table_key,
        ).first()
        if not preference:
            return JsonResponse({
                'ok': True,
                'preference': {
                    'page_key': page_key,
                    'table_key': table_key,
                    'density': UserTablePreference.DENSITY_COMFORTABLE,
                    'page_size': 50,
                    'hidden_columns': [],
                },
            })
        return JsonResponse({'ok': True, 'preference': self._serialize(preference)})

    def post(self, request):
        try:
            payload = json.loads(request.body.decode('utf-8') or '{}')
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JsonResponse({'ok': False, 'error': 'Invalid preference data.'}, status=400)
        try:
            page_key, table_key = self._keys(request, payload)
        except ValidationError as exc:
            return JsonResponse({'ok': False, 'error': exc.message}, status=400)

        if payload.get('reset'):
            UserTablePreference.objects.filter(
                user=request.user,
                page_key=page_key,
                table_key=table_key,
            ).delete()
            return JsonResponse({'ok': True, 'reset': True})

        density = payload.get('density', UserTablePreference.DENSITY_COMFORTABLE)
        if density not in dict(UserTablePreference.DENSITY_CHOICES):
            return JsonResponse({'ok': False, 'error': 'Choose a valid table density.'}, status=400)
        try:
            page_size = int(payload.get('page_size', 50))
        except (TypeError, ValueError):
            page_size = 0
        if page_size not in TABLE_PAGE_SIZES:
            return JsonResponse({'ok': False, 'error': 'Choose 25, 50, 100, or 200 rows.'}, status=400)

        hidden_columns = payload.get('hidden_columns', [])
        if not isinstance(hidden_columns, list):
            return JsonResponse({'ok': False, 'error': 'Invalid hidden-column list.'}, status=400)
        hidden_columns = [
            str(value).strip()[:100]
            for value in hidden_columns[:50]
            if str(value).strip()
        ]
        preference, _ = UserTablePreference.objects.update_or_create(
            user=request.user,
            page_key=page_key,
            table_key=table_key,
            defaults={
                'density': density,
                'page_size': page_size,
                'hidden_columns': hidden_columns,
            },
        )
        return JsonResponse({'ok': True, 'preference': self._serialize(preference)})


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
            'description': product.description or '',
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


# Quantity-bearing ProductLot rows are authoritative once a product adopts lot
# tracking. ProductExpiryDate remains as a legacy compatibility layer only.
def _positive_expiry_lot_rows(product):
    grouped = defaultdict(int)
    for lot in product.lots.all():
        if (
            lot.archived_at is None
            and lot.quantity_on_hand > 0
            and lot.expiry_date is not None
        ):
            grouped[lot.expiry_date] += lot.quantity_on_hand
    if grouped:
        return [
            {'date': expiry, 'quantity': quantity}
            for expiry, quantity in sorted(grouped.items())
        ]
    if product.expiry_date and product.quantity_in_stock > 0:
        return [{
            'date': product.expiry_date,
            'quantity': product.quantity_in_stock,
            'legacy': True,
        }]
    return []


def _expiry_bounds(date_filter, date_from='', date_to=''):
    today = date.today()
    if date_filter == 'custom' and (date_from or date_to):
        try:
            return (
                date.fromisoformat(date_from) if date_from else None,
                date.fromisoformat(date_to) if date_to else None,
            )
        except (ValueError, TypeError):
            return None, today - timedelta(days=1)
    periods = {
        '1_week': timedelta(weeks=1),
        '2_weeks': timedelta(weeks=2),
        '1_month': relativedelta(months=1),
        '2_months': relativedelta(months=2),
        '3_months': relativedelta(months=3),
    }
    if date_filter in periods:
        return today, today + periods[date_filter]
    return None, today - timedelta(days=1)


def _date_in_expiry_window(value, lower, upper):
    return ((lower is None or value >= lower) and
            (upper is None or value <= upper))


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
        product = (Product.objects.filter(pk=pid).select_related('category').prefetch_related('lots').first()
                   if pid else None)

        # Per-product expiry breakdown for the log-mode detail card.
        product_extra = self._product_expiry_summary(product) if product else None

        lower, upper = _expiry_bounds(date_filter, date_from, date_to)
        total_units = 0
        value_at_risk = Decimal('0.00')
        total_expired_units = 0
        for listed_product in products:
            matching_rows = [
                row for row in _positive_expiry_lot_rows(listed_product)
                if _date_in_expiry_window(row['date'], lower, upper)
            ]
            listed_product.expiry_lot_rows = matching_rows
            listed_product.at_risk_units = sum(
                row['quantity'] for row in matching_rows
            )
            listed_product.at_risk_value = (
                (listed_product.price or Decimal('0.00'))
                * listed_product.at_risk_units
            )
            total_units += listed_product.at_risk_units
            value_at_risk += listed_product.at_risk_value
            total_expired_units += listed_product.stock_expired

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
            "product_count": len(products),
            "total_units_on_shelf": total_units,
            "value_at_risk": value_at_risk,
            "total_expired_units": total_expired_units,
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
        lots = _positive_expiry_lot_rows(product)

        def classify(d):
            delta = (d - today).days
            if delta < 0:
                return 'expired', delta
            if delta <= 30:
                return 'soon', delta
            return 'ok', delta

        lot_rows = []
        for lot in lots:
            d = lot['date']
            status, delta = classify(d)
            lot_rows.append({
                'date': d,
                'quantity': lot['quantity'],
                'days': delta,
                'days_abs': abs(delta),
                'status': status,
            })

        expired_quantity = sum(
            row['quantity'] for row in lot_rows if row['status'] == 'expired'
        )
        soon_quantity = sum(
            row['quantity'] for row in lot_rows if row['status'] == 'soon'
        )
        at_risk_quantity = expired_quantity + soon_quantity
        value = (product.price or Decimal('0.00')) * at_risk_quantity
        return {
            'lots': lot_rows,
            'status': lot_rows[0]['status'] if lot_rows else 'none',
            'days': lot_rows[0]['days'] if lot_rows else None,
            'days_abs': lot_rows[0]['days_abs'] if lot_rows else None,
            'value': value,
            'expired_quantity': expired_quantity,
            'at_risk_quantity': at_risk_quantity,
        }

    def post(self, request):
        barcode = request.POST.get("barcode", "").strip()
        product = None
        # Set on a successful retire so the redirect can trigger the "what to do
        # next" pop-out (instead of a toast) on the rebuilt page.
        retired_qty = 0
        mis_scan = False

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

            expiry_summary = self._product_expiry_summary(product)
            expired_available = expiry_summary['expired_quantity']

            if qty <= 0:
                messages.error(request, "Quantity must be greater than 0.")
            elif expired_available <= 0:
                messages.error(
                    request,
                    "No quantity-bearing lot for this product is currently expired.",
                )
            elif qty > expired_available:
                messages.error(
                    request,
                    f"Cannot retire {qty} units. Only {expired_available} expired "
                    "unit(s) are on shelf.",
                )
            else:
                # ✅ FIXED: Wrap in transaction with row locking
                with transaction.atomic():
                    # ✅ Lock the product row
                    product = Product.objects.select_for_update().get(pk=product.pk)
                    
                    # Double-check stock hasn't changed
                    locked_summary = self._product_expiry_summary(product)
                    locked_expired_available = locked_summary['expired_quantity']
                    if qty > locked_expired_available:
                        messages.error(
                            request,
                            "Expired lot quantities changed. Please try again.",
                        )
                    else:
                        # Update stock
                        product.quantity_in_stock -= qty
                        product.save(update_fields=["quantity_in_stock"])

                        # Log the change
                        stock_change = record_stock_change(
                            product,
                            qty=qty,
                            change_type="expired",
                            note="Marked as expired from expired product view",
                            user=request.user,
                        )
                        remove_stock_from_lots(product, qty, stock_change)
                        UserAction.objects.create(user=request.user, action='retire_expired',
                            target=product.name, detail=f'{qty} units retired')

                        # Success + the "what to do next" instructions are shown
                        # as a pop-out on the rebuilt page (not a toast) — flag it
                        # via the redirect below.
                        retired_qty = qty
                        mis_scan = False

        # Post/Redirect/Get: bounce back to the GET handler so the page is
        # rebuilt with the full context — including a fresh `expired_logs`
        # query, so the pull-out Expired Log reflects what was just retired.
        # Also avoids re-submitting the retire on refresh. Messages survive
        # the redirect via the messages framework.
        redirect_url = f"{reverse('expired_products')}?mode=log"
        if product:
            redirect_url += f"&pid={product.pk}"
        if retired_qty:
            redirect_url += f"&retired=1&rq={retired_qty}"
            if mis_scan:
                redirect_url += "&warn=1"
        return redirect(redirect_url)

    ALLOWED_SORTS = {"expiry_date", "-expiry_date", "name", "-name", "barcode", "-barcode", "category__name", "-category__name"}

    def _filter_products(self, date_filter, name_query, sort="expiry_date", date_from=None, date_to=None):
        lower, upper = _expiry_bounds(date_filter, date_from, date_to)
        positive_dated_lots = ProductLot.objects.filter(
            product_id=OuterRef('pk'),
            archived_at__isnull=True,
            quantity_on_hand__gt=0,
            expiry_date__isnull=False,
        )
        qs = Product.objects.filter(quantity_in_stock__gt=0).annotate(
            has_positive_dated_lot=Exists(positive_dated_lots),
        )

        lot_window = Q(
            lots__archived_at__isnull=True,
            lots__quantity_on_hand__gt=0,
            lots__expiry_date__isnull=False,
        )
        legacy_window = Q(has_positive_dated_lot=False)
        if lower is not None:
            lot_window &= Q(lots__expiry_date__gte=lower)
            legacy_window &= Q(expiry_date__gte=lower)
        if upper is not None:
            lot_window &= Q(lots__expiry_date__lte=upper)
            legacy_window &= Q(expiry_date__lte=upper)
        qs = qs.filter(lot_window | legacy_window)

        if name_query:
            qs = qs.filter(name__icontains=name_query)

        order_field = sort if sort in self.ALLOWED_SORTS else "expiry_date"
        return list(
            qs.select_related('category').prefetch_related('lots')
            .distinct().order_by(order_field)
        )


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

        products = ExpiredProductView()._filter_products(
            date_filter, '', sort,
        )
        lower, upper = _expiry_bounds(date_filter)
        total_units = 0
        value_at_risk = Decimal('0.00')
        for product in products:
            matching_rows = [
                row for row in _positive_expiry_lot_rows(product)
                if _date_in_expiry_window(row['date'], lower, upper)
            ]
            product.report_expiry_rows = matching_rows
            product.at_risk_units = sum(
                row['quantity'] for row in matching_rows
            )
            total_units += product.at_risk_units
            value_at_risk += (
                (product.price or Decimal('0.00')) * product.at_risk_units
            )
        product_count = len(products)

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
        kpi_line = f"{product_count} products  ·  {total_units} units at risk  ·  ${value_at_risk:,.2f} value at risk"
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
            ("Qty at Risk", margin + 515, usable - 515 + margin),
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

            expiry_str = (
                p.report_expiry_rows[0]['date'].strftime("%b %d, %Y")
                if p.report_expiry_rows else "N/A"
            )
            cat_name = p.category.name if p.category else "--"
            name_display = p.name[:38] + "..." if len(p.name) > 38 else p.name

            row_data = [
                (expiry_str, cols[0][1]),
                (name_display, cols[1][1]),
                (cat_name, cols[2][1]),
                (str(p.barcode or ""), cols[3][1]),
                (f"${p.price:.2f}", cols[4][1]),
                (str(p.at_risk_units), cols[5][1]),
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
            .filter(
                product__recentlypurchasedproduct__isnull=False,
                product__recentlypurchasedproduct__archived_at__isnull=True,
            )
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
            .filter(archived_at__isnull=True)
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

        preferred_size = preferred_table_page_size(request, 100)
        paginator_low_stock = Paginator(low_stock_products, preferred_size)
        page_obj_low_stock = paginator_low_stock.get_page(request.GET.get('page'))

        # Use the account's saved table size for both full and seamless/AJAX
        # responses so filtering never silently disables pagination.
        paginator_recent = Paginator(recently_purchased, preferred_size)
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
            pager_html = render_to_string(
                'partials/rp_pager.html',
                {
                    'page_obj_recent': page_obj_recent,
                    'q': q,
                    'category_filter': category_filter,
                    'sort': sort_col,
                    'dir': sort_dir,
                    'hide_snacks': hide_snacks,
                },
                request=request,
            )
            return JsonResponse({
                'html': rows_html,
                'pager_html': pager_html,
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
            'category_filter':    category_filter,
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


class ExportRecentlyPurchasedCSVView(AdminRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        # Create the HttpResponse object with the appropriate CSV header.
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="recently_purchased_{now().strftime("%Y%m%d_%H%M")}.csv"'

        writer = csv.writer(response)
        # Write Header Row
        writer.writerow(['Product Name', 'Barcode', 'Item Number', 'Brand', 'Units Bought', 'Current Stock Level'])

        # Fetch all items
        items = RecentlyPurchasedProduct.objects.filter(
            archived_at__isnull=True,
        ).select_related('product')

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


# ── McKesson PharmaClik ordering (drives mckesson_order.py) ──────────────────
# 'paused' is still an active run (process alive, waiting). 'cancelled' is terminal.
MCKESSON_ACTIVE_STATES = ('starting', 'login', 'waiting_user', 'running', 'paused', 'review')


def _order_process_creationflags():
    """Hide a directly launched worker console on supported Windows callers."""
    if os.name != 'nt':
        return 0
    return getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)


def _launch_order_process(cmd, base, log_path):
    """Launch a supplier-ordering worker without inheriting server stdin."""
    with open(log_path, 'a', encoding='utf-8') as logf:
        return subprocess.Popen(
            cmd,
            cwd=str(base),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            creationflags=_order_process_creationflags(),
            close_fds=True,
        )


def _launch_or_schedule_order_process(run, cmd, base, log_path):
    """Use Task Scheduler for every Windows web-originated browser worker.

    A Job Object membership probe is not a reliable safety gate: Waitress can
    report that it is outside a job while a child Playwright process is still
    denied Windows pipe/process creation. The interactive Scheduled Task is the
    one supported Windows boundary. Non-Windows development remains direct.
    """
    if os.name == 'nt':
        from app.supplier_orders import queue_scheduled_supplier_launch

        queue_scheduled_supplier_launch(run, base)
        return None
    return _launch_order_process(cmd, base, log_path)


def _pid_alive(pid):
    """True if the given Windows process id is still running."""
    if not pid:
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def _supplier_run_status(vendor, plan_id=None):
    """Current database-backed run status, including stale-process detection."""
    from app.supplier_orders import serialize_run

    try:
        plan_id = int(plan_id) if plan_id not in (None, '') else None
    except (TypeError, ValueError):
        plan_id = None
    runs = SupplierOrderRun.objects.filter(vendor=vendor)
    if plan_id:
        run = runs.filter(plan_id=plan_id).order_by('-created_at').first()
    else:
        run = runs.filter(state__in=MCKESSON_ACTIVE_STATES).order_by('-created_at').first()
        if run is None:
            run = runs.order_by('-created_at').first()
    if run is None:
        return {'state': 'idle', 'added': [], 'skipped': []}

    if run.state in MCKESSON_ACTIVE_STATES:
        age = (now() - run.updated_at).total_seconds()
        from app.supplier_orders import (
            SCHEDULED_LAUNCH_START_TIMEOUT,
            scheduled_launcher_repair_hint,
        )

        startup_timeout = SCHEDULED_LAUNCH_START_TIMEOUT.total_seconds()
        alive = _pid_alive(run.process_id) if run.process_id else age < startup_timeout
        if not alive:
            run.state = SupplierOrderRun.STATE_ERROR
            if run.process_id:
                run.message = 'The supplier browser ended unexpectedly.'
            elif os.name == 'nt':
                run.message = (
                    'The Windows supplier launcher did not acknowledge this '
                    f'request within {int(startup_timeout)} seconds. '
                    f'{scheduled_launcher_repair_hint()}'
                )
            else:
                run.message = 'The supplier worker did not start.'
            run.completed_at = now()
            run.save(update_fields=['state', 'message', 'completed_at', 'updated_at'])
    return serialize_run(run)


def _mckesson_status(plan_id=None):
    return _supplier_run_status(SupplierOrderRun.VENDOR_MCKESSON, plan_id=plan_id)


def _clean_supplier_items(raw_items):
    clean = []
    if not isinstance(raw_items, list):
        return clean
    product_ids = set(Product.objects.values_list('product_id', flat=True))
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        barcode = str(item.get('barcode') or '').strip()[:64]
        try:
            quantity = int(item.get('quantity'))
        except (TypeError, ValueError):
            continue
        if not barcode or quantity < 1:
            continue
        try:
            product_id = int(item.get('product_id'))
        except (TypeError, ValueError):
            product_id = None
        clean.append({
            'product_id': product_id if product_id in product_ids else None,
            'name': str(item.get('name') or '')[:200],
            'barcode': barcode,
            'quantity': quantity,
        })
    return clean


def _start_supplier_run(request, vendor, script_name):
    current = _supplier_run_status(vendor)
    if current.get('state') in MCKESSON_ACTIVE_STATES:
        return JsonResponse({
            'ok': False,
            'error': f"A {dict(SupplierOrderRun.VENDOR_CHOICES)[vendor]} ordering run is already in progress.",
            'run_id': current.get('run_id'),
            'plan_id': current.get('plan_id'),
        }, status=409)

    try:
        body = json.loads(request.body or '{}')
    except ValueError:
        body = {}
    plan = None
    plan_id = body.get('plan_id')
    if plan_id:
        plan = SupplierOrderPlan.objects.filter(
            pk=plan_id, created_by=request.user,
            status__in=[SupplierOrderPlan.STATUS_PLANNED, SupplierOrderPlan.STATUS_RUNNING],
        ).first()
        if plan is None:
            return JsonResponse({'ok': False, 'error': 'The saved ordering plan is no longer active.'}, status=409)

    clean = _clean_supplier_items(body.get('items'))
    skipped = []
    if not clean:
        try:
            exclude_ids = [int(value) for value in body.get('exclude_category_ids', [])]
        except (TypeError, ValueError):
            return JsonResponse({'ok': False, 'error': 'Invalid category ids.'}, status=400)
        from app.mckesson import collect_order_items
        collected, skipped = collect_order_items(exclude_category_ids=exclude_ids)
        clean = _clean_supplier_items(collected)
    if not clean:
        return JsonResponse({'ok': False, 'error': 'No valid items to order.'}, status=400)

    sequence_position = 0
    if plan and vendor in plan.vendor_sequence:
        sequence_position = plan.vendor_sequence.index(vendor)
    try:
        with transaction.atomic():
            run = SupplierOrderRun.objects.create(
                plan=plan, created_by=request.user, vendor=vendor,
                source=SupplierOrderRun.SOURCE_WEB,
                sequence_position=sequence_position, total=len(clean),
                state=SupplierOrderRun.STATE_STARTING, message='Starting...',
            )
            rows = [
                SupplierOrderRunItem(
                    run=run, product_id=item['product_id'], product_name=item['name'],
                    barcode=item['barcode'], quantity_requested=item['quantity'], position=position,
                )
                for position, item in enumerate(clean)
            ]
            offset = len(rows)
            for position, item in enumerate(skipped, start=offset):
                rows.append(SupplierOrderRunItem(
                    run=run,
                    product_id=item.get('product_id') if item.get('product_id') in
                        set(Product.objects.values_list('product_id', flat=True)) else None,
                    product_name=str(item.get('name') or '')[:200],
                    barcode=str(item.get('barcode') or '')[:64],
                    quantity_requested=max(1, int(item.get('quantity') or 1)),
                    position=position, outcome=SupplierOrderRunItem.OUTCOME_SKIPPED,
                    reason=str(item.get('reason') or '')[:500], processed_at=now(),
                ))
            SupplierOrderRunItem.objects.bulk_create(rows)
            if plan:
                plan.status = SupplierOrderPlan.STATUS_RUNNING
                if plan.started_at is None:
                    plan.started_at = now()
                plan.save(update_fields=['status', 'started_at'])
    except IntegrityError:
        existing = SupplierOrderRun.objects.filter(plan=plan, vendor=vendor).first()
        return JsonResponse({
            'ok': False, 'error': 'This supplier step has already started.',
            'run_id': existing.pk if existing else None,
        }, status=409)

    base = Path(settings.BASE_DIR)
    python = base / 'env' / 'Scripts' / 'python.exe'
    script = base / script_name
    if not python.exists() or not script.exists():
        run.state = SupplierOrderRun.STATE_ERROR
        run.message = f'{script_name} or the application environment was not found.'
        run.completed_at = now()
        run.save(update_fields=['state', 'message', 'completed_at', 'updated_at'])
        return JsonResponse({'ok': False, 'error': run.message}, status=500)

    logs_dir = base / 'logs'
    logs_dir.mkdir(exist_ok=True)
    try:
        process = _launch_or_schedule_order_process(
            run,
            [str(python), str(script), '--no-input', '--run-id', str(run.pk)],
            base, logs_dir / f'{vendor}_order.log',
        )
        if process is not None:
            run.process_id = process.pid
            run.save(update_fields=['process_id', 'updated_at'])
    except OSError as exc:
        run.state = SupplierOrderRun.STATE_ERROR
        run.message = f'Could not start supplier ordering: {exc}'
        run.completed_at = now()
        run.save(update_fields=['state', 'message', 'completed_at', 'updated_at'])
        return JsonResponse({'ok': False, 'error': run.message}, status=500)
    return JsonResponse({
        'ok': True,
        'run_id': run.pk,
        'plan_id': run.plan_id,
        'launch_mode': 'scheduled' if process is None else 'direct',
    })


class McKessonOrderStartView(AdminRequiredMixin, View):
    def post(self, request):
        return _start_supplier_run(
            request, SupplierOrderRun.VENDOR_MCKESSON, 'mckesson_order.py',
        )


class McKessonOrderStatusView(AdminRequiredMixin, View):
    def get(self, request):
        return JsonResponse(_mckesson_status(request.GET.get('plan_id')))


class McKessonOrderPreviewView(AdminRequiredMixin, View):
    """What WOULD be ordered with the chosen filters — no browser, DB only."""

    def post(self, request):
        try:
            body = json.loads(request.body or '{}')
        except ValueError:
            body = {}
        try:
            exclude_ids = [int(x) for x in body.get('exclude_category_ids', [])]
        except (TypeError, ValueError):
            return JsonResponse({'ok': False, 'error': 'Invalid category ids.'}, status=400)

        from app.mckesson import collect_order_items
        items, skipped = collect_order_items(exclude_category_ids=exclude_ids)

        # Stock movement over the last 120 days per product: units in
        # (check-ins/corrections) vs units out (sold/expired/removed).
        pids = [r['product_id'] for r in items + skipped if r.get('product_id')]
        plus_map, minus_map = {}, {}
        if pids:
            PLUS_TYPES = {'checkin', 'error_add'}
            MINUS_TYPES = {'checkout', 'expired', 'error_subtract', 'deletion'}
            since = now() - timedelta(days=120)
            for r in (StockChange.objects
                      .filter(product_id__in=pids, timestamp__gte=since,
                              change_type__in=PLUS_TYPES | MINUS_TYPES)
                      .values('product_id', 'change_type')
                      .annotate(total=Sum('quantity'))):
                target = plus_map if r['change_type'] in PLUS_TYPES else minus_map
                target[r['product_id']] = target.get(r['product_id'], 0) + (r['total'] or 0)
        for r in items + skipped:
            pid = r.get('product_id')
            r['plus_120'] = plus_map.get(pid, 0)
            r['minus_120'] = minus_map.get(pid, 0)

        return JsonResponse({'ok': True, 'items': items, 'skipped': skipped})


# ── Kohl & Frisch (KFConnect) ordering (drives kohlfrisch_order.py) ──────────
# Same Recently-Purchased data and preview as McKesson — only the script that
# fills the vendor cart differs, so the preview endpoint is reused as-is.
KOHLFRISCH_ACTIVE_STATES = MCKESSON_ACTIVE_STATES


def _kohlfrisch_status(plan_id=None):
    return _supplier_run_status(SupplierOrderRun.VENDOR_KOHLFRISCH, plan_id=plan_id)


class KohlFrischOrderStartView(AdminRequiredMixin, View):
    def post(self, request):
        return _start_supplier_run(
            request, SupplierOrderRun.VENDOR_KOHLFRISCH, 'kohlfrisch_order.py',
        )


class KohlFrischOrderStatusView(AdminRequiredMixin, View):
    def get(self, request):
        return JsonResponse(_kohlfrisch_status(request.GET.get('plan_id')))


class OrderControlView(AdminRequiredMixin, View):
    """Save pause/resume/cancel controls on the active database run."""

    def post(self, request):
        try:
            body = json.loads(request.body or '{}')
        except ValueError:
            body = {}
        vendor = body.get('vendor')
        action = body.get('action')
        if vendor not in ('mck', 'kf') or action not in ('pause', 'resume', 'cancel'):
            return JsonResponse({'ok': False, 'error': 'Invalid control request.'}, status=400)

        runs = SupplierOrderRun.objects.filter(
            vendor=vendor, state__in=MCKESSON_ACTIVE_STATES,
        )
        plan_id = body.get('plan_id')
        if plan_id:
            runs = runs.filter(plan_id=plan_id)
        run = runs.order_by('-created_at').first()
        if run is None:
            return JsonResponse({'ok': False, 'error': 'No active supplier run was found.'}, status=404)
        if action == 'pause':
            run.pause_requested = True
        elif action == 'resume':
            run.pause_requested = False
        elif action == 'cancel':
            run.cancel_requested = True
            run.pause_requested = False
        run.save(update_fields=['pause_requested', 'cancel_requested', 'updated_at'])
        return JsonResponse({'ok': True, 'run_id': run.pk})


class SupplierOrderPlanView(AdminRequiredMixin, View):
    """Create, resume, and finish durable multi-supplier handoff plans."""

    @staticmethod
    def _serialize(plan):
        return {
            'id': plan.pk,
            'status': plan.status,
            'seq': plan.vendor_sequence,
            'items': [
                {
                    'product_id': item.product_id,
                    'name': item.product_name,
                    'barcode': item.barcode,
                    'quantity': item.quantity,
                }
                for item in plan.items.all()
            ],
            'created_at': plan.created_at.isoformat(),
            'started_at': plan.started_at.isoformat() if plan.started_at else None,
        }

    def get(self, request):
        plan = SupplierOrderPlan.objects.filter(
            created_by=request.user,
            status__in=[SupplierOrderPlan.STATUS_PLANNED, SupplierOrderPlan.STATUS_RUNNING],
        ).prefetch_related('items').order_by('-created_at').first()
        return JsonResponse({'ok': True, 'plan': self._serialize(plan) if plan else None})

    def post(self, request):
        try:
            body = json.loads(request.body or '{}')
        except ValueError:
            return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)
        action = body.get('action', 'create')

        if action == 'finish':
            plan = SupplierOrderPlan.objects.filter(
                pk=body.get('plan_id'), created_by=request.user,
                status__in=[SupplierOrderPlan.STATUS_PLANNED, SupplierOrderPlan.STATUS_RUNNING],
            ).first()
            if plan is None:
                return JsonResponse({'ok': True})
            if body.get('cancelled'):
                plan.status = SupplierOrderPlan.STATUS_CANCELLED
            elif plan.runs.filter(state=SupplierOrderRun.STATE_ERROR).exists():
                plan.status = SupplierOrderPlan.STATUS_ERROR
            else:
                plan.status = SupplierOrderPlan.STATUS_COMPLETED
            plan.completed_at = now()
            plan.save(update_fields=['status', 'completed_at'])
            return JsonResponse({'ok': True, 'status': plan.status})

        sequence = body.get('seq') if isinstance(body.get('seq'), list) else []
        sequence = [vendor for vendor in ['mck', 'kf'] if vendor in sequence]
        if not sequence:
            return JsonResponse({'ok': False, 'error': 'Pick at least one distributor.'}, status=400)
        clean = _clean_supplier_items(body.get('items'))
        if not clean:
            return JsonResponse({'ok': False, 'error': 'No valid items to order.'}, status=400)
        existing = SupplierOrderPlan.objects.filter(
            created_by=request.user,
            status__in=[SupplierOrderPlan.STATUS_PLANNED, SupplierOrderPlan.STATUS_RUNNING],
        ).prefetch_related('items').order_by('-created_at').first()
        if existing:
            return JsonResponse({
                'ok': False,
                'error': 'An ordering plan is already in progress. It will resume automatically.',
                'plan': self._serialize(existing),
            }, status=409)

        with transaction.atomic():
            plan = SupplierOrderPlan.objects.create(
                created_by=request.user, vendor_sequence=sequence,
            )
            SupplierOrderPlanItem.objects.bulk_create([
                SupplierOrderPlanItem(
                    plan=plan, product_id=item['product_id'], product_name=item['name'],
                    barcode=item['barcode'], quantity=item['quantity'], position=position,
                )
                for position, item in enumerate(clean)
            ])
        plan = SupplierOrderPlan.objects.prefetch_related('items').get(pk=plan.pk)
        return JsonResponse({'ok': True, 'plan': self._serialize(plan)})


class SupplierPurchaseOrderView(AdminRequiredMixin, View):
    """Track human-confirmed supplier orders independently of browser automation."""

    template_name = 'supplier_purchase_orders.html'

    def get(self, request):
        orders = (
            SupplierPurchaseOrder.objects.filter(archived_at__isnull=True)
            .select_related('plan', 'created_by')
            .prefetch_related('lines')
        )
        return render(request, self.template_name, {
            'orders': orders,
            'plans': SupplierOrderPlan.objects.prefetch_related('items').order_by('-created_at')[:30],
            'supplier_choices': SupplierPurchaseOrder.SUPPLIER_CHOICES,
            'status_choices': SupplierPurchaseOrder.STATUS_CHOICES,
        })

    @staticmethod
    def _header_values(request):
        supplier = request.POST.get('supplier', '')
        if supplier not in dict(SupplierPurchaseOrder.SUPPLIER_CHOICES):
            raise ValidationError('Choose a valid supplier.')
        order_date = parse_date(request.POST.get('order_date', ''))
        expected_raw = request.POST.get('expected_date', '').strip()
        expected_date = parse_date(expected_raw) if expected_raw else None
        if not order_date or (expected_raw and not expected_date):
            raise ValidationError('Enter valid order and expected dates.')
        status = request.POST.get('status', SupplierPurchaseOrder.STATUS_DRAFT)
        if status not in dict(SupplierPurchaseOrder.STATUS_CHOICES):
            raise ValidationError('Choose a valid order status.')
        return {
            'supplier': supplier,
            'supplier_name': request.POST.get('supplier_name', '').strip()[:120],
            'confirmation_number': request.POST.get('confirmation_number', '').strip()[:100],
            'order_date': order_date,
            'expected_date': expected_date,
            'status': status,
            'notes': request.POST.get('notes', '').strip(),
        }

    def post(self, request):
        action = request.POST.get('action', 'create')
        if action == 'archive':
            purchase_order = get_object_or_404(
                SupplierPurchaseOrder,
                pk=request.POST.get('purchase_order_id'), archived_at__isnull=True,
            )
            purchase_order.archived_at = now()
            purchase_order.archived_by = request.user
            purchase_order.save(update_fields=['archived_at', 'archived_by', 'updated_at'])
            UserAction.objects.create(
                user=request.user, action='supplier_order_archive',
                target=str(purchase_order),
            )
            messages.success(request, 'Supplier order moved to Recovery.')
            return redirect('supplier_purchase_orders')

        try:
            values = self._header_values(request)
        except ValidationError as exc:
            messages.error(request, exc.message)
            return redirect('supplier_purchase_orders')

        if action == 'create':
            plan = None
            plan_id = request.POST.get('plan_id', '').strip()
            if plan_id:
                plan = SupplierOrderPlan.objects.filter(pk=plan_id).first()
            with transaction.atomic():
                purchase_order = SupplierPurchaseOrder.objects.create(
                    plan=plan, created_by=request.user, **values,
                )
                if plan:
                    SupplierPurchaseOrderLine.objects.bulk_create([
                        SupplierPurchaseOrderLine(
                            purchase_order=purchase_order,
                            product=item.product,
                            product_name=item.product_name,
                            product_barcode=item.barcode,
                            quantity_ordered=item.quantity,
                        )
                        for item in plan.items.all()
                    ])
            UserAction.objects.create(
                user=request.user, action='supplier_order_create',
                target=str(purchase_order),
                detail=f'Status: {purchase_order.get_status_display()}',
            )
            messages.success(
                request,
                f'Supplier order {purchase_order.confirmation_number or purchase_order.pk} is now tracked.',
            )
            return redirect('supplier_purchase_orders')

        purchase_order = get_object_or_404(
            SupplierPurchaseOrder,
            pk=request.POST.get('purchase_order_id'), archived_at__isnull=True,
        )
        with transaction.atomic():
            purchase_order = SupplierPurchaseOrder.objects.select_for_update().get(pk=purchase_order.pk)
            for field, value in values.items():
                setattr(purchase_order, field, value)
            received_total = 0
            ordered_total = 0
            for line in purchase_order.lines.select_for_update():
                try:
                    received = int(request.POST.get(f'received_{line.pk}', line.quantity_received))
                except (TypeError, ValueError):
                    messages.error(request, f'{line.product_name}: received quantity must be a whole number.')
                    transaction.set_rollback(True)
                    return redirect('supplier_purchase_orders')
                if received < 0 or received > line.quantity_ordered:
                    messages.error(request, f'{line.product_name}: received cannot exceed ordered.')
                    transaction.set_rollback(True)
                    return redirect('supplier_purchase_orders')
                line.quantity_received = received
                line.save(update_fields=['quantity_received'])
                received_total += received
                ordered_total += line.quantity_ordered
            if purchase_order.status != SupplierPurchaseOrder.STATUS_CANCELLED:
                if ordered_total and received_total == ordered_total:
                    purchase_order.status = SupplierPurchaseOrder.STATUS_RECEIVED
                elif received_total:
                    purchase_order.status = SupplierPurchaseOrder.STATUS_PARTIAL
                elif ordered_total and purchase_order.status in {
                    SupplierPurchaseOrder.STATUS_PARTIAL,
                    SupplierPurchaseOrder.STATUS_RECEIVED,
                }:
                    purchase_order.status = SupplierPurchaseOrder.STATUS_SUBMITTED
            purchase_order.save()
        UserAction.objects.create(
            user=request.user, action='supplier_order_update',
            target=str(purchase_order),
            detail=f'Status: {purchase_order.get_status_display()}',
        )
        messages.success(request, 'Supplier order tracking updated.')
        return redirect('supplier_purchase_orders')


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
        'delete_product': ['delete_product', 'archive_product'],
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
        'supplier_orders': [
            'supplier_order_create', 'supplier_order_update',
            'supplier_order_archive', 'restore_archived_record',
        ],
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

        paginator = Paginator(events, preferred_table_page_size(request, 50))
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
class DeleteRecentlyPurchasedProductView(AdminRequiredMixin, View):
   def post(self, request, id):
       is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
       try:
           recently_purchased = RecentlyPurchasedProduct.objects.get(
               id=id, archived_at__isnull=True,
           )
           product_name = recently_purchased.product.name if recently_purchased.product else "Unknown"
           recently_purchased.archived_at = now()
           recently_purchased.archived_by = request.user
           recently_purchased.archive_reason = 'Removed from Recently Purchased'
           recently_purchased.save(update_fields=['archived_at', 'archived_by', 'archive_reason'])
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


class DeleteAllRecentlyPurchasedView(AdminRequiredMixin, View):
   def post(self, request):
       is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
       active = RecentlyPurchasedProduct.objects.filter(archived_at__isnull=True)
       count = active.count()
       active.update(
           archived_at=now(), archived_by=request.user,
           archive_reason='Cleared from Recently Purchased',
       )
       UserAction.objects.create(
           user=request.user, action='delete_all_recently_purchased',
           target=f'{count} items',
       )
       if is_ajax:
           return JsonResponse({'success': True})
       messages.success(request, "All recently purchased products have been deleted.")
       return redirect('low_stock')


class BulkDeleteRecentlyPurchasedView(AdminRequiredMixin, View):
   def post(self, request):
       try:
           data = json.loads(request.body)
           ids = data.get('ids', [])
           if not ids:
               return JsonResponse({'success': False, 'error': 'No IDs provided'}, status=400)
           qs = RecentlyPurchasedProduct.objects.filter(
               id__in=ids, archived_at__isnull=True,
           )
           deleted_count = qs.count()
           qs.update(
               archived_at=now(), archived_by=request.user,
               archive_reason='Bulk removed from Recently Purchased',
           )
           UserAction.objects.create(
               user=request.user, action='bulk_delete_recently_purchased',
               target=f'{deleted_count} items',
           )
           return JsonResponse({'success': True, 'deleted_count': deleted_count})
       except (json.JSONDecodeError, Exception) as e:
           return JsonResponse({'success': False, 'error': str(e)}, status=400)


class DeleteByCategoryRecentlyPurchasedView(AdminRequiredMixin, View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            category_id = data.get('category_id')
            if not category_id:
                return JsonResponse({'success': False, 'error': 'No category ID'}, status=400)
            qs = RecentlyPurchasedProduct.objects.filter(
                product__category_id=category_id, archived_at__isnull=True,
            )
            deleted_count = qs.count()
            qs.update(
                archived_at=now(), archived_by=request.user,
                archive_reason='Category removed from Recently Purchased',
            )
            UserAction.objects.create(
                user=request.user, action='bulk_delete_recently_purchased',
                target=f'{deleted_count} items (by category)',
            )
            return JsonResponse({'success': True, 'deleted_count': deleted_count})
        except (json.JSONDecodeError, Exception) as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)


class DeleteOlderThanRecentlyPurchasedView(AdminRequiredMixin, View):
    ALLOWED_DAYS = {30, 60, 90}

    def post(self, request):
        try:
            data = json.loads(request.body)
            days = data.get('days')
            if days not in self.ALLOWED_DAYS:
                return JsonResponse({'success': False, 'error': 'Invalid days value'}, status=400)
            cutoff = now() - timedelta(days=days)
            qs = RecentlyPurchasedProduct.objects.filter(
                order_date__lt=cutoff, archived_at__isnull=True,
            )
            deleted_count = qs.count()
            qs.update(
                archived_at=now(), archived_by=request.user,
                archive_reason=f'Older than {days} days',
            )
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
    if not has_admin_access(request):
        target = reverse('passkey_unlock')
        return redirect(
            f"{target}?{urlencode({'next': request.get_full_path()})}"
        )
    
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
        
        # Keep the complete product and lot record recoverable. Product.objects
        # hides archived rows from operational pages; Product.all_objects is
        # used only by Recovery and explicit integrity work.
        product.status_before_archive = product.status
        product.status = False
        product.archived_at = now()
        product.archived_by = request.user
        product.archive_reason = (
            request.POST.get('archive_reason', '').strip()[:255]
            or 'Removed from inventory'
        )
        product.save(update_fields=[
            'status_before_archive', 'status', 'archived_at', 'archived_by',
            'archive_reason',
        ])
        RecentlyPurchasedProduct.objects.filter(
            product=product, archived_at__isnull=True,
        ).update(
            archived_at=product.archived_at,
            archived_by=request.user,
            archive_reason='Product moved to Recovery',
        )

    UserAction.objects.create(
        user=request.user, action='archive_product',
        target=product_name,
        detail=f"Had {remaining_stock} units remaining",
    )
    messages.success(
        request,
        f"Product '{product_name}' was moved to Recovery and can be restored.",
    )

    # Redirect back to inventory page with query parameters
    page = request.POST.get('page', 1)
    category_id = request.POST.get('category_id', '')
    search_query = (
        request.POST.get('q')
        or request.POST.get('barcode_query')
        or request.POST.get('name_query')
        or ''
    ).strip()

    redirect_url = f"{reverse('inventory_display')}?page={page}"
    if category_id:
        redirect_url += f"&category_id={category_id}"
    if search_query:
        redirect_url += f"&{urlencode({'q': search_query})}"

    return redirect(redirect_url)

# Delete all orders
class DeleteAllOrdersView(AdminRequiredMixin, View):
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


class ArchiveRecoveryView(AdminRequiredMixin, View):
    """One recoverable-history page for operational records removed from views."""

    template_name = 'archive_recovery.html'

    TYPE_LABELS = {
        'product': 'Products',
        'order': 'Sales',
        'ordering': 'Ordering sheet',
        'delivery': 'Delivery',
        'recent_purchase': 'Recently Purchased',
        'special_order': 'Special orders',
        'supplier_order': 'Supplier orders',
    }

    def _queryset(self, kind, query='', date_from=None, date_to=None):
        if kind == 'product':
            qs = Product.all_objects.filter(archived_at__isnull=False).select_related('archived_by')
            date_field = 'archived_at__date'
            if query:
                qs = qs.filter(
                    Q(name__icontains=query) | Q(brand__icontains=query)
                    | Q(item_number__icontains=query) | Q(barcode__icontains=query)
                    | Q(archive_reason__icontains=query)
                )
            order_field = '-archived_at'
        elif kind == 'order':
            qs = Order.objects.filter(is_deleted=True).select_related('deleted_by', 'user')
            date_field = 'deleted_at__date'
            if query:
                lookup = (
                    Q(user__username__icontains=query) | Q(deleted_by__username__icontains=query)
                    | Q(details__product_name__icontains=query)
                    | Q(details__product_barcode__icontains=query)
                )
                if query.isdigit():
                    lookup |= Q(pk=int(query))
                qs = qs.filter(lookup).distinct()
            order_field = '-deleted_at'
        elif kind == 'ordering':
            qs = OrderingSheetEntry.objects.filter(is_deleted=True).select_related('deleted_by')
            date_field = 'deleted_at__date'
            if query:
                qs = qs.filter(
                    Q(name__icontains=query) | Q(patient_name__icontains=query)
                    | Q(initials__icontains=query) | Q(supplier_name__icontains=query)
                    | Q(order_note__icontains=query) | Q(phone_number__icontains=query)
                )
            order_field = '-deleted_at'
        elif kind == 'delivery':
            qs = DeliveryCheckIn.objects.filter(archived_at__isnull=False).select_related('archived_by')
            date_field = 'archived_at__date'
            if query:
                qs = qs.filter(
                    Q(first_name__icontains=query) | Q(last_name__icontains=query)
                    | Q(barcode__icontains=query) | Q(comment__icontains=query)
                    | Q(archive_reason__icontains=query)
                )
            order_field = '-archived_at'
        elif kind == 'recent_purchase':
            qs = RecentlyPurchasedProduct.objects.filter(
                archived_at__isnull=False,
            ).select_related('product', 'archived_by')
            date_field = 'archived_at__date'
            if query:
                qs = qs.filter(
                    Q(product__name__icontains=query) | Q(product__brand__icontains=query)
                    | Q(product__barcode__icontains=query) | Q(product__item_number__icontains=query)
                    | Q(archive_reason__icontains=query)
                )
            order_field = '-archived_at'
        elif kind == 'special_order':
            qs = Item.objects.filter(
                archived_at__isnull=False,
            ).select_related('archived_by')
            date_field = 'archived_at__date'
            if query:
                qs = qs.filter(
                    Q(first_name__icontains=query) | Q(last_name__icontains=query)
                    | Q(item_name__icontains=query) | Q(item_number__icontains=query)
                    | Q(phone_number__icontains=query) | Q(archive_reason__icontains=query)
                )
            order_field = '-archived_at'
        elif kind == 'supplier_order':
            qs = SupplierPurchaseOrder.objects.filter(
                archived_at__isnull=False,
            ).select_related('archived_by')
            date_field = 'archived_at__date'
            if query:
                qs = qs.filter(
                    Q(supplier_name__icontains=query) | Q(confirmation_number__icontains=query)
                    | Q(notes__icontains=query) | Q(lines__product_name__icontains=query)
                    | Q(lines__product_barcode__icontains=query)
                ).distinct()
            order_field = '-archived_at'
        else:
            raise ValueError('Unknown recovery record type.')

        if date_from:
            qs = qs.filter(**{f'{date_field}__gte': date_from})
        if date_to:
            qs = qs.filter(**{f'{date_field}__lte': date_to})
        return qs.order_by(order_field)

    @staticmethod
    def _username(user):
        return user.get_username() if user else ''

    def _row(self, kind, obj):
        if kind == 'product':
            identity = obj.barcode or obj.item_number or f'Product #{obj.pk}'
            return {
                'kind': kind, 'object_id': obj.pk, 'type_label': 'Product',
                'title': obj.name, 'reference': identity,
                'detail': f'{obj.quantity_in_stock} units retained',
                'reason': obj.archive_reason or 'Removed from inventory',
                'archived_at': obj.archived_at, 'archived_by': self._username(obj.archived_by),
            }
        if kind == 'order':
            detail = f'Original sale: {localtime(obj.order_date).strftime("%b %d, %Y %H:%M")}'
            if obj.user:
                detail += f' · Created by {self._username(obj.user)}'
            return {
                'kind': kind, 'object_id': obj.pk, 'type_label': 'Sale',
                'title': f'Sale #{obj.pk}', 'reference': f'${obj.total_price:.2f}',
                'detail': detail, 'reason': 'Removed from Transactions',
                'archived_at': obj.deleted_at, 'archived_by': self._username(obj.deleted_by),
            }
        if kind == 'ordering':
            detail_parts = [obj.get_status_display(), obj.get_entry_type_display()]
            if obj.patient_name:
                detail_parts.append(f'Patient: {obj.patient_name}')
            return {
                'kind': kind, 'object_id': obj.pk, 'type_label': 'Ordering sheet',
                'title': obj.name, 'reference': obj.initials,
                'detail': ' · '.join(detail_parts), 'reason': obj.order_note or 'Removed from ordering sheet',
                'archived_at': obj.deleted_at, 'archived_by': self._username(obj.deleted_by),
            }
        if kind == 'delivery':
            state = 'Checked out' if obj.checked_out_at else 'Was on site'
            return {
                'kind': kind, 'object_id': obj.pk, 'type_label': 'Delivery',
                'title': f'{obj.first_name} {obj.last_name}'.strip(), 'reference': obj.barcode,
                'detail': state + (f' · {obj.comment}' if obj.comment else ''),
                'reason': obj.archive_reason or 'Removed from delivery history',
                'archived_at': obj.archived_at, 'archived_by': self._username(obj.archived_by),
            }
        if kind == 'recent_purchase':
            return {
                'kind': kind, 'object_id': obj.pk, 'type_label': 'Recently Purchased',
                'title': obj.product.name, 'reference': obj.product.barcode or f'Product #{obj.product_id}',
                'detail': f'{obj.quantity} purchased',
                'reason': obj.archive_reason or 'Removed from Recently Purchased',
                'archived_at': obj.archived_at, 'archived_by': self._username(obj.archived_by),
            }
        if kind == 'special_order':
            customer = f'{obj.first_name} {obj.last_name}'.strip()
            item_context = f'{obj.get_size_display()} · {obj.get_side_display()}'
            if customer:
                item_context += f' · For {customer}'
            return {
                'kind': kind, 'object_id': obj.pk, 'type_label': 'Special order',
                'title': obj.item_name, 'reference': obj.item_number,
                'detail': item_context,
                'reason': obj.archive_reason or 'Removed from Special Orders',
                'archived_at': obj.archived_at,
                'archived_by': self._username(obj.archived_by),
            }
        return {
            'kind': kind, 'object_id': obj.pk, 'type_label': 'Supplier order',
            'title': obj.display_supplier,
            'reference': obj.confirmation_number or f'Order #{obj.pk}',
            'detail': f'{obj.get_status_display()} · ordered {obj.order_date.strftime("%b %d, %Y")}',
            'reason': obj.notes or 'Removed from supplier orders',
            'archived_at': obj.archived_at, 'archived_by': self._username(obj.archived_by),
        }

    def _redirect_with_filters(self, request):
        allowed = ('type', 'q', 'date_from', 'date_to', 'page')
        query = urlencode({key: request.POST.get(key, '') for key in allowed if request.POST.get(key)})
        url = reverse('archive_recovery')
        return redirect(f'{url}?{query}' if query else url)

    def get(self, request):
        selected_type = request.GET.get('type', 'all')
        if selected_type not in {'all', *self.TYPE_LABELS}:
            selected_type = 'all'
        query = request.GET.get('q', '').strip()[:200]
        date_from_value = request.GET.get('date_from', '')
        date_to_value = request.GET.get('date_to', '')
        date_from = parse_date(date_from_value) if date_from_value else None
        date_to = parse_date(date_to_value) if date_to_value else None

        filtered = {
            kind: self._queryset(kind, query, date_from, date_to)
            for kind in self.TYPE_LABELS
        }
        total_counts = {
            kind: self._queryset(kind).count()
            for kind in self.TYPE_LABELS
        }
        has_filters = bool(query or date_from or date_to)
        filtered_counts = (
            {kind: qs.count() for kind, qs in filtered.items()}
            if has_filters else total_counts.copy()
        )
        type_tabs = [{
            'key': 'all',
            'label': 'All records',
            'count': sum(filtered_counts.values()),
            'query': urlencode({
                key: value for key, value in {
                    'q': query, 'date_from': date_from_value, 'date_to': date_to_value,
                }.items() if value
            }),
        }]
        for kind, label in self.TYPE_LABELS.items():
            type_tabs.append({
                'key': kind,
                'label': label,
                'count': filtered_counts[kind],
                'query': urlencode({
                    key: value for key, value in {
                        'type': kind, 'q': query,
                        'date_from': date_from_value, 'date_to': date_to_value,
                    }.items() if value
                }),
            })

        if selected_type == 'all':
            per_page = preferred_table_page_size(request, 50)
            paginator = Paginator(range(sum(filtered_counts.values())), per_page)
            page_obj = paginator.get_page(request.GET.get('page'))
            upper_bound = page_obj.end_index() if paginator.count else 0
            lower_bound = page_obj.start_index() - 1 if paginator.count else 0
            candidates = [
                self._row(kind, obj)
                for kind, qs in filtered.items()
                for obj in qs[:upper_bound]
            ]
            candidates.sort(
                key=lambda row: row['archived_at'] or datetime.min.replace(tzinfo=now().tzinfo),
                reverse=True,
            )
            page_obj.object_list = candidates[lower_bound:upper_bound]
        else:
            paginator = Paginator(
                filtered[selected_type],
                preferred_table_page_size(request, 50),
            )
            model_page = paginator.get_page(request.GET.get('page'))
            model_page.object_list = [self._row(selected_type, obj) for obj in model_page.object_list]
            page_obj = model_page

        filter_query = urlencode({
            key: value for key, value in {
                'type': selected_type if selected_type != 'all' else '',
                'q': query, 'date_from': date_from_value, 'date_to': date_to_value,
            }.items() if value
        })
        return render(request, self.template_name, {
            'page_obj': page_obj,
            'selected_type': selected_type,
            'query': query,
            'date_from': date_from_value,
            'date_to': date_to_value,
            'type_tabs': type_tabs,
            'total_counts': total_counts,
            'filtered_total': sum(filtered_counts.values()),
            'recovery_total': sum(total_counts.values()),
            'filter_query': filter_query,
            'invalid_date': bool((date_from_value and not date_from) or (date_to_value and not date_to)),
        })

    def post(self, request):
        kind = request.POST.get('kind', '')
        object_id = request.POST.get('object_id', '')
        restored_label = None
        with transaction.atomic():
            if kind == 'product':
                obj = get_object_or_404(
                    Product.all_objects.select_for_update(),
                    pk=object_id, archived_at__isnull=False,
                )
                obj.archived_at = None
                obj.archived_by = None
                obj.archive_reason = ''
                obj.status = obj.status_before_archive
                obj.save(update_fields=[
                    'archived_at', 'archived_by', 'archive_reason', 'status',
                ])
                if obj.quantity_in_stock > 0:
                    record_stock_change(
                        product=obj,
                        qty=obj.quantity_in_stock,
                        change_type='restoration',
                        note='Product restored from Recovery',
                        user=request.user,
                    )
                restored_label = obj.name
            elif kind == 'order':
                obj = get_object_or_404(Order, pk=object_id, is_deleted=True)
                obj.is_deleted = False
                obj.deleted_at = None
                obj.deleted_by = None
                obj.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])
                restored_label = f'Sale #{obj.pk}'
            elif kind == 'ordering':
                obj = get_object_or_404(OrderingSheetEntry, pk=object_id, is_deleted=True)
                obj.is_deleted = False
                obj.deleted_at = None
                obj.deleted_by = None
                obj.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])
                restored_label = obj.name
            elif kind == 'delivery':
                obj = get_object_or_404(DeliveryCheckIn, pk=object_id, archived_at__isnull=False)
                obj.archived_at = None
                obj.archived_by = None
                obj.archive_reason = ''
                obj.save(update_fields=['archived_at', 'archived_by', 'archive_reason'])
                restored_label = f'{obj.first_name} {obj.last_name}'
            elif kind == 'recent_purchase':
                obj = get_object_or_404(RecentlyPurchasedProduct, pk=object_id, archived_at__isnull=False)
                if RecentlyPurchasedProduct.objects.filter(
                    product=obj.product, archived_at__isnull=True,
                ).exists():
                    messages.error(
                        request,
                        f'{obj.product.name} already has an active Recently Purchased row.',
                    )
                    return self._redirect_with_filters(request)
                obj.archived_at = None
                obj.archived_by = None
                obj.archive_reason = ''
                obj.save(update_fields=['archived_at', 'archived_by', 'archive_reason'])
                restored_label = obj.product.name
            elif kind == 'special_order':
                obj = get_object_or_404(
                    Item.objects.select_for_update(),
                    pk=object_id, archived_at__isnull=False,
                )
                obj.archived_at = None
                obj.archived_by = None
                obj.archive_reason = ''
                obj.save(update_fields=['archived_at', 'archived_by', 'archive_reason'])
                restored_label = obj.item_name
            elif kind == 'supplier_order':
                obj = get_object_or_404(SupplierPurchaseOrder, pk=object_id, archived_at__isnull=False)
                obj.archived_at = None
                obj.archived_by = None
                obj.save(update_fields=['archived_at', 'archived_by', 'updated_at'])
                restored_label = str(obj)
            else:
                messages.error(request, 'Unknown recovery record.')
                return self._redirect_with_filters(request)
            UserAction.objects.create(
                user=request.user, action='restore_archived_record',
                target=restored_label, detail=kind,
            )
        messages.success(request, f'Restored {restored_label}.')
        return self._redirect_with_filters(request)


# Item list view
class ItemListView(LoginRequiredMixin,View):
   template_name = 'item_list.html'
   form_class = ItemForm

   def get(self, request):
       form = self.form_class()
       items = Item.objects.filter(archived_at__isnull=True)
       return render(request, self.template_name, {
           'form': form,
           'items': items,
           'can_administer': has_admin_access(request),
       })

   def post(self, request):
       if 'delete' in request.POST:
           if not has_admin_access(request):
               unlock_url = reverse('passkey_unlock')
               return redirect(
                   f"{unlock_url}?{urlencode({'next': request.get_full_path()})}"
               )
           item_id = request.POST.get('item_id')
           with transaction.atomic():
               item = get_object_or_404(
                   Item.objects.select_for_update(),
                   id=item_id, archived_at__isnull=True,
               )
               item_name = item.item_name
               customer = f'{item.first_name} {item.last_name}'.strip()
               item.archived_at = now()
               item.archived_by = request.user
               item.archive_reason = 'Removed from Special Orders'
               item.save(update_fields=[
                   'archived_at', 'archived_by', 'archive_reason',
               ])
               UserAction.objects.create(
                   user=request.user,
                   action='delete_item_list',
                   target=item_name,
                   detail=f'Moved to Recovery · {customer}',
               )
           messages.success(
               request,
               f"Item '{item_name}' was moved to Recovery and can be restored.",
           )
           return redirect('item_list')
       elif 'update_checked' in request.POST:
           item_id = request.POST.get('item_id')
           is_checked = request.POST.get('is_checked') == 'on'
           item = get_object_or_404(
               Item, id=item_id, archived_at__isnull=True,
           )
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


       items = Item.objects.filter(archived_at__isnull=True)
       return render(request, self.template_name, {
           'form': form,
           'items': items,
           'can_administer': has_admin_access(request),
       })

def _delivery_average_minutes():
    """Average completed check-in-to-check-out duration across visible records."""
    completed = DeliveryCheckIn.objects.filter(
        checked_out_at__isnull=False,
        checked_out_at__gte=F('checked_in_at'),
        archived_at__isnull=True,
    )
    duration = ExpressionWrapper(
        F('checked_out_at') - F('checked_in_at'),
        output_field=DurationField(),
    )
    stats = completed.aggregate(
        total_duration=Sum(duration),
        completed_count=Count('pk'),
    )
    if not stats['completed_count'] or stats['total_duration'] is None:
        return None
    average_seconds = (
        stats['total_duration'].total_seconds() / stats['completed_count']
    )
    return int(round(average_seconds / 60))


class DeliveryView(LoginRequiredMixin, View):
    template_name = 'delivery.html'

    def get(self, request):
        from django.utils.timezone import localdate
        active_records = DeliveryCheckIn.objects.filter(
            checked_out_at__isnull=True, archived_at__isnull=True,
        ).order_by('-checked_in_at')
        history_records = DeliveryCheckIn.objects.filter(
            checked_out_at__isnull=False, archived_at__isnull=True,
        ).order_by('-checked_out_at')[:150]
        today = localdate()
        completed_today = DeliveryCheckIn.objects.filter(
            checked_out_at__date=today, archived_at__isnull=True,
        )
        avg_minutes = _delivery_average_minutes()
        return render(request, self.template_name, {
            'active_records': active_records,
            'history_records': history_records,
            'on_site_count': active_records.count(),
            'checkin_today': DeliveryCheckIn.objects.filter(
                checked_in_at__date=today, archived_at__isnull=True,
            ).count(),
            'checkout_today': completed_today.count(),
            'avg_minutes_on_site': avg_minutes,
            'avg_time_on_site': f'{avg_minutes}m' if avg_minutes is not None else '—',
            'can_administer': has_admin_access(request),
        })

    def post(self, request):
        action = request.POST.get('action')

        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        if action == 'checkin':
            raw_barcode = request.POST.get('barcode', '').strip()
            first_name = request.POST.get('first_name', '').strip()
            last_name = request.POST.get('last_name', '').strip()
            comment = request.POST.get('comment', '').strip()

            no_barcode = _is_no_barcode(raw_barcode)
            barcode = 'NB' if no_barcode else _normalize_barcode(raw_barcode)

            def _fail(msg):
                if is_ajax:
                    return JsonResponse({'status': 'error', 'message': msg})
                messages.error(request, msg)
                return redirect('delivery')

            if not barcode or not first_name or not last_name:
                return _fail("Barcode, first name, and last name are all required.")

            # Skip duplicate check for no-barcode entries
            if not no_barcode:
                already = DeliveryCheckIn.objects.filter(
                    barcode=barcode, checked_out_at__isnull=True,
                    archived_at__isnull=True,
                ).first()
                if already:
                    return _fail(f"{already.first_name} {already.last_name} is already checked in with that barcode.")

            record = DeliveryCheckIn.objects.create(
                barcode=barcode, first_name=first_name, last_name=last_name, comment=comment,
            )
            UserAction.objects.create(user=request.user, action='delivery_checkin',
                target=f'{first_name} {last_name}', detail=f'Barcode: {barcode}')
            if is_ajax:
                return JsonResponse({
                    'status': 'ok',
                    'record_id': record.pk,
                    'name': f'{first_name} {last_name}',
                    'barcode': barcode,
                    'comment': comment,
                    'checked_in_at': record.checked_in_at.strftime('%d %b, %H:%M'),
                    'checked_in_iso': record.checked_in_at.isoformat(),
                })
            messages.success(request, f"{first_name} {last_name} checked in.")
            return redirect('delivery')

        elif action == 'checkout':
            record_id = request.POST.get('record_id', '').strip()
            barcode_raw = request.POST.get('barcode', '').strip()

            if record_id:
                record = DeliveryCheckIn.objects.filter(
                    pk=record_id, checked_out_at__isnull=True,
                    archived_at__isnull=True,
                ).first()
            elif _is_no_barcode(barcode_raw):
                return JsonResponse({
                    'status': 'error',
                    'message': 'No-barcode deliveries must be checked out from the table.',
                })
            else:
                barcode = _normalize_barcode(barcode_raw)
                record = DeliveryCheckIn.objects.filter(
                    barcode=barcode, checked_out_at__isnull=True,
                    archived_at__isnull=True,
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
                    'checked_in_at': record.checked_in_at.strftime('%d %b, %H:%M'),
                    'checked_out_at': record.checked_out_at.strftime('%d %b, %H:%M'),
                    'checked_in_iso': record.checked_in_at.isoformat(),
                    'checked_out_iso': record.checked_out_at.isoformat(),
                    'avg_minutes_on_site': _delivery_average_minutes(),
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'No active check-in found for this barcode.'})

        elif action == 'undo_checkout':
            if not has_admin_access(request):
                return JsonResponse({'status': 'error', 'message': 'Admin passkey required.'}, status=403)
            record_id = request.POST.get('record_id', '').strip()
            record = DeliveryCheckIn.objects.filter(
                pk=record_id, checked_out_at__isnull=False,
                archived_at__isnull=True,
            ).first()
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
                    'checked_in_at': record.checked_in_at.strftime('%d %b, %H:%M'),
                    'checked_in_iso': record.checked_in_at.isoformat(),
                    'avg_minutes_on_site': _delivery_average_minutes(),
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'Record not found or already active.'})

        elif action == 'delete_record':
            if not has_admin_access(request):
                return JsonResponse({'status': 'error', 'message': 'Admin passkey required.'}, status=403)
            record_id = request.POST.get('record_id', '').strip()
            record = DeliveryCheckIn.objects.filter(
                pk=record_id, archived_at__isnull=True,
            ).first()
            if record:
                name = f"{record.first_name} {record.last_name}"
                barcode = record.barcode
                record.archived_at = now()
                record.archived_by = request.user
                record.archive_reason = 'Removed from delivery page'
                record.save(update_fields=['archived_at', 'archived_by', 'archive_reason'])
                UserAction.objects.create(user=request.user, action='delivery_delete_record',
                    target=name, detail=f'Barcode: {barcode}')
                return JsonResponse({
                    'status': 'ok',
                    'record_id': int(record_id),
                    'name': name,
                    'avg_minutes_on_site': _delivery_average_minutes(),
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'Record not found.'})

        elif action == 'clear_history':
            if not has_admin_access(request):
                return redirect(
                    f"{reverse('passkey_unlock')}?{urlencode({'next': reverse('delivery')})}"
                )
            history = DeliveryCheckIn.objects.filter(
                checked_out_at__isnull=False, archived_at__isnull=True,
            )
            del_count = history.count()
            history.update(
                archived_at=now(), archived_by=request.user,
                archive_reason='Delivery history cleared',
            )
            UserAction.objects.create(user=request.user, action='delivery_clear_history',
                target=f'{del_count} records cleared')
            messages.success(request, "Checkout history cleared.")
            return redirect('delivery')

        return redirect('delivery')


class OrderingSheetView(LoginRequiredMixin, View):
    """Daily ordering sheet.

    Any logged-in user can add a row and edit/delete their own pending rows.
    Staff or a passkey-unlocked session can manage every row and advance the
    structured ordering lifecycle.
    """
    template_name = 'ordering_sheet.html'
    embed_template_name = 'ordering_sheet_embed.html'

    @staticmethod
    def _is_embed(request):
        # The dashboard opens this page in an iframe modal with ?embed=1 — render
        # without the nav chrome and keep the flag across post→redirect.
        return request.GET.get('embed') == '1'

    def _redirect(self, request):
        view_mode = request.GET.get('view', '')
        suffix = f"&view={view_mode}" if view_mode else ''
        if self._is_embed(request):
            return redirect(f"{reverse('ordering_sheet')}?embed=1{suffix}")
        if view_mode:
            return redirect(f"{reverse('ordering_sheet')}?view={view_mode}")
        return redirect('ordering_sheet')

    @staticmethod
    def _can_edit_entry(request, entry):
        return has_admin_access(request) or (
            entry.created_by_id == request.user.id
            and entry.status == OrderingSheetEntry.STATUS_PENDING
        )

    def _render_page(self, request, *, form=None, otc_form=None, status=200):
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
        view_mode = request.GET.get('view', 'active')
        if view_mode not in {'active', 'completed', 'all'}:
            view_mode = 'active'
        entries = OrderingSheetEntry.objects.filter(is_deleted=False)
        if view_mode == 'active':
            # Keep Not for Sale visible in the working table: the page has a
            # dedicated filter chip for it and staff may still need to review
            # the pharmacist decision. Picked-up/cancelled rows are historical.
            entries = entries.exclude(status__in=(
                OrderingSheetEntry.STATUS_PICKED_UP,
                OrderingSheetEntry.STATUS_CANCELLED,
            ))
        elif view_mode == 'completed':
            entries = entries.filter(status__in=OrderingSheetEntry.TERMINAL_STATUSES)
        entries = (entries
                   .annotate(type_rank=type_rank, urgency_rank=urgency_rank)
                   .prefetch_related('status_events__changed_by')
                   .order_by('type_rank', 'urgency_rank', '-created_at'))

        # (value, label) pairs GINA can pick from the inline status dropdown.
        status_labels = dict(OrderingSheetEntry.STATUS_CHOICES)
        gina_status_options = [(v, status_labels[v]) for v in OrderingSheetEntry.ADMIN_STATUS_CHOICES]
        entries = list(entries)
        known_supplier_values = dict(OrderingSheetEntry.SUPPLIER_CHOICES)
        for entry in entries:
            entry.can_user_edit = self._can_edit_entry(request, entry)
            entry.has_legacy_supplier = bool(
                entry.supplier_name and entry.supplier_name not in known_supplier_values
            )
            allowed_statuses = {
                entry.status,
                *OrderingSheetEntry.STATUS_TRANSITIONS.get(entry.status, set()),
            }
            entry.status_options = [
                option for option in gina_status_options
                if option[0] in allowed_statuses
            ]

        # Google Sheet sync status and next database-backed pre-closing run.
        from app.gsheet_sync import is_configured as gsheet_configured, load_state as gsheet_state
        from app.scheduled_jobs import next_gsheet_pull
        gsheet_enabled = gsheet_configured()
        gsheet_last_sync = None
        gsheet_last_status = None
        gsheet_last_summary = ''
        gsheet_next_pull = None
        gsheet_pull_due = False
        if gsheet_enabled:
            latest_sync_run = ScheduledJobRun.objects.filter(
                job_key=ScheduledJobRun.JOB_GSHEET_PRECLOSE,
            ).exclude(status=ScheduledJobRun.STATUS_RUNNING).first()
            if latest_sync_run:
                completed = latest_sync_run.completed_at or latest_sync_run.started_at
                gsheet_last_sync = localtime(completed).strftime('%b %d at %H:%M')
                gsheet_last_status = latest_sync_run.status
                gsheet_last_summary = latest_sync_run.summary
            else:
                state = gsheet_state()
                if state and state.get('last_sync'):
                    gsheet_last_sync = datetime.fromtimestamp(state['last_sync']).strftime('%b %d at %H:%M')
            next_pull = next_gsheet_pull()
            if next_pull:
                gsheet_next_pull = next_pull['scheduled_for']
                gsheet_pull_due = next_pull['due']

        embed = self._is_embed(request)
        template = self.embed_template_name if embed else self.template_name
        response = render(request, template, {
            'form': form if form is not None else OrderingSheetForm(prefix='drug'),
            'otc_form': otc_form if otc_form is not None else OTCOrderingForm(prefix='otc'),
            'entries': entries,
            'gina_status_options': gina_status_options,
            'ordering_supplier_choices': OrderingSheetEntry.SUPPLIER_CHOICES,
            'can_administer': has_admin_access(request),
            'view_mode': view_mode,
            'embed': embed,
            'gsheet_enabled': gsheet_enabled,
            'gsheet_last_sync': gsheet_last_sync,
            'gsheet_last_status': gsheet_last_status,
            'gsheet_last_summary': gsheet_last_summary,
            'gsheet_next_pull': gsheet_next_pull,
            'gsheet_pull_due': gsheet_pull_due,
        }, status=status)
        if embed:
            # Project default is X-Frame-Options: DENY. Allow this page to load
            # inside the dashboard's same-origin iframe modal. Setting the header
            # here pre-empts XFrameOptionsMiddleware (it won't overwrite it).
            response['X-Frame-Options'] = 'SAMEORIGIN'
        return response

    def get(self, request):
        return self._render_page(request)

    def post(self, request):
        action = request.POST.get('action')

        if action == 'sync_gsheet':
            if not has_admin_access(request):
                messages.error(request, "Admin passkey required to sync the shared sheet.")
                return self._redirect(request)
            from app.gsheet_sync import is_configured as gsheet_configured
            from app.scheduled_jobs import run_google_sheet_sync
            if not gsheet_configured():
                messages.error(request, "Google Sheet sync is not configured.")
                return self._redirect(request)
            _run, result = run_google_sheet_sync(created_by=request.user)
            if result['errors']:
                messages.error(request, f"Google Sheet pull problem: {result['errors'][0]}")
            elif result['imported']:
                n = result['imported']
                messages.success(request, f"Pulled {n} new item{'s' if n != 1 else ''} from the Google Sheet.")
            else:
                messages.success(request, "Google Sheet checked — no new items.")
            return self._redirect(request)

        if action == 'add':
            form = OrderingSheetForm(request.POST, prefix='drug')
            if form.is_valid():
                entry = form.save(commit=False)
                entry.entry_type = OrderingSheetEntry.ENTRY_DRUG
                entry.created_by = request.user
                entry.status = OrderingSheetEntry.STATUS_PENDING
                entry.save()
                OrderingSheetStatusEvent.objects.create(
                    entry=entry, from_status=entry.status, to_status=entry.status,
                    note='Entry created', changed_by=request.user,
                )
                messages.success(request, f"Added “{entry.name}” to the ordering sheet.")
            else:
                first_error = next(iter(form.errors.values()))[0]
                messages.error(request, f"Could not add entry: {first_error}")
                return self._render_page(
                    request,
                    form=form,
                    otc_form=OTCOrderingForm(prefix='otc'),
                    status=422,
                )
            return self._redirect(request)

        elif action == 'add_otc':
            form = OTCOrderingForm(request.POST, prefix='otc')
            if form.is_valid():
                entry = form.save(commit=False)
                entry.entry_type = OrderingSheetEntry.ENTRY_OTC
                entry.reasoning = ''
                entry.urgency = OrderingSheetEntry.URGENCY_NA
                entry.created_by = request.user
                entry.status = OrderingSheetEntry.STATUS_PENDING
                entry.save()
                OrderingSheetStatusEvent.objects.create(
                    entry=entry, from_status=entry.status, to_status=entry.status,
                    note='Entry created', changed_by=request.user,
                )
                messages.success(request, f"Added OTC product “{entry.name}” to the ordering sheet.")
            else:
                first_error = next(iter(form.errors.values()))[0]
                messages.error(request, f"Could not add OTC product: {first_error}")
                return self._render_page(
                    request,
                    form=OrderingSheetForm(prefix='drug'),
                    otc_form=form,
                    status=422,
                )
            return self._redirect(request)

        elif action == 'update_status':
            if not has_admin_access(request):
                messages.error(request, "Admin passkey required to change ordering progress.")
                return self._redirect(request)

            entry = OrderingSheetEntry.objects.filter(
                pk=request.POST.get('entry_id'), is_deleted=False,
            ).first()
            new_status = request.POST.get('status', '')
            if not entry:
                messages.error(request, "Ordering-sheet entry not found.")
            elif new_status not in OrderingSheetEntry.ADMIN_STATUS_CHOICES:
                messages.error(request, "Invalid status.")
            elif not entry.can_transition_to(new_status):
                messages.error(
                    request,
                    f"{entry.get_status_display()} cannot move directly to "
                    f"{dict(OrderingSheetEntry.STATUS_CHOICES).get(new_status, new_status)}.",
                )
            else:
                supplier_name = request.POST.get('supplier_name', '').strip()
                valid_suppliers = dict(OrderingSheetEntry.SUPPLIER_CHOICES)
                if (
                    supplier_name
                    and supplier_name not in valid_suppliers
                    and supplier_name != entry.supplier_name
                ):
                    messages.error(request, "Choose McKesson, K&F, or Direct as the supplier.")
                    return self._redirect(request)

                fully_received_statuses = {
                    OrderingSheetEntry.STATUS_RECEIVED,
                    OrderingSheetEntry.STATUS_READY,
                    OrderingSheetEntry.STATUS_CONTACTED,
                    OrderingSheetEntry.STATUS_PICKED_UP,
                }
                quantity_required_statuses = {
                    OrderingSheetEntry.STATUS_BACKORDERED,
                    OrderingSheetEntry.STATUS_ORDERED,
                    OrderingSheetEntry.STATUS_PARTIAL_RECEIVED,
                    *fully_received_statuses,
                }
                try:
                    quantity_ordered = request.POST.get('quantity_ordered', '').strip()
                    quantity_ordered = int(quantity_ordered) if quantity_ordered else None
                    if quantity_ordered is not None and quantity_ordered < 0:
                        raise ValueError

                    if new_status == OrderingSheetEntry.STATUS_PARTIAL_RECEIVED:
                        quantity_received_raw = request.POST.get('quantity_received', '').strip()
                        quantity_received = int(quantity_received_raw) if quantity_received_raw else 0
                    elif new_status in fully_received_statuses:
                        # A full-received lifecycle state is authoritative: avoid
                        # making staff type the ordered amount a second time. A
                        # value from an older cached form is still validated.
                        quantity_received = quantity_ordered or 0
                        posted_received = request.POST.get('quantity_received', '').strip()
                        if posted_received and int(posted_received) != quantity_received:
                            raise ValueError
                    else:
                        # Qty received is intentionally hidden outside the partial
                        # state. Preserve any earlier partial receipt instead of
                        # silently resetting it during a later status change.
                        quantity_received = entry.quantity_received or 0

                    if quantity_received < 0 or (
                        quantity_ordered is not None and quantity_received > quantity_ordered
                    ):
                        raise ValueError
                except (TypeError, ValueError):
                    messages.error(
                        request,
                        "Enter valid quantities; Qty received cannot exceed Qty ordered.",
                    )
                    return self._redirect(request)

                if new_status in quantity_required_statuses and not quantity_ordered:
                    messages.error(request, "Enter a Qty ordered greater than zero for this status.")
                    return self._redirect(request)

                expected_raw = request.POST.get('expected_date', '').strip()
                expected_date = parse_date(expected_raw) if expected_raw else None
                if expected_raw and not expected_date:
                    messages.error(request, "Enter a valid expected date.")
                    return self._redirect(request)

                if new_status == OrderingSheetEntry.STATUS_PARTIAL_RECEIVED and not (
                    quantity_ordered
                    and 0 < quantity_received < quantity_ordered
                ):
                    messages.error(
                        request,
                        "Partially Received needs an ordered quantity and a received "
                        "quantity greater than zero but below it.",
                    )
                    return self._redirect(request)
                if new_status in fully_received_statuses and not (
                    quantity_ordered and quantity_received == quantity_ordered
                ):
                    messages.error(
                        request,
                        "Received, Ready, Contacted, and Picked Up require the full "
                        "ordered quantity to be recorded as received.",
                    )
                    return self._redirect(request)

                with transaction.atomic():
                    entry = OrderingSheetEntry.objects.select_for_update().get(pk=entry.pk)
                    old_status = entry.status
                    if not entry.can_transition_to(new_status):
                        messages.error(request, "This entry changed; refresh and try again.")
                        return self._redirect(request)
                    timestamp = now()
                    entry.status = new_status
                    entry.supplier_name = supplier_name
                    entry.expected_date = expected_date
                    entry.quantity_ordered = quantity_ordered
                    entry.quantity_received = quantity_received
                    entry.order_note = request.POST.get('order_note', '').strip()[:255]
                    entry.status_updated_by = request.user
                    entry.status_updated_at = timestamp
                    if new_status == OrderingSheetEntry.STATUS_ORDERED and not entry.ordered_at:
                        entry.ordered_at = timestamp
                    if new_status in {
                        OrderingSheetEntry.STATUS_RECEIVED,
                        OrderingSheetEntry.STATUS_READY,
                        OrderingSheetEntry.STATUS_CONTACTED,
                        OrderingSheetEntry.STATUS_PICKED_UP,
                    } and not entry.received_at:
                        entry.received_at = timestamp
                    if new_status == OrderingSheetEntry.STATUS_CONTACTED and not entry.contacted_at:
                        entry.contacted_at = timestamp
                    if new_status in OrderingSheetEntry.TERMINAL_STATUSES:
                        entry.completed_at = timestamp
                    entry.save()
                    OrderingSheetStatusEvent.objects.create(
                        entry=entry, from_status=old_status, to_status=new_status,
                        note=entry.order_note, changed_by=request.user,
                    )
                UserAction.objects.create(user=request.user, action='ordering_status_update',
                    target=entry.name, detail=f'Status → {entry.get_status_display()}')
                messages.success(request, f"“{entry.name}” is now {entry.get_status_display()}.")
            return self._redirect(request)

        elif action == 'update_note':
            if not has_admin_access(request):
                messages.error(request, "Admin passkey required to edit progress notes.")
                return self._redirect(request)
            entry = OrderingSheetEntry.objects.filter(pk=request.POST.get('entry_id'), is_deleted=False).first()
            if entry:
                entry.order_note = request.POST.get('order_note', '').strip()[:255]
                entry.save(update_fields=['order_note'])
                messages.success(request, "Note saved.")
            else:
                messages.error(request, "Ordering-sheet entry not found.")
            return self._redirect(request)

        elif action == 'edit':
            entry = OrderingSheetEntry.objects.filter(pk=request.POST.get('entry_id'), is_deleted=False).first()
            if not entry:
                messages.error(request, "Ordering-sheet entry not found.")
                return self._redirect(request)
            if not self._can_edit_entry(request, entry):
                messages.error(request, "You can only edit your own pending entries; use the admin passkey for others.")
                return self._redirect(request)
            name = (request.POST.get('name') or '').strip()
            initials = (request.POST.get('initials') or '').strip()
            if not name or not initials:
                messages.error(request, "Name and initials are required.")
                return self._redirect(request)
            entry.name = name[:200]
            entry.initials = initials[:20]
            entry.patient_name = (request.POST.get('patient_name') or '').strip()[:200]
            entry.quantity_needed = (request.POST.get('quantity_needed') or '').strip()[:50]
            entry.quantity_remaining = (request.POST.get('quantity_remaining') or '').strip()[:50]
            if entry.entry_type == OrderingSheetEntry.ENTRY_OTC:
                side = request.POST.get('side', '')
                if side in dict(OrderingSheetEntry.SIDE_CHOICES):
                    entry.side = side
                entry.phone_number = (request.POST.get('phone_number') or '').strip()[:20]
            else:
                reasoning = request.POST.get('reasoning', '')
                if reasoning in dict(OrderingSheetEntry.REASON_CHOICES):
                    entry.reasoning = reasoning
                urgency = request.POST.get('urgency', '')
                if urgency in dict(OrderingSheetEntry.URGENCY_CHOICES):
                    entry.urgency = urgency
            entry.save()
            UserAction.objects.create(user=request.user, action='ordering_edit',
                target=entry.name)
            messages.success(request, f"Updated “{entry.name}”.")
            return self._redirect(request)

        elif action == 'delete_selected':
            raw = request.POST.get('entry_ids', '')
            ids = [int(x) for x in raw.split(',') if x.strip().isdigit()]
            count = 0
            for entry in OrderingSheetEntry.objects.filter(pk__in=ids, is_deleted=False):
                if not self._can_edit_entry(request, entry):
                    continue
                entry.is_deleted = True
                entry.deleted_at = now()
                entry.deleted_by = request.user
                entry.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])
                UserAction.objects.create(user=request.user, action='ordering_delete',
                    target=entry.name)
                count += 1
            if count:
                messages.success(request, f"Removed {count} entr{'y' if count == 1 else 'ies'} from the ordering sheet.")
            else:
                messages.error(request, "No matching ordering-sheet entries found.")
            return self._redirect(request)

        elif action == 'delete':
            entry = OrderingSheetEntry.objects.filter(pk=request.POST.get('entry_id'), is_deleted=False).first()
            if entry and self._can_edit_entry(request, entry):
                entry.is_deleted = True
                entry.deleted_at = now()
                entry.deleted_by = request.user
                entry.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])
                UserAction.objects.create(user=request.user, action='ordering_delete',
                    target=entry.name)
                messages.success(request, f"Removed “{entry.name}” from the ordering sheet.")
            elif entry:
                messages.error(request, "You can only remove your own pending entries; use the admin passkey for others.")
            else:
                messages.error(request, "Ordering-sheet entry not found.")
            return self._redirect(request)

        return self._redirect(request)


@login_required
def update_product_settings(request, product_id):
    if request.method != 'POST':
        return redirect('create_order')
    if not has_admin_access(request):
        return redirect(
            f"{reverse('passkey_unlock')}?{urlencode({'next': reverse('create_order')})}"
        )

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
