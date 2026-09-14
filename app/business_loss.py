"""Editable business-loss reports derived from the expired-stock audit trail."""

import io
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.views import View
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.models import StockChange
from app.navigation import safe_local_return_url


MAX_QUANTITY = 2147483647
MAX_UNIT_COST = Decimal("99999999.99")
# A line can contain MAX_QUANTITY units at the largest stored unit cost.
MAX_TOTAL_COST = Decimal("999999999999999999.99")
ROW_FIELDS = ("product", "quantity", "cost_per_unit", "total_cost", "year")
WHOLE_NUMBER = re.compile(r"[0-9]+\Z")
MONEY_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]{0,2})?|\.[0-9]{1,2})\Z")
INVALID_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _return_url(request, raw):
    candidate = safe_local_return_url(request, raw, fallback_name="expired_log")
    if urlsplit(candidate).path.rstrip("/") != reverse("expired_log").rstrip("/"):
        return reverse("expired_log")
    return candidate


def _context(rows, report_date, return_url, **errors):
    total_units = 0
    total_cost = Decimal("0.00")
    has_missing_costs = False
    for row in rows:
        quantity = row["quantity"].strip()
        cost = row["total_cost"].strip()
        if WHOLE_NUMBER.fullmatch(quantity) and len(quantity) <= 10:
            total_units += int(quantity)
        if MONEY_NUMBER.fullmatch(cost) and len(cost) <= 21:
            total_cost += Decimal(cost)
        if not row["cost_per_unit"].strip() or not cost:
            has_missing_costs = True
    context = {
        "rows": rows,
        "report_date": report_date,
        "return_url": return_url,
        "page_return": {
            "url": return_url,
            "destination": "Expired Log",
            "label": "Back to Expired Log",
            "source": "workflow-parent",
        },
        "filter_error": "",
        "form_error": "",
        "date_error": "",
        "total_units": total_units,
        "total_cost": format(total_cost, ".2f"),
        "has_missing_costs": has_missing_costs,
    }
    context.update(errors)
    return context


def _validate_rows(raw_rows):
    """Keep submitted strings for correction, returning typed rows only if valid."""
    try:
        submitted = json.loads(raw_rows)
    except (json.JSONDecodeError, TypeError, ValueError):
        return [], [], "The report data could not be read. Reload the form and try again."
    if not isinstance(submitted, list) or not submitted:
        return [], [], "Add at least one product before generating the PDF."

    rows = []
    validated = []
    has_errors = False
    for item in submitted:
        values = item if isinstance(item, dict) else {}
        row = {
            field: values[field] if isinstance(values.get(field), str) else ""
            for field in ROW_FIELDS
        }
        # Display-only context survives validation errors without entering the PDF.
        try:
            added_date = values.get("added_date", "")
            row["added_date"] = date.fromisoformat(added_date)
            if row["added_date"].isoformat() != added_date:
                row["added_date"] = None
        except (TypeError, ValueError):
            row["added_date"] = None
        errors = {}
        product = row["product"].strip()
        if not product:
            errors["product"] = "Enter a product name."
        elif len(product) > 200:
            errors["product"] = "Use 200 characters or fewer."
        elif INVALID_TEXT.search(product):
            errors["product"] = "Use a product name without control characters."

        quantity_text = row["quantity"].strip()
        if (
            not WHOLE_NUMBER.fullmatch(quantity_text)
            or len(quantity_text) > 10
            or int(quantity_text) > MAX_QUANTITY
        ):
            errors["quantity"] = f"Enter a whole number from 0 to {MAX_QUANTITY}."
            quantity = 0
        else:
            quantity = int(quantity_text)

        money = {}
        for field, maximum in (
            ("cost_per_unit", MAX_UNIT_COST),
            ("total_cost", MAX_TOTAL_COST),
        ):
            value = row[field].strip()
            try:
                if not MONEY_NUMBER.fullmatch(value) or len(value) > 21:
                    raise InvalidOperation
                amount = Decimal(value)
                if not amount.is_finite() or amount < 0 or amount > maximum:
                    raise InvalidOperation
                money[field] = amount.quantize(Decimal("0.01"))
            except InvalidOperation:
                errors[field] = f"Enter a cost from 0 to {maximum:,.2f}, with at most 2 decimal places."

        year_text = row["year"].strip()
        if (
            not WHOLE_NUMBER.fullmatch(year_text)
            or len(year_text) > 4
            or not 1 <= int(year_text) <= 9999
        ):
            errors["year"] = "Enter a year from 1 to 9999."
            year = 0
        else:
            year = int(year_text)

        row["errors"] = errors
        rows.append(row)
        if errors:
            has_errors = True
        else:
            validated.append({
                "product": product,
                "quantity": quantity,
                "cost_per_unit": money["cost_per_unit"],
                "total_cost": money["total_cost"],
                "year": year,
            })
    if has_errors:
        return rows, [], "Review the highlighted fields before generating the PDF."
    return rows, validated, ""


class BusinessLossView(LoginRequiredMixin, View):
    """Prepare a report and download edits without changing inventory or its log."""

    def get(self, request):
        query = request.GET.get("q", "").strip()
        date_from = request.GET.get("from", "").strip()
        date_to = request.GET.get("to", "").strip()
        logs = StockChange.objects.filter(change_type="expired")
        if query:
            logs = logs.filter(
                Q(product_name__icontains=query)
                | Q(product_barcode__icontains=query)
                | Q(product__name__icontains=query)
                | Q(product__barcode__icontains=query)
                | Q(user__username__icontains=query)
                | Q(lot_movements__lot_number__icontains=query)
            ).distinct()

        filter_error = ""
        try:
            lower = date.fromisoformat(date_from) if date_from else None
            upper = date.fromisoformat(date_to) if date_to else None
            if lower and upper and lower > upper:
                filter_error = "The start date must be on or before the end date."
            if lower:
                logs = logs.filter(timestamp__date__gte=lower)
            if upper:
                logs = logs.filter(timestamp__date__lte=upper)
        except ValueError:
            filter_error = "Enter valid logged dates."
        if filter_error:
            logs = logs.none()

        rows = []
        for log in logs.select_related("product").order_by("-timestamp", "-pk"):
            quantity = abs(log.quantity)
            cost = log.product.price_per_unit if log.product else None
            added_date = timezone.localdate(log.timestamp)
            rows.append({
                "product": log.product_name or log.display_name,
                "quantity": str(quantity),
                "cost_per_unit": format(cost, ".2f") if cost is not None else "",
                "total_cost": format(cost * quantity, ".2f") if cost is not None else "",
                "year": str(added_date.year),
                "added_date": added_date,
                "errors": {},
            })
        return render(request, "business_loss.html", _context(
            rows,
            timezone.localdate().isoformat(),
            _return_url(request, request.GET.get("return_to")),
            filter_error=filter_error,
        ))

    def post(self, request):
        rows, validated, form_error = _validate_rows(request.POST.get("rows"))
        date_text = request.POST.get("report_date", "").strip()
        report_date = None
        date_error = ""
        try:
            report_date = date.fromisoformat(date_text)
            if report_date.isoformat() != date_text:
                raise ValueError
        except ValueError:
            date_error = "Enter a valid report date."
        if form_error or date_error:
            return render(request, "business_loss.html", _context(
                rows, date_text,
                _return_url(request, request.POST.get("return_to")),
                form_error=form_error,
                date_error=date_error,
            ), status=400)

        response = HttpResponse(
            build_business_loss_pdf(validated, report_date),
            content_type="application/pdf",
        )
        response["Content-Disposition"] = (
            f'attachment; filename="MPCP-Business-Loss-{report_date:%Y%m%d}.pdf"'
        )
        return response


def build_business_loss_pdf(rows, report_date):
    """Render validated editable values, including explicit total-cost overrides."""
    buffer = io.BytesIO()
    navy = colors.HexColor("#17334C")
    muted = colors.HexColor("#526477")
    border = colors.HexColor("#DDE5EC")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BusinessLossTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=28, textColor=navy, alignment=0, spaceAfter=5,
    )
    note_style = ParagraphStyle(
        "BusinessLossNote", parent=styles["Normal"], fontSize=9,
        leading=13, textColor=muted,
    )
    cell_style = ParagraphStyle(
        "BusinessLossCell", parent=styles["Normal"], fontSize=8.5,
        leading=12, textColor=navy, splitLongWords=True,
    )
    number_style = ParagraphStyle(
        "BusinessLossNumber", parent=cell_style, fontSize=8,
        alignment=TA_RIGHT,
    )
    header_style = ParagraphStyle(
        "BusinessLossHeader", parent=cell_style, fontName="Helvetica-Bold",
        fontSize=8, leading=11, textColor=colors.white,
    )
    header_number_style = ParagraphStyle(
        "BusinessLossHeaderNumber", parent=header_style, alignment=TA_RIGHT,
    )
    total_units = sum(row["quantity"] for row in rows)
    total_cost = sum((row["total_cost"] for row in rows), Decimal("0.00"))
    document = SimpleDocTemplate(
        buffer, pagesize=letter, leftMargin=36, rightMargin=36,
        topMargin=133, bottomMargin=49, title="MPCP Business Loss", author="MPCP",
    )

    def page_decoration(pdf_canvas, doc):
        pdf_canvas.saveState()
        width, height = letter
        pdf_canvas.setFillColor(navy)
        pdf_canvas.setFont("Helvetica-Bold", 28)
        pdf_canvas.drawCentredString(width / 2, height - 48, "MPCP")
        pdf_canvas.setFont("Helvetica-Bold", 12)
        pdf_canvas.drawCentredString(
            width / 2, height - 67, "Meadowvale Professional Center Pharmacy",
        )
        pdf_canvas.setFont("Helvetica", 9)
        pdf_canvas.setFillColor(muted)
        pdf_canvas.drawCentredString(
            width / 2, height - 83,
            "6855 Meadowvale Town Centre Cir, Mississauga, ON L5N 2Y1",
        )
        pdf_canvas.drawCentredString(width / 2, height - 98, "Phone: (905) 821-9992")
        pdf_canvas.setStrokeColor(border)
        pdf_canvas.line(36, height - 113, width - 36, height - 113)
        pdf_canvas.setStrokeColor(navy)
        pdf_canvas.setLineWidth(2)
        pdf_canvas.line(width / 2 - 24, height - 113, width / 2 + 24, height - 113)
        pdf_canvas.setStrokeColor(border)
        pdf_canvas.setLineWidth(1)
        pdf_canvas.line(36, 37, width - 36, 37)
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.drawString(36, 23, f"MPCP  |  {report_date:%B %d, %Y}")
        pdf_canvas.drawRightString(width - 36, 23, f"Page {doc.page}")
        pdf_canvas.restoreState()

    story = [
        Paragraph("Business Loss", title_style),
        Paragraph(f"Report date: {report_date:%B %d, %Y}", note_style),
        Spacer(1, 16),
    ]
    summary_style = ParagraphStyle(
        "BusinessLossSummary", parent=cell_style, fontSize=11, leading=17,
    )
    summary = Table([[
        Paragraph(f"<font size='8'>TOTAL COST</font><br/><b>${total_cost:,.2f}</b>", summary_style),
        Paragraph(f"<font size='8'>NUMBER OF UNITS</font><br/><b>{total_units:,}</b>", summary_style),
        Paragraph(f"<font size='8'>PRODUCT ENTRIES</font><br/><b>{len(rows):,}</b>", summary_style),
    ]], colWidths=[240, 160, 140])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF3F7")),
        ("BOX", (0, 0), (-1, -1), 0.5, border),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([summary, Spacer(1, 20)])

    headers = ["Product", "Qty", "Cost per unit", "Total cost", "Year"]
    table_rows = [[
        Paragraph(text, header_style if index == 0 else header_number_style)
        for index, text in enumerate(headers)
    ]]
    for row in rows:
        table_rows.append([
            Paragraph(escape(" ".join(row["product"].split())), cell_style),
            Paragraph(f'{row["quantity"]:,}', number_style),
            Paragraph(f'${row["cost_per_unit"]:,.2f}', number_style),
            Paragraph(f'${row["total_cost"]:,.2f}', number_style),
            Paragraph(str(row["year"]), number_style),
        ])
    table_rows.append([
        Paragraph(f"<b>Total units: {total_units:,}</b>", cell_style),
        "",
        Paragraph(f"<b>${total_cost:,.2f}</b>", number_style),
        "",
        "",
    ])
    table = Table(table_rows, colWidths=[205, 70, 85, 125, 55], repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F7F9FB")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E8EFF5")),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, navy),
        ("SPAN", (0, -1), (1, -1)),
        ("SPAN", (2, -1), (3, -1)),
    ]))
    story.append(table)
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    return buffer.getvalue()
