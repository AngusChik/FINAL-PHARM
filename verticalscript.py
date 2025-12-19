import csv
from reportlab.lib.pagesizes import letter, portrait
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.barcode import code128

# ---------- Label size ----------
LABEL_WIDTH  = 2.00 * inch
LABEL_HEIGHT = 1.25 * inch

PAGE_WIDTH, PAGE_HEIGHT = portrait(letter)

# ---------- REQUIRED margins ----------
LEFT_MARGIN   = 0.25 * inch
RIGHT_MARGIN  = 0.25 * inch
TOP_MARGIN    = 0.50 * inch
BOTTOM_MARGIN = 0.50 * inch

# Grid: 4 columns × 8 rows
COLUMNS = 4
ROWS    = 8
LABELS_PER_PAGE = COLUMNS * ROWS

# Inner padding (points)
LEFT_PADDING = 6
RIGHT_PADDING = 6
TOP_PADDING = 4
BOTTOM_PADDING = 4

# Optional printer calibration (points). 1 pt = 1/72 inch.
# If col 1 is still too far left, try X_NUDGE = +2 or +4.
# If everything sits too high/low, tweak Y_NUDGE.
X_NUDGE = 0
Y_NUDGE = 0

CSV_FILE = "current.csv"
OUTPUT_PDF = "labels_portrait.pdf"


def wrap_text(text, font_name, font_size, max_width):
    words = text.split()
    lines, current = [], ""
    for w in words:
        test = (current + " " + w) if current else w
        if stringWidth(test, font_name, font_size) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def draw_label(c, x, y, data):
    """Draw a single label (x,y = bottom-left)."""
    name = (data.get("name") or "").strip().title()
    brand = (data.get("brand") or "").strip().title()
    item_number = (data.get("item_number") or "").strip()
    barcode_value = (data.get("barcode") or "").strip()

    # Price
    try:
        price_val = float(data.get("price", 0) or 0)
    except:
        price_val = 0.0
    price = f"${price_val:.2f}"

    # === Product name (top, centered) ===
    name_font = "Helvetica-Bold"
    name_size = 10
    max_text_width = LABEL_WIDTH - LEFT_PADDING - RIGHT_PADDING
    lines = wrap_text(name, name_font, name_size, max_text_width)[:4]

    NAME_BLOCK_TOP = LABEL_HEIGHT - 10
    c.setFont(name_font, name_size)
    for i, line in enumerate(lines):
        text_y = y + NAME_BLOCK_TOP - (i * (name_size + 1))
        c.drawCentredString(x + LABEL_WIDTH / 2, text_y, line)

    # Left stack (flush-left)
    base_y = y + BOTTOM_PADDING
    body_x = x + LEFT_PADDING

    if barcode_value:
        barcode = code128.Code128(
            barcode_value,
            barHeight=16,
            barWidth=0.9,
            humanReadable=False,
            quiet=False
        )
        lq = getattr(barcode, "lquiet", 0) if getattr(barcode, "quiet", False) else 0
        barcode_bottom_y = base_y + 20
        barcode.drawOn(c, body_x - lq, barcode_bottom_y)

        c.setFont("Helvetica", 6)
        c.drawString(body_x, base_y + 14, barcode_value)

    if item_number:
        c.setFont("Helvetica", 6)
        c.drawString(body_x, base_y + 8, f"Item #: {item_number}")

    if brand:
        c.setFont("Helvetica", 6)
        brand_line = wrap_text(brand, "Helvetica", 6, LABEL_WIDTH - LEFT_PADDING - RIGHT_PADDING)[:1]
        if brand_line:
            c.drawString(body_x, base_y + 2, brand_line[0])

    # === Price (bottom-right) ===
    c.setFont("Helvetica-Bold", 17)
    c.drawRightString(x + LABEL_WIDTH - RIGHT_PADDING, base_y + 4, price)


def generate_labels(csv_file, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=portrait(letter))
    count = 0

    # Compute usable area inside margins
    usable_w = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
    usable_h = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN

    # Compute gutters so grid fills usable area evenly
    # (for your numbers, these should end up as 0, but this keeps it exact/robust)
    h_gutter = 0
    v_gutter = 0
    if COLUMNS > 1:
        h_gutter = (usable_w - (COLUMNS * LABEL_WIDTH)) / (COLUMNS - 1)
    if ROWS > 1:
        v_gutter = (usable_h - (ROWS * LABEL_HEIGHT)) / (ROWS - 1)

    # If a gutter goes negative (doesn't fit), clamp to 0 to avoid overlapping
    if h_gutter < 0:
        h_gutter = 0
    if v_gutter < 0:
        v_gutter = 0

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            col = count % COLUMNS
            row_num = (count // COLUMNS) % ROWS

            # Exact grid placement, respecting margins
            x = LEFT_MARGIN + col * (LABEL_WIDTH + h_gutter) + X_NUDGE

            # Top-based layout: row 0 is the first row under TOP_MARGIN
            y_top = PAGE_HEIGHT - TOP_MARGIN - row_num * (LABEL_HEIGHT + v_gutter) + Y_NUDGE
            y = y_top - LABEL_HEIGHT

            draw_label(c, x, y, row)
            count += 1

            if count % LABELS_PER_PAGE == 0:
                c.showPage()

    if count % LABELS_PER_PAGE != 0:
        c.showPage()

    c.save()
    print(f"✅ Labels saved to {output_pdf}")


if __name__ == "__main__":
    generate_labels(CSV_FILE, OUTPUT_PDF)
