import csv
from reportlab.lib.pagesizes import letter, portrait
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.barcode import code128

# ---------- Stock + margins ----------
LABEL_WIDTH  = 2.00 * inch    # 144 pt
LABEL_HEIGHT = 1.25 * inch    # 90 pt

PAGE_WIDTH, PAGE_HEIGHT = portrait(letter)

LEFT_MARGIN   = 0.25 * inch   # 18 pt
RIGHT_MARGIN  = 0.25 * inch   # 18 pt
TOP_MARGIN    = 0.5 * inch    # 36 pt
BOTTOM_MARGIN = 0.5 * inch    # 36 pt

# Grid: 4 columns × 8 rows
COLUMNS = 4
ROWS    = 8
LABELS_PER_PAGE = COLUMNS * ROWS

# Inner padding
LEFT_PADDING = 6
RIGHT_PADDING = 6
TOP_PADDING = 4
BOTTOM_PADDING = 4

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
    lines = wrap_text(name, name_font, name_size, max_text_width)[:4]  # allow up to 3 lines

    NAME_BLOCK_TOP = LABEL_HEIGHT - 10
    for i, line in enumerate(lines):
        text_y = y + NAME_BLOCK_TOP - (i * (name_size + 1))
        c.setFont(name_font, name_size)
        c.drawCentredString(x + LABEL_WIDTH / 2, text_y, line)

    # Left stack (all flush-left) — ORDER: barcode → digits → item # → brand
    base_y = y + BOTTOM_PADDING
    body_x = x + LEFT_PADDING

    if barcode_value:
        # 1) Barcode image first (top of the left stack)
        barcode = code128.Code128(
            barcode_value,
            barHeight=16,
            barWidth=0.9,
            humanReadable=False,
            quiet=False
        )
        lq = getattr(barcode, "lquiet", 0) if getattr(barcode, "quiet", False) else 0
        barcode_bottom_y = base_y + 20     # higher so it's visually "first"
        barcode.drawOn(c, body_x - lq, barcode_bottom_y)

        # 2) Barcode number (just below the bars)
        c.setFont("Helvetica", 6)
        c.drawString(body_x, base_y + 14, barcode_value)

    # 3) Item number
    if item_number:
        c.setFont("Helvetica", 6)
        c.drawString(body_x, base_y + 8, f"Item #: {item_number}")

    # 4) Brand (bottom of the left stack)
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

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            col = count % COLUMNS
            row_num = (count // COLUMNS) % ROWS

            x = LEFT_MARGIN + col * LABEL_WIDTH
            y = PAGE_HEIGHT - TOP_MARGIN - (row_num + 1) * LABEL_HEIGHT

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
