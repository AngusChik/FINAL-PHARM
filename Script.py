import csv
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.barcode import code128

# ---------- Stock + margins ----------
LABEL_WIDTH  = 3.00 * inch    # 216 pt
LABEL_HEIGHT = 1.25 * inch    # 90 pt

PAGE_WIDTH, PAGE_HEIGHT = landscape(letter)

LEFT_MARGIN   = 1.0 * inch    # 72 pt
RIGHT_MARGIN  = 1.0 * inch    # 72 pt
TOP_MARGIN    = 0.5 * inch    # 36 pt
BOTTOM_MARGIN = 0.5 * inch    # 36 pt

# Grid is exact: 3 columns × 6 rows
COLUMNS = 3
ROWS    = 6
LABELS_PER_PAGE = COLUMNS * ROWS

# Inner padding for text inside each label
LEFT_PADDING = 8
RIGHT_PADDING = 8
TOP_PADDING = 5
BOTTOM_PADDING = 5

CSV_FILE = 'current.csv'
OUTPUT_PDF = 'labels.pdf'


def wrap_text(text, font_name, font_size, max_width):
    """Split text into wrapped lines within max width."""
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
    """Draw a single label inside the given cell (x,y = bottom-left)."""
    name = (data.get('name') or '').strip().title()
    item_number = (data.get('item_number') or '').strip()
    barcode_value = (data.get('barcode') or '').strip()

    # Price
    try:
        price_val = float(data.get('price', 0) or 0)
    except:
        price_val = 0.0
    price = f"${price_val:.2f}"

    # === Name (top-centered) ===
    name_font = "Helvetica-Bold"
    name_size = 11
    max_text_width = LABEL_WIDTH - LEFT_PADDING - RIGHT_PADDING
    lines = wrap_text(name, name_font, name_size, max_text_width)  # no limit

    NAME_BLOCK_TOP = LABEL_HEIGHT - 10
    for i, line in enumerate(lines):
        text_y = y + NAME_BLOCK_TOP - (i * (name_size + 2))
        c.setFont(name_font, name_size)
        c.drawCentredString(x + LABEL_WIDTH / 2, text_y, line)

    # === Bottom-left block ===
    text_base_y = y + BOTTOM_PADDING
    body_x = x + LEFT_PADDING

    if item_number:
        c.setFont("Helvetica", 8)
        c.drawString(body_x + 6, text_base_y + 30, f"Item #: {item_number}")

    if barcode_value:
        c.setFont("Helvetica", 8)
        c.drawString(body_x + 6, text_base_y + 22, barcode_value)
        try:
            barcode = code128.Code128(barcode_value, barHeight=20, barWidth=0.9)
            barcode.drawOn(c, body_x - 12, text_base_y)
        except Exception as e:
            print(f"⚠️ Error drawing barcode for {barcode_value}: {e}")

    # === Price (bottom-right) ===
    c.setFont("Helvetica-Bold", 24)
    c.drawRightString(x + LABEL_WIDTH - RIGHT_PADDING, text_base_y, price)



def generate_labels(csv_file, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=landscape(letter))
    count = 0

    # Draw perforation (cut) lines for alignment
    c.setStrokeColorRGB(0.7, 0.7, 0.7)  # light gray
    c.setLineWidth(0.3)

    # Vertical lines
    for col in range(COLUMNS + 1):
        x = LEFT_MARGIN + col * LABEL_WIDTH
        c.line(x, BOTTOM_MARGIN, x, PAGE_HEIGHT - TOP_MARGIN)

    # Horizontal lines
    for row in range(ROWS + 1):
        y = BOTTOM_MARGIN + row * LABEL_HEIGHT
        c.line(LEFT_MARGIN, y, PAGE_WIDTH - RIGHT_MARGIN, y)

    c.setStrokeColorRGB(0, 0, 0)  # reset back to black

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            col = count % COLUMNS
            row_num = (count // COLUMNS) % ROWS

            # Perfect grid placement
            x = LEFT_MARGIN + col * LABEL_WIDTH
            y = PAGE_HEIGHT - TOP_MARGIN - row_num * LABEL_HEIGHT - LABEL_HEIGHT

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
