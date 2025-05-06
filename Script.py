import csv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def generate_price_labels(csv_file, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    page_width, page_height = letter

    # Define margins and label sizes
    left_margin = 30
    right_margin = 30
    column_width = (page_width - (left_margin + right_margin)) / 3  # Three columns
    y_start = page_height - 50
    bottom_margin = 30  # Adjusted bottom margin
    label_spacing = 10  # Spacing between labels
    line_spacing = 12  # Spacing between lines in text

    x_positions = [
        left_margin,
        left_margin + column_width,
        left_margin + 2 * column_width
    ]  # Three column positions
    y = y_start
    column = 0  # Start with the first column

    # Open the file with proper encoding
    with open(csv_file, newline='', encoding='latin1') as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            # Check if a new page starts
            if row_index == 0 or (y == y_start and column == 0):
                # Reset font and spacing explicitly for the first label on each page
                c.setFont("Helvetica", 10)
                y = y_start

            name = row.get("name", "")
            item_number = row.get("itemNumber", "")  # Retrieve item number
            price = row.get("Price", "")

            # Calculate dynamic line spacing and wrap text
            c.setFont("Helvetica", 10)  # Set font for name
            lines = split_text(name, column_width - 30, c)
            extra_spacing = line_spacing if price == "0" else 0  # Add extra spacing if price is 0
            total_height = (len(lines) + 1) * line_spacing + extra_spacing + (30 if price != "0" else 10)

            # If the current label doesn't fit the remaining space, move to the next column or page
            if y - total_height < bottom_margin:  # Use adjusted bottom margin
                column += 1
                if column > 2:  # If all three columns are full, start a new page
                    c.showPage()
                    column = 0
                    y = y_start
                    c.setFont("Helvetica", 11)  # Reset font for new page
                else:  # Move to the next column
                    y = y_start

            # Draw product name
            for i, line in enumerate(lines):
                c.drawString(x_positions[column], y - (i * line_spacing), line)

            # Draw bold item number below the name
            c.setFont("Helvetica-Bold", 10)  # Bold font for item number
            c.drawString(x_positions[column], y - (len(lines) * line_spacing) - 3, f"Item Number: {item_number}")

            # Add extra line spacing if price is 0
            if price == "0":
                y -= line_spacing  # Add spacing as an empty line
            else:
                # Draw "Price:" in regular font
                c.setFont("Helvetica", 12)
                c.drawString(x_positions[column], y - (len(lines) * line_spacing) - 20, "Price: ")

                # Draw bold and larger price
                c.setFont("Helvetica-Bold", 20)  # Bold, larger font for price
                price_x_offset = c.stringWidth("Price: ")
                c.drawString(x_positions[column] + price_x_offset, y - (len(lines) * line_spacing) - 20, f"${price}")

            # Move to the next label
            y -= total_height + label_spacing

    c.save()
    print(f"Price labels saved to {output_pdf}")

def split_text(text, max_width, canvas_obj):
    """
    Splits text into multiple lines based on the maximum width, ensuring no line has
    only one word unless it's the first word.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if canvas_obj.stringWidth(test_line) < max_width:
            current_line = test_line
        else:
            # Ensure at least two words in the current line if possible
            if current_line:
                if len(current_line.split()) == 1 and lines:
                    # Push last word from the previous line to the current line
                    last_line = lines.pop()
                    last_word = last_line.split()[-1]
                    last_line = " ".join(last_line.split()[:-1])
                    if last_line:
                        lines.append(last_line)
                    current_line = f"{last_word} {current_line}".strip()
                lines.append(current_line)
            current_line = word

    if current_line:
        # Handle the last line: ensure no single-word line at the end
        if len(current_line.split()) == 1 and lines:
            last_line = lines.pop()
            last_word = last_line.split()[-1]
            last_line = " ".join(last_line.split()[:-1])
            if last_line:
                lines.append(last_line)
            current_line = f"{last_word} {current_line}".strip()
        lines.append(current_line)

    return lines

# Usage
csv_file = "temp.csv"  # Replace with your CSV file name
output_pdf = "temp.pdf"  # Output file
generate_price_labels(csv_file, output_pdf)
