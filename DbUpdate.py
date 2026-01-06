import csv
from itertools import product
import os
import django
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'inventory.settings')  # 🔁 Replace with your project name
django.setup()

from app.models import Product  # 🔁 Replace 'app' with your actual app name

# Optional: handle boolean parsing
def parse_bool(value):
    return value.strip().lower() == 't'

# Load and import products
with open('current.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            expiry = row['expiry_date']
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date() if expiry else None

            Product.objects.update_or_create(
                product_id=row['product_id'],
                defaults={
                    'name': row['name'],
                    'brand': row['brand'],
                    'item_number': row['item_number'],
                    'barcode': row['barcode'],
                    'price': float(row['price']) if row['price'] else 0.0,
                    'quantity_in_stock': int(row['quantity_in_stock']) if row['quantity_in_stock'] else 0,
                    'unit_size': row['unit_size'],
                    'description': row['description'],
                    'expiry_date': expiry_date,
                    'taxable': parse_bool(row['taxable']),
                    'stock_bought': int(row['stock_bought']) if row['stock_bought'] else 0,
                    'stock_sold': int(row['stock_sold']) if row['stock_sold'] else 0,
                    'stock_expired': int(row['stock_expired']) if row['stock_expired'] else 0,
                    'price_per_unit': float(row['price_per_unit']) if row['price_per_unit'] else 0.0,
                    'category_id': int(row['category_id']) if row['category_id'] else None,
                }
            )
            print(product)
        except Exception as e:
            print(f"⚠️ Error importing product_id {row['product_id']}: {e}")

print("✅ Import completed.")
