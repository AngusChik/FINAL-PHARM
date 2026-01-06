import csv
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'inventory.settings')
django.setup()

from app.models import Product, Category  # Replace 'app' with your actual app name

def import_products(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Retrieve the category by name
                category = Category.objects.get(name=row['category_id'])
            except Category.DoesNotExist:
                print(f"Category '{row['category_id']}' does not exist. Skipping product: {row['name']}")
                continue

            # Create the product
            Product.objects.create(
                name=row['name'],
                brand=row['brand'],
                item_number=row['item_number'],
                price=row['price'],
                barcode=row['barcode'],
                quantity_in_stock=row['quantity_in_stock'],
                category=category,
                unit_size=row['unit_size'],
                description=row['description'],
            )

if __name__ == '__main__':
    product_csv_file_path = 'updated.csv'
    import_products(product_csv_file_path)
