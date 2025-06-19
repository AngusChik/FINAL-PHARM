from django.db import models
from django.utils import timezone

class Customer(models.Model):
   customer_id = models.AutoField(primary_key=True)
   name = models.CharField(max_length=100)

   def __str__(self):
       return self.name


class Category(models.Model):
   id = models.AutoField(primary_key=True)  # Explicit primary key
   name = models.CharField(max_length=100)

   def __str__(self):
       return self.name


# Inventory
class Product(models.Model):
    product_id = models.AutoField(primary_key=True)  # Explicit primary key
    name = models.CharField(max_length=200)
    brand = models.CharField(max_length=100)  # Renamed field
    item_number = models.CharField(max_length=50, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    barcode = models.CharField(max_length=30)
    quantity_in_stock = models.IntegerField(blank=True)  # Renamed field
    category = models.ForeignKey(Category, on_delete=models.CASCADE)  # ForeignKey field
    unit_size = models.CharField(max_length=50, blank=True)  # Unit Size field
    description = models.TextField(blank=True)  # Description field
    expiry_date = models.DateField(null=True, blank=True)  # Expiry Date field
    taxable = models.BooleanField() # Tax Field 

    stock_bought = models.IntegerField()
    stock_sold = models.IntegerField()
    stock_expired = models.IntegerField()
    price_per_unit = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
       return self.name
  

   
# Change 
class StockChange(models.Model):
    CHANGE_TYPE_CHOICES = [
        ('checkin', 'Stock Added'),
        ('checkout', 'Stock Removed'),
        ('expired', 'Expired'),
        ('error', 'Manual Adjustment'),
        ('return', 'Customer Return'), # change 
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='stock_changes')
    change_type = models.CharField(max_length=20, choices=CHANGE_TYPE_CHOICES)
    quantity = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)
    note = models.TextField(blank=True, null=True)  # Optional reason/comment

    def __str__(self):
        direction = "+" if self.quantity >= 0 else "-"
        return f"{self.product.name}: {direction}{abs(self.quantity)} ({self.get_change_type_display()})" 
    

### Purchase - Update inventory
class Order(models.Model):  # the order
    order_id = models.AutoField(primary_key=True)  # Explicit primary key
    total_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)  # Ensure default is set to 0
    order_date = models.DateTimeField(auto_now_add=True)
    submitted = models.BooleanField(default=False)  # Track whether the order is completed

    def __str__(self):
        return f"Order {self.order_id}"

class OrderDetail(models.Model):
   od_id = models.AutoField(primary_key=True)
   order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='details')
   product = models.ForeignKey(Product, on_delete=models.CASCADE)
   quantity = models.PositiveIntegerField()
   price = models.DecimalField(max_digits=10, decimal_places=2)
   order_date = models.DateTimeField(auto_now_add=True)

   def __str__(self):
       return f"{self.quantity} x {self.product.name}"


class RecentlyPurchasedProduct(models.Model):
   id = models.AutoField(primary_key=True)  # Auto-increment primary key without default
   product = models.ForeignKey(Product, on_delete=models.CASCADE)
   quantity = models.IntegerField(default=0)
   order_date = models.DateTimeField(auto_now_add=True)

   def __str__(self):
       return f"{self.product.name} ({self.quantity})"


class Item(models.Model):
   SIZE_CHOICES = [
       ('xxsmall', 'XX-Small'),
       ('xsmall', 'X-Small'),
       ('small', 'Small'),
       ('medium', 'Medium'),
       ('large', 'Large'),
       ('xlarge', 'X-Large'),
       ('xxlarge', 'XX-Large'),
       ('na', 'N/A'),
       ('Bathroom', 'Bathroom')
   ]
 
   SIDE_CHOICES = [
       ('left', 'Left'),
       ('right', 'Right'),
       ('na', 'N/A'),
       ('Bathroom', 'Bathroom')
   ]
 
   first_name = models.CharField(max_length=100)
   last_name = models.CharField(max_length=100)
   item_name = models.CharField(max_length=100)
   size = models.CharField(max_length=100, choices=SIZE_CHOICES)
   side = models.CharField(max_length=100, choices=SIDE_CHOICES)
   item_number = models.CharField(max_length=100)
   phone_number = models.CharField(max_length=15)
   is_checked = models.BooleanField(default=False)
 
   def __str__(self):
       return f"{self.first_name} {self.last_name} - {self.item_name}"