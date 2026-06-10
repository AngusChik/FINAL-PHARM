from django.conf import settings
from django.db import models
from django.utils import timezone
from django.db.models import Q

class Customer(models.Model):
   customer_id = models.AutoField(primary_key=True)
   name = models.CharField(max_length=100)

   def __str__(self):
       return self.name


class Category(models.Model):
   id = models.AutoField(primary_key=True)  # Explicit primary key
   name = models.CharField(max_length=100)
   low_stock_threshold = models.PositiveIntegerField(default=3)

   def __str__(self):
       return self.name


# Inventory
class Product(models.Model):
    product_id = models.AutoField(primary_key=True)  # Explicit primary key
    name = models.CharField(max_length=200)
    brand = models.CharField(max_length=100, blank=True)  # Renamed field
    item_number = models.CharField(max_length=50, blank=True, null=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    barcode = models.CharField(max_length=64, null=True, blank=True)
    quantity_in_stock = models.IntegerField(default=0)  # Renamed field
    category = models.ForeignKey('Category', on_delete=models.SET_NULL, null=True, blank=True)    
    previous_category = models.ForeignKey(
        'Category', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='revert_products'
    )
    unit_size = models.CharField(max_length=50, blank=True)  # Unit Size field
    description = models.TextField(blank=True)  # Description field
    expiry_date = models.DateField(null=True, blank=True)  # Expiry Date field
    taxable = models.BooleanField(default=True) # Tax Field 
    status = models.BooleanField(default=True)  # Active/Inactive status

    stock_bought = models.IntegerField(default = 0)
    stock_sold = models.IntegerField(default = 0)
    stock_expired = models.IntegerField(default = 0)
    stock_unfulfilled = models.IntegerField(default=0)  # Tracks missed sales due to stockouts

    price_per_unit = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True,default=None)

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)
    

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["barcode"],
                condition=Q(barcode__isnull=False),
                name="uniq_product_barcode_not_null",
            )
        ]

    indexes = [
        models.Index(fields=['barcode'], name='product_barcode_idx'),
        models.Index(fields=['name'], name='product_name_idx'),
        models.Index(fields=['status', 'quantity_in_stock'], name='product_stock_status_idx'),
        models.Index(fields=['category', 'status'], name='product_cat_status_idx'),
        models.Index(fields=['expiry_date'], name='product_expiry_idx'),
    ]

    def __str__(self):
       return self.name
  
    @classmethod
    def active(cls):
        return cls.objects.filter(status=True)

    def refresh_earliest_expiry(self):
        earliest = self.expiry_dates.order_by('expiry_date').values_list('expiry_date', flat=True).first()
        if self.expiry_date != earliest:
            self.expiry_date = earliest
            self.save(update_fields=['expiry_date'])


class ProductExpiryDate(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='expiry_dates')
    expiry_date = models.DateField()

    class Meta:
        ordering = ['expiry_date']
        indexes = [
            models.Index(fields=['product', 'expiry_date'], name='prodexpiry_prod_date_idx'),
        ]

    def __str__(self):
        return f"{self.product.name} — {self.expiry_date}"


class CheckinSession(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='checkin_sessions',
    )
    scanned_by = models.CharField(max_length=100, blank=True, default="")
    inventory_mode = models.BooleanField(default=False)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    reopened_at = models.DateTimeField(null=True, blank=True)
    note = models.TextField(blank=True)

    class Meta:
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['-started_at'], name='session_started_idx'),
            models.Index(fields=['user', '-started_at'], name='session_user_started_idx'),
        ]

    def __str__(self):
        status = "Active" if self.is_active else "Completed"
        return f"Session #{self.pk} — {status} ({self.started_at:%b %d %H:%M})"

    @property
    def is_active(self):
        return self.ended_at is None

    @property
    def is_reopened(self):
        return self.reopened_at is not None

    @property
    def duration(self):
        end = self.ended_at or timezone.now()
        return end - self.started_at

    @property
    def items_scanned(self):
        return self.stock_changes.filter(
            change_type__in=['checkin', 'checkin_delete1', 'error_add', 'error_subtract']
        ).count()


# Change
class StockChange(models.Model):
    CHANGE_TYPE_CHOICES = [
        ('checkin', 'Stock Added'),
        ('checkout', 'Stock Removed (Sale)'),
        ('checkout_unfulfilled', 'Unfulfilled Sale (Stockout)'),  # ✅ Already exists
        ('expired', 'Expired Stock'),
        ('error_add', 'Manual Addition'),
        ('error_subtract', 'Manual Adjustment'),
        ('checkin_delete1', 'Stock Removed via Delete Button'),
        ('deletion', 'Product Deletion'),  # ✅ ADD THIS
        ('return', 'Customer Return'),
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='stock_changes')
    session = models.ForeignKey(
        'CheckinSession', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='stock_changes',
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='stock_changes',
    )
    change_type = models.CharField(max_length=20, choices=CHANGE_TYPE_CHOICES)
    quantity = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)
    note = models.TextField(blank=True, null=True)  # Optional reason/comment

    def __str__(self):
        direction = "+" if self.quantity >= 0 else "-"
        return f"{self.product.name}: {direction}{abs(self.quantity)} ({self.get_change_type_display()})"


class LoginAudit(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='login_audits',
    )
    username = models.CharField(max_length=150)
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    success = models.BooleanField(default=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp'], name='loginaudit_ts_idx'),
            models.Index(fields=['user', '-timestamp'], name='loginaudit_user_ts_idx'),
        ]

    def __str__(self):
        status = "OK" if self.success else "FAIL"
        return f"{self.username} [{status}] @ {self.timestamp:%Y-%m-%d %H:%M}"


class UserAction(models.Model):
    ACTION_CHOICES = [
        # Original actions
        ('delete_product', 'Deleted Product'),
        ('delete_order', 'Deleted Order'),
        ('delete_all_orders', 'Deleted All Orders'),
        ('delete_recently_purchased', 'Deleted Recently Purchased'),
        ('delete_all_recently_purchased', 'Deleted All Recently Purchased'),
        ('bulk_delete_recently_purchased', 'Bulk Deleted Recently Purchased'),
        ('submit_order', 'Submitted Order'),
        ('add_product', 'Added New Product'),
        # Check-in Sessions
        ('start_session', 'Started Check-in Session'),
        ('end_session', 'Ended Check-in Session'),
        ('reopen_session', 'Reopened Check-in Session'),
        ('adjust_session_line', 'Adjusted Session Line'),
        ('remove_session_line', 'Removed Session Line'),
        ('delete_session', 'Deleted Check-in Session'),
        ('clear_session_history', 'Cleared Session History'),
        # Delivery
        ('delivery_checkin', 'Delivery Check-in'),
        ('delivery_checkout', 'Delivery Check-out'),
        ('delivery_undo_checkout', 'Delivery Undo Check-out'),
        ('delivery_clear_history', 'Delivery Cleared History'),
        # Product
        ('edit_product', 'Edited Product'),
        ('update_product_settings', 'Updated Product Settings'),
        ('revert_label_category', 'Reverted Label Categories'),
        # Other
        ('create_account', 'Created Account'),
        ('clear_label_queue', 'Cleared Label Queue'),
        # Item list
        ('delete_item_list', 'Deleted Item List Entry'),
        ('add_item_list', 'Added Item List Entry'),
        # Delivery single delete
        ('delivery_delete_record', 'Deleted Delivery Record'),
        # Stock operations
        ('cycle_count', 'Cycle Count Completed'),
        ('retire_expired', 'Retired Expired Stock'),
        # Label sessions
        ('print_labels', 'Printed Labels'),
        ('delete_label_session', 'Deleted Label Session'),
        ('regenerate_label_session', 'Regenerated Label Session'),
        ('clear_all_label_sessions', 'Cleared All Label Sessions'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='user_actions',
    )
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    target = models.CharField(max_length=200)
    detail = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp'], name='useraction_ts_idx'),
        ]

    def __str__(self):
        return f"{self.get_action_display()}: {self.target}"


### Purchase - Update inventory
class Order(models.Model):  # the order
    order_id = models.AutoField(primary_key=True)  # Explicit primary key
    total_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)  # Ensure default is set to 0
    order_date = models.DateTimeField(auto_now_add=True)
    submitted = models.BooleanField(default=False)  # Track whether the order is completed
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='orders',
    )

    def __str__(self):
        return f"Order {self.order_id}"
    
    @property
    def calculated_total(self):
        return sum(detail.line_total for detail in self.details.all())

class OrderDetail(models.Model):
   od_id = models.AutoField(primary_key=True)
   order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='details')
   product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
   product_name = models.CharField(max_length=200, default="Unknown Product")
   product_barcode = models.CharField(max_length=64, blank=True, default="")
   quantity = models.PositiveIntegerField()
   price = models.DecimalField(max_digits=10, decimal_places=2)
   order_date = models.DateTimeField(auto_now_add=True)

   def __str__(self):
        name = self.product.name if self.product else self.product_name
        return f"{self.quantity} x {name}"

   @property
   def line_total(self):
        return self.quantity * self.price

   @property
   def display_name(self):
        """Returns product name, falling back to stored name if product was deleted."""
        if self.product:
            return self.product.name
        return self.product_name

   @property
   def display_barcode(self):
        """Returns barcode, falling back to stored barcode if product was deleted."""
        if self.product:
            return self.product.barcode or ""
        return self.product_barcode
    
class RecentlyPurchasedProduct(models.Model):
   id = models.AutoField(primary_key=True)  # Auto-increment primary key without default
   product = models.ForeignKey(Product, on_delete=models.CASCADE)
   quantity = models.IntegerField(default=0)
   order_date = models.DateTimeField(auto_now_add=True)

   def __str__(self):
       return f"{self.product.name} ({self.quantity})"


class DeliveryCheckIn(models.Model):
    barcode        = models.CharField(max_length=64)
    first_name     = models.CharField(max_length=100)
    last_name      = models.CharField(max_length=100)
    comment        = models.CharField(max_length=255, blank=True, default='')
    checked_in_at  = models.DateTimeField(auto_now_add=True)
    checked_out_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-checked_in_at']
        indexes = [
            models.Index(fields=['barcode'],       name='delivery_barcode_idx'),
            models.Index(fields=['checked_in_at'], name='delivery_checkin_date_idx'),
        ]

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.barcode})"


class LabelQueueItem(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='label_queue_entries')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='label_queue_items',
    )
    qty = models.PositiveIntegerField(default=1)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-added_at']
        indexes = [
            models.Index(fields=['user', '-added_at'], name='labelqueue_user_added_idx'),
        ]

    def __str__(self):
        return f"{self.product.name} x{self.qty} (user={self.user_id})"


class LabelSession(models.Model):
    """Snapshot of a label print run — created each time Generate PDF is clicked."""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='label_sessions',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    label_count = models.PositiveIntegerField(default=0)
    note = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at'], name='labelsession_user_created_idx'),
        ]

    def __str__(self):
        return f"Label Session #{self.pk} — {self.label_count} labels ({self.created_at:%b %d %H:%M})"


class LabelSessionItem(models.Model):
    """Individual label snapshot — stores product data at time of printing."""
    session = models.ForeignKey(LabelSession, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    product_name = models.CharField(max_length=200)
    product_barcode = models.CharField(max_length=64, blank=True)
    product_price = models.DecimalField(max_digits=10, decimal_places=2)
    product_brand = models.CharField(max_length=100, blank=True)
    product_item_number = models.CharField(max_length=50, blank=True)
    qty = models.PositiveIntegerField(default=1)

    class Meta:
        ordering = ['pk']

    def __str__(self):
        return f"{self.product_name} x{self.qty}"


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


class UserSession(models.Model):
    """Tracks active Django sessions per user for concurrent session limiting."""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='user_sessions',
    )
    session_key = models.CharField(max_length=40, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['user', 'created_at'], name='usersession_user_created_idx'),
        ]

    def __str__(self):
        return f"{self.user} — session {self.session_key[:8]}…"