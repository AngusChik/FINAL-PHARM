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
    stock_giveaway = models.IntegerField(default=0)  # Cumulative units given away via PU terminals
    stock_deleted = models.IntegerField(default=0)  # Units lost when a product is deleted (shrinkage/discontinuation, not expiry)

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

    @property
    def counted_units(self):
        """Total units tallied so far in an inventory-count session (buffer)."""
        return self.count_lines.aggregate(total=models.Sum('counted_qty'))['total'] or 0


class InventoryCountLine(models.Model):
    """Per-session tally for Inventory Count Mode.

    Scanning during an inventory-count session increments `counted_qty` here
    instead of touching live `Product.quantity_in_stock`. At reconcile the
    counted value is applied to the product (in-scope but never scanned → 0) and
    the expected→counted variance is recorded as a StockChange. This keeps live
    stock correct during the count and makes it interruption-safe.
    """
    session = models.ForeignKey(
        CheckinSession, on_delete=models.CASCADE, related_name='count_lines',
    )
    product = models.ForeignKey(
        Product, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='inventory_count_lines',
    )
    # Snapshots so the line survives product deletion.
    product_name = models.CharField(max_length=200, blank=True, default="")
    product_barcode = models.CharField(max_length=64, blank=True, default="")
    # Live stock captured when the product was added to the count's scope.
    expected_qty = models.IntegerField(default=0)
    # Running tally of units physically scanned/counted in this session.
    counted_qty = models.IntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('session', 'product')
        ordering = ['product_name']

    def __str__(self):
        return f"{self.product_name or self.product_id}: counted {self.counted_qty} (exp {self.expected_qty})"

    @property
    def variance(self):
        return self.counted_qty - self.expected_qty


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
        ('deletion', 'Product Deletion'),
        ('giveaway', 'No Sale (Terminal)'),  # PU checkout terminal — no-sale removal
        ('giveaway_unfulfilled', 'Unfulfilled No Sale'),
    ]

    # SET_NULL (not CASCADE) so deleting a product never erases its audit trail.
    # product_name / product_barcode snapshot the product's identity at write time
    # so the ledger stays readable after the product row is gone.
    product = models.ForeignKey(
        Product, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='stock_changes',
    )
    product_name = models.CharField(max_length=200, blank=True, default="")
    product_barcode = models.CharField(max_length=64, blank=True, default="")
    session = models.ForeignKey(
        'CheckinSession', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='stock_changes',
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='stock_changes',
    )
    change_type = models.CharField(max_length=30, choices=CHANGE_TYPE_CHOICES)
    quantity = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)
    note = models.TextField(blank=True, null=True)  # Optional reason/comment

    @property
    def display_name(self):
        """Product name, falling back to the snapshot when the product was deleted."""
        if self.product:
            return self.product.name
        return self.product_name or "(deleted product)"

    @property
    def display_barcode(self):
        """Barcode, falling back to the snapshot when the product was deleted."""
        if self.product:
            return self.product.barcode or ""
        return self.product_barcode or ""

    def __str__(self):
        direction = "+" if self.quantity >= 0 else "-"
        return f"{self.display_name}: {direction}{abs(self.quantity)} ({self.get_change_type_display()})"


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
        ('passkey_unlock', 'Unlocked Admin Passkey'),
        ('passkey_lockout', 'Passkey Attempt Lockout'),
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
        # PU Checkout
        ('checkout_submit', 'Submitted PU Checkout'),
        ('checkout_new', 'Started New PU Checkout'),
        # Ordering sheet
        ('ordering_status_update', 'Updated Ordering Sheet Status'),
        ('ordering_delete', 'Removed Ordering Sheet Entry'),
        # Session management
        ('boot_session', 'Logged Off User'),
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
    # Seniors discount: 10% off the pre-tax subtotal, toggled on the purchase page.
    seniors_discount = models.BooleanField(default=False)
    # In-progress cart for an unsubmitted order, so it survives logout/login.
    draft_cart = models.JSONField(default=dict, blank=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='orders',
    )
    # Soft delete: a "deleted" order is hidden from the order list but its data
    # (OrderDetail lines, StockChange ledger, stock counters) is preserved so
    # reports and reorder predictions keep working.
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='deleted_orders',
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
   # Product's earliest expiry date captured at submit time, so "expired when sold"
   # stays accurate even if the product's expiry data changes later. Null for lines
   # created before this was tracked, or for products with no expiry at sale.
   expiry_at_sale = models.DateField(null=True, blank=True)
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


### PU Checkout — durable, per-user checkout classified separately from admin Orders
class CheckoutOrder(models.Model):
    STATUS_DRAFT = 'draft'
    STATUS_SUBMITTED = 'submitted'
    STATUS_CHOICES = [
        (STATUS_DRAFT, 'Draft'),
        (STATUS_SUBMITTED, 'Submitted'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='checkout_orders',
    )
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=STATUS_DRAFT)
    # Session that currently "owns" the active draft (drives the concurrency warning).
    active_session_key = models.CharField(max_length=40, blank=True, default="")
    subtotal = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    tax = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    submitted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        # NOTE: multiple draft checkouts per user are allowed — each checkout
        # terminal (browser session) keeps its own active session at a time.
        indexes = [
            models.Index(fields=['user', 'status'], name='checkout_user_status_idx'),
            models.Index(fields=['-created_at'], name='checkout_created_idx'),
        ]

    def __str__(self):
        return f"PU Checkout #{self.pk} — {self.get_status_display()} ({self.user})"

    @property
    def item_count(self):
        return sum(i.quantity for i in self.items.all())


class CheckoutOrderItem(models.Model):
    checkout = models.ForeignKey(CheckoutOrder, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    product_name = models.CharField(max_length=200)          # snapshot at add time
    product_barcode = models.CharField(max_length=64, blank=True, default="")
    price = models.DecimalField(max_digits=10, decimal_places=2)  # snapshot at add time
    taxable = models.BooleanField(default=True)              # snapshot, for tax calc
    quantity = models.PositiveIntegerField(default=0)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['pk']
        unique_together = [('checkout', 'product')]          # one line per product; increment quantity

    def __str__(self):
        return f"{self.quantity} x {self.product_name}"

    @property
    def line_total(self):
        return self.price * self.quantity

    @property
    def display_name(self):
        if self.product:
            return self.product.name
        return self.product_name

    @property
    def display_barcode(self):
        if self.product:
            return self.product.barcode or ""
        return self.product_barcode


class PagePresence(models.Model):
    """Tracks which single computer (browser session) currently 'holds' a guarded
    page, so only one computer can be on a given page at a time. Refreshed by a
    heartbeat; a holder is considered gone once last_seen is older than the TTL."""
    page = models.CharField(max_length=200, unique=True)   # the page key (URL path)
    session_key = models.CharField(max_length=40)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='page_presences',
    )
    ip_address = models.CharField(max_length=45, blank=True, default="")
    user_agent = models.CharField(max_length=300, blank=True, default="")
    last_seen = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.page} → {self.session_key}"


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
    # How this computer signed in. A "phone" session is one created via the
    # dashboard "Connect Phone" QR flow (see ConnectPhone / CustomLoginView);
    # it gets a shorter 2-hour expiry (settings.PHONE_SESSION_AGE) and is shown
    # distinctly on the Active Sessions page. Everything else is a "computer".
    DEVICE_COMPUTER = 'computer'
    DEVICE_PHONE = 'phone'
    DEVICE_CHOICES = [
        (DEVICE_COMPUTER, 'Computer'),
        (DEVICE_PHONE, 'Phone'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='user_sessions',
    )
    session_key = models.CharField(max_length=40, unique=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    # Nullable so a migrate-before-restart deploy window can't 500 on insert.
    user_agent = models.CharField(max_length=300, blank=True, null=True, default="")
    device_type = models.CharField(
        max_length=10, choices=DEVICE_CHOICES, default=DEVICE_COMPUTER,
    )
    # The URL path this computer is currently viewing — powers the live nav
    # "who's on which screen" bubble (refreshed by a client heartbeat).
    current_path = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['user', 'created_at'], name='usersession_user_created_idx'),
        ]

    def __str__(self):
        return f"{self.user} — session {self.session_key[:8]}…"


class OrderingSheetEntry(models.Model):
    """A line on the daily ordering sheet.

    Any logged-in user (PU or GINA) can add an entry to flag an item that
    needs ordering. Only GINA (the staff account) may change its Status after
    submission — that's enforced in OrderingSheetView.
    """
    REASON_STOCK = 'stock'
    REASON_BASKET = 'basket'
    REASON_EXPIRING = 'expiring'
    REASON_BLISTER = 'blister'
    REASON_CHOICES = [
        (REASON_STOCK, 'Order for stock'),
        (REASON_BASKET, 'Order for basket'),
        (REASON_EXPIRING, 'Expiring'),
        (REASON_BLISTER, 'Order for BLISTER'),
    ]

    # An entry is either a drug (the original use) or an OTC product. OTC rows
    # capture Side / Phone instead of reasoning / quantities / urgency.
    ENTRY_DRUG = 'drug'
    ENTRY_OTC = 'otc'
    ENTRY_TYPE_CHOICES = [
        (ENTRY_DRUG, 'Drug'),
        (ENTRY_OTC, 'OTC Product'),
    ]

    SIDE_LEFT = 'left'
    SIDE_RIGHT = 'right'
    SIDE_NA = 'na'
    SIDE_CHOICES = [
        (SIDE_LEFT, 'Left'),
        (SIDE_RIGHT, 'Right'),
        (SIDE_NA, 'N/A'),
    ]

    URGENCY_LOW = 'low'
    URGENCY_MEDIUM = 'medium'
    URGENCY_HIGH = 'high'
    URGENCY_NA = 'na'
    URGENCY_CHOICES = [
        (URGENCY_HIGH, 'High (TOMORROW PU)'),
        (URGENCY_MEDIUM, 'Medium (4 days PU)'),
        (URGENCY_LOW, 'Low (1 week PU)'),
        (URGENCY_NA, 'N/A'),
    ]

    STATUS_PENDING = 'pending'
    STATUS_BACKORDERED = 'backordered'
    STATUS_ORDERED = 'ordered'
    STATUS_NOT_FOR_SALE = 'not_for_sale'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_BACKORDERED, 'Back-Ordered'),
        (STATUS_ORDERED, 'Ordered'),
        (STATUS_NOT_FOR_SALE, 'Not for Sale (Consult Pharmacist)'),
    ]
    # The values GINA may set manually. Pending is the un-actioned default and
    # is never chosen by hand.
    GINA_STATUS_CHOICES = [STATUS_BACKORDERED, STATUS_ORDERED, STATUS_NOT_FOR_SALE]

    name = models.CharField(max_length=200)  # the drug name, or the OTC product name
    entry_type = models.CharField(max_length=10, choices=ENTRY_TYPE_CHOICES, default=ENTRY_DRUG)
    reasoning = models.CharField(max_length=20, choices=REASON_CHOICES, blank=True, default="")
    quantity_needed = models.CharField(max_length=50, blank=True, default="")
    quantity_remaining = models.CharField(max_length=50, blank=True)
    patient_name = models.CharField(max_length=200, blank=True, default="")
    # OTC-only: which side, and a contact phone number for the patient.
    side = models.CharField(max_length=10, choices=SIDE_CHOICES, blank=True, default="")
    phone_number = models.CharField(max_length=20, blank=True, default="")
    urgency = models.CharField(max_length=10, choices=URGENCY_CHOICES, default=URGENCY_LOW)
    initials = models.CharField(max_length=20)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    # Free-text note GINA can attach when marking a row "Ordered" (qty ordered, supplier, ETA…).
    order_note = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)  # the auto-filled submission date
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='ordering_entries',
    )
    status_updated_at = models.DateTimeField(null=True, blank=True)
    status_updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='ordering_status_edits',
    )

    # Soft delete, mirroring Order: GINA can clear finished rows without losing data.
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='ordering_deletions',
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['is_deleted', 'status'])]

    def __str__(self):
        return f"{self.name} ({self.get_status_display()})"

    @property
    def is_out(self):
        """True when the free-text 'quantity remaining' indicates zero on hand."""
        raw = (self.quantity_remaining or '').strip().lower()
        if not raw:
            return False
        if raw in ('0', 'none', 'nil', 'out', 'n/a', 'na', 'zero'):
            return True
        import re
        nums = re.findall(r'\d+', raw)
        return bool(nums) and int(nums[0]) == 0

    @property
    def is_low(self):
        """True when the free-text 'quantity remaining' indicates a single unit left."""
        raw = (self.quantity_remaining or '').strip().lower()
        if not raw or self.is_out:
            return False
        if raw in ('1', 'one'):
            return True
        import re
        nums = re.findall(r'\d+', raw)
        return bool(nums) and int(nums[0]) == 1

class DailyReportArchive(models.Model):
    """A stored snapshot (rendered PDF) of a day's end-of-day report.

    One row per day (upserted). Rows older than RETENTION_DAYS are pruned
    whenever a new snapshot is saved, so the archive self-cleans at ~30 days.
    """
    RETENTION_DAYS = 30

    report_date = models.DateField(unique=True)
    pdf = models.BinaryField()
    summary = models.CharField(max_length=200, blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-report_date']

    def __str__(self):
        return f"Daily report {self.report_date}"
