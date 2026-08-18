from decimal import Decimal

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone
from django.db.models import F, Q
from django.db.models.functions import Lower


def normalize_barcode_key(value):
    """Return the durable comparison key used for barcode uniqueness.

    Numeric UPC/EAN values are tolerant of spaces, dashes, and leading zeroes.
    Supplier alphanumeric codes retain their letters and compare case-insensitively.
    """
    compact = ''.join(
        ch for ch in str(value or '').strip().upper()
        if not ch.isspace() and ch != '-'
    )
    if not compact:
        return None
    if compact.isdigit():
        return compact.lstrip('0') or '0'
    return compact


class ActiveProductManager(models.Manager):
    """Default product manager: operational pages never show archived rows."""

    def get_queryset(self):
        return super().get_queryset().filter(archived_at__isnull=True)

class Customer(models.Model):
   customer_id = models.AutoField(primary_key=True)
   name = models.CharField(max_length=100)

   def __str__(self):
       return self.name


class Category(models.Model):
   id = models.AutoField(primary_key=True)  # Explicit primary key
   name = models.CharField(max_length=100)
   low_stock_threshold = models.PositiveIntegerField(default=3)

   class Meta:
       ordering = ['name']  # categories list alphabetically everywhere by default
       constraints = [
           models.UniqueConstraint(
               Lower('name'), name='uniq_category_name_casefold',
           ),
       ]

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
    normalized_barcode = models.CharField(
        max_length=64, null=True, blank=True, unique=True, editable=False,
    )
    archived_at = models.DateTimeField(null=True, blank=True)
    archived_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='archived_products',
    )
    archive_reason = models.CharField(max_length=255, blank=True, default='')
    status_before_archive = models.BooleanField(default=True)
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

    objects = ActiveProductManager()
    all_objects = models.Manager()

    class Meta:
        default_manager_name = 'objects'
        base_manager_name = 'all_objects'
        constraints = [
            models.UniqueConstraint(
                fields=["barcode"],
                condition=Q(barcode__isnull=False),
                name="uniq_product_barcode_not_null",
            ),
            models.CheckConstraint(
                condition=Q(quantity_in_stock__gte=0),
                name='product_stock_nonnegative',
            ),
            models.CheckConstraint(
                condition=Q(price__gte=0),
                name='product_price_nonnegative',
            ),
            models.CheckConstraint(
                condition=Q(price_per_unit__isnull=True) | Q(price_per_unit__gte=0),
                name='product_cost_nonnegative',
            ),
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

    def save(self, *args, **kwargs):
        self.normalized_barcode = normalize_barcode_key(self.barcode)
        update_fields = kwargs.get('update_fields')
        if update_fields is not None and 'barcode' in update_fields:
            kwargs['update_fields'] = set(update_fields) | {'normalized_barcode'}
        super().save(*args, **kwargs)
  
    @classmethod
    def active(cls):
        return cls.objects.filter(status=True)

    def refresh_earliest_expiry(self):
        earliest = self.expiry_dates.order_by('expiry_date').values_list('expiry_date', flat=True).first()
        if self.expiry_date != earliest:
            self.expiry_date = earliest
            self.save(update_fields=['expiry_date'])

    @property
    def lot_numbers(self):
        return list(
            self.lots.filter(archived_at__isnull=True)
            .exclude(lot_number=ProductLot.UNASSIGNED)
            .values_list('lot_number', flat=True)
        )


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


class ProductLot(models.Model):
    """A quantity-bearing product lot used for receiving and FEFO deduction."""

    UNASSIGNED = 'UNASSIGNED'

    product = models.ForeignKey(
        Product, on_delete=models.CASCADE, related_name='lots',
    )
    lot_number = models.CharField(max_length=64, default=UNASSIGNED)
    expiry_date = models.DateField(null=True, blank=True)
    quantity_on_hand = models.PositiveIntegerField(default=0)
    received_at = models.DateTimeField(default=timezone.now)
    checkin_session = models.ForeignKey(
        'CheckinSession', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='received_lots',
    )
    notes = models.CharField(max_length=255, blank=True, default='')
    archived_at = models.DateTimeField(null=True, blank=True)
    archived_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='archived_product_lots',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['expiry_date', 'lot_number', 'pk']
        constraints = [
            models.UniqueConstraint(
                fields=['product', 'lot_number', 'expiry_date'],
                name='productlot_identity_uniq',
                nulls_distinct=False,
            ),
            models.CheckConstraint(
                condition=Q(quantity_on_hand__gte=0),
                name='productlot_qty_nonnegative',
            ),
        ]
        indexes = [
            models.Index(
                fields=['product', 'archived_at', 'expiry_date'],
                name='productlot_fefo_idx',
            ),
            models.Index(fields=['lot_number'], name='productlot_number_idx'),
        ]

    def save(self, *args, **kwargs):
        self.lot_number = (self.lot_number or self.UNASSIGNED).strip().upper()
        super().save(*args, **kwargs)

    def __str__(self):
        expiry = self.expiry_date.isoformat() if self.expiry_date else 'no expiry'
        return f'{self.product.name} / {self.lot_number} / {expiry} ({self.quantity_on_hand})'


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
        ('restoration', 'Product Restored'),
        ('giveaway', 'No Sale (Terminal)'),  # PU checkout terminal — no-sale removal
        ('giveaway_unfulfilled', 'Unfulfilled No Sale'),
        ('return', 'Transaction Return — Restocked'),
        ('return_no_restock', 'Transaction Return — Not Restocked'),
        ('void', 'Transaction Void'),
        ('correction_undo', 'Transaction Void Undone'),
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
    order_detail = models.ForeignKey(
        'OrderDetail', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='stock_changes',
    )
    checkout_item = models.ForeignKey(
        'CheckoutOrderItem', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='stock_changes',
    )
    correction_line = models.ForeignKey(
        'TransactionCorrectionLine', on_delete=models.SET_NULL,
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
        ('archive_product', 'Moved Product to Recovery'),
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
        ('ordering_edit', 'Edited Ordering Sheet Entry'),
        # Durable corrections, supplier tracking, and recovery
        ('transaction_correction', 'Corrected Transaction'),
        ('transaction_correction_undo', 'Undid Transaction Void'),
        ('supplier_order_create', 'Created Supplier Order Tracking'),
        ('supplier_order_update', 'Updated Supplier Order Tracking'),
        ('supplier_order_archive', 'Moved Supplier Order to Recovery'),
        ('restore_archived_record', 'Restored Archived Record'),
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
    SNAPSHOT_CAPTURED = 'captured'
    SNAPSHOT_BACKFILLED = 'backfilled'
    SNAPSHOT_SOURCE_CHOICES = [
        (SNAPSHOT_CAPTURED, 'Captured at sale'),
        (SNAPSHOT_BACKFILLED, 'Backfilled from available history'),
    ]

    order_id = models.AutoField(primary_key=True)  # Explicit primary key
    # Finalized together at submission. These fields are the authoritative
    # transaction snapshot and must not be recomputed from mutable products.
    subtotal = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    discount_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    tax = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    tax_rate = models.DecimalField(
        max_digits=6, decimal_places=4, default=Decimal('0.1300'),
    )
    total_price = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    financial_snapshot_source = models.CharField(
        max_length=10, choices=SNAPSHOT_SOURCE_CHOICES, blank=True, default='',
    )
    order_date = models.DateTimeField(auto_now_add=True)
    submitted = models.BooleanField(default=False)  # Track whether the order is completed
    # Seniors discount: 10% off the pre-tax subtotal, toggled on the purchase page.
    seniors_discount = models.BooleanField(default=False)
    # In-progress cart for an unsubmitted order, so it survives logout/login.
    draft_cart = models.JSONField(default=dict, blank=True)
    # Server-owned auto-submit deadline for this draft. Keeping the deadline on
    # the order makes the countdown follow the user across computers and leaves
    # useful timing data behind for later workflow analysis.
    draft_expires_at = models.DateTimeField(null=True, blank=True)
    last_timer_reset_at = models.DateTimeField(null=True, blank=True)
    timer_reset_count = models.PositiveIntegerField(default=0)
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
   # Exact sale-time values for future reporting. Nullable means a legacy value
   # could not be recovered during the historical backfill.
   taxable_at_sale = models.BooleanField(null=True, blank=True)
   cost_per_unit_at_sale = models.DecimalField(
       max_digits=10, decimal_places=2, null=True, blank=True,
   )
   # Product's earliest expiry date captured at submit time, so "expired when sold"
   # stays accurate even if the product's expiry data changes later. Null for lines
   # created before this was tracked, or for products with no expiry at sale.
   expiry_at_sale = models.DateField(null=True, blank=True)
   order_date = models.DateTimeField(auto_now_add=True)

   def __str__(self):
        return f"{self.quantity} x {self.product_name}"

   @property
   def line_total(self):
        return self.quantity * self.price

   @property
   def display_name(self):
        """The immutable name captured when this transaction was submitted."""
        return self.product_name

   @property
   def display_barcode(self):
        """The immutable barcode captured when this transaction was submitted."""
        return self.product_barcode
    
class RecentlyPurchasedProduct(models.Model):
   id = models.AutoField(primary_key=True)  # Auto-increment primary key without default
   product = models.ForeignKey(Product, on_delete=models.CASCADE)
   quantity = models.IntegerField(default=0)
   order_date = models.DateTimeField(auto_now_add=True)
   archived_at = models.DateTimeField(null=True, blank=True)
   archived_by = models.ForeignKey(
       settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
       null=True, blank=True, related_name='archived_recent_purchases',
   )
   archive_reason = models.CharField(max_length=255, blank=True, default='')

   class Meta:
       constraints = [
           models.UniqueConstraint(
               fields=['product'], condition=Q(archived_at__isnull=True),
               name='recentpurchase_one_active_product',
           ),
       ]

   def __str__(self):
       return f"{self.product.name} ({self.quantity})"


class SupplierOrderPlan(models.Model):
    """Durable multi-distributor ordering plan created from Recently Purchased."""

    STATUS_PLANNED = 'planned'
    STATUS_RUNNING = 'running'
    STATUS_COMPLETED = 'completed'
    STATUS_CANCELLED = 'cancelled'
    STATUS_ERROR = 'error'
    STATUS_CHOICES = [
        (STATUS_PLANNED, 'Planned'),
        (STATUS_RUNNING, 'Running'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_CANCELLED, 'Cancelled'),
        (STATUS_ERROR, 'Error'),
    ]

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='supplier_order_plans',
    )
    vendor_sequence = models.JSONField(default=list)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default=STATUS_PLANNED)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', '-created_at'], name='supplierplan_status_idx'),
            models.Index(fields=['created_by', '-created_at'], name='supplierplan_user_idx'),
        ]

    def __str__(self):
        return f"Supplier plan #{self.pk} ({self.get_status_display()})"


class SupplierOrderPlanItem(models.Model):
    """Original quantity-adjusted item list saved before any supplier is tried."""

    plan = models.ForeignKey(SupplierOrderPlan, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    product_name = models.CharField(max_length=200)
    barcode = models.CharField(max_length=64, blank=True, default='')
    quantity = models.PositiveIntegerField(default=1)
    position = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['position', 'pk']
        constraints = [
            models.UniqueConstraint(fields=['plan', 'position'], name='supplierplanitem_position_uniq'),
        ]

    def __str__(self):
        return f"{self.quantity} x {self.product_name} (plan {self.plan_id})"


class SupplierPurchaseOrder(models.Model):
    """A human-entered supplier order record.

    This deliberately tracks the ordering lifecycle and confirmation details;
    it does not pretend that supplier websites have confirmed a receipt.
    """

    SUPPLIER_MCKESSON = 'mck'
    SUPPLIER_KOHLFRISCH = 'kf'
    SUPPLIER_OTHER = 'other'
    SUPPLIER_CHOICES = [
        (SUPPLIER_MCKESSON, 'McKesson'),
        (SUPPLIER_KOHLFRISCH, 'Kohl & Frisch'),
        (SUPPLIER_OTHER, 'Other supplier'),
    ]
    STATUS_DRAFT = 'draft'
    STATUS_SUBMITTED = 'submitted'
    STATUS_PARTIAL = 'partial'
    STATUS_RECEIVED = 'received'
    STATUS_CANCELLED = 'cancelled'
    STATUS_CHOICES = [
        (STATUS_DRAFT, 'Draft'),
        (STATUS_SUBMITTED, 'Submitted'),
        (STATUS_PARTIAL, 'Partially received'),
        (STATUS_RECEIVED, 'Received'),
        (STATUS_CANCELLED, 'Cancelled'),
    ]

    plan = models.ForeignKey(
        SupplierOrderPlan, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='purchase_orders',
    )
    supplier = models.CharField(max_length=10, choices=SUPPLIER_CHOICES)
    supplier_name = models.CharField(max_length=120, blank=True, default='')
    confirmation_number = models.CharField(max_length=100, blank=True, default='')
    order_date = models.DateField(default=timezone.localdate)
    expected_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default=STATUS_DRAFT)
    notes = models.TextField(blank=True, default='')
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='supplier_purchase_orders',
    )
    archived_at = models.DateTimeField(null=True, blank=True)
    archived_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='archived_supplier_purchase_orders',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-order_date', '-created_at']
        indexes = [
            models.Index(fields=['status', '-order_date'], name='supplierpo_status_date_idx'),
            models.Index(fields=['supplier', '-order_date'], name='supplierpo_supplier_idx'),
        ]

    @property
    def display_supplier(self):
        if self.supplier == self.SUPPLIER_OTHER and self.supplier_name:
            return self.supplier_name
        return self.get_supplier_display()

    def __str__(self):
        reference = self.confirmation_number or f'#{self.pk}'
        return f'{self.display_supplier} {reference}'


class SupplierPurchaseOrderLine(models.Model):
    purchase_order = models.ForeignKey(
        SupplierPurchaseOrder, on_delete=models.CASCADE, related_name='lines',
    )
    product = models.ForeignKey(
        Product, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='supplier_purchase_order_lines',
    )
    product_name = models.CharField(max_length=200)
    product_barcode = models.CharField(max_length=64, blank=True, default='')
    quantity_ordered = models.PositiveIntegerField(default=1)
    quantity_received = models.PositiveIntegerField(default=0)
    unit_cost = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True,
    )

    class Meta:
        ordering = ['pk']
        constraints = [
            models.CheckConstraint(
                condition=Q(quantity_received__lte=models.F('quantity_ordered')),
                name='supplierpo_received_not_over_ordered',
            ),
            models.CheckConstraint(
                condition=Q(unit_cost__isnull=True) | Q(unit_cost__gte=0),
                name='supplierpo_unit_cost_nonnegative',
            ),
        ]

    @property
    def remaining(self):
        return self.quantity_ordered - self.quantity_received

    def __str__(self):
        return f'{self.quantity_ordered} x {self.product_name}'


class SupplierOrderRun(models.Model):
    """One supplier browser-automation attempt with durable progress/control."""

    VENDOR_MCKESSON = 'mck'
    VENDOR_KOHLFRISCH = 'kf'
    VENDOR_CHOICES = [
        (VENDOR_MCKESSON, 'McKesson'),
        (VENDOR_KOHLFRISCH, 'Kohl & Frisch'),
    ]
    SOURCE_WEB = 'web'
    SOURCE_CLI = 'cli'
    SOURCE_LEGACY_STATUS = 'legacy_status'
    SOURCE_LEGACY_REPORT = 'legacy_report'
    SOURCE_CHOICES = [
        (SOURCE_WEB, 'Web ordering workflow'),
        (SOURCE_CLI, 'Command line'),
        (SOURCE_LEGACY_STATUS, 'Imported status file'),
        (SOURCE_LEGACY_REPORT, 'Imported CSV report'),
    ]
    STATE_STARTING = 'starting'
    STATE_LOGIN = 'login'
    STATE_WAITING_USER = 'waiting_user'
    STATE_RUNNING = 'running'
    STATE_PAUSED = 'paused'
    STATE_REVIEW = 'review'
    STATE_DONE = 'done'
    STATE_ERROR = 'error'
    STATE_CANCELLED = 'cancelled'
    STATE_CHOICES = [
        (STATE_STARTING, 'Starting'),
        (STATE_LOGIN, 'Login required'),
        (STATE_WAITING_USER, 'Waiting for user'),
        (STATE_RUNNING, 'Running'),
        (STATE_PAUSED, 'Paused'),
        (STATE_REVIEW, 'Ready for review'),
        (STATE_DONE, 'Done'),
        (STATE_ERROR, 'Error'),
        (STATE_CANCELLED, 'Cancelled'),
    ]

    plan = models.ForeignKey(
        SupplierOrderPlan, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='runs',
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='supplier_order_runs',
    )
    vendor = models.CharField(max_length=3, choices=VENDOR_CHOICES)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default=SOURCE_WEB)
    sequence_position = models.PositiveIntegerField(default=0)
    state = models.CharField(max_length=20, choices=STATE_CHOICES, default=STATE_STARTING)
    message = models.CharField(max_length=500, blank=True, default='')
    current = models.PositiveIntegerField(default=0)
    total = models.PositiveIntegerField(default=0)
    process_id = models.PositiveIntegerField(null=True, blank=True)
    pause_requested = models.BooleanField(default=False)
    cancel_requested = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['vendor', 'state', '-created_at'], name='supplierrun_vendor_idx'),
            models.Index(fields=['plan', 'sequence_position'], name='supplierrun_plan_idx'),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['plan', 'vendor'], condition=Q(plan__isnull=False),
                name='supplierrun_plan_vendor_uniq',
            ),
        ]

    def __str__(self):
        return f"{self.get_vendor_display()} run #{self.pk} ({self.get_state_display()})"


class SupplierOrderRunItem(models.Model):
    """Per-product result from a supplier attempt, retained for reporting."""

    OUTCOME_PENDING = 'pending'
    OUTCOME_ADDED = 'added'
    OUTCOME_SKIPPED = 'skipped'
    OUTCOME_CHOICES = [
        (OUTCOME_PENDING, 'Pending'),
        (OUTCOME_ADDED, 'Added'),
        (OUTCOME_SKIPPED, 'Not added'),
    ]

    run = models.ForeignKey(SupplierOrderRun, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    product_name = models.CharField(max_length=200)
    barcode = models.CharField(max_length=64, blank=True, default='')
    quantity_requested = models.PositiveIntegerField(default=1)
    position = models.PositiveIntegerField(default=0)
    outcome = models.CharField(max_length=10, choices=OUTCOME_CHOICES, default=OUTCOME_PENDING)
    reason = models.CharField(max_length=500, blank=True, default='')
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['position', 'pk']
        constraints = [
            models.UniqueConstraint(fields=['run', 'position'], name='supplierrunitem_position_uniq'),
        ]
        indexes = [
            models.Index(fields=['run', 'outcome'], name='supplierrunitem_outcome_idx'),
            models.Index(fields=['product', 'outcome'], name='supplierrunitem_product_idx'),
        ]

    def __str__(self):
        return f"{self.product_name} x{self.quantity_requested}: {self.get_outcome_display()}"


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
    # Hides the row from the checkout chooser's History panel only; the record
    # still appears on the Transactions page and in reports/Stock Log.
    hidden_from_history = models.BooleanField(default=False)

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


class TransactionCorrection(models.Model):
    """Immutable return/void record attached to an original transaction."""

    TYPE_RETURN = 'return'
    TYPE_VOID = 'void'
    TYPE_CORRECTION = 'correction'
    TYPE_CHOICES = [
        (TYPE_RETURN, 'Return'),
        (TYPE_VOID, 'Void'),
        (TYPE_CORRECTION, 'Correction'),
    ]

    correction_type = models.CharField(max_length=12, choices=TYPE_CHOICES)
    order = models.ForeignKey(
        Order, on_delete=models.PROTECT, null=True, blank=True,
        related_name='corrections',
    )
    checkout = models.ForeignKey(
        CheckoutOrder, on_delete=models.PROTECT, null=True, blank=True,
        related_name='corrections',
    )
    reason = models.CharField(max_length=255)
    note = models.TextField(blank=True, default='')
    adjustment_amount = models.DecimalField(
        max_digits=12, decimal_places=2, default=0,
        help_text='Financial adjustment recorded for reporting only.',
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='transaction_corrections',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        constraints = [
            models.CheckConstraint(
                condition=(
                    Q(order__isnull=False, checkout__isnull=True)
                    | Q(order__isnull=True, checkout__isnull=False)
                ),
                name='correction_exactly_one_transaction',
            ),
            models.CheckConstraint(
                condition=Q(adjustment_amount__gte=0),
                name='correction_amount_nonnegative',
            ),
        ]

    @property
    def transaction_label(self):
        if self.order_id:
            return f'Sale #{self.order_id}'
        return f'No-sale checkout #{self.checkout_id}'

    def __str__(self):
        return f'{self.get_correction_type_display()} for {self.transaction_label}'


class TransactionCorrectionUndo(models.Model):
    """Append-only audit record that reverses an accidental transaction void."""

    correction = models.OneToOneField(
        TransactionCorrection, on_delete=models.PROTECT, related_name='undo',
    )
    reason = models.CharField(
        max_length=255, default='Void entered by mistake',
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='transaction_correction_undos',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Undo for {self.correction}'


class TransactionCorrectionLine(models.Model):
    DISPOSITION_RESTOCK = 'restock'
    DISPOSITION_QUARANTINE = 'quarantine'
    DISPOSITION_DAMAGED = 'damaged'
    DISPOSITION_EXPIRED = 'expired'
    DISPOSITION_NO_RESTOCK = 'no_restock'
    DISPOSITION_CHOICES = [
        (DISPOSITION_RESTOCK, 'Return to stock'),
        (DISPOSITION_QUARANTINE, 'Quarantine'),
        (DISPOSITION_DAMAGED, 'Damaged'),
        (DISPOSITION_EXPIRED, 'Expired'),
        (DISPOSITION_NO_RESTOCK, 'Do not restock'),
    ]

    correction = models.ForeignKey(
        TransactionCorrection, on_delete=models.CASCADE, related_name='lines',
    )
    order_detail = models.ForeignKey(
        OrderDetail, on_delete=models.PROTECT, null=True, blank=True,
        related_name='correction_lines',
    )
    checkout_item = models.ForeignKey(
        CheckoutOrderItem, on_delete=models.PROTECT, null=True, blank=True,
        related_name='correction_lines',
    )
    product = models.ForeignKey(
        Product, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='transaction_correction_lines',
    )
    product_name = models.CharField(max_length=200)
    product_barcode = models.CharField(max_length=64, blank=True, default='')
    quantity = models.PositiveIntegerField()
    unit_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    disposition = models.CharField(
        max_length=16, choices=DISPOSITION_CHOICES,
        default=DISPOSITION_RESTOCK,
    )

    class Meta:
        ordering = ['pk']
        constraints = [
            models.CheckConstraint(
                condition=(
                    Q(order_detail__isnull=False, checkout_item__isnull=True)
                    | Q(order_detail__isnull=True, checkout_item__isnull=False)
                ),
                name='correctionline_exactly_one_source',
            ),
            models.CheckConstraint(
                condition=Q(quantity__gt=0),
                name='correctionline_qty_positive',
            ),
            models.CheckConstraint(
                condition=Q(unit_price__gte=0),
                name='correctionline_price_nonnegative',
            ),
        ]

    @property
    def line_adjustment(self):
        return self.quantity * self.unit_price

    def __str__(self):
        return f'{self.quantity} x {self.product_name}'


class ProductLotMovement(models.Model):
    """Connects each stock-ledger entry to the exact lots it changed."""

    DIRECTION_IN = 'in'
    DIRECTION_OUT = 'out'
    DIRECTION_CHOICES = [
        (DIRECTION_IN, 'Added to lot'),
        (DIRECTION_OUT, 'Removed from lot'),
    ]

    stock_change = models.ForeignKey(
        StockChange, on_delete=models.CASCADE, related_name='lot_movements',
    )
    lot = models.ForeignKey(
        ProductLot, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='movements',
    )
    lot_number = models.CharField(max_length=64)
    expiry_date = models.DateField(null=True, blank=True)
    quantity = models.PositiveIntegerField()
    direction = models.CharField(max_length=3, choices=DIRECTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['pk']
        constraints = [
            models.CheckConstraint(
                condition=Q(quantity__gt=0), name='lotmovement_qty_positive',
            ),
        ]
        indexes = [
            models.Index(fields=['lot', '-created_at'], name='lotmovement_lot_date_idx'),
        ]

    def __str__(self):
        return f'{self.direction} {self.quantity} / {self.lot_number}'


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
    archived_at    = models.DateTimeField(null=True, blank=True)
    archived_by    = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='archived_deliveries',
    )
    archive_reason = models.CharField(max_length=255, blank=True, default='')

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


class CustomLabelQueueItem(models.Model):
    """Durable free-form label queued by a user.

    Custom labels used to live in Django's browser session, which meant they
    disappeared on logout, session expiry, or when the same user moved to a
    different computer. Keep them beside the product-label queue in the
    database instead.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='custom_label_queue_items',
    )
    title = models.CharField(max_length=200)
    lines = models.JSONField(default=list, blank=True)
    copies = models.PositiveIntegerField(default=1)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['added_at', 'pk']
        indexes = [
            models.Index(fields=['user', 'added_at'], name='customlabel_user_added_idx'),
        ]

    def __str__(self):
        return f"{self.title} x{self.copies} (user={self.user_id})"


class LabelPrintOverride(models.Model):
    """Durable per-user print-only changes for a product or queued label."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='label_print_overrides',
    )
    product = models.ForeignKey(
        Product, on_delete=models.CASCADE, null=True, blank=True,
        related_name='label_print_overrides',
    )
    queue_item = models.ForeignKey(
        LabelQueueItem, on_delete=models.CASCADE, null=True, blank=True,
        related_name='print_overrides',
    )
    name = models.CharField(max_length=200, blank=True, default='')
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    barcode = models.CharField(max_length=64, blank=True, default='')
    barcode_overridden = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=['user', '-updated_at'], name='labeloverride_user_idx')]
        constraints = [
            models.CheckConstraint(
                condition=(
                    (Q(product__isnull=False) & Q(queue_item__isnull=True)) |
                    (Q(product__isnull=True) & Q(queue_item__isnull=False))
                ),
                name='labeloverride_exactly_one_target',
            ),
            models.UniqueConstraint(
                fields=['user', 'product'], condition=Q(product__isnull=False),
                name='labeloverride_user_product_uniq',
            ),
            models.UniqueConstraint(
                fields=['user', 'queue_item'], condition=Q(queue_item__isnull=False),
                name='labeloverride_user_queue_uniq',
            ),
        ]

    def __str__(self):
        target = f"product {self.product_id}" if self.product_id else f"queue {self.queue_item_id}"
        return f"Label override for {target} (user={self.user_id})"


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
    is_custom = models.BooleanField(default=False)
    custom_lines = models.JSONField(default=list, blank=True)

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
   archived_at = models.DateTimeField(null=True, blank=True)
   archived_by = models.ForeignKey(
       settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
       null=True, blank=True, related_name='archived_special_orders',
   )
   archive_reason = models.CharField(max_length=255, blank=True, default='')
 
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
    needs ordering. Admin or passkey-unlocked users advance the lifecycle;
    creators may edit their own entry while it is still pending.
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
    STATUS_PARTIAL_RECEIVED = 'partial_received'
    STATUS_RECEIVED = 'received'
    STATUS_READY = 'ready'
    STATUS_CONTACTED = 'contacted'
    STATUS_PICKED_UP = 'picked_up'
    STATUS_CANCELLED = 'cancelled'
    STATUS_NOT_FOR_SALE = 'not_for_sale'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_BACKORDERED, 'Back-Ordered'),
        (STATUS_ORDERED, 'Ordered'),
        (STATUS_PARTIAL_RECEIVED, 'Partially Received'),
        (STATUS_RECEIVED, 'Received'),
        (STATUS_READY, 'Ready for Pickup'),
        (STATUS_CONTACTED, 'Patient Contacted'),
        (STATUS_PICKED_UP, 'Picked Up'),
        (STATUS_CANCELLED, 'Cancelled'),
        (STATUS_NOT_FOR_SALE, 'Not for Sale (Consult Pharmacist)'),
    ]
    ADMIN_STATUS_CHOICES = [
        STATUS_PENDING, STATUS_BACKORDERED, STATUS_ORDERED,
        STATUS_PARTIAL_RECEIVED, STATUS_RECEIVED, STATUS_READY,
        STATUS_CONTACTED, STATUS_PICKED_UP, STATUS_CANCELLED,
        STATUS_NOT_FOR_SALE,
    ]
    # Compatibility name retained for existing integrations.
    GINA_STATUS_CHOICES = ADMIN_STATUS_CHOICES
    TERMINAL_STATUSES = [STATUS_PICKED_UP, STATUS_CANCELLED, STATUS_NOT_FOR_SALE]
    STATUS_TRANSITIONS = {
        STATUS_PENDING: {STATUS_ORDERED, STATUS_BACKORDERED, STATUS_CANCELLED, STATUS_NOT_FOR_SALE},
        STATUS_BACKORDERED: {STATUS_ORDERED, STATUS_CANCELLED, STATUS_NOT_FOR_SALE},
        STATUS_ORDERED: {STATUS_PARTIAL_RECEIVED, STATUS_RECEIVED, STATUS_BACKORDERED, STATUS_CANCELLED},
        STATUS_PARTIAL_RECEIVED: {STATUS_RECEIVED, STATUS_BACKORDERED, STATUS_CANCELLED},
        STATUS_RECEIVED: {STATUS_READY, STATUS_CONTACTED, STATUS_PICKED_UP},
        STATUS_READY: {STATUS_CONTACTED, STATUS_PICKED_UP},
        STATUS_CONTACTED: {STATUS_READY, STATUS_PICKED_UP},
        STATUS_PICKED_UP: set(),
        STATUS_CANCELLED: set(),
        STATUS_NOT_FOR_SALE: set(),
    }

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
    supplier_name = models.CharField(max_length=120, blank=True, default='')
    expected_date = models.DateField(null=True, blank=True)
    quantity_ordered = models.PositiveIntegerField(null=True, blank=True)
    quantity_received = models.PositiveIntegerField(default=0)
    ordered_at = models.DateTimeField(null=True, blank=True)
    received_at = models.DateTimeField(null=True, blank=True)
    contacted_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Where the row was created: in the app, or imported from the Google
    # Sheet / Form. gsheet_synced_at marks the first successful export to the
    # sheet — it lets the sync tell "never exported" apart from "someone
    # deleted the row in Google".
    SOURCE_APP = 'app'
    SOURCE_GSHEET = 'gsheet'
    SOURCE_CHOICES = [(SOURCE_APP, 'App'), (SOURCE_GSHEET, 'Google Sheet')]
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default=SOURCE_APP)
    gsheet_synced_at = models.DateTimeField(null=True, blank=True)

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
        indexes = [
            models.Index(fields=['is_deleted', 'status']),
            models.Index(fields=['expected_date', 'status'], name='ordering_expected_status_idx'),
        ]
        constraints = [
            models.CheckConstraint(
                condition=(
                    Q(quantity_ordered__isnull=True)
                    | Q(quantity_received__lte=models.F('quantity_ordered'))
                ),
                name='ordering_received_not_over_ordered',
            ),
        ]

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

    @property
    def is_terminal(self):
        return self.status in self.TERMINAL_STATUSES

    def can_transition_to(self, new_status):
        if new_status == self.status:
            return True
        return new_status in self.STATUS_TRANSITIONS.get(self.status, set())


class OrderingSheetStatusEvent(models.Model):
    entry = models.ForeignKey(
        OrderingSheetEntry, on_delete=models.CASCADE, related_name='status_events',
    )
    from_status = models.CharField(max_length=20, choices=OrderingSheetEntry.STATUS_CHOICES)
    to_status = models.CharField(max_length=20, choices=OrderingSheetEntry.STATUS_CHOICES)
    note = models.CharField(max_length=255, blank=True, default='')
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='ordering_status_events',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['entry', '-created_at'], name='orderingevent_entry_date_idx'),
        ]

    def __str__(self):
        return f'{self.entry_id}: {self.from_status} -> {self.to_status}'

class DailyReportArchive(models.Model):
    """A stored snapshot (rendered PDF) of a day's end-of-day report.

    One row per day (upserted). Rows older than RETENTION_DAYS are pruned when
    a new snapshot is saved; no independent daily cleanup is scheduled.
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


class DashboardTask(models.Model):
    """Shared pharmacy task with soft-archive and completion history."""

    text = models.CharField(max_length=200)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='dashboard_tasks_created',
    )
    created_by_name = models.CharField(max_length=150, blank=True, default='')
    completed = models.BooleanField(default=False)
    completed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='dashboard_tasks_completed',
    )
    completed_at = models.DateTimeField(null=True, blank=True)
    archived_at = models.DateTimeField(null=True, blank=True)
    archived_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='dashboard_tasks_archived',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at', '-pk']
        indexes = [
            models.Index(fields=['archived_at', 'completed', '-created_at'], name='dashtask_active_idx'),
        ]

    def __str__(self):
        return self.text


class UserTablePreference(models.Model):
    """Per-user display choices for one large table on one application page."""

    DENSITY_COMFORTABLE = 'comfortable'
    DENSITY_COMPACT = 'compact'
    DENSITY_CHOICES = [
        (DENSITY_COMFORTABLE, 'Comfortable'),
        (DENSITY_COMPACT, 'Compact'),
    ]
    PAGE_SIZE_CHOICES = [
        (25, '25 rows'),
        (50, '50 rows'),
        (100, '100 rows'),
        (200, '200 rows'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='table_preferences',
    )
    page_key = models.CharField(max_length=100)
    table_key = models.CharField(max_length=100, default='main')
    density = models.CharField(
        max_length=12,
        choices=DENSITY_CHOICES,
        default=DENSITY_COMFORTABLE,
    )
    page_size = models.PositiveSmallIntegerField(
        choices=PAGE_SIZE_CHOICES,
        default=50,
    )
    hidden_columns = models.JSONField(default=list, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['page_key', 'table_key']
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'page_key', 'table_key'],
                name='user_table_preference_uniq',
            ),
            models.CheckConstraint(
                condition=Q(page_size__in=[25, 50, 100, 200]),
                name='user_table_page_size_allowed',
            ),
        ]
        indexes = [
            models.Index(
                fields=['user', 'page_key'],
                name='tablepref_user_page_idx',
            ),
        ]

    def __str__(self):
        return f'{self.user_id}: {self.page_key}/{self.table_key}'


class StoreHours(models.Model):
    """One shared source of truth for the pharmacy's weekly opening hours."""

    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6
    DAY_CHOICES = [
        (MONDAY, 'Monday'),
        (TUESDAY, 'Tuesday'),
        (WEDNESDAY, 'Wednesday'),
        (THURSDAY, 'Thursday'),
        (FRIDAY, 'Friday'),
        (SATURDAY, 'Saturday'),
        (SUNDAY, 'Sunday'),
    ]

    weekday = models.PositiveSmallIntegerField(choices=DAY_CHOICES, unique=True)
    is_closed = models.BooleanField(default=False)
    opens_at = models.TimeField(null=True, blank=True)
    closes_at = models.TimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['weekday']
        verbose_name_plural = 'Store hours'
        constraints = [
            models.CheckConstraint(
                condition=Q(weekday__gte=0, weekday__lte=6),
                name='storehours_weekday_valid',
            ),
            models.CheckConstraint(
                condition=(
                    Q(is_closed=True)
                    | Q(
                        opens_at__isnull=False,
                        closes_at__isnull=False,
                        closes_at__gt=F('opens_at'),
                    )
                ),
                name='storehours_open_times_valid',
            ),
        ]

    def __str__(self):
        if self.is_closed:
            return f'{self.get_weekday_display()}: closed'
        return (
            f'{self.get_weekday_display()}: '
            f'{self.opens_at.strftime("%H:%M")}–{self.closes_at.strftime("%H:%M")}'
        )

    def clean(self):
        super().clean()
        if (
            not self.is_closed
            and self.closes_at is not None
            and (
                self.closes_at.minute != 0
                or self.closes_at.second != 0
                or self.closes_at.microsecond != 0
            )
        ):
            raise ValidationError({
                'closes_at': (
                    'Closing time must be on the hour so the hourly automation '
                    'can run exactly 30 minutes before closing.'
                ),
            })


class ScheduledJobRun(models.Model):
    """Durable status and output for scheduled and manually-triggered jobs."""

    JOB_GSHEET_PRECLOSE = 'gsheet_preclose'
    JOB_DATABASE_BACKUP = 'database_backup'
    JOB_REPORT_CLEANUP = 'report_cleanup'
    JOB_CHOICES = [
        (JOB_GSHEET_PRECLOSE, 'Google Sheet pre-closing pull'),
        (JOB_DATABASE_BACKUP, 'Pre-closing database backup'),
        # Retained so historical cleanup runs keep a readable label. The
        # cleanup is no longer part of the automatic schedule.
        (JOB_REPORT_CLEANUP, 'Daily report archive cleanup'),
    ]

    TRIGGER_SCHEDULED = 'scheduled'
    TRIGGER_MANUAL = 'manual'
    TRIGGER_CHOICES = [
        (TRIGGER_SCHEDULED, 'Scheduled'),
        (TRIGGER_MANUAL, 'Manual'),
    ]

    STATUS_RUNNING = 'running'
    STATUS_SUCCESS = 'success'
    STATUS_ERROR = 'error'
    STATUS_SKIPPED = 'skipped'
    STATUS_CHOICES = [
        (STATUS_RUNNING, 'Running'),
        (STATUS_SUCCESS, 'Successful'),
        (STATUS_ERROR, 'Failed'),
        (STATUS_SKIPPED, 'Skipped'),
    ]

    job_key = models.CharField(max_length=40, choices=JOB_CHOICES)
    trigger = models.CharField(
        max_length=12, choices=TRIGGER_CHOICES, default=TRIGGER_SCHEDULED,
    )
    business_date = models.DateField(null=True, blank=True)
    scheduled_for = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default=STATUS_RUNNING)
    attempt_count = models.PositiveSmallIntegerField(default=1)
    summary = models.CharField(max_length=500, blank=True, default='')
    error = models.TextField(blank=True, default='')
    result = models.JSONField(default=dict, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='scheduled_job_runs',
    )
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-started_at', '-pk']
        constraints = [
            models.UniqueConstraint(
                fields=['job_key', 'business_date'],
                condition=Q(trigger='scheduled'),
                name='scheduled_job_once_per_day',
            ),
            models.CheckConstraint(
                condition=(
                    Q(trigger='manual')
                    | Q(business_date__isnull=False)
                ),
                name='scheduled_job_date_required',
            ),
        ]
        indexes = [
            models.Index(fields=['job_key', 'status', '-started_at'], name='jobrun_job_status_idx'),
            models.Index(fields=['trigger', '-started_at'], name='jobrun_trigger_date_idx'),
        ]

    def __str__(self):
        return f'{self.get_job_key_display()} — {self.get_status_display()}'


class InventoryAuditRun(models.Model):
    """One durable execution of the inventory-integrity checks."""

    STATUS_RUNNING = 'running'
    STATUS_PASSED = 'passed'
    STATUS_ISSUES = 'issues'
    STATUS_REPAIRED = 'repaired'
    STATUS_ERROR = 'error'
    STATUS_CHOICES = [
        (STATUS_RUNNING, 'Running'),
        (STATUS_PASSED, 'Passed'),
        (STATUS_ISSUES, 'Problems found'),
        (STATUS_REPAIRED, 'Problems repaired'),
        (STATUS_ERROR, 'Audit failed'),
    ]

    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default=STATUS_RUNNING)
    repair_requested = models.BooleanField(default=False)
    issue_count = models.PositiveIntegerField(default=0)
    repaired_count = models.PositiveIntegerField(default=0)
    checks = models.JSONField(default=list, blank=True)
    summary = models.CharField(max_length=500, blank=True, default='')
    error = models.TextField(blank=True, default='')
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='inventory_audit_runs',
    )
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-started_at', '-pk']
        indexes = [
            models.Index(fields=['status', '-started_at'], name='invaudit_status_date_idx'),
        ]

    def __str__(self):
        return f'Inventory audit #{self.pk} — {self.get_status_display()}'


class InventoryAuditIssue(models.Model):
    """A structured, linkable finding retained with an inventory audit."""

    SEVERITY_WARNING = 'warning'
    SEVERITY_ERROR = 'error'
    SEVERITY_CHOICES = [
        (SEVERITY_WARNING, 'Warning'),
        (SEVERITY_ERROR, 'Error'),
    ]

    run = models.ForeignKey(
        InventoryAuditRun, on_delete=models.CASCADE, related_name='issues',
    )
    code = models.CharField(max_length=60)
    severity = models.CharField(
        max_length=10, choices=SEVERITY_CHOICES, default=SEVERITY_ERROR,
    )
    product = models.ForeignKey(
        Product, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='inventory_audit_issues',
    )
    product_name = models.CharField(max_length=200, blank=True, default='')
    title = models.CharField(max_length=200)
    detail = models.TextField(blank=True, default='')
    expected_value = models.CharField(max_length=100, blank=True, default='')
    actual_value = models.CharField(max_length=100, blank=True, default='')
    repairable = models.BooleanField(default=False)
    repaired = models.BooleanField(default=False)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['severity', 'code', 'pk']
        indexes = [
            models.Index(fields=['run', 'code'], name='auditissue_run_code_idx'),
            models.Index(fields=['product', '-created_at'], name='auditissue_product_idx'),
        ]

    def __str__(self):
        return f'{self.code}: {self.title}'
