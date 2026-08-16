from django.contrib import admin
from .models import (
    Category, Product, Order, OrderDetail, Customer, StockChange,
    CheckinSession, LoginAudit, UserAction,
    CheckoutOrder, CheckoutOrderItem,
    DashboardTask, LabelPrintOverride, SupplierOrderPlan,
    SupplierOrderPlanItem, SupplierOrderRun, SupplierOrderRunItem,
    StoreHours, ScheduledJobRun, InventoryAuditRun, InventoryAuditIssue,
)

admin.site.register(Customer)
admin.site.register(Category)
admin.site.register(Product)
admin.site.register(Order)
admin.site.register(OrderDetail)
admin.site.register(StockChange)
admin.site.register(CheckinSession)
admin.site.register(LoginAudit)
admin.site.register(UserAction)


class CheckoutOrderItemInline(admin.TabularInline):
    model = CheckoutOrderItem
    extra = 0
    readonly_fields = ('product', 'product_name', 'product_barcode', 'price', 'taxable', 'quantity', 'added_at')


@admin.register(CheckoutOrder)
class CheckoutOrderAdmin(admin.ModelAdmin):
    list_display = ('pk', 'user', 'status', 'total_price', 'created_at', 'submitted_at')
    list_filter = ('status', 'created_at')
    search_fields = ('user__username',)
    inlines = [CheckoutOrderItemInline]


admin.site.register(CheckoutOrderItem)


@admin.register(DashboardTask)
class DashboardTaskAdmin(admin.ModelAdmin):
    list_display = ('text', 'created_by_name', 'completed', 'created_at', 'completed_at', 'archived_at')
    list_filter = ('completed', 'created_at', 'archived_at')
    search_fields = ('text', 'created_by_name', 'created_by__username')


@admin.register(LabelPrintOverride)
class LabelPrintOverrideAdmin(admin.ModelAdmin):
    list_display = ('user', 'product', 'queue_item', 'name', 'price', 'updated_at')
    search_fields = ('user__username', 'product__name', 'name', 'barcode')


class SupplierOrderPlanItemInline(admin.TabularInline):
    model = SupplierOrderPlanItem
    extra = 0
    readonly_fields = ('product', 'product_name', 'barcode', 'quantity', 'position')


@admin.register(SupplierOrderPlan)
class SupplierOrderPlanAdmin(admin.ModelAdmin):
    list_display = ('pk', 'created_by', 'status', 'vendor_sequence', 'created_at', 'completed_at')
    list_filter = ('status', 'created_at')
    inlines = [SupplierOrderPlanItemInline]


class SupplierOrderRunItemInline(admin.TabularInline):
    model = SupplierOrderRunItem
    extra = 0
    readonly_fields = (
        'product', 'product_name', 'barcode', 'quantity_requested',
        'position', 'outcome', 'reason', 'processed_at',
    )


@admin.register(SupplierOrderRun)
class SupplierOrderRunAdmin(admin.ModelAdmin):
    list_display = ('pk', 'vendor', 'source', 'state', 'plan', 'created_by', 'current', 'total', 'created_at')
    list_filter = ('vendor', 'source', 'state', 'created_at')
    search_fields = ('created_by__username', 'message')
    inlines = [SupplierOrderRunItemInline]


@admin.register(StoreHours)
class StoreHoursAdmin(admin.ModelAdmin):
    list_display = ('day_name', 'is_closed', 'opens_at', 'closes_at', 'updated_at')
    ordering = ('weekday',)

    @admin.display(ordering='weekday', description='Day')
    def day_name(self, obj):
        return obj.get_weekday_display()


@admin.register(ScheduledJobRun)
class ScheduledJobRunAdmin(admin.ModelAdmin):
    list_display = (
        'job_key', 'trigger', 'business_date', 'status', 'attempt_count',
        'started_at', 'completed_at',
    )
    list_filter = ('job_key', 'trigger', 'status')
    readonly_fields = (
        'job_key', 'trigger', 'business_date', 'scheduled_for', 'status',
        'attempt_count', 'summary', 'error', 'result', 'created_by',
        'started_at', 'completed_at', 'created_at', 'updated_at',
    )


class InventoryAuditIssueInline(admin.TabularInline):
    model = InventoryAuditIssue
    extra = 0
    readonly_fields = (
        'code', 'severity', 'product', 'product_name', 'title', 'detail',
        'expected_value', 'actual_value', 'repairable', 'repaired', 'created_at',
    )


@admin.register(InventoryAuditRun)
class InventoryAuditRunAdmin(admin.ModelAdmin):
    list_display = (
        'pk', 'status', 'issue_count', 'repaired_count', 'repair_requested',
        'created_by', 'started_at', 'completed_at',
    )
    list_filter = ('status', 'repair_requested', 'started_at')
    readonly_fields = (
        'status', 'repair_requested', 'issue_count', 'repaired_count', 'checks',
        'summary', 'error', 'created_by', 'started_at', 'completed_at',
    )
    inlines = [InventoryAuditIssueInline]
