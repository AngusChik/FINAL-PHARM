from django.contrib import admin
from .models import (
    Category, Product, Order, OrderDetail, Customer, StockChange,
    CheckinSession, LoginAudit, UserAction,
    CheckoutOrder, CheckoutOrderItem,
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
