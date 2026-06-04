from django.contrib import admin
from .models import Category, Product, Order, OrderDetail, Customer, StockChange, LoginAudit, UserAction

admin.site.register(Customer)
admin.site.register(Category)
admin.site.register(Product)
admin.site.register(Order)
admin.site.register(OrderDetail)
admin.site.register(StockChange)
admin.site.register(LoginAudit)
admin.site.register(UserAction)