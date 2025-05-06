from django.contrib import admin
from .models import Category, Product, Order, OrderDetail, Customer

admin.site.register(Customer)
admin.site.register(Category)
admin.site.register(Product)
admin.site.register(Order)
admin.site.register(OrderDetail)