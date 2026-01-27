from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from app.views import (
  InventoryView, EditProductView, AddProductView, CheckinProductView,
  LowStockView, CreateOrderView, OrderView, SubmitOrderView, delete_item,
  delete_order_item, ItemListView, DeleteRecentlyPurchasedProductView,
  DeleteAllOrdersView, DeleteAllRecentlyPurchasedView, signup, CustomLoginView, delete_one, update_product_settings,
  AddQuantityView, ExpiredProductView,OrderDetailView,AddProductByIdView, AddProductByIdCheckinView,
  ProductTrendView, CheckinEditProductView, LabelPrintingView, GenerateLabelPDFView, ExportRecentlyPurchasedCSVView
)


urlpatterns = [
  # Admin Site
  path('admin/', admin.site.urls),

  path("product-trend/", ProductTrendView.as_view(), name="product_trend"),

  # Authentication
  path('login/', CustomLoginView.as_view(template_name='login.html'), name='login'),
  path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
 
  path('signup/', signup, name='signup'),

  # Default route
  path('', CustomLoginView.as_view(template_name='login.html'), name='home'),

  #Expired
  path('expired-products/', ExpiredProductView.as_view(), name='expired_products'),

  # Orders
  path('order/', CreateOrderView.as_view(), name='create_order'),
  path('order/submit/', SubmitOrderView.as_view(), name='submit_order'),
  path('order/delete-item/<int:product_id>/', delete_order_item, name='delete_order_item'),

  path('orders/', OrderView.as_view(), name='order_view'),

  path('add-product/<int:product_id>/', AddProductByIdView.as_view(), name='add_product_by_id'),

  # Inventory
  path('inventory/', InventoryView.as_view(), name='inventory_display'),
  path('product/edit/<int:product_id>/', EditProductView.as_view(), name='edit_product'),
  path('new-product/', AddProductView.as_view(), name='new_product'),
  path('product/delete/<int:product_id>/', delete_item, name='delete_item'),

  # Low Stock
  path('low-stock/', LowStockView.as_view(), name='low_stock'),
  path('low-stock/delete/<int:id>/', DeleteRecentlyPurchasedProductView.as_view(), name='delete_recently_purchased_product'),
  path('low-stock/delete_all/', DeleteAllRecentlyPurchasedView.as_view(), name='delete_all_recently_purchased'),

  # Check-in
  path('checkin/', CheckinProductView.as_view(), name='checkin'),
  path('product/add-quantity/<int:product_id>/', AddQuantityView, name='add_quantity'),
  path('delete_one/<int:product_id>/', delete_one, name='delete_one'),
  path('checkin/update-product-settings/<int:product_id>/', update_product_settings, name='update_product_settings'),

  path("checkin/product/<int:product_id>/edit/", CheckinEditProductView.as_view(), name="checkin_edit_product"),
  path('checkin/add/<int:product_id>/', AddProductByIdCheckinView.as_view(), name='checkin_add_by_id'),

  # Order Item Management
  path('orders/<int:order_id>/', OrderDetailView.as_view(), name='order_detail'),  # Order details page
  path('delete-orders/', DeleteAllOrdersView.as_view(), name='delete_all_orders'),

  path('labels/', LabelPrintingView.as_view(), name='label_printing'),
  path('labels/generate/', GenerateLabelPDFView.as_view(), name='generate_label_pdf'),
  path('export-recently-purchased/', ExportRecentlyPurchasedCSVView.as_view(), name='export_recently_purchased_csv'),

  # Item List
  path('item_list/', ItemListView.as_view(), name='item_list'),
]

