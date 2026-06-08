from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from app.views import (
  InventoryView, EditProductView, AddProductView, CheckinProductView,
  LowStockView, CreateOrderView, OrderView, SubmitOrderView, delete_item,
  delete_order_item, ItemListView, DeleteRecentlyPurchasedProductView,
  DeleteAllOrdersView, DeleteOrderView, OrderPDFView, ExportAllOrdersPDFView, DeleteAllRecentlyPurchasedView, signup, CustomLoginView, delete_one, update_product_settings,
  AddQuantityView, ExpiredProductView, ExpiredProductPDFView, ExpiredLogPDFView, OrderDetailView,AddProductByIdView, AddProductByIdCheckinView,
  ProductTrendView, CheckinEditProductView, LabelPrintingView, GenerateLabelPDFView, ExportRecentlyPurchasedCSVView,
  RevertPrintLabelCategoryView, LabelSessionListView, LabelSessionDetailView, LabelSessionDeleteView, LabelSessionRegenerateView, LabelSessionAddToQueueView, LabelSessionClearAllView,
  OutOfStockView, LowStockTrendView, ExportInventoryCSVView, ExportTransactionsCSVView, OrderSuccessView,
  GlobalSearchAPIView, AlertBannerAPIView, ProductDetailAPIView, BulkDeleteRecentlyPurchasedView,
  DeleteByCategoryRecentlyPurchasedView, DeleteOlderThanRecentlyPurchasedView, home,
  DeliveryView,
  SalesAnalyticsView,
  ActivityLogView,
  CheckinDashboardView, StartCheckinSessionView, EndCheckinSessionView, CheckinSessionDetailView,
  DeleteCheckinSessionView, ClearCheckinHistoryView, CheckinSessionPDFView, CheckinAllSessionsPDFView,
  ReopenCheckinSessionView, SessionAdjustLineView, SessionRemoveLineView,
)


urlpatterns = [
  # Admin Site
  path('admin/', admin.site.urls),

  path("product-trend/", ProductTrendView.as_view(), name="product_trend"),
  path("out-of-stock/", OutOfStockView.as_view(), name="out_of_stock"),
  path("low-stock-alert/", LowStockTrendView.as_view(), name="low_stock_trend"),

  # Authentication
  path('login/', CustomLoginView.as_view(template_name='login.html'), name='login'),
  path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
 
  path('signup/', signup, name='signup'),

  # Default route
  path('', CustomLoginView.as_view(template_name='login.html'), name='home'),
  path('dashboard/', home, name='dashboard'),

  #Expired
  path('expired-products/', ExpiredProductView.as_view(), name='expired_products'),
  path('expired-products/pdf/', ExpiredProductPDFView.as_view(), name='expired_products_pdf'),
  path('expired-products/log-pdf/', ExpiredLogPDFView.as_view(), name='expired_log_pdf'),

  # Orders
  path('order/', CreateOrderView.as_view(), name='create_order'),
  path('order/submit/', SubmitOrderView.as_view(), name='submit_order'),
  path('order/delete-item/<int:product_id>/', delete_order_item, name='delete_order_item'),

  path('orders/', OrderView.as_view(), name='order_view'),
  path('sales/', SalesAnalyticsView.as_view(), name='sales_analytics'),

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
  path('low-stock/bulk-delete/', BulkDeleteRecentlyPurchasedView.as_view(), name='bulk_delete_recently_purchased'),
  path('low-stock/delete-by-category/', DeleteByCategoryRecentlyPurchasedView.as_view(), name='delete_rp_by_category'),
  path('low-stock/delete-older-than/', DeleteOlderThanRecentlyPurchasedView.as_view(), name='delete_rp_older_than'),

  # Check-in — session dashboard & lifecycle
  path('checkin/', CheckinDashboardView.as_view(), name='checkin_dashboard'),
  path('checkin/start/', StartCheckinSessionView.as_view(), name='checkin_start'),
  path('checkin/session/<int:session_id>/', CheckinProductView.as_view(), name='checkin_session'),
  path('checkin/session/<int:session_id>/end/', EndCheckinSessionView.as_view(), name='checkin_end'),
  path('checkin/session/<int:session_id>/detail/', CheckinSessionDetailView.as_view(), name='checkin_session_detail'),
  path('checkin/session/<int:session_id>/delete/', DeleteCheckinSessionView.as_view(), name='checkin_session_delete'),
  path('checkin/session/<int:session_id>/reopen/', ReopenCheckinSessionView.as_view(), name='checkin_session_reopen'),
  path('checkin/session/<int:session_id>/adjust/<int:change_id>/', SessionAdjustLineView.as_view(), name='checkin_session_adjust'),
  path('checkin/session/<int:session_id>/remove-line/<int:change_id>/', SessionRemoveLineView.as_view(), name='checkin_session_remove_line'),
  path('checkin/session/<int:session_id>/pdf/', CheckinSessionPDFView.as_view(), name='checkin_session_pdf'),
  path('checkin/clear-history/', ClearCheckinHistoryView.as_view(), name='checkin_clear_history'),
  path('checkin/export-all-pdf/', CheckinAllSessionsPDFView.as_view(), name='checkin_all_sessions_pdf'),

  # Check-in — session-scoped actions
  path('checkin/session/<int:session_id>/add-quantity/<int:product_id>/', AddQuantityView, name='add_quantity'),
  path('checkin/session/<int:session_id>/delete-one/<int:product_id>/', delete_one, name='delete_one'),
  path('checkin/session/<int:session_id>/product/<int:product_id>/edit/', CheckinEditProductView.as_view(), name='checkin_edit_product'),
  path('checkin/session/<int:session_id>/add/<int:product_id>/', AddProductByIdCheckinView.as_view(), name='checkin_add_by_id'),
  path('checkin/update-product-settings/<int:product_id>/', update_product_settings, name='update_product_settings'),

  # Order Item Management
  path('orders/<int:order_id>/', OrderDetailView.as_view(), name='order_detail'),  # Order details page
  path('delete-orders/', DeleteAllOrdersView.as_view(), name='delete_all_orders'),
  path('orders/<int:order_id>/delete/', DeleteOrderView.as_view(), name='delete_order'),
  path('orders/<int:order_id>/pdf/', OrderPDFView.as_view(), name='order_pdf'),

  path('labels/', LabelPrintingView.as_view(), name='label_printing'),
  path('labels/generate/', GenerateLabelPDFView.as_view(), name='generate_label_pdf'),
  path('export-recently-purchased/', ExportRecentlyPurchasedCSVView.as_view(), name='export_recently_purchased_csv'),
  path('export-inventory/', ExportInventoryCSVView.as_view(), name='export_inventory_csv'),
  path('export-transactions/', ExportTransactionsCSVView.as_view(), name='export_transactions_csv'),
  path('export-transactions-pdf/', ExportAllOrdersPDFView.as_view(), name='export_transactions_pdf'),
  path('labels/revert/', RevertPrintLabelCategoryView.as_view(), name='revert_labels'),
  path('labels/sessions/', LabelSessionListView.as_view(), name='label_sessions'),
  path('labels/sessions/<int:session_id>/', LabelSessionDetailView.as_view(), name='label_session_detail'),
  path('labels/sessions/<int:session_id>/delete/', LabelSessionDeleteView.as_view(), name='label_session_delete'),
  path('labels/sessions/<int:session_id>/regenerate/', LabelSessionRegenerateView.as_view(), name='label_session_regenerate'),
  path('labels/sessions/<int:session_id>/add-to-queue/', LabelSessionAddToQueueView.as_view(), name='label_session_add_to_queue'),
  path('labels/sessions/clear/', LabelSessionClearAllView.as_view(), name='label_sessions_clear'),

  # Order Success
  path('order/success/<int:order_id>/', OrderSuccessView.as_view(), name='order_success'),

  # Item List
  path('item_list/', ItemListView.as_view(), name='item_list'),

  # New Feature Routes
  path('api/search/', GlobalSearchAPIView.as_view(), name='global_search'),
  path('api/product-detail/', ProductDetailAPIView.as_view(), name='product_detail_api'),
  path('api/alerts/', AlertBannerAPIView.as_view(), name='alert_banner_api'),

  # Delivery check-in / check-out
  path('delivery/', DeliveryView.as_view(), name='delivery'),

  # Activity Log
  path('activity-log/', ActivityLogView.as_view(), name='activity_log'),
]

