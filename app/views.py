from decimal import Decimal
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.views import View
from django.contrib import messages
from django.db.models import Sum
from django.db import transaction
from django.core.paginator import Paginator
from django.core.cache import cache
from app.mixins import AdminRequiredMixin
from django.contrib.auth.decorators import login_required #for @login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView
from django.contrib.auth.forms import UserCreationForm
from datetime import date, datetime, timedelta
from django.utils.timezone import now
from django.db import IntegrityError  # To handle IntegrityError
from .utils import recalculate_order_totals, get_product_stock_records, recommend_inventory_action # Import the function
from .forms import  EditProductForm, OrderDetailForm, BarcodeForm, ItemForm, AddProductForm
from .models import Item, Product, Category, Order, OrderDetail, RecentlyPurchasedProduct, StockChange
from dateutil.relativedelta import relativedelta # change pip install python-dateutil
from django.db.models import Sum
from django.db.models.functions import TruncMonth, TruncWeek, TruncDay
    
class ProductTrendView(AdminRequiredMixin, View):
    template_name = "product_trend.html"

    def get(self, request):
        query = request.GET.get("q", "").strip()
        chart_type = request.GET.get("type", "bar")
        granularity = request.GET.get("granularity", "month")

        try:
            end_date = datetime.strptime(request.GET.get("end", ""), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            end_date = date.today()

        try:
            start_date = datetime.strptime(request.GET.get("start", ""), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            start_date = end_date - timedelta(days=365)

        # Fetch all products for dropdown search
        all_products = list(Product.objects.values("product_id", "name", "barcode", "item_number", "price", "quantity_in_stock"))

        context = {
            "query": query,
            "chart_type": chart_type,
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity,
            "all_products": all_products,
        }

        # Resolve query from dropdown to product barcode or name
        product = None
        if query:
            # Attempt exact barcode match first
            product = Product.objects.filter(barcode__iexact=query).first()

            sold, restocked, labels, cumulative_stock = self._grouped_totals(product, start_date, end_date, granularity)

            # error handling for missing cost per unit
            if product.price_per_unit is None:
                messages.error(
                    request,
                    f"Product '{product.name}' is missing a defined price per unit. Please update it to view recommendations."
                )
                context.update({
                    "product": product,
                    "sold": sold,
                    "restocked": restocked,
                    "periods": labels,
                    "cumulative_stock": cumulative_stock,
                    "price_per_unit_missing_message": "Adjust price through edit product.",
                })
            else:
                # Call algorithm functions
                purchases, sales, expiries = get_product_stock_records(product, str(start_date), str(end_date))
                recommendation_data = recommend_inventory_action(
                    purchase_history=purchases,
                    sale_history=sales,
                    expiry_history=expiries,
                    timeframe_start=str(start_date),
                    timeframe_end=str(end_date),
                    cost_per_unit=float(product.price_per_unit),
                    price_per_unit=float(product.price),
                    granularity=granularity,
                )

                context.update({
                    "product": product,
                    "sold": sold,
                    "restocked": restocked,
                    "periods": labels,
                    "cumulative_stock": cumulative_stock,
                    "recommendation_data": recommendation_data,
                    "granularity": granularity,
                    "total_price": product.price * recommendation_data["suggested_order_quantity"],
                })

        return render(request, self.template_name, context)


    def _grouped_totals(self, product, start_date, end_date, granularity):
        trunc_map = {
            "day": TruncDay("timestamp"),
            "week": TruncWeek("timestamp"),
            "month": TruncMonth("timestamp"),
        }

        trunc = trunc_map.get(granularity, TruncMonth("timestamp"))
        # Filter by product and date range
        qs = (
            StockChange.objects.filter(
                product=product,
                timestamp__date__gte=start_date,
                timestamp__date__lte=end_date
            )
            .annotate(period=trunc)
            .values("period", "change_type")
            .annotate(total=Sum("quantity"))
            .order_by("period")
        )

        # Generate label list
        periods = []
        current = start_date
        while current <= end_date:
            if granularity == "day":
                label = current.strftime("%Y-%m-%d")
                current += timedelta(days=1)
            elif granularity == "week":
                week_start = current - timedelta(days=current.weekday())
                label = f"Week of {week_start.strftime('%Y-%m-%d')}"
                current += timedelta(weeks=1)
            else:  # month
                label = current.strftime("%b %Y")
                current = (current + timedelta(days=32)).replace(day=1)

            periods.append(label)

        sold = [0] * len(periods)
        restocked = [0] * len(periods)
        total_stock_changes = [0] * len(periods)

        # Build lookup to index
        label_to_index = {label: i for i, label in enumerate(periods)}

        for row in qs:
            period_date = row["period"].date()
            if granularity == "day":
                label = period_date.strftime("%Y-%m-%d")
            elif granularity == "week":
                label = f"Week of {(period_date - timedelta(days=period_date.weekday())).strftime('%Y-%m-%d')}"
            else:  # month
                label = period_date.strftime("%b %Y")

            idx = label_to_index.get(label)
            if idx is not None:
                change_type = row["change_type"]
                qty = row["total"]
                if change_type == "checkout":
                    sold[idx] += abs(qty)
                    total_stock_changes[idx] -= abs(qty)  # sold decreases stock
                elif change_type == "checkin":
                    restocked[idx] += qty
                    total_stock_changes[idx] += qty  # restock increases stock
                elif change_type == "error_add":
                    total_stock_changes[idx] += qty  # manual add increases stock
                elif change_type == "error_subtract":
                    total_stock_changes[idx] -= abs(qty)  # manual subtract decreases stock
                elif change_type == "checkin_delete1":
                    restocked[idx] -= abs(qty)  # treat like removed
                    total_stock_changes[idx] -= abs(qty)

        # Now calculate cumulative total stock over time
        cumulative_stock = []
        running_total = 0
        for change in total_stock_changes:
            running_total += change
            cumulative_stock.append(running_total)

        return sold, restocked, periods, cumulative_stock


# Home view
@login_required
def home(request):
   if not request.user.is_authenticated:
       return redirect('login')  # Redirect to login page
   return render(request, 'home.html')

def signup(request):
   if request.method == 'POST':
       form = UserCreationForm(request.POST)
       if form.is_valid():
           form.save()
           messages.success(request, "Your account has been created successfully! You can now log in.")
           return redirect('login')
   else:
       form = UserCreationForm()
   return render(request, 'signup.html', {'form': form})
 
class CustomLoginView(LoginView):
    def get(self, request, *args, **kwargs):
        # Redirect authenticated users to the appropriate page
        if request.user.is_authenticated:
            if request.user.is_staff:  # Redirect admins
                return redirect('inventory_display')  # Example: Admin page
            return redirect('checkin')  # Example: Regular user page
        return super().get(request, *args, **kwargs)


    def get_success_url(self):
        """
        Redirect users based on their role after a successful login.
        """
        if self.request.user.is_staff:
            return reverse('inventory_display')  # Admin-specific page
        return reverse('checkin')  # Regular user page
   
# Display all orders - Transaction page.
class OrderView(AdminRequiredMixin, View):
    template_name = 'order_view.html'

    def get(self, request):
        orders = Order.objects.all().order_by('-order_id')
        current_order_id = request.session.get('order_id')  # Get current active order
        return render(request, self.template_name, {
            'orders': orders,
            'current_order_id': current_order_id  # Pass it to template
        })

   
class OrderDetailView(View):
    template_name = 'order_detail.html'

    def get(self, request, order_id):
        # Get the order and its details
        order = get_object_or_404(Order, order_id=order_id)
        order_details = order.details.all()  # Assuming 'details' is the related name for the OrderDetail model

        # Calculate total price per item (quantity × price)
        order_details_with_total = [
            {
                'detail': detail,
                'total_price': detail.product.price * detail.quantity
            }
            for detail in order_details
        ]

        # Calculate order total before tax and after tax
        total_price_before_tax = sum(item['total_price'] for item in order_details_with_total)
        total_price_after_tax = total_price_before_tax * Decimal('1.13')  # Assuming 13% tax

        return render(request, self.template_name, {
            'order': order,
            'order_details_with_total': order_details_with_total,
            'total_price_before_tax': total_price_before_tax,
            'total_price_after_tax': total_price_after_tax,
        })
        
# change
class AddProductByIdView(LoginRequiredMixin, View):
    def post(self, request, product_id):
        order = CreateOrderView().get_order(request)
        quantity = int(request.POST.get('quantity', 1))

        try:
            product = Product.objects.get(product_id=product_id)
        except Product.DoesNotExist:
            messages.error(request, "Product not found.",extra_tags='order')
            return redirect('create_order')
        
        # ✅ Check for expired product
        if product.expiry_date and product.expiry_date < now().date():
            messages.error(
                request,
                f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                extra_tags='order'
            )
            return redirect('create_order')  # 🔁 Redirect instead of rendering

        quantity_to_add = min(quantity, product.quantity_in_stock)

        with transaction.atomic():
            order_detail, created = OrderDetail.objects.select_for_update().get_or_create(
                order=order,
                product=product,
                defaults={'quantity': quantity_to_add, 'price': product.price}
            )
            if not created:
                order_detail.quantity += quantity_to_add
                order_detail.save()

            product.quantity_in_stock -= quantity_to_add
            product.save()

        recalculate_order_totals(order)

        if quantity_to_add == 0:
            messages.warning(request, f"Product '{product.name}' added with quantity 0 due to limited stock.",extra_tags='order')
        elif quantity_to_add < quantity:
            messages.warning(request, f"Only {quantity_to_add} units of '{product.name}' added due to limited stock.",extra_tags='order')
        else:
            messages.success(request, f"{quantity_to_add} units of '{product.name}' successfully added.",extra_tags='order')

        return redirect('create_order')
    

class CreateOrderView(LoginRequiredMixin, View):
    template_name = 'order_form.html'

    def get_order(self, request):
        order_id = request.session.get('order_id')
        
        if order_id:
            try:
                # Retrieve only non-submitted orders
                return Order.objects.get(order_id=order_id, submitted=False)
            except Order.DoesNotExist:
                del request.session['order_id']  # Clear invalid session data

        # Create a new order if none is found or session is invalid
        return self._create_new_order(request)

    def _create_new_order(self, request):
            last_order = Order.objects.order_by('-order_id').first()
            next_order_id = 1 if not last_order else last_order.order_id + 1
            order = Order.objects.create(order_id=next_order_id, total_price=Decimal('0.00'))
            request.session['order_id'] = order.order_id
            return order

    def get(self, request, *args, **kwargs):
        order = self.get_order(request)
        form = BarcodeForm()

        # change
        name_query = request.GET.get('name_query', '')
        search_results = []
        all_products = [
            {
                'id': p['product_id'],
                'name': p['name'],
                'price': str(p['price']),
                'quantity_in_stock': p['quantity_in_stock'],
                'item_number': p['item_number']  # ✅ Add this line
            } for p in Product.objects.values('product_id', 'name', 'price', 'quantity_in_stock', 'item_number')
        ]


        # change
        if name_query:
            search_results = Product.objects.filter(name__icontains=name_query).order_by('name')

        # Order details and totals
        order_details = order.details.all().order_by('-order_date')
        total_price_before_tax = sum(detail.product.price * detail.quantity for detail in order_details)
        total_price_after_tax = total_price_before_tax * Decimal('1.13')
        
        # Change name_query, search results
        return render(request, self.template_name, {
            'order': order,
            'form': form,
            'order_details': order_details,
            'total_price_before_tax': total_price_before_tax,
            'total_price_after_tax': total_price_after_tax,
            'search_results': search_results,
            'name_query': name_query,
            'all_products': all_products,

        })

    def post(self, request, *args, **kwargs):
        order = self.get_order(request)
        form = BarcodeForm(request.POST)

        if form.is_valid():
            barcode = form.cleaned_data['barcode']
            requested_quantity = form.cleaned_data.get('quantity', 1)

            try:
                product = Product.objects.get(barcode=barcode)
            except Product.DoesNotExist:
                messages.error(request, f"No product found with barcode '{barcode}'.",extra_tags='order')
                return self._render_order_page(request, order, form)
            
            # Check for expired product
            if product.expiry_date and product.expiry_date < now().date():
                messages.error(
                    request,
                    f"Cannot add '{product.name}' — product is expired (Expiry: {product.expiry_date}).",
                    extra_tags='order'
                )
                return redirect('create_order')  # Redirect instead of rendering

            quantity_to_add = min(requested_quantity, product.quantity_in_stock)

            with transaction.atomic():
                order_detail, created = OrderDetail.objects.select_for_update().get_or_create(
                    order=order,
                    product=product,
                    defaults={'quantity': quantity_to_add, 'price': product.price}
                )
                if not created:
                    order_detail.quantity += quantity_to_add
                    order_detail.save()

                # Update product stock
                product.quantity_in_stock -= quantity_to_add
                product.save()

            # Recalculate order totals
            recalculate_order_totals(order)

            # Provide appropriate feedback
            if quantity_to_add == 0:
                messages.warning(request, f"Product '{product.name}' added with quantity 0 due to limited stock.", extra_tags='order')
            elif quantity_to_add < requested_quantity:
                messages.warning(request, f"Only {quantity_to_add} units of '{product.name}' added due to limited stock.", extra_tags='order')
            else:
                messages.success(request, f"{quantity_to_add} units of '{product.name}' successfully added.",extra_tags='order')

        return self._render_order_page(request, order, form)

    def _render_order_page(self, request, order, form):
        order_details = order.details.all().order_by('-order_date')
        total_price_before_tax = sum(detail.product.price * detail.quantity for detail in order_details)
        total_price_after_tax = total_price_before_tax * Decimal('1.13')

        # Reconstruct the product list for autocomplete
        all_products = [
            {
                'id': p['product_id'],
                'name': p['name'],
                'price': str(p['price']),
                'quantity_in_stock': p['quantity_in_stock']
            } for p in Product.objects.values('product_id', 'name', 'price', 'quantity_in_stock','item_number')
        ]

        return render(request, self.template_name, {
            'order': order,
            'form': form,
            'order_details': order_details,
            'total_price_before_tax': total_price_before_tax,
            'total_price_after_tax': total_price_after_tax,
            'all_products': all_products,
            'name_query': '',  # Clear or maintain the previous query if needed
            'search_results': [],  # Optional: keep previous manual search results if needed
        })


# Update Order Item
class UpdateOrderItemView(LoginRequiredMixin, View):
    def post(self, request, item_id):
        order_detail = get_object_or_404(OrderDetail, od_id=item_id)
        new_quantity = int(request.POST.get('quantity', 1))

        if new_quantity < 1:
            messages.error(request, "Quantity must be at least 1.", extra_tags='order')
            return redirect('create_order')

        product = order_detail.product
        quantity_difference = new_quantity - order_detail.quantity

        # Check stock availability if increasing the quantity
        if quantity_difference > 0 and product.quantity_in_stock < quantity_difference:
            messages.error(request, f"Not enough stock for {product.name}.", extra_tags='order')
            return redirect('create_order')

        with transaction.atomic():
            # Update stock
            product.quantity_in_stock -= quantity_difference
            product.save()

            # Update order detail
            order_detail.quantity = new_quantity
            order_detail.save()

            # Recalculate and update the order total
            order = order_detail.order
            recalculate_order_totals(order)

        messages.success(request, f"Order item updated successfully.", extra_tags='order')
        return redirect('create_order')

# change -> function LOL delete this 
# submit order button 
class SubmitOrderView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        if 'order_id' not in request.session:
            messages.error(request, "No active order found.")
            return redirect('create_order')

        order = get_object_or_404(Order, order_id=request.session['order_id'])

        if not order.details.exists():
            messages.error(request, "Cannot submit an empty order.")
            return redirect('create_order')

        with transaction.atomic():
            # Lock the order details
            details = order.details.select_for_update()

            for detail in details:
                product = detail.product

                # 1. Update or create RecentlyPurchasedProduct
                recently_purchased, created = RecentlyPurchasedProduct.objects.get_or_create(
                    product=product
                )
                if created:
                    recently_purchased.quantity = detail.quantity
                else:
                    recently_purchased.quantity += detail.quantity
                recently_purchased.save(update_fields=["quantity"])

                # 2. Record the checkout as a stock change
                record_stock_change(
                    product=product,
                    qty=detail.quantity,  # negative for checkout
                    change_type="checkout",
                    note=f"Order {order.order_id} submission"
                )

            # 3. Finalize order
            order.submitted = True
            order.save(update_fields=["submitted"])

            # 4. Clear session
            del request.session['order_id']

        messages.success(request, "Order submitted successfully.")
        return redirect('create_order')

# deletes item from the order
@login_required
def delete_order_item(request, item_id):
    order_detail = get_object_or_404(OrderDetail, od_id=item_id)
    order = order_detail.order
    product = order_detail.product

    if order_detail.quantity > 1:
        order_detail.quantity -= 1
        order_detail.save()

        # Return stock
        product.quantity_in_stock += 1
        product.save()
    else:
        product.quantity_in_stock += order_detail.quantity
        product.save()
        order_detail.delete()

    # Recalculate the order total
    recalculate_order_totals(order)

    messages.success(request, f"1 unit of {product.name} removed from the order.")
    return redirect('create_order')
    

# View for order success page
"""
class OrderSuccessView(View):
   template_name = 'order_success.html'
 
   def get(self, request, *args, **kwargs):
       return render(request, self.template_name)
"""
#Change - Function to annotate changes

def record_stock_change(
    product: Product,
    qty: int,
    change_type: str,
    note: str = ""
) -> None:
    """
    Creates a StockChange row *and* updates per-product counters.

    • Positive qty  -> stock added
    • Negative qty  -> stock removed
    """
    with transaction.atomic():

        # 1) persist the audit trail
        StockChange.objects.create(
            product=product,
            change_type=change_type,
            quantity=qty,
            note=note or None,
        )

        # 2) update running totals on Product
        if change_type == "checkin":
            product.stock_bought += qty  # qty is positive
        elif change_type == "checkout":
            product.stock_sold += qty    # qty is negative
        elif change_type == "expired":
            product.stock_expired += abs(qty)
        elif change_type == "error_subtract":
            product.stock_bought -= qty
        elif change_type == "error_add":
            product.stock_bought += qty
        elif change_type == "checkin_delete1":
            product.stock_bought -= qty

        # -- optional: keep other change types (return/adjustment) out of the counters,
        #             or handle them however you prefer.

        product.save(
            update_fields=["stock_bought", "stock_sold", "stock_expired"]
        )


# DELETES ON ITEM ON CHECKIN BUTTON -- CHANGE - there is a bug with the checkout technically, not a checkout but a misclick 
def delete_one(request, product_id):
    if request.method != "POST":
        return redirect("checkin")

    with transaction.atomic():
        product = get_object_or_404(Product.objects.select_for_update(), pk=product_id)

        if product.quantity_in_stock <= 0:
            messages.error(
                request,
                f"Cannot subtract. {product.name} is already out of stock.",
                extra_tags="checkin",
            )
        else:
            product.quantity_in_stock -= 1
            product.save(update_fields=["quantity_in_stock"])
            record_stock_change(
                product,
                qty=1,
                change_type="checkin_delete1",
                note="1 unit removed due to UI misclick during check-in"
            )
            messages.success(
                request,
                f"Adjusted: 1 unit removed from {product.name}'s stock.",
                extra_tags="checkin",
            )
    return redirect(f"{reverse('checkin')}?barcode={product.barcode}")


#add1 checkin
def AddQuantityView(request, product_id):
    if request.method != "POST":
        return redirect("checkin")

    try:
        quantity_to_add = int(request.POST.get("quantity_to_add", 1))
    except ValueError:
        messages.error(request, "Please enter a valid quantity.", extra_tags="checkin")
        return redirect("inventory_display")

    if quantity_to_add < 1:
        messages.error(request, "Quantity to add must be at least 1.", extra_tags="checkin")
        return redirect("inventory_display")

    with transaction.atomic():
        product = get_object_or_404(Product.objects.select_for_update(), product_id=product_id)
        product.quantity_in_stock += quantity_to_add
        product.save(update_fields=["quantity_in_stock"])
        record_stock_change(product, qty=quantity_to_add, change_type="checkin", note="Manual add via UI")

    messages.success(
        request,
        f"{quantity_to_add} unit(s) of {product.name} added to stock.",
        extra_tags="checkin",
    )

    # Redirect to GET checkin with barcode query param to avoid double-post on refresh
    return redirect(f"{reverse('checkin')}?barcode={product.barcode}")


#add products without barcode
class AddProductByIdCheckinView(LoginRequiredMixin, View):
    def post(self, request, product_id):
        quantity = int(request.POST.get("quantity", 1))

        with transaction.atomic():
            try:
                product = Product.objects.select_for_update().get(product_id=product_id)
            except Product.DoesNotExist:
                messages.error(request, "Product not found.", extra_tags="checkin")
                return redirect("checkin")

            product.quantity_in_stock += quantity
            product.save(update_fields=["quantity_in_stock"])
            record_stock_change(product, qty=quantity, change_type="checkin", note="Add via ID button")

        messages.success(
            request,
            f"{quantity} unit(s) of '{product.name}' successfully added to stock.",
            extra_tags="checkin",
        )
        return render(
            request,
            "checkin.html",
            {
                "product": product,
                "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number")),
            },
        )


# change 
# product.stock_bought += 1
#checkin views
class CheckinProductView(LoginRequiredMixin, View):
    template_name = "checkin.html"

    # GET → show search box & current stock table
    def get(self, request):
        barcode = request.GET.get("barcode")
        product = None
        if barcode:
            product = Product.objects.filter(barcode=barcode).first()
        query = request.GET.get("name_query", "").strip()
        search_results = Product.objects.filter(name__icontains=query) if query else []
        return render(
            request,
            self.template_name,
            {
                "search_results": search_results,
                "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock", "item_number")),
                "product": product,
                "categories": Category.objects.all(),
            },
        )

    # POST → barcode submitted
    def post(self, request):
        barcode = request.POST.get("barcode", "").strip()
        if not barcode:
            messages.error(request, "No barcode provided.", extra_tags="checkin")
            return self._render_no_product(request)

        with transaction.atomic():
            try:
                product = Product.objects.select_for_update().get(barcode=barcode)
            except Product.DoesNotExist:
                messages.error(request, "Product does not exist. Please add the product first.", extra_tags="checkin")
                return redirect("checkin")

            product.quantity_in_stock += 1
            product.save(update_fields=["quantity_in_stock"])
            record_stock_change(product, qty=1, change_type="checkin", note="Barcode scan")

        messages.success(request, f"1 unit of {product.name} added to stock.", extra_tags="checkin")
        # Redirect to checkin page with the product barcode as query param to show product info again
        return redirect(f"{reverse('checkin')}?barcode={product.barcode}")

    # helper
    def _render_no_product(self, request):
        return render(
            request,
            self.template_name,
            {
                "all_products": list(Product.objects.values("product_id", "name", "price", "quantity_in_stock",  "item_number")),
                "categories": Category.objects.all(),
            },
        )



   
# Edit product.
class EditProductView(View):
    template_name = 'edit_product.html'

    def get(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        form = EditProductForm(instance=product)

        # Use GET param or fallback to referring page
        next_url = request.GET.get('next') or request.META.get('HTTP_REFERER', '/inventory_display')
        return render(request, self.template_name, {
            'form': form,
            'next': next_url,
            'product': product
        })

    def post(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        old_quantity = product.quantity_in_stock  # Store old quantity for stock change
        form = EditProductForm(request.POST, instance=product)
        next_url = request.POST.get('next', '/inventory_display')

        if form.is_valid():
            updated_product = form.save(commit=False)  # Save without committing to DB yet
            new_quantity = updated_product.quantity_in_stock  # Get updated quantity
            
            if new_quantity - old_quantity != 0:
                # Record stock change only if quantity has changed
                change = "error_add" if new_quantity > old_quantity else "error_subtract"
                record_stock_change(
                    product=product,
                    qty=abs(new_quantity - old_quantity),
                    change_type=change,
                    note="Product updated via edit form"
                )
            
            form.save()  # Now save the updated product
            messages.success(request, "Product updated successfully.")
            return redirect(next_url)
        else:
            messages.error(request, "Failed to update the product.")
            return render(request, self.template_name, {
                'form': form,
                'next': next_url,
                'product': product
            })


# Add a new product
class AddProductView(LoginRequiredMixin, View):
    template_name = 'new_product.html'

    def get(self, request):
        # Capture the 'next' parameter from the query string
        next_url = request.GET.get('next', '')
        categories = Category.objects.all()

        # Initialize an empty form or prefill it with query parameters
        initial_data = {
            'name': request.GET.get('name', ''),
            'barcode': request.GET.get('barcode', ''),
            'price': request.GET.get('price', ''),
        }
        form = AddProductForm(initial=initial_data)

        return render(request, self.template_name, {
            'categories': categories,
            'form': form,  # Pass the form to the template
            'next': next_url  # Pass the next URL to the template
        })

    def post(self, request):
        form = AddProductForm(request.POST)
        # Capture the 'next' parameter from the hidden input
        next_url = request.POST.get('next', '')

        if form.is_valid():
            barcode = form.cleaned_data.get('barcode')  # Extract the barcode from the form
            try:
                # Check if a product with the same barcode already exists
                if Product.objects.filter(barcode=barcode).exists():
                    messages.error(request, f"A product with barcode '{barcode}' already exists.", extra_tags='new_product')
                    return render(request, self.template_name, {
                        'categories': Category.objects.all(),
                        'form': form,
                        'next': next_url
                    })

                # Save the product if no duplicates found
                form.save()
                messages.success(request, "Product added successfully.")
                product = Product.objects.filter(barcode=barcode).first()
                record_stock_change(product=product, qty=product.quantity_in_stock, change_type="checkin", note="New product added via form")
                
                # Redirect to the captured 'next' URL or default to 'checkin'
                return redirect(next_url) if next_url else redirect('checkin')
            except IntegrityError:
                messages.error(request, "A product with this barcode or item number already exists.", extra_tags='new_product')
            except Exception as e:
                # Handle unexpected errors gracefully
                messages.error(request, f"An unexpected error occurred: {str(e)}", extra_tags='new_product')
        else:
            messages.error(request, "Failed to add product. Please check the form fields.", extra_tags='new_product')

        # Re-render the form with categories and pass the 'next' URL
        categories = Category.objects.all()
        return render(request, self.template_name, {
            'categories': categories,
            'form': form,
            'next': next_url
        })


# Display inventory
class InventoryView(LoginRequiredMixin, View):
    template_name = 'inventory_display.html'

    def get(self, request):
        # Get filter parameters from the request
        selected_category_id = request.GET.get('category_id', '')
        barcode_query = request.GET.get('barcode_query', '')
        name_query = request.GET.get('name_query', '')
        sort_column = request.GET.get('sort', 'name')  # Default sorting column is 'name'
        sort_direction = request.GET.get('direction', 'asc')  # Default sorting direction is ascending

        # Query products based on filters
        products = Product.objects.all()
        if selected_category_id:
            products = products.filter(category_id=selected_category_id)
        if barcode_query:
            products = products.filter(barcode__icontains=barcode_query)
        if name_query:
            products = products.filter(name__icontains=name_query)

        # Apply sorting dynamically
        valid_sort_columns = ['name', 'quantity_in_stock', 'price', 'expiry_date']

        if sort_column in valid_sort_columns:
            sort_prefix = '-' if sort_direction == 'desc' else ''
            products = products.order_by(f'{sort_prefix}{sort_column}')
        else:
            # Fallback to default sort
            products = products.order_by('name')

        # Paginate the filtered products
        paginator = Paginator(products, 100)  # Show 100 items per page
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)


        # Pass all query parameters and the paginator to the template
        return render(request, self.template_name, {
            'page_obj': page_obj,
            'categories': Category.objects.all(),
            'selected_category_id': selected_category_id,
            'barcode_query': barcode_query,
            'name_query': name_query,
            'sort_column': sort_column,
            'sort_direction': sort_direction,
        })

# search bar for expired product view 
def get(self, request):
    date_filter = request.GET.get('date_filter', '')
    name_query = request.GET.get('name_query', '').strip()
    pid = request.GET.get("pid", None)

    products = self._filter_products(date_filter, name_query)
    product = Product.objects.filter(pk=pid).first() if pid else None

    return render(request, self.template_name, {
        "products": products,
        "product": product,
        "date_filter": date_filter,
        "name_query": name_query,
        "all_products": list(Product.objects.values("product_id", "name")),
    })


# Change 
class ExpiredProductView(LoginRequiredMixin, View):
    template_name = 'expired_products.html'

    def get(self, request):
        date_filter = request.GET.get('date_filter', '')
        name_query = request.GET.get('name_query', '').strip()
        pid = request.GET.get("pid", None)

        products = self._filter_products(date_filter, name_query)
        product = Product.objects.filter(pk=pid).first() if pid else None

        return render(request, self.template_name, {
            "products": products,
            "product": product,
            "date_filter": date_filter,
            "name_query": name_query,
            "all_products": list(Product.objects.values("product_id", "name")),
        })

    def post(self, request):
        barcode      = request.POST.get("barcode", "").strip()
        date_filter  = request.POST.get("date_filter", "")
        name_query   = request.POST.get("name_query", "").strip()
        product      = Product.objects.filter(barcode__iexact=barcode).first()
        products     = self._filter_products(date_filter, name_query)

        if product and request.POST.get("retire_expired") == "1":
            try:
                qty = int(request.POST.get("retire_quantity"))
            except (ValueError, TypeError):
                qty = 0

            if qty > 0 and qty <= product.quantity_in_stock:
                # Update stock
                product.quantity_in_stock -= qty
                product.save(update_fields=["quantity_in_stock", "stock_expired"])

                # Log the change
                record_stock_change(product, qty=qty, change_type="expired", note="Marked as expired from expired product view")
                messages.success(request, f"{qty} units of '{product.name}' marked as expired.")
            else:
                messages.error(request, "Invalid quantity to retire.")

        return render(request, self.template_name, {
            "products": products,
            "product": product,
            "date_filter": date_filter,
            "name_query": name_query,
            "all_products": list(Product.objects.values("product_id", "name")),
        })


    def _filter_products(self, date_filter, name_query):
        today = date.today()
        if date_filter == "1_week":
            end = today + timedelta(weeks=1)
            qs  = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        elif date_filter == "3_months":
            end = today + relativedelta(months=3)
            qs  = Product.objects.filter(expiry_date__gte=today, expiry_date__lte=end)
        else:
            qs  = Product.objects.filter(expiry_date__lt=today)

        if name_query:
            qs = qs.filter(name__icontains=name_query)
        return qs.exclude(expiry_date__isnull=True).order_by("expiry_date")
    
     
# View for displaying low-stock items
class LowStockView(AdminRequiredMixin, View):
    template_name = 'low_stock.html'
    threshold = 2


    def get(self, request):
        low_stock_products = Product.objects.filter(quantity_in_stock__lt=self.threshold).order_by('name')
        recently_purchased = RecentlyPurchasedProduct.objects.all().order_by('-order_date')


        paginator_low_stock = Paginator(low_stock_products, 100)
        page_number_low_stock = request.GET.get('page')
        page_obj_low_stock = paginator_low_stock.get_page(page_number_low_stock)


        paginator_recent = Paginator(recently_purchased, 80)
        page_number_recent = request.GET.get('page_recent')
        page_obj_recent = paginator_recent.get_page(page_number_recent)




        return render(request, self.template_name, {
            'page_obj_low_stock': page_obj_low_stock,
            'page_obj_recent': page_obj_recent,
            'threshold': self.threshold,
        })


# Delete a recently purchased product
class DeleteRecentlyPurchasedProductView(LoginRequiredMixin, View):
   def post(self, request, id):  # Use 'id' to match the model's primary key field name
       try:
           recently_purchased = RecentlyPurchasedProduct.objects.get(id=id)
           product_name = recently_purchased.product.name  # Capture the name before deletion
           recently_purchased.delete()
           messages.success(request, f"{product_name} has been deleted from the recently purchased list.")
       except RecentlyPurchasedProduct.DoesNotExist:
           messages.error(request, "The selected product does not exist in the recently purchased list.")
       return redirect('low_stock')


class DeleteAllRecentlyPurchasedView(LoginRequiredMixin, View):
   def post(self, request):
       # Delete all recently purchased products
       RecentlyPurchasedProduct.objects.all().delete()
       messages.success(request, "All recently purchased products have been deleted.")
       return redirect('low_stock')

# Delete an item
@login_required
def delete_item(request, product_id):
    product = get_object_or_404(Product, product_id=product_id)
    product.delete()
    messages.success(request, f"Product '{product.name}' has been deleted.")


    # Redirect back to inventory page with query parameters
    page = request.POST.get('page', 1)
    category_id = request.POST.get('category_id', '')
    barcode_query = request.POST.get('barcode_query', '')
    name_query = request.POST.get('name_query', '')


    redirect_url = f"{reverse('inventory_display')}?page={page}"
    if category_id:
        redirect_url += f"&category_id={category_id}"
    if barcode_query:
        redirect_url += f"&barcode_query={barcode_query}"
    if name_query:
        redirect_url += f"&name_query={name_query}"

    return redirect(redirect_url)

# Delete all orders
class DeleteAllOrdersView(LoginRequiredMixin, View):
   def post(self, request, *args, **kwargs):
       Order.objects.all().delete()
       request.session['next_order_id'] = 1
       messages.success(request, "All orders have been deleted successfully.")
       return redirect('order_view')




# Item list view
class ItemListView(LoginRequiredMixin,View):
   template_name = 'item_list.html'
   form_class = ItemForm

   def get(self, request):
       form = self.form_class()
       items = Item.objects.all()
       return render(request, self.template_name, {'form': form, 'items': items})

   def post(self, request):
       if 'delete' in request.POST:
           item_id = request.POST.get('item_id')
           item = get_object_or_404(Item, id=item_id)
           item.delete()
           messages.success(request, f"Item '{item.item_name}' has been deleted.")
           return redirect('item_list')
       elif 'update_checked' in request.POST:
           item_id = request.POST.get('item_id')
           is_checked = request.POST.get('is_checked') == 'on'
           item = get_object_or_404(Item, id=item_id)
           item.is_checked = is_checked
           item.save()
           return redirect('item_list')
       else:
           form = self.form_class(request.POST)
           if form.is_valid():
               form.save()
               return redirect('item_list')


       items = Item.objects.all()
       return render(request, self.template_name, {'form': form, 'items': items})

@login_required
def update_product_settings(request, product_id):
    if request.method == 'POST':
        product = get_object_or_404(Product, product_id=product_id)
        expiry_input = request.POST.get('expiry_date')
        taxable_input = request.POST.get('taxable')
        category_id = request.POST.get('category')

        try:
            from django.utils.dateparse import parse_date

            if expiry_input:
                product.expiry_date = parse_date(expiry_input)

            if taxable_input in ['True', 'False']:
                product.taxable = (taxable_input == 'True')

            if category_id:
                try:
                    new_category = Category.objects.get(pk=category_id)
                    product.category = new_category
                except Category.DoesNotExist:
                    pass  # silently ignore bad input

            product.save()
            messages.success(request, "Product settings updated.", extra_tags='checkin')
        except Exception as e:
            messages.error(request, "Failed to update product settings.", extra_tags='checkin')

    return redirect(request.META.get('HTTP_REFERER', 'checkin'))


