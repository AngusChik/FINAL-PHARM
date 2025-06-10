from decimal import Decimal
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.views import View
from django.views.generic.edit import FormView
from django.contrib import messages
from django.db.models import Sum
from django.db.models import F
from django.db import transaction
from django.utils import timezone
from django.core.paginator import Paginator
from django.core.cache import cache
from app.mixins import AdminRequiredMixin
from django.contrib.auth.decorators import login_required #for @login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.mixins import UserPassesTestMixin
from urllib.parse import urlparse
from datetime import date, timedelta
from django.http import HttpResponseRedirect  # To handle redirection to the referrer
from django.db import IntegrityError  # To handle IntegrityError
from .utils import recalculate_order_totals  # Import the function
import time
from .forms import  EditProductForm, OrderDetailForm, BarcodeForm, ItemForm, AddProductForm
from .models import Item, Product, Category, Order, OrderDetail, Customer, RecentlyPurchasedProduct
from django.core.serializers import serialize
from django.forms.models import model_to_dict
from dateutil.relativedelta import relativedelta # change pip install python-dateutil



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
            messages.error(request, "Product not found.")
            return redirect('create_order')

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
            messages.warning(request, f"Product '{product.name}' added with quantity 0 due to limited stock.")
        elif quantity_to_add < quantity:
            messages.warning(request, f"Only {quantity_to_add} units of '{product.name}' added due to limited stock.")
        else:
            messages.success(request, f"{quantity_to_add} units of '{product.name}' successfully added.")

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
                'quantity_in_stock': p['quantity_in_stock']
            } for p in Product.objects.values('product_id', 'name', 'price', 'quantity_in_stock')
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
                messages.error(request, f"No product found with barcode '{barcode}'.")
                return self._render_order_page(request, order, form)

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
                messages.warning(request, f"Product '{product.name}' added with quantity 0 due to limited stock.")
            elif quantity_to_add < requested_quantity:
                messages.warning(request, f"Only {quantity_to_add} units of '{product.name}' added due to limited stock.")
            else:
                messages.success(request, f"{quantity_to_add} units of '{product.name}' successfully added.")

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
            } for p in Product.objects.values('product_id', 'name', 'price', 'quantity_in_stock')
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
            messages.error(request, "Quantity must be at least 1.")
            return redirect('create_order')

        product = order_detail.product
        quantity_difference = new_quantity - order_detail.quantity

        # Check stock availability if increasing the quantity
        if quantity_difference > 0 and product.quantity_in_stock < quantity_difference:
            messages.error(request, f"Not enough stock for {product.name}.")
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

        messages.success(request, f"Order item updated successfully.")
        return redirect('create_order')


#Submit Order
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
            for detail in order.details.all():
                recently_purchased, created = RecentlyPurchasedProduct.objects.get_or_create(
                    product=detail.product
                )
                if not created:
                    recently_purchased.quantity += detail.quantity
                else:
                    recently_purchased.quantity = detail.quantity
                recently_purchased.save()

            order.submitted = True
            order.save()

            del request.session['order_id']

        messages.success(request, "Order submitted successfully.")
        return redirect('create_order')


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

# DELETES ON ITEM ON CHECKIN BUTTON
def delete_one(request, product_id):
    if request.method == "POST":
        product = get_object_or_404(Product, pk=product_id)

        # Check if there is stock to subtract
        if product.quantity_in_stock > 0:
            product.quantity_in_stock -= 1
            product.save()
            messages.success(request, f"Successfully subtracted 1 from {product.name}'s stock.")
        else:
            messages.error(request, f"Cannot subtract. {product.name} is already out of stock.")

        # Pass the product details back to the template
        return render(request, 'checkin.html', {'product': product})

    return redirect('checkin')  # Replace 'checkin' with the actual name of your check-in view


#add1 checkin
def AddQuantityView(request, product_id):
    if request.method == 'POST':
        product = get_object_or_404(Product, product_id=product_id)

        try:
            quantity_to_add = int(request.POST.get('quantity_to_add', 1))
        except ValueError:
            messages.error(request, "Please enter a valid quantity.")
            return redirect('inventory_display')  # Safe fallback if input is invalid

        if quantity_to_add < 1:
            messages.error(request, "Quantity to add must be at least 1.")
            return redirect('inventory_display')

        # Update the product stock
        product.quantity_in_stock += quantity_to_add
        product.save()

        messages.success(request, f"{quantity_to_add} unit(s) of {product.name} added to stock.")

        # Render the check-in template with product details
        return render(request, 'checkin.html', {'product': product})

    return redirect('checkin')  # Fallback in case of invalid HTTP method


#checkin views
class CheckinProductView(LoginRequiredMixin, View):
    template_name = 'checkin.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        barcode = request.POST.get('barcode')
        if barcode:
            try:
                # Use a transaction to ensure atomic updates
                with transaction.atomic():
                    product = Product.objects.select_for_update().get(barcode=barcode)
                    product.quantity_in_stock += 1
                    product.save()

                    messages.success(request, f"1 unit of {product.name} added to stock.")
                    # Render template with product details
                    return render(request, self.template_name, {'product': product})

            except Product.DoesNotExist:
                messages.error(request, "Product does not exist. Please add the product first.")
                return redirect('checkin')
        else:
            messages.error(request, "No barcode provided.")

        return render(request, self.template_name)
   
# Edit product.
class EditProductView(View):
    template_name = 'edit_product.html'

    def get(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        form = EditProductForm(instance=product)

        # Preserve the full URL with query parameters for redirection
        next_url = request.GET.get('next', request.META.get('HTTP_REFERER', '/inventory_display'))
       
        return render(request, self.template_name, {'form': form, 'next': next_url, 'product': product})

    def post(self, request, product_id):
        product = get_object_or_404(Product, product_id=product_id)
        form = EditProductForm(request.POST, instance=product)
        next_url = request.POST.get('next', '/inventory_display')

        if form.is_valid():
            form.save()
            messages.success(request, "Product updated successfully.")
            return redirect(next_url)  # Redirect back with preserved parameters
        else:
            messages.error(request, "Failed to update the product.")
        return render(request, self.template_name, {'form': form, 'next': next_url, 'product': product})


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
                    messages.error(request, f"A product with barcode '{barcode}' already exists.")
                    return render(request, self.template_name, {
                        'categories': Category.objects.all(),
                        'form': form,
                        'next': next_url
                    })

                # Save the product if no duplicates found
                form.save()
                messages.success(request, "Product added successfully.")

                # Redirect to the captured 'next' URL or default to 'checkin'
                return redirect(next_url) if next_url else redirect('checkin')
            except IntegrityError:
                messages.error(request, "A product with this barcode or item number already exists.")
            except Exception as e:
                # Handle unexpected errors gracefully
                messages.error(request, f"An unexpected error occurred: {str(e)}")
        else:
            messages.error(request, "Failed to add product. Please check the form fields.")

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


# Change 
class ExpiredProductView(LoginRequiredMixin, View):
    template_name = 'expired_products.html'

    def get(self, request):
        # Get filter from query
        date_filter = request.GET.get('date_filter', '')
        today = date.today()

        # Build date ranges using accurate calendar months
        if date_filter == '1_week':
            end_date = today + timedelta(weeks=1)
        elif date_filter == '3_months':
            end_date = today + relativedelta(months=3)
        else:
            # Default: show already expired items
            products = Product.objects.filter(
                expiry_date__lt=today
            ).exclude(expiry_date__isnull=True).order_by('expiry_date')
            return render(request, self.template_name, {
                'products': products,
                'date_filter': date_filter,
            })

        # Only include products expiring between today and end_date
        products = Product.objects.filter(
            expiry_date__gte=today,
            expiry_date__lte=end_date
        ).exclude(expiry_date__isnull=True).order_by('expiry_date')

        return render(request, self.template_name, {
            'products': products,
            'date_filter': date_filter,
        })
    
     
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
