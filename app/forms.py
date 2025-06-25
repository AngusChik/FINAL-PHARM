from django import forms
from .models import Product, OrderDetail, Item


class EditProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            'name', 'item_number', 'brand', 'barcode', 'price',
            'quantity_in_stock', 'description', 'category',
            'unit_size', 'expiry_date', 'taxable', 'price_per_unit'
        ]
        widgets = {
            'expiry_date': forms.DateInput(attrs={'type': 'date'}),
        }

       
class AddProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            'name',
            'item_number',
            'brand',
            'barcode',
            'price',
            'quantity_in_stock',
            'description',
            'category',
            'unit_size',
            'expiry_date',
            'taxable',
            'price_per_unit'
        ]


    # Optional: Add custom validation or widget settings if needed
    def __init__(self, *args, **kwargs):
        super(AddProductForm, self).__init__(*args, **kwargs)
        # Example: Add placeholders or customize widgets
        self.fields['name'].widget.attrs.update({'placeholder': 'Enter product name'})
        self.fields['item_number'].widget.attrs.update({'placeholder': 'Enter item number'})
        self.fields['brand'].widget.attrs.update({'placeholder': 'Enter brand name'})
        self.fields['barcode'].widget.attrs.update({'placeholder': 'Enter barcode'})
        self.fields['price'].widget.attrs.update({'placeholder': 'Enter price'})
        self.fields['quantity_in_stock'].widget.attrs.update({'placeholder': 'Enter quantity'})
        self.fields['description'].widget.attrs.update({'placeholder': 'Enter description'})
        self.fields['unit_size'].widget.attrs.update({'placeholder': 'Enter unit size'})
        self.fields['expiry_date'].widget.attrs.update({'placeholder': 'Enter expiry date'})
        self.fields['taxable'].widget.attrs.update({'placeholder': 'Enter taxable'})
        self.fields['price_per_unit'].widget.attrs.update({'placeholder': 'Enter cost per unit'})  

class OrderDetailForm(forms.ModelForm):
    class Meta:
        model = OrderDetail
        fields = ['product', 'quantity']


class BarcodeForm(forms.Form):
    barcode = forms.CharField(max_length=30, label="Scan or Enter Barcode")
    quantity = forms.IntegerField(min_value=1, label="Quantity", initial=1)


class ItemForm(forms.ModelForm):
    class Meta:
        model = Item
        fields = ['first_name', 'last_name', 'item_name', 'size', 'side', 'item_number', 'phone_number']




