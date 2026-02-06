from django import forms
from .models import Product, OrderDetail, Item
from datetime import datetime, date

class EditProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            "name", "brand", "item_number", "price", "barcode",
            "quantity_in_stock", "category", "unit_size",
            "description", "expiry_date", "taxable", "status", "price_per_unit",
        ]
        widgets = {
            "expiry_date": forms.DateInput(
                attrs={
                    "type": "text", 
                    "class": "flatpickr-date",
                    "placeholder": "DD-MM-YYYY",
                },
                format='%d-%m-%Y'
            ),
            "taxable": forms.CheckboxInput(),
            "status": forms.CheckboxInput(),
        }

    def clean_expiry_date(self):
            # 1. Get the raw string exactly as it was typed/scanned
            raw_date = self.data.get('expiry_date', '').strip().rstrip('-')
            
            if not raw_date:
                return None

            # 2. Try parsing the custom format first
            formats_to_try = ['%d-%m-%Y', '%Y-%m-%d']
            for fmt in formats_to_try:
                try:
                    # Convert the string into a real Python date
                    return datetime.strptime(raw_date, fmt).date()
                except (ValueError, TypeError):
                    continue
                    
            # 3. If no formats match, Django triggers this error
            raise forms.ValidationError("Enter a valid date (DD-MM-YYYY).")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            existing_classes = field.widget.attrs.get("class", "")
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = f"{existing_classes} form-check-input".strip()
            else:
                field.widget.attrs["class"] = f"{existing_classes} form-control".strip()
       
class AddProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            "name", "item_number", "brand", "barcode", "price",
            "quantity_in_stock", "description", "category",
            "unit_size", "expiry_date", "taxable", "price_per_unit", "status",
        ]
        widgets = {
            "brand": forms.HiddenInput(),
            "expiry_date": forms.DateInput(
                attrs={
                    "type": "text",              # ✅ Required to bypass browser default
                    "class": "flatpickr-date",   # ✅ Required for the JS to find it
                    "placeholder": "DD-MM-YYYY",
                },
                format='%d-%m-%Y'                # ✅ Tells Django how to display existing data
            ),
            "taxable": forms.CheckboxInput(),
            "status": forms.CheckboxInput(),
        }

    # ✅ ADD THIS CLEAN METHOD
    def clean_expiry_date(self):
        raw_date = self.data.get('expiry_date', '').strip().rstrip('-')
        if not raw_date:
            return None
        for fmt in ('%d-%m-%Y', '%Y-%m-%d'):
            try:
                return datetime.strptime(raw_date, fmt).date()
            except (ValueError, TypeError):
                continue
        raise forms.ValidationError("Enter a valid date (DD-MM-YYYY).")

    def clean_brand(self):
        return (self.cleaned_data.get("brand") or "Generic").strip()

    def clean_quantity_in_stock(self):
        qty = self.cleaned_data.get("quantity_in_stock") or 0
        if qty < 0:
            raise forms.ValidationError("Quantity cannot be negative.")
        return qty

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        required_fields = ['name', 'category', 'item_number', 'price', 'quantity_in_stock']

        placeholders = {
            "name": "Enter product name",
            "item_number": "Enter item number",
            "barcode": "Enter barcode",
            "price": "Enter selling price",
            "quantity_in_stock": "Enter quantity",
            "description": "Enter description",
            "unit_size": "Enter unit size",
            "price_per_unit": "Enter cost per unit",
        }

        for name, field in self.fields.items():
            # Add required attribute for browser validation
            if name in required_fields:
                field.required = True
                field.widget.attrs['required'] = 'required'

            # Keep your existing class-appending logic
            existing_classes = field.widget.attrs.get("class", "")
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = f"{existing_classes} form-check-input".strip()
            else:
                field.widget.attrs["class"] = f"{existing_classes} form-control".strip()


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