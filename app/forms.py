from django import forms
from .models import Product, OrderDetail, Item



class EditProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            "name",
            "brand",
            "item_number",
            "price",
            "barcode",
            "quantity_in_stock",
            "category",
            "unit_size",
            "description",
            "expiry_date",
            "taxable",
            "status",
            "price_per_unit",
        ]
        widgets = {
            "expiry_date": forms.DateInput(
                attrs={
                    "type": "date",
                    "class": "expiry-input form-control",
                }
            ),
            "taxable": forms.CheckboxInput(),
            "status": forms.CheckboxInput(),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for name, field in self.fields.items():
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = "form-check-input"
            else:
                field.widget.attrs["class"] = "form-control"

       
class AddProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            "name",
            "item_number",
            "brand",
            "barcode",
            "price",
            "quantity_in_stock",
            "description",
            "category",
            "unit_size",
            "expiry_date",
            "taxable",
            "price_per_unit",
            "status",
        ]
        widgets = {
            "brand": forms.HiddenInput(),
            "expiry_date": forms.DateInput(attrs={"type": "date", "class": "form-control"}),
            "taxable": forms.CheckboxInput(),
            "status": forms.CheckboxInput(),
        }

    def clean_brand(self):
        return (self.cleaned_data.get("brand") or "Generic").strip()

    def clean_quantity_in_stock(self):
        qty = self.cleaned_data.get("quantity_in_stock") or 0
        if qty < 0:
            raise forms.ValidationError("Quantity cannot be negative.")
        return qty

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            if name in placeholders:
                field.widget.attrs["placeholder"] = placeholders[name]

            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = "form-check-input"
            else:
                field.widget.attrs["class"] = "form-control"


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