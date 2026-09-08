from django import forms
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db import IntegrityError, transaction
from django.db.models import F, Prefetch, Q
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.views.generic import DetailView, FormView, ListView
from urllib.parse import urlencode, urlsplit

from .mixins import AdminRequiredMixin
from .navigation import safe_local_return_url
from .prescription_drug_learning import matching_prescription_drugs
from .models import (
    PrescriptionDrug, PrescriptionDrugLearningRecord,
    PrescriptionDrugRequestRevision, PrescriptionDrugPackage,
    PrescriptionDrugSupplierItem, Product,
)
from .prescription_drug_management import (
    DRUG_FIELDS, PACKAGE_FIELDS, SUPPLIER_FIELDS,
    StalePrescriptionDrugEdit, save_catalogue_form,
)


class PrescriptionDrugCreateForm(forms.ModelForm):
    class Meta:
        model = PrescriptionDrug
        fields = ['name', 'brand', 'strength', 'pack_size']
        labels = {'pack_size': 'Pack size (optional)'}
        help_texts = {
            'name': 'Saved in uppercase.',
            'brand': 'Saved in uppercase.',
        }
        widgets = {
            'name': forms.TextInput(attrs={'autofocus': True}),
            'strength': forms.TextInput(attrs={'placeholder': 'e.g. 250 mg/5 mL'}),
            'pack_size': forms.TextInput(attrs={'placeholder': 'e.g. 100 tablets'}),
        }

    def clean(self):
        cleaned = super().clean()
        if all(cleaned.get(field) for field in ('name', 'brand', 'strength')):
            details = {
                'name': cleaned['name'].strip().upper(),
                'brand': cleaned['brand'].strip().upper(),
                'strength': cleaned['strength'],
            }
            if matching_prescription_drugs(details).filter(dosage_form='', route='').exists():
                self.add_error('name', 'This name, brand, and strength already exist in the catalogue.')
        return cleaned


class PrescriptionDrugCreateView(AdminRequiredMixin, FormView):
    template_name = 'prescription_drug_form.html'
    form_class = PrescriptionDrugCreateForm

    def get_success_url(self):
        raw = self.request.POST.get('return_to') if self.request.method == 'POST' else self.request.GET.get('return_to')
        url = safe_local_return_url(self.request, raw, fallback_name='prescription_drugs')
        if urlsplit(url).path != reverse('prescription_drugs'):
            return reverse('prescription_drugs')
        return url

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['return_to'] = self.get_success_url()
        return context

    def form_valid(self, form):
        try:
            with transaction.atomic():
                drug = save_catalogue_form(form, actor=self.request.user)
        except IntegrityError as exc:
            constraint = getattr(getattr(exc.__cause__, 'diag', None), 'constraint_name', None)
            if constraint != 'uniq_prescription_drug_identity':
                raise
            form.add_error(None, 'This prescription drug was just added. It already exists in the catalogue.')
            return self.form_invalid(form)
        except ValidationError:
            form.add_error(None, 'This prescription drug was just added. It already exists in the catalogue.')
            return self.form_invalid(form)
        messages.success(self.request, f'Added {drug.name} to Prescription Drugs.')
        return HttpResponseRedirect(self.get_success_url())


class PrescriptionDrugListView(AdminRequiredMixin, ListView):
    model = PrescriptionDrug
    template_name = 'prescription_drugs.html'
    context_object_name = 'drugs'
    paginate_by = 50

    def get_queryset(self):
        self.query = self.request.GET.get('q', '').strip()[:200]
        drugs = super().get_queryset()
        if self.query:
            drugs = drugs.filter(
                Q(name__icontains=self.query)
                | Q(brand__icontains=self.query)
                | Q(strength__icontains=self.query)
                | Q(din__icontains=self.query)
                | Q(generic_name__icontains=self.query)
                | Q(manufacturer__icontains=self.query)
                | Q(packages__upc__icontains=self.query)
                | Q(packages__supplier_items__item_number__icontains=self.query)
            )
        return drugs.distinct().order_by('name', 'brand', 'strength', 'pk')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['query'] = self.query
        return context


def _catalogue_return(request, raw):
    candidate = safe_local_return_url(request, raw, fallback_name='prescription_drugs')
    return candidate if urlsplit(candidate).path == reverse('prescription_drugs') else reverse('prescription_drugs')


class PrescriptionDrugDetailView(AdminRequiredMixin, DetailView):
    model = PrescriptionDrug
    template_name = 'prescription_drug_detail.html'
    context_object_name = 'drug'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        drug = self.object
        packages = list(drug.packages.select_related('inventory_product').prefetch_related('supplier_items', 'inventory_product__lots'))
        for package in packages:
            product = package.inventory_product
            package.stock_product = product
            package.stock_lots = []
            if product:
                package.stock_lots = sorted(
                    (lot for lot in product.lots.all() if lot.archived_at is None),
                    key=lambda lot: (lot.expiry_date is None, str(lot.expiry_date), lot.pk),
                )
                package.lot_total = sum(lot.quantity_on_hand for lot in package.stock_lots)
                package.stock_mismatch = package.lot_total != product.quantity_in_stock
        edit_page = Paginator(drug.changes.all(), 25).get_page(self.request.GET.get('changes_page'))
        changes = list(edit_page.object_list)
        for change in changes:
            change.changes = [
                {'field': name.replace('_', ' ').capitalize(), 'before': change.before.get(name), 'after': value}
                for name, value in change.after.items() if change.before.get(name) != value
            ]
        list_return = _catalogue_return(self.request, self.request.GET.get('return_to'))
        context.update(
            packages=packages, edit_records=changes, edit_page=edit_page, list_return=list_return,
            detail_url=reverse('prescription_drug_detail', args=[drug.pk]) + '?' + urlencode({'return_to': list_return}),
            recent_requests=drug.learning_records.order_by(F('requested_at').desc(nulls_last=True), '-pk')[:10],
        )
        return context


class VersionedCatalogueForm(forms.ModelForm):
    record_version = forms.IntegerField(widget=forms.HiddenInput(), required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields['record_version'].required = True
            self.initial['record_version'] = self.instance.version


class PrescriptionDrugEditForm(VersionedCatalogueForm):
    class Meta:
        model = PrescriptionDrug
        fields = [field for field in DRUG_FIELDS if field not in {'pack_size', 'source'}]
        labels = {'din': 'DIN', 'drug_schedule': 'Schedule', 'route': 'Route of administration'}
        help_texts = {'review_status': 'Mark reviewed after checking this record against your source.', 'din': 'Eight digits, including leading zeros.'}
        widgets = {'notes': forms.Textarea(attrs={'rows': 3}), 'storage_notes': forms.Textarea(attrs={'rows': 3})}

    def clean_din(self):
        din = self.cleaned_data['din']
        if din and PrescriptionDrug.objects.filter(din=din).exclude(pk=self.instance.pk).exists():
            raise ValidationError('This DIN already belongs to a prescription drug.')
        return din

    def clean(self):
        cleaned = super().clean()
        if all(cleaned.get(field) for field in ('name', 'brand', 'strength')):
            duplicate = matching_prescription_drugs(cleaned).filter(
                dosage_form__iexact=cleaned.get('dosage_form', ''), route__iexact=cleaned.get('route', ''),
            ).exclude(pk=self.instance.pk)
            if duplicate.exists():
                self.add_error('name', 'This drug, strength, form, and route already exist in the catalogue.')
        return cleaned


class PrescriptionDrugPackageForm(VersionedCatalogueForm):
    class Meta:
        model = PrescriptionDrugPackage
        fields = list(PACKAGE_FIELDS)
        labels = {'label': 'Pack size', 'quantity': 'Units per pack', 'upc': 'UPC / barcode', 'inventory_product': 'Linked inventory product'}
        help_texts = {
            'label': 'For example, 100 tablets or 250 mL.',
            'quantity': 'Optional. Enter the amount and unit together; this does not convert inventory quantities.',
            'inventory_product': 'Link only the inventory product for this exact pack. Stock stays managed in Inventory.',
            'reorder_point': 'In the linked inventory product’s units.',
            'target_stock': 'In the linked inventory product’s units. A planning reference; orders are not placed automatically.',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['inventory_product'].queryset = Product.all_objects.filter(
            Q(archived_at__isnull=True) | Q(pk=self.instance.inventory_product_id),
        ).order_by('name', 'pk')

    def clean_label(self):
        label = self.cleaned_data['label']
        if PrescriptionDrugPackage.objects.filter(drug_id=self.instance.drug_id, label__iexact=label).exclude(pk=self.instance.pk).exists():
            raise ValidationError('This pack size already exists for this drug.')
        return label


class PrescriptionDrugSupplierForm(VersionedCatalogueForm):
    class Meta:
        model = PrescriptionDrugSupplierItem
        fields = list(SUPPLIER_FIELDS)
        labels = {'pack_cost': 'Purchase cost per pack', 'catalogue_price': 'Catalogue price per pack'}
        help_texts = {'is_preferred': 'Selecting this supplier replaces the previous preferred supplier for this pack.', 'order_multiple': 'Number of packs per ordering multiple.'}

    def _get_validation_exclusions(self):
        # Preference replacement is validated atomically under the parent lock.
        return super()._get_validation_exclusions() | {'is_preferred'}

    def clean(self):
        cleaned = super().clean()
        if cleaned.get('supplier_name') and PrescriptionDrugSupplierItem.objects.filter(
            package_id=self.instance.package_id, supplier_name__iexact=cleaned['supplier_name'],
            item_number__iexact=cleaned.get('item_number', ''),
        ).exclude(pk=self.instance.pk).exists():
            self.add_error('supplier_name', 'This supplier and item number already exist for this pack.')
        return cleaned


class PrescriptionDrugEditorView(AdminRequiredMixin, FormView):
    template_name = 'prescription_drug_edit.html'
    form_class = PrescriptionDrugEditForm
    title = 'Edit Prescription Drug'
    submit_label = 'Save changes'

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        self.drug = get_object_or_404(PrescriptionDrug, pk=self.kwargs['pk'])
        kwargs['instance'] = self.get_instance()
        return kwargs

    def get_instance(self):
        return self.drug

    def get_success_url(self):
        fallback = reverse('prescription_drug_detail', args=[self.kwargs['pk']])
        raw = self.request.POST.get('return_to') or self.request.GET.get('return_to')
        candidate = safe_local_return_url(self.request, raw, fallback_name='prescription_drugs')
        return candidate if urlsplit(candidate).path == fallback else fallback

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(drug=self.drug, return_to=self.get_success_url(), title=self.title, submit_label=self.submit_label)
        return context

    def form_valid(self, form):
        try:
            save_catalogue_form(form, actor=self.request.user)
        except StalePrescriptionDrugEdit as exc:
            form.add_error(None, str(exc))
            return self.render_to_response(self.get_context_data(form=form), status=409)
        except ValidationError as exc:
            errors = exc.message_dict if hasattr(exc, 'message_dict') else {'__all__': exc.messages}
            for field, messages_ in errors.items():
                messages_ = ['A record with these identifiers already exists. Check the existing records.' if 'Constraint ' in message else message for message in messages_]
                form.add_error(field if field in form.fields else None, messages_)
            return self.form_invalid(form)
        except IntegrityError:
            form.add_error(None, 'A record with these identifiers was just saved. Check the existing records and try again.')
            return self.form_invalid(form)
        messages.success(self.request, 'Prescription drug details saved.')
        return HttpResponseRedirect(self.get_success_url())


class PrescriptionDrugEditView(PrescriptionDrugEditorView):
    pass


class PrescriptionDrugPackageCreateView(PrescriptionDrugEditorView):
    template_name = 'prescription_drug_package_form.html'
    form_class = PrescriptionDrugPackageForm
    title = 'Add pack size'
    submit_label = 'Add pack size'

    def get_instance(self):
        return PrescriptionDrugPackage(drug=self.drug)


class PrescriptionDrugPackageUpdateView(PrescriptionDrugPackageCreateView):
    title = 'Edit pack size'
    submit_label = 'Save pack size'

    def get_instance(self):
        return get_object_or_404(PrescriptionDrugPackage, pk=self.kwargs['package_pk'], drug=self.drug)


class PrescriptionDrugSupplierCreateView(PrescriptionDrugEditorView):
    template_name = 'prescription_drug_supplier_form.html'
    form_class = PrescriptionDrugSupplierForm
    title = 'Add supplier'
    submit_label = 'Add supplier'

    def get_instance(self):
        self.package = get_object_or_404(PrescriptionDrugPackage, pk=self.kwargs['package_pk'], drug=self.drug)
        return PrescriptionDrugSupplierItem(package=self.package)


class PrescriptionDrugSupplierUpdateView(PrescriptionDrugSupplierCreateView):
    title = 'Edit supplier'
    submit_label = 'Save supplier'

    def get_instance(self):
        super().get_instance()
        return get_object_or_404(PrescriptionDrugSupplierItem, pk=self.kwargs['supplier_pk'], package=self.package)


class PrescriptionDrugHistoryView(AdminRequiredMixin, ListView):
    model = PrescriptionDrugLearningRecord
    template_name = 'prescription_drug_history.html'
    context_object_name = 'request_entries'
    paginate_by = 50

    def get_queryset(self):
        self.drug = get_object_or_404(PrescriptionDrug, pk=self.kwargs['pk'])
        return PrescriptionDrugLearningRecord.objects.filter(
            Q(drug=self.drug) | Q(revisions__drug=self.drug),
        ).distinct().select_related('drug').prefetch_related(
            Prefetch(
                'revisions',
                queryset=PrescriptionDrugRequestRevision.objects.order_by('-observed_at', '-pk'),
                to_attr='request_changes',
            ),
        ).order_by(F('requested_at').desc(nulls_last=True), '-pk')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['drug'] = self.drug
        context['return_to'] = self.request.GET.get('return_to', '')
        return context
