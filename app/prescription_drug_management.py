"""Transactional drug master edits with version checks and durable audit records."""

from decimal import Decimal

from django.db import transaction

from .models import (
    PrescriptionDrug, PrescriptionDrugPackage, PrescriptionDrugSupplierItem,
    PrescriptionDrugChange,
)


DRUG_FIELDS = (
    'name', 'brand', 'strength', 'pack_size', 'din', 'generic_name', 'manufacturer',
    'dosage_form', 'route', 'drug_schedule', 'storage_notes', 'notes', 'status',
    'review_status', 'source',
)
PACKAGE_FIELDS = (
    'label', 'quantity', 'unit', 'upc', 'inventory_product', 'location',
    'reorder_point', 'target_stock', 'is_active',
)
SUPPLIER_FIELDS = (
    'supplier_name', 'item_number', 'pack_cost', 'catalogue_price',
    'order_multiple', 'is_preferred', 'is_active',
)


class StalePrescriptionDrugEdit(Exception):
    pass


def catalogue_snapshot(obj):
    fields = DRUG_FIELDS if isinstance(obj, PrescriptionDrug) else PACKAGE_FIELDS if isinstance(obj, PrescriptionDrugPackage) else SUPPLIER_FIELDS
    result = {}
    for name in fields:
        value = obj._meta.get_field(name).value_from_object(obj)
        result[name] = str(value) if isinstance(value, Decimal) else value
    return result


def record_catalogue_change(obj, before, *, actor=None, actor_label=None):
    after = catalogue_snapshot(obj)
    if before == after:
        return
    if isinstance(obj, PrescriptionDrug):
        drug_id, entity_type = obj.pk, 'drug'
    elif isinstance(obj, PrescriptionDrugPackage):
        drug_id, entity_type = obj.drug_id, 'package'
    else:
        drug_id, entity_type = obj.package.drug_id, 'supplier'
    PrescriptionDrugChange.objects.create(
        drug_id=drug_id, entity_type=entity_type, entity_id=obj.pk,
        action='updated' if before else 'created', actor=actor,
        actor_label=(actor_label or (actor.get_username() if actor else 'System'))[:150],
        before=before, after=after,
    )


@transaction.atomic
def save_catalogue_form(form, actor):
    """Lock parents first; never persist posted stock or demand summaries."""
    submitted = form.instance
    model = type(submitted)
    if isinstance(submitted, PrescriptionDrug):
        drug_id = submitted.pk
    elif isinstance(submitted, PrescriptionDrugPackage):
        drug_id = submitted.drug_id
    else:
        drug_id = submitted.package.drug_id
    if drug_id:
        PrescriptionDrug.objects.select_for_update().get(pk=drug_id)
    if isinstance(submitted, PrescriptionDrugSupplierItem):
        PrescriptionDrugPackage.objects.select_for_update().get(pk=submitted.package_id, drug_id=drug_id)

    if submitted.pk:
        obj = model.objects.select_for_update().get(pk=submitted.pk)
        if form.cleaned_data.get('record_version') != obj.version:
            raise StalePrescriptionDrugEdit('This record changed after you opened it. Reload the page and apply your changes again.')
        before = catalogue_snapshot(obj)
        for field in form._meta.fields:
            setattr(obj, field, form.cleaned_data[field])
    else:
        obj, before = submitted, {}

    if isinstance(obj, PrescriptionDrugSupplierItem) and obj.is_preferred:
        previous_items = PrescriptionDrugSupplierItem.objects.select_for_update().filter(
            package_id=obj.package_id, is_preferred=True,
        ).exclude(pk=obj.pk)
        for previous in previous_items:
            previous_snapshot = catalogue_snapshot(previous)
            previous.is_preferred = False
            previous.version += 1
            previous.save(update_fields=['is_preferred', 'version', 'updated_at'])
            record_catalogue_change(previous, previous_snapshot, actor=actor)

    obj.full_clean()
    if before == catalogue_snapshot(obj):
        return obj
    if before:
        obj.version += 1
        obj.save(update_fields=[*form._meta.fields, 'version', 'updated_at'])
    else:
        obj.save()
    record_catalogue_change(obj, before, actor=actor)

    # Keep the original pack_size field useful for legacy callers. Package
    # records become the editable source once a structured pack exists.
    if isinstance(obj, PrescriptionDrug) and not before and obj.pack_size:
        package = PrescriptionDrugPackage.objects.create(drug=obj, label=obj.pack_size)
        record_catalogue_change(package, {}, actor=actor)
    elif isinstance(obj, PrescriptionDrugPackage):
        drug = PrescriptionDrug.objects.get(pk=obj.drug_id)
        preferred_label = drug.packages.filter(is_active=True).order_by('pk').values_list('label', flat=True).first() or ''
        if drug.pack_size != preferred_label:
            drug_before = catalogue_snapshot(drug)
            drug.pack_size = preferred_label
            drug.version += 1
            drug.save(update_fields=['pack_size', 'version', 'updated_at'])
            record_catalogue_change(drug, drug_before, actor=actor)
    return obj
