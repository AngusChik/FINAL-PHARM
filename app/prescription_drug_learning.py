"""Build reference records from explicit ordering-sheet drug labels only."""

from django.core.exceptions import ValidationError
from django.db import models, transaction
from django.db.models.functions import JSONObject, Lower, Replace, Trim

from .models import (
    OrderingSheetEntry, PrescriptionDrug, PrescriptionDrugLearningRecord,
    PrescriptionDrugRequestRevision,
    PrescriptionDrugChange,
)
from .prescription_drug_parser import parse_prescription_drug_label
from .prescription_drug_quantities import parse_requested_quantity
from .prescription_drug_management import record_catalogue_change


PARSER_VERSION = 3


def pending_prescription_drug_entries():
    """Ordering rows are the durable work queue, including imports and edits."""
    catalogue_revision = PrescriptionDrugChange.objects.filter(entity_type='drug').exclude(
        actor_label='Ordering sheet', actor__isnull=True,
    ).aggregate(latest=models.Max('pk'))['latest'] or 0
    return (
        OrderingSheetEntry.objects
        .filter(
            models.Q(entry_type=OrderingSheetEntry.ENTRY_DRUG)
            | models.Q(prescription_learning__isnull=False)
        )
        .annotate(current_snapshot=JSONObject(
            entry_id=models.F('pk'), name=models.F('name'),
            quantity_needed=models.F('quantity_needed'),
            quantity_remaining=models.F('quantity_remaining'),
            quantity_ordered=models.F('quantity_ordered'),
            quantity_received=models.F('quantity_received'),
            status=models.F('status'), is_deleted=models.F('is_deleted'),
            entry_type=models.F('entry_type'),
        ))
        .annotate(current_catalogue_revision=models.Value(catalogue_revision, output_field=models.BigIntegerField()))
        .exclude(
            prescription_learning__source_snapshot=models.F('current_snapshot'),
            prescription_learning__parser_version=PARSER_VERSION,
            prescription_learning__catalogue_revision=catalogue_revision,
        )
        .only('pk', 'name', 'created_at')
        .order_by('pk')
    )


def matching_prescription_drugs(details):
    # Match the database uniqueness constraint, including manually entered
    # strengths such as "20MG" versus the parsed display form "20 mg".
    return PrescriptionDrug.objects.annotate(
        identity_name=Lower(Trim('name')),
        identity_brand=Lower(Trim('brand')),
        identity_strength=Lower(Replace(Trim('strength'), models.Value(' '), models.Value(''))),
    ).filter(
        identity_name=details['name'].strip().lower(),
        identity_brand=details['brand'].strip().lower(),
        identity_strength=details['strength'].strip().replace(' ', '').lower(),
    )


def _validated_details(label):
    details, reason = parse_prescription_drug_label(label)
    if details is None:
        return None, reason
    candidate = PrescriptionDrug(**details)
    try:
        candidate.full_clean(validate_unique=False, validate_constraints=False)
    except ValidationError:
        return None, 'invalid_details'
    return {
        'name': candidate.name, 'brand': candidate.brand, 'strength': candidate.strength,
    }, ''


def _details_for_entry(entry):
    if entry.current_snapshot['entry_type'] != OrderingSheetEntry.ENTRY_DRUG:
        return None, 'ineligible_source'
    return _validated_details(entry.name)


def _refresh_request_totals(drug_ids):
    """Replace summaries from latest source records, never from revisions.

    Lock drug rows in a stable order before querying totals. A concurrent
    worker waits here and then sees the first worker's committed source rows.
    Cancelled, completed, and archived requests remain historical requests.
    """
    drugs = list(PrescriptionDrug.objects.filter(pk__in=drug_ids).order_by('pk').select_for_update())
    records = PrescriptionDrugLearningRecord.objects.filter(drug_id__in=drug_ids)
    counts = {
        row['drug_id']: row for row in records.values('drug_id').annotate(
            request_count=models.Count('pk'),
            unknown_count=models.Count('pk', filter=models.Q(quantity_needed__isnull=True)),
        )
    }
    totals = {}
    for row in records.filter(quantity_needed__isnull=False).values('drug_id', 'quantity_unit').annotate(total=models.Sum('quantity_needed')):
        totals.setdefault(row['drug_id'], {})[row['quantity_unit']] = format(row['total'], 'f')
    for drug in drugs:
        summary = counts.get(drug.pk, {})
        PrescriptionDrug.objects.filter(pk=drug.pk).update(
            total_quantity_needed=totals.get(drug.pk, {}),
            request_count=summary.get('request_count', 0),
            unknown_quantity_count=summary.get('unknown_count', 0),
        )


def learn_prescription_drugs(*, batch_size=100):
    """Process one bounded batch atomically; concurrent workers skip locked rows.

    The source snapshot is written with the catalogue result. A stopped or
    failed pass remains eligible for retry. Only drug and quantity/lifecycle
    details are read; patient/contact and note fields are never read or copied.
    Ordering-sheet rows are never rewritten.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    result = dict(processed=0, created=0, matched=0, skipped=0)
    with transaction.atomic():
        affected_drugs = set()
        entries = list(
            pending_prescription_drug_entries()
            .select_for_update(skip_locked=True, of=('self',))[:batch_size]
        )
        for entry in entries:
            previous = PrescriptionDrugLearningRecord.objects.filter(entry=entry).first()
            details, reason = _details_for_entry(entry)
            drug = None
            if details is None:
                result['skipped'] += 1
            else:
                matches = list(matching_prescription_drugs(details)[:2])
                if len(matches) > 1:
                    reason = 'ambiguous'
                    result['skipped'] += 1
                else:
                    if matches:
                        drug, created = matches[0], False
                    else:
                        # The blank form/route identity is unique even when
                        # distinct manually identified dosage forms coexist.
                        drug, created = matching_prescription_drugs(details).filter(
                            dosage_form='', route='',
                        ).get_or_create(defaults={**details, 'source': 'ordering_sheet'})
                    if created:
                        record_catalogue_change(drug, {}, actor_label='Ordering sheet')
                    result['created' if created else 'matched'] += 1
                    affected_drugs.add(drug.pk)
            if previous and previous.drug_id:
                affected_drugs.add(previous.drug_id)
            snapshot = entry.current_snapshot
            quantity, unit = parse_requested_quantity(snapshot['quantity_needed'])
            record, _created = PrescriptionDrugLearningRecord.objects.update_or_create(
                entry=entry,
                defaults={
                    'source_name': entry.name, 'parser_version': PARSER_VERSION,
                    'catalogue_revision': entry.current_catalogue_revision,
                    'source_snapshot': snapshot, 'requested_at': entry.created_at,
                    'quantity_needed': quantity, 'quantity_unit': unit,
                    'drug': drug, 'reason': reason,
                },
            )
            if (
                previous is None or previous.source_snapshot != snapshot
                or previous.drug_id != record.drug_id
                or previous.quantity_needed != quantity or previous.quantity_unit != unit
            ):
                PrescriptionDrugRequestRevision.objects.create(
                    learning_record=record, drug=drug,
                    snapshot={
                        **snapshot,
                        'quantity_needed_value': str(quantity) if quantity is not None else None,
                        'quantity_unit': unit,
                    },
                )
            result['processed'] += 1
        if affected_drugs:
            _refresh_request_totals(affected_drugs)
    return result


def preview_prescription_drugs():
    """Read-only counts for all currently pending rows; no raw labels in output."""
    result = dict(processed=0, created=0, matched=0, skipped=0)
    identities = set()
    for entry in pending_prescription_drug_entries().iterator(chunk_size=100):
        details, _reason = _details_for_entry(entry)
        result['processed'] += 1
        if details is None:
            result['skipped'] += 1
            continue
        identity = (
            details['name'].strip().lower(), details['brand'].strip().lower(),
            details['strength'].strip().replace(' ', '').lower(),
        )
        matches = list(matching_prescription_drugs(details).values_list('pk', flat=True)[:2])
        if len(matches) > 1:
            result['skipped'] += 1
        elif identity in identities or matches:
            result['matched'] += 1
        else:
            result['created'] += 1
            identities.add(identity)
    return result
