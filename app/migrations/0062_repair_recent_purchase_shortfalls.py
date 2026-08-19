from collections import defaultdict
from datetime import timedelta

from django.db import migrations


# A checkout ledger row is written immediately before a newly created
# RecentlyPurchasedProduct row. On production data the gap is milliseconds;
# keep the allowance deliberately small so an event cannot drift into a later
# generation of the same product.
FORWARD_CREATION_GRACE = timedelta(seconds=5)


def match_recent_purchase_generation(generations, event_at):
    """Return the one RP generation that safely owns a ledger event.

    archived_at is the end of a generation. A restored row has archived_at
    cleared, so it can overlap a later generation in historical data; those
    ambiguous cases are intentionally skipped rather than risking corruption.
    """
    active_at_event = [
        generation
        for generation in generations
        if generation.order_date <= event_at
        and (
            generation.archived_at is None
            or event_at <= generation.archived_at
        )
    ]
    if len(active_at_event) == 1:
        return active_at_event[0]
    if active_at_event:
        return None

    created_immediately_after = [
        generation
        for generation in generations
        if event_at < generation.order_date <= event_at + FORWARD_CREATION_GRACE
        and (
            generation.archived_at is None
            or event_at <= generation.archived_at
        )
    ]
    if len(created_immediately_after) == 1:
        return created_immediately_after[0]
    return None


def plan_recent_purchase_repairs(apps):
    """Build conservative, idempotent quantity repairs without writing data."""
    RecentlyPurchasedProduct = apps.get_model(
        'app', 'RecentlyPurchasedProduct',
    )
    StockChange = apps.get_model('app', 'StockChange')

    affected_product_ids = set(
        StockChange.objects.filter(
            change_type='checkout_unfulfilled',
            product_id__isnull=False,
        ).values_list('product_id', flat=True)
    )
    if not affected_product_ids:
        return []

    generations_by_product = defaultdict(list)
    generations_by_id = {}
    generations = RecentlyPurchasedProduct.objects.filter(
        product_id__in=affected_product_ids,
    ).order_by('product_id', 'order_date', 'pk')
    for generation in generations.iterator():
        generations_by_product[generation.product_id].append(generation)
        generations_by_id[generation.pk] = generation

    ledger_totals = defaultdict(lambda: {'fulfilled': 0, 'unfulfilled': 0})
    changes = StockChange.objects.filter(
        product_id__in=generations_by_product,
        change_type__in=['checkout', 'checkout_unfulfilled'],
    ).order_by('timestamp', 'pk')
    for change in changes.iterator():
        generation = match_recent_purchase_generation(
            generations_by_product[change.product_id], change.timestamp,
        )
        if generation is None:
            continue
        quantity = max(0, int(change.quantity or 0))
        if change.change_type == 'checkout':
            ledger_totals[generation.pk]['fulfilled'] += quantity
        else:
            ledger_totals[generation.pk]['unfulfilled'] += quantity

    repairs = []
    for generation_id, totals in ledger_totals.items():
        unfulfilled = totals['unfulfilled']
        if not unfulfilled:
            continue
        generation = generations_by_id[generation_id]
        before = max(0, int(generation.quantity or 0))
        # Old code added fulfilled + unfulfilled units to RP; fixed code adds
        # only fulfilled units. Limiting the repair to ledger-backed excess
        # makes this safe to run after fixed transactions have also accumulated
        # in the same active row.
        ledger_backed_excess = max(0, before - totals['fulfilled'])
        deduction = min(unfulfilled, ledger_backed_excess)
        if not deduction:
            continue
        repairs.append({
            'row_id': generation.pk,
            'product_id': generation.product_id,
            'before': before,
            'after': before - deduction,
            'deduction': deduction,
            'archived': generation.archived_at is not None,
        })
    return repairs


def repair_recent_purchase_shortfalls(apps, schema_editor):
    RecentlyPurchasedProduct = apps.get_model(
        'app', 'RecentlyPurchasedProduct',
    )
    for repair in plan_recent_purchase_repairs(apps):
        RecentlyPurchasedProduct.objects.filter(pk=repair['row_id']).update(
            quantity=repair['after'],
        )


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0061_repair_unfulfilled_sale_totals'),
    ]

    operations = [
        migrations.RunPython(
            repair_recent_purchase_shortfalls,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
