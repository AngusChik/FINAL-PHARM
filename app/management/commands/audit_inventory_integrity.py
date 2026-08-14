from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Count, F, Sum

from app.inventory_services import get_or_create_lot
from app.models import (
    Product, ProductLot, SupplierPurchaseOrderLine, normalize_barcode_key,
)


class Command(BaseCommand):
    help = (
        'Audit durable inventory relationships: product totals versus lots, '
        'normalized barcodes, negative values, and supplier receiving totals.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--repair-unassigned', action='store_true',
            help='Add missing positive stock to UNASSIGNED lots. Never reduces named lots.',
        )

    def handle(self, *args, **options):
        issues = 0
        repaired = 0
        duplicate_keys = (
            Product.all_objects.exclude(normalized_barcode__isnull=True)
            .values('normalized_barcode').annotate(count=Count('pk')).filter(count__gt=1)
        )
        for row in duplicate_keys:
            issues += 1
            self.stdout.write(self.style.ERROR(
                f"Duplicate normalized barcode {row['normalized_barcode']}: {row['count']} products"
            ))

        for product in Product.all_objects.order_by('pk').iterator():
            expected_key = normalize_barcode_key(product.barcode)
            if product.normalized_barcode != expected_key:
                issues += 1
                self.stdout.write(self.style.ERROR(
                    f'Barcode key mismatch product #{product.pk} {product.name}: '
                    f'stored={product.normalized_barcode!r}, expected={expected_key!r}'
                ))
            tracked = (
                ProductLot.objects.filter(product=product, archived_at__isnull=True)
                .aggregate(total=Sum('quantity_on_hand'))['total'] or 0
            )
            expected = int(product.quantity_in_stock or 0)
            if tracked == expected:
                continue
            issues += 1
            difference = expected - tracked
            if options['repair_unassigned'] and difference > 0:
                with transaction.atomic():
                    locked = Product.objects.select_for_update().get(pk=product.pk)
                    lot = get_or_create_lot(locked)
                    lot.quantity_on_hand = F('quantity_on_hand') + difference
                    lot.save(update_fields=['quantity_on_hand', 'updated_at'])
                repaired += 1
                self.stdout.write(self.style.WARNING(
                    f'Repaired product #{product.pk} {product.name}: +{difference} UNASSIGNED'
                ))
            else:
                self.stdout.write(self.style.ERROR(
                    f'Lot mismatch product #{product.pk} {product.name}: product={expected}, lots={tracked}'
                ))

        invalid_products = Product.all_objects.filter(
            quantity_in_stock__lt=0,
        ).count() + Product.all_objects.filter(price__lt=0).count() + Product.all_objects.filter(
            price_per_unit__lt=0,
        ).count()
        if invalid_products:
            issues += invalid_products
            self.stdout.write(self.style.ERROR(
                f'{invalid_products} invalid negative product value(s) found.'
            ))

        over_received = SupplierPurchaseOrderLine.objects.filter(
            quantity_received__gt=F('quantity_ordered'),
        ).count()
        if over_received:
            issues += over_received
            self.stdout.write(self.style.ERROR(
                f'{over_received} supplier line(s) received above ordered quantity.'
            ))

        if issues == 0:
            self.stdout.write(self.style.SUCCESS('Inventory integrity audit passed.'))
        else:
            self.stdout.write(self.style.WARNING(
                f'Inventory integrity audit found {issues} issue(s); repaired {repaired}.'
            ))
