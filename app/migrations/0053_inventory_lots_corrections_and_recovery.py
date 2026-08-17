from decimal import Decimal
import re

import django.db.models.deletion
import django.db.models.functions.text
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


def _barcode_key(value):
    compact = ''.join(
        ch for ch in str(value or '').strip().upper()
        if not ch.isspace() and ch != '-'
    )
    if not compact:
        return None
    if compact.isdigit():
        return compact.lstrip('0') or '0'
    return compact


def backfill_inventory_relationships(apps, schema_editor):
    Product = apps.get_model('app', 'Product')
    ProductLot = apps.get_model('app', 'ProductLot')
    StockChange = apps.get_model('app', 'StockChange')
    OrderDetail = apps.get_model('app', 'OrderDetail')
    CheckoutOrderItem = apps.get_model('app', 'CheckoutOrderItem')
    lots = []
    seen_barcode_keys = {}
    for product in Product.objects.all().iterator():
        barcode_key = _barcode_key(product.barcode)
        if barcode_key and barcode_key in seen_barcode_keys:
            raise RuntimeError(
                'Cannot enable normalized barcode uniqueness: products '
                f"#{seen_barcode_keys[barcode_key]} and #{product.pk} both map "
                f"to '{barcode_key}'. Resolve the duplicate barcodes first."
            )
        if barcode_key:
            seen_barcode_keys[barcode_key] = product.pk
        Product.objects.filter(pk=product.pk).update(
            normalized_barcode=barcode_key,
        )
        lots.append(ProductLot(
            product_id=product.pk,
            lot_number='UNASSIGNED',
            quantity_on_hand=max(0, int(product.quantity_in_stock or 0)),
        ))
    ProductLot.objects.bulk_create(lots, batch_size=500)

    # Older ledger rows only named their source transaction in a note. Link
    # those rows once so later reporting can use durable foreign keys.
    for change in StockChange.objects.filter(
        order_detail_id__isnull=True,
        change_type__in=['checkout', 'checkout_unfulfilled'],
    ).iterator():
        match = re.search(r'Order\s+(\d+)', change.note or '')
        if not match or not change.product_id:
            continue
        detail = OrderDetail.objects.filter(
            order_id=int(match.group(1)), product_id=change.product_id,
        ).first()
        if detail:
            StockChange.objects.filter(pk=change.pk).update(order_detail_id=detail.pk)
    for change in StockChange.objects.filter(
        checkout_item_id__isnull=True,
        change_type__in=['giveaway', 'giveaway_unfulfilled'],
    ).iterator():
        match = re.search(r'PU Checkout\s+(\d+)', change.note or '')
        if not match or not change.product_id:
            continue
        item = CheckoutOrderItem.objects.filter(
            checkout_id=int(match.group(1)), product_id=change.product_id,
        ).first()
        if item:
            StockChange.objects.filter(pk=change.pk).update(checkout_item_id=item.pk)


def reverse_inventory_backfill(apps, schema_editor):
    apps.get_model('app', 'ProductLot').objects.all().delete()
    apps.get_model('app', 'Product').objects.update(normalized_barcode=None)


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0052_durable_purchase_financials_and_product_indexes'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ProductLot',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lot_number', models.CharField(default='UNASSIGNED', max_length=64)),
                ('expiry_date', models.DateField(blank=True, null=True)),
                ('quantity_on_hand', models.PositiveIntegerField(default=0)),
                ('received_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('notes', models.CharField(blank=True, default='', max_length=255)),
                ('archived_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('archived_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='archived_product_lots', to=settings.AUTH_USER_MODEL)),
                ('checkin_session', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='received_lots', to='app.checkinsession')),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='lots', to='app.product')),
            ],
            options={
                'ordering': ['expiry_date', 'lot_number', 'pk'],
                'indexes': [
                    models.Index(fields=['product', 'archived_at', 'expiry_date'], name='productlot_fefo_idx'),
                    models.Index(fields=['lot_number'], name='productlot_number_idx'),
                ],
                'constraints': [
                    models.UniqueConstraint(fields=('product', 'lot_number', 'expiry_date'), name='productlot_identity_uniq', nulls_distinct=False),
                    models.CheckConstraint(condition=models.Q(('quantity_on_hand__gte', 0)), name='productlot_qty_nonnegative'),
                ],
            },
        ),
        migrations.CreateModel(
            name='SupplierPurchaseOrder',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('supplier', models.CharField(choices=[('mck', 'McKesson'), ('kf', 'Kohl & Frisch'), ('other', 'Other supplier')], max_length=10)),
                ('supplier_name', models.CharField(blank=True, default='', max_length=120)),
                ('confirmation_number', models.CharField(blank=True, default='', max_length=100)),
                ('order_date', models.DateField(default=django.utils.timezone.localdate)),
                ('expected_date', models.DateField(blank=True, null=True)),
                ('status', models.CharField(choices=[('draft', 'Draft'), ('submitted', 'Submitted'), ('partial', 'Partially received'), ('received', 'Received'), ('cancelled', 'Cancelled')], default='draft', max_length=12)),
                ('notes', models.TextField(blank=True, default='')),
                ('archived_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('archived_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='archived_supplier_purchase_orders', to=settings.AUTH_USER_MODEL)),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='supplier_purchase_orders', to=settings.AUTH_USER_MODEL)),
                ('plan', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='purchase_orders', to='app.supplierorderplan')),
            ],
            options={
                'ordering': ['-order_date', '-created_at'],
                'indexes': [
                    models.Index(fields=['status', '-order_date'], name='supplierpo_status_date_idx'),
                    models.Index(fields=['supplier', '-order_date'], name='supplierpo_supplier_idx'),
                ],
            },
        ),
        migrations.CreateModel(
            name='SupplierPurchaseOrderLine',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_name', models.CharField(max_length=200)),
                ('product_barcode', models.CharField(blank=True, default='', max_length=64)),
                ('quantity_ordered', models.PositiveIntegerField(default=1)),
                ('quantity_received', models.PositiveIntegerField(default=0)),
                ('unit_cost', models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True)),
                ('product', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='supplier_purchase_order_lines', to='app.product')),
                ('purchase_order', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='lines', to='app.supplierpurchaseorder')),
            ],
            options={
                'ordering': ['pk'],
                'constraints': [
                    models.CheckConstraint(condition=models.Q(('quantity_received__lte', models.F('quantity_ordered'))), name='supplierpo_received_not_over_ordered'),
                    models.CheckConstraint(condition=models.Q(('unit_cost__isnull', True), ('unit_cost__gte', 0), _connector='OR'), name='supplierpo_unit_cost_nonnegative'),
                ],
            },
        ),
        migrations.CreateModel(
            name='TransactionCorrection',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('correction_type', models.CharField(choices=[('return', 'Return'), ('void', 'Void'), ('correction', 'Correction')], max_length=12)),
                ('reason', models.CharField(max_length=255)),
                ('note', models.TextField(blank=True, default='')),
                ('adjustment_amount', models.DecimalField(decimal_places=2, default=Decimal('0'), help_text='Financial adjustment recorded for reporting only.', max_digits=12)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('checkout', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='corrections', to='app.checkoutorder')),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='transaction_corrections', to=settings.AUTH_USER_MODEL)),
                ('order', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='corrections', to='app.order')),
            ],
            options={
                'ordering': ['-created_at'],
                'constraints': [
                    models.CheckConstraint(condition=models.Q(models.Q(('checkout__isnull', True), ('order__isnull', False)), models.Q(('checkout__isnull', False), ('order__isnull', True)), _connector='OR'), name='correction_exactly_one_transaction'),
                    models.CheckConstraint(condition=models.Q(('adjustment_amount__gte', 0)), name='correction_amount_nonnegative'),
                ],
            },
        ),
        migrations.CreateModel(
            name='TransactionCorrectionLine',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_name', models.CharField(max_length=200)),
                ('product_barcode', models.CharField(blank=True, default='', max_length=64)),
                ('quantity', models.PositiveIntegerField()),
                ('unit_price', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('disposition', models.CharField(choices=[('restock', 'Return to stock'), ('quarantine', 'Quarantine'), ('damaged', 'Damaged'), ('expired', 'Expired'), ('no_restock', 'Do not restock')], default='restock', max_length=16)),
                ('checkout_item', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='correction_lines', to='app.checkoutorderitem')),
                ('correction', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='lines', to='app.transactioncorrection')),
                ('order_detail', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='correction_lines', to='app.orderdetail')),
                ('product', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='transaction_correction_lines', to='app.product')),
            ],
            options={
                'ordering': ['pk'],
                'constraints': [
                    models.CheckConstraint(condition=models.Q(models.Q(('checkout_item__isnull', True), ('order_detail__isnull', False)), models.Q(('checkout_item__isnull', False), ('order_detail__isnull', True)), _connector='OR'), name='correctionline_exactly_one_source'),
                    models.CheckConstraint(condition=models.Q(('quantity__gt', 0)), name='correctionline_qty_positive'),
                    models.CheckConstraint(condition=models.Q(('unit_price__gte', 0)), name='correctionline_price_nonnegative'),
                ],
            },
        ),
        migrations.CreateModel(
            name='OrderingSheetStatusEvent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('from_status', models.CharField(choices=[('pending', 'Pending'), ('backordered', 'Back-Ordered'), ('ordered', 'Ordered'), ('partial_received', 'Partially Received'), ('received', 'Received'), ('ready', 'Ready for Pickup'), ('contacted', 'Patient Contacted'), ('picked_up', 'Picked Up'), ('cancelled', 'Cancelled'), ('not_for_sale', 'Not for Sale (Consult Pharmacist)')], max_length=20)),
                ('to_status', models.CharField(choices=[('pending', 'Pending'), ('backordered', 'Back-Ordered'), ('ordered', 'Ordered'), ('partial_received', 'Partially Received'), ('received', 'Received'), ('ready', 'Ready for Pickup'), ('contacted', 'Patient Contacted'), ('picked_up', 'Picked Up'), ('cancelled', 'Cancelled'), ('not_for_sale', 'Not for Sale (Consult Pharmacist)')], max_length=20)),
                ('note', models.CharField(blank=True, default='', max_length=255)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('changed_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='ordering_status_events', to=settings.AUTH_USER_MODEL)),
                ('entry', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='status_events', to='app.orderingsheetentry')),
            ],
            options={
                'ordering': ['-created_at'],
                'indexes': [models.Index(fields=['entry', '-created_at'], name='orderingevent_entry_date_idx')],
            },
        ),
        migrations.CreateModel(
            name='ProductLotMovement',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lot_number', models.CharField(max_length=64)),
                ('expiry_date', models.DateField(blank=True, null=True)),
                ('quantity', models.PositiveIntegerField()),
                ('direction', models.CharField(choices=[('in', 'Added to lot'), ('out', 'Removed from lot')], max_length=3)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('lot', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='movements', to='app.productlot')),
                ('stock_change', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='lot_movements', to='app.stockchange')),
            ],
            options={
                'ordering': ['pk'],
                'indexes': [models.Index(fields=['lot', '-created_at'], name='lotmovement_lot_date_idx')],
                'constraints': [models.CheckConstraint(condition=models.Q(('quantity__gt', 0)), name='lotmovement_qty_positive')],
            },
        ),
        migrations.AddField(model_name='deliverycheckin', name='archive_reason', field=models.CharField(blank=True, default='', max_length=255)),
        migrations.AddField(model_name='deliverycheckin', name='archived_at', field=models.DateTimeField(blank=True, null=True)),
        migrations.AddField(model_name='deliverycheckin', name='archived_by', field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='archived_deliveries', to=settings.AUTH_USER_MODEL)),
        migrations.AddField(model_name='orderingsheetentry', name='completed_at', field=models.DateTimeField(blank=True, null=True)),
        migrations.AddField(model_name='orderingsheetentry', name='contacted_at', field=models.DateTimeField(blank=True, null=True)),
        migrations.AddField(model_name='orderingsheetentry', name='expected_date', field=models.DateField(blank=True, null=True)),
        migrations.AddField(model_name='orderingsheetentry', name='ordered_at', field=models.DateTimeField(blank=True, null=True)),
        migrations.AddField(model_name='orderingsheetentry', name='quantity_ordered', field=models.PositiveIntegerField(blank=True, null=True)),
        migrations.AddField(model_name='orderingsheetentry', name='quantity_received', field=models.PositiveIntegerField(default=0)),
        migrations.AddField(model_name='orderingsheetentry', name='received_at', field=models.DateTimeField(blank=True, null=True)),
        migrations.AddField(model_name='orderingsheetentry', name='supplier_name', field=models.CharField(blank=True, default='', max_length=120)),
        migrations.AddField(model_name='product', name='normalized_barcode', field=models.CharField(blank=True, editable=False, max_length=64, null=True)),
        migrations.AddField(model_name='recentlypurchasedproduct', name='archive_reason', field=models.CharField(blank=True, default='', max_length=255)),
        migrations.AddField(model_name='recentlypurchasedproduct', name='archived_at', field=models.DateTimeField(blank=True, null=True)),
        migrations.AddField(model_name='recentlypurchasedproduct', name='archived_by', field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='archived_recent_purchases', to=settings.AUTH_USER_MODEL)),
        migrations.AddConstraint(model_name='recentlypurchasedproduct', constraint=models.UniqueConstraint(condition=models.Q(('archived_at__isnull', True)), fields=('product',), name='recentpurchase_one_active_product')),
        migrations.AddField(model_name='stockchange', name='checkout_item', field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='stock_changes', to='app.checkoutorderitem')),
        migrations.AddField(model_name='stockchange', name='order_detail', field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='stock_changes', to='app.orderdetail')),
        migrations.AddField(model_name='stockchange', name='correction_line', field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='stock_changes', to='app.transactioncorrectionline')),
        migrations.AlterField(model_name='orderingsheetentry', name='status', field=models.CharField(choices=[('pending', 'Pending'), ('backordered', 'Back-Ordered'), ('ordered', 'Ordered'), ('partial_received', 'Partially Received'), ('received', 'Received'), ('ready', 'Ready for Pickup'), ('contacted', 'Patient Contacted'), ('picked_up', 'Picked Up'), ('cancelled', 'Cancelled'), ('not_for_sale', 'Not for Sale (Consult Pharmacist)')], default='pending', max_length=20)),
        migrations.AlterField(model_name='stockchange', name='change_type', field=models.CharField(choices=[('checkin', 'Stock Added'), ('checkout', 'Stock Removed (Sale)'), ('checkout_unfulfilled', 'Unfulfilled Sale (Stockout)'), ('expired', 'Expired Stock'), ('error_add', 'Manual Addition'), ('error_subtract', 'Manual Adjustment'), ('checkin_delete1', 'Stock Removed via Delete Button'), ('deletion', 'Product Deletion'), ('restoration', 'Product Restored'), ('giveaway', 'No Sale (Terminal)'), ('giveaway_unfulfilled', 'Unfulfilled No Sale'), ('return', 'Transaction Return — Restocked'), ('return_no_restock', 'Transaction Return — Not Restocked'), ('void', 'Transaction Void')], max_length=30)),
        migrations.AddIndex(model_name='orderingsheetentry', index=models.Index(fields=['expected_date', 'status'], name='ordering_expected_status_idx')),
        migrations.AddConstraint(model_name='category', constraint=models.UniqueConstraint(django.db.models.functions.text.Lower('name'), name='uniq_category_name_casefold')),
        migrations.AddConstraint(model_name='orderingsheetentry', constraint=models.CheckConstraint(condition=models.Q(('quantity_ordered__isnull', True), ('quantity_received__lte', models.F('quantity_ordered')), _connector='OR'), name='ordering_received_not_over_ordered')),
        migrations.AddConstraint(model_name='product', constraint=models.CheckConstraint(condition=models.Q(('quantity_in_stock__gte', 0)), name='product_stock_nonnegative')),
        migrations.AddConstraint(model_name='product', constraint=models.CheckConstraint(condition=models.Q(('price__gte', 0)), name='product_price_nonnegative')),
        migrations.AddConstraint(model_name='product', constraint=models.CheckConstraint(condition=models.Q(('price_per_unit__isnull', True), ('price_per_unit__gte', 0), _connector='OR'), name='product_cost_nonnegative')),
        migrations.RunPython(backfill_inventory_relationships, reverse_inventory_backfill),
        migrations.AlterField(model_name='product', name='normalized_barcode', field=models.CharField(blank=True, editable=False, max_length=64, null=True, unique=True)),
    ]
