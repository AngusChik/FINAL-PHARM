"""
Fix OrderDetail CASCADE deletion bug.

When a Product was deleted, all OrderDetail rows referencing it were
CASCADE-deleted too, making historical orders appear empty.

Changes:
- OrderDetail.product: CASCADE → SET_NULL (null=True)
- Add product_name field to preserve name after product deletion
- Add product_barcode field to preserve barcode after product deletion
- Backfill product_name/product_barcode from existing product FK
"""

from django.db import migrations, models
import django.db.models.deletion


def backfill_product_names(apps, schema_editor):
    """Copy product name & barcode into OrderDetail for existing rows."""
    OrderDetail = apps.get_model("app", "OrderDetail")
    for detail in OrderDetail.objects.select_related("product").filter(product__isnull=False):
        changed = False
        if detail.product_name == "Unknown Product":
            detail.product_name = detail.product.name
            changed = True
        if not detail.product_barcode and detail.product.barcode:
            detail.product_barcode = detail.product.barcode
            changed = True
        if changed:
            detail.save(update_fields=["product_name", "product_barcode"])


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0007_order_submitted"),
    ]

    operations = [
        # 1. Add product_name field (with default so existing rows aren't broken)
        migrations.AddField(
            model_name="orderdetail",
            name="product_name",
            field=models.CharField(default="Unknown Product", max_length=200),
        ),
        # 2. Add product_barcode field
        migrations.AddField(
            model_name="orderdetail",
            name="product_barcode",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        # 3. Change product FK from CASCADE to SET_NULL
        migrations.AlterField(
            model_name="orderdetail",
            name="product",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="app.product",
            ),
        ),
        # 4. Backfill product_name and product_barcode from the FK
        migrations.RunPython(
            backfill_product_names,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
