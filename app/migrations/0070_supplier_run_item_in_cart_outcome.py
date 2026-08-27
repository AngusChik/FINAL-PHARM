from django.db import migrations, models


LEGACY_MCKESSON_IN_CART_REASON = (
    'already present in the current order — left as-is'
)


def classify_existing_mckesson_cart_items(apps, schema_editor):
    SupplierOrderRunItem = apps.get_model('app', 'SupplierOrderRunItem')
    SupplierOrderRunItem.objects.filter(
        run__vendor='mck',
        outcome='skipped',
        reason=LEGACY_MCKESSON_IN_CART_REASON,
    ).update(outcome='in_cart')


def restore_legacy_skipped_outcome(apps, schema_editor):
    SupplierOrderRunItem = apps.get_model('app', 'SupplierOrderRunItem')
    SupplierOrderRunItem.objects.filter(
        outcome='in_cart',
    ).update(outcome='skipped')


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0069_mckesson_recovery_claim'),
    ]

    operations = [
        migrations.AlterField(
            model_name='supplierorderrunitem',
            name='outcome',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending'),
                    ('added', 'Added'),
                    ('in_cart', 'Already in current cart'),
                    ('skipped', 'Not added'),
                ],
                default='pending',
                max_length=10,
            ),
        ),
        migrations.RunPython(
            classify_existing_mckesson_cart_items,
            restore_legacy_skipped_outcome,
        ),
    ]
