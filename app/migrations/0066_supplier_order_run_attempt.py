from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0065_supplier_order_run_heartbeat'),
    ]

    operations = [
        migrations.AddField(
            model_name='supplierorderrun',
            name='attempt',
            field=models.PositiveIntegerField(default=1),
        ),
    ]
