# Seniors discount flag on Order (10% off pre-tax subtotal, toggled on purchase page).

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0040_useraction_boot_session'),
    ]

    operations = [
        migrations.AddField(
            model_name='order',
            name='seniors_discount',
            field=models.BooleanField(default=False),
        ),
    ]
