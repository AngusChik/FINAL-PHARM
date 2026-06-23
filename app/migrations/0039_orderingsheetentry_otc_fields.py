# Generated for OTC ordering-sheet entries + "Order for BLISTER" reasoning.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0038_dailyreportarchive'),
    ]

    operations = [
        migrations.AddField(
            model_name='orderingsheetentry',
            name='entry_type',
            field=models.CharField(choices=[('drug', 'Drug'), ('otc', 'OTC Product')], default='drug', max_length=10),
        ),
        migrations.AddField(
            model_name='orderingsheetentry',
            name='side',
            field=models.CharField(blank=True, choices=[('left', 'Left'), ('right', 'Right'), ('na', 'N/A')], default='', max_length=10),
        ),
        migrations.AddField(
            model_name='orderingsheetentry',
            name='phone_number',
            field=models.CharField(blank=True, default='', max_length=20),
        ),
        migrations.AlterField(
            model_name='orderingsheetentry',
            name='reasoning',
            field=models.CharField(blank=True, choices=[('stock', 'Order for stock'), ('basket', 'Order for basket'), ('expiring', 'Expiring'), ('blister', 'Order for BLISTER')], default='', max_length=20),
        ),
    ]
