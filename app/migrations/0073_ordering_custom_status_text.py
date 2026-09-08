from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0072_inline_lot_assignment_choices'),
    ]

    operations = [
        migrations.AddField(
            model_name='orderingsheetentry',
            name='custom_status_text',
            field=models.CharField(blank=True, default='', max_length=80),
        ),
        migrations.AlterField(
            model_name='orderingsheetentry',
            name='status',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending'),
                    ('backordered', 'Back-Ordered'),
                    ('ordered', 'Ordered'),
                    ('partial_received', 'Partially Received'),
                    ('received', 'Received'),
                    ('ready', 'Ready for Pickup'),
                    ('contacted', 'Patient Contacted'),
                    ('picked_up', 'Picked Up'),
                    ('cancelled', 'Cancelled'),
                    ('not_for_sale', 'Not for Sale (Consult Pharmacist)'),
                    ('custom', 'Custom text'),
                ],
                default='pending',
                max_length=20,
            ),
        ),
        migrations.AlterField(
            model_name='orderingsheetstatusevent',
            name='from_status',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending'),
                    ('backordered', 'Back-Ordered'),
                    ('ordered', 'Ordered'),
                    ('partial_received', 'Partially Received'),
                    ('received', 'Received'),
                    ('ready', 'Ready for Pickup'),
                    ('contacted', 'Patient Contacted'),
                    ('picked_up', 'Picked Up'),
                    ('cancelled', 'Cancelled'),
                    ('not_for_sale', 'Not for Sale (Consult Pharmacist)'),
                    ('custom', 'Custom text'),
                ],
                max_length=20,
            ),
        ),
        migrations.AlterField(
            model_name='orderingsheetstatusevent',
            name='to_status',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending'),
                    ('backordered', 'Back-Ordered'),
                    ('ordered', 'Ordered'),
                    ('partial_received', 'Partially Received'),
                    ('received', 'Received'),
                    ('ready', 'Ready for Pickup'),
                    ('contacted', 'Patient Contacted'),
                    ('picked_up', 'Picked Up'),
                    ('cancelled', 'Cancelled'),
                    ('not_for_sale', 'Not for Sale (Consult Pharmacist)'),
                    ('custom', 'Custom text'),
                ],
                max_length=20,
            ),
        ),
    ]
