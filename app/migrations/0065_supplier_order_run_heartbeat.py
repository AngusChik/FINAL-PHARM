from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0064_checkin_receiving_draft'),
    ]

    operations = [
        migrations.AddField(
            model_name='supplierorderrun',
            name='heartbeat_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
