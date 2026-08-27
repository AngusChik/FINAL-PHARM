from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('app', '0068_one_active_mckesson_run'),
    ]

    operations = [
        migrations.AddField(
            model_name='supplierorderplan',
            name='mckesson_recovery_claimed_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='supplierorderplan',
            name='mckesson_recovery_claimed_by',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='mckesson_recovery_plans',
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
