from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0067_usersession_pu_slot'),
    ]

    operations = [
        migrations.AddConstraint(
            model_name='supplierorderrun',
            constraint=models.UniqueConstraint(
                fields=('vendor',),
                condition=(
                    models.Q(vendor='mck')
                    & models.Q(state__in=[
                        'starting',
                        'login',
                        'waiting_user',
                        'running',
                        'paused',
                        'review',
                    ])
                ),
                name='supplierrun_one_active_mck',
            ),
        ),
    ]
