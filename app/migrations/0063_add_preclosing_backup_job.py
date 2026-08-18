from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0062_repair_recent_purchase_shortfalls'),
    ]

    operations = [
        migrations.AlterField(
            model_name='scheduledjobrun',
            name='job_key',
            field=models.CharField(
                choices=[
                    ('gsheet_preclose', 'Google Sheet pre-closing pull'),
                    ('database_backup', 'Pre-closing database backup'),
                    ('report_cleanup', 'Daily report archive cleanup'),
                ],
                max_length=40,
            ),
        ),
    ]
