import django.db.models.manager
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0054_recoverable_products'),
    ]

    operations = [
        migrations.AlterModelManagers(
            name='product',
            managers=[
                ('objects', django.db.models.manager.Manager()),
                ('all_objects', django.db.models.manager.Manager()),
            ],
        ),
    ]
