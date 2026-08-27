from django.db import migrations, models


def assign_existing_pu_slots(apps, schema_editor):
    """Give the six freshest existing regular sessions distinct identities."""
    UserSession = apps.get_model('app', 'UserSession')
    sessions = (
        UserSession.objects
        .filter(user__is_staff=False)
        .order_by('-last_activity', 'pk')[:6]
    )
    for slot, session in enumerate(sessions, start=1):
        session.pu_slot = slot
        session.save(update_fields=['pu_slot'])


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0066_supplier_order_run_attempt'),
    ]

    operations = [
        migrations.AddField(
            model_name='usersession',
            name='pu_slot',
            field=models.PositiveSmallIntegerField(
                blank=True,
                null=True,
                unique=True,
            ),
        ),
        migrations.RunPython(assign_existing_pu_slots, migrations.RunPython.noop),
    ]
