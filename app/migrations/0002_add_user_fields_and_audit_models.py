import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('app', '0001_initial'),
    ]

    operations = [
        # Order.user
        migrations.AddField(
            model_name='order',
            name='user',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='orders',
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        # StockChange.user
        migrations.AddField(
            model_name='stockchange',
            name='user',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='stock_changes',
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        # LoginAudit model
        migrations.CreateModel(
            name='LoginAudit',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=150)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('ip_address', models.GenericIPAddressField(blank=True, null=True)),
                ('success', models.BooleanField(default=True)),
                ('user', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='login_audits',
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                'ordering': ['-timestamp'],
                'indexes': [
                    models.Index(fields=['-timestamp'], name='loginaudit_ts_idx'),
                    models.Index(fields=['user', '-timestamp'], name='loginaudit_user_ts_idx'),
                ],
            },
        ),
        # UserAction model
        migrations.CreateModel(
            name='UserAction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('action', models.CharField(choices=[
                    ('delete_product', 'Deleted Product'),
                    ('delete_order', 'Deleted Order'),
                    ('delete_all_orders', 'Deleted All Orders'),
                    ('delete_recently_purchased', 'Deleted Recently Purchased'),
                    ('delete_all_recently_purchased', 'Deleted All Recently Purchased'),
                    ('bulk_delete_recently_purchased', 'Bulk Deleted Recently Purchased'),
                    ('submit_order', 'Submitted Order'),
                    ('add_product', 'Added New Product'),
                ], max_length=40)),
                ('target', models.CharField(max_length=200)),
                ('detail', models.TextField(blank=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='user_actions',
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                'ordering': ['-timestamp'],
                'indexes': [
                    models.Index(fields=['-timestamp'], name='useraction_ts_idx'),
                ],
            },
        ),
    ]
