# Add the boot_session action type (admin logs another user off).

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0039_orderingsheetentry_otc_fields'),
    ]

    operations = [
        migrations.AlterField(
            model_name='useraction',
            name='action',
            field=models.CharField(
                max_length=50,
                choices=[
                    ('delete_product', 'Deleted Product'),
                    ('delete_order', 'Deleted Order'),
                    ('delete_all_orders', 'Deleted All Orders'),
                    ('delete_recently_purchased', 'Deleted Recently Purchased'),
                    ('delete_all_recently_purchased', 'Deleted All Recently Purchased'),
                    ('bulk_delete_recently_purchased', 'Bulk Deleted Recently Purchased'),
                    ('submit_order', 'Submitted Order'),
                    ('add_product', 'Added New Product'),
                    ('start_session', 'Started Check-in Session'),
                    ('end_session', 'Ended Check-in Session'),
                    ('reopen_session', 'Reopened Check-in Session'),
                    ('adjust_session_line', 'Adjusted Session Line'),
                    ('remove_session_line', 'Removed Session Line'),
                    ('delete_session', 'Deleted Check-in Session'),
                    ('clear_session_history', 'Cleared Session History'),
                    ('delivery_checkin', 'Delivery Check-in'),
                    ('delivery_checkout', 'Delivery Check-out'),
                    ('delivery_undo_checkout', 'Delivery Undo Check-out'),
                    ('delivery_clear_history', 'Delivery Cleared History'),
                    ('edit_product', 'Edited Product'),
                    ('update_product_settings', 'Updated Product Settings'),
                    ('revert_label_category', 'Reverted Label Categories'),
                    ('create_account', 'Created Account'),
                    ('passkey_unlock', 'Unlocked Admin Passkey'),
                    ('clear_label_queue', 'Cleared Label Queue'),
                    ('delete_item_list', 'Deleted Item List Entry'),
                    ('add_item_list', 'Added Item List Entry'),
                    ('delivery_delete_record', 'Deleted Delivery Record'),
                    ('cycle_count', 'Cycle Count Completed'),
                    ('retire_expired', 'Retired Expired Stock'),
                    ('print_labels', 'Printed Labels'),
                    ('delete_label_session', 'Deleted Label Session'),
                    ('regenerate_label_session', 'Regenerated Label Session'),
                    ('clear_all_label_sessions', 'Cleared All Label Sessions'),
                    ('checkout_submit', 'Submitted PU Checkout'),
                    ('checkout_new', 'Started New PU Checkout'),
                    ('ordering_status_update', 'Updated Ordering Sheet Status'),
                    ('ordering_delete', 'Removed Ordering Sheet Entry'),
                    ('boot_session', 'Logged Off User'),
                ],
            ),
        ),
    ]
