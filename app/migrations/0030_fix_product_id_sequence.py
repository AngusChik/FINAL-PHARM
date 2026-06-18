from django.db import migrations


def fix_product_id_sequence(apps, schema_editor):
    """Realign the Product primary-key sequence with the table's max id.

    Bulk CSV imports (DbUpdate.py / import.py) insert rows with EXPLICIT
    product_id values, which does not advance the Postgres sequence. The
    sequence can therefore lag far behind MAX(product_id), so the next
    Product.objects.create() collides on the primary key. This bumps the
    sequence so the next auto id is MAX(product_id) + 1. Idempotent.
    """
    connection = schema_editor.connection
    if connection.vendor != "postgresql":
        return
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_get_serial_sequence('app_product', 'product_id')")
        row = cursor.fetchone()
        seq = row[0] if row else None
        if not seq:
            return
        cursor.execute("SELECT COALESCE(MAX(product_id), 0) FROM app_product")
        max_id = cursor.fetchone()[0]
        if max_id and max_id > 0:
            # is_called=true → next nextval() returns max_id + 1
            cursor.execute("SELECT setval(%s, %s, true)", [seq, max_id])
        else:
            # Empty table → next nextval() returns 1
            cursor.execute("SELECT setval(%s, 1, false)", [seq])


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0029_product_stock_deleted_stockchange_product_barcode_and_more'),
    ]

    operations = [
        # Sequence state isn't reversible in a meaningful way; reverse is a no-op.
        migrations.RunPython(fix_product_id_sequence, migrations.RunPython.noop),
    ]
