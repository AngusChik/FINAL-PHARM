from decimal import Decimal, ROUND_HALF_UP

from django.db import migrations
from django.db.models import Sum


MONEY = Decimal('0.01')


def repair_unfulfilled_sale_totals(apps, schema_editor):
    """Remove stockout quantities from historical customer-facing sale totals.

    The missed demand remains permanently available in checkout_unfulfilled
    StockChange rows. Fully unfulfilled OrderDetail rows are removed after their
    ledger references are detached; partially fulfilled rows retain only the
    quantity that actually left inventory.
    """
    Order = apps.get_model('app', 'Order')
    OrderDetail = apps.get_model('app', 'OrderDetail')
    StockChange = apps.get_model('app', 'StockChange')

    affected_order_ids = set()
    details = OrderDetail.objects.filter(
        stock_changes__change_type='checkout_unfulfilled',
    ).distinct()
    for detail in details.iterator():
        fulfilled = (
            StockChange.objects.filter(
                order_detail_id=detail.pk,
                change_type='checkout',
            ).aggregate(total=Sum('quantity'))['total']
            or 0
        )
        fulfilled = max(0, int(fulfilled))
        if fulfilled >= detail.quantity:
            continue
        affected_order_ids.add(detail.order_id)
        if fulfilled:
            OrderDetail.objects.filter(pk=detail.pk).update(quantity=fulfilled)
        else:
            StockChange.objects.filter(order_detail_id=detail.pk).update(
                order_detail_id=None,
            )
            OrderDetail.objects.filter(pk=detail.pk).delete()

    for order in Order.objects.filter(pk__in=affected_order_ids).iterator():
        subtotal = Decimal('0.00')
        taxable_subtotal = Decimal('0.00')
        for line in OrderDetail.objects.filter(order_id=order.pk).iterator():
            line_total = line.price * line.quantity
            subtotal += line_total
            if line.taxable_at_sale is True:
                taxable_subtotal += line_total

        subtotal = subtotal.quantize(MONEY, rounding=ROUND_HALF_UP)
        discount = Decimal('0.00')
        taxable_base = taxable_subtotal
        if order.seniors_discount:
            discount = (subtotal * Decimal('0.10')).quantize(
                MONEY, rounding=ROUND_HALF_UP,
            )
            taxable_base = taxable_subtotal * Decimal('0.90')
        tax_rate = Decimal(order.tax_rate or Decimal('0.1300'))
        tax = (taxable_base * tax_rate).quantize(MONEY, rounding=ROUND_HALF_UP)
        total = (subtotal - discount + tax).quantize(
            MONEY, rounding=ROUND_HALF_UP,
        )
        Order.objects.filter(pk=order.pk).update(
            subtotal=subtotal,
            discount_amount=discount,
            tax=tax,
            total_price=total,
        )


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0060_special_order_recovery'),
    ]

    operations = [
        migrations.RunPython(
            repair_unfulfilled_sale_totals,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
