from decimal import Decimal

def recalculate_order_totals(order):
    order_details = order.details.all()
    total_price_before_tax = Decimal('0.00')
    total_tax = Decimal('0.00')
    tax_rate = Decimal('0.13')  # Assuming 13% tax

    for detail in order_details:
        item_price = detail.product.price * detail.quantity
        total_price_before_tax += item_price

        # Apply tax if the product is taxable
        if detail.product.taxable:
            total_tax += item_price * tax_rate

    total_price_after_tax = total_price_before_tax + total_tax

    # Update the order with the calculated totals
    order.total_price = total_price_after_tax
    order.save()

    return total_price_before_tax, total_price_after_tax
