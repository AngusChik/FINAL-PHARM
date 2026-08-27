import re
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .models import Category, Order, OrderDetail, Product


class TransactionNoticeStyleTests(SimpleTestCase):
    def test_notice_badge_uses_red_pill_styling(self):
        template = (
            Path(__file__).resolve().parent / 'templates' / 'order_view.html'
        ).read_text(encoding='utf-8')
        shared_rules = re.search(
            r'\.tx-page \.current-pill,\s*\.tx-page \.notice-pill\{([^}]+)\}',
            template,
        )
        notice_rule_blocks = re.findall(
            r'\.tx-page \.notice-pill\{([^}]+)\}',
            template,
        )

        self.assertIsNotNone(shared_rules)
        self.assertIn('border-radius: 999px', shared_rules.group(1))
        self.assertGreaterEqual(len(notice_rule_blocks), 2)
        notice_rules = notice_rule_blocks[-1]
        self.assertIn('background: #fef2f2', notice_rules)
        self.assertIn('color: #b91c1c', notice_rules)
        self.assertIn('border: 1px solid #fca5a5', notice_rules)


@override_settings(AXES_ENABLED=False)
class TransactionNoticeTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='transaction-notice-user',
            password='test-pass',
        )
        self.category = Category.objects.create(name='Notice tests')
        self.product = Product.objects.create(
            name='Notice Product',
            barcode='NOTICE-001',
            price=Decimal('10.00'),
            quantity_in_stock=10,
            category=self.category,
        )
        self.client.force_login(self.user)

    def create_order(self, *, expiry_at_sale=None, quantity=1):
        order = Order.objects.create(
            user=self.user,
            submitted=True,
            subtotal=Decimal('10.00'),
            total_price=Decimal('10.00'),
            financial_snapshot_source=Order.SNAPSHOT_CAPTURED,
        )
        OrderDetail.objects.create(
            order=order,
            product=self.product,
            product_name=self.product.name,
            product_barcode=self.product.barcode,
            quantity=quantity,
            price=Decimal('10.00'),
            taxable_at_sale=False,
            expiry_at_sale=expiry_at_sale,
        )
        return order

    @staticmethod
    def row_for(response, order):
        return next(
            row for row in response.context['page_obj'].object_list
            if row['source'] == 'pos' and row['id'] == order.pk
        )

    def test_expired_at_sale_order_has_notice_immediately_after_id(self):
        order = self.create_order(expiry_at_sale=date.today() - timedelta(days=1))
        session = self.client.session
        session['order_id'] = order.pk
        session.save()

        response = self.client.get(reverse('order_view'))

        self.assertTrue(self.row_for(response, order)['requires_notice'])
        html = response.content.decode()
        detail_url = re.escape(reverse('order_detail', args=[order.pk]))
        self.assertRegex(
            html,
            rf'href="{detail_url}" class="order-link">\s*#{order.pk}\s*</a>\s*'
            rf'<span class="notice-pill"[^>]*>Notice</span>\s*'
            rf'<span class="current-pill">Current</span>',
        )

    def test_unexpired_order_has_no_notice(self):
        order = self.create_order(expiry_at_sale=date.today() + timedelta(days=30))

        response = self.client.get(reverse('order_view'))

        self.assertFalse(self.row_for(response, order)['requires_notice'])
        self.assertNotContains(response, 'class="notice-pill"')

    def test_zero_quantity_expired_stockout_has_no_notice(self):
        order = self.create_order(
            expiry_at_sale=date.today() - timedelta(days=1),
            quantity=0,
        )

        response = self.client.get(reverse('order_view'))

        self.assertFalse(self.row_for(response, order)['requires_notice'])
        self.assertNotContains(response, 'class="notice-pill"')

    def test_sale_snapshot_takes_precedence_over_changed_product_expiry(self):
        order = self.create_order(expiry_at_sale=date.today() + timedelta(days=30))
        self.product.expiry_date = date.today() - timedelta(days=1)
        self.product.save(update_fields=['expiry_date'])

        response = self.client.get(reverse('order_view'))

        self.assertFalse(self.row_for(response, order)['requires_notice'])

    def test_legacy_line_falls_back_to_product_expiry(self):
        order = self.create_order(expiry_at_sale=None)
        self.product.expiry_date = date.today() - timedelta(days=1)
        self.product.save(update_fields=['expiry_date'])

        response = self.client.get(reverse('order_view'))

        self.assertTrue(self.row_for(response, order)['requires_notice'])
