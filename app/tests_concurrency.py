from decimal import Decimal
from datetime import timedelta

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils.timezone import now

from .models import Category, CheckinSession, CheckoutOrder, Product


@override_settings(AXES_ENABLED=False)
class GuardedMutationTests(TestCase):
    def setUp(self):
        self.first = User.objects.create_user(
            username='lock-first', password='pass1234', is_staff=True,
        )
        self.second = User.objects.create_user(
            username='lock-second', password='pass1234', is_staff=True,
        )
        category = Category.objects.create(name='Concurrency')
        self.product = Product.objects.create(
            name='Locked Product', barcode='LOCK-001', price=Decimal('5.00'),
            quantity_in_stock=2, category=category,
        )

    def _client(self, user):
        client = Client()
        client.force_login(user)
        return client

    def test_purchase_takeover_blocks_stale_tab_mutation(self):
        first = self._client(self.first)
        second = self._client(self.second)
        self.assertEqual(first.get(reverse('create_order')).status_code, 200)

        blocked = second.post(
            reverse('add_product_by_id', args=[self.product.pk]),
            {'quantity': '1'},
        )

        self.assertEqual(blocked.status_code, 409)
        self.assertFalse(
            self.second.orders.filter(submitted=False).exclude(draft_cart={}).exists()
        )

        takeover = second.post(
            reverse('presence_takeover'),
            {'page': reverse('create_order')},
        )
        self.assertEqual(takeover.status_code, 200)
        stale_write = first.post(
            reverse('add_product_by_id', args=[self.product.pk]),
            {'quantity': '1'},
        )
        self.assertEqual(stale_write.status_code, 409)

    def test_checkin_takeover_blocks_stale_quantity_edit(self):
        session = CheckinSession.objects.create(user=self.first, scanned_by='LF')
        first = self._client(self.first)
        second = self._client(self.second)
        page = reverse('checkin_session', args=[session.pk])
        self.assertEqual(first.get(page).status_code, 200)

        blocked = second.post(
            reverse('add_quantity', args=[session.pk, self.product.pk]),
            {'amount': '1'},
        )

        self.assertEqual(blocked.status_code, 409)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 2)

    def test_checkout_continue_and_write_respect_live_owner(self):
        owner = self._client(self.first)
        other = self._client(self.second)
        owner.get(reverse('dashboard'))
        other.get(reverse('dashboard'))
        checkout = CheckoutOrder.objects.create(
            user=self.first,
            status=CheckoutOrder.STATUS_DRAFT,
            active_session_key=owner.session.session_key,
        )

        continue_response = other.post(
            reverse('checkout_continue', args=[checkout.pk]),
        )
        self.assertRedirects(
            continue_response, reverse('checkout'), fetch_redirect_response=False,
        )
        checkout.refresh_from_db()
        self.assertEqual(checkout.active_session_key, owner.session.session_key)

        other_session = other.session
        other_session['checkout_id'] = checkout.pk
        other_session.save()
        write_response = other.post(
            reverse('checkout_add', args=[self.product.pk]),
            {'quantity': '1'},
        )
        self.assertRedirects(
            write_response, reverse('checkout'), fetch_redirect_response=False,
        )
        self.assertFalse(checkout.items.exists())

    def test_old_checkin_requires_explicit_review_before_inventory_changes(self):
        session = CheckinSession.objects.create(user=self.first, scanned_by='LF')
        CheckinSession.objects.filter(pk=session.pk).update(
            started_at=now() - timedelta(hours=25),
        )
        session.refresh_from_db()
        client = self._client(self.first)

        dashboard = client.get(reverse('checkin_dashboard'))
        listed = next(
            item for item in dashboard.context['active_sessions']
            if item.pk == session.pk
        )
        self.assertTrue(listed.needs_review)

        review_page = client.get(reverse('checkin_session', args=[session.pk]))
        self.assertEqual(review_page.status_code, 409)
        self.assertContains(review_page, 'Needs review', status_code=409)

        blocked = client.post(
            reverse('add_quantity', args=[session.pk, self.product.pk]),
            {'amount': '1'},
        )
        self.assertEqual(blocked.status_code, 409)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 2)

        resumed = client.post(
            reverse('checkin_session_reopen', args=[session.pk]),
        )
        self.assertRedirects(
            resumed,
            reverse('checkin_session', args=[session.pk]),
            fetch_redirect_response=False,
        )
        session.refresh_from_db()
        self.assertIsNotNone(session.reopened_at)

        accepted = client.post(
            reverse('add_quantity', args=[session.pk, self.product.pk]),
            {'amount': '1'},
        )
        self.assertEqual(accepted.status_code, 302)
        self.product.refresh_from_db()
        self.assertEqual(self.product.quantity_in_stock, 3)
