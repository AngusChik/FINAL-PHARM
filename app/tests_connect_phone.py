"""Tests for the dashboard "Connect Phone" QR flow.

A phone scans the dashboard QR -> /connect-phone/ -> flag set on its session ->
login as a PU account gives a 2-hour session tagged device_type='phone'. The
admin (is_staff) account is never connected by phone.
"""
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.conf import settings

from app.models import UserSession

User = get_user_model()


class ConnectPhoneTests(TestCase):
    def setUp(self):
        self.pu = User.objects.create_user(username="pu", password="pass1234", is_staff=False)
        self.admin = User.objects.create_user(username="gina", password="pass1234", is_staff=True)
        self.client = Client()

    def _flag_phone(self):
        """Mimic having scanned the QR: set the connect_phone session flag."""
        session = self.client.session
        session["connect_phone"] = True
        session.save()

    def test_connect_phone_anonymous_redirects_to_login_and_sets_flag(self):
        resp = self.client.get(reverse("connect_phone"))
        self.assertRedirects(resp, reverse("login"), fetch_redirect_response=False)
        self.assertTrue(self.client.session.get("connect_phone"))

    def test_pu_login_via_connect_flow_gets_2h_phone_session(self):
        self._flag_phone()
        resp = self.client.post(
            reverse("login"), {"username": "pu", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.50",
        )
        self.assertEqual(resp.status_code, 302)
        us = UserSession.objects.get(user=self.pu)
        self.assertEqual(us.device_type, UserSession.DEVICE_PHONE)
        # ~2 hours, allow a little slack for test runtime.
        self.assertLessEqual(self.client.session.get_expiry_age(), settings.PHONE_SESSION_AGE)
        self.assertGreater(self.client.session.get_expiry_age(), settings.PHONE_SESSION_AGE - 120)
        # The one-shot flag is consumed.
        self.assertIsNone(self.client.session.get("connect_phone"))

    def test_admin_login_via_connect_flow_is_not_a_phone(self):
        self._flag_phone()
        resp = self.client.post(
            reverse("login"), {"username": "gina", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.51",
        )
        self.assertEqual(resp.status_code, 302)
        us = UserSession.objects.get(user=self.admin)
        self.assertEqual(us.device_type, UserSession.DEVICE_COMPUTER)
        # Admin keeps the full shift-length session, not the 2-hour phone window.
        self.assertGreater(self.client.session.get_expiry_age(), settings.PHONE_SESSION_AGE)

    def test_normal_pu_login_without_flag_is_a_computer(self):
        resp = self.client.post(
            reverse("login"), {"username": "pu", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.52",
        )
        self.assertEqual(resp.status_code, 302)
        us = UserSession.objects.get(user=self.pu)
        self.assertEqual(us.device_type, UserSession.DEVICE_COMPUTER)

    def test_authenticated_pu_scan_converts_session_in_place(self):
        # Sign in normally first (computer session).
        self.client.post(
            reverse("login"), {"username": "pu", "password": "pass1234"},
            REMOTE_ADDR="192.168.0.53",
        )
        us = UserSession.objects.get(user=self.pu)
        self.assertEqual(us.device_type, UserSession.DEVICE_COMPUTER)
        # Now scan the QR while signed in -> converts to a 2-hour phone session.
        resp = self.client.get(reverse("connect_phone"))
        self.assertRedirects(resp, reverse("dashboard"), fetch_redirect_response=False)
        us.refresh_from_db()
        self.assertEqual(us.device_type, UserSession.DEVICE_PHONE)
        self.assertLessEqual(self.client.session.get_expiry_age(), settings.PHONE_SESSION_AGE)
