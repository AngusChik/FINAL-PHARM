"""Tests for the dashboard "Connect Phone" QR flow.

A phone scans the dashboard QR -> /connect-phone/ -> flag set on its session ->
login as a PU account gives a 2-hour session tagged device_type='phone'. The
admin (is_staff) account is never connected by phone.
"""
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.test import TestCase, Client, RequestFactory, SimpleTestCase, override_settings
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.conf import settings

from app.models import UserSession
from app.views import _lan_base_url

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


class PhoneAddressTests(SimpleTestCase):
    @override_settings(SECURE=True, PHARMACY_HOST='192.168.0.22', ALLOWED_HOSTS=['192.168.0.149', 'localhost'])
    def test_production_prefers_configured_host_even_on_local_backend_request(self):
        request = RequestFactory().get('/', HTTP_HOST='localhost:8000', SERVER_PORT='8000')
        self.assertEqual(_lan_base_url(request), 'https://192.168.0.22')

    @override_settings(SECURE=False, PHARMACY_HOST='192.168.0.22', ALLOWED_HOSTS=['192.168.0.40', 'localhost'])
    def test_local_development_keeps_its_own_http_port(self):
        request = RequestFactory().get('/', HTTP_HOST='localhost:8002', SERVER_PORT='8002')
        self.assertEqual(_lan_base_url(request), 'http://192.168.0.40:8002')

    def test_configure_ip_updates_all_public_addresses_and_preserves_database_settings(self):
        import configure_ip
        with TemporaryDirectory(prefix='pharmacy-address-test-') as temporary:
            env_file = Path(temporary) / '.env'
            env_file.write_text(
                'PHARMACY_HOST=192.168.0.149\n'
                'DJANGO_ALLOWED_HOSTS=192.168.0.149,localhost,127.0.0.1\n'
                'DJANGO_CSRF_TRUSTED_ORIGINS=https://192.168.0.149,https://localhost\n'
                'DB_NAME=keep_this_database\nDB_PASSWORD="test-only-unchanged-secret"\n'
                'DJANGO_SECRET_KEY=test-only-unchanged-signing-key\n', encoding='utf-8',
            )
            with patch.object(configure_ip, 'ENV_PATH', env_file), patch('sys.argv', ['configure_ip.py', '192.168.0.22']), redirect_stdout(io.StringIO()):
                self.assertEqual(configure_ip.main(), 0)
            values = env_file.read_text(encoding='utf-8')
            self.assertIn('PHARMACY_HOST=192.168.0.22\n', values)
            self.assertIn('DJANGO_ALLOWED_HOSTS=192.168.0.22,localhost,127.0.0.1\n', values)
            self.assertIn('DJANGO_CSRF_TRUSTED_ORIGINS=https://192.168.0.22,https://localhost\n', values)
            self.assertIn('DB_NAME=keep_this_database\nDB_PASSWORD="test-only-unchanged-secret"\n', values)
            self.assertIn('DJANGO_SECRET_KEY=test-only-unchanged-signing-key\n', values)
            self.assertNotIn('192.168.0.149', values)
