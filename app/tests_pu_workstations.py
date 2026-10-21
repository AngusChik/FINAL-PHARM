"""Exercise workstation logins through the production Waitress proxy settings."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
from unittest import skipUnless

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.sessions.models import Session as DjangoSession
from django.db import close_old_connections, connections
from django.test import Client, SimpleTestCase, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils.timezone import now
from waitress.adjustments import Adjustments
from waitress.proxy_headers import proxy_headers_middleware

from app import session_limits
from app.models import UserSession


def _ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@lru_cache(maxsize=1)
def production_proxy_settings():
    """Run the real launch expression against an argv-only fake Waitress module.

    This checks Windows Start-Process quoting too. No server or application is
    started: the temporary module only prints its received command arguments.
    """
    source = (Path(settings.BASE_DIR) / 'scripts' / 'production.ps1').read_text(encoding='utf-8')
    expression = re.search(
        r'\$waitressProcess = Start-Process -FilePath \$pythonw.*?(?=\r?\n\s*\$healthy = \$false)',
        source, re.DOTALL,
    )
    if expression is None:
        raise AssertionError('Cannot find the production Waitress launch expression.')
    with TemporaryDirectory(prefix='pharmacy-proxy-argv-') as temporary:
        root = Path(temporary)
        (root / 'waitress.py').write_text(
            'import json, sys\nprint(json.dumps(sys.argv[1:]))\n', encoding='utf-8',
        )
        command = '\n'.join([
            "$ErrorActionPreference = 'Stop'",
            f'$pythonw = {_ps_literal(sys.executable)}',
            f'$projectRoot = {_ps_literal(root)}',
            f'$waitressOut = {_ps_literal(root / "stdout.txt")}',
            f'$waitressErr = {_ps_literal(root / "stderr.txt")}',
            expression.group(0),
            '$null = $waitressProcess.Handle',
            'if (-not $waitressProcess.WaitForExit(15000)) { $waitressProcess.Kill(); throw "Argv probe timed out." }',
            '$waitressProcess.WaitForExit()',
            'if ($waitressProcess.ExitCode -ne 0) { throw (Get-Content -LiteralPath $waitressErr -Raw) }',
            'Get-Content -LiteralPath $waitressOut -Raw',
        ])
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
            capture_output=True, text=True, timeout=25, check=False,
        )
        if result.returncode:
            raise AssertionError(f'Production argv probe failed: {result.stderr}')
        arguments = json.loads(result.stdout)
    options, application = Adjustments.parse_args(arguments)
    if application != ['inventory.wsgi:application']:
        raise AssertionError(f'Unexpected production application arguments: {application}')
    options.pop('help')
    options.pop('call')
    return Adjustments(**options)


class ProxiedClient(Client):
    def __init__(self, address, *, peer='127.0.0.1', forwarded=None):
        super().__init__(
            REMOTE_ADDR=peer, HTTP_X_FORWARDED_FOR=forwarded or address,
            HTTP_X_FORWARDED_PROTO='https', HTTP_HOST='192.168.0.22', SERVER_PORT='8000',
        )
        proxy = production_proxy_settings()
        handler = self.handler
        wrapped = proxy_headers_middleware(
            lambda environ, start_response: handler(environ),
            trusted_proxy=proxy.trusted_proxy,
            trusted_proxy_count=proxy.trusted_proxy_count,
            trusted_proxy_headers=proxy.trusted_proxy_headers,
            clear_untrusted=proxy.clear_untrusted_proxy_headers,
        )
        self.handler = lambda environ: wrapped(environ, lambda *args: None)


TEST_SETTINGS = {
    'AXES_ENABLED': False,
    'MAX_PU_SESSIONS': 6,
    'MAX_PU_PHONE_SESSIONS': 4,
    'SESSION_ACTIVE_WINDOW': 2700,
    'PHARMACY_HOST': '192.168.0.22',
    'ALLOWED_HOSTS': ['192.168.0.22', 'testserver', 'localhost'],
    'PASSWORD_HASHERS': ['django.contrib.auth.hashers.MD5PasswordHasher'],
}


@skipUnless(os.name == 'nt', 'Production launcher uses Windows Start-Process.')
class ProductionProxyArgumentsTests(SimpleTestCase):
    def test_only_local_caddy_can_forward_client_address_and_https(self):
        proxy = production_proxy_settings()
        self.assertEqual(proxy.host, '127.0.0.1')
        self.assertEqual(proxy.trusted_proxy, '127.0.0.1')
        self.assertEqual(proxy.trusted_proxy_count, 1)
        self.assertEqual(proxy.trusted_proxy_headers, {'x-forwarded-for', 'x-forwarded-proto'})
        self.assertTrue(proxy.clear_untrusted_proxy_headers)


@skipUnless(os.name == 'nt', 'Exercises the Windows production launcher.')
@override_settings(**TEST_SETTINGS)
class PUWorkstationTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.pu = get_user_model().objects.create_user('PU', password='test-only-password')
        cls.admin = get_user_model().objects.create_user('GINA', password='test-only-password', is_staff=True)

    def login(self, client, username='PU'):
        return client.post(reverse('login'), {'username': username, 'password': 'test-only-password'})

    def six_computers(self):
        clients = [ProxiedClient(f'192.168.0.{100 + index}') for index in range(6)]
        for client in clients:
            self.assertEqual(self.login(client).status_code, 302)
        return clients

    def four_phones(self):
        phones = [ProxiedClient(f'192.168.0.{210 + index}') for index in range(4)]
        for phone in phones:
            self.assertEqual(phone.get(reverse('connect_phone')).status_code, 302)
            self.assertEqual(self.login(phone).status_code, 302)
        return phones

    def assert_signed_in(self, clients):
        for client in clients:
            self.assertEqual(client.get(reverse('dashboard')).status_code, 200)

    def test_six_shared_logins_keep_distinct_ips_and_all_remain_signed_in(self):
        clients = self.six_computers()
        self.assert_signed_in(clients)
        sessions = UserSession.objects.filter(user=self.pu).order_by('pu_slot')
        self.assertEqual(list(sessions.values_list('pu_slot', flat=True)), [1, 2, 3, 4, 5, 6])
        self.assertEqual(set(sessions.values_list('ip_address', flat=True)), {
            f'192.168.0.{100 + index}' for index in range(6)
        })

    def test_seventh_computer_is_refused_without_disconnecting_six(self):
        clients = self.six_computers()
        extra = ProxiedClient('192.168.0.200')
        self.assertEqual(self.login(extra).status_code, 200)
        self.assertFalse(extra.session.get('_auth_user_id'))
        self.assert_signed_in(clients)
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 6)

    def test_separate_browsers_with_the_same_ip_do_not_replace_each_other(self):
        clients = [ProxiedClient('192.168.0.100') for _ in range(6)]
        for client in clients:
            self.assertEqual(self.login(client).status_code, 302)
        self.assert_signed_in(clients)
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 6)

    def test_same_browser_relogin_reuses_its_slot_at_capacity(self):
        clients = self.six_computers()
        before = set(UserSession.objects.exclude(session_key=clients[0].session.session_key).values_list('session_key', flat=True))
        self.assertEqual(self.login(clients[0]).status_code, 302)
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 6)
        self.assertTrue(before.issubset(set(UserSession.objects.values_list('session_key', flat=True))))
        self.assert_signed_in(clients)

    def test_one_logout_leaves_other_computers_signed_in(self):
        clients = self.six_computers()
        clients[0].logout()
        self.assert_signed_in(clients[1:])
        self.assertEqual(UserSession.objects.filter(user=self.pu).count(), 5)

    def test_admin_login_does_not_displace_pu_computers(self):
        clients = self.six_computers()
        self.assertEqual(self.login(ProxiedClient('192.168.0.50'), 'GINA').status_code, 302)
        self.assert_signed_in(clients)

    def test_four_phones_can_connect_when_all_six_computers_are_signed_in(self):
        clients = self.six_computers()
        phones = self.four_phones()
        self.assert_signed_in(clients + phones)
        self.assertEqual(session_limits.active_pu_count(), 6)
        self.assertEqual(session_limits.active_phone_count(), 4)
        for phone in phones:
            row = UserSession.objects.get(session_key=phone.session.session_key)
            self.assertEqual(row.device_type, UserSession.DEVICE_PHONE)
            self.assertIsNone(row.pu_slot)
            self.assertLessEqual(phone.session.get_expiry_age(), settings.PHONE_SESSION_AGE)
            self.assertGreater(phone.session.get_expiry_age(), settings.PHONE_SESSION_AGE - 120)

    def test_fifth_phone_is_refused_without_disconnecting_existing_sessions(self):
        clients = self.six_computers()
        phones = self.four_phones()
        extra = ProxiedClient('192.168.0.214')
        extra.get(reverse('connect_phone'))
        self.assertEqual(self.login(extra).status_code, 200)
        self.assert_signed_in(clients + phones)
        self.assertFalse(extra.session.get('_auth_user_id'))
        self.assertEqual(session_limits.active_phone_count(), 4)

    def test_phone_can_sign_in_again_after_two_hour_expiry(self):
        clients = self.six_computers()
        phones = self.four_phones()
        phone = phones[0]
        old_key = phone.session.session_key
        DjangoSession.objects.filter(session_key=old_key).update(expire_date=now() - timedelta(seconds=1))
        phone.get(reverse('connect_phone'))
        self.assertEqual(self.login(phone).status_code, 302)
        self.assertFalse(UserSession.objects.filter(session_key=old_key).exists())
        self.assert_signed_in(clients + phones)
        self.assertEqual(session_limits.active_phone_count(), 4)

    def test_phone_conversion_frees_computer_slot_without_logging_it_out(self):
        clients = self.six_computers()
        clients[0].get(reverse('connect_phone'))
        self.assertEqual(session_limits.active_pu_count(), 5)
        self.assertIsNone(UserSession.objects.get(session_key=clients[0].session.session_key).pu_slot)
        replacement = ProxiedClient('192.168.0.200')
        self.assertEqual(self.login(replacement).status_code, 302)
        self.assert_signed_in(clients + [replacement])

    def test_phone_conversion_at_phone_capacity_preserves_computer_session(self):
        clients = self.six_computers()
        phones = self.four_phones()
        clients[0].get(reverse('connect_phone'))
        self.assertEqual(UserSession.objects.get(session_key=clients[0].session.session_key).device_type, UserSession.DEVICE_COMPUTER)
        self.assertGreater(clients[0].session.get_expiry_age(), settings.PHONE_SESSION_AGE)
        self.assert_signed_in(clients + phones)

    def test_old_phone_slot_is_released_when_computer_signs_in(self):
        phone = ProxiedClient('192.168.0.210')
        phone.get(reverse('connect_phone'))
        self.assertEqual(self.login(phone).status_code, 302)
        UserSession.objects.filter(session_key=phone.session.session_key).update(pu_slot=1)
        clients = self.six_computers()
        self.assert_signed_in(clients + [phone])
        self.assertIsNone(UserSession.objects.get(session_key=phone.session.session_key).pu_slot)

    def test_forwarded_chain_uses_only_the_address_supplied_by_caddy(self):
        client = ProxiedClient('192.168.0.100', forwarded='203.0.113.55, 192.168.0.100')
        self.assertEqual(self.login(client).status_code, 302)
        self.assertEqual(UserSession.objects.get(user=self.pu).ip_address, '192.168.0.100')

    def test_untrusted_peer_cannot_supply_another_workstation_address(self):
        client = ProxiedClient('192.168.0.100', peer='192.168.0.200')
        self.assertEqual(self.login(client).status_code, 302)
        self.assertEqual(UserSession.objects.get(user=self.pu).ip_address, '192.168.0.200')

    @override_settings(SECURE=True, ALLOWED_HOSTS=['192.168.0.149', '192.168.0.22', 'localhost'])
    def test_dashboard_phone_link_uses_new_public_https_address_without_backend_port(self):
        client = ProxiedClient('192.168.0.100')
        self.assertEqual(self.login(client).status_code, 302)
        response = client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['connect_phone_url'], 'https://192.168.0.22/connect-phone/')
        self.assertContains(response, 'https://192.168.0.22/connect-phone/')


@skipUnless(os.name == 'nt', 'Exercises the Windows production launcher.')
@override_settings(**TEST_SETTINGS)
class ConcurrentPUWorkstationTests(TransactionTestCase):
    def test_seven_simultaneous_logins_admit_six_without_evicting_any(self):
        user = get_user_model().objects.create_user('PU', password='test-only-password')
        production_proxy_settings()
        barrier = Barrier(7)

        def sign_in(index):
            close_old_connections()
            try:
                client = ProxiedClient(f'192.168.0.{100 + index}')
                barrier.wait(timeout=15)
                response = client.post(reverse('login'), {'username': 'PU', 'password': 'test-only-password'})
                return client, response.status_code
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=7) as executor:
            results = list(executor.map(sign_in, range(7)))
        self.assertEqual(sorted(status for _, status in results), [200] + [302] * 6)
        self.assertEqual(UserSession.objects.filter(user=user).count(), 6)
        for client, status in results:
            if status == 302:
                self.assertEqual(client.get(reverse('dashboard')).status_code, 200)
