from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils.timezone import now

from .models import UserSession
from .page_lock import PRESENCE_TTL


@override_settings(AXES_ENABLED=False, MAX_PU_SESSIONS=20)
class PresenceActiveTests(TestCase):
    def setUp(self):
        self.viewer = User.objects.create_user(
            username='presence-viewer', password='pass1234', is_staff=True,
        )
        self.client.force_login(self.viewer)

    def test_returns_plain_language_identity_without_technical_details(self):
        pharmacy_user = User.objects.create_user(
            username='PU', password='pass1234', is_staff=False,
        )
        UserSession.objects.create(
            user=pharmacy_user,
            session_key='other-pharmacy-computer',
            pu_slot=2,
            current_path=reverse('sales_analytics'),
            ip_address='127.0.0.1',
            user_agent='Mozilla/5.0 Edg/140.0 Windows NT 10.0',
        )

        response = self.client.get(
            reverse('presence_active'),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['count'], 1)
        self.assertEqual(data['pages'], [{
            'page': 'Sales Analytics',
            'user': 'Pharmacy user 2',
        }])
        self.assertNotIn('ip', data['pages'][0])
        self.assertNotIn('browser', data['pages'][0])

    def test_uses_staff_full_name_and_omits_stale_sessions(self):
        named_staff = User.objects.create_user(
            username='technical-username',
            first_name='Jamie',
            last_name='Lee',
            password='pass1234',
            is_staff=True,
        )
        UserSession.objects.create(
            user=named_staff,
            session_key='named-staff-computer',
            current_path=reverse('inventory_display'),
        )
        stale_user = User.objects.create_user(
            username='stale-user', password='pass1234', is_staff=True,
        )
        stale = UserSession.objects.create(
            user=stale_user,
            session_key='stale-computer',
            current_path=reverse('expired_products'),
        )
        UserSession.objects.filter(pk=stale.pk).update(
            last_activity=now() - timedelta(seconds=PRESENCE_TTL + 1),
        )

        response = self.client.get(reverse('presence_active'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['pages'], [{
            'page': 'Stock',
            'user': 'Jamie Lee',
        }])


class PresenceCardPresentationTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        base_dir = Path(settings.BASE_DIR)
        cls.template = (base_dir / 'app' / 'templates' / 'base.html').read_text(
            encoding='utf-8',
        )
        cls.styles = (base_dir / 'static' / 'css' / 'ui-system.css').read_text(
            encoding='utf-8',
        )

    def test_card_uses_short_plain_language_content(self):
        self.assertIn('No one else online', self.template)
        self.assertIn("labelEl.textContent = 'Online now';", self.template)
        self.assertIn('class="np-page"', self.template)
        self.assertIn('class="np-who"', self.template)
        self.assertNotIn('class="np-count"', self.template)
        self.assertNotIn('p.ip', self.template)
        self.assertNotIn('p.browser', self.template)
        self.assertNotIn('animation: npPulse', self.template)

    def test_compact_card_wraps_only_at_normal_word_boundaries(self):
        self.assertIn(
            'body.app-shell .app-nav .np-label {\n'
            '    min-width: 0;\n'
            '    overflow-wrap: normal;\n'
            '    word-break: normal;',
            self.styles,
        )
        self.assertIn(
            'body.app-shell .app-nav .np-item {\n'
            '    font-size: 0.71875rem;\n'
            '    white-space: normal;\n'
            '    overflow-wrap: normal;\n'
            '    word-break: normal;',
            self.styles,
        )
