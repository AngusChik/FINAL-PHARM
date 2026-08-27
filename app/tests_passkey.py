import time
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .mixins import PASSKEY_SESSION_KEY, passkey_unlocked
from .models import UserAction


TEST_ADMIN_PASSKEY = "test-only-private-admin-passkey"


@override_settings(ADMIN_PASSKEY=TEST_ADMIN_PASSKEY)
class PasskeyUnlockBehaviorTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="passkey-behavior-pu",
            password="test-only-login-password",
            is_staff=False,
        )
        self.client.force_login(self.user)

    def test_valid_configured_passkey_unlocks_session_and_writes_audit(self):
        destination = reverse("new_product")

        response = self.client.post(
            reverse("passkey_unlock"),
            {"passkey": TEST_ADMIN_PASSKEY, "next": destination},
        )

        self.assertRedirects(response, destination, fetch_redirect_response=False)
        self.assertIn(PASSKEY_SESSION_KEY, self.client.session)
        self.assertTrue(
            UserAction.objects.filter(
                user=self.user,
                action="passkey_unlock",
                target="admin access",
            ).exists()
        )

    def test_fifth_invalid_attempt_locks_session_and_writes_audit(self):
        endpoint = reverse("passkey_unlock")

        for attempt in range(1, 6):
            response = self.client.post(
                endpoint,
                {"passkey": "incorrect-test-passkey"},
            )
            self.assertEqual(response.status_code, 200)
            session = self.client.session
            if attempt < 5:
                self.assertEqual(session["passkey_failed_attempts"], attempt)
                self.assertNotIn("passkey_locked_until", session)

        session = self.client.session
        self.assertEqual(session["passkey_failed_attempts"], 0)
        self.assertGreater(session["passkey_locked_until"], time.time())
        self.assertNotIn(PASSKEY_SESSION_KEY, session)
        self.assertTrue(
            UserAction.objects.filter(
                user=self.user,
                action="passkey_lockout",
                target="admin access",
                detail="5 failed passkey attempts",
            ).exists()
        )

    def test_correct_passkey_remains_blocked_during_lockout(self):
        session = self.client.session
        session["passkey_locked_until"] = time.time() + 300
        session.save()

        response = self.client.post(
            reverse("passkey_unlock"),
            {"passkey": TEST_ADMIN_PASSKEY},
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(PASSKEY_SESSION_KEY, self.client.session)
        self.assertFalse(
            UserAction.objects.filter(
                user=self.user,
                action="passkey_unlock",
            ).exists()
        )


class PasskeyExpiryTests(SimpleTestCase):
    @override_settings(ADMIN_PASSKEY_TTL=60)
    @patch("app.mixins.time.time", return_value=1_000.0)
    def test_passkey_unlock_expires_after_configured_ttl(self, _mock_time):
        fresh_request = SimpleNamespace(
            session={PASSKEY_SESSION_KEY: 950.0},
        )
        expired_request = SimpleNamespace(
            session={PASSKEY_SESSION_KEY: 939.0},
        )

        self.assertTrue(passkey_unlocked(fresh_request))
        self.assertFalse(passkey_unlocked(expired_request))
