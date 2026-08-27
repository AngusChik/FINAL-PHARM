"""Hardened settings used by the production launcher."""

import os

from django.core.exceptions import ImproperlyConfigured

# Production is always served through Caddy. These values are set before the
# shared settings module is imported so all conditional security settings apply.
os.environ["DJANGO_DEBUG"] = "false"
os.environ["DJANGO_SECURE"] = "1"

from .settings import *  # noqa: F401,F403,E402

DEBUG = False

_unsafe_secrets = {
    "",
    "replace-with-a-real-secret-key",
    "django-insecure-fallback-for-dev-only",
}
if SECRET_KEY in _unsafe_secrets:
    raise ImproperlyConfigured(
        "Production requires a real DJANGO_SECRET_KEY in .env."
    )

_configured_admin_passkey = os.environ.get("ADMIN_PASSKEY")
_unsafe_admin_passkeys = {
    "",
    "pharmacy-admin",
    "replace-with-a-private-admin-passkey",
}
_normalized_admin_passkey = (
    _configured_admin_passkey.strip()
    if _configured_admin_passkey is not None
    else ""
)
if (
    _configured_admin_passkey is None
    or _configured_admin_passkey != _normalized_admin_passkey
    or len(_normalized_admin_passkey) < 12
    or _normalized_admin_passkey in _unsafe_admin_passkeys
):
    raise ImproperlyConfigured(
        "Production requires a private ADMIN_PASSKEY of at least 12 characters "
        "with no leading or trailing whitespace; missing, blank, default, and "
        "placeholder values are not allowed."
    )
