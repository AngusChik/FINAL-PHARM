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
