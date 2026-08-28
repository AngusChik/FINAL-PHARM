"""Hardened settings used by the production launcher."""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

from .production_guard import validate_production_role

_BASE_DIR = Path(__file__).resolve().parent.parent
_PRODUCTION_ROLE_ROOT_VALUE = os.environ.get("PHARMACY_PRODUCTION_ROLE_ROOT")
_PRODUCTION_ROLE_ROOT = validate_production_role(
    _BASE_DIR,
    _PRODUCTION_ROLE_ROOT_VALUE,
)
_PRODUCTION_ENV_FILE = Path(
    os.environ.get("PHARMACY_PRODUCTION_ENV_FILE")
    or _PRODUCTION_ROLE_ROOT / ".env"
).resolve()
if not _PRODUCTION_ENV_FILE.is_file():
    raise ImproperlyConfigured(
        f"Production environment file is missing: {_PRODUCTION_ENV_FILE}"
    )
if _PRODUCTION_ENV_FILE.parent != _PRODUCTION_ROLE_ROOT:
    raise ImproperlyConfigured(
        "Production environment file must belong to the authorized production "
        "worktree."
    )
os.environ["PHARMACY_ENV_FILE"] = str(_PRODUCTION_ENV_FILE)
load_dotenv(_PRODUCTION_ENV_FILE, override=True)

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
