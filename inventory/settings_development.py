"""Isolated settings for local development and restored data snapshots."""

import os
import json
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import dotenv_values, load_dotenv

from .development_guard import (
    validate_development_database_override,
    validate_development_environment,
)


_BASE_DIR = Path(__file__).resolve().parent.parent
DEVELOPMENT_ENV_FILE = _BASE_DIR / ".env.development"

# Development has its own explicit configuration file.  Loading it with
# ``override=True`` prevents values from the production-oriented root ``.env``
# from winning when the shared settings module imports it afterwards.
os.environ["PHARMACY_ENV_FILE"] = str(DEVELOPMENT_ENV_FILE)
load_dotenv(DEVELOPMENT_ENV_FILE, override=True)

_production_env_file = _BASE_DIR / ".env"
_workflow_file = _BASE_DIR / ".runtime" / "development-workflow.json"
if _workflow_file.is_file():
    try:
        _workflow = json.loads(_workflow_file.read_text(encoding="utf-8-sig"))
        _configured_production_root = Path(
            _workflow["production_worktree"]
        ).resolve()
        if (
            _configured_production_root.parent != _BASE_DIR.parent
            or _configured_production_root == _BASE_DIR
        ):
            raise ValueError("production worktree is not a sibling")
        _production_env_file = _configured_production_root / ".env"
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ImproperlyConfigured(
            "Development workflow configuration cannot resolve the production "
            f"environment: {exc}"
        ) from exc

_production_values = dotenv_values(_production_env_file)
_resolved_production_database = str(
    _production_values.get("DB_NAME") or "postgres"
).strip()

_configured_development_database = os.environ.get("DB_NAME")
_development_database_override = os.environ.get(
    "PHARMACY_DEVELOPMENT_DB_OVERRIDE"
)
_development_test_database = os.environ.get("DEVELOPMENT_TEST_DB_NAME")
_development_database_user = os.environ.get("DB_USER")
_resolved_production_user = str(
    _production_values.get("DB_USER") or "postgres"
).strip()

_development_database, _production_database = validate_development_environment(
    os.environ.get("PHARMACY_ENVIRONMENT"),
    _configured_development_database,
    os.environ.get("PRODUCTION_DB_NAME"),
    _resolved_production_database,
    _development_test_database,
    _development_database_user,
    _resolved_production_user,
)
if _development_database_override:
    _development_database = validate_development_database_override(
        _development_database_override,
        _production_database,
    )

# Force development-safe behaviour even if a copied production value appears
# in either environment file.
os.environ["DJANGO_DEBUG"] = "true"
os.environ["DJANGO_SECURE"] = "false"

from .settings import *  # noqa: F401,F403,E402

PHARMACY_ENVIRONMENT = "development"
DEVELOPMENT_DATABASE_ISOLATED = True
DEVELOPMENT_DATABASE_NAME = _development_database
PRODUCTION_DATABASE_NAME = _production_database

DEBUG = True
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# Development remains fail-closed even during tests. Tests that exercise an
# integration's enabled branch must opt in with override_settings and mock the
# transport/process boundary explicitly.
SUPPLIER_AUTOMATION_ENABLED = False
GOOGLE_SHEETS_SYNC_ENABLED = False
SCHEDULED_JOBS_ENABLED = False
EMAIL_DELIVERY_ENABLED = False
EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
EMAIL_HOST_USER = ""
EMAIL_HOST_PASSWORD = ""

DEVELOPMENT_SAFETY_MESSAGE = (
    "DEVELOPMENT - test data only. Supplier ordering, scheduled jobs, "
    "Google Sheets, and email delivery are disabled."
)

DATABASES["default"]["NAME"] = _development_database
DATABASES["default"].setdefault("TEST", {})["NAME"] = (
    _development_test_database.strip()
)
