"""Development settings for local coding and testing."""

import os

# Force development-safe behaviour even if the production .env is present.
os.environ["DJANGO_DEBUG"] = "true"
os.environ["DJANGO_SECURE"] = "false"

from .settings import *  # noqa: F401,F403,E402

DEBUG = True
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
