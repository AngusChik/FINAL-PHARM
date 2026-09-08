"""
WSGI config for inventory project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'inventory.settings_production')

application = get_wsgi_application()

# Start local database learning only in a running website, after Django is ready.
from app.prescription_drug_worker import start_prescription_drug_worker

start_prescription_drug_worker()
