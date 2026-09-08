"""
ASGI config for inventory project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'inventory.settings_production')

application = get_asgi_application()

# Keep ASGI deployments consistent with the WSGI website's local learning.
from app.prescription_drug_worker import start_prescription_drug_worker

start_prescription_drug_worker()
