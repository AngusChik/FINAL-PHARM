"""Runtime environment and external-integration safety helpers."""

from django.conf import settings


class ExternalIntegrationDisabled(RuntimeError):
    """Raised before development can contact an operational external system."""


def is_development_environment():
    return getattr(settings, "PHARMACY_ENVIRONMENT", "production") == "development"


def _integration_enabled(setting_name):
    return bool(getattr(settings, setting_name, True))


def supplier_automation_enabled():
    return _integration_enabled("SUPPLIER_AUTOMATION_ENABLED")


def google_sheets_sync_enabled():
    return _integration_enabled("GOOGLE_SHEETS_SYNC_ENABLED")


def email_delivery_enabled():
    return _integration_enabled("EMAIL_DELIVERY_ENABLED")


def integration_disabled_message(label):
    return (
        f"{label} is disabled in the development environment. "
        "Use production only after the tested change has been promoted."
    )


def require_supplier_automation():
    if not supplier_automation_enabled():
        raise ExternalIntegrationDisabled(
            integration_disabled_message("Supplier ordering")
        )


def require_email_delivery():
    if not email_delivery_enabled():
        raise ExternalIntegrationDisabled(
            integration_disabled_message("Email delivery")
        )
