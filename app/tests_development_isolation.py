import os
from datetime import timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.contrib.sessions.models import Session
from django.core.exceptions import ImproperlyConfigured
from django.core.management import call_command
from django.core.management.base import CommandError
from django.template.loader import render_to_string
from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from app import gsheet_sync
from app.context_processors import ui_context
from app.environment import (
    email_delivery_enabled,
    supplier_automation_enabled,
)
from app.models import (
    CheckoutOrder,
    PagePresence,
    ScheduledJobRun,
    SupplierOrderPlan,
    SupplierOrderRun,
    UserSession,
)
from app.views import _start_supplier_run
from app.management.commands.run_scheduled_jobs import (
    Command as RunScheduledJobsCommand,
)
from inventory.development_guard import validate_development_environment


class DevelopmentConfigurationTests(SimpleTestCase):
    def test_explicit_development_database_is_accepted(self):
        names = validate_development_environment(
            "development",
            "pharmacy_development",
            "postgres",
            "postgres",
            "test_pharmacy_development",
            "pharmacy_development",
            "postgres",
        )
        self.assertEqual(names, ("pharmacy_development", "postgres"))

    def test_missing_environment_identity_is_rejected(self):
        with self.assertRaisesRegex(
            ImproperlyConfigured, "PHARMACY_ENVIRONMENT=development",
        ):
            validate_development_environment(
                "", "pharmacy_development", "postgres",
            )

    def test_missing_database_names_are_rejected(self):
        with self.assertRaisesRegex(ImproperlyConfigured, "explicit DB_NAME"):
            validate_development_environment("development", "", "postgres")
        with self.assertRaisesRegex(ImproperlyConfigured, "PRODUCTION_DB_NAME"):
            validate_development_environment(
                "development", "pharmacy_development", "",
            )

    def test_matching_database_names_are_rejected_case_insensitively(self):
        with self.assertRaisesRegex(
            ImproperlyConfigured, "DB_NAME must differ",
        ):
            validate_development_environment(
                "development", "Pharmacy", "pharmacy",
            )

    def test_declared_production_name_must_match_resolved_live_database(self):
        with self.assertRaisesRegex(
            ImproperlyConfigured, "does not match the live DB_NAME",
        ):
            validate_development_environment(
                "development",
                "pharmacy_development",
                "claimed_production",
                "postgres",
            )

    def test_example_and_gitignore_define_separate_development_config(self):
        root = Path(settings.BASE_DIR)
        example = (root / ".env.development.example").read_text(
            encoding="utf-8",
        )
        gitignore = (root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("PHARMACY_ENVIRONMENT=development", example)
        self.assertIn("DB_NAME=pharmacy_development", example)
        self.assertIn("PRODUCTION_DB_NAME=postgres", example)
        self.assertIn(".env.development", gitignore.splitlines())


class DevelopmentIntegrationSafetyTests(SimpleTestCase):
    @override_settings(
        PHARMACY_ENVIRONMENT="development",
        SUPPLIER_AUTOMATION_ENABLED=False,
        GOOGLE_SHEETS_SYNC_ENABLED=False,
        EMAIL_DELIVERY_ENABLED=False,
    )
    def test_context_and_banner_identify_safe_development_environment(self):
        request = RequestFactory().get("/")
        request.user = AnonymousUser()
        context = ui_context(request)
        rendered = render_to_string(
            "partials/_development_banner.html", context,
        )

        self.assertTrue(context["is_development_environment"])
        self.assertFalse(context["supplier_automation_enabled"])
        self.assertFalse(context["google_sheets_sync_enabled"])
        self.assertIn("Development", rendered)
        self.assertIn("Supplier ordering", rendered)
        self.assertIn("email delivery", rendered)

    @override_settings(
        PHARMACY_ENVIRONMENT="development",
        SUPPLIER_AUTOMATION_ENABLED=False,
    )
    def test_supplier_start_is_rejected_before_database_or_process_work(self):
        request = RequestFactory().post(
            "/low-stock/mckesson-order/start/",
            data="{}",
            content_type="application/json",
        )
        response = _start_supplier_run(
            request,
            SupplierOrderRun.VENDOR_MCKESSON,
            "mckesson_order.py",
        )

        self.assertEqual(response.status_code, 403)
        self.assertIn("disabled", response.content.decode().lower())
        self.assertFalse(supplier_automation_enabled())

    @override_settings(
        PHARMACY_ENVIRONMENT="development",
        GOOGLE_SHEETS_SYNC_ENABLED=False,
    )
    def test_google_sheet_sync_never_reaches_service_account(self):
        with (
            patch.dict(
                os.environ,
                {"GSHEET_SPREADSHEET_ID": "a" * 24},
                clear=False,
            ),
            patch("app.gsheet_sync.get_spreadsheet") as get_spreadsheet,
        ):
            self.assertFalse(gsheet_sync.is_configured())
            result = gsheet_sync.sync_all()

        get_spreadsheet.assert_not_called()
        self.assertEqual(result["imported"], 0)
        self.assertIn("disabled", result["errors"][0].lower())

    @override_settings(
        PHARMACY_ENVIRONMENT="development",
        EMAIL_DELIVERY_ENABLED=False,
    )
    def test_development_email_backend_is_local_only(self):
        self.assertEqual(
            settings.EMAIL_BACKEND,
            "django.core.mail.backends.locmem.EmailBackend",
        )
        self.assertFalse(email_delivery_enabled())

    @override_settings(SCHEDULED_JOBS_ENABLED=False)
    def test_scheduled_job_dispatch_is_rejected_in_development(self):
        with self.assertRaisesRegex(CommandError, "disabled"):
            RunScheduledJobsCommand().handle(
                at=None,
                force=None,
                self_test=True,
            )


class PrepareDevelopmentSnapshotTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="snapshot-admin",
            password="test-only-password",
        )
        Session.objects.create(
            session_key="copied-django-session",
            session_data="",
            expire_date=timezone.now() + timedelta(hours=1),
        )
        UserSession.objects.create(
            user=self.user,
            session_key="copied-user-session",
        )
        PagePresence.objects.create(
            page="/inventory/",
            session_key="copied-user-session",
            user=self.user,
        )
        self.checkout = CheckoutOrder.objects.create(
            user=self.user,
            active_session_key="copied-user-session",
        )
        self.plan = SupplierOrderPlan.objects.create(
            created_by=self.user,
            status=SupplierOrderPlan.STATUS_RUNNING,
            mckesson_recovery_claimed_by=self.user,
            mckesson_recovery_claimed_at=timezone.now(),
        )
        self.run = SupplierOrderRun.objects.create(
            plan=self.plan,
            created_by=self.user,
            vendor=SupplierOrderRun.VENDOR_MCKESSON,
            state=SupplierOrderRun.STATE_ERROR,
            message="Copied retryable failure",
            process_id=12345,
        )
        self.job = ScheduledJobRun.objects.create(
            job_key=ScheduledJobRun.JOB_GSHEET_PRECLOSE,
            trigger=ScheduledJobRun.TRIGGER_MANUAL,
            status=ScheduledJobRun.STATUS_RUNNING,
        )

    @override_settings(
        PHARMACY_ENVIRONMENT="development",
        DEVELOPMENT_DATABASE_ISOLATED=True,
    )
    def test_command_clears_sessions_and_neutralizes_automation(self):
        output = StringIO()
        call_command("prepare_development_snapshot", stdout=output)

        self.assertFalse(Session.objects.exists())
        self.assertFalse(UserSession.objects.exists())
        self.assertFalse(PagePresence.objects.exists())

        self.checkout.refresh_from_db()
        self.plan.refresh_from_db()
        self.run.refresh_from_db()
        self.job.refresh_from_db()
        self.assertEqual(self.checkout.active_session_key, "")
        self.assertEqual(self.plan.status, SupplierOrderPlan.STATUS_CANCELLED)
        self.assertIsNone(self.plan.mckesson_recovery_claimed_by)
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_CANCELLED)
        self.assertIsNone(self.run.process_id)
        self.assertTrue(self.run.cancel_requested)
        self.assertEqual(self.job.status, ScheduledJobRun.STATUS_SKIPPED)
        self.assertTrue(self.job.result["neutralized_for_development"])
        self.assertIn("Development snapshot prepared", output.getvalue())

    @override_settings(
        PHARMACY_ENVIRONMENT="production",
        DEVELOPMENT_DATABASE_ISOLATED=False,
    )
    def test_command_refuses_non_development_database(self):
        with self.assertRaisesRegex(CommandError, "development-only"):
            call_command("prepare_development_snapshot")

        self.assertTrue(Session.objects.exists())
        self.run.refresh_from_db()
        self.assertEqual(self.run.state, SupplierOrderRun.STATE_ERROR)
