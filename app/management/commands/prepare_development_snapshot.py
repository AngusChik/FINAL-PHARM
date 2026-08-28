"""Neutralize live-only transient state after restoring a development copy."""

from django.conf import settings
from django.contrib.sessions.models import Session
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from app.models import (
    CheckoutOrder,
    PagePresence,
    ScheduledJobRun,
    SupplierOrderPlan,
    SupplierOrderRun,
    UserSession,
)


class Command(BaseCommand):
    help = (
        "Clear copied browser state and neutralize in-flight automation after "
        "a production snapshot is restored into development."
    )

    def handle(self, *args, **options):
        if (
            getattr(settings, "PHARMACY_ENVIRONMENT", "") != "development"
            or not getattr(settings, "DEVELOPMENT_DATABASE_ISOLATED", False)
        ):
            raise CommandError(
                "prepare_development_snapshot is development-only and refused "
                "to modify this database."
            )

        finished_at = timezone.now()
        active_supplier_states = [
            SupplierOrderRun.STATE_STARTING,
            SupplierOrderRun.STATE_LOGIN,
            SupplierOrderRun.STATE_WAITING_USER,
            SupplierOrderRun.STATE_RUNNING,
            SupplierOrderRun.STATE_PAUSED,
            SupplierOrderRun.STATE_REVIEW,
            # Failed copied runs can otherwise expose a Retry action that starts
            # a real supplier browser from development.
            SupplierOrderRun.STATE_ERROR,
        ]

        with transaction.atomic():
            session_count, _ = Session.objects.all().delete()
            user_session_count, _ = UserSession.objects.all().delete()
            presence_count, _ = PagePresence.objects.all().delete()
            checkout_claim_count = CheckoutOrder.objects.exclude(
                active_session_key="",
            ).update(active_session_key="")

            supplier_run_count = SupplierOrderRun.objects.filter(
                state__in=active_supplier_states,
            ).update(
                state=SupplierOrderRun.STATE_CANCELLED,
                message=(
                    "Cancelled while preparing the isolated development snapshot."
                ),
                process_id=None,
                pause_requested=False,
                cancel_requested=True,
                heartbeat_at=None,
                completed_at=finished_at,
                updated_at=finished_at,
            )
            supplier_plan_count = SupplierOrderPlan.objects.filter(
                Q(status=SupplierOrderPlan.STATUS_RUNNING)
                | Q(mckesson_recovery_claimed_by__isnull=False)
                | Q(mckesson_recovery_claimed_at__isnull=False)
            ).update(
                status=SupplierOrderPlan.STATUS_CANCELLED,
                mckesson_recovery_claimed_by=None,
                mckesson_recovery_claimed_at=None,
                completed_at=finished_at,
            )
            scheduled_run_count = ScheduledJobRun.objects.filter(
                status=ScheduledJobRun.STATUS_RUNNING,
            ).update(
                status=ScheduledJobRun.STATUS_SKIPPED,
                summary=(
                    "Neutralized while preparing the isolated development snapshot."
                ),
                error="",
                result={"neutralized_for_development": True},
                completed_at=finished_at,
                updated_at=finished_at,
            )

        self.stdout.write(self.style.SUCCESS(
            "Development snapshot prepared: "
            f"{session_count} Django session(s), "
            f"{user_session_count} user session(s), "
            f"{presence_count} page presence row(s), "
            f"{checkout_claim_count} checkout claim(s), "
            f"{supplier_run_count} supplier run(s), "
            f"{supplier_plan_count} supplier plan(s), and "
            f"{scheduled_run_count} scheduled run(s) neutralized."
        ))
