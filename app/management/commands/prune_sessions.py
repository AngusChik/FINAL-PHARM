"""
Prune stale session rows so the concurrent-session count stays honest.

Usage:
    python manage.py prune_sessions
    python manage.py prune_sessions --dry-run

Behaviour:
  * Deletes UserSession rows whose heartbeat is older than
    settings.SESSION_ACTIVE_WINDOW (and their matching django_session rows), via
    app.session_limits.prune_stale(). A computer that closed its browser without
    logging out frees its slot here instead of lingering forever.
  * Then runs Django's built-in `clearsessions` to drop expired django_session
    rows the active-window prune didn't cover.

Run once after deploying the global-cap change to collapse the existing backlog
of duplicate/stale rows, then schedule daily via Windows Task Scheduler
(alongside daily_report.bat).
"""
from django.core.management import call_command
from django.core.management.base import BaseCommand

from app import session_limits
from app.models import UserSession


class Command(BaseCommand):
    help = "Delete stale UserSession rows (and expired django_session rows)."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Report what would be pruned without deleting anything.',
        )

    def handle(self, *args, **options):
        cutoff = session_limits.active_cutoff()

        if options['dry_run']:
            stale = UserSession.objects.filter(last_activity__lt=cutoff).count()
            active = session_limits.active_count()
            self.stdout.write(
                f"[dry-run] {stale} stale session row(s) would be pruned; "
                f"{active} would remain active (cap {session_limits.global_max()})."
            )
            return

        pruned = session_limits.prune_stale()
        call_command('clearsessions')
        self.stdout.write(self.style.SUCCESS(
            f"Pruned {pruned} stale session row(s); "
            f"{session_limits.active_count()} active "
            f"(cap {session_limits.global_max()}). Ran clearsessions."
        ))
