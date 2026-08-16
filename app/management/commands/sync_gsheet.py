"""
Pull new Ordering Sheet entries from the configured Google Spreadsheet.

Usage:
    python manage.py sync_gsheet

Behaviour:
  * Reads every ordering-shaped tab (Form responses or hand-typed rows) and
    imports new rows as OrderingSheetEntry rows (source=gsheet).
  * Pull-only: the app never rewrites or deletes sheet content. Durable app
    records provide deduplication.
  * No-op with a clear message when GSHEET_SPREADSHEET_ID isn't configured.

The automatic pre-closing pull is dispatched by run_scheduled_jobs. This
command remains available for a manual pull.
"""

from django.core.management.base import BaseCommand, CommandError

from app.models import ScheduledJobRun
from app.scheduled_jobs import run_google_sheet_sync


class Command(BaseCommand):
    help = "Pull new Ordering Sheet entries from the configured Google Spreadsheet."

    def handle(self, *args, **options):
        run, result = run_google_sheet_sync()
        if run.status == ScheduledJobRun.STATUS_SKIPPED:
            self.stdout.write(
                "gsheet pull: not configured (set GSHEET_SPREADSHEET_ID in .env) — skipping."
            )
            return
        if result['errors']:
            raise CommandError('; '.join(str(err) for err in result['errors']))
        tabs = ", ".join(f"{t['title']}: {t['imported']}" for t in result.get('tabs', [])) or "no tabs read"
        self.stdout.write(
            f"gsheet pull: {result['imported']} imported ({tabs}), {len(result['errors'])} error(s)."
        )
