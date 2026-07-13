"""
Pull new Ordering Sheet entries from the configured Google Spreadsheet.

Usage:
    python manage.py sync_gsheet

Behaviour:
  * Reads every ordering-shaped tab (Form responses or hand-typed rows) and
    imports unmarked rows as OrderingSheetEntry rows (source=gsheet).
  * Pull-only: the app never rewrites or deletes sheet content — its only
    write-back is the "Imported" marker column used for dedup.
  * No-op with a clear message when GSHEET_SPREADSHEET_ID isn't configured.

Schedule every ~5 minutes via Windows Task Scheduler using sync_gsheet.bat.
"""

from django.core.management.base import BaseCommand

from app.gsheet_sync import is_configured, sync_all


class Command(BaseCommand):
    help = "Pull new Ordering Sheet entries from the configured Google Spreadsheet."

    def handle(self, *args, **options):
        if not is_configured():
            self.stdout.write("gsheet pull: not configured (set GSHEET_SPREADSHEET_ID in .env) — skipping.")
            return
        result = sync_all()
        if result['errors']:
            for err in result['errors']:
                self.stderr.write(f"gsheet pull error: {err}")
        tabs = ", ".join(f"{t['title']}: {t['imported']}" for t in result.get('tabs', [])) or "no tabs read"
        self.stdout.write(
            f"gsheet pull: {result['imported']} imported ({tabs}), {len(result['errors'])} error(s)."
        )
