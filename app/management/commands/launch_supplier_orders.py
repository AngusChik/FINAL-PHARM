from django.core.management.base import BaseCommand, CommandError

from app.supplier_orders import dispatch_scheduled_supplier_launches


def _run_browser_smoke():
    """Open and close a local blank Chromium page without external traffic."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto('about:blank')
        finally:
            browser.close()


class Command(BaseCommand):
    help = 'Launch supplier workers requested through the Windows task broker.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--browser-smoke-if-idle',
            action='store_true',
            help=(
                'When no supplier request is pending, open and close a local '
                'headless browser to validate scheduled-task process creation.'
            ),
        )

    def handle(self, *args, **options):
        results = dispatch_scheduled_supplier_launches(wait_for_workers=True)
        if not results:
            if options['browser_smoke_if_idle']:
                _run_browser_smoke()
                self.stdout.write('supplier launcher: browser smoke passed')
                return
            self.stdout.write('supplier launcher: no pending requests')
            return

        failed = []
        for result in results:
            if result['pid']:
                self.stdout.write(self.style.SUCCESS(
                    f"supplier run {result['run_id']}: started process {result['pid']}"
                ))
            else:
                failed.append(
                    result['error'] or
                    f"supplier run {result['run_id']}: worker did not start"
                )
        if failed:
            raise CommandError(' '.join(failed))
