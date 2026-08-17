from django.core.management.base import BaseCommand, CommandError

from app.inventory_audit import run_inventory_audit
from app.models import InventoryAuditRun


class Command(BaseCommand):
    help = (
        'Audit durable inventory relationships: product totals versus lots, '
        'normalized barcodes, negative values, and supplier receiving totals.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--repair-unassigned', action='store_true',
            help='Add missing positive stock to UNASSIGNED lots. Never reduces named lots.',
        )

    def handle(self, *args, **options):
        run = run_inventory_audit(
            repair_unassigned=options['repair_unassigned'],
        )
        if run.status == InventoryAuditRun.STATUS_ERROR:
            raise CommandError(run.error or run.summary)

        for issue in run.issues.all():
            label = issue.product_name or issue.title
            detail = f'{label}: {issue.detail}'
            if issue.repaired:
                self.stdout.write(self.style.WARNING(f'Repaired {detail}'))
            else:
                self.stdout.write(self.style.ERROR(detail))

        if run.status == InventoryAuditRun.STATUS_PASSED:
            self.stdout.write(self.style.SUCCESS(run.summary))
        else:
            self.stdout.write(self.style.WARNING(run.summary))
