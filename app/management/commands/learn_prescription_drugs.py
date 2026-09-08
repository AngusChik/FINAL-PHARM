from django.core.management.base import BaseCommand, CommandError

from app.prescription_drug_learning import (
    learn_prescription_drugs, preview_prescription_drugs,
)


class Command(BaseCommand):
    help = 'Learn complete prescription drug details from existing ordering-sheet labels.'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Preview counts without saving records.')
        parser.add_argument('--batch-size', type=int, default=100)

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        if batch_size < 1:
            raise CommandError('--batch-size must be positive.')
        if options['dry_run']:
            result = preview_prescription_drugs()
            prefix = 'Preview'
        else:
            result = dict(processed=0, created=0, matched=0, skipped=0)
            while True:
                batch = learn_prescription_drugs(batch_size=batch_size)
                for key in result:
                    result[key] += batch[key]
                if batch['processed'] < batch_size:
                    break
            prefix = 'Completed'
        self.stdout.write(
            f"{prefix}: {result['processed']} rows checked; "
            f"{result['created']} new drugs; {result['matched']} existing matches; "
            f"{result['skipped']} incomplete or ambiguous rows skipped."
        )
