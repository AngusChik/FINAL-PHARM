from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from app.models import ScheduledJobRun
from app.scheduled_jobs import run_due_jobs


class Command(BaseCommand):
    help = 'Run database-backed pharmacy jobs that are due at the current local time.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--at', metavar='YYYY-MM-DDTHH:MM',
            help='Evaluate schedules at a specific local time (primarily for verification).',
        )
        parser.add_argument(
            '--force',
            choices=[
                'all',
                ScheduledJobRun.JOB_GSHEET_PRECLOSE,
                ScheduledJobRun.JOB_REPORT_CLEANUP,
            ],
            help='Run one job now even when it is not due or already succeeded today.',
        )

    def handle(self, *args, **options):
        at = None
        if options.get('at'):
            at = parse_datetime(options['at'])
            if at is None:
                raise CommandError('Invalid --at value. Use YYYY-MM-DDTHH:MM.')
            if timezone.is_naive(at):
                at = timezone.make_aware(at, timezone.get_current_timezone())

        runs = run_due_jobs(at=at, force_job=options.get('force'))
        if not runs:
            self.stdout.write('scheduled jobs: nothing due')
            return

        failed = []
        for run in runs:
            line = f'{run.get_job_key_display()}: {run.summary}'
            if run.status == ScheduledJobRun.STATUS_ERROR:
                failed.append(line)
                self.stderr.write(self.style.ERROR(line))
            elif run.status == ScheduledJobRun.STATUS_SKIPPED:
                self.stdout.write(self.style.WARNING(line))
            else:
                self.stdout.write(self.style.SUCCESS(line))
        if failed:
            raise CommandError(f'{len(failed)} scheduled job(s) failed.')
