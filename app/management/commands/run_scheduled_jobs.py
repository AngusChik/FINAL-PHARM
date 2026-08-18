from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from app.models import ScheduledJobRun, StoreHours
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
                ScheduledJobRun.JOB_DATABASE_BACKUP,
            ],
            help='Run one job now even when it is not due or already succeeded today.',
        )
        parser.add_argument(
            '--self-test', action='store_true',
            help='Verify the database and store-hours scheduler without running jobs.',
        )

    def handle(self, *args, **options):
        if options.get('self_test'):
            if options.get('at') or options.get('force'):
                raise CommandError('--self-test cannot be combined with --at or --force.')
            try:
                connection.ensure_connection()
                hours = list(StoreHours.objects.order_by('weekday'))
            except Exception as exc:
                raise CommandError(
                    f'Scheduled jobs self-test could not access configuration: {exc}'
                ) from exc

            weekdays = {row.weekday for row in hours}
            if weekdays != set(range(7)):
                raise CommandError('Scheduled jobs self-test requires all seven StoreHours rows.')
            open_days = [row for row in hours if not row.is_closed]
            invalid_days = [
                row.get_weekday_display()
                for row in open_days
                if (
                    row.opens_at is None
                    or row.closes_at is None
                    or row.closes_at <= row.opens_at
                )
            ]
            if not open_days:
                raise CommandError('Scheduled jobs self-test found no open business days.')
            if invalid_days:
                raise CommandError(
                    'Scheduled jobs self-test found invalid hours for: '
                    + ', '.join(invalid_days)
                )
            misaligned_days = [
                row.get_weekday_display()
                for row in open_days
                if (
                    row.closes_at.minute != 0
                    or row.closes_at.second != 0
                    or row.closes_at.microsecond != 0
                )
            ]
            if misaligned_days:
                raise CommandError(
                    'Hourly pre-closing automation requires whole-hour closing '
                    'times for: ' + ', '.join(misaligned_days)
                )
            self.stdout.write(self.style.SUCCESS(
                'scheduled jobs self-test passed: '
                f'database={connection.vendor}, store_hours=7, '
                f'open_days={len(open_days)}'
            ))
            return

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
