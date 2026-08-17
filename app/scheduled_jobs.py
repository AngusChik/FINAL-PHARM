"""Database-backed scheduling helpers invoked by Windows Task Scheduler."""

from datetime import datetime, time, timedelta

from django.db import transaction
from django.utils import timezone

from .models import ScheduledJobRun, StoreHours


DEFAULT_STORE_HOURS = {
    0: (time(9, 30), time(18, 0), False),
    1: (time(9, 30), time(19, 0), False),
    2: (time(9, 30), time(18, 0), False),
    3: (time(9, 30), time(19, 0), False),
    4: (time(9, 30), time(18, 0), False),
    5: (time(9, 30), time(16, 0), False),
    6: (None, None, True),
}

SCHEDULE_RETRY_LIMIT = 3
SCHEDULE_RETRY_DELAY = timedelta(minutes=10)
RUN_STALE_AFTER = timedelta(minutes=20)


def ensure_store_hours():
    """Backstop for databases created before the data migration was applied."""
    existing = set(StoreHours.objects.values_list('weekday', flat=True))
    missing = []
    for weekday, (opens_at, closes_at, is_closed) in DEFAULT_STORE_HOURS.items():
        if weekday not in existing:
            missing.append(StoreHours(
                weekday=weekday,
                opens_at=opens_at,
                closes_at=closes_at,
                is_closed=is_closed,
            ))
    if missing:
        StoreHours.objects.bulk_create(missing, ignore_conflicts=True)
    return list(StoreHours.objects.order_by('weekday'))


def store_hours_payload():
    """Return the Dashboard's JavaScript-day-indexed weekly schedule."""
    payload = {str(day): None for day in range(7)}
    for row in ensure_store_hours():
        # Python: Monday=0. JavaScript: Sunday=0.
        js_weekday = (row.weekday + 1) % 7
        if row.is_closed or not row.opens_at or not row.closes_at:
            payload[str(js_weekday)] = None
        else:
            payload[str(js_weekday)] = [
                row.opens_at.hour,
                row.opens_at.minute,
                row.closes_at.hour,
                row.closes_at.minute,
            ]
    return payload


def _aware_on(day, value):
    return timezone.make_aware(
        datetime.combine(day, value),
        timezone.get_current_timezone(),
    )


def gsheet_schedule_for(day):
    hours = next(
        (row for row in ensure_store_hours() if row.weekday == day.weekday()),
        None,
    )
    if not hours or hours.is_closed or not hours.closes_at:
        return None
    return _aware_on(day, hours.closes_at) - timedelta(minutes=30)


def next_gsheet_pull(at=None):
    local_now = timezone.localtime(at or timezone.now())
    for offset in range(8):
        day = local_now.date() + timedelta(days=offset)
        scheduled_for = gsheet_schedule_for(day)
        if scheduled_for is None:
            continue
        completed = ScheduledJobRun.objects.filter(
            job_key=ScheduledJobRun.JOB_GSHEET_PRECLOSE,
            trigger=ScheduledJobRun.TRIGGER_SCHEDULED,
            business_date=day,
            status__in=[
                ScheduledJobRun.STATUS_SUCCESS,
                ScheduledJobRun.STATUS_SKIPPED,
            ],
        ).exists()
        if day == local_now.date() and scheduled_for <= local_now and not completed:
            return {'scheduled_for': scheduled_for, 'due': True}
        if scheduled_for > local_now:
            return {'scheduled_for': scheduled_for, 'due': False}
    return None


def _finish_run(run, *, status, summary, result=None, error=''):
    run.status = status
    run.summary = summary[:500]
    run.result = result or {}
    run.error = error[:4000]
    run.completed_at = timezone.now()
    run.save(update_fields=[
        'status', 'summary', 'result', 'error', 'completed_at', 'updated_at',
    ])
    return run


def run_google_sheet_sync(
    *, trigger=ScheduledJobRun.TRIGGER_MANUAL, created_by=None,
    run=None, scheduled_for=None, business_date=None,
):
    """Execute one Sheet pull and retain its result in the database."""
    from .gsheet_sync import is_configured, sync_all

    if run is None:
        run = ScheduledJobRun.objects.create(
            job_key=ScheduledJobRun.JOB_GSHEET_PRECLOSE,
            trigger=trigger,
            created_by=created_by,
            scheduled_for=scheduled_for,
            business_date=business_date,
        )

    if not is_configured():
        result = {'imported': 0, 'tabs': [], 'errors': ['Google Sheet sync is not configured.']}
        return _finish_run(
            run,
            status=ScheduledJobRun.STATUS_SKIPPED,
            summary='Google Sheet pull skipped because it is not configured.',
            result=result,
            error=result['errors'][0],
        ), result

    try:
        result = sync_all()
    except Exception as exc:
        result = {'imported': 0, 'tabs': [], 'errors': [str(exc)]}

    if result.get('errors'):
        error = '; '.join(str(value) for value in result['errors'])
        return _finish_run(
            run,
            status=ScheduledJobRun.STATUS_ERROR,
            summary=f'Google Sheet pull failed: {result["errors"][0]}',
            result=result,
            error=error,
        ), result

    imported = int(result.get('imported') or 0)
    summary = (
        f'Google Sheet pull imported {imported} new item'
        f'{"" if imported == 1 else "s"}.'
    )
    return _finish_run(
        run,
        status=ScheduledJobRun.STATUS_SUCCESS,
        summary=summary,
        result=result,
    ), result


def _claim_scheduled_job(job_key, business_date, scheduled_for, *, force=False):
    current = timezone.now()
    with transaction.atomic():
        run, created = ScheduledJobRun.objects.get_or_create(
            job_key=job_key,
            trigger=ScheduledJobRun.TRIGGER_SCHEDULED,
            business_date=business_date,
            defaults={
                'scheduled_for': scheduled_for,
                'status': ScheduledJobRun.STATUS_RUNNING,
                'started_at': current,
            },
        )
        if not created:
            run = ScheduledJobRun.objects.select_for_update().get(pk=run.pk)
            if run.status == ScheduledJobRun.STATUS_SUCCESS and not force:
                return None
            if run.status == ScheduledJobRun.STATUS_SKIPPED and not force:
                return None
            if (
                run.status == ScheduledJobRun.STATUS_RUNNING
                and current - run.started_at < RUN_STALE_AFTER
                and not force
            ):
                return None
            if (
                run.status == ScheduledJobRun.STATUS_ERROR
                and run.attempt_count >= SCHEDULE_RETRY_LIMIT
                and not force
            ):
                return None
            if (
                run.status == ScheduledJobRun.STATUS_ERROR
                and run.completed_at
                and current - run.completed_at < SCHEDULE_RETRY_DELAY
                and not force
            ):
                return None
            run.attempt_count += 1
            run.status = ScheduledJobRun.STATUS_RUNNING
            run.scheduled_for = scheduled_for
            run.started_at = current
            run.completed_at = None
            run.summary = ''
            run.error = ''
            run.result = {}
            run.save(update_fields=[
                'attempt_count', 'status', 'scheduled_for', 'started_at',
                'completed_at', 'summary', 'error', 'result', 'updated_at',
            ])
        return run


def _run_report_cleanup(run, business_date):
    from .reporting import prune_daily_report_archives

    try:
        deleted = prune_daily_report_archives(reference_date=business_date)
    except Exception as exc:
        return _finish_run(
            run,
            status=ScheduledJobRun.STATUS_ERROR,
            summary='Daily report cleanup failed.',
            error=str(exc),
        )
    return _finish_run(
        run,
        status=ScheduledJobRun.STATUS_SUCCESS,
        summary=f'Daily report cleanup removed {deleted} expired snapshot(s).',
        result={'deleted': deleted},
    )


def run_due_jobs(*, at=None, force_job=None):
    """Run jobs due at the supplied local time and return claimed run rows."""
    local_now = timezone.localtime(at or timezone.now())
    day = local_now.date()
    force_all = force_job == 'all'
    completed_runs = []

    cleanup_at = _aware_on(day, time(2, 5))
    if force_all or force_job == ScheduledJobRun.JOB_REPORT_CLEANUP or local_now >= cleanup_at:
        run = _claim_scheduled_job(
            ScheduledJobRun.JOB_REPORT_CLEANUP,
            day,
            cleanup_at,
            force=bool(force_job),
        )
        if run:
            completed_runs.append(_run_report_cleanup(run, day))

    gsheet_at = gsheet_schedule_for(day)
    if gsheet_at and (
        force_all
        or force_job == ScheduledJobRun.JOB_GSHEET_PRECLOSE
        or local_now >= gsheet_at
    ):
        run = _claim_scheduled_job(
            ScheduledJobRun.JOB_GSHEET_PRECLOSE,
            day,
            gsheet_at,
            force=bool(force_job),
        )
        if run:
            finished, _ = run_google_sheet_sync(
                trigger=ScheduledJobRun.TRIGGER_SCHEDULED,
                run=run,
                scheduled_for=gsheet_at,
                business_date=day,
            )
            completed_runs.append(finished)

    return completed_runs
