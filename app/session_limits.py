"""Concurrent PU-session limiting and backend identity allocation.

All regular workstations sign in with the shared visible username ``PU``. This
module allocates a unique backend identity (PU1..PU6 by default) to each live
computer session. Phones have a separate allowance; staff/admin sessions are
also separate and do not consume a PU computer slot.
The login view, Active Sessions page, and prune_sessions command share this rule.

A session is "active" while its heartbeat (UserSession.last_activity, refreshed
every ~10s by base.html / presence_heartbeat) is fresher than
SESSION_ACTIVE_WINDOW. Anything staler is dead — pruned along with its
django_session row so the browser is truly logged out and cannot silently
re-register via ConcurrentSessionMiddleware.
"""

from datetime import timedelta

from django.conf import settings
from django.contrib.sessions.models import Session as DjangoSession
from django.db import connection
from django.db.models import Q
from django.utils.timezone import now

from app.models import UserSession

# Fixed key for pg_advisory_xact_lock — serializes the login critical section
# (check-count-allocate-insert) across all processes so slots cannot be raced.
SESSION_CAP_LOCK_KEY = 727274


def active_window_seconds():
    return getattr(settings, 'SESSION_ACTIVE_WINDOW', 2700)


def active_cutoff():
    """Heartbeats at or after this instant count as active."""
    return now() - timedelta(seconds=active_window_seconds())


def pu_max():
    """Maximum number of simultaneously active regular/PU computer sessions."""
    return getattr(
        settings,
        'MAX_PU_SESSIONS',
        getattr(settings, 'GLOBAL_MAX_SESSIONS', 6),
    )


def phone_max():
    return getattr(settings, 'MAX_PU_PHONE_SESSIONS', 4)


def global_max():
    """Compatibility alias for callers using the former setting name."""
    return pu_max()


def _delete_django_sessions(session_keys):
    """Drop the Django session rows for these keys so the browsers are logged out."""
    keys = [k for k in session_keys if k]
    if keys:
        DjangoSession.objects.filter(session_key__in=keys).delete()


def take_global_lock():
    """Acquire the transaction-scoped advisory lock (auto-released at commit).

    Must be called inside a transaction.atomic() block. Serializes concurrent
    logins so the active-count check and the insert are effectively one step.
    """
    with connection.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", [SESSION_CAP_LOCK_KEY])


def prune_stale():
    """Release disconnected or expired sessions and their Django session rows.

    A phone reaching its two-hour expiry must be able to sign in again even
    when its last heartbeat was recent.
    """
    expired_keys = DjangoSession.objects.filter(expire_date__lte=now()).values('session_key')
    stale = list(
        UserSession.objects.filter(
            Q(last_activity__lt=active_cutoff()) | Q(session_key__in=expired_keys)
        )
        .values_list('session_key', flat=True)
    )
    if not stale:
        return 0
    _delete_django_sessions(stale)
    UserSession.objects.filter(session_key__in=stale).delete()
    return len(stale)


def active_count():
    """How many tracked sessions are active, including staff/admin."""
    return UserSession.objects.filter(last_activity__gte=active_cutoff()).count()


def active_pu_count(exclude_session_key=None):
    """How many active computers consume PU slots, excluding a re-login cookie."""
    sessions = UserSession.objects.filter(
        user__is_staff=False,
        device_type=UserSession.DEVICE_COMPUTER,
        last_activity__gte=active_cutoff(),
    )
    if exclude_session_key:
        sessions = sessions.exclude(session_key=exclude_session_key)
    return sessions.count()


def active_phone_count(exclude_session_key=None):
    sessions = UserSession.objects.filter(
        user__is_staff=False,
        device_type=UserSession.DEVICE_PHONE,
        last_activity__gte=active_cutoff(),
    )
    if exclude_session_key:
        sessions = sessions.exclude(session_key=exclude_session_key)
    return sessions.count()


def release_phone_slots():
    """Upgrade older tracked phone sessions while holding the login lock.

    Phones used to occupy PU1..PU6. Keep them signed in, but release those
    numbers before allocating a computer identity under the new policy.
    """
    return UserSession.objects.filter(
        user__is_staff=False, device_type=UserSession.DEVICE_PHONE,
        pu_slot__isnull=False,
    ).update(pu_slot=None)


def next_pu_slot(exclude_session_key=None):
    """Return the lowest available PU slot, or ``None`` when all are occupied.

    Call this while holding ``take_global_lock()`` after ``prune_stale()``.
    ``pu_slot`` is also unique at the database layer as a final race guard.
    """
    sessions = UserSession.objects.filter(
        user__is_staff=False,
        pu_slot__isnull=False,
    )
    if exclude_session_key:
        sessions = sessions.exclude(session_key=exclude_session_key)
    used = set(sessions.values_list('pu_slot', flat=True))
    return next((slot for slot in range(1, pu_max() + 1) if slot not in used), None)


def drop_session_tracking(session_key):
    """Replace only the browser presenting this cookie, never an IP neighbour.

    Leave Django's session row alone: auth.login will reuse or rotate it. This
    runs only after capacity validation, inside the locked login transaction.
    """
    if not session_key:
        return 0
    deleted, _ = UserSession.objects.filter(session_key=session_key).delete()
    return deleted


def evict_for_user(user, exclude_session_key=None):
    """Admin singleton: remove ALL of this user's sessions (and their django_session
    rows). Their next request hits ConcurrentSessionMiddleware and is logged out."""
    sessions = user.user_sessions.all()
    if exclude_session_key:
        sessions = sessions.exclude(session_key=exclude_session_key)
    keys = list(sessions.values_list('session_key', flat=True))
    if not keys:
        return 0
    _delete_django_sessions(keys)
    sessions.delete()
    return len(keys)
