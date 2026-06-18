"""Global concurrent-session limiting.

One place for the rule "at most GLOBAL_MAX_SESSIONS active computers at a time,
across all accounts, one slot per computer." Shared by the login view, the
Active Sessions page, and the prune_sessions management command.

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
from django.utils.timezone import now

from app.models import UserSession

# Fixed key for pg_advisory_xact_lock — serializes the login critical section
# (check-count-then-insert) across all processes so a global cap can't be raced.
SESSION_CAP_LOCK_KEY = 727274


def active_window_seconds():
    return getattr(settings, 'SESSION_ACTIVE_WINDOW', 300)


def active_cutoff():
    """Heartbeats at or after this instant count as active."""
    return now() - timedelta(seconds=active_window_seconds())


def global_max():
    return getattr(settings, 'GLOBAL_MAX_SESSIONS', 5)


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
    """Delete UserSession rows whose heartbeat is older than the active window,
    plus their django_session rows. Returns the number of UserSession rows removed."""
    stale = list(
        UserSession.objects.filter(last_activity__lt=active_cutoff())
        .values_list('session_key', flat=True)
    )
    if not stale:
        return 0
    _delete_django_sessions(stale)
    UserSession.objects.filter(session_key__in=stale).delete()
    return len(stale)


def active_count():
    """How many computers are currently active (fresh heartbeat)."""
    return UserSession.objects.filter(last_activity__gte=active_cutoff()).count()


def drop_computer(user, ip):
    """Free this account's previous slot on the same computer (matched by IP),
    so a re-login replaces rather than stacks. No-op if ip is falsy."""
    if not ip:
        return 0
    keys = list(
        UserSession.objects.filter(user=user, ip_address=ip)
        .values_list('session_key', flat=True)
    )
    if not keys:
        return 0
    _delete_django_sessions(keys)
    UserSession.objects.filter(session_key__in=keys).delete()
    return len(keys)


def evict_for_user(user):
    """Admin singleton: remove ALL of this user's sessions (and their django_session
    rows). Their next request hits ConcurrentSessionMiddleware and is logged out."""
    keys = list(user.user_sessions.values_list('session_key', flat=True))
    if not keys:
        return 0
    _delete_django_sessions(keys)
    user.user_sessions.all().delete()
    return len(keys)


def evict_stalest():
    """Remove the single least-recently-active session (and its django_session).
    Used only to let an admin in when the global cap is full. Returns True if one
    was evicted."""
    victim = UserSession.objects.order_by('last_activity').first()
    if not victim:
        return False
    _delete_django_sessions([victim.session_key])
    victim.delete()
    return True
