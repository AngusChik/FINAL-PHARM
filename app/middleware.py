from django.contrib.auth import logout
from django.contrib import messages
from django.shortcuts import redirect, render
from django.db import transaction
from django.utils import timezone


class ConcurrentSessionMiddleware:
    """
    Validates that the current session is still registered in UserSession.
    If a session was evicted (e.g., admin logged in elsewhere), the user
    is logged out and redirected to login with an explanatory message.

    Sessions created outside CustomLoginView (e.g., Django admin login)
    are auto-registered rather than kicked.

    Also throttle-updates last_activity (once per 60 seconds) to avoid
    a DB write on every single request.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated and request.session.session_key:
            from app.models import UserSession

            try:
                user_session = UserSession.objects.get(
                    session_key=request.session.session_key
                )
                # Throttled last_activity update — only if >60s since last
                now = timezone.now()
                if (now - user_session.last_activity).total_seconds() > 60:
                    user_session.last_activity = now
                    user_session.save(update_fields=['last_activity'])
            except UserSession.DoesNotExist:
                # Check if this user has ANY tracked sessions — if yes, this
                # session was evicted. If no, it's a login that bypassed
                # CustomLoginView (e.g., Django admin) — auto-register it.
                if UserSession.objects.filter(user=request.user).exists():
                    # Session was evicted by a newer login
                    logout(request)
                    messages.warning(
                        request,
                        'Your session was ended because this account logged in elsewhere.'
                    )
                    return redirect('login')
                else:
                    # First-time session (admin login, etc.) — register it
                    xff = request.META.get('HTTP_X_FORWARDED_FOR')
                    ip = xff.split(',')[0].strip() if xff else request.META.get('REMOTE_ADDR')
                    UserSession.objects.create(
                        user=request.user,
                        session_key=request.session.session_key,
                        ip_address=ip,
                        user_agent=request.META.get('HTTP_USER_AGENT', '')[:300],
                    )

        return self.get_response(request)


class PageLockMiddleware:
    """Limits guarded work pages to one computer at a time.

    On a GET to a guarded page, if another computer currently holds it (fresh
    heartbeat), render the 'busy' page naming that computer. Otherwise this
    browser session claims/refreshes the page lock.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_view(self, request, view_func, view_args, view_kwargs):
        if request.method != 'GET':
            return None
        rm = request.resolver_match
        from app.page_lock import GUARDED_PAGE_NAMES
        if not rm or rm.url_name not in GUARDED_PAGE_NAMES:
            return None
        if not request.user.is_authenticated:
            return None
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return None

        from app.models import PagePresence
        from app.page_lock import is_fresh, holder_info, presence_defaults

        if not request.session.session_key:
            request.session.save()
        my = request.session.session_key
        key = request.path

        with transaction.atomic():
            holder = PagePresence.objects.select_for_update().filter(page=key).first()
            if holder and holder.session_key != my and is_fresh(holder):
                return render(request, 'page_busy.html', {
                    'holder': holder_info(holder),
                    'last_seen': holder.last_seen,
                    'page_busy': True,
                    'page_key': key,
                }, status=409)
            PagePresence.objects.update_or_create(page=key, defaults=presence_defaults(request))
        return None
