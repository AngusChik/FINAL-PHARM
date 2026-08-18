from django.contrib.auth import logout
from django.contrib import messages
from django.shortcuts import redirect, render
from django.http import JsonResponse
from django.db import transaction
from django.utils import timezone


CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'self'; "
    "object-src 'none'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self' data:; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "frame-src 'self'"
)


class ContentSecurityPolicyMiddleware:
    """Keep every browser page on local, same-origin runtime resources.

    Legacy templates still contain inline scripts and styles, so those two
    directives remain explicitly allowed. Remote origins are not allowed; all
    third-party browser libraries and fonts are committed under static/vendor.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response.headers.setdefault('Content-Security-Policy', CONTENT_SECURITY_POLICY)
        return response


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

    A guarded page and every mapped mutation share one canonical lock key. If
    another computer holds it, reads and writes both stop with HTTP 409;
    otherwise this browser session claims/refreshes the lock.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_view(self, request, view_func, view_args, view_kwargs):
        from app.page_lock import (
            CHECKIN_SESSION_MUTATION_NAMES,
            checkin_session_last_activity,
            checkin_session_needs_review,
            guarded_page_path,
        )
        key = guarded_page_path(request)
        if not key:
            return None
        if not request.user.is_authenticated:
            return None
        if (
            request.method in {'GET', 'HEAD'}
            and request.headers.get('x-requested-with') == 'XMLHttpRequest'
        ):
            return None

        match = request.resolver_match
        session_id = match.kwargs.get('session_id') if match else None
        if (
            session_id is not None
            and match.url_name in CHECKIN_SESSION_MUTATION_NAMES
            and match.url_name != 'checkin_end'
        ):
            from app.models import CheckinSession
            session = CheckinSession.objects.filter(pk=session_id).first()
            if session and checkin_session_needs_review(session):
                context = {
                    'session': session,
                    'last_activity': checkin_session_last_activity(session),
                }
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({
                        'ok': False,
                        'error': 'checkin_needs_review',
                        'message': 'This old check-in session must be resumed first.',
                        'resume_url': f'/checkin/session/{session.pk}/reopen/',
                    }, status=409)
                return render(
                    request, 'checkin_needs_review.html', context, status=409,
                )

        from app.models import PagePresence
        from app.page_lock import is_fresh, holder_info, presence_defaults

        if not request.session.session_key:
            request.session.save()
        my = request.session.session_key
        with transaction.atomic():
            holder = PagePresence.objects.select_for_update().filter(page=key).first()
            if holder and holder.session_key != my and is_fresh(holder):
                info = holder_info(holder)
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({
                        'ok': False,
                        'error': 'page_taken_over',
                        'message': 'This page is now active on another computer.',
                        'holder': info,
                    }, status=409)
                return render(request, 'page_busy.html', {
                    'holder': info,
                    'last_seen': holder.last_seen,
                    'page_busy': True,
                    'page_key': key,
                }, status=409)
            PagePresence.objects.update_or_create(page=key, defaults=presence_defaults(request))
        return None
