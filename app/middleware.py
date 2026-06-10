from django.contrib.auth import logout
from django.contrib import messages
from django.shortcuts import redirect
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
                    UserSession.objects.create(
                        user=request.user,
                        session_key=request.session.session_key,
                    )

        return self.get_response(request)
