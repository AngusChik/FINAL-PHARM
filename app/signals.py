from django.contrib.auth.signals import user_logged_out
from django.dispatch import receiver


@receiver(user_logged_out)
def on_user_logout(sender, request, user, **kwargs):
    """Remove the UserSession row when a user logs out, freeing a session slot."""
    from app.models import UserSession

    if request and request.session.session_key:
        UserSession.objects.filter(session_key=request.session.session_key).delete()
