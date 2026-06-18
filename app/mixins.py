import time

from django.conf import settings
from django.contrib.auth.mixins import UserPassesTestMixin
from django.shortcuts import redirect
from django.urls import reverse
from urllib.parse import urlencode

# Session key holding the timestamp (epoch seconds) at which this session
# unlocked admin access by entering the passkey.
PASSKEY_SESSION_KEY = 'admin_passkey_unlocked_at'


def passkey_unlocked(request):
    """True if this session unlocked admin access via the passkey and it hasn't expired."""
    ts = request.session.get(PASSKEY_SESSION_KEY)
    if not ts:
        return False
    ttl = getattr(settings, 'ADMIN_PASSKEY_TTL', 0)
    if ttl and (time.time() - ts) > ttl:
        return False
    return True


def has_admin_access(request):
    """
    Admin functions are allowed for staff users (e.g. GINA) OR for a regular
    user (e.g. PU) whose session has been unlocked with the admin passkey.
    """
    return request.user.is_staff or passkey_unlocked(request)


class AdminRequiredMixin(UserPassesTestMixin):
   """
   Restrict a view to admin functions.

   Staff users pass straight through. A non-staff user (PU) is sent to the
   passkey-unlock page; once they enter the correct passkey, has_admin_access()
   stays True for the rest of their session (or until ADMIN_PASSKEY_TTL elapses)
   and they can use the restricted pages.
   """
   def test_func(self):
       return has_admin_access(self.request)


   def handle_no_permission(self):
       # Unauthenticated users go to login; authenticated-but-locked users get
       # the passkey prompt, returning to where they were headed afterwards.
       if not self.request.user.is_authenticated:
           return redirect('login')
       target = reverse('passkey_unlock')
       return redirect(f"{target}?{urlencode({'next': self.request.get_full_path()})}")




class UserRequiredMixin(UserPassesTestMixin):
   """
   Gate for the checkout (terminal / giveaway) flow.

   Any authenticated user may use it — both regular pharmacy users and staff.
   Staff reach it from the purchase page's Checkout button; regular users from
   their own checkout nav. Only unauthenticated requests are turned away.
   """
   def test_func(self):
       return self.request.user.is_authenticated


   def handle_no_permission(self):
       # Unauthenticated → login.
       return redirect('login')
