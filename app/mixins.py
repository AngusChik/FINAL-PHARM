from django.contrib.auth.mixins import UserPassesTestMixin
from django.shortcuts import redirect


class AdminRequiredMixin(UserPassesTestMixin):
   """
   Mixin to allow only admin users to access the view.
   Redirects non-admin users to the home page.
   """
   def test_func(self):
       # Check if the user is an admin (is_staff=True)
       return self.request.user.is_staff


   def handle_no_permission(self):
       # Redirect non-admin users to the home page
       return redirect('home')




class UserRequiredMixin(UserPassesTestMixin):
   """
   This mixin restricts access to views for regular users only.
   Admin users or unauthenticated users are redirected to a different page.
   """
   def test_func(self):
       # Check if the user is authenticated and NOT an admin
       return self.request.user.is_authenticated and not self.request.user.is_staff


   def handle_no_permission(self):
       # Redirect admin users to the admin dashboard or a no-access page
       return redirect('admin_dashboard')  # Change to your desired redirect URL







