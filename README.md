# FINAL-PHARM

Pharmacy inventory & checkout management (Django).

## Mobile / tablet support

The UI is responsive and works on phones (iPhone-size) and tablets (iPad-size)
as well as shop computers. The left dock navigation collapses to a bottom bar on
small screens, multi-column layouts stack to a single column, wide data tables
scroll inside their card, and modals/toasts size to the viewport. Styling lives
in `app/templates/base.html` (global rules + breakpoints) and per-page
`<style>` blocks; breakpoints are standardized at `1024px`, `768px`, and `480px`.

## Connect a phone

The dashboard has a **Connect Phone** button (under *Active Sessions*). It shows
a QR code and the server's LAN URL. A staff member scans it with their phone
camera to open the app over the shop Wi-Fi, then signs in with their **PU
account**.

- **Session length:** a phone login lasts **2 hours**
  (`PHONE_SESSION_AGE` in `inventory/settings.py`, default `7200s`), versus the
  8-hour shift session on a shop computer (`SESSION_COOKIE_AGE`). Override with
  the `PHONE_SESSION_AGE` env var if needed.
- **Admin excluded:** the admin (GINA / `is_staff`) account is *not* connected by
  phone — it stays a single-device session on its main computer. Scanning the QR
  while signed in as admin just shows a note.
- **Visibility:** phone-connected devices show a green **📱 Phone** badge on the
  *Active Sessions* page.

How it works: the QR points at `/connect-phone/`, which flags the phone's
pre-login session; `CustomLoginView` honours that flag for PU accounts only —
setting the 2-hour expiry and tagging the `UserSession` as a phone
(`app/views.py`). The LAN address comes from `DJANGO_ALLOWED_HOSTS` (set via
`configure_ip.py`); the QR PNG is generated server-side with `qrcode` (no
internet needed at runtime).

## Setup

```
pip install -r requirements.txt
python manage.py migrate
python configure_ip.py <this-machine-LAN-IP>   # so phones can reach the server
python manage.py runserver 0.0.0.0:8000
```
