# FINAL-PHARM

Pharmacy inventory & checkout management (Django).

## Mobile / tablet support

The UI is responsive and works on phones (iPhone-size) and tablets (iPad-size)
as well as shop computers. The left dock navigation collapses to a bottom bar on
small screens, multi-column layouts stack to a single column, wide data tables
scroll inside their card, and modals/toasts size to the viewport. Styling lives
in the shared design tokens and interface system under `static/css/`; legacy
page-level styles are normalized by that shared layer. Breakpoints are
standardized at `1024px`, `768px`, and `480px`. See `UI_SYSTEM.md` before adding
new pages or components.

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

## First-time setup

On the main pharmacy/server computer, double-click `setup-main-computer.bat` and
accept its Windows Administrator prompt. This one-time setup configures Python,
dependencies, the LAN address, Caddy HTTPS, Windows Firewall, database
migrations, certificate trust, and starts production in the normal user session.

The setup creates `Pharmacy-Root-Certificate.crt`. Copy that certificate to
each other pharmacy computer and install it under **Trusted Root Certification
Authorities** before opening the server URL.

`setup_env.bat` remains available as a development-only environment setup.

Copy `.env.example` to `.env` first if `.env` does not exist. Production will
refuse to start until `DJANGO_SECRET_KEY` contains a real secret.

## Development

Double-click:

```
development.bat
```

This opens a control console that stays available while you work. It shows the
current server state and provides Start, Stop, Restart, Open Website, and Open
Logs options. Development uses `inventory.settings_development`, Django's
auto-reloading server, detailed error pages, and `http://127.0.0.1:8001`.
Port 8001 lets development run without colliding with production on port 8000.

The same controls are available directly from a terminal:

```
development.bat start
development.bat status
development.bat stop
development.bat restart
```

To make a development server temporarily reachable on the LAN:

```
development.bat -Lan
```

Never use the development launcher for the pharmacy's live deployment.

## Production

Production runs Waitress on localhost behind Caddy HTTPS. Its launcher performs
Django deployment checks, a verified pre-start database backup, migrations,
static collection, and a database-backed health check before reporting success.

Double-click `production.bat` to open its persistent control console. It shows
the current health and provides Start, Stop, Restart/Update, Open Website, and
Open Logs options. Clicking Start while production is already healthy is safe;
it reports the current state instead of failing.

The same controls are available directly from a terminal:

```
production.bat start
production.bat status
production.bat stop
production.bat update
production.bat backup
```

`production.bat update` performs a controlled stop and full prepared restart.
Runtime process IDs are stored under `.runtime/`, and output is written to
`logs/`. See `DEPLOYMENT_HTTPS.md` for the one-time Caddy and certificate setup.

## Database backup and recovery

Main-computer setup installs a Windows task named **Pharmacy Database Backup**.
It creates and verifies a PostgreSQL backup every day at 2:00 AM. Production
also creates a verified backup before every restart or migration. Backups are
kept for 30 days by default.

- Run an extra backup from the production menu, with
  `production.bat backup`, or by double-clicking `database_backup.bat`.
- Restore only while production is stopped. Run `production.bat stop`, then
  drag a `.dump` file onto `database_restore.bat` (or pass its path on the
  command line). The restore verifies the checksum and creates another safety
  backup before replacing database objects.
- Configure the destination and retention in `.env` using
  `PHARMACY_BACKUP_DIR` and `PHARMACY_BACKUP_RETENTION_DAYS`. A secured external
  drive or network location protects against failure or loss of the server PC;
  the default `backups\database` folder protects only against database-level
  mistakes.
