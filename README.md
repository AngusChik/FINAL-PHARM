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

This is the developer-only control panel. It manages development, refreshes its
test data, runs release checks, publishes a tested release, and shows both
development and production health. Ordinary staff do not use this console.

Development uses the local-only `development` branch, a dedicated
`pharmacy_development` PostgreSQL database and role, and
`http://127.0.0.1:8001`. It is always localhost-only and displays a prominent
**DEVELOPMENT – TEST DATA** banner. The Windows launcher uses Django's stable
`--noreload` process mode so Stop and Restart can terminate the exact tracked
process tree; choose **Restart development** after changing code. Real email,
supplier browsers, Google Sheet synchronization, and scheduled jobs remain
disabled, including during automated tests unless an individual mocked test
explicitly opts in.

The same controls are available directly from a terminal:

```
development.bat start
development.bat status
development.bat stop
development.bat restart
development.bat refresh-data
development.bat check
development.bat publish
```

Provision the sibling production worktree once after committing the workflow:

```
setup-development-workflow.bat -CreateDevelopmentBranch
```

The first cutover requires the old production launcher to be stopped. The setup
refuses to continue while ports 8000/443 or tracked legacy processes are still
active. The initial setup intentionally does not install shortcuts until the
new controller has been released to production. See `RELEASE_WORKFLOW.md` for
the exact bootstrap and rollback sequence.

## Production

Production runs Waitress on localhost behind Caddy HTTPS. Its launcher performs
Django deployment checks, a verified pre-start database backup, migrations,
static collection, and a database-backed health check before reporting success.

Staff double-click **Pharmacy**. The shortcut silently ensures the complete
production stack is healthy and opens the HTTPS site without showing a command
window. A hidden **Pharmacy Production Startup** task runs shortly after the
designated pharmacy user signs in and repeats every five minutes for recovery.
It runs under that interactive user with limited rights, never as SYSTEM.

Administrators use **Pharmacy Admin Control** for credentials, status, backup,
restart, recovery controls, and logs. A deliberate administrator Stop is
remembered so the recovery task does not immediately start production again;
double-clicking **Pharmacy** or choosing Start clears that deliberate-stop state.

If the PostgreSQL password is missing or rejected, the production console asks
for it with hidden input, verifies the database connection, and saves it to
`.env`. This is a one-time repair; leave the production console open for normal
Start, Stop, and Restart controls afterward.

The same controls are available directly from a terminal:

```
production.bat start
production.bat ensure
production.bat status
production.bat stop
production.bat update
production.bat backup
```

`production.bat ensure` is idempotent: it leaves a healthy stack alone, starts
a stopped stack, and replaces a tracked partial/unhealthy stack. Untracked port
conflicts, missing credentials, invalid production-role markers, and incomplete
release recovery all fail closed with an actionable log entry.

Application changes never use `production.bat update` as a deployment method.
Use **Publish Tested Release** in `development.bat`: it tests the clean local
development commit, creates release recovery artifacts, deploys and verifies
production, and only then atomically pushes that exact `main` commit and tag to
GitHub. A failed Git push leaves healthy production running and blocks another
release until the pending synchronization succeeds.
Runtime process IDs are stored under `.runtime/`, and output is written to
`logs/`. See `DEPLOYMENT_HTTPS.md` for the one-time Caddy and certificate setup.

## Database backup and recovery

Main-computer setup includes the database backup in **Pharmacy Scheduled Jobs**.
It creates and verifies one PostgreSQL backup on each open business day,
30 minutes before the configured closing time. Production also creates a
verified backup before every restart or migration. Backups are kept for 30
days by default.

For a server that was set up before this feature was added, double-click
`install_database_backup_task.bat` once and accept the Administrator prompt.
This refreshes the pharmacy automation task without rerunning server setup.

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

## Scheduled pharmacy jobs

Main-computer setup also installs **Pharmacy Scheduled Jobs**, a lightweight
Windows task that checks the database schedule once per hour on the hour. It
runs hidden, without opening a console window. The app runs each due
job once and saves its result in PostgreSQL; it does not run every job on every
check.

The task uses the signed-in main-computer Windows account and remains fully
windowless while the screen is locked. After a Windows restart, sign in to the
main computer so the hourly dispatcher and the separate on-demand
supplier-browser task have their required interactive Windows session. It does
not run as SYSTEM because the project folder is writable by that user.

- Google Sheet ordering entries are pulled one hour before closing on each
  open day. Closing times come from the shared `StoreHours` database rows used
  by the Dashboard clock. Closing times must remain on the hour so the hourly
  `:00` dispatcher can meet that timing exactly.
- A verified database backup is created immediately after the pre-closing Sheet
  pull on each open day. If the pull later succeeds on retry, a fresh backup is
  created so it includes the imported rows.
- There is no scheduled Daily Report cleanup job. Old PDF snapshots are pruned
  only when a new snapshot is saved; source transactions and inventory history
  are never removed.
- Failed Sheet pulls retry up to three times with a cooldown, and overlapping
  manual/scheduled pulls are blocked.

For an existing server, double-click `install_automation_task.bat` once and
accept the Administrator prompt. Job output is written to
`logs\scheduled-jobs.log`; the durable run history remains available in the
database.
