# Production HTTPS deployment

The live pharmacy deployment has one supported architecture:

```
Staff browser -> Caddy :443 -> Waitress 127.0.0.1:8000 -> Django -> PostgreSQL
```

The production launcher forces `DEBUG=False`, enables Django's secure-cookie and
HTTPS settings, runs deployment checks and migrations, collects static assets,
and waits for Django plus PostgreSQL to pass `/healthz/` before starting Caddy.

## One-time server setup

1. Run `setup_env.bat`.
2. Put `caddy.exe` in the project folder or install Caddy on `PATH`.
3. Ensure `.env` contains a real `DJANGO_SECRET_KEY`, database credentials, and:

   ```env
   PHARMACY_HOST=192.168.0.15
   DJANGO_ALLOWED_HOSTS=192.168.0.15,localhost,127.0.0.1
   DJANGO_CSRF_TRUSTED_ORIGINS=https://192.168.0.15,https://localhost
   ```

4. Open TCP ports 80 and 443 to the pharmacy LAN. Do not expose port 8000;
   Waitress binds to localhost only.
5. Trust Caddy's internal root certificate on each pharmacy workstation. Run
   `caddy trust` on the server and distribute Caddy's root certificate to the
   Trusted Root Certification Authorities store on the other workstations.

Use `configure_ip.bat` whenever the server PC's LAN address changes. It updates
the shared `.env` values used by Django, Caddy, and the production launcher.

## Operation

Staff use the installed **Pharmacy** shortcut. It invokes the hidden production
launcher, repairs a stopped or tracked unhealthy stack, and opens the HTTPS
site only after health checks pass. **Pharmacy Admin Control** opens the visible
administrative console.

```
production.bat          # open the interactive control console
production.bat ensure   # idempotently establish a healthy complete stack
production.bat start    # prepare and start without the menu
production.bat status   # process and Django/database health
production.bat stop     # stop only the tracked pharmacy processes
production.bat update   # controlled stop, prepare, and restart
```

Production logs are stored in `logs/`. Process tracking is stored in `.runtime/`.
Both directories are excluded from Git.

The launcher opens `https://<PHARMACY_HOST>` after a successful start. Hidden
unattended launches use the VBS wrapper installed from the authorized
production worktree:

```
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\install-production-startup.ps1 -EnableAutoStart
```

The installer creates **Pharmacy** and **Pharmacy Admin Control** shortcuts and
registers **Pharmacy Production Startup** for the designated signed-in pharmacy
user. Its triggers run 30 seconds after sign-in and every five minutes for
recovery. Do not change it to SYSTEM: the application checkout is user-writable
and supplier-browser workflows require the signed-in interactive desktop.

Other workstations never run application code. Build a copyable URL-only kit
on the server with `scripts\build-workstation-kit.ps1`, then run
`setup-workstation.bat` on each client to install the trusted certificate and a
shortcut that opens only the server HTTPS URL.

Production deployment is intentionally separate from startup. Use the guarded
publisher documented in `RELEASE_WORKFLOW.md`; it deploys and health-checks the
tested local commit first, then pushes that exact commit and release tag to
GitHub atomically.
