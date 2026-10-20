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
   PHARMACY_HOST=192.168.0.22
   DJANGO_ALLOWED_HOSTS=192.168.0.22,localhost,127.0.0.1
   DJANGO_CSRF_TRUSTED_ORIGINS=https://192.168.0.22,https://localhost
   ```

4. Open TCP ports 80 and 443 to the pharmacy LAN. Do not expose port 8000;
   Waitress binds to localhost only.
5. Trust Caddy's internal root certificate on each pharmacy workstation. Run
   `caddy trust` on the server and distribute Caddy's root certificate to the
   Trusted Root Certification Authorities store on the other workstations.

Use `configure_ip.bat` whenever the server PC's LAN address changes. It updates
the shared `.env` values used by Django, Caddy, and the production launcher.

## Six PU computers and phone access

Six computers can use the same PU login at once. Each browser has its own
session, displayed as PU1 through PU6. A seventh computer is refused without
disconnecting the first six. Reopening a browser profile without its old login
cookie is a new session; the previous session expires or can be logged off from
Active Sessions. IP addresses are for identification, not for replacing sessions.

`MAX_PU_SESSIONS=6` sets computer capacity. `MAX_PU_PHONE_SESSIONS=4` allows up to
four additional phones through the **Connect Phone** QR without using computer
places. A fifth phone is refused without disconnecting any signed-in device.
Phone sessions expire after two hours. Admin sessions remain separate. If the
server's `.env` already sets `MAX_PU_PHONE_SESSIONS`, change it to `4` before
restarting Pharmacy; an explicit value overrides the new default.

The production launcher accepts client IP and HTTPS headers only from the local
Caddy proxy. Keep Waitress bound to `127.0.0.1` and port 8000 closed to the LAN.

For the new desktop at `192.168.0.22`, after installing this release and stopping
Pharmacy, run in its production folder:

```powershell
.\env\Scripts\python.exe configure_ip.py 192.168.0.22
.\production.bat start
.\production.bat status
```

The website is `https://192.168.0.22` and the dashboard phone QR must encode
`https://192.168.0.22/connect-phone/`, with no `:8000` port. Do not copy another
computer's `.env`, database, or Caddy keys to change the address.

Once the server is healthy, rebuild its workstation kit with
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build-workstation-kit.ps1`.
The kit uses the active server's public `caddy_data\caddy\pki\authorities\local\root.crt`
and the configured server address. Run `setup-workstation.bat` from that kit on
each Windows workstation. Phones need the same public root certificate trusted
in their operating system and must connect to the pharmacy Wi-Fi before scanning
the dashboard QR. Never distribute `.key` files or an IP-named leaf certificate.

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
