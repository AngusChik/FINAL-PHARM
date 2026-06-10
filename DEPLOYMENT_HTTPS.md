# HTTPS Deployment Guide

This app is served by **waitress** (plain HTTP). To run it securely over the LAN,
**Caddy** sits in front and terminates TLS:

```
Staff browsers  ->  Caddy :443 (HTTPS)  ->  waitress 127.0.0.1:8000  ->  Django  ->  PostgreSQL
```

By default the repo ships in **plain-HTTP LAN mode** so nothing breaks. HTTPS is
opt-in via environment variables — follow the steps below to turn it on.

---

## Current default (no change needed)
- `DEBUG = False` (production-safe; set `DJANGO_DEBUG=true` in `.env` only for dev)
- HTTPS hardening **off** (`DJANGO_SECURE` unset) → cookies work over HTTP
- waitress still serves `http://192.168.0.15:8000` exactly as before

> ⚠️ After pulling this change, **restart waitress** (it does not auto-reload code).
> Because `DEBUG=False`, make sure `collectstatic` has run (the `.bat` scripts do this)
> — whitenoise serves the static files.

---

## Enabling HTTPS (one-time setup on the server PC)

1. **Install Caddy** — download the single `caddy.exe` from
   https://caddyserver.com/download and put it in the project folder (or on PATH).

2. **Bind waitress to localhost only** so it is reachable *only* through Caddy:
   ```
   waitress-serve --host=127.0.0.1 --port=8000 --threads=4 inventory.wsgi:application
   ```
   (`start_https.bat` does this for you.)

3. **Turn on Django's secure settings** — set the env var before launching:
   ```
   set DJANGO_SECURE=1
   ```
   or add `DJANGO_SECURE=1` to your `.env` file.

4. **Run Caddy** from the project folder (Administrator — it binds port 443):
   ```
   caddy run --config Caddyfile
   ```

5. **Firewall** — open **443** for the LAN and (optionally) close 8000:
   ```
   netsh advfirewall firewall add rule name="Pharmacy HTTPS" dir=in action=allow protocol=TCP localport=443 remoteip=192.168.0.0/24
   netsh advfirewall firewall delete rule name="Pharmacy App"
   ```

6. **Browse to** `https://192.168.0.15`

---

## Certificates on a LAN (IP, no public domain)
`tls internal` in the `Caddyfile` issues a **locally-trusted** certificate. Staff PCs
will see a certificate warning until Caddy's local root CA is trusted on each machine:

- On the **server PC**: `caddy trust` (installs the root CA locally), **or**
- Export Caddy's root CA and install it on each staff PC's "Trusted Root Certification
  Authorities" store.

For a small pharmacy LAN this is a one-time per-machine step.

---

## Environment variables (see `.env.example`)
| Variable | Purpose | Default |
|---|---|---|
| `DJANGO_SECRET_KEY` | Django secret key (set a real one!) | insecure dev fallback |
| `DJANGO_DEBUG` | `true` only for development | `False` |
| `DJANGO_SECURE` | `1` to enable HTTPS hardening (use with Caddy) | off |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | comma-separated https origins for CSRF | `https://192.168.0.15,https://localhost` |

## Rollback
Unset `DJANGO_SECURE` (and stop Caddy, re-bind waitress to `0.0.0.0:8000`) to return
to plain-HTTP LAN mode. `DEBUG` stays `False` in production regardless.
