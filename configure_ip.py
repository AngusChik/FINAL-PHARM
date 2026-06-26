#!/usr/bin/env python
"""Point this deployment at a new machine's LAN IP address.

Updates .env (DJANGO_ALLOWED_HOSTS + DJANGO_CSRF_TRUSTED_ORIGINS) and swaps the
display IP in the server start scripts. Stdlib only — no Django needed.

Run it via configure_ip.bat, or directly:

    python configure_ip.py 192.168.1.42
"""
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
START_SCRIPTS = ["server_control.bat", "update.bat", "start_server.bat"]
LOOPBACK = {"127.0.0.1", "0.0.0.0", "localhost"}
LEGACY_DEFAULT_IP = "192.168.0.15"
IP_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")


def valid_ip(ip):
    m = IP_RE.match(ip or "")
    return bool(m) and all(0 <= int(octet) <= 255 for octet in m.groups())


def read_env_lines():
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def current_ip(lines):
    """The LAN IP currently configured (first non-loopback host), else the legacy default."""
    for line in lines:
        if line.strip().startswith("DJANGO_ALLOWED_HOSTS="):
            for host in line.split("=", 1)[1].split(","):
                host = host.strip()
                if host and host not in LOOPBACK:
                    return host
    return LEGACY_DEFAULT_IP


def set_env_key(lines, key, value):
    prefix = key + "="
    updated, found = [], False
    for line in lines:
        if line.strip().startswith(prefix):
            updated.append(f"{key}={value}")
            found = True
        else:
            updated.append(line)
    if not found:
        updated.append(f"{key}={value}")
    return updated


def main():
    new_ip = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not valid_ip(new_ip):
        print("Usage: python configure_ip.py <IPv4>   (e.g. 192.168.1.42)")
        return 1

    lines = read_env_lines()
    old_ip = current_ip(lines)

    lines = set_env_key(lines, "DJANGO_ALLOWED_HOSTS", f"{new_ip},localhost,127.0.0.1")
    lines = set_env_key(lines, "DJANGO_CSRF_TRUSTED_ORIGINS", f"https://{new_ip},https://localhost")
    ENV_PATH.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    print(f"  .env updated -> server will accept requests for {new_ip}")

    if old_ip != new_ip:
        for name in START_SCRIPTS:
            path = BASE_DIR / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if old_ip in text:
                path.write_text(text.replace(old_ip, new_ip), encoding="utf-8")
                print(f"  {name} -> updated {old_ip} to {new_ip}")

    print(f"\n  Done. Reach the app at http://{new_ip}:8000")
    print("  Restart the server for the change to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
