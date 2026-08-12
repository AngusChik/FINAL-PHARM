#!/usr/bin/env python
"""Write the server PC's LAN address to the shared .env configuration."""

import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
IP_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")


def valid_ip(ip):
    match = IP_RE.match(ip or "")
    return bool(match) and all(0 <= int(octet) <= 255 for octet in match.groups())


def read_env_lines():
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def set_env_key(lines, key, value):
    prefix = key + "="
    updated = []
    found = False
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
    lines = set_env_key(lines, "PHARMACY_HOST", new_ip)
    lines = set_env_key(
        lines,
        "DJANGO_ALLOWED_HOSTS",
        f"{new_ip},localhost,127.0.0.1",
    )
    lines = set_env_key(
        lines,
        "DJANGO_CSRF_TRUSTED_ORIGINS",
        f"https://{new_ip},https://localhost",
    )
    ENV_PATH.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")

    print(f"  .env updated for server {new_ip}")
    print(f"  Production URL: https://{new_ip}")
    print("  Restart production for the change to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
