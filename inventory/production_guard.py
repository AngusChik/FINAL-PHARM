"""Validation for code that is allowed to load production settings."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured


def _normalized_path(value: Path | str) -> Path:
    try:
        return Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ImproperlyConfigured(
            f"Production role path is invalid or unavailable: {value}"
        ) from exc


def validate_production_role(
    code_root: Path | str,
    role_root: Path | str | None = None,
) -> Path:
    """Return the authorized production root or fail closed.

    ``role_root`` differs from ``code_root`` only while the release controller
    checks candidate code against the real production environment. Scheduled
    jobs and ordinary runtime processes do not set that override, so moving the
    original checkout to development immediately prevents it from loading
    production settings.
    """

    resolved_code_root = _normalized_path(code_root)
    resolved_role_root = _normalized_path(role_root or resolved_code_root)
    marker_path = resolved_role_root / ".runtime" / "production-role.json"

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ImproperlyConfigured(
            "This checkout is not authorized to use production settings; "
            f"the production role marker is missing: {marker_path}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImproperlyConfigured(
            f"The production role marker is unreadable or invalid: {marker_path}"
        ) from exc

    required = {
        "schema_version",
        "role",
        "worktree",
        "branch",
        "remote",
        "created_at",
    }
    missing = sorted(required.difference(marker))
    if missing:
        raise ImproperlyConfigured(
            "The production role marker is missing: " + ", ".join(missing)
        )
    if marker["schema_version"] != 1 or marker["role"] != "production":
        raise ImproperlyConfigured(
            "The production role marker does not identify a supported "
            "production checkout."
        )
    if marker["branch"] != "main" or marker["remote"] != "origin":
        raise ImproperlyConfigured(
            "Production must be the main branch of the origin remote."
        )

    try:
        marked_root = _normalized_path(marker["worktree"])
    except (TypeError, ImproperlyConfigured) as exc:
        raise ImproperlyConfigured(
            "The production role marker has an invalid worktree path."
        ) from exc
    if marked_root != resolved_role_root:
        raise ImproperlyConfigured(
            "The production role marker belongs to a different checkout."
        )

    try:
        datetime.fromisoformat(str(marker["created_at"]).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ImproperlyConfigured(
            "The production role marker has an invalid created_at timestamp."
        ) from exc

    try:
        branch_result = subprocess.run(
            [
                "git",
                "-C",
                str(resolved_role_root),
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ImproperlyConfigured(
            "Git could not verify the production checkout branch."
        ) from exc
    if branch_result.returncode != 0 or branch_result.stdout.strip() != "main":
        raise ImproperlyConfigured(
            "Production settings require the authorized worktree to be on main."
        )
    try:
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(resolved_role_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ImproperlyConfigured(
            "Git could not verify production worktree cleanliness."
        ) from exc
    if status_result.returncode != 0 or status_result.stdout.strip():
        raise ImproperlyConfigured(
            "Production settings require a clean authorized main worktree."
        )

    production_env = resolved_role_root / ".env"
    if not production_env.is_file():
        raise ImproperlyConfigured(
            f"The authorized production environment file is missing: {production_env}"
        )

    if resolved_code_root != resolved_role_root and role_root is None:
        raise ImproperlyConfigured(
            "Candidate production checks require an explicit production role root."
        )

    return resolved_role_root
