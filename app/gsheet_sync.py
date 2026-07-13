"""Pull-only sync: import Ordering Sheet entries FROM a Google Spreadsheet.

Staff type rows into the Google Sheet (or submit a Google Form that feeds
it); the app pulls new rows in — via the "Pull from Google Sheet" button on
the Ordering Sheet page and automatically every 5 minutes (Task Scheduler:
PharmacyGSheetSync). The app never rewrites or deletes sheet content; its
only write-back is an "Imported" marker column so each row imports once.

Which tabs are read: every worksheet that *looks like* ordering data — its
header row must have a product/drug-name column plus at least one other
recognizable column (patient / qty / reasoning / urgency / initials / side /
phone). Google Form response tabs qualify automatically. Other tabs are
ignored.

Config (in .env):
  GSHEET_SPREADSHEET_ID    - required; the spreadsheet ID **or its full URL**
  GSHEET_CREDENTIALS_FILE  - optional; defaults to google_credentials.json

One-time setup: see GSHEET_SETUP.md (service account + share the sheet).
"""

import json
import os
import re
import time
from pathlib import Path

from django.conf import settings
from django.utils.timezone import localtime, now

from app.models import OrderingSheetEntry

BASE_DIR = Path(settings.BASE_DIR)
STATE_FILE = BASE_DIR / 'gsheet_sync_state.json'

IMPORTED_MARKER = 'Imported'


def _spreadsheet_id():
    """Accept a bare spreadsheet ID or a full docs.google.com URL."""
    raw = os.environ.get('GSHEET_SPREADSHEET_ID', '').strip()
    if not raw:
        return ''
    m = re.search(r'/d/([a-zA-Z0-9_-]{20,})', raw)
    if m:
        return m.group(1)
    # Bare id (possibly with stray spaces/params pasted around it)
    m = re.search(r'[a-zA-Z0-9_-]{20,}', raw)
    return m.group(0) if m else raw


def is_configured():
    return bool(_spreadsheet_id())


def _tab_selected(ws):
    """True if this worksheet should be pulled from. GSHEET_TAB (.env) may
    name ONE tab by title (case-insensitive) or by gid; unset = all
    ordering-shaped tabs."""
    target = os.environ.get('GSHEET_TAB', '').strip()
    if not target:
        return True
    return str(ws.id) == target or _norm(ws.title) == _norm(target)


def get_spreadsheet():
    """Open the configured spreadsheet with the service-account client."""
    if not is_configured():
        raise RuntimeError(
            "Google Sheet sync is not configured — set GSHEET_SPREADSHEET_ID "
            "in .env and provide google_credentials.json.")
    import gspread
    cred_path = BASE_DIR / os.environ.get('GSHEET_CREDENTIALS_FILE', 'google_credentials.json')
    if not cred_path.exists():
        raise RuntimeError(f"Google credentials file not found: {cred_path}")
    client = gspread.service_account(filename=str(cred_path))
    return client.open_by_key(_spreadsheet_id())


# ─── Choice mapping (forgiving: accepts keys, labels, or close variants) ──────

def _norm(s):
    return (s or '').strip().lower()


def _match_choice(value, choices, default=''):
    """Map a free-text answer to a choice key by key, label, or prefix."""
    v = _norm(value)
    if not v:
        return default
    for key, label in choices:
        if v == _norm(key) or v == _norm(label):
            return key
    for key, label in choices:
        if _norm(label).startswith(v) or v.startswith(_norm(key)) or v in _norm(label):
            return key
    # Free text may EMBED a choice ("order for basket, 3 boxes + more for
    # stock"): pick the longest label/key that appears as a whole phrase in
    # the text (longest wins, so "order for basket" beats "stock"). Short
    # needles (<4 chars, e.g. urgency key "na") are skipped to avoid false
    # hits inside ordinary words.
    best_key, best_len = None, 0
    for key, label in choices:
        for needle in (_norm(label), _norm(key)):
            if len(needle) >= 4 and len(needle) > best_len and re.search(
                    r'(?<![a-z0-9])' + re.escape(needle) + r'(?![a-z0-9])', v):
                best_key, best_len = key, len(needle)
    if best_key is not None:
        return best_key
    return default


def map_reasoning(value):
    return _match_choice(value, OrderingSheetEntry.REASON_CHOICES,
                         default=OrderingSheetEntry.REASON_STOCK)


def map_urgency(value):
    return _match_choice(value, OrderingSheetEntry.URGENCY_CHOICES,
                         default=OrderingSheetEntry.URGENCY_LOW)


def map_side(value):
    return _match_choice(value, OrderingSheetEntry.SIDE_CHOICES,
                         default=OrderingSheetEntry.SIDE_NA)


def map_entry_type(value):
    v = _norm(value)
    return OrderingSheetEntry.ENTRY_OTC if v.startswith('otc') else OrderingSheetEntry.ENTRY_DRUG


# ─── Worksheet import ────────────────────────────────────────────────────────

def _column_index(header, *needles):
    for i, h in enumerate(header):
        if any(n in h for n in needles):
            return i
    return None


def _detect_columns(header_row):
    """Map recognizable columns by keyword. Returns None if the tab doesn't
    look like ordering data (needs a name column + one other known column)."""
    header = [_norm(h) for h in header_row]
    idx = {
        'type': _column_index(header, 'type'),
        'patient': _column_index(header, 'patient'),
        'name': _column_index(header, 'drug', 'product', 'item', 'med', 'name'),
        'qty_needed': _column_index(header, 'needed'),
        'qty_remaining': _column_index(header, 'remaining', 'left'),
        'reasoning': _column_index(header, 'reason'),
        'urgency': _column_index(header, 'urgency', 'priority'),
        'side': _column_index(header, 'side'),
        'phone': _column_index(header, 'phone'),
        'initials': _column_index(header, 'initial'),
        # Free-text note column (may be the SAME column as reasoning, e.g. a
        # "Reasoning / Notes" header) — kept verbatim as the entry's comment.
        'note': _column_index(header, 'note', 'comment'),
    }
    if idx['name'] is None:
        return None, None
    others = [k for k, v in idx.items() if k != 'name' and v is not None]
    if not others:
        return None, None
    marker = _column_index(header, 'imported')
    return idx, marker


def import_worksheet(ws):
    """Import unmarked data rows from one worksheet. Returns count (or None
    if the tab doesn't look like ordering data and was skipped)."""
    values = ws.get_all_values()
    if not values:
        return None
    idx, marker_col = _detect_columns(values[0])
    if idx is None:
        return None
    if marker_col is None:
        marker_col = len(values[0])
        ws.update_cell(1, marker_col + 1, IMPORTED_MARKER)

    def cell(row, key):
        i = idx.get(key)
        return row[i].strip() if i is not None and i < len(row) and isinstance(row[i], str) else ''

    # Content-based dedup: never add a row that already matches an ACTIVE
    # entry in the app (by name + patient, case-insensitive). This covers the
    # marker being cleared, the same item retyped, or an item already added in
    # the app directly. Key = (name_norm, patient_norm).
    existing = {
        (_norm(n), _norm(p))
        for n, p in OrderingSheetEntry.objects.filter(is_deleted=False)
                       .values_list('name', 'patient_name')
    }

    imported = 0
    for row_index, row in enumerate(values[1:], start=2):
        if not any(str(c).strip() for c in row):
            continue
        if marker_col < len(row) and str(row[marker_col]).strip():
            continue  # already imported
        name = cell(row, 'name')
        if not name:
            continue
        key = (_norm(name), _norm(cell(row, 'patient')))
        if key in existing:
            # Already on the webapp — mark it so we don't re-check, but don't
            # create a duplicate.
            ws.update_cell(row_index, marker_col + 1,
                           f"exists {localtime(now()).strftime('%d %b %H:%M')}")
            continue
        entry_type = map_entry_type(cell(row, 'type'))
        kwargs = {
            'entry_type': entry_type,
            'name': name[:200],
            'patient_name': cell(row, 'patient')[:200],
            'quantity_needed': cell(row, 'qty_needed')[:50],
            'quantity_remaining': cell(row, 'qty_remaining')[:50],
            'initials': (cell(row, 'initials') or 'GS')[:20],
            'source': OrderingSheetEntry.SOURCE_GSHEET,
        }
        if entry_type == OrderingSheetEntry.ENTRY_OTC:
            kwargs['side'] = map_side(cell(row, 'side'))
            kwargs['phone_number'] = cell(row, 'phone')[:20]
            kwargs['urgency'] = OrderingSheetEntry.URGENCY_NA
        else:
            kwargs['reasoning'] = map_reasoning(cell(row, 'reasoning'))
            kwargs['urgency'] = map_urgency(cell(row, 'urgency'))
        # Preserve free text as the entry's comment: an explicit note column
        # if there is one, else the raw reasoning text when it says more than
        # just a reasoning label ("order for basket, 3 boxes + more…").
        reason_labels = {_norm(l) for _, l in OrderingSheetEntry.REASON_CHOICES}
        note = cell(row, 'note') or (
            cell(row, 'reasoning') if entry_type == OrderingSheetEntry.ENTRY_DRUG else '')
        if note and _norm(note) not in reason_labels:
            kwargs['order_note'] = note[:255]
        OrderingSheetEntry.objects.create(**kwargs)
        existing.add(key)  # guard against duplicate rows within this same sheet
        ws.update_cell(row_index, marker_col + 1,
                       f"✓ {localtime(now()).strftime('%d %b %H:%M')}")
        imported += 1
    return imported


# ─── Entry point ─────────────────────────────────────────────────────────────

def sync_all():
    """Pull new rows from every ordering-shaped tab. Returns a result dict
    (also saved to the state file for the page caption)."""
    result = {'last_sync': time.time(), 'imported': 0, 'tabs': [], 'errors': []}
    try:
        ss = get_spreadsheet()
    except Exception as e:
        result['errors'].append(str(e))
        _save_state(result)
        return result

    try:
        worksheets = ss.worksheets()
    except Exception as e:
        result['errors'].append(str(e))
        _save_state(result)
        return result

    worksheets = [ws for ws in worksheets if _tab_selected(ws)]
    if not worksheets:
        result['errors'].append(
            f"Configured tab (GSHEET_TAB={os.environ.get('GSHEET_TAB', '')!r}) "
            "was not found in the spreadsheet.")
        _save_state(result)
        return result

    for ws in worksheets:
        try:
            count = import_worksheet(ws)
        except Exception as e:
            result['errors'].append(f"{ws.title}: {e}")
            continue
        if count is None:
            continue  # not ordering data — ignored
        result['tabs'].append({'title': ws.title, 'imported': count})
        result['imported'] += count

    if not result['tabs'] and not result['errors']:
        result['errors'].append(
            "No readable tab has a recognizable header row "
            "(needs a Drug/Product Name column plus e.g. Initials/Urgency).")

    _save_state(result)
    return result


def _save_state(result):
    try:
        STATE_FILE.write_text(json.dumps(result), encoding='utf-8')
    except Exception:
        pass


def load_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return None
