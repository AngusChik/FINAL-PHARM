"""
Kohl & Frisch (KFConnect) re-order helper.

Reads the Recently Purchased list from the pharmacy database, opens KFConnect
(https://kfconnect.kohlandfrisch.com), signs in, goes to the Item Catalogue,
and searches each product by barcode to add it to the cart with the recorded
quantity. It NEVER submits the order — it stops at the cart so you review and
place the order yourself.

This is the Kohl & Frisch twin of mckesson_order.py: identical data (Recently
Purchased -> barcode + prediction quantity), different website. It can be run
from the console OR driven by the web app (the "Order on Kohl & Frisch" button
on the Recently Purchased page) via --no-input + --status-file.

Setup (one time, inside the FINAL-PHARM venv):
    env\\Scripts\\activate
    pip install playwright
    playwright install chromium

Usage:
    python kohlfrisch_order.py --dry-run        # print what would be ordered
    python kohlfrisch_order.py --limit 2        # first live test with 2 items
    python kohlfrisch_order.py --days 7         # only items sold in the last 7 days
    python kohlfrisch_order.py                  # full run

Login: the first run opens the KFConnect / Microsoft (Azure B2C) sign-in page
— sign in manually in the browser window, then press Enter in this console.
The session is saved in .kohlfrisch_profile\\ so later runs skip the login.
No credentials are stored or typed by this script.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Django bootstrap (same DB the app uses; .env is loaded by settings) ---
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "inventory.settings")
import django  # noqa: E402

django.setup()

from app.mckesson import collect_order_items  # noqa: E402

# ---------------------------------------------------------------------------
# KFConnect page config — THE ONLY PART THAT SHOULD NEED TUNING.
#
# Each entry is a list of candidate CSS selectors tried in order; the first
# VISIBLE one wins. The real DOM is unknown until we run against it logged in,
# so these are best-guess starting points — fix them here after the first run
# using the on-screen page / the kf_debug_*.html snapshots it saves.
# ---------------------------------------------------------------------------
PORTAL_URL = "https://kfconnect.kohlandfrisch.com/en-US/"
# Any of these appearing in the URL means we're on the Microsoft/B2C login flow.
LOGIN_HOSTS = ("b2clogin.com", "kohlandfrischprod", "login.microsoftonline.com")

SELECTORS = {
    # Landing-page "Sign in" button/link (only shown when logged out)
    "sign_in_button": [
        "a:has-text('Sign in')",
        "button:has-text('Sign in')",
        "a:has-text('Sign In')",
        "button:has-text('Log in')",
        "a[href*='signin' i]",
    ],
    # Top-nav "Item Catalogue" tab
    "item_catalogue_tab": [
        "a:has-text('Item Catalogue')",
        "a:has-text('Item Catalog')",
        "a:has-text('Catalogue')",
        "button:has-text('Item Catalogue')",
        "[href*='catalog' i]",
    ],
    # Barcode / product search box on the catalogue page
    "barcode_search": [
        "input[placeholder*='barcode' i]",
        "input[placeholder*='UPC' i]",
        "input[placeholder*='search' i]",
        "input[type='search']",
        "input[name*='search' i]",
    ],
    # The cart icon at the end of a search-result row. Confirmed from the live
    # DOM: <button class="catalogueBtn addtocart" aria-label="Add to Cart"
    #              onclick="getDetails('<guid>')"></button>
    "row_cart_button": [
        "table tbody tr button.addtocart",
        "table tbody tr button.catalogueBtn.addtocart",
        "table tbody tr button[aria-label='Add to Cart' i]",
        "button.addtocart",
        "[aria-label='Add to Cart' i]",
        "table tbody tr [class*='addtocart' i]",
        "table tbody tr td:last-child button",
    ],
    # A red circle-X on the result row meaning the product is unavailable /
    # out of stock. Best-guess (title/aria/class); tune from a live snapshot.
    "unavailable_marker": [
        "table tbody tr [title*='navailable' i]",
        "table tbody tr [title*='not available' i]",
        "table tbody tr [title*='out of stock' i]",
        "table tbody tr [title*='discontinued' i]",
        "table tbody tr [aria-label*='navailable' i]",
        "table tbody tr [data-bs-original-title*='navailable' i]",
        "table tbody tr [data-bs-original-title*='not available' i]",
        "table tbody tr [class*='unavailable' i]",
        "table tbody tr [class*='outofstock' i]",
        "table tbody tr [class*='out-of-stock' i]",
    ],
    # The "Add to Cart" modal that opens after clicking the cart icon.
    "cart_modal": [
        "[class*='modal' i]:has-text('Add to Cart')",
        "[role='dialog']:has-text('Add to Cart')",
        "div:has-text('Add to Cart')",
    ],
    # Some products open an "Add to Watchlist" modal instead of Add to Cart.
    "watchlist_modal": [
        "[class*='modal' i]:has-text('Add to Watchlist')",
        "[role='dialog']:has-text('Add to Watchlist')",
        "[class*='modal' i]:has-text('Watch List')",
        "div:has-text('Add to Watchlist')",
    ],
    # Quantity box inside the Add to Cart modal.
    "qty_input": [
        "input[type='number']",
        "input[name*='qty' i]",
        "input[name*='quant' i]",
    ],
    # "Order Reference" box in the Create-a-new-Cart section (placeholder
    # "Add Order Reference").
    "order_reference": [
        "input[placeholder*='Order Reference' i]",
        "input[placeholder*='Reference' i]",
        "input[name*='reference' i]",
    ],
    # Name box in the Create-a-new-Watchlist section (best-guess until the
    # watchlist modal DOM is seen).
    "watchlist_reference": [
        "input[placeholder*='Watchlist' i]",
        "input[placeholder*='Watch List' i]",
        "input[placeholder*='Reference' i]",
        "input[name*='watch' i]",
    ],
    # The "ADD" button at the bottom of the modal (styled uppercase; has-text
    # is case-insensitive).
    "modal_add_button": [
        "button:has-text('Add')",
        "span.btn:has-text('Add')",
        "a:has-text('Add')",
    ],
    # Something only present when logged in (used to confirm login state)
    "logged_in_marker": [
        "a:has-text('Item Catalogue')",
        "[class*='cart' i]",
        "[class*='account' i]",
        "nav",
    ],
}

# KFConnect drops a full-screen loading overlay over the page during ajax
# calls (search, opening the cart modal). It swallows clicks, so we wait for it
# to clear before interacting.
OVERLAY_SELECTORS = ("#processing-screen", ".full-screen-loading-div")

# DataTables shows this text in the results grid when a barcode matches nothing.
NO_DATA_TEXT = "no data available in table"
# Label next to the "create a new cart" checkbox in the Add to Cart modal.
CREATE_NEW_CART_LABEL = "Create a new Cart"
# Label next to the "create a new watchlist" checkbox in the Add to Watchlist
# modal (best-guess wording until the modal DOM is seen).
CREATE_NEW_WATCHLIST_LABEL = "Create a new Watchlist"

# Pause between items. The real pacing is settle() waiting out the overlay, so
# this can be ~0 — the next item's settle() naturally throttles us anyway.
THROTTLE_SECONDS = 0.05
PROFILE_DIR = BASE_DIR / ".kohlfrisch_profile"
REPORT_PATH = BASE_DIR / "kohlfrisch_order_report.csv"

# How long to wait for the user to act in the browser (login) in --no-input mode.
USER_ACTION_TIMEOUT_S = 300


class Status:
    """Progress written to a JSON file so the web app can poll it.

    States: starting | login | waiting_user | running | review | done | error
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.data = {
            "state": "starting", "current": 0, "total": 0, "message": "",
            "added": [], "skipped": [], "report_path": str(REPORT_PATH),
            "pid": os.getpid(),
        }
        self.update()

    def update(self, **kw):
        self.data.update(kw)
        self.data["updated_at"] = time.time()
        if not self.path:
            return
        try:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            pass


def first_visible(scope, candidates, timeout_ms=0):
    """First visible locator among candidate selectors, else None. Candidates
    are polled in rounds every 200 ms so a non-matching first guess can't burn
    the whole timeout."""
    deadline = time.time() + timeout_ms / 1000
    while True:
        for sel in candidates:
            loc = scope.locator(sel).first
            try:
                if loc.is_visible():
                    return loc
            except Exception:
                continue
        if time.time() >= deadline:
            return None
        time.sleep(0.2)


def settle(page, timeout_ms=20000):
    """Wait for KFConnect's full-screen loading overlay to clear before we act.
    Its clicks are otherwise intercepted by <div id="processing-screen">."""
    for sel in OVERLAY_SELECTORS:
        try:
            page.locator(sel).first.wait_for(state="hidden", timeout=timeout_ms)
        except Exception:
            pass


def robust_click(page, locator, timeout_ms=8000):
    """Click through KFConnect's loading overlay: wait it out, then fall back to
    a forced click that ignores pointer-event interception."""
    settle(page)
    try:
        locator.click(timeout=timeout_ms)
        return True
    except Exception as e:
        if "closed" in str(e).lower():
            raise
    try:
        settle(page, timeout_ms=6000)
        locator.click(timeout=timeout_ms, force=True)
        return True
    except Exception as e:
        if "closed" in str(e).lower():
            raise
    return False


def open_cart_modal(page, cart_btn):
    """Open the Add to Cart modal for a result row. Prefer invoking the row's
    own getDetails('<guid>') handler so the loading overlay can't swallow the
    click; fall back to a robust click on the button."""
    try:
        onclick = cart_btn.get_attribute("onclick") or ""
        m = re.search(r"getDetails\(\s*'([^']+)'\s*\)", onclick)
        if m:
            settle(page)
            page.evaluate("(id) => getDetails(id)", m.group(1))
            return True
    except Exception:
        pass
    return robust_click(page, cart_btn)


def input_after_label(scope, text):
    """The first <input> that follows a bit of label text (e.g. "Quantity")."""
    try:
        node = scope.get_by_text(re.compile(text, re.I)).first
        inp = node.locator("xpath=following::input[1]").first
        if inp.count():
            return inp
    except Exception:
        pass
    return None


def checkbox_for_label(scope, text):
    """The checkbox associated with a label (e.g. "Create a new Cart")."""
    # A <label> wrapping the checkbox
    try:
        lbl = scope.locator(f"label:has-text('{text}')").first
        if lbl.count():
            cb = lbl.locator("input[type='checkbox']").first
            if cb.count():
                return cb
    except Exception:
        pass
    # Otherwise the checkbox immediately preceding the text
    try:
        node = scope.get_by_text(re.compile(text, re.I)).first
        cb = node.locator("xpath=preceding::input[@type='checkbox'][1]").first
        if cb.count():
            return cb
    except Exception:
        pass
    return None


def radio_for_ref(scope, ref):
    """The radio button of the "existing cart/watchlist" row matching ref."""
    try:
        node = scope.get_by_text(ref, exact=False).first
        if node.count():
            # Radio inside the same row as the reference text…
            radio = node.locator(
                "xpath=ancestor-or-self::*[.//input[@type='radio']][1]"
                "//input[@type='radio']").first
            if radio.count():
                return radio
            # …otherwise the nearest radio after/before the text.
            radio = node.locator("xpath=following::input[@type='radio'][1]").first
            if radio.count():
                return radio
            radio = node.locator("xpath=preceding::input[@type='radio'][1]").first
            if radio.count():
                return radio
    except Exception:
        pass
    return None


def set_checkbox(cb, checked):
    """Force a checkbox to the desired state, falling back to a click. Bounded
    to 5s so it can't hang on the 10s default actionability timeout."""
    if cb is None:
        return
    try:
        if cb.is_checked() != checked:
            cb.check(timeout=5000) if checked else cb.uncheck(timeout=5000)
    except Exception:
        try:
            cb.click(timeout=5000, force=True)
        except Exception:
            pass


def dump_debug(page, tag):
    """Save the page HTML so failing selectors can be fixed from the real DOM."""
    path = BASE_DIR / f"kf_debug_{tag}.html"
    try:
        path.write_text(page.content(), encoding="utf-8")
        print(f"    (page snapshot saved to {path.name} — send this file to Claude to fix the selectors)")
    except Exception:
        pass


def on_login_page(page):
    return any(host in page.url for host in LOGIN_HOSTS)


def ensure_logged_in(page, status, no_input=False):
    """Open KFConnect and make sure we're signed in (manual first run)."""
    page.goto(PORTAL_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)  # let the SPA settle

    # If a "Sign in" button is on the landing page, click it to start the flow.
    if not on_login_page(page):
        btn = first_visible(page, SELECTORS["sign_in_button"], timeout_ms=4000)
        if btn is not None:
            try:
                btn.click()
                page.wait_for_timeout(2500)
            except Exception:
                pass

    if on_login_page(page):
        if no_input:
            status.update(state="login",
                          message="Sign in to Kohl & Frisch in the browser window that just opened")
            print(">>> Waiting for KFConnect login in the browser window...")
            deadline = time.time() + USER_ACTION_TIMEOUT_S
            while on_login_page(page):
                if time.time() > deadline:
                    raise RuntimeError("Timed out waiting for login (5 minutes).")
                page.wait_for_timeout(1000)
            page.wait_for_timeout(3000)
        else:
            print("\n>>> KFConnect login required.")
            print(">>> Sign in (and complete any 2FA) in the browser window,")
            input(">>> then come back here and press Enter to continue... ")
            for _ in range(40):
                if not on_login_page(page):
                    break
                page.wait_for_timeout(1000)
            if on_login_page(page):
                raise RuntimeError("Still on the login page — aborting.")
            page.wait_for_timeout(3000)
    print(f"Logged in — KFConnect at {page.url}")


def open_catalogue(page, status, no_input=False):
    """Click the Item Catalogue nav tab and confirm the search box appears."""
    tab = first_visible(page, SELECTORS["item_catalogue_tab"], timeout_ms=10000)
    if tab is not None:
        try:
            tab.click()
            page.wait_for_load_state("domcontentloaded", timeout=10000)
            page.wait_for_timeout(1500)
        except Exception:
            pass

    search = first_visible(page, SELECTORS["barcode_search"], timeout_ms=8000)
    if search is not None:
        print("Item Catalogue open.")
        return
    dump_debug(page, "catalogue")
    if no_input:
        status.update(state="waiting_user",
                      message="Open the Item Catalogue in the Kohl & Frisch window so the "
                              "barcode search box is visible")
        print(">>> Waiting for the Item Catalogue / barcode search in the browser...")
        deadline = time.time() + USER_ACTION_TIMEOUT_S
        while time.time() < deadline:
            page.wait_for_timeout(1500)
            if first_visible(page, SELECTORS["barcode_search"], timeout_ms=0) is not None:
                return
        raise RuntimeError("Timed out waiting for the Item Catalogue (5 minutes).")
    print("\n>>> Couldn't open the Item Catalogue / find the barcode search box.")
    print(">>> In the browser: click 'Item Catalogue' so the barcode search is visible,")
    input(">>> then press Enter here to start adding products... ")


def kf_search_codes(barcode):
    """Barcodes to try in the K&F catalogue search, in priority order.

    K&F stores UPC-A as the full 12 digits, but our app often holds the
    11-digit form with the leading zero dropped (e.g. 59972101604 ->
    059972101604). So for an 11-digit numeric barcode we search the PADDED
    12-digit form first — that's the canonical K&F code — and fall back to the
    raw 11-digit only if the padded one finds nothing. Everything else is
    searched as-is.
    """
    b = (barcode or "").strip()
    if not b:
        return []
    if len(b) == 11 and b.isdigit():
        return ["0" + b, b]
    return [b]


def add_item(page, item, state, cart_ref, wl_ref):
    """Search one barcode and add it to the cart (or watchlist). (ok, reason).

    KFConnect flow (from screenshots):
      search barcode -> results table -> click the row's cart icon -> a modal
      opens. Most products open "Add to Cart"; some open "Add to Watchlist"
      instead. We react to whichever one appears.

    Destination choice (mirrors for both cart and watchlist):
      * The FIRST item that lands in a given destination ticks "Create a new
        Cart/Watchlist" and names it (cart_ref / wl_ref), so ADD makes a fresh
        one we can find again.
      * Every item after picks that same named entry under "Add to an existing
        Cart/Watchlist" (falling back to the first listed one).

    `state` is a dict of {'cart_created': bool, 'wl_created': bool} updated in
    place so the caller knows a cart/watchlist now exists.
    """
    settle(page)  # a previous add may still be committing behind the overlay
    t0 = time.time()  # per-phase timing so a slow run shows where the time goes

    # Try the exact barcode first; for 11-digit UPCs fall back to the padded
    # 12-digit form only if the exact search finds nothing.
    codes = kf_search_codes(item["barcode"])
    if not codes:
        return False, "no barcode to search"

    cart_btn = None
    saw_no_data = False
    unavailable = False
    for code in codes:
        search = first_visible(page, SELECTORS["barcode_search"], timeout_ms=8000)
        if search is None:
            return False, "barcode search box not found (adjust SELECTORS['barcode_search'])"
        search.click()
        search.fill("")
        search.fill(code)
        search.press("Enter")
        # The search is a server round-trip that raises a loading overlay; a
        # short beat lets it appear, then settle() waits for it to clear so
        # results are rendered and clicks aren't swallowed.
        page.wait_for_timeout(120)
        settle(page)

        # Race three outcomes so a search returns FAST: the row's cart icon
        # renders, the grid shows "No data available", or the row is flagged
        # unavailable (red circle-X = out of stock).
        saw_no_data = False
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                if page.get_by_text(re.compile(NO_DATA_TEXT, re.I)).first.is_visible():
                    saw_no_data = True
                    break
            except Exception:
                pass
            if first_visible(page, SELECTORS["unavailable_marker"], timeout_ms=0) is not None:
                unavailable = True
                break
            cart_btn = first_visible(page, SELECTORS["row_cart_button"], timeout_ms=0)
            if cart_btn is not None:
                break
            page.wait_for_timeout(150)
        if unavailable or cart_btn is not None:
            break  # resolved this code — don't try the fallback
        # else: no match for this code — try the next candidate (padded UPC)

    if unavailable:
        return False, "unavailable — out of stock at Kohl & Frisch"
    if cart_btn is None:
        if saw_no_data:
            return False, "no results — barcode not in the Kohl & Frisch catalogue"
        dump_debug(page, "search_results")
        return False, "cart icon not found on the result row (adjust SELECTORS['row_cart_button'])"
    t_search = time.time()

    # Open the modal by invoking the row's own getDetails() handler — this
    # bypasses the loading overlay that otherwise swallows the click.
    if not open_cart_modal(page, cart_btn):
        dump_debug(page, "search_results")
        return False, "couldn't open the Add to Cart modal for this row"

    settle(page)  # getDetails() shows the overlay while the modal loads

    # K&F opens EITHER an "Add to Cart" or an "Add to Watchlist" modal; detect
    # which so we route the item to the right destination.
    modal = None
    is_watchlist = False
    deadline = time.time() + 10
    while time.time() < deadline and modal is None:
        m = first_visible(page, SELECTORS["cart_modal"], timeout_ms=0)
        if m is not None:
            modal, is_watchlist = m, False
            break
        m = first_visible(page, SELECTORS["watchlist_modal"], timeout_ms=0)
        if m is not None:
            modal, is_watchlist = m, True
            break
        page.wait_for_timeout(120)
    if modal is None:
        dump_debug(page, "cart_modal")
        return False, "Add to Cart / Watchlist modal did not open"
    t_modal = time.time()

    # Route to cart vs watchlist.
    if is_watchlist:
        dest = "watchlist"
        create_new = not state["wl_created"]
        create_label = CREATE_NEW_WATCHLIST_LABEL
        ref = wl_ref
        ref_selectors = SELECTORS["watchlist_reference"]
        ref_label = "Watchlist"
        debug_tag = "watchlist_modal"
    else:
        dest = "cart"
        create_new = not state["cart_created"]
        create_label = CREATE_NEW_CART_LABEL
        ref = cart_ref
        ref_selectors = SELECTORS["order_reference"]
        ref_label = "Order Reference"
        debug_tag = "cart_modal"

    # Quantity = the suggested quantity from the reorder prediction. Required
    # for the cart; a watchlist may not offer it, so don't fail if it's absent.
    qty = item["quantity"]
    qty_input = (first_visible(modal, SELECTORS["qty_input"], timeout_ms=700)
                 or input_after_label(modal, "Quantity"))
    if qty_input is not None:
        try:
            qty_input.click()
            qty_input.fill(str(qty))
        except Exception:
            pass
    elif not is_watchlist:
        page.keyboard.press("Escape")
        dump_debug(page, debug_tag)
        return False, "Quantity field not found in the Add to Cart modal"

    # Choose the destination (create a new one, or attach to the one we made).
    cb = checkbox_for_label(modal, create_label)
    if create_new:
        set_checkbox(cb, True)  # tick "Create a new Cart/Watchlist"
        # Name it so later items can attach to this exact cart/watchlist.
        ref_input = (first_visible(modal, ref_selectors, timeout_ms=600)
                     or input_after_label(modal, ref_label))
        if ref_input is not None:
            try:
                ref_input.click()
                ref_input.fill(ref)
            except Exception:
                pass
    else:
        # Add to the SAME cart created earlier. Untick "Create a new Cart"
        # FIRST so K&F switches to add-to-existing mode and renders the cart
        # list — unticking AFTER selecting resets that list and drops our
        # choice, which is why items were "found but not added". Then select
        # OUR cart by its reference so every item lands in the same one.
        set_checkbox(cb, False)
        settle(page)
        radio = radio_for_ref(modal, ref)
        if radio is None or not radio.count():
            radio = modal.locator("input[type='radio']").first
        try:
            radio.check(timeout=5000)
        except Exception:
            try:
                radio.click(timeout=5000, force=True)
            except Exception:
                pass
        # A cart MUST be selected before ADD, or the add silently no-ops —
        # re-select once if the first attempt didn't take.
        try:
            if not radio.is_checked():
                radio.check(timeout=3000, force=True)
        except Exception:
            pass

    add = first_visible(modal, SELECTORS["modal_add_button"], timeout_ms=1000)
    if add is None:
        dump_debug(page, debug_tag)
        page.keyboard.press("Escape")
        return False, f"ADD button not found in the Add to {dest.title()} modal"
    # Fast path: click ADD straight away (the overlay is normally gone by now,
    # so no leading settle wait). Fall back to the overlay-safe click only if
    # this one is actually blocked.
    try:
        add.click(timeout=1500)
    except Exception as e:
        if "closed" in str(e).lower():
            raise
        if not robust_click(page, add, timeout_ms=6000):
            dump_debug(page, debug_tag)
            return False, f"couldn't click ADD in the Add to {dest.title()} modal (overlay?)"

    # The modal closing is the signal the add registered. Creating a cart is
    # slower, so allow time; if it never closes, the add did NOT go through —
    # report the item as not-added instead of a silent false success.
    try:
        modal.wait_for(state="hidden", timeout=7000)
    except Exception:
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass
        settle(page)
        dump_debug(page, debug_tag)
        return False, f"add did not register — the {dest} modal stayed open"
    settle(page)  # let the save overlay clear before the next item

    # Record that this destination now exists, so later items attach to it.
    if is_watchlist:
        state["wl_created"] = True
    else:
        state["cart_created"] = True
    # Per-phase timing (search / open-modal / add) so slow items are diagnosable.
    now_t = time.time()
    timing = f" [search {t_search - t0:.1f}s, modal {t_modal - t_search:.1f}s, add {now_t - t_modal:.1f}s]"
    return True, f"added x{qty} to {dest}{' (new)' if create_new else ''}{timing}"


def write_report(results):
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["status", "name", "barcode", "quantity", "reason"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nReport written to {REPORT_PATH}")


def run(args, status):
    if args.items_file:
        # The exact (possibly user-edited) list from the web preview — use it
        # verbatim instead of recomputing.
        data = json.loads(Path(args.items_file).read_text(encoding="utf-8"))
        items = data.get("items", [])
        pre_skipped = data.get("skipped", [])
        if args.limit:
            items = items[: args.limit]
    else:
        exclude_ids = []
        if args.exclude_category_ids:
            exclude_ids = [int(x) for x in args.exclude_category_ids.split(",") if x.strip()]
        items, pre_skipped = collect_order_items(
            days=args.days, limit=args.limit, qty_mode=args.qty,
            exclude_category_ids=exclude_ids,
        )

    print(f"\n{len(items)} item(s) to order, {len(pre_skipped)} skipped:\n")
    for it in items:
        print(f"  {it['quantity']:>4} x {it['name']}  [{it['barcode']}]")
    for sk in pre_skipped:
        print(f"  SKIP   {sk['name']} — {sk['reason']}")

    status.update(total=len(items),
                  skipped=[{"name": sk["name"], "reason": sk["reason"],
                            "barcode": sk.get("barcode", "")} for sk in pre_skipped])

    if args.dry_run:
        status.update(state="done", message="Dry run — nothing sent to Kohl & Frisch")
        return
    if not items:
        print("Nothing to order.")
        status.update(state="done", message="Nothing to order after filtering")
        return

    from playwright.sync_api import sync_playwright

    results = [{"status": "skipped", "reason": sk["reason"],
                **{k: sk[k] for k in ("name", "barcode", "quantity")}}
               for sk in pre_skipped]

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            args=["--start-maximized"],
            no_viewport=True,
        )
        ctx.set_default_timeout(10000)  # fail fast instead of 30 s hangs

        # Closing the Chrome window at any point stops the whole process —
        # no orphaned background run. `finishing` guards our own intentional
        # close at the end so it doesn't look like an abort.
        state = {"finishing": False}

        def _on_close(*_):
            if state["finishing"]:
                return
            print("\nBrowser closed — ending the run.")
            status.update(state="done", message="Browser closed — run ended by user")
            os._exit(0)

        ctx.on("close", _on_close)
        for pg in ctx.pages:
            pg.on("close", _on_close)

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        ensure_logged_in(page, status, no_input=args.no_input)
        status.update(state="running", message="Opening the Item Catalogue")
        open_catalogue(page, status, no_input=args.no_input)

        # The first item that lands in each destination creates it (named with
        # these references); everything after attaches to the same one.
        cart_ref = "autosession(" + datetime.now().strftime("%Y%m%d-%H%M%S") + ")"
        wl_ref = "AUTOSESSION(" + datetime.now().strftime("%Y%m%d") + ")"
        print(f"New cart reference: {cart_ref}")
        print(f"New watchlist reference (if needed): {wl_ref}")
        state = {"cart_created": False, "wl_created": False}
        for i, item in enumerate(items, 1):
            status.update(state="running", current=i,
                          message=f"{item['name']} x{item['quantity']}")
            print(f"[{i}/{len(items)}] {item['name']} x{item['quantity']} ... ", end="", flush=True)
            try:
                ok, reason = add_item(page, item, state, cart_ref=cart_ref, wl_ref=wl_ref)
            except Exception as e:
                msg = str(e)
                if "closed" in msg.lower():
                    raise RuntimeError("The Kohl & Frisch browser window was closed before the run finished.")
                ok, reason = False, f"error: {msg}"
            print(reason)
            results.append({"status": "added" if ok else "skipped", "reason": reason, **item})
            if ok:
                status.data["added"].append({"product_id": item.get("product_id"),
                                             "name": item["name"], "qty": item["quantity"],
                                             "barcode": item.get("barcode", "")})
            else:
                status.data["skipped"].append({"name": item["name"], "reason": reason,
                                               "barcode": item.get("barcode", "")})
            status.update()
            time.sleep(THROTTLE_SECONDS)

        write_report(results)
        added = sum(1 for r in results if r["status"] == "added")
        print(f"\nDone: {added} added, {len(results) - added} skipped.")
        print("The browser stays open — review the cart and submit the order yourself.")
        if args.no_input:
            status.update(state="review",
                          message=f"{added} added — review and submit in the Kohl & Frisch window, "
                                  "then close it")
            print("Close the browser window when finished.")
            # The ctx/page 'close' handler ends the process the moment the
            # window is closed. This loop just heartbeats the status file (so
            # the server sees the run as alive) and is a backstop in case the
            # close event doesn't fire.
            while True:
                try:
                    if not ctx.pages:
                        break
                    ctx.pages[0].evaluate("1")  # cheap liveness ping
                except Exception:
                    break  # browser/page gone
                status.update()  # heartbeat so the server knows we're alive
                time.sleep(2)
            status.update(state="done", message=f"{added} added, {len(results) - added} skipped")
        else:
            input("Press Enter here when finished to close the browser... ")
            state["finishing"] = True  # our own close — don't treat as an abort
            ctx.close()
            status.update(state="done", message=f"{added} added, {len(results) - added} skipped")


def main():
    ap = argparse.ArgumentParser(description="Fill the Kohl & Frisch cart from Recently Purchased products.")
    ap.add_argument("--days", type=int, default=None, help="only include items sold in the last N days")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N items (for testing)")
    ap.add_argument("--dry-run", action="store_true", help="print the order list and exit (no browser)")
    ap.add_argument("--qty", choices=["predicted", "sold"], default="predicted",
                    help="quantity source: the app's reorder-prediction formula (default) or units sold")
    ap.add_argument("--exclude-category-ids", default="",
                    help="comma-separated category ids to skip (e.g. Snacks)")
    ap.add_argument("--items-file", default=None,
                    help="JSON file with the exact items to order (from the web preview); "
                         "overrides --exclude-category-ids/--days/--qty")
    ap.add_argument("--status-file", default=None,
                    help="write progress JSON here (used by the web app)")
    ap.add_argument("--no-input", action="store_true",
                    help="never prompt on the console; wait for browser actions instead")
    args = ap.parse_args()

    status = Status(args.status_file)
    try:
        run(args, status)
    except Exception as e:
        status.update(state="error", message=str(e))
        raise


if __name__ == "__main__":
    main()
