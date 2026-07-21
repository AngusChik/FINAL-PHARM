"""
McKesson PharmaClik re-order helper.

Reads the Recently Purchased list from the pharmacy database, opens
PharmaClik (https://clients.mckesson.ca) in a real browser, searches each
product by barcode/UPC and adds it to the cart with the recorded quantity.
It NEVER submits the order — it stops at the cart so you can review and
place the order yourself.

Setup (one time, inside the FINAL-PHARM venv):
    env\\Scripts\\activate
    pip install playwright
    playwright install chromium

Usage:
    python mckesson_order.py --dry-run          # just print what would be ordered
    python mckesson_order.py --limit 2          # first live test with 2 items
    python mckesson_order.py --days 7           # only items sold in the last 7 days
    python mckesson_order.py                    # full run

Login: the first run opens the PharmaClik/Okta login page — sign in
manually in the browser window, then press Enter in this console. The
session is saved in .mckesson_profile\\ so later runs skip the login.
No credentials are stored or typed by this script.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Django bootstrap (same DB the app uses; .env is loaded by settings) ---
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "inventory.settings")
import django  # noqa: E402

django.setup()

from app.mckesson import collect_order_items  # noqa: E402

# ---------------------------------------------------------------------------
# PharmaClik page config — THE ONLY PART THAT SHOULD NEED TUNING.
#
# Each entry is a list of candidate CSS selectors tried in order; the first
# one that exists on the page wins. If the site changes (or my initial
# guesses are wrong on the first run), fix them here.
# ---------------------------------------------------------------------------
PORTAL_URL = "https://clients.mckesson.ca/index.html"
LOGIN_HOSTS = ("pharmaclik-login.mckesson.ca", "okta")

SELECTORS = {
    # Global product search input (placeholder mentions "GTIN, Home Health Care, etc.")
    "search_input": [
        "input[placeholder*='GTIN' i]",
        "input[placeholder*='Search' i]",
        "input[type='search']",
        "input[name*='search' i]",
    ],
    # The per-row cart button ("Quick Add") at the far right of a search
    # result row — confirmed from a real page snapshot:
    #   <a class="localTP jqsAddItem" data-action="/ordering?action=AddItem">
    #     <img title="Quick Add" src="/images/icons/cart_product_details.svg">
    "row_cart_button": [
        "a.jqsAddItem",
        "a[data-action*='AddItem']",
        "tr td:last-child a:has(img[alt*='Add' i])",
        "table:has-text('Item #') tr td:last-child a",
    ],
    # The "Item Order Detail" popup — PharmaClik loads dialogs into #modalPH
    "order_detail_modal": [
        "#modalPH:has-text('Item Order Detail')",
        "[class*='modal' i]:has-text('Item Order Detail')",
        "[role='dialog']:has-text('Item Order Detail')",
        "div:has-text('Item Order Detail')",
    ],
    # "Qty Ord:" input inside the popup
    "qty_input": [
        "input[name*='qty' i]",
        "input[id*='qty' i]",
        "input[type='number']",
        "input[type='text']",
    ],
    # "Add item" button inside the popup (PharmaClik styles <span> as buttons)
    "add_button": [
        "button:has-text('Add item')",
        "button:has-text('Add Item')",
        "span.btn:has-text('Add item')",
        "span.btn:has-text('Add Item')",
    ],
    # A red circle-X on the result row meaning the product is unavailable /
    # out of stock. Best-guess (title/alt/class/src); tune from a live snapshot.
    "unavailable_marker": [
        "img[title*='navailable' i]",
        "img[title*='not available' i]",
        "img[title*='out of stock' i]",
        "img[title*='discontinued' i]",
        "img[alt*='navailable' i]",
        "img[src*='unavailable' i]",
        "img[src*='notavailable' i]",
        "[title*='navailable' i]",
        "[title*='out of stock' i]",
        "[class*='unavailable' i]",
    ],
    # Something that only exists when logged in (used to detect login state)
    "logged_in_marker": [
        "[class*='cart' i]",
        "[class*='account' i]",
        "nav",
    ],
}

# Seconds to wait between items so we behave like a human, not a scraper.
THROTTLE_SECONDS = 0.5
PROFILE_DIR = BASE_DIR / ".mckesson_profile"
REPORT_PATH = BASE_DIR / "mckesson_order_report.csv"

# How long to wait for the user to act in the browser (login / manual order
# creation) when running in --no-input mode.
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
    """Return the first visible locator among candidate selectors, else None.

    Candidates are polled in ROUNDS every 200 ms until the deadline — a
    non-matching first candidate can't burn the whole timeout the way a
    sequential wait_for() per candidate would.
    """
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


def on_login_page(page):
    return any(host in page.url for host in LOGIN_HOSTS)


def ensure_logged_in(page, status, no_input=False):
    page.goto(PORTAL_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)  # let SPA redirect settle
    if on_login_page(page):
        if no_input:
            status.update(state="login",
                          message="Log in to McKesson in the browser window that just opened")
            print(">>> Waiting for PharmaClik login in the browser window...")
            deadline = time.time() + USER_ACTION_TIMEOUT_S
            while on_login_page(page):
                if time.time() > deadline:
                    raise RuntimeError("Timed out waiting for login (5 minutes).")
                page.wait_for_timeout(1000)
            page.wait_for_timeout(3000)
        else:
            print("\n>>> PharmaClik login required.")
            print(">>> Sign in (and complete any 2FA) in the browser window,")
            input(">>> then come back here and press Enter to continue... ")
            # Give the SSO redirect time to land back on the portal
            for _ in range(30):
                if not on_login_page(page):
                    break
                page.wait_for_timeout(1000)
            if on_login_page(page):
                raise RuntimeError("Still on the login page — aborting.")
            page.wait_for_timeout(3000)
    print(f"Logged in — portal at {page.url}")


def click_control(page, text, timeout_ms=5000):
    """Click a visible button/link/element by its visible text (case-insensitive)."""
    deadline = time.time() + timeout_ms / 1000
    pat = re.compile(re.escape(text), re.I)
    while True:
        for get in (
            lambda: page.get_by_role("button", name=pat).first,
            lambda: page.get_by_role("link", name=pat).first,
            lambda: page.get_by_text(pat).last,
        ):
            try:
                loc = get()
                if loc.is_visible():
                    loc.click()
                    return True
            except Exception:
                continue
        if time.time() >= deadline:
            return False
        page.wait_for_timeout(300)


def click_create_order_button(page, timeout_ms=15000):
    """Click the blue 'Create Order' box in the 'Select an Order' dialog.

    Confirmed from a real page snapshot — it is a styled <span>, NOT a
    button or link (which is why role/button matching kept missing it):

        PO: <input type="text" id="jqsPONum">
        <span class="btn btn-default" id="jqsNewOrder"
              data-action="/ordering?action=orderSelectorStartNew">Create Order</span>

    The top-bar dropdown also contains a 'Create Order' MENU ITEM
    (a.jqsCrtOrderLink) that merely OPENS this dialog — if the dialog
    isn't up yet, click that first, then press #jqsNewOrder.
    """
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        # The blue box itself
        for sel in ("#jqsNewOrder", "span.btn:has-text('Create Order')"):
            try:
                el = page.locator(sel).first
                if el.is_visible():
                    el.click()
                    return True
            except Exception:
                continue
        # Dialog not open yet — click the dropdown menu item that opens it
        try:
            menu = page.locator("a.jqsCrtOrderLink").first
            if menu.is_visible():
                menu.click()
        except Exception:
            pass
        page.wait_for_timeout(400)
    return False


def dismiss_modal(page):
    """Close whatever PharmaClik dialog is open (jQuery-UI dialog / #modalPH)."""
    for sel in (
        ".ui-dialog-titlebar-close",
        "#modalPH .ui-dialog-titlebar-close",
        "button[title='Close']",
        ".modal .close",
        "span.btn:has-text('Cancel')",
        "button:has-text('Cancel')",
    ):
        try:
            el = page.locator(sel).first
            if el.is_visible():
                el.click()
                page.wait_for_timeout(300)
                return
        except Exception:
            continue
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass


def dump_debug(page, tag):
    """Save the page HTML so failing selectors can be fixed from the real DOM."""
    path = BASE_DIR / f"mckesson_debug_{tag}.html"
    try:
        path.write_text(page.content(), encoding="utf-8")
        print(f"    (page snapshot saved to {path.name} — send this file to Claude to fix the selectors)")
    except Exception:
        pass


def order_is_active(page):
    """True when the top bar shows 'Current Order: ...' (an order is open)."""
    try:
        return page.get_by_text(re.compile("Current Order", re.I)).first.is_visible()
    except Exception:
        return False


def start_new_order(page, status, no_input=False):
    """Create a fresh order via the 'Select an Order' dialog.

    The top-bar button reads 'Select Order' when no order is active, or
    'Current Order: ...' when one is. Clicking it opens the 'Select an
    Order' dialog, which has a 'Create Order' section (with optional PO
    box) and a 'Create Order' button. PO is left blank.
    """
    opened = (click_control(page, "Select Order", timeout_ms=8000)
              or click_control(page, "Current Order", timeout_ms=4000))
    if opened:
        page.wait_for_timeout(1500)
        if click_create_order_button(page):
            # Verify it worked: the top-bar button flips to "Current Order: ..."
            for _ in range(10):
                page.wait_for_timeout(1000)
                if order_is_active(page):
                    print("Created a new order.")
                    return
    dump_debug(page, "create_order")
    if no_input:
        status.update(state="waiting_user",
                      message="Couldn't create the order automatically — in the McKesson "
                              "window click 'Select Order' then 'Create Order'")
        print(">>> Waiting for the order to be created in the browser...")
        deadline = time.time() + USER_ACTION_TIMEOUT_S
        while time.time() < deadline:
            page.wait_for_timeout(1500)
            if order_is_active(page):
                return
        raise RuntimeError("Timed out waiting for an order to be created (5 minutes).")
    print("\n>>> Couldn't create the order automatically.")
    print(">>> In the browser: click 'Select Order' -> 'Create Order'")
    print(">>> (or select the open order you want the items added to),")
    input(">>> then press Enter here to start adding products... ")


def add_item_to_cart(page, item):
    """Search one barcode and add it to the current order. Returns (ok, reason).

    PharmaClik flow (confirmed by screenshots):
      search GTIN -> results table -> click the row's cart button ->
      "Item Order Detail" popup -> fill "Qty Ord" -> click "Add item".
    """
    search = first_visible(page, SELECTORS["search_input"], timeout_ms=10000)
    if search is None:
        return False, "search box not found (adjust SELECTORS['search_input'])"

    search.click()
    search.fill("")
    search.fill(item["barcode"])
    search.press("Enter")
    # The search is a server round-trip; wait for the new page/DOM so we
    # never read the PREVIOUS search's results counter.
    try:
        page.wait_for_load_state("domcontentloaded", timeout=10000)
    except Exception:
        pass

    # Race two outcomes so a no-results search returns FAST instead of
    # waiting out a 10 s timeout: either the "Search results - N" counter
    # appears, OR PharmaClik shows a no-results message. Poll both every
    # 300 ms up to 8 s.
    n_results = None
    counter = page.locator(r"text=/Search results\s*-\s*\d+/").first
    no_results = page.locator(
        r"text=/no\s+(results|record|product|match|item)/i").first
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            if no_results.is_visible():
                return False, "no results — barcode not in McKesson catalog"
        except Exception:
            pass
        try:
            if counter.is_visible():
                m = re.search(r"Search results\s*-\s*(\d+)", counter.inner_text())
                if m:
                    n_results = int(m.group(1))
                    break
        except Exception:
            pass
        page.wait_for_timeout(300)
    if n_results == 0:
        return False, "no results — barcode not in McKesson catalog"
    if n_results is not None and n_results > 1:
        return False, f"ambiguous: {n_results} results"

    # A single result: if the row is flagged unavailable (red circle-X = out
    # of stock), skip it and make that clear on the review.
    if first_visible(page, SELECTORS["unavailable_marker"], timeout_ms=0) is not None:
        return False, "unavailable — out of stock at McKesson"

    # Collect only VISIBLE candidates — PharmaClik keeps hidden leftover
    # markup (e.g. a calendar widget) that raw selectors can match. Poll in
    # rounds until the row's cart button renders (no fixed pause).
    cart_buttons = []
    deadline = time.time() + 4
    while not cart_buttons:
        for sel in SELECTORS["row_cart_button"]:
            loc = page.locator(sel)
            for i in range(min(loc.count(), 20)):
                el = loc.nth(i)
                try:
                    if el.is_visible():
                        cart_buttons.append(el)
                except Exception:
                    continue
            if cart_buttons:
                break
        if not cart_buttons:
            if time.time() >= deadline:
                break
            page.wait_for_timeout(200)
    if not cart_buttons:
        if n_results is None:
            return False, "no search results"
        dump_debug(page, "search_results")
        return False, "result row's cart button not found"
    if len(cart_buttons) > 1 and n_results is None:
        return False, f"ambiguous: {len(cart_buttons)} results"

    # A single result row: the cart button sits at the far right, so if the
    # selector matched several controls in the row, take the last one.
    cart_buttons[-1].click(timeout=5000)

    # Clicking the cart button opens EITHER the "Item Order Detail" popup, OR
    # an "Item already included in orders listed below" dialog when the item
    # is already in an open order. Race both so we never wait out a timeout.
    already_msg = page.locator(
        r"text=/already included in orders/i").first
    modal = None
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            if already_msg.is_visible():
                dismiss_modal(page)
                return False, "already in an existing order — left as-is"
        except Exception:
            pass
        modal = first_visible(page, SELECTORS["order_detail_modal"], timeout_ms=0)
        if modal is not None:
            break
        page.wait_for_timeout(200)
    if modal is None:
        dump_debug(page, "order_detail")
        return False, "Item Order Detail popup did not open"

    # Respect the minimum sell quantity if the popup shows one (e.g. "MSQ:6")
    qty = item["quantity"]
    msq_note = ""
    m = re.search(r"MSQ:\s*(\d+)", modal.inner_text())
    if m and int(m.group(1)) > qty:
        qty = int(m.group(1))
        msq_note = f" (raised to MSQ {qty})"

    qty_input = first_visible(modal, SELECTORS["qty_input"], timeout_ms=5000)
    if qty_input is None:
        page.keyboard.press("Escape")
        return False, "Qty Ord field not found in popup"
    qty_input.click()
    qty_input.fill(str(qty))

    add = first_visible(modal, SELECTORS["add_button"], timeout_ms=2000)
    if add is not None:
        add.click()
    elif not click_control(page, "Add item", timeout_ms=3000):
        dump_debug(page, "order_detail")
        page.keyboard.press("Escape")
        return False, "'Add item' button not found in popup"
    # Wait for the popup to actually close (= the add registered) rather
    # than a fixed pause.
    try:
        page.locator("#modalPH:has-text('Item Order Detail')").first.wait_for(
            state="hidden", timeout=5000)
    except Exception:
        page.wait_for_timeout(800)
    return True, f"added x{qty}{msq_note}"


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
        status.update(state="done", message="Dry run — nothing sent to McKesson")
        return
    if not items:
        print("Nothing to order.")
        status.update(state="done", message="Nothing to order after filtering")
        return

    from playwright.sync_api import sync_playwright

    results = [{"status": "skipped", "reason": sk["reason"], **{k: sk[k] for k in ("name", "barcode", "quantity")}}
               for sk in pre_skipped]

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            args=["--start-maximized"],
            no_viewport=True,
        )
        ctx.set_default_timeout(10000)  # fail fast instead of 30 s hangs

        # If the user closes the Chrome window at ANY point, stop the whole
        # process — no orphaned background run. `finishing` guards our own
        # intentional close at the end so it doesn't look like an abort.
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
        status.update(state="running", message="Creating a new order")
        start_new_order(page, status, no_input=args.no_input)

        for i, item in enumerate(items, 1):
            status.update(state="running", current=i,
                          message=f"{item['name']} x{item['quantity']}")
            print(f"[{i}/{len(items)}] {item['name']} x{item['quantity']} ... ", end="", flush=True)
            try:
                ok, reason = add_item_to_cart(page, item)
            except Exception as e:
                msg = str(e)
                if "closed" in msg.lower():
                    raise RuntimeError("The McKesson browser window was closed before the run finished.")
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
                          message=f"{added} added — review and submit in the McKesson window, "
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
    ap = argparse.ArgumentParser(description="Fill the PharmaClik cart from Recently Purchased products.")
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
