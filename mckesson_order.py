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
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Django bootstrap (same DB the app uses; .env is loaded by settings) ---
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "inventory.settings")
import django  # noqa: E402

django.setup()

from app.mckesson import collect_order_items  # noqa: E402
from app.supplier_orders import DatabaseRunStatus  # noqa: E402

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
    # Top-right order selector. PharmaClik renders this as a button whose
    # label is "Select Order" before an order exists and "Current Order: ..."
    # after one has been created.
    "select_order_button": [
        "button.select-order-btn.jqsOrderInfoLink",
        "button[data-action*='OrderSelectorShow']",
        "button:has(#currentOrderLabel)",
    ],
    # The AJAX-loaded "Select an Order" dialog and its new-order controls.
    "order_selector_dialog": [
        "#orderSelector",
        "#modalPH:has-text('Select an Order')",
        "#jqsNewOrder",
    ],
    "create_order_toggle": [
        "a.toggle-orderitem-link[href='#collapseNewOrder']",
        "#newOrder a:has-text('Create Order')",
    ],
    "create_order_button": [
        "#jqsNewOrder[data-action*='orderSelectorStartNew']",
        "#jqsNewOrder",
        "span.btn:has-text('Create Order')",
    ],
    # Global product search input (placeholder mentions "GTIN, Home Health Care, etc.")
    "search_input": [
        "#searchInput",
        "input[placeholder*='GTIN' i]",
        "input[placeholder*='Search' i]",
        "input[type='search']",
        "input[name*='search' i]",
    ],
    # Main product rows only. The following collapsible detail row deliberately
    # has no line* id and must never be treated as a second search result.
    "product_rows": [
        "#productList tbody > tr[id^='line']",
    ],
    # The per-row cart button ("Quick Add") at the far right of a search
    # result row — confirmed from a real page snapshot:
    #   <a class="localTP jqsAddItem" data-action="/ordering?action=AddItem">
    #     <img title="Quick Add" src="/images/icons/cart_product_details.svg">
    "row_cart_button": [
        "a.jqsAddItem:not([data-suggestion='true'])",
        "a[data-action*='AddItem']:not([data-suggestion='true'])",
        "tr td:last-child a:has(img[alt*='Add' i])",
        "table:has-text('Item #') tr td:last-child a",
    ],
    # When the product is already present in an older open order, PharmaClik
    # offers this exact link to add it to the one current order selected at
    # startup. Never click one of the old .jqsOrderRow order rows.
    "add_to_current_order": [
        "#jqsAdd2CurrentOrderLink",
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

# How long to wait for the user to act in the browser (login / manual order
# creation) when running in --no-input mode.
USER_ACTION_TIMEOUT_S = 300


@dataclass
class OrderTarget:
    """The one new McKesson order selected at startup.

    PharmaClik labels a freshly-created empty order ``Unsaved`` and assigns a
    numeric transaction only after the first confirmed add.  ``candidate`` may
    observe that number while the add dialog is open, but ``token`` is not
    promoted until the add itself is positively confirmed.
    """

    token: str = ""
    previous_order_id: str = ""
    candidate_token: str = ""
    add_verified: bool = False


class SupplierRunCancelled(RuntimeError):
    pass


def heartbeat_or_cancel(status):
    """Refresh the worker lease during long user/portal waits."""
    control = status.control()
    if (
        control.get("cancel_requested") is True
        or control.get("lease_active", True) is False
    ):
        raise SupplierRunCancelled("Supplier ordering was cancelled.")
    if status.update() is False:
        raise SupplierRunCancelled("Supplier ordering lease was released.")


def control_gate(status, control_file, current):
    """Honor pause/cancel from the web app between items. Blocks while paused
    (setting status to 'paused'); returns 'cancel' if cancellation was
    requested, else 'continue'."""
    announced = False
    while True:
        ctrl = status.control()
        if ctrl.get("cancel_requested") or ctrl.get("lease_active", True) is False:
            return "cancel"
        if ctrl.get("pause_requested"):
            if not announced:
                alive = status.update(
                    state="paused", current=current,
                    message="Paused — resume from the web app",
                )
                announced = True
            else:
                alive = status.update()
            if alive is False:
                return "cancel"
            time.sleep(1)
            continue
        return "continue"


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
            except Exception as exc:
                if is_closed_error(exc):
                    raise RuntimeError(
                        "The McKesson browser page was closed before the run finished."
                    ) from exc
                continue
        if time.time() >= deadline:
            return None
        time.sleep(0.2)


def is_closed_error(exc):
    """Return True for Playwright errors caused by a closed page/browser."""
    message = f"{type(exc).__name__}: {exc}".casefold()
    return any(token in message for token in (
        "targetclosederror",
        "target page, context or browser has been closed",
        "page has been closed",
        "browser has been closed",
        "target closed",
    ))


def page_is_open(page):
    """Check browser-page liveness without changing portal state."""
    try:
        closed = page.is_closed()
        if isinstance(closed, bool):
            return not closed
    except Exception as exc:
        if is_closed_error(exc):
            return False
    try:
        page.evaluate("1")
        return True
    except Exception:
        return False


def require_page_open(page):
    if not page_is_open(page):
        raise RuntimeError(
            "The McKesson browser page was closed before the run finished."
        )


def modal_is_visible(page):
    """Return whether PharmaClik's shared AJAX dialog is currently open."""
    require_page_open(page)
    try:
        return page.locator("#modalPH").first.is_visible()
    except Exception as exc:
        if is_closed_error(exc):
            require_page_open(page)
        raise RuntimeError("Could not verify McKesson dialog state.") from exc


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
                heartbeat_or_cancel(status)
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


def open_order_selector(page, timeout_ms=15000):
    """Open PharmaClik's Select an Order dialog from the top toolbar."""
    control = first_visible(page, SELECTORS["select_order_button"], timeout_ms=timeout_ms)
    if control is None:
        # Retain a text fallback in case McKesson changes only the toolbar CSS.
        if not (
            click_control(page, "Select Order", timeout_ms=3000)
            or click_control(page, "Current Order", timeout_ms=3000)
        ):
            return False
    else:
        control.click(timeout=5000)

    return first_visible(
        page, SELECTORS["order_selector_dialog"], timeout_ms=timeout_ms,
    ) is not None


def click_create_order_button(page, timeout_ms=15000):
    """Click the blue 'Create Order' box in the 'Select an Order' dialog.

    Confirmed from a real page snapshot — it is a styled <span>, NOT a
    button or link (which is why role/button matching kept missing it):

        PO: <input type="text" id="jqsPONum">
        <span class="btn btn-default" id="jqsNewOrder"
              data-action="/ordering?action=orderSelectorStartNew">Create Order</span>

    `open_order_selector()` opens this dialog first. This helper then expands
    the Create Order accordion when necessary and presses #jqsNewOrder.
    """
    # Usually the Create Order accordion is already open. If McKesson restores
    # it collapsed, expand that exact section before looking for the button.
    create = first_visible(page, SELECTORS["create_order_button"], timeout_ms=2500)
    if create is None:
        toggle = first_visible(page, SELECTORS["create_order_toggle"], timeout_ms=2500)
        if toggle is not None:
            toggle.click(timeout=5000)
        create = first_visible(
            page, SELECTORS["create_order_button"],
            timeout_ms=max(0, timeout_ms - 5000),
        )
    if create is None:
        return False

    create.click(timeout=5000)
    return True


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


def current_order_label(page):
    """Return the normalized text in PharmaClik's order toolbar control."""
    try:
        label = page.locator("#currentOrderLabel").first
        if label.is_visible():
            return re.sub(r"\s+", " ", label.inner_text()).strip()
    except Exception:
        pass
    return ""


def current_order_id(page):
    """Return the active PharmaClik transaction number, or an empty string."""
    match = re.fullmatch(r"Current Order:\s*(\d+)", current_order_label(page), re.I)
    return match.group(1) if match else ""


def current_order_token(page):
    """Return a numeric transaction or the portal's exact new-draft token."""
    order_id = current_order_id(page)
    if order_id:
        return order_id
    label = current_order_label(page)
    if re.fullmatch(r"Current Order:\s*Unsaved", label, re.I):
        return "unsaved"
    return ""


def order_is_active(page):
    """True only for a numeric order or the exact newly-created draft label."""
    return bool(current_order_token(page))


def wait_for_active_order(page, previous_label="", timeout_ms=30000):
    """Wait until the toolbar confirms a newly created order.

    When an older order was already active, require its transaction label to
    change. Otherwise the old "Current Order" state could be mistaken for
    confirmation that the Create Order click succeeded.
    """
    previous_match = re.fullmatch(
        r"Current Order:\s*(\d+|Unsaved)", previous_label, re.I,
    )
    previous_token = previous_match.group(1).casefold() if previous_match else ""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        token = current_order_token(page)
        if token and token != previous_token:
            return True
        page.wait_for_timeout(300)
    return False


def wait_for_order_selector_closed(page, target, timeout_ms=8000):
    """Ensure the one-time Select Order dialog is gone before Quick Add.

    PharmaClik leaves the old #orderSelector HTML inside #modalPH even after a
    successful close, so detached/content checks are incorrect. Visibility of
    #modalPH is the authoritative state. A non-empty Create response means the
    dialog stays open; that is not success and must never be hidden/retried.
    """
    modal = page.locator("#modalPH").first
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        require_page_open(page)
        try:
            if not modal.is_visible():
                return current_order_token(page) == target.token
        except Exception as exc:
            if is_closed_error(exc):
                require_page_open(page)
            return False
        page.wait_for_timeout(200)
    return False


def start_new_order(page, status, no_input=False):
    """Create a fresh order via the 'Select an Order' dialog.

    The top-bar button reads 'Select Order' when no order is active, or
    'Current Order: ...' when one is. Clicking it opens the 'Select an
    Order' dialog, which has a 'Create Order' section (with optional PO
    box) and a 'Create Order' button. PO is left blank.
    """
    previous_label = current_order_label(page)
    previous_match = re.fullmatch(
        r"Current Order:\s*(\d+)", previous_label, re.I,
    )
    previous_order_id = previous_match.group(1) if previous_match else ""
    status.update(state="running", message="Opening Select Order")
    heartbeat_or_cancel(status)
    opened = open_order_selector(page)
    create_clicked = False
    if opened:
        status.update(state="running", message="Clicking Create Order")
        heartbeat_or_cancel(status)
        create_clicked = click_create_order_button(page)
        if create_clicked and wait_for_active_order(
            page, previous_label=previous_label,
        ):
            target = OrderTarget(
                token=current_order_token(page),
                previous_order_id=previous_order_id,
            )
            if target.token and wait_for_order_selector_closed(page, target):
                print(f"Created new order target {target.token}.")
                return target
            dump_debug(page, "create_order")
            raise RuntimeError(
                "McKesson changed the order target, but Select Order did not "
                "close. "
                "Stopped before searching products; do not click Create Order again."
            )
        if create_clicked:
            dump_debug(page, "create_order")
            raise RuntimeError(
                "Create Order was clicked, but McKesson did not confirm a new "
                "transaction. Stopped to prevent a duplicate order."
            )
    dump_debug(page, "create_order")
    raise RuntimeError(
        "McKesson Select Order/Create Order could not be completed automatically. "
        "No products were searched and no existing order was selected."
    )


def _coerce_order_target(expected_order_id):
    if isinstance(expected_order_id, OrderTarget):
        return expected_order_id
    return OrderTarget(token=str(expected_order_id or ""))


def promote_order_target(page, target, add_verified=False):
    """Promote Unsaved to its new number only after a confirmed first add."""
    target = _coerce_order_target(target)
    actual = current_order_token(page)
    if target.token != "unsaved":
        if actual != target.token:
            raise RuntimeError(
                "McKesson active order changed "
                f"(expected {target.token or 'a confirmed order'}, "
                f"found {actual or 'none'})."
            )
        return target.token
    if not (add_verified or target.add_verified):
        raise RuntimeError(
            "McKesson exposed an order number before a verified add; stopped "
            "without promoting the Unsaved draft."
        )
    target.add_verified = True
    if actual == "unsaved":
        return target.token
    if actual == target.previous_order_id:
        raise RuntimeError(
            "McKesson returned to the previous order instead of the newly "
            "created draft."
        )
    if not actual.isdigit():
        raise RuntimeError("McKesson no longer shows the newly-created order.")
    if target.candidate_token and actual != target.candidate_token:
        raise RuntimeError("McKesson active order changed during the first add.")
    target.token = actual
    target.candidate_token = ""
    return target.token


def assert_active_order(page, expected_order_id, allow_candidate=False):
    """Fail closed if PharmaClik is no longer targeting the startup order."""
    require_page_open(page)
    target = _coerce_order_target(expected_order_id)
    actual = current_order_token(page)
    if target.token == "unsaved":
        if actual == "unsaved":
            return target.token
        if target.add_verified:
            return promote_order_target(page, target, add_verified=True)
        if (
            allow_candidate
            and actual.isdigit()
            and actual != target.previous_order_id
        ):
            if target.candidate_token and target.candidate_token != actual:
                raise RuntimeError("McKesson active order changed during the add.")
            target.candidate_token = actual
            return actual
        if target.candidate_token and actual == target.candidate_token:
            return actual
    elif target.token and actual == target.token:
        return actual
    raise RuntimeError(
        "McKesson active order changed "
        f"(expected {target.token or 'a confirmed order'}, "
        f"found {actual or 'none'}). Stopped before adding another item."
    )


def wait_for_expected_order(
    page, expected_order_id, timeout_ms=5000, allow_candidate=False,
):
    """Wait through #orderInfoPH refreshes, but never accept another order."""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        require_page_open(page)
        try:
            assert_active_order(
                page, expected_order_id, allow_candidate=allow_candidate,
            )
            return True
        except RuntimeError:
            actual = current_order_token(page)
            target = _coerce_order_target(expected_order_id)
            if actual and (
                actual == target.previous_order_id
                or (target.token != "unsaved" and actual != target.token)
            ):
                raise
        page.wait_for_timeout(200)
    target = _coerce_order_target(expected_order_id)
    raise RuntimeError(
        f"McKesson did not restore current order {target.token} after refresh."
    )


def submit_product_search(page, search, barcode, timeout_ms=15000):
    """Submit one barcode and return the input from that exact new document.

    Waiting for load state *after* Enter can accidentally observe the already
    loaded previous document. Binding Enter to expect_navigation prevents an
    out-of-stock result from leaking into the next item's decision.
    """
    require_page_open(page)
    barcode = str(barcode).strip()
    search.click(timeout=5000)
    search.fill("")
    search.fill(barcode)
    try:
        with page.expect_navigation(
            wait_until="domcontentloaded", timeout=timeout_ms,
        ):
            search.press("Enter")
    except Exception as exc:
        require_page_open(page)
        raise RuntimeError(
            f"McKesson search navigation did not complete for barcode {barcode}."
        ) from exc

    require_page_open(page)
    fresh_search = first_visible(
        page, SELECTORS["search_input"], timeout_ms=5000,
    )
    if fresh_search is None:
        raise RuntimeError(
            f"McKesson search page for barcode {barcode} has no search input."
        )
    try:
        submitted_value = fresh_search.input_value(timeout=5000).strip()
    except Exception as exc:
        require_page_open(page)
        raise RuntimeError(
            f"Could not verify McKesson's completed search for barcode {barcode}."
        ) from exc
    if submitted_value != barcode:
        raise RuntimeError(
            "McKesson returned a different search document "
            f"(expected {barcode}, found {submitted_value or 'blank'})."
        )
    return fresh_search


def visible_product_rows(page):
    """Return only visible rows in PharmaClik's main product-results table."""
    require_page_open(page)
    rows = []
    for selector in SELECTORS["product_rows"]:
        locator = page.locator(selector)
        try:
            count = min(locator.count(), 20)
        except Exception as exc:
            require_page_open(page)
            raise RuntimeError("Could not inspect McKesson product results.") from exc
        for index in range(count):
            row = locator.nth(index)
            try:
                if row.is_visible():
                    rows.append(row)
            except Exception as exc:
                if is_closed_error(exc):
                    require_page_open(page)
        if rows:
            break
    return rows


def normalized_product_identifier(value):
    """Normalize GTIN/UPC text while tolerating PharmaClik's U-000 prefix."""
    digits = re.sub(r"\D", "", str(value or ""))
    return digits.lstrip("0") or "0"


def verify_result_barcode(page, row, barcode):
    """Verify the single result's paired detail contains the searched GTIN."""
    expected = normalized_product_identifier(barcode)
    detail_selector = ""
    try:
        toggle = row.locator("a[aria-controls]").first
        detail_selector = (toggle.get_attribute("aria-controls") or "").strip()
    except Exception as exc:
        require_page_open(page)
        raise RuntimeError("Could not inspect the McKesson result identity.") from exc
    if not detail_selector.startswith("#multi-collapse-catalog"):
        raise RuntimeError(
            "McKesson result did not expose its paired product details for "
            "barcode verification."
        )
    try:
        detail_text = page.locator(detail_selector).first.text_content(timeout=5000) or ""
    except Exception as exc:
        require_page_open(page)
        raise RuntimeError("Could not read the McKesson result GTIN details.") from exc

    candidates = {
        normalized_product_identifier(match)
        for match in re.findall(r"(?<!\d)\d{8,}(?!\d)", detail_text)
    }
    if expected not in candidates:
        raise RuntimeError(
            "McKesson's returned product GTIN does not match the submitted "
            f"barcode {barcode}."
        )
    return True


def search_result_count(page):
    """Read PharmaClik's fresh 'Search results - N' counter when present."""
    counter = page.locator(r"text=/Search results\s*-\s*\d+/").first
    try:
        if counter.is_visible():
            match = re.search(
                r"Search results\s*-\s*(\d+)", counter.inner_text(), re.I,
            )
            if match:
                return int(match.group(1))
    except Exception as exc:
        if is_closed_error(exc):
            require_page_open(page)
    return None


def current_cart_count(page):
    """Return the toolbar cart badge, when PharmaClik exposes a numeric one."""
    badge = page.locator(
        "button.cart-btn.jqsOrderInfoLink span.position-absolute.rounded-circle"
    ).first
    try:
        if badge.is_visible():
            match = re.search(r"\d+", badge.inner_text())
            if match:
                return int(match.group(0))
    except Exception as exc:
        if is_closed_error(exc):
            require_page_open(page)
    return None


def wait_for_modal_closed(page, expected_order_id, timeout_ms=8000):
    """Wait for a portal-confirmed add (the AJAX dialog closes on success)."""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        require_page_open(page)
        assert_active_order(page, expected_order_id, allow_candidate=True)
        if not modal_is_visible(page):
            promote_order_target(page, expected_order_id, add_verified=True)
            return True
        page.wait_for_timeout(200)
    return False


def resolve_duplicate_order_dialog(
    page, item, expected_order_id, cart_count_before, status=None,
):
    """Route an older-order duplicate into the one preselected current order.

    The only allowed target is #jqsAdd2CurrentOrderLink and its text must name
    the exact transaction captured at startup. Old open-order rows are never
    selected and Select/Create is never reopened here.
    """
    modal = page.locator("#modalPH").first
    link = first_visible(
        page, SELECTORS["add_to_current_order"], timeout_ms=3000,
    )
    if link is None:
        try:
            dialog_text = re.sub(r"\s+", " ", modal.inner_text()).strip()
        except Exception as exc:
            require_page_open(page)
            raise RuntimeError("Could not inspect McKesson's duplicate-order dialog.") from exc
        if re.search(r"Add to current order\s*\(", dialog_text, re.I):
            raise RuntimeError(
                "McKesson offered Add to current order, but its verified action "
                "could not be selected."
            )
        target = _coerce_order_target(expected_order_id)
        expected_numeric = (
            target.token if target.token.isdigit() else target.candidate_token
        )
        current_order_is_listed = False
        listed_rows = modal.locator("#orderList tr.jqsOrderRow")
        try:
            for index in range(min(listed_rows.count(), 50)):
                row = listed_rows.nth(index)
                if not row.is_visible():
                    continue
                transaction_text = re.sub(
                    r"\s+", " ", row.locator("td").nth(1).inner_text(),
                ).strip()
                if expected_numeric and transaction_text == expected_numeric:
                    current_order_is_listed = True
                    break
        except Exception as exc:
            require_page_open(page)
            raise RuntimeError("Could not verify McKesson's listed open orders.") from exc
        if current_order_is_listed:
            dismiss_modal(page)
            return False, "already present in the current order — left as-is", None
        raise RuntimeError(
            "Product is listed in another McKesson order, but the portal did "
            "not offer the verified Add to current order action."
        )

    target = _coerce_order_target(expected_order_id)
    link_text = re.sub(r"\s+", " ", link.inner_text()).strip()
    match = re.search(r"Add to current order\s*\(\s*(\d+)", link_text, re.I)
    link_order_id = match.group(1) if match else ""
    expected_numeric = (
        target.token if target.token.isdigit() else target.candidate_token
    )
    if (
        target.token == "unsaved"
        and not expected_numeric
        and link_order_id
        and link_order_id != target.previous_order_id
    ):
        target.candidate_token = link_order_id
        expected_numeric = link_order_id
    if not link_order_id or link_order_id != expected_numeric:
        raise RuntimeError(
            "McKesson's Add to current order link targets the wrong transaction "
            f"({link_order_id or 'unknown'})."
        )
    if status is not None:
        heartbeat_or_cancel(status)
    wait_for_expected_order(page, expected_order_id)
    link.click(timeout=5000)

    # The AJAX response either advances to Item Order Detail (so quantity can
    # be entered) or directly completes the default x1 add and closes.
    deadline = time.time() + 10
    while time.time() < deadline:
        require_page_open(page)
        assert_active_order(page, expected_order_id, allow_candidate=True)
        detail = first_visible(
            page, SELECTORS["order_detail_modal"], timeout_ms=0,
        )
        if detail is not None:
            return None, "", detail
        if not modal_is_visible(page):
            requested_qty = max(1, int(item["quantity"]))
            if cart_count_before is None:
                raise RuntimeError(
                    "McKesson directly closed the duplicate dialog, but the "
                    "current-order cart count could not be verified."
                )
            after = current_cart_count(page)
            count_deadline = time.time() + 5
            while (
                (after is None or after <= cart_count_before)
                and time.time() < count_deadline
            ):
                page.wait_for_timeout(200)
                after = current_cart_count(page)
            if after is None or after <= cart_count_before:
                raise RuntimeError(
                    "McKesson closed the duplicate dialog without increasing "
                    "the current order cart."
                )
            if requested_qty != 1:
                raise RuntimeError(
                    "McKesson directly added the duplicate as x1 before the "
                    f"requested x{requested_qty} quantity could be entered. "
                    "Stopped for cart review."
                )
            promote_order_target(page, expected_order_id, add_verified=True)
            return True, "added x1 to the preselected current order", None
        page.wait_for_timeout(200)
    raise RuntimeError(
        "McKesson did not complete Add to current order or open Item Order Detail."
    )


def add_item_to_cart(page, item, expected_order_id=None, status=None):
    """Search one barcode and add it to the one startup order.

    Verified catalog outcomes (no result, ambiguous, unavailable, already in
    this current order) are safe skips. Navigation, selector, order-target, or
    add-confirmation failures are fatal so later products cannot be misrouted.
    """
    require_page_open(page)
    active_order_id = current_order_token(page)
    expected_order_id = expected_order_id or OrderTarget(token=active_order_id)
    assert_active_order(page, expected_order_id)
    if modal_is_visible(page):
        raise RuntimeError(
            "McKesson has an unexpected dialog open before product search."
        )

    search = first_visible(page, SELECTORS["search_input"], timeout_ms=10000)
    if search is None:
        raise RuntimeError("McKesson product search box was not found.")
    submit_product_search(page, search, item["barcode"])
    assert_active_order(page, expected_order_id)

    # Poll the fresh document only. Visible main rows are authoritative; the
    # counter/no-results copy merely lets zero-result searches return quickly.
    rows = []
    n_results = None
    explicit_no_results = False
    no_results = page.locator(
        r"text=/no\s+(results|record|product|match|item)/i"
    ).first
    deadline = time.time() + 8
    while time.time() < deadline:
        require_page_open(page)
        rows = visible_product_rows(page)
        n_results = search_result_count(page)
        if rows or n_results == 0:
            break
        try:
            if no_results.is_visible():
                n_results = 0
                explicit_no_results = True
                break
        except Exception as exc:
            if is_closed_error(exc):
                require_page_open(page)
        page.wait_for_timeout(200)

    if n_results == 0 or explicit_no_results:
        return False, "no results — barcode not in McKesson catalog"
    if n_results is not None and n_results > 1:
        return False, f"ambiguous: {n_results} results"
    if len(rows) > 1:
        return False, f"ambiguous: {len(rows)} visible results"
    if not rows:
        if n_results is None:
            raise RuntimeError(
                "McKesson did not expose a verifiable result for this search."
            )
        raise RuntimeError(
            "McKesson reported one result but did not render its product row."
        )
    row = rows[0]
    verify_result_barcode(page, row, item["barcode"])

    if first_visible(
        row, SELECTORS["unavailable_marker"], timeout_ms=0,
    ) is not None:
        return False, "unavailable — out of stock at McKesson"

    cart_button = first_visible(
        row, SELECTORS["row_cart_button"], timeout_ms=4000,
    )
    if cart_button is None:
        dump_debug(page, "search_results")
        raise RuntimeError(
            "McKesson returned an available matching row, but its verified "
            "Quick Add control was not present."
        )

    cart_count_before = current_cart_count(page)
    if status is not None:
        heartbeat_or_cancel(status)
    wait_for_expected_order(page, expected_order_id)
    cart_button.click(timeout=5000)

    # Quick Add opens either Item Order Detail or the older-order duplicate
    # dialog. Both are resolved only inside #modalPH.
    modal = None
    already_msg = page.locator(
        "#modalPH:has-text('already included in orders')"
    ).first
    deadline = time.time() + 10
    while time.time() < deadline:
        require_page_open(page)
        assert_active_order(page, expected_order_id, allow_candidate=True)
        try:
            if already_msg.is_visible():
                resolved, reason, modal = resolve_duplicate_order_dialog(
                    page, item, expected_order_id, cart_count_before,
                    status=status,
                )
                if resolved is not None:
                    return resolved, reason
                break
        except Exception as exc:
            if is_closed_error(exc):
                require_page_open(page)
            raise
        modal = first_visible(
            page, SELECTORS["order_detail_modal"], timeout_ms=0,
        )
        if modal is not None:
            break
        page.wait_for_timeout(200)
    if modal is None:
        dump_debug(page, "order_detail")
        raise RuntimeError("Item Order Detail did not open after Quick Add.")

    qty = max(1, int(item["quantity"]))
    msq_note = ""
    modal_text = modal.inner_text()
    match = re.search(r"MSQ:\s*(\d+)", modal_text)
    if match and int(match.group(1)) > qty:
        qty = int(match.group(1))
        msq_note = f" (raised to MSQ {qty})"

    qty_input = first_visible(modal, SELECTORS["qty_input"], timeout_ms=5000)
    if qty_input is None:
        dump_debug(page, "order_detail")
        raise RuntimeError("Qty Ord field was not found in Item Order Detail.")
    qty_input.click(timeout=5000)
    qty_input.fill(str(qty))

    add = first_visible(modal, SELECTORS["add_button"], timeout_ms=3000)
    if add is None:
        dump_debug(page, "order_detail")
        raise RuntimeError("Add item was not found in Item Order Detail.")
    if status is not None:
        heartbeat_or_cancel(status)
    wait_for_expected_order(page, expected_order_id)
    add.click(timeout=5000)
    if not wait_for_modal_closed(page, expected_order_id, timeout_ms=10000):
        dump_debug(page, "order_detail")
        raise RuntimeError(
            "McKesson did not confirm Add item; stopped before the next product."
        )
    return True, f"added x{qty}{msq_note}"


def run(args, status):
    if args.run_id:
        items = status.pending_items()
        pre_skipped = []
        if args.limit:
            items = items[: args.limit]
    elif args.items_file:
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

    if not args.run_id:
        items = status.ensure_items(items, pre_skipped)
        pre_skipped = []

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
        # intentional close; `in_review` means adding is done and the window is
        # open only for review, so closing it then is a normal finish.
        ctl = {"finishing": False, "in_review": False}

        def _on_close(*_):
            if ctl["finishing"]:
                return
            try:
                if ctl["in_review"]:
                    status.update(state="done", message="Browser closed — run ended")
                else:
                    print("\nBrowser closed — stopping the run.")
                    status.update(state="cancelled", message="Browser closed — run stopped")
            finally:
                # Even a transient database failure must not leave an orphaned
                # Playwright worker blocking the next Control Manager run.
                os._exit(0)

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        ctx.on("close", _on_close)
        page.on("close", _on_close)
        ensure_logged_in(page, status, no_input=args.no_input)
        status.update(state="running", message="Creating a new order")
        active_order_id = start_new_order(
            page, status, no_input=args.no_input,
        )

        cancelled = False
        for i, item in enumerate(items, 1):
            # Pause / cancel from the web app (checked between items).
            if control_gate(status, None, i - 1) == "cancel":
                cancelled = True
                print("\nCancelled from the web app — stopping.")
                break
            status.update(state="running", current=i,
                          message=f"{item['name']} x{item['quantity']}")
            print(f"[{i}/{len(items)}] {item['name']} x{item['quantity']} ... ", end="", flush=True)
            try:
                assert_active_order(page, active_order_id)
                ok, reason = add_item_to_cart(
                    page, item, expected_order_id=active_order_id,
                    status=status,
                )
            except SupplierRunCancelled:
                raise
            except Exception as e:
                msg = str(e) or type(e).__name__
                safe_message = (
                    f"Stopped safely on {item['name']}: {msg}. "
                    "This and remaining items are still pending."
                )
                print(f"STOPPED — {safe_message}")
                status.update(
                    state="error", current=i - 1,
                    message=safe_message,
                )
                raise RuntimeError(safe_message) from e
            print(reason)
            results.append({"status": "added" if ok else "skipped", "reason": reason, **item})
            status.record_result(item, ok, reason)
            time.sleep(THROTTLE_SECONDS)

        added = sum(1 for r in results if r["status"] == "added")
        stopped = " (stopped early)" if cancelled else ""
        print(f"\nDone: {added} added, {len(results) - added} skipped.")
        print("The browser stays open — review the cart and submit the order yourself.")
        if args.no_input:
            ctl["in_review"] = True  # adding is done; closing the window now is a normal finish
            status.update(state="review",
                          message=f"{added} added{stopped} — review and submit in the McKesson "
                                  "window, then close it")
            print("Close the browser window when finished.")
            # The ctx/page 'close' handler ends the process the moment the
            # window is closed. This loop just heartbeats the database record
            # and is a backstop in case the
            # close event doesn't fire.
            while True:
                if status.control().get("cancel_requested"):
                    status.update(
                        state="cancelled",
                        message="Cancelled from the Control Manager",
                    )
                    return
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
            ctl["finishing"] = True  # our own close — don't treat as an abort
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
    ap.add_argument("--run-id", type=int, default=None,
                    help="database SupplierOrderRun id created by the web app")
    ap.add_argument("--attempt", type=int, default=None,
                    help="supplier run attempt lease created by the web app")
    ap.add_argument("--no-input", action="store_true",
                    help="never prompt on the console; wait for browser actions instead")
    args = ap.parse_args()

    status = DatabaseRunStatus('mck', args.run_id, attempt=args.attempt)
    try:
        run(args, status)
    except SupplierRunCancelled:
        status.update(state="cancelled", message="Supplier ordering was cancelled")
    except Exception as e:
        status.update(state="error", message=str(e))
        raise


if __name__ == "__main__":
    main()
