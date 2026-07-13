# Google Sheet → Ordering Sheet — One-Time Setup

The app **pulls** entries from a Google Spreadsheet: staff add items from any
phone/browser (via a Google Form or by typing rows into the sheet), and they
appear on the app's Ordering Sheet when someone clicks the **⟳ Pull from
Google Sheet** button on the Ordering Sheet page (or the pull-out). There is
no automatic schedule — pulls happen only when the button is pressed. Each
row is imported once, and a row is skipped if a matching entry (same
name + patient) already exists in the app, so re-pulling never duplicates.

**Pull-only:** the app never rewrites or deletes anything in the sheet. Its
only write-back is an **"Imported"** marker column so each row is imported
exactly once. Managing entries (status, comments, deleting) happens in the app.

Until the steps below are done, the integration is silently OFF (no errors).

---

## 1. Create the service account (~5 min)

1. Go to https://console.cloud.google.com → sign in with the pharmacy Google account.
2. Top bar → project picker → **New Project** → name it `pharmacy-sheet-sync` → Create.
3. Menu → **APIs & Services → Library** → search **Google Sheets API** → **Enable**.
4. Menu → **APIs & Services → Credentials** → **+ Create credentials → Service account**.
   - Name: `ordering-sheet-sync` → Create and continue → skip the role screens → Done.
5. Click the new service account → **Keys** tab → **Add key → Create new key → JSON** → Create.
   A `.json` file downloads.
6. Move/rename that file to exactly:
   `C:\Users\Angus\Documents\projects\pharmacy\FINAL-PHARM\google_credentials.json`
   (it is gitignored — never commit it).
7. On the service account page, copy its **email** (looks like
   `ordering-sheet-sync@pharmacy-sheet-sync.iam.gserviceaccount.com`) — needed in step 3.

## 2. Prepare the spreadsheet (existing sheet is fine)

The app reads **any tab whose header row it recognizes**: it needs a
**Drug / Product / Item Name** column plus at least one other known column.
Recognized headers (matched by keyword, case-insensitive):

| Column header contains… | Used as              | Notes                                                    |
|--------------------------|----------------------|----------------------------------------------------------|
| `Drug` / `Product` / `Item` / `Name` | **Product name (required)** |                                             |
| `Type`                   | Drug vs OTC          | values `Drug` or `OTC Product`; defaults to Drug          |
| `Patient`                | Patient name         | optional                                                  |
| `Needed`                 | Quantity needed      | optional                                                  |
| `Remaining` / `Left`     | Quantity remaining   | optional                                                  |
| `Reason`                 | Reasoning            | `Order for stock` / `Order for basket` / `Expiring` / `Order for BLISTER` |
| `Urgency` / `Priority`   | Urgency              | `High (TOMORROW PU)` / `Medium (4 days PU)` / `Low (1 week PU)` / `N/A` |
| `Side`                   | Side (OTC)           | `Left` / `Right` / `N/A`                                  |
| `Phone`                  | Phone (OTC)          | optional                                                  |
| `Initial`                | Initials             | defaults to `GS` if blank                                 |

Optional but recommended: a **Google Form** with those questions, linked to
this spreadsheet (Responses → Sheets icon), gives staff a phone-friendly
entry page with proper dropdowns — its responses tab is picked up
automatically.

## 3. Share the spreadsheet with the app

1. Open the spreadsheet → **Share** → paste the service-account **email** from step 1.7
   → role **Editor** (needed to write the Imported markers) → uncheck "Notify" → Share.

## 4. Tell the app about it

Add to `FINAL-PHARM\.env` — **either the full URL or just the ID works**:

```
GSHEET_SPREADSHEET_ID=https://docs.google.com/spreadsheets/d/15W_sIIC_tOJE7xB5cX6CXkuQSwwnXkOIt42QBYycSKs/edit
```

Then restart the server (server_control.bat → Stop → Start).

## 5. First pull

- Open the Ordering Sheet page → you'll see "📊 Google Sheet connected" and the
  **⟳ Pull from Google Sheet** button. Click it.
- Every recognizable row that isn't marked Imported comes in (so the first
  pull imports everything currently in the sheet — check the app afterwards).
- Imported rows show a green **📱 Form** pill in the app, and get a
  "✓ date" stamp in the sheet's Imported column.

Sync log: `logs\gsheet_sync.log`. Manual run: `sync_gsheet.bat` or
`python manage.py sync_gsheet`.
