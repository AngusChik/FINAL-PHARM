# Inventory data workflows

This file describes the durable records and guardrails behind inventory changes.
It is intended for operators troubleshooting a workflow and for future development.

## Permissions

| Area | Signed-in PU user | Staff admin or unlocked admin passkey |
| --- | --- | --- |
| Dashboard and inventory viewing | Use | Use |
| Product add, full edit, and removal | Use | Use |
| Check-in and check-in inline edit | Use | Use |
| Expired stock | Use | Use |
| Product Details, Out of Stock, Low Stock Alert, and Expiring Soon | View | View |
| Inventory Health run and repair | Use | Use |
| Purchase / sales checkout | Use | Use |
| PU no-sale checkout | Use | Use |
| Transactions and transaction details | View | View and correct |
| Labels | Use | Use |
| Delivery | Use normal workflow, including undo checkout | Use normal workflow and destructive controls |
| Recently Purchased | View | Add, edit, or remove |
| Ordering sheet | Add and edit own pending requests | Manage full lifecycle and shared entries |
| Supplier purchase-order tracking | Passkey prompt | Use |
| Recovery | Passkey prompt | Use |
| Reports, analytics, and administrative history | Passkey prompt | Use |

An unlocked admin passkey grants the same protected workflow access as staff for
the configured session lifetime. It does not change the user's account role.

### Shared PU workstation identities

- Staff on every regular device enter the same visible username, `PU`, and the
  same PU password.
- The backend assigns the lowest free live identity from `PU1` through `PU6`.
  The identity is tied to that browser session and appears in the navigation,
  presence indicators, login audit, page-lock messages, and Active Sessions.
- A seventh PU device is refused until a PU session logs out or its heartbeat is
  stale. Stale identities are reclaimed automatically.
- Admin sessions are separate from the six-PU pool. Signing in as an admin does
  not consume a PU identity or disconnect one of the six PU devices.
- The durable Django account remains `PU`; the numbered distinction identifies
  the live device/session rather than requiring six user-managed passwords.

## Prescription drug reference records

- `PrescriptionDrug` stores required name, brand, and strength fields. The
  Dashboard's Prescription Drugs button beside
  Supplier Orders opens a searchable catalogue for staff or passkey-unlocked users.
- The catalogue header's Add Prescription Drug button opens a manual-entry form
  for name, brand, strength, and optional pack size. Existing uppercase and
  duplicate protections apply; request totals/history stay background-derived.
- Model validation and normal saves trim and uppercase name and brand. Database
  constraints reject lowercase values from bulk writes that bypass model saves.
- Strength is free text (for example, `250 mg/5 mL`); its casing is preserved.
- The catalogue is a read-only, searchable list. Selecting a row opens the
  ID-based drug details page; links through editors and inventory preserve the
  list's search/page return context. Details include optional eight-digit DIN,
  generic name, manufacturer, dosage form, route, schedule, storage instructions,
  notes, active/inactive/discontinued status, and a manual review flag.
- `PrescriptionDrugPackage` stores multiple pack sizes, optional quantity/unit,
  normalized unique UPC, location, active status, and reorder/target levels.
  Existing `pack_size` text is preserved as a package without guessing its
  quantity or unit. Thereafter `pack_size` summarizes the first active package.
  It is never inferred from demand or strength and learning never overwrites it.
- Packages can explicitly link one-to-one to existing inventory Products.
  Details read existing stock and active lots, including archived-link status
  and discrepancies; viewing a drug never creates or balances stock. Reorder
  levels are in that Product's inventory units, without automatic conversion.
- `PrescriptionDrugSupplierItem` stores supplier item codes, purchase and
  catalogue prices per pack, order multiples, and one active preferred supplier.
  These are manually maintained purchasing references, not automatic orders or
  supplier price feeds. Supplier preference changes and price edits are audited.
- Manual master, package, and supplier writes lock their parent records, reject
  stale form versions, and append `PrescriptionDrugChange` snapshots with actor,
  timestamp, and before/after values atomically. Details paginate this history.
  Catalogue admin screens are read-only; edits use these audited application
  forms. Records are deactivated rather than deleted through the application.
- Drug identity includes name, brand, normalized strength, dosage form and route;
  a nonblank DIN is unique. Multiple matching forms are ambiguous to the learner.
  A durable catalogue revision causes existing source rows to be reconsidered
  after master edits, without adding unchanged observations to request history.
- Structure is informed by the TELUS Kroll 2025 User Guide, pages 117, 120, 134
  and 218: https://go.telushealth.com/hubfs/Pharmacy/support/kroll/Kroll_User_Guide_2025.pdf.
  This implements local drug-master management. Clinical interaction data,
  therapeutic equivalence, patient dispensing, claims, and Kroll connectivity
  are not implemented or inferred. New learned records remain unreviewed.
- `total_quantity_needed` stores lifetime requested totals grouped by explicit
  unit, together with `request_count` and `unknown_quantity_count`. Bare numbers
  retain an unspecified unit; tablets, capsules, bottles, boxes, and other units
  are not combined or converted. Blank/unrecognized quantities are counted as
  unknown, not zero. These summaries are read-only and rebuilt from source records.
- Demand totals remain separate from actual inventory and stock totals.
- While the web server runs, a local background worker checks the ordering sheet
  every 60 seconds. It catches up on existing rows and reads new or edited drug
  labels, including imported rows. It does not read patient/contact details or
  change ordering-sheet rows. No external AI service or network call is involved.
- Only explicit, complete name/brand/strength labels are added. Unrecognized
  brands, missing strength units, package-only amounts, and ambiguous alternatives
  are skipped rather than guessed. OTC rows do not teach drug records. Historical
  drug requests, including completed, cancelled, and archived rows, contribute to
  lifetime demand. Drug spelling is retained exactly apart from uppercase/spacing;
  the catalogue does not certify that a typed name is medically correct.
- `PrescriptionDrugLearningRecord` retains one latest snapshot per ordering row:
  source label, request date, raw needed/remaining quantities, ordered/received
  quantities, status, and deletion/type flags, plus parsed Qty Needed and unit.
  Changes to these fields or the parser version trigger another pass. Skipped
  drug labels still retain raw quantity history, but do not contribute to a drug's
  total until the label can be matched.
- The catalogue's Entries link shows request history and observed revisions in
  `PrescriptionDrugRequestRevision`. Repeated passes do not add duplicate revisions.
  This is observed history: edits made and reversed between background checks
  cannot be recovered. Patient/contact details and notes are never copied.
- Each request contributes its latest corrected quantity once; revisions never
  contribute to totals. Fulfillment, cancellation, and archiving preserve lifetime
  totals. Correcting a label transfers its current contribution to the correct
  drug; changing a row to OTC removes its contribution. Prior revisions remain
  visible. Purging a source row retains the recorded request and revision history.
- Qty Remaining is on-hand context and is never subtracted from Qty Needed.
  Ordered/received quantities also remain separate; their unit cannot be inferred.
- Database uniqueness prevents duplicate name/brand/strength combinations,
  ignoring case and spaces in strength. Existing catalogue records are reused,
  never overwritten. Learning a corrected source can add a new catalogue record;
  previously learned records remain available for administrator correction.
- Each bounded batch saves its catalogue and learning records atomically. A
  restart or failure leaves unfinished work eligible for retry; concurrent workers
  skip locked source rows. Unchanged rows are not repeatedly learned.
- `python manage.py learn_prescription_drugs --dry-run` previews pending counts;
  run without `--dry-run` for an immediate catch-up. The worker starts with WSGI or
  ASGI, not with migration or maintenance commands. Its local-only settings are
  `PRESCRIPTION_DRUG_LEARNING_ENABLED` (default on) and
  `PRESCRIPTION_DRUG_LEARNING_INTERVAL_SECONDS` (default 60).

## Product lots and stock totals

- `Product.quantity_in_stock` is the operational total shown throughout the app.
- Active `ProductLot.quantity_on_hand` values must sum to that total.
- Existing stock is migrated to the explicit `UNASSIGNED` lot. The migration does
  not guess historical lot numbers.
- Check-in can record a lot number and expiry date. Repeated check-ins for the same
  product, normalized lot number, and expiry date add to the same lot.
- A sale or no-sale checkout removes stock using FEFO: the earliest dated usable
  lot is consumed first, then later dated lots, then undated lots.
- Every automatic lot allocation is stored in `ProductLotMovement` and linked to
  its `StockChange`. This makes the exact source lot traceable later.
- Product add/edit screens accept multiple lot rows and reject the save if the lot
  total does not match Units in Stock.
- `python manage.py audit_inventory_integrity` checks negative quantities,
  product/lot total mismatches, normalized barcode conflicts, and other database
  invariants without changing data.

## Returns, voids, and transaction corrections

- The original `Order`, `OrderDetail`, `CheckoutOrder`, and checkout item are never
  rewritten. Each correction is a new immutable `TransactionCorrection` with one
  or more `TransactionCorrectionLine` rows.
- Correction lines link directly to their original transaction lines and to their
  resulting stock-ledger records.
- Only units recorded as physically supplied are correctable. Unfulfilled units
  cannot be returned to inventory, and the same unit cannot be corrected twice.
- Return-to-stock restores the original consumed lot where movement history is
  available. Older transactions without lot history use `UNASSIGNED` rather than
  inventing a lot.
- Quarantine, damaged, expired, and do-not-restock dispositions correct transaction
  counters without increasing usable stock.
- The financial adjustment includes the sale-time discount and tax. It is an audit
  and reporting value; this application does not send a refund to a payment system.
- Returns and voids appear in the daily corrections report while the original sale
  remains available for audit.

## Supplier orders and ordering lifecycle

- `SupplierPurchaseOrder` records supplier, confirmation number, dates, notes, and
  received progress. Lines may be copied from a saved supplier order plan.
- Updating supplier-order progress never changes inventory. Staff must use Check-in
  when stock physically arrives so quantities, lots, and the stock ledger agree.
- Ordering-sheet entries retain structured supplier, expected-date, ordered quantity,
  received quantity, and note fields.
- Status changes are validated and recorded in `OrderingSheetStatusEvent`. A request
  cannot claim full receipt until received quantity reaches ordered quantity.
- Completed and cancelled requests remain queryable instead of being deleted.

## Recovery instead of destructive deletion

- Product removal archives the product while preserving lots, movements, stock
  changes, transaction links, and counters. Operational product queries hide it.
- Sales, ordering entries, deliveries, Recently Purchased rows, and supplier orders
  use their existing soft-delete/archive fields and are available from Recovery.
- Restoring a product records a restoration ledger entry. A Recently Purchased row
  cannot be restored when another active row already exists for the same product.
- Database constraints reject negative stock, negative monetary values, invalid
  correction relationships, duplicate normalized barcodes, and duplicate active
  Recently Purchased rows even if a future code path misses a form-level check.

## Inventory integrity and scheduled operations

- Inventory Health on the Inventory page runs read-only barcode, lot-balance,
  non-negative-value, and supplier-receiving checks without reloading the page.
- Every audit and structured finding is retained in `InventoryAuditRun` and
  `InventoryAuditIssue`. Every signed-in user can assign positive missing
  balances to `UNASSIGNED`; this repair never changes product stock.
- `StoreHours` is the shared schedule for the Dashboard clock and automatic
  work. `ScheduledJobRun` records attempts, imported counts, failures, and
  retries for later troubleshooting.
- The Google Sheet pull runs one hour before closing on open days. It is
  pull-only, mutually exclusive with a manual pull, and deduplicates against
  durable Ordering Sheet records.
- Daily Report PDF snapshots older than the retention window are removed by an
  independent scheduled cleanup. Underlying transactions and stock history are
  retained.
