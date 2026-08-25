# Inventory data workflows

This file describes the durable records and guardrails behind inventory changes.
It is intended for operators troubleshooting a workflow and for future development.

## Permissions

| Area | Signed-in PU user | Staff admin or unlocked admin passkey |
| --- | --- | --- |
| Dashboard and inventory viewing | Use | Use |
| Product add, full edit, and removal | Passkey prompt | Use |
| Check-in and check-in inline edit | Use | Use |
| Expired stock | Use | Use |
| Purchase / sales checkout | Use | Use |
| PU no-sale checkout | Use | Use |
| Transactions and transaction details | View | View and correct |
| Labels | Use | Use |
| Delivery | Use normal workflow | Use normal workflow and destructive controls |
| Recently Purchased | View | Add, edit, or remove |
| Ordering sheet | Add and edit own pending requests | Manage full lifecycle and shared entries |
| Supplier purchase-order tracking | Passkey prompt | Use |
| Recovery | Passkey prompt | Use |
| Reports, analytics, and administrative history | Passkey prompt | Use |

An unlocked admin passkey grants the same protected workflow access as staff for
the configured session lifetime. It does not change the user's account role.

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
  `InventoryAuditIssue`. Assigning positive missing balances to `UNASSIGNED`
  requires staff access or the admin passkey and never changes product stock.
- `StoreHours` is the shared schedule for the Dashboard clock and automatic
  work. `ScheduledJobRun` records attempts, imported counts, failures, and
  retries for later troubleshooting.
- The Google Sheet pull runs one hour before closing on open days. It is
  pull-only, mutually exclusive with a manual pull, and deduplicates against
  durable Ordering Sheet records.
- Daily Report PDF snapshots older than the retention window are removed by an
  independent scheduled cleanup. Underlying transactions and stock history are
  retained.
