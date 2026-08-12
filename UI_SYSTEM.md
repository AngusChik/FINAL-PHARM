# Interface system

FINAL-PHARM uses one shared visual and interaction layer across inventory,
check-in, stock exceptions, purchasing, checkout, fulfillment, management,
login, error, and embedded pages. Existing form names, element IDs, URLs, and
page-specific JavaScript remain the source of business behavior.

## Source files

- `static/css/tokens.css` defines color, type, spacing, radius, shadow, motion,
  layout, and stacking values. Change a token when a visual decision should
  affect the whole product.
- `static/css/ui-system.css` defines the application shell and reusable visual
  patterns. It loads after legacy styles so old pages can be improved without
  rewriting their working markup.
- `static/js/ui-system.js` contains progressive interface behavior: submit
  feedback, table semantics, mobile utility shortcuts, and Escape-key closing.
- `app/templates/base.html` owns the authenticated shell, primary navigation,
  workflow navigation, mobile utilities, and shared overlays.

## Page structure

New authenticated pages should extend `base.html` and use the existing page
frame, header, action, card, table, status badge, pagination, empty-state,
drawer, modal, and toast patterns before creating page-specific equivalents.
Keep business hooks stable: do not rename fields, IDs, URLs, or JavaScript data
attributes for visual reasons.

Use these workflow groups for navigation and page relationships:

- Products: inventory, add product, product trends
- Check-in: sessions, inventory, activity
- Stock exceptions: expired, expiring, out of stock, low stock, recent stock
- Purchasing: current order, transactions, analytics, daily report
- Checkout: active checkout flow
- Fulfillment: ordering, delivery, labels, special orders
- Management: activity, active sessions, dashboard

## Responsive and accessibility rules

- Desktop keeps the compact expandable side navigation; small screens use the
  bottom navigation and mobile Search/Ordering utility bar.
- Tables stay inside horizontally scrollable card regions rather than widening
  the viewport.
- Scanner fields may keep autofocus, but scripted focus must use
  `{ preventScroll: true }` so phone users remain at the page header.
- Interactive elements need visible keyboard focus, a descriptive accessible
  label, and an adequate touch target.
- Respect reduced-motion preferences and keep print layouts free of navigation,
  overlays, and nonessential actions.

## Making a visual change

1. Prefer changing a token or shared component rule.
2. Add a page-level rule only when the page has a genuinely unique need.
3. Verify desktop and phone widths, keyboard focus, empty/loading/error states,
   drawers/modals, and one successful form submission.
4. Run `manage.py check`, compile the templates, and run the relevant tests.
5. Restart production with `production.bat update` so static files are collected
   and served through Caddy.
