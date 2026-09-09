"""
Generate, archive, and optionally email the daily management report.

Usage:
    python manage.py send_daily_report
    python manage.py send_daily_report --date 2026-06-17
    python manage.py send_daily_report --to someone@example.com --to other@example.com
    python manage.py send_daily_report --dry-run
    python manage.py send_daily_report --no-attach

Behaviour:
  * Builds the same enhanced digest as the Daily Report page via
    app.daily_reporting.build_daily_report() and a PDF via
    app.reporting.build_daily_report_pdf().
  * Sales and activity use the selected date; inventory reflects current
    balances at generation time. Sales include currently active corrections.
  * Retains each changed full report in Report history, including during --dry-run.
  * Emails an HTML + plain-text summary with the PDF attached to
    settings.DAILY_REPORT_RECIPIENTS (or --to).
  * Uses the configured email backend and delivery setting. With the console
    backend or no recipients, no external email is delivered. Configure EMAIL_*
    and DAILY_REPORT_RECIPIENTS to use SMTP when email delivery is enabled.

Schedule once daily via Windows Task Scheduler using daily_report.bat.
"""
from datetime import date

from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.core.management.base import BaseCommand, CommandError
from django.utils.dateparse import parse_date
from django.utils.html import escape

from app import reporting
from app.daily_reporting import build_daily_report
from app.environment import (
    email_delivery_enabled,
    integration_disabled_message,
)


class Command(BaseCommand):
    help = "Build and archive the daily management report, then email it to configured recipients."

    def add_arguments(self, parser):
        parser.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD",
                            help="Report date (default: today).")
        parser.add_argument("--to", action="append", default=None, metavar="EMAIL",
                            help="Override recipient(s). Repeat for multiple. Defaults to settings.DAILY_REPORT_RECIPIENTS.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Build, archive, and print the report but do not send any email.")
        parser.add_argument("--no-attach", action="store_true",
                            help="Do not attach the PDF to the email.")

    def handle(self, *args, **options):
        day = parse_date(options["date"]) if options["date"] else date.today()
        if day is None:
            self.stderr.write(self.style.ERROR("Invalid --date (use YYYY-MM-DD)."))
            return

        digest = build_daily_report(day)

        # Retain the snapshot alongside earlier versions in Report history.
        try:
            reporting.archive_daily_report(digest=digest)
        except Exception as exc:  # never let archiving break the email job
            self.stderr.write(self.style.WARNING(f"Could not archive report: {exc}"))

        subject = f"Pharmacy daily report — {day:%b %d, %Y}"
        text_body = self._text(digest)

        if options["dry_run"]:
            self.stdout.write(subject)
            self.stdout.write(text_body)
            self.stdout.write(self.style.WARNING("[dry-run] No email sent."))
            return

        recipients = [r for r in (options["to"] or getattr(settings, "DAILY_REPORT_RECIPIENTS", [])) if r]
        if not recipients:
            # Email scaffold is intentionally inert until recipients/SMTP are set.
            self.stdout.write(subject)
            self.stdout.write(text_body)
            self.stdout.write(self.style.WARNING(
                "No DAILY_REPORT_RECIPIENTS configured — nothing sent. "
                "Set DAILY_REPORT_RECIPIENTS + EMAIL_* env vars to enable delivery."
            ))
            return

        if not email_delivery_enabled():
            raise CommandError(integration_disabled_message("Email delivery"))

        msg = EmailMultiAlternatives(
            subject, text_body, getattr(settings, "DEFAULT_FROM_EMAIL", None), recipients,
        )
        msg.attach_alternative(self._html(digest), "text/html")
        if not options["no_attach"]:
            msg.attach(f"daily_report_{day:%Y%m%d}.pdf",
                       reporting.build_daily_report_pdf(digest), "application/pdf")
        sent = msg.send(fail_silently=False)
        self.stdout.write(self.style.SUCCESS(
            f"Daily report sent to {', '.join(recipients)} (send()={sent})."
        ))

    # ── body builders ───────────────────────────────────────────────────────
    def _text(self, d):
        s, h, inv = d["sales"], d["stock_health"], d["inventory"]
        inventory_day = d.get('inventory_day', d['day'])
        lines = [
            f"Daily Management Report — {d['day']:%A, %B %d, %Y}",
            "",
            f"Sales on {d['day']:%b %d, %Y}",
            f"Net revenue: ${float(s['revenue_today']):,.2f}   Orders: {s['orders_today']}   Units sold: {s['units_sold']}",
            "Revenue is before tax, after discounts and currently active returns or voids.",
            "",
            f"Current inventory as of {inventory_day:%b %d, %Y}",
            f"Out of stock: {h['out_of_stock_count']}   Low stock: {h['low_stock_count']}   "
            f"Expiring <=7d: {h['expiring_soon_count']}",
            f"Inventory retail value: ${float(inv['total_retail']):,.2f}   Stock valuation margin: {inv['gross_margin_pct']}%",
            "Stock valuation margin uses current retail prices and recorded product costs; it is not realized sales margin.",
            "",
            f"Top movers (7 days ending {d['day']:%b %d, %Y}):",
        ]
        lines += [f"  {m['total_qty']:>4}  x  {m['product_name']}" for m in d["top_movers"]] or ["  (none)"]
        lines += [
            "",
            f"Low stock: {d['low_stock']['count']}   Out of stock: {d['out_of_stock']['count']}",
            f"Currently expiring within 7 days: {d['expiring_week']['count']}   Dead stock: {d['dead_stock']['count']}",
            f"Corrections on {d['day']:%b %d, %Y}: {d['corrections']['correction_count']}   "
            f"Expiries: {d['corrections']['expired_count']}",
        ]
        return "\n".join(lines)

    def _html(self, d):
        s, h, inv = d["sales"], d["stock_health"], d["inventory"]
        inventory_day = d.get('inventory_day', d['day'])
        movers = "".join(
            f"<li>{m['total_qty']} &times; {escape(m['product_name'])}</li>" for m in d["top_movers"]
        ) or "<li>(none)</li>"
        return f"""
        <h2>Daily Management Report</h2>
        <p><strong>{d['day']:%A, %B %d, %Y}</strong></p>
        <h3>Sales on {d['day']:%b %d, %Y}</h3>
        <p>Net revenue: <strong>${float(s['revenue_today']):,.2f}</strong> &nbsp;|&nbsp;
           Orders: {s['orders_today']} &nbsp;|&nbsp; Units sold: {s['units_sold']}</p>
        <p>Revenue is before tax, after discounts and currently active returns or voids.</p>
        <h3>Current inventory as of {inventory_day:%b %d, %Y}</h3>
        <p>Out of stock: {h['out_of_stock_count']} &nbsp;|&nbsp; Low stock: {h['low_stock_count']}
           &nbsp;|&nbsp; Expiring &le;7d: {h['expiring_soon_count']}</p>
        <p>Inventory retail value: ${float(inv['total_retail']):,.2f} &nbsp;|&nbsp;
           Stock valuation margin: {inv['gross_margin_pct']}%</p>
        <p>Stock valuation margin uses current retail prices and recorded product costs; it is not realized sales margin.</p>
        <h3>Top movers (7 days ending {d['day']:%b %d, %Y})</h3><ul>{movers}</ul>
        <p>Low stock: {d['low_stock']['count']} &nbsp;|&nbsp; Out of stock: {d['out_of_stock']['count']}
           &nbsp;|&nbsp; Currently expiring within 7 days: {d['expiring_week']['count']}
           &nbsp;|&nbsp; Dead stock: {d['dead_stock']['count']}</p>
        <p>Corrections on {d['day']:%b %d, %Y}: {d['corrections']['correction_count']} &nbsp;|&nbsp;
           Expiries: {d['corrections']['expired_count']}</p>
        <p>Full breakdown attached as PDF.</p>
        """
