"""
Generate (and optionally email) the end-of-day daily report.

Usage:
    python manage.py send_daily_report
    python manage.py send_daily_report --date 2026-06-17
    python manage.py send_daily_report --to someone@example.com --to other@example.com
    python manage.py send_daily_report --dry-run
    python manage.py send_daily_report --no-attach

Behaviour:
  * Builds the digest via app.reporting.daily_digest() and a PDF via
    app.reporting.build_daily_report_pdf().
  * Emails an HTML + plain-text summary with the PDF attached to
    settings.DAILY_REPORT_RECIPIENTS (or --to).
  * EMAIL IS SCAFFOLDED, NOT YET FUNCTIONAL: with the default console backend
    (and/or no recipients) nothing actually leaves the machine — it just prints.
    Set the EMAIL_* env vars + DAILY_REPORT_RECIPIENTS to deliver for real.

Schedule once daily via Windows Task Scheduler using daily_report.bat.
"""
from datetime import date

from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.core.management.base import BaseCommand
from django.utils.dateparse import parse_date
from django.utils.html import escape

from app import reporting


class Command(BaseCommand):
    help = "Build the end-of-day report (PDF + digest) and email it to the configured recipients."

    def add_arguments(self, parser):
        parser.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD",
                            help="Report date (default: today).")
        parser.add_argument("--to", action="append", default=None, metavar="EMAIL",
                            help="Override recipient(s). Repeat for multiple. Defaults to settings.DAILY_REPORT_RECIPIENTS.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Build and print the digest but do not send any email.")
        parser.add_argument("--no-attach", action="store_true",
                            help="Do not attach the PDF to the email.")

    def handle(self, *args, **options):
        day = parse_date(options["date"]) if options["date"] else date.today()
        if day is None:
            self.stderr.write(self.style.ERROR("Invalid --date (use YYYY-MM-DD)."))
            return

        digest = reporting.daily_digest(day)

        # Archive the snapshot (30-day retention) so it shows in "View Reports".
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
        lines = [
            f"Daily End-of-Day Report — {d['day']:%A, %B %d, %Y}",
            "",
            f"Revenue: ${float(s['revenue_today']):,.2f}   Orders: {s['orders_today']}   Units sold: {s['units_sold']}",
            f"Out of stock: {h['out_of_stock_count']}   Low stock: {h['low_stock_count']}   "
            f"Expiring <=7d: {h['expiring_soon_count']}",
            f"Inventory value: ${float(inv['total_retail']):,.2f}   Gross margin: {inv['gross_margin_pct']}%",
            "",
            "Top movers (7d):",
        ]
        lines += [f"  {m['total_qty']:>4}  x  {m['product_name']}" for m in d["top_movers"]] or ["  (none)"]
        lines += [
            "",
            f"Low stock: {d['low_stock']['count']}   Out of stock: {d['out_of_stock']['count']}",
            f"Expiring this week: {d['expiring_week']['count']}   Dead stock: {d['dead_stock']['count']}",
            f"Today's corrections: {d['corrections']['correction_count']}   "
            f"Expiries: {d['corrections']['expired_count']}",
        ]
        return "\n".join(lines)

    def _html(self, d):
        s, h, inv = d["sales"], d["stock_health"], d["inventory"]
        movers = "".join(
            f"<li>{m['total_qty']} &times; {escape(m['product_name'])}</li>" for m in d["top_movers"]
        ) or "<li>(none)</li>"
        return f"""
        <h2>Daily End-of-Day Report</h2>
        <p><strong>{d['day']:%A, %B %d, %Y}</strong></p>
        <p>Revenue: <strong>${float(s['revenue_today']):,.2f}</strong> &nbsp;|&nbsp;
           Orders: {s['orders_today']} &nbsp;|&nbsp; Units sold: {s['units_sold']}</p>
        <p>Out of stock: {h['out_of_stock_count']} &nbsp;|&nbsp; Low stock: {h['low_stock_count']}
           &nbsp;|&nbsp; Expiring &le;7d: {h['expiring_soon_count']}</p>
        <p>Inventory value: ${float(inv['total_retail']):,.2f} &nbsp;|&nbsp;
           Gross margin: {inv['gross_margin_pct']}%</p>
        <h3>Top movers (7 days)</h3><ul>{movers}</ul>
        <p>Low stock: {d['low_stock']['count']} &nbsp;|&nbsp; Out of stock: {d['out_of_stock']['count']}
           &nbsp;|&nbsp; Expiring this week: {d['expiring_week']['count']}
           &nbsp;|&nbsp; Dead stock: {d['dead_stock']['count']}</p>
        <p>Today's corrections: {d['corrections']['correction_count']} &nbsp;|&nbsp;
           Expiries: {d['corrections']['expired_count']}</p>
        <p>Full breakdown attached as PDF.</p>
        """
