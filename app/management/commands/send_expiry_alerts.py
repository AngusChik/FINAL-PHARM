"""
Send a daily digest email of expired and soon-to-expire products.

Usage:
    python manage.py send_expiry_alerts
    python manage.py send_expiry_alerts --window 30
    python manage.py send_expiry_alerts --to someone@example.com --to other@example.com
    python manage.py send_expiry_alerts --dry-run

Behaviour:
  * Collects products that are already EXPIRED (expiry_date < today) and products
    EXPIRING SOON (today <= expiry_date <= today + window days).
  * Builds an HTML + plain-text digest and emails it to the configured recipients.
  * Recipients default to settings.EXPIRY_ALERT_RECIPIENTS (anguscwebsite@gmail.com).
  * Window defaults to settings.EXPIRY_ALERT_WINDOW_DAYS (14 days).
  * If there is nothing to report, no email is sent (use --always-send to override).
  * Uses settings.EMAIL_BACKEND — with no SMTP credentials configured this prints
    the email to the console (safe for testing).

Schedule it to run once daily (e.g. via cron / Task Scheduler / a Cowork scheduled task).
"""
from datetime import date, timedelta

from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.core.management.base import BaseCommand
from django.db.models import F
from django.utils.html import escape

from app.models import Product


class Command(BaseCommand):
    help = "Email a digest of expired and soon-to-expire products to the alert recipients."

    def add_arguments(self, parser):
        parser.add_argument(
            "--window", type=int, default=None,
            help="Days ahead to treat as 'expiring soon' (default: settings.EXPIRY_ALERT_WINDOW_DAYS).",
        )
        parser.add_argument(
            "--to", action="append", default=None, metavar="EMAIL",
            help="Override recipient(s). Repeat for multiple. Defaults to settings.EXPIRY_ALERT_RECIPIENTS.",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Build and print the digest but do not send any email.",
        )
        parser.add_argument(
            "--always-send", action="store_true",
            help="Send the email even when there is nothing to report.",
        )

    def handle(self, *args, **options):
        window = options["window"]
        if window is None:
            window = getattr(settings, "EXPIRY_ALERT_WINDOW_DAYS", 14)

        recipients = options["to"] or getattr(settings, "EXPIRY_ALERT_RECIPIENTS", [])
        recipients = [r for r in recipients if r]
        if not recipients:
            self.stderr.write(self.style.ERROR(
                "No recipients configured. Set EXPIRY_ALERT_RECIPIENTS or pass --to."
            ))
            return

        today = date.today()
        soon_cutoff = today + timedelta(days=window)

        base = (
            Product.objects.filter(status=True)
            .exclude(expiry_date__isnull=True)
            .select_related("category")
        )
        expired = list(
            base.filter(expiry_date__lt=today).order_by("expiry_date")
        )
        expiring_soon = list(
            base.filter(expiry_date__gte=today, expiry_date__lte=soon_cutoff)
            .order_by("expiry_date")
        )

        total = len(expired) + len(expiring_soon)
        if total == 0 and not options["always_send"]:
            self.stdout.write(self.style.SUCCESS(
                "Nothing expired or expiring soon — no email sent."
            ))
            return

        subject = (
            f"Pharmacy expiry alert — {len(expired)} expired, "
            f"{len(expiring_soon)} expiring within {window} days ({today:%b %d, %Y})"
        )
        text_body = self._build_text(today, window, expired, expiring_soon)
        html_body = self._build_html(today, window, expired, expiring_soon)

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("[DRY RUN] Would send to: " + ", ".join(recipients)))
            self.stdout.write(f"Subject: {subject}\n")
            self.stdout.write(text_body)
            return

        from_email = getattr(settings, "DEFAULT_FROM_EMAIL", None)
        msg = EmailMultiAlternatives(subject, text_body, from_email, recipients)
        msg.attach_alternative(html_body, "text/html")
        sent = msg.send(fail_silently=False)

        self.stdout.write(self.style.SUCCESS(
            f"Expiry digest sent to {', '.join(recipients)} "
            f"({len(expired)} expired, {len(expiring_soon)} expiring soon; send()={sent})."
        ))

    # ── digest builders ────────────────────────────────────────────────
    def _row_text(self, p, today):
        days = (p.expiry_date - today).days
        when = f"{abs(days)}d ago" if days < 0 else (f"in {days}d" if days > 0 else "today")
        cat = p.category.name if p.category else "—"
        return (
            f"  - {p.name} | {p.barcode or '-'} | qty {p.quantity_in_stock} | "
            f"{cat} | exp {p.expiry_date:%Y-%m-%d} ({when})"
        )

    def _build_text(self, today, window, expired, expiring_soon):
        lines = [f"Expiry report for {today:%A, %B %d, %Y}", ""]
        lines.append(f"EXPIRED ({len(expired)}):")
        lines += [self._row_text(p, today) for p in expired] or ["  (none)"]
        lines.append("")
        lines.append(f"EXPIRING WITHIN {window} DAYS ({len(expiring_soon)}):")
        lines += [self._row_text(p, today) for p in expiring_soon] or ["  (none)"]
        lines.append("")
        lines.append("— Automated pharmacy inventory alert")
        return "\n".join(lines)

    def _html_section(self, title, products, today, accent):
        head = (
            f'<h3 style="margin:18px 0 8px;color:{accent};">{escape(title)} '
            f'({len(products)})</h3>'
        )
        if not products:
            return head + '<p style="color:#888;margin:0 0 8px;">None.</p>'
        rows = []
        for p in products:
            days = (p.expiry_date - today).days
            when = f"{abs(days)} days ago" if days < 0 else (f"in {days} days" if days > 0 else "today")
            cat = escape(p.category.name) if p.category else "—"
            rows.append(
                "<tr>"
                f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">{escape(p.name)}</td>'
                f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">{escape(p.barcode or "—")}</td>'
                f'<td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:right;">{p.quantity_in_stock}</td>'
                f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">{cat}</td>'
                f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">{p.expiry_date:%Y-%m-%d}</td>'
                f'<td style="padding:6px 10px;border-bottom:1px solid #eee;color:{accent};">{when}</td>'
                "</tr>"
            )
        table = (
            '<table style="border-collapse:collapse;width:100%;font-size:14px;">'
            '<thead><tr style="text-align:left;background:#f5f5f7;">'
            '<th style="padding:6px 10px;">Product</th>'
            '<th style="padding:6px 10px;">Barcode</th>'
            '<th style="padding:6px 10px;text-align:right;">Qty</th>'
            '<th style="padding:6px 10px;">Category</th>'
            '<th style="padding:6px 10px;">Expiry</th>'
            '<th style="padding:6px 10px;">When</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
        )
        return head + table

    def _build_html(self, today, window, expired, expiring_soon):
        return (
            '<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;'
            'max-width:760px;margin:0 auto;color:#222;">'
            f'<h2 style="margin:0 0 4px;">Pharmacy Expiry Report</h2>'
            f'<p style="color:#666;margin:0 0 12px;">{today:%A, %B %d, %Y}</p>'
            + self._html_section("Expired", expired, today, "#c0392b")
            + self._html_section(f"Expiring within {window} days", expiring_soon, today, "#b8860b")
            + '<p style="color:#999;font-size:12px;margin-top:18px;">'
            'Automated pharmacy inventory alert.</p></div>'
        )
