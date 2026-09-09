"""PDF compatibility and complete-content checks without database access."""

import base64
import re
import zlib
from datetime import date, timedelta
from decimal import Decimal
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase

from app.management.commands.send_daily_report import Command
from app.reporting import build_daily_report_pdf


def pdf_text(payload):
    """Read ReportLab's text streams using only standard-library decoders."""
    text = []
    for stream in re.findall(rb'stream\r?\n(.*?)endstream', payload, re.DOTALL):
        decoded = zlib.decompress(base64.a85decode(stream.strip(), adobe=True))
        for literal in re.findall(rb'\(((?:\\.|[^\\)])*)\)\s*Tj', decoded):
            literal = re.sub(
                rb'\\([0-7]{1,3})', lambda match: bytes([int(match[1], 8)]), literal,
            )
            literal = re.sub(rb'\\([\\()])', rb'\1', literal)
            text.append(literal.decode('cp1252'))
    return '\n'.join(text)


def legacy_digest():
    return {
        'day': date(2026, 9, 7),
        'exclude_snacks': False,
        'sales': {'revenue_today': Decimal('30.00'), 'orders_today': 2, 'units_sold': 3},
        'stock_health': {
            'out_of_stock_count': 0, 'low_stock_count': 0,
            'expiring_soon_count': 0, 'total_products': 1,
        },
        'inventory': {'total_retail': Decimal('90.00'), 'gross_margin_pct': Decimal('60.0')},
        'top_movers': [],
        'low_stock': {'count': 0, 'items': []},
        'out_of_stock': {'count': 0, 'items': []},
        'expiring_week': {'count': 0, 'items': []},
        'dead_stock': {'count': 1, 'items': [{
            'name': 'Old stock', 'quantity_in_stock': 9,
            'capital_tied': Decimal('90.00'), 'days_since_sale': 'Never',
        }]},
        'corrections': {
            'correction_count': 0, 'expired_count': 0,
            'corrections': [], 'expired_today': [],
        },
    }


class DailyReportPDFTests(SimpleTestCase):
    def test_legacy_scheduled_digest_still_renders_without_enhanced_keys(self):
        payload = build_daily_report_pdf(legacy_digest())
        self.assertTrue(payload.startswith(b'%PDF-'))
        self.assertTrue(payload.rstrip().endswith(b'%%EOF'))
        content = pdf_text(payload)
        self.assertIn('Daily End-of-Day Report', content)
        self.assertIn('Sales - Sep 07, 2026', content)
        self.assertIn('Net revenue: $30.00', content)
        self.assertIn('Retail value $90.00', content)
        self.assertNotIn('tied', content)
        self.assertIn('No corrections or expiries logged on this date.', content)

    def test_enhanced_report_distinguishes_selected_sales_and_current_inventory(self):
        digest = legacy_digest()
        digest['inventory_day'] = date(2026, 9, 8)
        digest['exclude_snacks'] = True
        digest['sales'].update({
            'cost': Decimal('12.00'), 'profit': Decimal('18.00'),
            'margin_pct': Decimal('60.0'), 'average_order': Decimal('15.00'),
            'missing_cost_units': 2,
        })
        digest['comparisons'] = {
            'previous_day': {
                'day': date(2026, 9, 6), 'revenue': Decimal('0.00'),
                'delta': Decimal('30.00'), 'pct': None,
            },
            'previous_week': {
                'day': date(2026, 8, 31), 'revenue': Decimal('40.00'),
                'delta': Decimal('-10.00'), 'pct': Decimal('-25.0'),
            },
        }
        digest['trend'] = [{
            'day': date(2026, 9, 1) + timedelta(days=offset),
            'revenue': Decimal('30.00'), 'orders': 2, 'units': 3,
        } for offset in range(7)]
        digest['top_products'] = [{
            'name': 'Selected-day seller', 'barcode': 'DAILY-001',
            'units': 3, 'revenue': Decimal('30.00'),
        }]
        digest['categories'] = [{
            'name': 'Vitamins', 'units': 3, 'revenue': Decimal('30.00'), 'share_pct': 100,
        }]
        digest['stock_health']['expired_count'] = 1
        digest['expired_stock'] = {'count': 1, 'items': [{
            'name': 'Expired lot', 'lot_number': 'LOT-123',
            'expiry_date': date(2026, 9, 5), 'days_left': -3, 'quantity_in_stock': 4,
        }]}
        digest['activity'] = {
            'count': 12, 'checkin_count': 2, 'checkin_units': 20,
            'correction_count': 1, 'expired_count': 1, 'expired_units': 4,
        }

        content = pdf_text(build_daily_report_pdf(digest))
        for expected in (
            'Daily Management Report', 'Snacks category excluded',
            'Current inventory - Sep 08, 2026', 'Average order: $15.00',
            'Profit before missing costs: $18.00',
            'Cost snapshots are missing for 2 units.',
            'no percentage baseline', 'revenue change $-10.00 (-25.0%)',
            'Net revenue: $210.00', 'Orders: 14', 'Units: 21',
            'Selected-day seller', 'DAILY-001', 'Vitamins',
            'Expired stock on hand (1)', 'Lot LOT-123', '3 days overdue',
            'Daily activity - Sep 07, 2026', 'Check-ins: 2 (20 units)',
            'Expiry retirements: 1 (4 units)',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, content)
        self.assertNotIn('Sales today', content)
        self.assertNotIn('immutable', content)

    def test_long_names_and_notes_wrap_without_truncation_across_pages(self):
        digest = legacy_digest()
        long_name = 'Long descriptive product ' * 14 + 'NAME-TAIL'
        long_note = 'Detailed stock correction note ' * 18 + 'NOTE-TAIL'
        digest['corrections']['correction_count'] = 35
        digest['corrections']['corrections'] = [{
            'name': f'{long_name} {index}', 'time': '14:30', 'action': 'Correction',
            'qty': 1, 'user': 'Report user', 'note': long_note,
        } for index in range(35)]
        payload = build_daily_report_pdf(digest)
        content = pdf_text(payload)
        self.assertEqual(content.count('NAME-TAIL'), 35)
        self.assertEqual(content.count('NOTE-TAIL'), 35)
        self.assertIn('(continued)', content)
        self.assertIn('Page 2', content)
        self.assertRegex(payload, rb'/Count [2-9][0-9]*\b')

    def test_long_unbroken_identifiers_and_limited_lists_remain_complete(self):
        digest = legacy_digest()
        identifier = 'ABCDEFGH' * 35 + 'IDENTIFIER-TAIL'
        digest['low_stock'] = {'count': 8, 'items': [{
            'name': identifier, 'quantity_in_stock': 1, 'threshold': 3,
        }]}
        content = pdf_text(build_daily_report_pdf(digest))
        self.assertIn(identifier, content.replace('\n', ''))
        self.assertIn('Showing 1 of 8; 7 more are available in the application.', content)


class DailyReportCommandTests(SimpleTestCase):
    def test_dry_run_uses_enhanced_digest_for_archive_without_sending_email(self):
        digest = legacy_digest()
        digest['inventory_day'] = date(2026, 9, 8)
        output = StringIO()
        with (
            patch('app.management.commands.send_daily_report.build_daily_report', return_value=digest) as builder,
            patch('app.management.commands.send_daily_report.reporting.archive_daily_report') as archive,
            patch('app.management.commands.send_daily_report.EmailMultiAlternatives') as email,
        ):
            call_command(
                'send_daily_report', date='2026-09-07', dry_run=True,
                to=['reports@example.test'], stdout=output,
            )

        builder.assert_called_once_with(date(2026, 9, 7))
        archive.assert_called_once_with(digest=digest)
        email.assert_not_called()
        text = output.getvalue()
        html = Command()._html(digest)
        for body in (text, html):
            self.assertIn('Sales on Sep 07, 2026', body)
            self.assertIn('Current inventory as of Sep 08, 2026', body)
            self.assertIn('Stock valuation margin: 60.0%', body)
            self.assertNotIn("Today's corrections", body)
            self.assertNotIn('Gross margin:', body)
        self.assertIn('[dry-run] No email sent.', text)
