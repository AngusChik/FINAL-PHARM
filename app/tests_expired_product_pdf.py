from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from .models import Category, Product, ProductLot


class RecordingCanvas:
    """Small ReportLab stand-in that records text drawn by the PDF view."""

    strings = []

    def __init__(self, buffer, pagesize=None):
        self.buffer = buffer
        type(self).strings = []

    def drawString(self, _x, _y, text):
        type(self).strings.append(str(text))

    def save(self):
        self.buffer.write(b'%PDF-1.4\n% expired product report test\n%%EOF')

    def setFont(self, *_args):
        pass

    def setFillColorRGB(self, *_args):
        pass

    def setStrokeColorRGB(self, *_args):
        pass

    def rect(self, *_args, **_kwargs):
        pass

    def line(self, *_args):
        pass

    def showPage(self):
        pass


@override_settings(AXES_ENABLED=False)
class ExpiredProductPDFTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='expired-pdf-user', password='pass1234',
        )
        self.client.force_login(self.user)
        self.category = Category.objects.create(name='PDF Expiry')
        self.today = date.today()
        self.lower = self.today - timedelta(days=1)
        self.upper = self.today + timedelta(days=1)

        self.mixed = self._product('Mixed range product', 'PDF-MIXED', 8, '10.00')
        ProductLot.objects.create(
            product=self.mixed,
            lot_number='IN-RANGE',
            expiry_date=self.lower,
            quantity_on_hand=2,
        )
        ProductLot.objects.create(
            product=self.mixed,
            lot_number='OUTSIDE-RANGE',
            expiry_date=self.upper + timedelta(days=10),
            quantity_on_hand=6,
        )

        self.boundary = self._product('Upper boundary product', 'PDF-UPPER', 3, '4.00')
        ProductLot.objects.create(
            product=self.boundary,
            lot_number='UPPER-BOUNDARY',
            expiry_date=self.upper,
            quantity_on_hand=3,
        )

        self.before = self._product('Before range product', 'PDF-BEFORE', 4, '2.00')
        ProductLot.objects.create(
            product=self.before,
            lot_number='BEFORE-RANGE',
            expiry_date=self.lower - timedelta(days=1),
            quantity_on_hand=4,
        )

        self.after = self._product('After range product', 'PDF-AFTER', 5, '3.00')
        ProductLot.objects.create(
            product=self.after,
            lot_number='AFTER-RANGE',
            expiry_date=self.upper + timedelta(days=1),
            quantity_on_hand=5,
        )

    def _product(self, name, barcode, quantity, price):
        return Product.objects.create(
            name=name,
            barcode=barcode,
            price=Decimal(price),
            quantity_in_stock=quantity,
            category=self.category,
        )

    def _custom_params(self, **overrides):
        params = {
            'date_filter': 'custom',
            'date_from': self.lower.isoformat(),
            'date_to': self.upper.isoformat(),
            'sort': 'expiry_date',
        }
        params.update(overrides)
        return params

    @patch('app.views.canvas.Canvas', RecordingCanvas)
    def test_custom_range_filters_products_lots_and_prints_effective_period(self):
        response = self.client.get(
            reverse('expired_products_pdf'), self._custom_params(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertTrue(response.content.startswith(b'%PDF'))
        self.assertIn('custom_expiry_date_range_report_', response['Content-Disposition'])

        drawn = RecordingCanvas.strings
        self.assertIn('Mixed range product', drawn)
        self.assertIn('Upper boundary product', drawn)
        self.assertNotIn('Before range product', drawn)
        self.assertNotIn('After range product', drawn)
        self.assertIn(
            'Expiry date range: '
            f'{self.lower.strftime("%b %d, %Y")} to '
            f'{self.upper.strftime("%b %d, %Y")}',
            drawn,
        )
        self.assertIn(
            '2 products  ·  5 units at risk  ·  $32.00 value at risk',
            drawn,
        )

    @patch('app.views.canvas.Canvas', RecordingCanvas)
    def test_custom_range_supports_one_sided_bounds(self):
        cases = (
            (
                {'date_from': self.lower.isoformat(), 'date_to': ''},
                f'Expiry date range: From {self.lower.strftime("%b %d, %Y")}',
            ),
            (
                {'date_from': '', 'date_to': self.upper.isoformat()},
                f'Expiry date range: Through {self.upper.strftime("%b %d, %Y")}',
            ),
        )

        for overrides, expected_label in cases:
            with self.subTest(overrides=overrides):
                response = self.client.get(
                    reverse('expired_products_pdf'),
                    self._custom_params(**overrides),
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn(expected_label, RecordingCanvas.strings)

    def test_custom_range_rejects_empty_malformed_and_reversed_dates(self):
        cases = (
            (
                {'date_from': '', 'date_to': ''},
                'Choose at least one expiry date',
            ),
            (
                {'date_from': 'not-a-date', 'date_to': ''},
                'Enter valid expiry dates',
            ),
            (
                {
                    'date_from': self.upper.isoformat(),
                    'date_to': self.lower.isoformat(),
                },
                'From date must be on or before the To date',
            ),
        )

        for overrides, expected_error in cases:
            with self.subTest(overrides=overrides):
                response = self.client.get(
                    reverse('expired_products_pdf'),
                    self._custom_params(**overrides),
                )
                self.assertEqual(response.status_code, 400)
                self.assertContains(
                    response, expected_error, status_code=400,
                )

    @patch('app.views.canvas.Canvas', RecordingCanvas)
    def test_preset_prints_resolved_range_and_ignores_stale_custom_dates(self):
        response = self.client.get(reverse('expired_products_pdf'), {
            'date_filter': '1_week',
            'date_from': 'not-a-date',
            'date_to': 'also-not-a-date',
        })

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'Expiry date range: '
            f'{self.today.strftime("%b %d, %Y")} to '
            f'{(self.today + timedelta(weeks=1)).strftime("%b %d, %Y")}',
            RecordingCanvas.strings,
        )

    def test_real_reportlab_response_is_a_pdf(self):
        response = self.client.get(
            reverse('expired_products_pdf'), self._custom_params(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertTrue(response.content.startswith(b'%PDF'))

    def test_pdf_requires_authentication(self):
        response = Client().get(reverse('expired_products_pdf'))

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response['Location'])
