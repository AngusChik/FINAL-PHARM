from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory, SimpleTestCase
from django.urls import reverse

from .views import OrderPDFView


class FakeDetailCollection(list):
    def select_related(self, *_args, **_kwargs):
        return self

    def all(self):
        return self


class OrderPDFViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.order_id = 77
        self.template_dir = Path(__file__).resolve().parent / "templates"
        self.fake_order = SimpleNamespace(
            order_id=self.order_id,
            order_date=SimpleNamespace(strftime=lambda fmt: "June 04, 2026 09:30" if "%H:%M" in fmt else "June 04, 2026"),
            submitted=True,
        )
        self.fake_order.details = FakeDetailCollection(
            [
                SimpleNamespace(
                    product=None,
                    quantity=2,
                    price=Decimal("12.99"),
                    display_name="Vitamin C",
                    display_barcode="123456789012",
                )
            ]
        )

    def test_order_pdf_requires_login(self):
        request = self.factory.get(reverse("order_pdf", args=[self.order_id]))
        request.user = AnonymousUser()

        response = OrderPDFView.as_view()(request, order_id=self.order_id)

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response.url)

    @patch("app.views.get_object_or_404")
    def test_order_pdf_download_returns_pdf_attachment(self, mock_get_object_or_404):
        mock_get_object_or_404.return_value = self.fake_order
        request = self.factory.get(reverse("order_pdf", args=[self.order_id]))
        request.user = SimpleNamespace(is_authenticated=True)

        response = OrderPDFView.as_view()(request, order_id=self.order_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("attachment;", response["Content-Disposition"])
        self.assertIn(f"order_{self.order_id}_transaction_record_", response["Content-Disposition"])
        self.assertTrue(response.content.startswith(b"%PDF"))

    def test_order_templates_reference_pdf_route(self):
        order_view_template = (self.template_dir / "order_view.html").read_text(encoding="utf-8")
        order_detail_template = (self.template_dir / "order_detail.html").read_text(encoding="utf-8")

        self.assertIn("url 'order_pdf' order.order_id", order_view_template)
        self.assertIn("url 'order_pdf' order.order_id", order_detail_template)
