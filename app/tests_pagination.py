from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

from django.core.paginator import Paginator
from django.template.loader import get_template, render_to_string
from django.test import SimpleTestCase


class BoundaryParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.controls = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        label = attrs.get('aria-label')
        if label in ('First page', 'Last page'):
            self.controls[label] = (tag, attrs)


class PaginationBoundaryTests(SimpleTestCase):
    def test_ajax_pager_boundaries_preserve_filters_and_last_page_state(self):
        cases = (
            ('partials/inv_pager.html', 'page_obj', 'page', {
                'sort_column': 'name', 'sort_direction': 'desc',
                'category_qs': '&category=2&category=4',
                'stock_filter_qs': '&stock=low', 'search_query': 'A & B',
            }, {'sort': ['name'], 'direction': ['desc'], 'category': ['2', '4'],
                'stock': ['low'], 'q': ['A & B']}),
            ('partials/rp_pager.html', 'page_obj_recent', 'page_recent', {
                'q': 'A & B', 'category_filter': '2', 'sort': 'quantity',
                'dir': 'desc', 'hide_snacks': '1',
            }, {'q': ['A & B'], 'category': ['2'], 'sort': ['quantity'],
                'dir': ['desc'], 'hide_snacks': ['1']}),
        )
        paginator = Paginator(range(21), 10)
        for template, page_key, parameter, filters, expected_filters in cases:
            for number in (1, 2, 3):
                with self.subTest(template=template, page=number):
                    html = render_to_string(template, {**filters, page_key: paginator.page(number)})
                    controls = BoundaryParser(html).controls
                    if number == 1:
                        self.assertEqual(controls, {})
                        continue
                    first_tag, first = controls['First page']
                    self.assertEqual(first_tag, 'a')
                    self.assertEqual(parse_qs(urlsplit(first['href']).query), {
                        **expected_filters, parameter: ['1'],
                    })
                    last_tag, last = controls['Last page']
                    if number == 3:
                        self.assertEqual(last_tag, 'span')
                        self.assertEqual(last['aria-disabled'], 'true')
                        self.assertNotIn('href', last)
                    else:
                        self.assertEqual(last_tag, 'a')
                        self.assertEqual(parse_qs(urlsplit(last['href']).query), {
                            **expected_filters, parameter: ['3'],
                        })
                    self.assertLess(html.index('aria-label="First page"'), html.index('aria-label="Last page"'))

    def test_single_page_ajax_lists_have_no_boundary_controls(self):
        page = Paginator(range(3), 10).page(1)
        for template in ('partials/inv_pager.html', 'partials/rp_pager.html'):
            with self.subTest(template=template):
                html = render_to_string(template, {'page_obj': page, 'page_obj_recent': page})
                self.assertEqual(BoundaryParser(html).controls, {})

    def test_all_updated_page_templates_compile(self):
        for name in (
            'activity_log.html', 'order_view.html', 'checkin_dashboard.html',
            'archive_recovery.html', 'expired_log.html', 'expiring_soon.html',
            'low_stock_trend.html', 'out_of_stock.html', 'prescription_drugs.html',
            'prescription_drug_history.html', 'prescription_drug_detail.html',
            'label_printing.html', 'home.html', 'partials/_stock_log_panel.html',
        ):
            with self.subTest(template=name):
                get_template(name)
