from decimal import Decimal

from django.test import SimpleTestCase

from .prescription_drug_quantities import (
    format_quantity_totals,
    parse_requested_quantity,
)


class PrescriptionDrugQuantityParserTests(SimpleTestCase):
    def test_bare_numbers_remain_unit_unspecified_and_exact(self):
        for raw, expected in (
            ('0', '0'), ('0002', '2'), ('12', '12'), ('1.500', '1.5'),
            ('.125', '0.125'), ('  0.001  ', '0.001'),
            ('999999999', '999999999'), ('999999998.999', '999999998.999'),
        ):
            with self.subTest(raw=raw):
                quantity, unit = parse_requested_quantity(raw)
                self.assertIsInstance(quantity, Decimal)
                self.assertEqual(quantity, Decimal(expected))
                self.assertEqual(unit, 'unspecified')

    def test_explicit_units_are_canonicalized_without_package_conversion(self):
        aliases = {
            'tablets': ('tabs', 'tablet', 'tablets'),
            'capsules': ('caps', 'capsule', 'capsules'),
            'packs': ('pack', 'packs'),
            'boxes': ('box', 'boxes'),
            'bottles': ('bottle', 'bottles'),
            'vials': ('vial', 'vials'),
            'ampoules': ('ampoule', 'ampoules'),
            'units': ('unit', 'units'),
            'mL': ('mL', 'ml', 'mls'),
            'g': ('g', 'gram', 'grams'),
            'tubes': ('tube', 'tubes'),
            'inhalers': ('inhaler', 'inhalers'),
        }
        for expected, variants in aliases.items():
            for variant in variants:
                for separator in ('', ' ', '\t', '  '):
                    raw = f'2.500{separator}{variant.upper()}'
                    with self.subTest(raw=raw):
                        self.assertEqual(
                            parse_requested_quantity(raw), (Decimal('2.5'), expected),
                        )

    def test_ambiguous_or_unsupported_text_is_not_guessed(self):
        for raw in (
            None, 1, 1.5, Decimal('2'), '', ' ', 'sj', 'JM', 'RS',
            '-1', '+1', '-0', '2-3', '2 – 3', '1x100', '1 x 100',
            '1,000', '1,5', '1 000', '1/2', '1+2', '1e3', '1E+3',
            'NaN', 'Infinity', 'inf', '500mg', '250 mcg', '2 L',
            'about 2', '2 please', '2 packs of 100', '2 packs + 3 tablets',
            '2 tablets (100)', '2.0.0', '1.', '0.0001', '1.0000',
            '1000000000', '999999999.001', '9' * 129,
            '2\n', '\n2', '2\r\ntablets', '2\x00 tablets', '２ tablets',
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse_requested_quantity(raw), (None, ''))

    def test_different_units_and_unknown_units_cannot_be_conflated(self):
        parsed = [parse_requested_quantity(value) for value in (
            '2', '2 units', '2 packs', '2 tablets', '2 capsules', '2 mL', '2 g',
        )]
        self.assertEqual(len({unit for _, unit in parsed}), len(parsed))


class PrescriptionDrugQuantityTotalsTests(SimpleTestCase):
    def test_totals_show_each_unit_in_stable_order_with_no_rounding(self):
        self.assertEqual(
            format_quantity_totals({
                'unspecified': '37.000', 'tablets': '100.000',
                'mL': '12.125', 'packs': '2', 'g': '0.010',
            }),
            '0.01 g; 12.125 mL; 2 packs; 100 tablets; 37 (unit unspecified)',
        )

    def test_zero_is_preserved_and_empty_totals_have_no_display(self):
        self.assertEqual(format_quantity_totals({'unspecified': '0.000'}),
                         '0 (unit unspecified)')
        self.assertEqual(format_quantity_totals({}), '')

    def test_large_cumulative_totals_and_decimal_objects_are_formatted(self):
        self.assertEqual(
            format_quantity_totals({'tablets': '1999999998.002', 'packs': Decimal('2')}),
            '2 packs; 1999999998.002 tablets',
        )
