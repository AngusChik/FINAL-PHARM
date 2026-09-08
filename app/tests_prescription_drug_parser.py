from django.test import SimpleTestCase

from .prescription_drug_parser import parse_prescription_drug_label


class PrescriptionDrugLabelParserTests(SimpleTestCase):
    def test_explicit_manufacturer_and_strength_are_extracted_without_spellcheck(self):
        examples = [
            ('pms-rivaroxaban 20mg', 'RIVAROXABAN', 'PMS', '20 mg'),
            ('pms-perindopril-amlodipine3.5mg/2.5mg',
             'PERINDOPRIL-AMLODIPINE', 'PMS', '3.5 mg/2.5 mg'),
            ('jamp-ciproflaxcin500mg', 'CIPROFLAXCIN', 'JAMP', '500 mg'),
            ('pms olanzapine5mg', 'OLANZAPINE', 'PMS', '5 mg'),
            ('ipratropium pms0.03%', 'IPRATROPIUM', 'PMS', '0.03%'),
            ('pms sulfasalazine ec500mg', 'SULFASALAZINE EC', 'PMS', '500 mg'),
            ('Jamp-Pantorapazole sodium40mg', 'PANTORAPAZOLE SODIUM', 'JAMP', '40 mg'),
            ('  amoxicillin (APO) 250 mg/5 mL  ', 'AMOXICILLIN', 'APO', '250 mg/5 mL'),
            ('amoxicillin teva 250MG/5ML', 'AMOXICILLIN', 'TEVA', '250 mg/5 mL'),
            ('taro-example .5000 mg', 'EXAMPLE', 'TARO', '0.5 mg'),
            ('sandoz example 5 mg/mL', 'EXAMPLE', 'SANDOZ', '5 mg/mL'),
            ('mylan-example 100mcg/0.5mL', 'EXAMPLE', 'MYLAN', '100 mcg/0.5 mL'),
        ]
        for label, name, brand, strength in examples:
            with self.subTest(label=label):
                details, reason = parse_prescription_drug_label(label)
                self.assertEqual(details, {'name': name, 'brand': brand, 'strength': strength})
                self.assertEqual(reason, '')

    def test_brand_is_never_inferred_from_a_trade_name_or_partial_word(self):
        for label in (
            'clobazam10mg', 'Grastofil300mcg/0.5mL', 'Janumet XR50/1000',
            'unknown-amoxicillin500mg', 'apomorphine5mg',
            'ipratropiumpms0.03%', 'APOAMOXICILLIN 500 mg',
        ):
            with self.subTest(label=label):
                self.assertEqual(parse_prescription_drug_label(label), (None, 'missing_brand'))

    def test_strength_is_not_inferred_from_unitless_numbers_or_package_volume(self):
        for label in (
            'apo mometasone nasal spray', 'apo levothyroxine25',
            'APO EXAMPLE 250MLS', 'APO EXAMPLE 250 mL', 'APO EXAMPLE 1 L',
            'PMS EXAMPLE 50/1000', 'JAMP EXAMPLE 100 tablets',
        ):
            with self.subTest(label=label):
                self.assertEqual(parse_prescription_drug_label(label), (None, 'missing_strength'))

    def test_options_multiple_brands_and_annotation_tails_are_rejected(self):
        for label in (
            'Mylan-APO-GINASTAN5mg', 'APO APO-AMOXICILLIN 500mg',
            'APO amoxicillin or clobazam 5mg', 'APO amoxicillin and clobazam 5mg',
            'APO AMOXICILLIN 250mg; 500mg', 'APO AMOXICILLIN 250mg, 500mg',
            'APO AMOXICILLIN 250mg + 500mg', 'APO AMOXICILLIN/CLAVULANATE 500mg',
            'APO AMOXICILLIN 500mg 100 caps', 'APO AMOXICILLIN 500mg (stock)',
            'APO AMOXICILLIN 250mg 500mg', 'APO AMOXICILLIN 250/500mg',
            'APO AMOXICILLIN 500mg\n250mg', 'AMOXICILLIN APO TABLETS 500mg',
            'APO AMOXICILLIN 0mg', 'APO EXAMPLE 100 CAPS 500mg',
            'APO EXAMPLE 250 500mg',
        ):
            with self.subTest(label=label):
                self.assertEqual(parse_prescription_drug_label(label), (None, 'ambiguous'))

    def test_missing_or_invalid_drug_name_is_rejected(self):
        for label in (None, 42, '', '  ', 'APO !!! 500mg', 'APO 123 500mg',
                      'APO EXAMPLE\x005mg', f'APO {"A" * 201} 5mg',
                      'APO 500mg', 'APO-500mg'):
            with self.subTest(label=label):
                self.assertEqual(parse_prescription_drug_label(label), (None, 'invalid_name'))

    def test_name_brand_strength_round_trip_has_stable_identity(self):
        original, _ = parse_prescription_drug_label('  APO amoxicillin  250.00 MG / 5.0 ml ')
        formatted = '{name} ({brand}) {strength}'.format(**original)
        self.assertEqual(parse_prescription_drug_label(formatted), (original, ''))
