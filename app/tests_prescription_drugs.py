from django.contrib import admin
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import RequestFactory, TestCase

from .models import PrescriptionDrug


class PrescriptionDrugTests(TestCase):
    def test_create_normalizes_names_and_preserves_strength(self):
        drug = PrescriptionDrug.objects.create(
            name='  Amoxicillin  ', brand='  Apo-Amoxi  ', strength='250 mg/5 mL',
        )

        drug.refresh_from_db()
        self.assertEqual(drug.name, 'AMOXICILLIN')
        self.assertEqual(drug.brand, 'APO-AMOXI')
        self.assertEqual(drug.strength, '250 mg/5 mL')

    def test_partial_edits_normalize_only_the_persisted_fields(self):
        drug = PrescriptionDrug.objects.create(
            name='Amoxicillin', brand='Apo-Amoxi', strength='250 mg/5 mL',
        )
        drug.name = ' amoxicillin trihydrate '
        drug.brand = ' teva-amoxicillin '
        drug.save(update_fields=['name', 'brand'])

        drug.refresh_from_db()
        self.assertEqual(drug.name, 'AMOXICILLIN TRIHYDRATE')
        self.assertEqual(drug.brand, 'TEVA-AMOXICILLIN')

        drug.name = 'Unsaved name change'
        drug.strength = '500 mg'
        drug.save(update_fields=['strength'])
        drug.refresh_from_db()
        self.assertEqual(drug.name, 'AMOXICILLIN TRIHYDRATE')
        self.assertEqual(drug.strength, '500 mg')

    def test_admin_is_read_only_so_edits_use_audited_workflow(self):
        model_admin = admin.site._registry[PrescriptionDrug]
        request = RequestFactory().get('/admin/')
        self.assertFalse(model_admin.has_add_permission(request))
        self.assertFalse(model_admin.has_change_permission(request))
        self.assertFalse(model_admin.has_delete_permission(request))
        self.assertIn('name', model_admin.get_readonly_fields(request))

    def test_validation_requires_all_three_fields(self):
        drug = PrescriptionDrug(name='   ', brand='   ', strength='')

        with self.assertRaises(ValidationError) as caught:
            drug.full_clean()

        self.assertEqual(set(caught.exception.message_dict), {'name', 'brand', 'strength'})

    def test_validation_checks_length_after_uppercase_expansion(self):
        drug = PrescriptionDrug(name='\u00df' * 101, brand='BRAND', strength='10 mg')

        with self.assertRaises(ValidationError) as caught:
            drug.full_clean()

        self.assertIn('name', caught.exception.message_dict)

    def test_database_rejects_lowercase_updates_that_bypass_save(self):
        drug = PrescriptionDrug.objects.create(
            name='Amoxicillin', brand='Apo-Amoxi', strength='250 mg/5 mL',
        )
        for field in ('name', 'brand'):
            with self.subTest(field=field):
                with self.assertRaises(IntegrityError), transaction.atomic():
                    PrescriptionDrug.objects.filter(pk=drug.pk).update(**{field: 'lowercase'})

    def test_database_rejects_lowercase_bulk_inserts(self):
        for field in ('name', 'brand'):
            values = {'name': 'AMOXICILLIN', 'brand': 'APO-AMOXI', 'strength': '250 mg/5 mL'}
            values[field] = 'lowercase'
            with self.subTest(field=field):
                with self.assertRaises(IntegrityError), transaction.atomic():
                    PrescriptionDrug.objects.bulk_create([PrescriptionDrug(**values)])
