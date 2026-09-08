import time
from urllib.parse import urlencode
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from .mixins import PASSKEY_SESSION_KEY
from .models import PrescriptionDrug
from .prescription_drug_views import PrescriptionDrugCreateView


@override_settings(AXES_ENABLED=False)
class PrescriptionDrugCreateTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.staff = get_user_model().objects.create_user(username='drug-create-admin', is_staff=True)
        cls.pu = get_user_model().objects.create_user(username='drug-create-pu')

    def setUp(self):
        self.url = reverse('add_prescription_drug')
        self.data = {'name': '  Amoxicillin  ', 'brand': '  Apo  ', 'strength': '250 mg/5 mL', 'pack_size': '100 mL'}
        self.client.force_login(self.staff)

    def test_header_button_opens_form_with_catalogue_context(self):
        catalogue = reverse('prescription_drugs') + '?q=APO&page=1'
        response = self.client.get(catalogue)
        self.assertContains(response, 'Add Prescription Drug')
        self.assertContains(response, 'class="pd-add-drug"')
        self.assertContains(response, self.url + '?return_to=/prescription-drugs/%3Fq%3DAPO%26page%3D1')

        response = self.client.get(self.url, {'return_to': catalogue})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(response.context['form'].fields), ['name', 'brand', 'strength', 'pack_size'])
        self.assertEqual(response.context['return_to'], catalogue)
        self.assertEqual(response.context['page_return']['url'], catalogue)

    def test_create_normalizes_names_and_keeps_totals_and_history_derived(self):
        destination = reverse('prescription_drugs') + '?q=APO&page=1#catalogue'
        response = self.client.post(self.url, {
            **self.data, 'return_to': destination,
            'request_count': 99, 'total_quantity_needed': '{"tablets":"999"}',
        })

        self.assertRedirects(response, destination, fetch_redirect_response=False)
        drug = PrescriptionDrug.objects.get()
        self.assertEqual((drug.name, drug.brand, drug.strength, drug.pack_size), ('AMOXICILLIN', 'APO', '250 mg/5 mL', '100 mL'))
        self.assertEqual(drug.total_quantity_needed, {})
        self.assertEqual(drug.request_count, 0)
        self.assertEqual(drug.unknown_quantity_count, 0)
        self.assertFalse(drug.learning_records.exists())

    def test_pack_size_is_optional_and_invalid_fields_stay_in_form(self):
        for changes, error_field in (
            ({'name': '   '}, 'name'),
            ({'brand': ''}, 'brand'),
            ({'strength': ''}, 'strength'),
            ({'pack_size': 'x' * 101}, 'pack_size'),
        ):
            with self.subTest(field=error_field):
                response = self.client.post(self.url, {**self.data, **changes})
                self.assertEqual(response.status_code, 200)
                self.assertIn(error_field, response.context['form'].errors)
                self.assertEqual(response.context['form']['strength'].value(), changes.get('strength', self.data['strength']))
                self.assertFalse(PrescriptionDrug.objects.exists())

        response = self.client.post(self.url, {**self.data, 'pack_size': ''})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(PrescriptionDrug.objects.get().pack_size, '')

    def test_normalized_duplicate_is_rejected_without_overwriting_existing(self):
        original = PrescriptionDrug.objects.create(name='AMOXICILLIN', brand='APO', strength='250MG/5ML', pack_size='75 mL')
        response = self.client.post(self.url, self.data)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'already exist in the catalogue')
        self.assertNotContains(response, 'uniq_prescription_drug_identity')
        self.assertEqual(PrescriptionDrug.objects.count(), 1)
        original.refresh_from_db()
        self.assertEqual(original.pack_size, '75 mL')

    def test_duplicate_created_after_validation_has_friendly_error(self):
        save_form = PrescriptionDrugCreateView.form_valid

        def concurrent_insert(view, form):
            PrescriptionDrug.objects.create(name='AMOXICILLIN', brand='APO', strength='250MG/5ML', pack_size='75 mL')
            return save_form(view, form)

        with patch.object(PrescriptionDrugCreateView, 'form_valid', concurrent_insert):
            response = self.client.post(self.url, self.data)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'This prescription drug was just added.')
        self.assertEqual(PrescriptionDrug.objects.count(), 1)
        self.assertEqual(PrescriptionDrug.objects.get().pack_size, '75 mL')

    def test_access_and_csrf_match_catalogue_rules(self):
        anonymous = Client()
        self.assertRedirects(anonymous.get(self.url), reverse('login'), fetch_redirect_response=False)
        self.client.force_login(self.pu)
        self.assertRedirects(self.client.post(self.url, self.data), reverse('passkey_unlock') + '?' + urlencode({'next': self.url}), fetch_redirect_response=False)
        self.assertFalse(PrescriptionDrug.objects.exists())

        session = self.client.session
        session[PASSKEY_SESSION_KEY] = time.time()
        session.save()
        self.assertEqual(self.client.get(self.url).status_code, 200)
        self.assertEqual(self.client.post(self.url, self.data).status_code, 302)

        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(self.staff)
        self.assertEqual(csrf_client.post(self.url, {**self.data, 'name': 'Other'}).status_code, 403)
        self.assertEqual(PrescriptionDrug.objects.count(), 1)

    def test_external_self_and_non_catalogue_returns_fall_back(self):
        for destination in ('https://example.org/away', '//example.org/away', self.url, reverse('dashboard')):
            with self.subTest(destination=destination):
                response = self.client.get(self.url, {'return_to': destination})
                self.assertEqual(response.context['return_to'], reverse('prescription_drugs'))

        response = self.client.post(self.url, {**self.data, 'return_to': 'https://example.org/away'})
        self.assertRedirects(response, reverse('prescription_drugs'), fetch_redirect_response=False)
