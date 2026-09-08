import time
from urllib.parse import quote

from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .mixins import PASSKEY_SESSION_KEY
from .models import (
    OrderingSheetEntry, PrescriptionDrug, PrescriptionDrugLearningRecord,
    PrescriptionDrugRequestRevision,
)


@override_settings(AXES_ENABLED=False)
class PrescriptionDrugCatalogueTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.staff = get_user_model().objects.create_user(username='catalogue-admin', is_staff=True)
        cls.pu = get_user_model().objects.create_user(username='catalogue-pu')
        cls.first = PrescriptionDrug.objects.create(name='Amoxicillin', brand='APO', strength='250 mg/5 mL')
        cls.second = PrescriptionDrug.objects.create(name='Rivaroxaban', brand='PMS', strength='20 mg')

    def test_anonymous_and_locked_users_follow_existing_access_rules(self):
        url = reverse('prescription_drugs')
        self.assertRedirects(self.client.get(url), reverse('login'), fetch_redirect_response=False)

        self.client.force_login(self.pu)
        response = self.client.get(url, {'q': 'APO'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('passkey_unlock') + '?next=%2Fprescription-drugs%2F%3Fq%3DAPO')

    def test_passkey_unlocked_user_can_browse_catalogue(self):
        self.client.force_login(self.pu)
        session = self.client.session
        session[PASSKEY_SESSION_KEY] = time.time()
        session.save()

        response = self.client.get(reverse('prescription_drugs'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'AMOXICILLIN')
        self.assertContains(response, 'RIVAROXABAN')

    def test_staff_page_shows_all_fields_and_dashboard_return(self):
        self.client.force_login(self.staff)

        response = self.client.get(reverse('prescription_drugs'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'AMOXICILLIN')
        self.assertContains(response, 'APO')
        self.assertContains(response, '250 mg/5 mL')
        self.assertEqual(response.context['page_return']['url'], reverse('dashboard'))
        self.assertEqual(list(response.context['drugs']), [self.first, self.second])

    def test_search_matches_name_brand_and_strength_case_insensitively(self):
        self.client.force_login(self.staff)
        for query in ('amox', 'apo', '250 mg'):
            with self.subTest(query=query):
                response = self.client.get(reverse('prescription_drugs'), {'q': query})
                self.assertEqual(list(response.context['drugs']), [self.first])

    def test_filtered_pagination_retains_search_and_returns_remaining_rows(self):
        self.client.force_login(self.staff)
        for index in range(51):
            PrescriptionDrug.objects.create(name=f'Catalogue {index:02d}', brand='TEVA', strength='10 mg')

        response = self.client.get(reverse('prescription_drugs'), {'q': 'TEVA'})
        self.assertEqual(len(response.context['drugs']), 50)
        self.assertContains(response, '?q=TEVA&amp;page=2')
        response = self.client.get(reverse('prescription_drugs'), {'q': 'TEVA', 'page': 2})
        self.assertEqual(len(response.context['drugs']), 1)
        self.assertEqual(response.context['query'], 'TEVA')

    def test_no_results_escape_search_text_and_offer_clear_search(self):
        self.client.force_login(self.staff)
        response = self.client.get(reverse('prescription_drugs'), {'q': '<script>missing</script>'})

        self.assertContains(response, 'No prescription drugs match your search.')
        self.assertContains(response, 'Clear search')
        self.assertNotContains(response, '<script>missing</script>')

    def test_catalogue_get_does_not_trigger_learning_or_edit_records(self):
        self.client.force_login(self.staff)
        before = list(PrescriptionDrug.objects.values())

        self.client.get(reverse('prescription_drugs'))

        self.assertEqual(list(PrescriptionDrug.objects.values()), before)

    def test_catalogue_is_a_clickable_list_with_preserved_search_context(self):
        self.client.force_login(self.staff)
        self.first.pack_size = '100 tablets'
        self.first.total_quantity_needed = {'tablets': '120', 'mL': '250.5'}
        self.first.request_count = 4
        self.first.save()
        catalogue_url = reverse('prescription_drugs') + '?q=APO&page=1'
        response = self.client.get(catalogue_url)
        self.assertContains(response, reverse('prescription_drug_detail', args=[self.first.pk])
                            + '?return_to=' + quote(catalogue_url, safe='/'))
        self.assertNotContains(response, '120 tablets')
        self.assertNotContains(response, 'name="pack_size"')
        self.assertNotContains(response, 'class="pd-pack-size-form"')
        self.assertNotContains(response, 'data-no-personalize')

    def test_catalogue_rejects_writes_and_leaves_details_unchanged(self):
        self.client.force_login(self.staff)
        before = PrescriptionDrug.objects.values().get(pk=self.first.pk)
        response = self.client.post(reverse('prescription_drugs'), {
            'drug_id': self.first.pk, 'pack_size': '30 tablets', 'request_count': 999,
        })
        self.assertEqual(response.status_code, 405)
        self.assertEqual(PrescriptionDrug.objects.values().get(pk=self.first.pk), before)


@override_settings(AXES_ENABLED=False)
class PrescriptionDrugHistoryTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.staff = get_user_model().objects.create_user(username='history-admin', is_staff=True)
        cls.pu = get_user_model().objects.create_user(username='history-pu')
        cls.drug = PrescriptionDrug.objects.create(name='Amoxicillin', brand='APO', strength='250 mg/5 mL')

    def history_url(self, **query):
        url = reverse('prescription_drug_history', args=[self.drug.pk])
        if query:
            from urllib.parse import urlencode
            url += '?' + urlencode(query)
        return url

    def make_record(self, **kwargs):
        defaults = {
            'drug': self.drug,
            'source_name': 'APO Amoxicillin 250 mg/5 mL',
            'source_snapshot': {
                'name': 'APO Amoxicillin 250 mg/5 mL',
                'quantity_needed': '2 bottles',
                'quantity_remaining': 'half bottle',
                'quantity_ordered': 2,
                'quantity_received': 0,
                'status': 'ordered',
                'is_deleted': False,
                'entry_type': 'drug',
            },
            'requested_at': timezone.now(),
        }
        defaults.update(kwargs)
        return PrescriptionDrugLearningRecord.objects.create(**defaults)

    def test_history_uses_existing_access_rules(self):
        url = self.history_url()
        self.assertRedirects(self.client.get(url), reverse('login'), fetch_redirect_response=False)
        self.client.force_login(self.pu)
        self.assertEqual(self.client.get(url).status_code, 302)
        session = self.client.session
        session[PASSKEY_SESSION_KEY] = time.time()
        session.save()
        self.assertEqual(self.client.get(url).status_code, 200)

    def test_history_shows_raw_quantities_archived_requests_and_escaped_revisions(self):
        self.client.force_login(self.staff)
        ordering_entry = OrderingSheetEntry.objects.create(
            name='APO Amoxicillin 250 mg/5 mL', initials='AB',
            patient_name='Patient Hidden Value', phone_number='555-987-1234',
        )
        record = self.make_record(entry=ordering_entry)
        record.source_snapshot['is_deleted'] = True
        record.save()
        PrescriptionDrugRequestRevision.objects.create(
            learning_record=record, drug=self.drug,
            snapshot={
                **record.source_snapshot,
                'quantity_needed': '<script>old quantity</script>',
                'quantity_remaining': 'less than one bottle',
            },
        )

        response = self.client.get(self.history_url())

        self.assertContains(response, '2 bottles')
        self.assertContains(response, 'half bottle')
        self.assertContains(response, '>0</td>', html=False)
        self.assertContains(response, 'Archived')
        self.assertContains(response, 'Recorded changes (1)')
        self.assertContains(response, '&lt;script&gt;old quantity&lt;/script&gt;')
        self.assertContains(response, 'less than one bottle')
        self.assertNotContains(response, '<script>old quantity</script>')
        self.assertNotContains(response, 'Patient Hidden Value')
        self.assertNotContains(response, '555-987-1234')
        self.assertEqual(response.context['request_entries'][0].requested_at, record.requested_at)

    def test_history_keeps_previous_drug_links_once_after_reassignment(self):
        self.client.force_login(self.staff)
        replacement = PrescriptionDrug.objects.create(name='Rivaroxaban', brand='PMS', strength='20 mg')
        record = self.make_record(drug=replacement)
        for quantity in ('1 bottle', '2 bottles'):
            PrescriptionDrugRequestRevision.objects.create(
                learning_record=record, drug=self.drug,
                snapshot={**record.source_snapshot, 'quantity_needed': quantity},
            )

        response = self.client.get(self.history_url())

        self.assertEqual(list(response.context['request_entries']), [record])
        self.assertContains(response, 'Reassigned to RIVAROXABAN')
        self.assertContains(response, 'Archived source')
        self.assertContains(response, 'Recorded changes (2)')

    def test_history_retains_catalogue_return_query_and_fragment(self):
        self.client.force_login(self.staff)
        catalogue_return = reverse('prescription_drugs') + '?q=APO&page=2#request-list'

        response = self.client.get(self.history_url(return_to=catalogue_return))

        self.assertEqual(response.context['page_return']['url'], catalogue_return)
        self.assertEqual(response.context['page_return']['destination'], 'Prescription Drugs')

    def test_history_rejects_external_return_and_uses_detail_fallback(self):
        self.client.force_login(self.staff)

        response = self.client.get(self.history_url(return_to='https://example.org/phishing'))

        self.assertEqual(response.context['page_return']['url'], reverse('prescription_drug_detail', args=[self.drug.pk]))

    def test_history_pagination_retains_return_and_orders_newest_requests_first(self):
        self.client.force_login(self.staff)
        records = [self.make_record() for _ in range(51)]
        catalogue_return = reverse('prescription_drugs') + '?q=APO&page=2'

        response = self.client.get(self.history_url(return_to=catalogue_return))

        self.assertEqual(len(response.context['request_entries']), 50)
        self.assertEqual(response.context['request_entries'][0], records[-1])
        self.assertContains(response, '?return_to=' + quote(catalogue_return, safe='/') + '&amp;page=2')
        response = self.client.get(self.history_url(return_to=catalogue_return, page=2))
        self.assertEqual(list(response.context['request_entries']), [records[0]])
        self.assertEqual(response.context['page_return']['url'], catalogue_return)

    def test_empty_and_missing_drug_history(self):
        self.client.force_login(self.staff)
        self.assertContains(self.client.get(self.history_url()), 'No ordering requests have been recorded')
        self.assertEqual(self.client.get(reverse('prescription_drug_history', args=[self.drug.pk + 1000])).status_code, 404)
