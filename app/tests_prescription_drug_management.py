import time
from decimal import Decimal
from unittest.mock import patch
from urllib.parse import urlencode

from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.forms.models import model_to_dict
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .mixins import PASSKEY_SESSION_KEY
from .models import (
    OrderingSheetEntry, PrescriptionDrug, PrescriptionDrugChange,
    PrescriptionDrugLearningRecord, PrescriptionDrugPackage,
    PrescriptionDrugSupplierItem, Product, ProductLot, ProductLotMovement,
    StockChange,
)
from .prescription_drug_learning import learn_prescription_drugs, pending_prescription_drug_entries
from .prescription_drug_management import (
    StalePrescriptionDrugEdit, save_catalogue_form,
)
from .prescription_drug_views import (
    PrescriptionDrugEditForm, PrescriptionDrugPackageForm,
    PrescriptionDrugSupplierForm,
)


def form_data(instance, **changes):
    """Submit a complete record as the browser would, retaining model defaults."""
    data = {key: '' if value is None else value for key, value in model_to_dict(instance).items()}
    data['record_version'] = instance.version
    data.update(changes)
    return data


@override_settings(AXES_ENABLED=False)
class PrescriptionDrugManagementTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.staff = get_user_model().objects.create_user(
            username='drug-management-admin', is_staff=True,
        )
        cls.pu = get_user_model().objects.create_user(username='drug-management-pu')
        cls.drug = PrescriptionDrug.objects.create(
            name='Rivaroxaban', brand='PMS', strength='20 mg',
            total_quantity_needed={'tablets': '12'}, request_count=2,
            unknown_quantity_count=1,
        )
        cls.other_drug = PrescriptionDrug.objects.create(
            name='Amoxicillin', brand='APO', strength='250 mg/5 mL',
        )
        cls.package = PrescriptionDrugPackage.objects.create(
            drug=cls.drug, label='100 tablets', quantity=100, unit='tablets',
        )
        cls.other_package = PrescriptionDrugPackage.objects.create(
            drug=cls.other_drug, label='100 mL', quantity=100, unit='mL',
        )
        cls.supplier = PrescriptionDrugSupplierItem.objects.create(
            package=cls.package, supplier_name='McKesson', item_number='ABC',
            pack_cost=Decimal('21.1200'), is_preferred=True,
        )
        cls.other_supplier = PrescriptionDrugSupplierItem.objects.create(
            package=cls.other_package, supplier_name='Kohl & Frisch',
            item_number='DEF',
        )

    def setUp(self):
        self.client.force_login(self.staff)
        self.detail_url = reverse('prescription_drug_detail', args=[self.drug.pk])
        self.edit_url = reverse('edit_prescription_drug', args=[self.drug.pk])

    def test_detail_and_edit_routes_follow_existing_staff_and_passkey_access(self):
        urls = [
            self.detail_url, self.edit_url,
            reverse('add_prescription_drug_package', args=[self.drug.pk]),
            reverse('edit_prescription_drug_package', args=[self.drug.pk, self.package.pk]),
            reverse('add_prescription_drug_supplier', args=[self.drug.pk, self.package.pk]),
            reverse('edit_prescription_drug_supplier', args=[self.drug.pk, self.package.pk, self.supplier.pk]),
        ]
        self.client.logout()
        for url in urls:
            with self.subTest(url=url, access='anonymous'):
                self.assertRedirects(self.client.get(url), reverse('login'), fetch_redirect_response=False)
        self.client.force_login(self.pu)
        for url in urls:
            with self.subTest(url=url, access='locked'):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertTrue(response.url.startswith(reverse('passkey_unlock') + '?next='))
        session = self.client.session
        session[PASSKEY_SESSION_KEY] = time.time()
        session.save()
        for url in urls:
            with self.subTest(url=url, access='unlocked'):
                self.assertEqual(self.client.get(url).status_code, 200)

    def test_all_editing_endpoints_require_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(self.staff)
        urls = [
            self.edit_url,
            reverse('add_prescription_drug_package', args=[self.drug.pk]),
            reverse('edit_prescription_drug_package', args=[self.drug.pk, self.package.pk]),
            reverse('add_prescription_drug_supplier', args=[self.drug.pk, self.package.pk]),
            reverse('edit_prescription_drug_supplier', args=[self.drug.pk, self.package.pk, self.supplier.pk]),
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(csrf_client.post(url, {}).status_code, 403)

    def test_child_routes_cannot_read_or_change_another_drugs_records(self):
        urls = [
            reverse('edit_prescription_drug_package', args=[self.drug.pk, self.other_package.pk]),
            reverse('add_prescription_drug_supplier', args=[self.drug.pk, self.other_package.pk]),
            reverse('edit_prescription_drug_supplier', args=[self.drug.pk, self.package.pk, self.other_supplier.pk]),
            reverse('edit_prescription_drug_supplier', args=[self.drug.pk, self.other_package.pk, self.other_supplier.pk]),
        ]
        for url in urls:
            for method in ('get', 'post'):
                with self.subTest(url=url, method=method):
                    self.assertEqual(getattr(self.client, method)(url, {}).status_code, 404)
        self.other_package.refresh_from_db()
        self.other_supplier.refresh_from_db()
        self.assertEqual(self.other_package.label, '100 mL')
        self.assertEqual(self.other_supplier.item_number, 'DEF')

    def test_detail_uses_explicit_inventory_link_and_never_repairs_stock_on_read(self):
        product = Product.objects.create(
            name='Linked inventory package', price=Decimal('31.00'), quantity_in_stock=9,
        )
        lot = ProductLot.objects.create(product=product, lot_number='VISIBLE-LOT', quantity_on_hand=7)
        ProductLot.objects.create(
            product=product, lot_number='ARCHIVED-LOT', quantity_on_hand=4,
            archived_at=timezone.now(),
        )
        unrelated = Product.objects.create(
            name=self.drug.name, brand=self.drug.brand, price=Decimal('2.00'), quantity_in_stock=99,
        )
        ProductLot.objects.create(product=unrelated, lot_number='UNRELATED-LOT', quantity_on_hand=99)
        self.package.inventory_product = product
        self.package.save(update_fields=['inventory_product'])
        models = [PrescriptionDrug, PrescriptionDrugPackage, PrescriptionDrugChange, ProductLot, ProductLotMovement, StockChange]
        counts = {model: model.objects.count() for model in models}

        response = self.client.get(self.detail_url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['drug'].pk, self.drug.pk)
        self.assertContains(response, 'Linked inventory package')
        self.assertContains(response, 'VISIBLE-LOT')
        self.assertNotContains(response, 'ARCHIVED-LOT')
        self.assertNotContains(response, 'UNRELATED-LOT')
        product.refresh_from_db()
        lot.refresh_from_db()
        self.assertEqual(product.quantity_in_stock, 9)
        self.assertEqual(lot.quantity_on_hand, 7)
        self.assertEqual(counts, {model: model.objects.count() for model in models})

    def test_details_return_preserves_catalogue_search_page_and_fragment(self):
        origin = reverse('prescription_drugs') + '?q=PMS&page=2#catalogue'
        response = self.client.get(self.detail_url, {'return_to': origin})
        self.assertEqual(response.context['page_return']['url'], origin)
        self.assertEqual(response.context['list_return'], origin)

    def test_edit_redirect_preserves_list_context_and_rejects_external_return(self):
        origin = reverse('prescription_drugs') + '?q=PMS&page=2#catalogue'
        detail_return = self.detail_url + '?' + urlencode({'return_to': origin})
        data = form_data(self.drug, notes='Staff-reviewed storage note', return_to=detail_return)
        response = self.client.post(self.edit_url, data)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith(self.detail_url))
        reopened = self.client.get(response.url)
        self.assertEqual(reopened.context['list_return'], origin)

        for unsafe in ('https://outside.example/', '//outside.example/'):
            with self.subTest(return_to=unsafe):
                response = self.client.get(self.detail_url, {'return_to': unsafe})
                self.assertEqual(response.context['page_return']['url'], reverse('prescription_drugs'))

    def test_manual_edit_preserves_learner_totals_and_records_actor_and_changes(self):
        form = PrescriptionDrugEditForm(
            data=form_data(self.drug, name='  Rivaroxaban  ', brand=' pms ', notes='Reviewed',
                           total_quantity_needed='{"tablets":"9000"}', request_count=500,
                           unknown_quantity_count=0, source='ordering_sheet', version=999),
            instance=PrescriptionDrug.objects.get(pk=self.drug.pk),
        )
        self.assertTrue(form.is_valid(), form.errors)
        # The background learner may update its own summaries after validation.
        PrescriptionDrug.objects.filter(pk=self.drug.pk).update(
            total_quantity_needed={'tablets': '22'}, request_count=3,
        )

        result = save_catalogue_form(form, actor=self.staff)

        result.refresh_from_db()
        self.assertEqual((result.name, result.brand), ('RIVAROXABAN', 'PMS'))
        self.assertEqual(result.notes, 'Reviewed')
        self.assertEqual(result.total_quantity_needed, {'tablets': '22'})
        self.assertEqual(result.request_count, 3)
        self.assertEqual(result.unknown_quantity_count, 1)
        self.assertEqual(result.source, 'manual')
        self.assertEqual(result.version, 2)
        change = PrescriptionDrugChange.objects.get(drug=self.drug)
        self.assertEqual(change.actor_id, self.staff.pk)
        self.assertEqual(change.actor_label, self.staff.username)
        self.assertEqual(change.before['notes'], '')
        self.assertEqual(change.after['notes'], 'Reviewed')

    def test_stale_service_edit_cannot_overwrite_newer_change_or_add_audit(self):
        first = PrescriptionDrugEditForm(
            data=form_data(self.drug, notes='First accepted change'),
            instance=PrescriptionDrug.objects.get(pk=self.drug.pk),
        )
        stale = PrescriptionDrugEditForm(
            data=form_data(self.drug, notes='Stale change'),
            instance=PrescriptionDrug.objects.get(pk=self.drug.pk),
        )
        self.assertTrue(first.is_valid(), first.errors)
        self.assertTrue(stale.is_valid(), stale.errors)
        save_catalogue_form(first, actor=self.staff)
        audit_count = PrescriptionDrugChange.objects.count()

        with self.assertRaises(StalePrescriptionDrugEdit):
            save_catalogue_form(stale, actor=self.staff)

        self.drug.refresh_from_db()
        self.assertEqual(self.drug.notes, 'First accepted change')
        self.assertEqual(self.drug.version, 2)
        self.assertEqual(PrescriptionDrugChange.objects.count(), audit_count)

    def test_failed_audit_rolls_back_the_catalogue_change(self):
        form = PrescriptionDrugEditForm(
            data=form_data(self.drug, notes='Must not survive a failed audit'),
            instance=self.drug,
        )
        self.assertTrue(form.is_valid(), form.errors)
        with patch.object(PrescriptionDrugChange.objects, 'create', side_effect=RuntimeError('audit unavailable')):
            with self.assertRaises(RuntimeError):
                save_catalogue_form(form, actor=self.staff)
        self.drug.refresh_from_db()
        self.assertEqual(self.drug.notes, '')
        self.assertEqual(self.drug.version, 1)
        self.assertFalse(PrescriptionDrugChange.objects.exists())

    def test_edit_route_returns_conflict_for_stale_version_and_preserves_posted_values(self):
        PrescriptionDrug.objects.filter(pk=self.drug.pk).update(notes='Newer saved notes', version=2)
        response = self.client.post(self.edit_url, form_data(self.drug, notes='Unsaved browser notes'))
        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.context['form'].non_field_errors())
        self.assertContains(response, 'Unsaved browser notes', status_code=409)
        self.drug.refresh_from_db()
        self.assertEqual(self.drug.notes, 'Newer saved notes')
        self.assertFalse(PrescriptionDrugChange.objects.exists())

    def test_edit_requires_record_version(self):
        data = form_data(self.drug, notes='Missing version')
        data.pop('record_version')
        response = self.client.post(self.edit_url, data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('record_version', response.context['form'].errors)
        self.drug.refresh_from_db()
        self.assertEqual(self.drug.notes, '')

    def test_din_keeps_leading_zeroes_rejects_invalid_and_duplicate_values(self):
        form = PrescriptionDrugEditForm(
            data=form_data(self.drug, din=' 01234567 '), instance=self.drug,
        )
        self.assertTrue(form.is_valid(), form.errors)
        saved = save_catalogue_form(form, actor=self.staff)
        self.assertEqual(saved.din, '01234567')
        for invalid in ('1234567', '123456789', 'A1234567', '1234-567'):
            with self.subTest(din=invalid):
                form = PrescriptionDrugEditForm(
                    data=form_data(self.other_drug, din=invalid), instance=self.other_drug,
                )
                self.assertFalse(form.is_valid())
                self.assertIn('din', form.errors)
        duplicate = PrescriptionDrugEditForm(
            data=form_data(self.other_drug, din='01234567'), instance=self.other_drug,
        )
        self.assertFalse(duplicate.is_valid())
        with self.assertRaises(IntegrityError), transaction.atomic():
            PrescriptionDrug.objects.filter(pk=self.other_drug.pk).update(din='01234567')

    def test_dosage_form_and_route_distinguish_drugs_but_duplicate_identity_is_rejected(self):
        self.drug.dosage_form = 'Tablet'
        self.drug.route = 'Oral'
        self.drug.save(update_fields=['dosage_form', 'route'])
        second = PrescriptionDrug.objects.create(
            name=self.drug.name, brand=self.drug.brand, strength=self.drug.strength,
            dosage_form='Solution', route='Oral',
        )
        form = PrescriptionDrugEditForm(data=form_data(second, notes='Distinct dosage form'), instance=second)
        self.assertTrue(form.is_valid(), form.errors)
        duplicate = PrescriptionDrugEditForm(
            data=form_data(second, strength='20MG', dosage_form=' tablet ', route=' oral '),
            instance=second,
        )
        self.assertFalse(duplicate.is_valid())
        response = self.client.post(
            reverse('edit_prescription_drug', args=[second.pk]),
            form_data(second, dosage_form='Tablet', route='Oral'),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['form'].errors)
        second.refresh_from_db()
        self.assertEqual(second.dosage_form, 'Solution')

    def test_learner_does_not_guess_between_dosage_forms(self):
        self.drug.dosage_form = 'Tablet'
        self.drug.route = 'Oral'
        self.drug.save(update_fields=['dosage_form', 'route'])
        PrescriptionDrug.objects.create(
            name=self.drug.name, brand=self.drug.brand, strength=self.drug.strength,
            dosage_form='Solution', route='Oral',
        )
        entry = OrderingSheetEntry.objects.create(
            name='PMS-rivaroxaban 20mg', initials='AB', quantity_needed='12 tablets',
        )
        before = PrescriptionDrug.objects.count()

        result = learn_prescription_drugs()

        record = PrescriptionDrugLearningRecord.objects.get(entry=entry)
        self.assertEqual(result['skipped'], 1)
        self.assertIsNone(record.drug_id)
        self.assertEqual(record.reason, 'ambiguous')
        self.assertEqual(PrescriptionDrug.objects.count(), before)

    def test_audited_master_edit_rechecks_unchanged_requests_and_restores_totals_after_resolution(self):
        entry = OrderingSheetEntry.objects.create(
            name='PMS-rivaroxaban 20mg', initials='AB', quantity_needed='12 tablets',
        )
        learn_prescription_drugs()
        record = PrescriptionDrugLearningRecord.objects.get(entry=entry)
        source_snapshot = record.source_snapshot
        self.assertEqual(record.drug_id, self.drug.pk)
        self.assertEqual(learn_prescription_drugs()['processed'], 0)

        form = PrescriptionDrugEditForm(
            data=form_data(self.other_drug, name=self.drug.name, brand=self.drug.brand,
                           strength=self.drug.strength, dosage_form='Solution', route='Oral'),
            instance=PrescriptionDrug.objects.get(pk=self.other_drug.pk),
        )
        self.assertTrue(form.is_valid(), form.errors)
        alternative = save_catalogue_form(form, actor=self.staff)
        revision = PrescriptionDrugChange.objects.filter(entity_type='drug').latest('pk').pk
        self.assertTrue(pending_prescription_drug_entries().filter(pk=entry.pk).exists())

        result = learn_prescription_drugs()

        self.assertEqual(result['processed'], 1)
        self.assertEqual(result['skipped'], 1)
        record.refresh_from_db()
        self.drug.refresh_from_db()
        self.assertEqual(record.source_snapshot, source_snapshot)
        self.assertEqual(record.catalogue_revision, revision)
        self.assertIsNone(record.drug_id)
        self.assertEqual(record.reason, 'ambiguous')
        self.assertEqual(record.revisions.count(), 2)
        self.assertEqual(self.drug.request_count, 0)
        self.assertEqual(self.drug.total_quantity_needed, {})
        self.assertEqual(learn_prescription_drugs()['processed'], 0)

        resolution = PrescriptionDrugEditForm(
            data=form_data(alternative, name='Amoxicillin', brand='APO', strength='250 mg/5 mL'),
            instance=alternative,
        )
        self.assertTrue(resolution.is_valid(), resolution.errors)
        save_catalogue_form(resolution, actor=self.staff)
        self.assertTrue(pending_prescription_drug_entries().filter(pk=entry.pk).exists())
        self.assertEqual(learn_prescription_drugs()['matched'], 1)
        record.refresh_from_db()
        self.drug.refresh_from_db()
        self.assertEqual(record.drug_id, self.drug.pk)
        self.assertEqual(record.revisions.count(), 3)
        self.assertEqual(self.drug.request_count, 1)
        self.assertEqual(Decimal(self.drug.total_quantity_needed['tablets']), Decimal('12'))
        self.assertEqual(learn_prescription_drugs()['processed'], 0)

    def test_audited_master_add_rechecks_existing_requests_when_a_second_form_is_added(self):
        entry = OrderingSheetEntry.objects.create(
            name='PMS-rivaroxaban 20mg', initials='AB', quantity_needed='3 tablets',
        )
        learn_prescription_drugs()
        self.assertEqual(learn_prescription_drugs()['processed'], 0)
        candidate = PrescriptionDrug(
            name=self.drug.name, brand=self.drug.brand, strength=self.drug.strength,
            dosage_form='Tablet', route='Oral',
        )
        form = PrescriptionDrugEditForm(data=form_data(candidate), instance=candidate)
        self.assertTrue(form.is_valid(), form.errors)
        save_catalogue_form(form, actor=self.staff)

        self.assertTrue(pending_prescription_drug_entries().filter(pk=entry.pk).exists())
        self.assertEqual(learn_prescription_drugs()['skipped'], 1)
        record = PrescriptionDrugLearningRecord.objects.get(entry=entry)
        self.assertEqual(record.reason, 'ambiguous')
        self.assertIsNone(record.drug_id)
        self.assertGreater(record.catalogue_revision, 0)
        self.assertEqual(learn_prescription_drugs()['processed'], 0)

    def test_package_service_saves_blank_and_normalized_upc_and_allows_clearing_it(self):
        for index, upc in enumerate(('', ' 000123-456 ')):
            with self.subTest(upc=upc):
                candidate = PrescriptionDrugPackage(drug=self.drug, label=f'Service pack {index}')
                form = PrescriptionDrugPackageForm(data=form_data(candidate, upc=upc), instance=candidate)
                self.assertTrue(form.is_valid(), form.errors)
                saved = save_catalogue_form(form, actor=self.staff)
                saved.refresh_from_db()
                self.assertEqual(saved.upc, upc.strip())
                self.assertEqual(saved.normalized_upc, '123456' if upc else None)
                edit = PrescriptionDrugPackageForm(data=form_data(saved, upc=''), instance=saved)
                self.assertTrue(edit.is_valid(), edit.errors)
                cleared = save_catalogue_form(edit, actor=self.staff)
                cleared.refresh_from_db()
                self.assertEqual(cleared.upc, '')
                self.assertIsNone(cleared.normalized_upc)

    def test_package_http_create_and_edit_save_optional_upc(self):
        create_url = reverse('add_prescription_drug_package', args=[self.drug.pk])
        for index, upc in enumerate(('', ' 000987-654 ')):
            with self.subTest(upc=upc):
                candidate = PrescriptionDrugPackage(drug=self.drug, label=f'Browser pack {index}')
                response = self.client.post(create_url, form_data(candidate, upc=upc))
                self.assertRedirects(response, self.detail_url, fetch_redirect_response=False)
                saved = self.drug.packages.get(label=candidate.label)
                self.assertEqual(saved.normalized_upc, '987654' if upc else None)
                edit_url = reverse('edit_prescription_drug_package', args=[self.drug.pk, saved.pk])
                replacement = f'000111-22{index}'
                response = self.client.post(edit_url, form_data(saved, upc=replacement))
                self.assertRedirects(response, self.detail_url, fetch_redirect_response=False)
                saved.refresh_from_db()
                self.assertEqual(saved.upc, replacement)
                self.assertEqual(saved.normalized_upc, f'11122{index}')

    def test_upc_normalization_prevents_duplicate_pack_codes(self):
        self.package.upc = '000123-456'
        self.package.save(update_fields=['upc'])
        self.assertEqual(self.package.normalized_upc, '123456')
        form = PrescriptionDrugPackageForm(
            data=form_data(self.other_package, upc=' 00123 456 '), instance=self.other_package,
        )
        self.assertFalse(form.is_valid())
        self.assertIn('upc', form.errors)
        with self.assertRaises(IntegrityError), transaction.atomic():
            PrescriptionDrugPackage.objects.create(drug=self.drug, label='Duplicate UPC', upc='123456')

    def test_pack_requires_quantity_and_unit_together_and_consistent_reorder_values(self):
        for changes in (
            {'quantity': 0, 'unit': 'tablets'},
            {'quantity': 10, 'unit': ''},
            {'quantity': '', 'unit': 'tablets'},
            {'reorder_point': -1},
            {'reorder_point': 10, 'target_stock': 9},
        ):
            with self.subTest(changes=changes):
                form = PrescriptionDrugPackageForm(
                    data=form_data(self.package, **changes),
                    instance=PrescriptionDrugPackage.objects.get(pk=self.package.pk),
                )
                self.assertFalse(form.is_valid())

    def test_inventory_product_cannot_be_linked_to_multiple_packs(self):
        product = Product.objects.create(name='Explicit stock link', price=Decimal('2.00'))
        self.package.inventory_product = product
        self.package.save(update_fields=['inventory_product'])
        form = PrescriptionDrugPackageForm(
            data=form_data(self.other_package, inventory_product=product.pk), instance=self.other_package,
        )
        self.assertFalse(form.is_valid())
        with self.assertRaises(IntegrityError), transaction.atomic():
            PrescriptionDrugPackage.objects.filter(pk=self.other_package.pk).update(inventory_product=product)

    def test_switching_preferred_supplier_audits_both_records_and_invalidates_old_form(self):
        previous_form = PrescriptionDrugSupplierForm(
            data=form_data(self.supplier, pack_cost='22.00'),
            instance=PrescriptionDrugSupplierItem.objects.get(pk=self.supplier.pk),
        )
        self.assertTrue(previous_form.is_valid(), previous_form.errors)
        candidate = PrescriptionDrugSupplierItem(package=self.package, supplier_name='Kohl & Frisch')
        form = PrescriptionDrugSupplierForm(
            data=form_data(candidate, item_number='NEW', is_preferred=True), instance=candidate,
        )
        self.assertTrue(form.is_valid(), form.errors)

        selected = save_catalogue_form(form, actor=self.staff)

        self.supplier.refresh_from_db()
        self.assertFalse(self.supplier.is_preferred)
        self.assertEqual(self.supplier.version, 2)
        self.assertTrue(selected.is_preferred)
        self.assertEqual(self.package.supplier_items.filter(is_preferred=True).count(), 1)
        changes = list(PrescriptionDrugChange.objects.filter(drug=self.drug))
        self.assertEqual(len(changes), 2)
        self.assertTrue(any(change.before.get('is_preferred') is True and change.after.get('is_preferred') is False for change in changes))
        with self.assertRaises(StalePrescriptionDrugEdit):
            save_catalogue_form(previous_form, actor=self.staff)
        self.supplier.refresh_from_db()
        self.assertEqual(self.supplier.pack_cost, Decimal('21.1200'))

    def test_supplier_validation_rejects_inactive_preferred_negative_cost_and_zero_multiple(self):
        for changes in (
            {'is_active': False, 'is_preferred': True},
            {'pack_cost': '-0.01'},
            {'catalogue_price': '-0.01'},
            {'order_multiple': 0},
        ):
            with self.subTest(changes=changes):
                form = PrescriptionDrugSupplierForm(
                    data=form_data(self.supplier, **changes),
                    instance=PrescriptionDrugSupplierItem.objects.get(pk=self.supplier.pk),
                )
                self.assertFalse(form.is_valid())
