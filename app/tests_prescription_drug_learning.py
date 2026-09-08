from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from threading import Barrier
from unittest.mock import patch

from django.core.management import call_command
from django.db import IntegrityError, close_old_connections, connection, connections, transaction
from django.test import TestCase, TransactionTestCase

from .models import OrderingSheetEntry, PrescriptionDrug, PrescriptionDrugLearningRecord, PrescriptionDrugRequestRevision
from .prescription_drug_learning import PARSER_VERSION, learn_prescription_drugs, preview_prescription_drugs


class PrescriptionDrugLearningTests(TestCase):
    def add_entry(self, name='PMS-rivaroxaban 20mg', **kwargs):
        return OrderingSheetEntry.objects.create(name=name, initials='AB', **kwargs)

    def test_existing_and_repeated_rows_create_one_drug_with_source_records(self):
        first = self.add_entry()
        self.add_entry('pms rivaroxaban 20 MG', source=OrderingSheetEntry.SOURCE_GSHEET)

        result = learn_prescription_drugs()

        self.assertEqual(result, dict(processed=2, created=1, matched=1, skipped=0))
        drug = PrescriptionDrug.objects.get()
        self.assertEqual((drug.name, drug.brand, drug.strength), ('RIVAROXABAN', 'PMS', '20 mg'))
        self.assertEqual(drug.learning_records.count(), 2)
        self.assertEqual(first.prescription_learning.source_name, first.name)
        self.assertEqual(learn_prescription_drugs()['processed'], 0)

    def test_matches_manual_strength_format_without_rewriting_catalogue(self):
        drug = PrescriptionDrug.objects.create(name='RIVAROXABAN', brand='PMS', strength='20MG')
        self.add_entry()

        self.assertEqual(learn_prescription_drugs()['matched'], 1)
        drug.refresh_from_db()
        self.assertEqual(drug.strength, '20MG')
        self.assertEqual(PrescriptionDrug.objects.count(), 1)

    def test_database_prevents_equivalent_catalogue_duplicates(self):
        PrescriptionDrug.objects.create(name='RIVAROXABAN', brand='PMS', strength='20MG')

        with self.assertRaises(IntegrityError), transaction.atomic():
            PrescriptionDrug.objects.create(name='rivaroxaban', brand='pms', strength='20 mg')

    def test_incomplete_rows_have_skip_reasons_and_are_retried_after_edit(self):
        entry = self.add_entry('PMS-rivaroxaban')

        self.assertEqual(learn_prescription_drugs()['skipped'], 1)
        self.assertEqual(entry.prescription_learning.reason, 'missing_strength')
        self.assertFalse(PrescriptionDrug.objects.exists())
        self.assertEqual(learn_prescription_drugs()['processed'], 0)

        OrderingSheetEntry.objects.filter(pk=entry.pk).update(name='PMS-rivaroxaban 20mg')
        self.assertEqual(learn_prescription_drugs()['created'], 1)
        record = PrescriptionDrugLearningRecord.objects.get(entry=entry)
        self.assertEqual(record.reason, '')
        self.assertIsNotNone(record.drug_id)

    def test_otc_rows_are_excluded_but_historical_drug_requests_are_retained(self):
        self.add_entry(entry_type=OrderingSheetEntry.ENTRY_OTC)
        self.add_entry(is_deleted=True)
        self.add_entry(status=OrderingSheetEntry.STATUS_CANCELLED)
        self.add_entry(status=OrderingSheetEntry.STATUS_NOT_FOR_SALE)

        self.assertEqual(learn_prescription_drugs()['processed'], 3)
        self.assertEqual(PrescriptionDrugLearningRecord.objects.count(), 3)
        self.assertEqual(PrescriptionDrug.objects.get().request_count, 3)

    def test_completed_rows_can_teach_and_restored_rows_are_picked_up(self):
        self.add_entry(status=OrderingSheetEntry.STATUS_PICKED_UP)
        restored = self.add_entry('APO-omeprazole 20mg', is_deleted=True)

        self.assertEqual(learn_prescription_drugs()['created'], 2)
        restored.is_deleted = False
        restored.save(update_fields=['is_deleted'])
        self.assertEqual(learn_prescription_drugs()['matched'], 1)

    def test_learning_does_not_read_private_fields_or_change_source(self):
        entry = self.add_entry(patient_name='PRIVATE PATIENT', phone_number='5551234567', order_note='PRIVATE NOTE')
        before = OrderingSheetEntry.objects.values().get(pk=entry.pk)

        with self.assertNumQueries(2):
            # The pending work query intentionally defers every unrelated field.
            from .prescription_drug_learning import pending_prescription_drug_entries
            source = pending_prescription_drug_entries().get(pk=entry.pk)
        self.assertTrue({'patient_name', 'phone_number', 'order_note'}.issubset(source.get_deferred_fields()))
        learn_prescription_drugs()
        self.assertEqual(OrderingSheetEntry.objects.values().get(pk=entry.pk), before)
        record = PrescriptionDrugLearningRecord.objects.values().get(entry=entry)
        self.assertNotIn('PRIVATE', str(record))

    def test_batch_size_limits_work_and_next_pass_continues(self):
        self.add_entry()
        self.add_entry('APO-omeprazole 20mg')

        self.assertEqual(learn_prescription_drugs(batch_size=1)['processed'], 1)
        self.assertEqual(learn_prescription_drugs(batch_size=1)['processed'], 1)
        self.assertEqual(learn_prescription_drugs(batch_size=1)['processed'], 0)

    def test_failed_pass_rolls_back_catalogue_and_remains_retryable(self):
        self.add_entry()
        with patch(
            'app.prescription_drug_learning.PrescriptionDrugLearningRecord.objects.update_or_create',
            side_effect=RuntimeError('simulated interrupted pass'),
        ):
            with self.assertRaises(RuntimeError):
                learn_prescription_drugs()

        self.assertFalse(PrescriptionDrug.objects.exists())
        self.assertFalse(PrescriptionDrugLearningRecord.objects.exists())
        self.assertEqual(learn_prescription_drugs()['created'], 1)

    def test_parser_version_change_rechecks_saved_results(self):
        self.add_entry()
        learn_prescription_drugs()
        with patch('app.prescription_drug_learning.PARSER_VERSION', PARSER_VERSION + 1):
            self.assertEqual(learn_prescription_drugs()['matched'], 1)
        self.assertEqual(PrescriptionDrugLearningRecord.objects.get().parser_version, PARSER_VERSION + 1)

    def test_invalid_parsed_details_are_skipped_without_truncation(self):
        self.add_entry()
        details = {'name': 'N' * 201, 'brand': 'PMS', 'strength': '20 mg'}
        with patch('app.prescription_drug_learning.parse_prescription_drug_label', return_value=(details, '')):
            self.assertEqual(learn_prescription_drugs()['skipped'], 1)
        self.assertEqual(PrescriptionDrugLearningRecord.objects.get().reason, 'invalid_details')

    def test_preview_and_dry_run_command_do_not_write(self):
        self.add_entry()
        self.add_entry()
        self.add_entry('Rivaroxaban 20mg')
        expected = dict(processed=3, created=1, matched=1, skipped=1)

        self.assertEqual(preview_prescription_drugs(), expected)
        output = StringIO()
        call_command('learn_prescription_drugs', dry_run=True, stdout=output)
        self.assertIn('Preview: 3 rows checked; 1 new drugs; 1 existing matches;', output.getvalue())
        self.assertFalse(PrescriptionDrug.objects.exists())
        self.assertFalse(PrescriptionDrugLearningRecord.objects.exists())

    def test_command_drains_all_pending_batches(self):
        self.add_entry()
        self.add_entry('APO-omeprazole 20mg')
        output = StringIO()
        call_command('learn_prescription_drugs', batch_size=1, stdout=output)

        self.assertIn('Completed: 2 rows checked; 2 new drugs;', output.getvalue())
        self.assertEqual(PrescriptionDrug.objects.count(), 2)


class PrescriptionDrugQuantityHistoryTests(TestCase):
    def add_entry(self, quantity='3', **kwargs):
        return OrderingSheetEntry.objects.create(
            name=kwargs.pop('name', 'PMS-rivaroxaban 20mg'),
            initials='AB', quantity_needed=quantity, **kwargs,
        )

    def test_repeated_passes_count_each_request_once_and_keep_request_date(self):
        first = self.add_entry('3 tablets')
        self.add_entry('2 tabs')
        learn_prescription_drugs()
        self.assertEqual(learn_prescription_drugs()['processed'], 0)

        drug = PrescriptionDrug.objects.get()
        self.assertEqual(drug.total_quantity_needed, {'tablets': '5.000'})
        self.assertEqual(drug.request_count, 2)
        self.assertEqual(drug.unknown_quantity_count, 0)
        self.assertEqual(drug.total_requested_display, '5 tablets')
        self.assertEqual(PrescriptionDrugRequestRevision.objects.count(), 2)
        self.assertEqual(first.prescription_learning.requested_at, first.created_at)

    def test_quantity_edit_replaces_latest_total_and_retains_observed_versions(self):
        entry = self.add_entry('3')
        learn_prescription_drugs()
        OrderingSheetEntry.objects.filter(pk=entry.pk).update(quantity_needed='5')

        self.assertEqual(learn_prescription_drugs()['processed'], 1)
        drug = PrescriptionDrug.objects.get()
        self.assertEqual(drug.total_quantity_needed, {'unspecified': '5.000'})
        self.assertEqual(drug.request_count, 1)
        record = PrescriptionDrugLearningRecord.objects.get(entry=entry)
        self.assertEqual(record.source_snapshot['quantity_needed'], '5')
        self.assertEqual(list(record.revisions.order_by('pk').values_list('snapshot__quantity_needed', flat=True)), ['3', '5'])
        self.assertEqual(learn_prescription_drugs()['processed'], 0)

    def test_explicit_units_bare_numbers_and_unknown_values_remain_separate(self):
        for quantity in ('2 boxes', '30 tablets', '10', '', 'JM'):
            self.add_entry(quantity)

        learn_prescription_drugs()

        drug = PrescriptionDrug.objects.get()
        self.assertEqual(drug.total_quantity_needed, {'boxes': '2.000', 'tablets': '30.000', 'unspecified': '10.000'})
        self.assertEqual(drug.request_count, 5)
        self.assertEqual(drug.unknown_quantity_count, 2)
        self.assertEqual(PrescriptionDrugLearningRecord.objects.filter(quantity_needed__isnull=True).count(), 2)

    def test_missing_to_zero_is_detected_without_treating_unknown_as_zero(self):
        entry = self.add_entry('')
        learn_prescription_drugs()
        self.assertEqual(PrescriptionDrug.objects.get().unknown_quantity_count, 1)
        entry.quantity_needed = '0'
        entry.save(update_fields=['quantity_needed'])
        learn_prescription_drugs()

        drug = PrescriptionDrug.objects.get()
        self.assertEqual(drug.total_quantity_needed, {'unspecified': '0.000'})
        self.assertEqual(drug.unknown_quantity_count, 0)

    def test_fulfillment_cancellation_archiving_and_restoring_preserve_historical_total(self):
        entry = self.add_entry('5 bottles', quantity_remaining='2 bottles')
        learn_prescription_drugs()
        for changes in (
            {'status': 'received', 'quantity_ordered': 5, 'quantity_received': 5},
            {'status': 'cancelled'}, {'is_deleted': True}, {'is_deleted': False},
        ):
            OrderingSheetEntry.objects.filter(pk=entry.pk).update(**changes)
            self.assertEqual(learn_prescription_drugs()['processed'], 1)
            self.assertEqual(PrescriptionDrug.objects.get().total_quantity_needed, {'bottles': '5.000'})
        record = PrescriptionDrugLearningRecord.objects.get(entry=entry)
        self.assertEqual(record.revisions.count(), 5)
        self.assertEqual(record.source_snapshot['quantity_remaining'], '2 bottles')
        self.assertEqual(record.source_snapshot['quantity_received'], 5)

    def test_reclassifying_as_otc_removes_totals_but_retains_prior_history(self):
        entry = self.add_entry('4')
        learn_prescription_drugs()
        drug = PrescriptionDrug.objects.get()
        entry.entry_type = OrderingSheetEntry.ENTRY_OTC
        entry.save(update_fields=['entry_type'])
        learn_prescription_drugs()

        drug.refresh_from_db()
        self.assertEqual(drug.total_quantity_needed, {})
        self.assertEqual(drug.request_count, 0)
        record = PrescriptionDrugLearningRecord.objects.get(entry=entry)
        self.assertIsNone(record.drug_id)
        self.assertEqual(record.reason, 'ineligible_source')
        self.assertEqual(drug.request_revisions.count(), 1)

    def test_drug_correction_moves_latest_request_without_losing_old_revision(self):
        entry = self.add_entry('3')
        learn_prescription_drugs()
        original = PrescriptionDrug.objects.get()
        entry.name = 'APO-omeprazole 20mg'
        entry.save(update_fields=['name'])
        learn_prescription_drugs()

        original.refresh_from_db()
        corrected = PrescriptionDrug.objects.get(name='OMEPRAZOLE')
        self.assertEqual(original.total_quantity_needed, {})
        self.assertEqual(original.request_count, 0)
        self.assertEqual(corrected.total_quantity_needed, {'unspecified': '3.000'})
        self.assertEqual(original.request_revisions.count(), 1)

    def test_source_purge_retains_request_history_and_total(self):
        entry = self.add_entry('3')
        learn_prescription_drugs()
        entry_id = entry.pk
        entry.delete()

        record = PrescriptionDrugLearningRecord.objects.get()
        self.assertIsNone(record.entry_id)
        self.assertEqual(record.source_snapshot['entry_id'], entry_id)
        self.assertEqual(record.revisions.count(), 1)
        self.assertEqual(PrescriptionDrug.objects.get().total_quantity_needed, {'unspecified': '3.000'})

    def test_learning_never_infers_or_overwrites_pack_size(self):
        drug = PrescriptionDrug.objects.create(name='RIVAROXABAN', brand='PMS', strength='20 mg', pack_size='100 tablets')
        self.add_entry('3 boxes')
        learn_prescription_drugs()
        drug.refresh_from_db()

        self.assertEqual(drug.pack_size, '100 tablets')
        self.assertEqual(drug.total_quantity_needed, {'boxes': '3.000'})

    def test_parser_recheck_does_not_create_fake_quantity_history(self):
        self.add_entry()
        learn_prescription_drugs()
        with patch('app.prescription_drug_learning.PARSER_VERSION', PARSER_VERSION + 1):
            learn_prescription_drugs()
        self.assertEqual(PrescriptionDrugRequestRevision.objects.count(), 1)

    def test_failed_summary_update_rolls_back_request_revision_and_can_retry(self):
        self.add_entry()
        with patch('app.prescription_drug_learning._refresh_request_totals', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                learn_prescription_drugs()
        self.assertFalse(PrescriptionDrugLearningRecord.objects.exists())
        self.assertFalse(PrescriptionDrugRequestRevision.objects.exists())
        self.assertEqual(learn_prescription_drugs()['created'], 1)


class ConcurrentPrescriptionDrugLearningTests(TransactionTestCase):
    def _run_concurrent_passes(self):
        if connection.vendor != 'postgresql':
            self.skipTest('Checks PostgreSQL row locks used by this deployment.')
        barrier = Barrier(2)

        def run_pass():
            close_old_connections()
            try:
                barrier.wait(timeout=10)
                return learn_prescription_drugs(batch_size=1)
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_pass) for _ in range(2)]
            for future in futures:
                future.result(timeout=20)
        learn_prescription_drugs()

    def test_concurrent_workers_do_not_duplicate_drugs_or_lose_sources(self):
        OrderingSheetEntry.objects.bulk_create([
            OrderingSheetEntry(name='PMS-rivaroxaban 20mg', initials='AB', quantity_needed='2 tablets'),
            OrderingSheetEntry(name='pms rivaroxaban 20 MG', initials='CD', quantity_needed='3 tablets'),
        ])
        self._run_concurrent_passes()

        self.assertEqual(PrescriptionDrug.objects.count(), 1)
        self.assertEqual(PrescriptionDrugLearningRecord.objects.count(), 2)
        self.assertEqual(PrescriptionDrug.objects.get().total_quantity_needed, {'tablets': '5.000'})
        self.assertEqual(PrescriptionDrugRequestRevision.objects.count(), 2)

    def test_concurrent_corrections_to_existing_requests_keep_final_total(self):
        entries = OrderingSheetEntry.objects.bulk_create([
            OrderingSheetEntry(name='PMS-rivaroxaban 20mg', initials='AB', quantity_needed='2 tablets'),
            OrderingSheetEntry(name='pms rivaroxaban 20 MG', initials='CD', quantity_needed='3 tablets'),
        ])
        learn_prescription_drugs()
        OrderingSheetEntry.objects.filter(pk=entries[0].pk).update(quantity_needed='4 tablets')
        OrderingSheetEntry.objects.filter(pk=entries[1].pk).update(quantity_needed='6 tablets')

        self._run_concurrent_passes()

        drug = PrescriptionDrug.objects.get()
        self.assertEqual(drug.total_quantity_needed, {'tablets': '10.000'})
        self.assertEqual(drug.request_count, 2)
        self.assertEqual(PrescriptionDrugLearningRecord.objects.count(), 2)
        self.assertEqual(PrescriptionDrugRequestRevision.objects.count(), 4)
