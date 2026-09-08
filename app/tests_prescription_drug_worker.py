from pathlib import Path
from unittest.mock import Mock, call, patch

from django.conf import settings
from django.test import SimpleTestCase, override_settings

from app import prescription_drug_worker as worker


class PrescriptionDrugWorkerStartupTests(SimpleTestCase):
    def setUp(self):
        self.saved_thread = worker._worker_thread
        worker._worker_thread = None
        self.addCleanup(setattr, worker, '_worker_thread', self.saved_thread)

    @override_settings(PRESCRIPTION_DRUG_LEARNING_ENABLED=False)
    @patch.object(worker.threading, 'Thread')
    def test_disabled_worker_never_starts(self, thread_class):
        self.assertFalse(worker.start_prescription_drug_worker())
        thread_class.assert_not_called()

    @override_settings(PRESCRIPTION_DRUG_LEARNING_ENABLED=True)
    @patch.object(worker.threading, 'Thread')
    def test_repeated_start_keeps_one_daemon_worker(self, thread_class):
        thread_class.return_value.is_alive.return_value = True

        self.assertTrue(worker.start_prescription_drug_worker())
        self.assertFalse(worker.start_prescription_drug_worker())

        thread_class.assert_called_once()
        self.assertIs(thread_class.call_args.kwargs['target'], worker._run_worker)
        self.assertTrue(thread_class.call_args.kwargs['daemon'])
        thread_class.return_value.start.assert_called_once_with()

    @override_settings(PRESCRIPTION_DRUG_LEARNING_ENABLED=True)
    @patch.object(worker.threading, 'Thread')
    def test_terminated_worker_can_be_restarted(self, thread_class):
        worker._worker_thread = Mock()
        worker._worker_thread.is_alive.return_value = False

        self.assertTrue(worker.start_prescription_drug_worker())

        thread_class.return_value.start.assert_called_once_with()

    @override_settings(PRESCRIPTION_DRUG_LEARNING_ENABLED=True)
    @patch.object(worker.threading, 'Thread')
    def test_start_failure_does_not_prevent_website_startup(self, thread_class):
        thread_class.return_value.start.side_effect = RuntimeError('private detail')

        with self.assertLogs(worker.logger, level='WARNING') as logs:
            self.assertFalse(worker.start_prescription_drug_worker())

        self.assertIsNone(worker._worker_thread)
        self.assertNotIn('private detail', '\n'.join(logs.output))

    def test_web_entry_points_start_worker_after_application_initialization(self):
        for module, factory in (
            ('wsgi', 'django.core.wsgi.get_wsgi_application'),
            ('asgi', 'django.core.asgi.get_asgi_application'),
        ):
            with self.subTest(module=module):
                calls = Mock()
                application = object()
                source_path = Path(settings.BASE_DIR) / 'inventory' / f'{module}.py'
                with patch(factory, return_value=application) as initialize:
                    with patch.object(worker, 'start_prescription_drug_worker') as start:
                        calls.attach_mock(initialize, 'initialize')
                        calls.attach_mock(start, 'start')
                        namespace = {'__name__': f'worker_test_{module}'}
                        exec(compile(source_path.read_text(encoding='utf-8'), str(source_path), 'exec'), namespace)

                self.assertIs(namespace['application'], application)
                self.assertEqual(calls.mock_calls, [call.initialize(), call.start()])


@override_settings(PRESCRIPTION_DRUG_LEARNING_INTERVAL_SECONDS=60)
class PrescriptionDrugWorkerLoopTests(SimpleTestCase):
    def run_worker(self, outcomes, waits):
        stop_event = Mock()
        stop_event.is_set.return_value = False
        stop_event.wait.side_effect = waits
        with patch.object(worker, 'learn_prescription_drugs', side_effect=outcomes) as learn:
            with patch.object(worker, 'close_old_connections') as close_old:
                with patch.object(worker.connections, 'close_all') as close_all:
                    worker._run_worker(stop_event)
        return stop_event, learn, close_old, close_all

    def test_first_pass_is_immediate_and_idle_connections_are_closed(self):
        events = Mock()
        stop_event = Mock()
        stop_event.is_set.return_value = False
        stop_event.wait.return_value = True
        events.attach_mock(stop_event.wait, 'wait')
        with patch.object(worker, 'learn_prescription_drugs', return_value={'processed': 1}) as learn:
            with patch.object(worker, 'close_old_connections') as close_old:
                with patch.object(worker.connections, 'close_all') as close_all:
                    events.attach_mock(close_old, 'close_old')
                    events.attach_mock(learn, 'learn')
                    events.attach_mock(close_all, 'close_all')
                    worker._run_worker(stop_event)

        self.assertEqual(events.mock_calls, [
            call.close_old(), call.learn(batch_size=100),
            call.close_all(), call.wait(60),
        ])

    def test_failure_retries_after_interval_without_logging_source_details(self):
        with self.assertLogs(worker.logger, level='WARNING') as logs:
            stop, learn, close_old, close_all = self.run_worker(
                [RuntimeError('private patient and drug text'), {'processed': 0}],
                [False, True],
            )

        self.assertEqual(learn.call_count, 2)
        self.assertEqual(close_old.call_count, 2)
        self.assertEqual(close_all.call_count, 2)
        self.assertEqual(stop.wait.call_args_list, [call(60), call(60)])
        self.assertIn('RuntimeError', '\n'.join(logs.output))
        self.assertNotIn('private patient and drug text', '\n'.join(logs.output))

    def test_full_batch_drains_again_after_short_delay(self):
        stop, learn, _, _ = self.run_worker(
            [{'processed': 100}, {'processed': 4}], [False, True],
        )

        self.assertEqual(learn.call_args_list, [call(batch_size=100)] * 2)
        self.assertEqual(stop.wait.call_args_list, [call(1), call(60)])

    @override_settings(PRESCRIPTION_DRUG_LEARNING_INTERVAL_SECONDS=12)
    def test_configured_interval_is_used(self):
        stop, _, _, _ = self.run_worker([{'processed': 0}], [True])

        stop.wait.assert_called_once_with(12)

    def test_connection_cleanup_failure_still_retries(self):
        stop_event = Mock()
        stop_event.is_set.return_value = False
        stop_event.wait.side_effect = [False, True]
        with patch.object(worker, 'learn_prescription_drugs', return_value={'processed': 0}) as learn:
            with patch.object(worker, 'close_old_connections'):
                with patch.object(worker.connections, 'close_all', side_effect=[RuntimeError('private'), None]):
                    with self.assertLogs(worker.logger, level='WARNING') as logs:
                        worker._run_worker(stop_event)

        self.assertEqual(learn.call_count, 2)
        self.assertNotIn('private', '\n'.join(logs.output))
