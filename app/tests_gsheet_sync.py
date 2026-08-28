import os
from pathlib import Path
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings
from gspread.http_client import HTTPClient

from app import gsheet_sync


class BoundedGoogleSheetClientTests(SimpleTestCase):
    def test_client_applies_request_timeout_with_shared_budget(self):
        class FakeHTTPClient:
            def __init__(self, auth, session=None):
                self.auth = auth
                self.session = session
                self.timeout = None

            def request(self, *args, **kwargs):
                return self.timeout

        with patch.object(gsheet_sync.time, 'monotonic', side_effect=[100, 105]):
            client_class = gsheet_sync._bounded_http_client(FakeHTTPClient)
            client = client_class('credentials')
            timeout = client.request('GET', 'https://example.test')

        self.assertEqual(timeout, gsheet_sync._GSHEET_REQUEST_TIMEOUT_SECONDS)

    def test_client_rejects_calls_after_total_network_budget(self):
        base_request = Mock()

        class FakeHTTPClient:
            def __init__(self, auth, session=None):
                self.timeout = None

            def request(self, *args, **kwargs):
                return base_request(*args, **kwargs)

        deadline = 100 + gsheet_sync._GSHEET_NETWORK_BUDGET_SECONDS
        with patch.object(
            gsheet_sync.time,
            'monotonic',
            side_effect=[100, deadline + 0.01],
        ):
            client_class = gsheet_sync._bounded_http_client(FakeHTTPClient)
            client = client_class('credentials')
            with self.assertRaisesRegex(TimeoutError, 'network budget expired'):
                client.request('GET', 'https://example.test')

        base_request.assert_not_called()

    @override_settings(GOOGLE_SHEETS_SYNC_ENABLED=True)
    def test_service_account_is_created_with_bounded_http_client(self):
        spreadsheet = object()
        client = Mock()
        client.open_by_key.return_value = spreadsheet

        base_dir = Path('C:/pharmacy-gsheet-test')
        credentials_path = base_dir / 'credentials.json'
        with (
            patch.object(gsheet_sync, 'BASE_DIR', base_dir),
            patch.object(Path, 'exists', return_value=True),
            patch.dict(
                os.environ,
                {
                    'GSHEET_SPREADSHEET_ID': 'a' * 24,
                    'GSHEET_CREDENTIALS_FILE': credentials_path.name,
                },
                clear=False,
            ),
            patch('gspread.service_account', return_value=client) as service_account,
        ):
            result = gsheet_sync.get_spreadsheet()

        self.assertIs(result, spreadsheet)
        kwargs = service_account.call_args.kwargs
        self.assertEqual(kwargs['filename'], str(credentials_path))
        self.assertEqual(kwargs['http_client'].__name__, 'BoundedHTTPClient')
        self.assertTrue(issubclass(kwargs['http_client'], HTTPClient))
        client.open_by_key.assert_called_once_with('a' * 24)
