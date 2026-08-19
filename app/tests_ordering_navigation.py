from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from app.models import OrderingSheetEntry


class OrderingNavigationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='ordering-navigation-user',
            password='test-password',
        )
        self.client.force_login(self.user)
        self.url = reverse('ordering_sheet')

    def test_full_page_keeps_primary_navigation_without_lifecycle_rail(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<nav class="app-nav" aria-label="Primary navigation">',
            count=1,
        )
        self.assertNotContains(response, 'class="os-view-tabs"')
        self.assertNotContains(response, 'Ordering lifecycle views')
        self.assertContains(response, 'aria-label="Filter by status"', count=1)

    def test_embed_remains_navless_and_uses_the_table_filters(self):
        response = self.client.get(f'{self.url}?embed=1')

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'class="app-nav"')
        self.assertNotContains(response, 'class="os-view-tabs"')
        self.assertContains(response, 'aria-label="Filter by status"', count=1)
        self.assertEqual(response.headers['X-Frame-Options'], 'SAMEORIGIN')

    def test_direct_completed_history_view_remains_supported(self):
        completed = OrderingSheetEntry.objects.create(
            name='Completed history item',
            initials='AB',
            status=OrderingSheetEntry.STATUS_PICKED_UP,
            created_by=self.user,
        )
        OrderingSheetEntry.objects.create(
            name='Active item',
            initials='AB',
            status=OrderingSheetEntry.STATUS_PENDING,
            created_by=self.user,
        )

        response = self.client.get(f'{self.url}?view=completed')

        self.assertContains(response, completed.name)
        self.assertNotContains(response, 'Active item')

    def test_not_for_sale_remains_available_to_its_table_filter(self):
        entry = OrderingSheetEntry.objects.create(
            name='Pharmacist review item',
            initials='AB',
            status=OrderingSheetEntry.STATUS_NOT_FOR_SALE,
            created_by=self.user,
        )

        response = self.client.get(self.url)

        self.assertContains(response, entry.name)
        self.assertContains(response, 'data-status="not_for_sale"')


class PrimaryNavigationScopeTests(SimpleTestCase):
    def test_primary_dock_styles_and_scripts_are_scoped_to_app_nav(self):
        root = Path(settings.BASE_DIR)
        base = (root / 'app' / 'templates' / 'base.html').read_text(encoding='utf-8')
        styles = (root / 'static' / 'css' / 'ui-system.css').read_text(encoding='utf-8')
        script = (root / 'static' / 'js' / 'ui-system.js').read_text(encoding='utf-8')
        labels = (root / 'app' / 'templates' / 'label_printing.html').read_text(encoding='utf-8')

        self.assertIn('<nav class="app-nav" aria-label="Primary navigation">', base)
        self.assertIn("document.querySelector('.app-nav')", base)
        self.assertNotRegex(base, r'(?m)^\s*nav(?:\s*\{|:hover|\.nav-force)')
        self.assertNotIn("document.querySelector('nav')", base)
        self.assertNotIn('body.app-shell nav', styles)
        self.assertIn('body.app-shell .app-nav', styles)
        self.assertIn("document.querySelector('.app-nav .nav-content')", script)
        self.assertEqual(labels.count("document.querySelector('.app-nav')"), 2)
        self.assertNotIn("document.querySelector('nav')", labels)
