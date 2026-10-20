"""Regression protection for the mobile matchday dashboard structure.

These tests intentionally use Python's standard library so they run with the
project's existing pytest/unittest setup, without requiring a browser driver.
"""
from pathlib import Path
import unittest


CSS_PATH = Path(__file__).resolve().parents[1] / 'src' / 'ui' / 'broadcast_styles.py'


class MobileDashboardLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.css = CSS_PATH.read_text(encoding='utf-8')

    def test_dashboard_mobile_override_targets_actual_panels(self):
        self.assertIn('/* Mobile scoreboard layout:', self.css)
        self.assertIn('max-width: 820px', self.css)
        # :has() avoids fragile Streamlit direct-child ancestor assumptions.
        self.assertIn(':has(.st-key-bp_fixtures):has(.st-key-bp_focus):has(.st-key-bp_sidebar)', self.css)
        self.assertIn('flex-flow: column nowrap !important', self.css)

    def test_featured_match_is_mobile_first(self):
        self.assertIn(':has(.st-key-bp_focus) {order: 0 !important;}', self.css)
        self.assertIn(':has(.st-key-bp_fixtures) {order: 1 !important;}', self.css)

    def test_nested_fixture_buttons_stack_below_full_width_club_names(self):
        self.assertIn('[class*="st-key-bp_fixture_item_"] [data-testid="stHorizontalBlock"]', self.css)
        self.assertIn('flex-direction: column !important', self.css)
        self.assertIn('grid-template-columns: 45px minmax(0,1fr) 58px !important', self.css)
        self.assertIn('width: 100% !important', self.css)

    def test_full_team_names_are_not_split_into_individual_letters(self):
        self.assertIn('word-break: normal !important', self.css)
        self.assertIn('overflow-wrap: normal !important', self.css)
        self.assertIn('.bp-focus-score strong', self.css)

    def test_monochrome_crests_still_work(self):
        self.assertIn('.bp-future-badge img.bp-club-crest', self.css)
        self.assertIn('filter:grayscale(1) saturate(0)', self.css)


if __name__ == '__main__':
    unittest.main()
