"""Verify that the future-fixture crest names resolve to real club images."""
from __future__ import annotations

import pandas as pd

from src.ui.broadcast_components import future_fixtures_html, CREST_URL_OVERRIDES


def test_lille_and_betis_future_crests():
    fixtures = pd.DataFrame([
        {"kickoff_utc": pd.Timestamp("2026-10-21T16:45:00Z"),
         "home_team": "Lille OSC", "away_team": "Galatasaray"},
        {"kickoff_utc": pd.Timestamp("2026-10-21T19:00:00Z"),
         "home_team": "Real Betis", "away_team": "Feyenoord"},
    ])
    html = future_fixtures_html(fixtures)
    assert 'crests.football-data.org/521.png' in html
    assert 'crests.football-data.org/90.png' in html
    assert 'alt="Lille OSC crest"' in html
    assert 'alt="Real Betis crest"' in html
    assert 'FORECAST PENDING' in html


def test_override_only_for_two_known_clubs():
    assert set(CREST_URL_OVERRIDES) == {"lille osc", "real betis"}
