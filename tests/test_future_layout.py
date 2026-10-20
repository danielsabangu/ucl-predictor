"""Regression checks for compact Future Fixtures team/crest alignment."""
from __future__ import annotations

import pandas as pd

from src.ui.broadcast_components import future_fixtures_html
from src.ui.broadcast_styles import BROADCAST_CSS, BROADCAST_LIGHT_CSS


def _sample_fixtures():
    return pd.DataFrame([
        {'kickoff_utc': pd.Timestamp('2026-10-21T16:45:00Z'),
         'home_team': 'Lille OSC', 'away_team': 'Galatasaray'},
        {'kickoff_utc': pd.Timestamp('2026-10-21T19:00:00Z'),
         'home_team': 'Real Betis', 'away_team': 'Feyenoord'},
    ])


def test_every_fixture_is_one_two_team_table_row():
    html = future_fixtures_html(_sample_fixtures())
    assert html.count('<tr class="bp-future-match">') == 2
    assert html.count('class="bp-future-club"') == 4
    assert html.count('class="bp-future-badge"') == 4
    assert html.count('class="bp-future-match-meta"') == 2
    assert html.count('FORECAST PENDING') == 2
    assert 'bp-future-list' in html
    assert 'crests.football-data.org/521.png' in html
    assert 'crests.football-data.org/90.png' in html


def test_no_fake_scores_or_forecasts_in_future_schedule():
    html = future_fixtures_html(_sample_fixtures())
    assert '18:45' in html and '21:00' in html
    assert '57.7%' not in html and 'LIVE' not in html


def test_fixed_crest_slot_and_table_layout():
    assert '.bp-future-table{table-layout:fixed;' in BROADCAST_CSS
    assert '.bp-future-badge{display:flex!important;' in BROADCAST_CSS
    assert '.bp-future-match-meta{width:108px!important;' in BROADCAST_CSS
    assert '.bp-future-name{color:#243e5b!important}' in BROADCAST_LIGHT_CSS
