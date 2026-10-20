"""Presentation-only matchday tests. No network or Streamlit server required."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.ui.matchday import (
    MatchScore,
    empty_pitch_html,
    fixture_card_html,
    forecast_vs_reality_html,
    matchday_hero_html,
    scoreboard_html,
)


@pytest.fixture
def fixture():
    return pd.Series({
        "kickoff_utc": pd.Timestamp("2026-10-20T16:45:00Z"),
        "home_team": "Fenerbahçe",
        "away_team": "Slavia Prague",
        "predicted_selection": "Fenerbahçe",
        "predicted_scoreline": "2-1",
        "home_probability": .5766609,
        "draw_probability": .1855513,
        "away_probability": .2377878,
    })


def test_hero_uses_real_snapshot_data(fixture):
    html = matchday_hero_html(
        pd.DataFrame([fixture]),
        now=datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc),
    )
    assert "Fenerbahçe" in html
    assert "Slavia Prague" in html
    assert "18:45 SAST" in html
    assert "01 MATCHES ON THE BOARD" in html


def test_unconnected_scoreboard_never_shows_a_live_score(fixture):
    html = scoreboard_html(
        fixture,
        now=datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc),
    )
    assert "– : –" in html
    assert "SCORE FEED NOT CONNECTED" in html
    assert "KICKOFF PASSED" in html
    assert "2 : 1" not in html


def test_actual_scores_only_when_explicitly_passed(fixture):
    html = scoreboard_html(
        fixture,
        score=MatchScore(home_goals=2, away_goals=1, status="LIVE", minute=67),
    )
    assert "2 : 1" in html
    assert "LIVE · 67′" in html


def test_forecast_probabilities_are_pre_match_not_live(fixture):
    html = forecast_vs_reality_html(fixture)
    assert "HOME 57.7%" in html
    assert "DRAW 18.6%" in html
    assert "AWAY 23.8%" in html
    assert "NOT AVAILABLE" in html
    assert "not live win probabilities" in html


def test_blocked_forecast_does_not_show_prediction(fixture):
    html = forecast_vs_reality_html(fixture, forecasts_available=False)
    assert "temporarily unavailable" in html
    assert "57.7%" not in html


def test_pitch_does_not_claim_tracking():
    html = empty_pitch_html()
    assert "MATCH TRACKING UNAVAILABLE" in html
    assert "No invented passes" in html


def test_score_must_be_consistent():
    with pytest.raises(ValueError):
        MatchScore(home_goals=1)
    with pytest.raises(ValueError):
        MatchScore(home_goals=-1, away_goals=1)


def test_team_names_escaped_in_text(fixture):
    fixture["home_team"] = '<script>alert("x")</script>'
    html = fixture_card_html(fixture)
    assert "&lt;script&gt;" in html
