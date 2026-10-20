"""Stage 5 schedule and navigation contracts, isolated from the prediction model."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ui.future_fixtures import load_future_fixtures
from src.ui.broadcast_components import future_fixtures_html
from src.ui.broadcast_styles import BROADCAST_CSS

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / 'data/app/future_fixtures.csv'


def _now():
    return datetime(2026, 10, 20, 13, 0, tzinfo=timezone.utc)


def test_tomorrow_schedule_only_contains_future_games():
    rows = load_future_fixtures(
        SCHEDULE, after_matchday=pd.Timestamp('2026-10-20T19:00:00Z'), now=_now()
    )
    assert len(rows) == 9
    assert 'Bayern Munich' in rows['home_team'].to_list()
    assert all(x.date().isoformat() == '2026-10-21'
               for x in rows['kickoff_utc'].dt.tz_convert('Africa/Johannesburg'))


def test_no_repeat_or_stale_future_games():
    rows = load_future_fixtures(
        SCHEDULE, after_matchday=pd.Timestamp('2026-10-21T19:00:00Z'), now=_now()
    )
    assert rows.empty
    after = load_future_fixtures(
        SCHEDULE,
        after_matchday=pd.Timestamp('2026-10-20T19:00:00Z'),
        now=datetime(2026, 10, 22, 0, 0, tzinfo=timezone.utc),
    )
    assert after.empty


def test_schedule_missing_is_honest(tmp_path):
    rows = load_future_fixtures(
        tmp_path/'missing.csv',
        after_matchday=pd.Timestamp('2026-10-20T19:00:00Z'),
        now=_now(),
    )
    assert rows.empty
    rendered = future_fixtures_html(rows)
    assert 'verified dates' in rendered
    assert 'PRE-MATCH' not in rendered


def test_future_cards_contain_no_model_probabilities():
    rows = load_future_fixtures(
        SCHEDULE, after_matchday=pd.Timestamp('2026-10-20T19:00:00Z'), now=_now()
    )
    rendered = future_fixtures_html(rows)
    assert 'Future fixtures' in rendered
    assert 'FORECAST PENDING' in rendered
    assert 'Bayern Munich' in rendered and 'Arsenal' in rendered
    assert '18:45' in rendered and '21:00' in rendered
    assert '57.7%' not in rendered
    assert 'Other key matches' not in rendered


def test_dashboard_actions_have_equal_css_and_real_fixture_navigation():
    assert 'Native st.columns(2) controls 50/50 width' in BROADCAST_CSS
    assert 'max-height:52px!important' in BROADCAST_CSS
    assert 'VIEW MATCH' in (ROOT/'app.py').read_text(encoding='utf-8')
    assert 'st.session_state["matchday_selected_id"] = choice_id' in (ROOT/'app.py').read_text(encoding='utf-8')
    assert 'render_html(future_fixtures_html(future_rows))' in (ROOT/'app.py').read_text(encoding='utf-8')
    assert 'strongest_picks_html(board)' in (ROOT/'app.py').read_text(encoding='utf-8')


def test_home_no_forced_streamlit_layout_or_double_print_button():
    app = (ROOT/'app.py').read_text(encoding='utf-8')
    css = BROADCAST_CSS
    home = app.split('if current_screen == "home":', 1)[1].split('elif current_screen == "live":', 1)[0]
    assert home.count('key="nf_home_print"') == 1
    assert home.count('key="nf_home_live"') == 1
    assert 'st.columns(2, gap="small")' in home
    assert 'st.columns([2.25, 1.25], gap="small")' in home
    css_actions = css.split('/* Equal Home actions:')[1].split('/* Maintain a stable compact fixture')[0]
    assert 'grid-template-columns' not in css_actions
    assert '[data-testid="stHorizontalBlock"]' not in css_actions
    assert '[data-testid="stColumn"]' not in css_actions
