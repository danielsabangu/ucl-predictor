"""Presentation contracts for Stage 3. No external scores or Streamlit server needed."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ui.broadcast_components import (
    banner_html, compact_fixture_html, featured_panel_html,
    featured_panel_header_html, featured_panel_body_html,
    strongest_picks_html, matchday_toolbar_html,
)
from src.ui.broadcast_identity import brand_icon_html, brand_lockup_html, stadium_data_uri
from src.ui.broadcast_styles import BROADCAST_CSS


def _rows():
    return pd.DataFrame([
        dict(kickoff_utc=pd.Timestamp('2026-10-20T16:45:00Z'),
             home_team='Fenerbahçe', away_team='Slavia Prague',
             home_probability=.5766609, draw_probability=.1855513,
             away_probability=.2377878, max_probability=.5766609,
             predicted_scoreline='2-1', predicted_selection='Fenerbahçe'),
        dict(kickoff_utc=pd.Timestamp('2026-10-20T19:00:00Z'),
             home_team='Liverpool', away_team='Villarreal',
             home_probability=.6098, draw_probability=.2102,
             away_probability=.18, max_probability=.6098,
             predicted_scoreline='2-1', predicted_selection='Liverpool')
    ])


def test_broadcast_is_about_real_matchday_not_fiction():
    rows = _rows()
    html = banner_html(rows, feature=rows.iloc[0])
    assert 'Fenerbahçe' in html
    assert 'Slavia Prague' in html
    assert '57.7%' in html
    assert '18:45' in html
    assert 'LIVE' not in html
    assert 'PRE-MATCH MODEL FORECAST' in html


def test_sidebar_uses_real_probabilities_and_escapes():
    rows = _rows()
    body = strongest_picks_html(rows)
    assert body.index('Liverpool') < body.index('Fenerbahçe')
    assert '60.98%' not in body # rounded to one decimal
    assert '61.0%' in body
    rows.loc[0, 'predicted_selection'] = '<script>bad</script>'
    assert '&lt;script&gt;' in strongest_picks_html(rows)


def test_fixture_status_is_not_faked():
    row = _rows().iloc[0]
    html = compact_fixture_html(row, now=datetime(2026,10,20,16,0,tzinfo=timezone.utc))
    assert 'Upcoming' in html
    html = compact_fixture_html(row, now=datetime(2026,10,20,19,0,tzinfo=timezone.utc))
    assert 'Kickoff passed' in html
    assert 'LIVE 90' not in html


def test_brand_assets_and_markup():
    assert 'viewBox="0 0 60 60"' in brand_icon_html()
    assert 'PREDICTOR' in brand_lockup_html()
    uri = stadium_data_uri()
    assert uri.startswith('data:image/webp;base64,')
    # The image variable now has a safe CSS fallback; the old no-fallback
    # reference is deliberately absent after the stadium visibility fix.
    assert 'var(--bp-stadium-image,none)' in BROADCAST_CSS
    assert '--bp-stadium-image:none' not in BROADCAST_CSS


def test_public_diagnostics_are_gated():
    src = (Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8')
    assert 'if CREATOR_MODE:' in src
    assert 'if CREATOR_MODE and health[' in src
    assert 'render_html(feed_notice_html())' not in src
    assert 'os.environ.get("UCL_CREATOR_MODE")' in src


def test_featured_tabs_have_distinct_real_content():
    row = _rows().iloc[0]
    top = featured_panel_header_html(row)
    assert 'Fenerbahçe' in top and 'Slavia Prague' in top
    preview = featured_panel_body_html(row, 'Preview')
    prediction = featured_panel_body_html(row, 'Prediction')
    evidence = featured_panel_body_html(row, 'Evidence')
    assert 'THE ROAD TO KICKOFF' in preview
    assert 'RESULT PROBABILITIES' in prediction
    assert '57.7%' in prediction
    assert 'Evidence strength' in evidence
    assert 'NOT AVAILABLE' in evidence  # not fabricated for sparse test row
    assert 'bp-panel-tabs' not in featured_panel_html(row)


def test_featured_evidence_uses_actual_evidence_fields():
    row = _rows().iloc[0].copy()
    row['confidence_label'] = 'Moderate'
    row['overall_feature_availability'] = 0.5946
    body = featured_panel_body_html(row, 'Evidence')
    assert 'MODERATE' in body
    assert '59.5%' in body
    assert '57.7%' not in body


def test_equal_home_actions_and_real_streamlit_tabs():
    src = (Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8')
    assert 'st.container(key="bp_hero_actions")' in src
    assert 'st.columns(2, gap="small")' in src
    assert 'st.container(key="bp_feature_tabs")' in src
    assert 'key=f"bp_tab_{tab_name.lower()}"' in src
    assert 'featured_panel_body_html(featured, active_feature_tab)' in src
    assert '"⚽  FIXTURES"' not in src
    css = (Path(__file__).resolve().parents[1] / 'src/ui/broadcast_styles.py').read_text(encoding='utf-8')
    assert 'flex:1 1 0!important' in css
    assert '.bp-fixture-clubs span{' in css
    assert 'font-size:12px;font-weight:750' in css


def test_stadium_background_variable_not_reset_by_broadcast_styles():
    # app.py injects the bundled artwork into this CSS custom property.
    # Declaring the property again in BROADCAST_CSS previously reset it to none.
    assert '--bp-stadium-image:none' not in BROADCAST_CSS
    assert 'var(--bp-stadium-image,none)' in BROADCAST_CSS
    app_source = (Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8')
    assert 'stadium_data_uri()' in app_source
    assert 'apply_broadcast_styles(render_html, theme=CURRENT_THEME)' in app_source
