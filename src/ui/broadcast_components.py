"""Broadcast-style matchday layouts. Only renders the real prediction snapshot.

No external scores, speculative match news, player ratings or events are inferred.
The real CatBoost probabilities are used without recalibration or modification.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape as esc

import pandas as pd
from src.production.club_crests import crest_html


def _t(row: pd.Series) -> str:
    return pd.Timestamp(row['kickoff_utc']).tz_convert('Africa/Johannesburg').strftime('%H:%M')


# Club names in the supplemental future-fixture schedule do not always match
# the names in the primary prediction-provider crest registry. Keep those
# identifiers local to presentation: never rename teams in prediction data.
# Images are hosted by the existing football-data.org crest provider.
CREST_URL_OVERRIDES = {
    'lille osc': 'https://crests.football-data.org/521.png',
    'real betis': 'https://crests.football-data.org/90.png',
}


def _c(row: pd.Series, side: str) -> str:
    team = str(row[f'{side}_team']).strip()
    override = CREST_URL_OVERRIDES.get(team.casefold())
    if override:
        return (f'<img src="{override}" alt="{esc(team, quote=True)} crest" '
                'loading="lazy" referrerpolicy="no-referrer" class="bp-club-crest">')
    return crest_html(team, side=side)


def _pct(row: pd.Series, key: str) -> str:
    value = float(row[key]) * 100.0
    return f'{value:.1f}%'


def _now(now: datetime | None = None) -> pd.Timestamp:
    return pd.Timestamp(now or datetime.now(timezone.utc))


def is_upcoming(row: pd.Series, now: datetime | None = None) -> bool:
    return pd.Timestamp(row['kickoff_utc']) > _now(now)


def banner_html(rows: pd.DataFrame, feature: pd.Series | None = None) -> str:
    if rows.empty:
        return ('<section class="bp-hero bp-hero-empty"><div class="bp-eyebrow">THE MATCHDAY HUB</div>'
                '<h1>THE NIGHT <em>IS OURS.</em></h1><p>Next matchday will appear when the snapshot is ready.</p></section>')
    row = feature if feature is not None else rows.iloc[0]
    day = pd.Timestamp(row['kickoff_utc']).tz_convert('Africa/Johannesburg').strftime('%a, %d %b %Y')
    odds = probability_triplet_html(row)
    return (f'<section class="bp-hero"><div class="bp-hero-content">'
            f'<div class="bp-eyebrow">EUROPEAN FOOTBALL NIGHTS · {esc(day)}</div>'
            '<h1>THE NIGHT<br><em>IS OURS.</em></h1>'
            '<p>Every match has a story. Every forecast has a receipt. '
            'Make your call before kickoff.</p></div>'
            '<div class="bp-hero-feature"><div class="bp-small-label">FEATURED FIXTURE</div>'
            f'<div class="bp-feature-teams"><div>{_c(row,"home")}<strong>{esc(str(row["home_team"]))}</strong></div>'
            f'<div class="bp-feature-time"><b>{_t(row)}</b><small>SAST</small></div>'
            f'<div>{_c(row,"away")}<strong>{esc(str(row["away_team"]))}</strong></div></div>'
            f'{odds}<div class="bp-feature-caption">PRE-MATCH MODEL FORECAST · FROZEN BEFORE KICKOFF</div>'
            '</div></section>')


def probability_triplet_html(row: pd.Series) -> str:
    return (f'<div class="bp-prob-triplet" aria-label="Home {_pct(row,"home_probability")}, '
            f'Draw {_pct(row,"draw_probability")}, Away {_pct(row,"away_probability")}">'
            f'<span><b>{_pct(row,"home_probability")}</b><small>Home</small></span>'
            f'<span><b>{_pct(row,"draw_probability")}</b><small>Draw</small></span>'
            f'<span><b>{_pct(row,"away_probability")}</b><small>Away</small></span></div>')


def compact_fixture_html(row: pd.Series, *, selected: bool = False, now: datetime | None = None) -> str:
    chosen = ' selected' if selected else ''
    match_phase = 'Upcoming' if is_upcoming(row, now) else 'Kickoff passed'
    return (f'<div class="bp-fixture-row{chosen}">'
            f'<div class="bp-fixture-time">{_t(row)}</div>'
            f'<div class="bp-fixture-clubs"><div>{_c(row,"home")}<span>{esc(str(row["home_team"]))}</span></div>'
            f'<div>{_c(row,"away")}<span>{esc(str(row["away_team"]))}</span></div></div>'
            f'<div class="bp-fixture-prob">{_pct(row,"max_probability")}</div>'
            f'<span class="bp-sr-only">{esc(match_phase)} · probability of most likely outcome</span></div>')


def featured_panel_header_html(row: pd.Series, now: datetime | None = None) -> str:
    """Stable match header shared by all three genuinely interactive Streamlit tabs."""
    phase = 'UPCOMING' if is_upcoming(row, now) else 'KICKOFF PASSED · SCORE UNAVAILABLE'
    return (f'<div class="bp-panel-head"><h2>Featured match</h2>'
            f'<span>{esc(phase)}</span></div>'
            f'<div class="bp-focus-score"><div>{_c(row,"home")}<strong>{esc(str(row["home_team"]))}</strong></div>'
            f'<div class="bp-focus-middle"><b>{_t(row)}</b><small>SAST · MATCH PREVIEW</small></div>'
            f'<div>{_c(row,"away")}<strong>{esc(str(row["away_team"]))}</strong></div></div>')


def featured_panel_body_html(row: pd.Series, tab: str = 'Preview') -> str:
    """One view at a time. No synthetic live data or inferred evidence strength."""
    if tab not in {'Preview', 'Prediction', 'Evidence'}:
        raise ValueError(f'Unknown featured tab: {tab}')

    predicted = esc(str(row['predicted_selection']))
    score = esc(str(row['predicted_scoreline']))
    if tab == 'Preview':
        return ('<div class="bp-focus-photo"><span>THE ROAD TO KICKOFF</span></div>'
                '<div class="bp-focus-body"><div class="bp-small-label">PRE-MATCH OVERVIEW</div>'
                f'<h3>{esc(str(row["home_team"]))} vs {esc(str(row["away_team"]))}</h3>'
                '<p>Explore the original forecast before kickoff. Follow the match in '
                'Match Centre when verified score data becomes available.</p>'
                f'<div class="bp-spotlight-line"><span>Model pick</span><b>{predicted}</b></div>'
                f'<div class="bp-projected">Projected score <strong>{score}</strong></div></div>')
    if tab == 'Prediction':
        return ('<div class="bp-focus-body bp-feature-tab-body">'
                '<div class="bp-small-label">RESULT PROBABILITIES</div>'
                f'<h3>{predicted}</h3>'
                '<p>Home, draw and away probabilities from the frozen pre-match forecast.</p>'
                f'{probability_triplet_html(row)}'
                f'<div class="bp-projected">Projected score <strong>{score}</strong></div>'
                '<p class="bp-feature-footnote">The displayed scoreline is an auxiliary estimate, '
                'not a separately trained goals model. Forecasts do not change during play.</p></div>')

    confidence = str(row.get('confidence_label', '')).strip()
    strength = esc(confidence.upper()) if confidence and confidence.lower() != 'nan' else 'NOT AVAILABLE'
    coverage = row.get('overall_feature_availability')
    coverage_display = 'NOT AVAILABLE'
    if coverage is not None and pd.notna(coverage):
        try:
            coverage_display = f'{max(0., min(100., float(coverage) * 100)):.1f}%'
        except (TypeError, ValueError, OverflowError):
            pass
    return ('<div class="bp-focus-body bp-feature-tab-body">'
            '<div class="bp-small-label">FORECAST EVIDENCE</div>'
            '<h3>What supports this forecast?</h3>'
            '<p>Evidence strength describes the supporting information; it is not the '
            'chance that the prediction will be correct.</p>'
            f'<div class="bp-spotlight-line"><span>Evidence strength</span><b>{strength}</b></div>'
            f'<div class="bp-spotlight-line"><span>Input availability</span><b>{coverage_display}</b></div>'
            '<p class="bp-feature-footnote">Detailed provenance and verified results are '
            'available in the Evidence Vault.</p></div>')


def featured_panel_html(row: pd.Series, now: datetime | None = None) -> str:
    """Backwards-compatible read-only preview; no fake navigation tabs."""
    return featured_panel_header_html(row, now) + featured_panel_body_html(row, 'Preview')


def future_fixtures_html(rows: pd.DataFrame) -> str:
    """Compact next-matchday rows. Schedule data only, never fake scores/picks.

    Every row contains two predictable club cells and one match-time cell.
    Crest markup is contained so an upstream crest wrapper cannot stretch
    either the layout or the scrollable fixture panel.
    """
    count = len(rows)
    heading = ('<div class="bp-side-panel bp-future-panel">'
               '<div class="bp-panel-head"><h2>Future fixtures</h2>'
               f'<span>{count} scheduled</span></div>')
    if rows.empty:
        return (heading + '<div class="bp-future-empty">The next fixture schedule '
                'will appear when verified dates are available.</div></div>')

    day = pd.Timestamp(rows.iloc[0]['kickoff_utc']).tz_convert('Africa/Johannesburg')
    out = [heading,
           '<p class="bp-future-date">' + esc(day.strftime('%A, %d %B %Y')) +
           ' · SOUTH AFRICA TIME</p>',
           '<div class="bp-future-list" role="region" aria-label="Upcoming fixtures" tabindex="0">',
           '<table class="bp-future-table"><tbody>']
    for _, row in rows.iterrows():
        home = esc(str(row['home_team']))
        away = esc(str(row['away_team']))
        out.append(
            '<tr class="bp-future-match">'
            '<td class="bp-future-match-clubs">'
            '<div class="bp-future-club"><span class="bp-future-badge">'
            f'{_c(row,"home")}</span><span class="bp-future-name">{home}</span></div>'
            '<div class="bp-future-club"><span class="bp-future-badge">'
            f'{_c(row,"away")}</span><span class="bp-future-name">{away}</span></div>'
            '</td>'
            '<td class="bp-future-match-meta">'
            f'<time>{_t(row)}</time>'
            '<small>FORECAST PENDING</small></td></tr>'
        )
    out.append('</tbody></table></div>'
               '<p class="bp-future-footnote">Schedule only. Fixtures may change; '
               'no forecast has been issued for these matches.</p></div>')
    return ''.join(out)


def strongest_picks_html(rows: pd.DataFrame) -> str:
    """Top actual model leanings, not player form or betting odds."""
    if rows.empty:
        return ''
    top = rows.sort_values('max_probability', ascending=False).head(4)
    out = ['<div class="bp-side-panel bp-strongest"><div class="bp-panel-head">'
           '<h2>Model watchlist</h2><span>PRE-MATCH</span></div>'
           '<p>Highest predicted result probabilities in this snapshot.</p>']
    for rank, (_,row) in enumerate(top.iterrows(),1):
        out.append(f'<div class="bp-strength-row"><span>{rank:02d}</span>'
                   f'<div><b>{esc(str(row["predicted_selection"]))}</b>'
                   f'<small>{esc(str(row["home_team"]))} vs {esc(str(row["away_team"]))}</small></div>'
                   f'<strong>{_pct(row,"max_probability")}</strong></div>')
    out.append('</div>')
    return ''.join(out)


def matchday_toolbar_html(rows: pd.DataFrame) -> str:
    if rows.empty:
        return '<div class="bp-toolbar"><b>MATCHDAY</b><span>Fixture data unavailable</span></div>'
    start = pd.Timestamp(rows['kickoff_utc'].min()).tz_convert('Africa/Johannesburg')
    date = start.strftime('%a, %d %b %Y')
    return (f'<div class="bp-toolbar"><span class="bp-toolbar-active">UEFA CHAMPIONS LEAGUE</span>'
            f'<span>{esc(date)}</span><span>{len(rows)} matches</span>'
            '<span>Pre-match analysis</span></div>')
