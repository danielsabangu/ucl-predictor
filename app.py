from __future__ import annotations

import textwrap
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.ui.realtime_match_scores import (
    fetch_champions_league_events, match_fixture_score,
)

from src.production.app_health import (
    evaluate_app_health,
    filter_upcoming_predictions,
)

from src.production.app_sources import (
    resolve_app_data_source,
)

from src.production.receipt_exports import (
    build_receipt_pdf,
    build_receipt_png,
    export_filename,
)

from src.ui.formatters import (
    LOCAL_TZ,
    confidence_explanation,
    display_confidence_label,
    display_profile,
    escape,
    format_health_finish_time,
    format_next_kickoff_local,
    health_css_class,
    match_identifier,
    percentage,
)


from src.ui.v2_components import (
    build_fixture_choice_html,
    build_forecast_presentation,
    build_idle_machine_html,
)


from src.ui.evidence_vault import (
    build_vault_card_html,
    build_vault_detail_html,
    filter_evaluations,
    find_evaluation,
    load_evidence_vault,
    receipt_asset_path,
)

from src.ui.startup_loader import (
    show_startup_loading_screen,
)

from src.ui.v2_styles import (
    apply_v2_styles,
)

from src.ui.matchday import (
    MatchScore,
    matchday_hero_html,
    matchday_section_heading,
    fixture_card_html,
    scoreboard_html,
    forecast_vs_reality_html,
    empty_pitch_html,
    feed_notice_html,
)

from src.ui.matchday_styles import (
    apply_matchday_styles,
)

from src.ui.nightfall_styles import (
    apply_nightfall_styles,
)
from src.ui.broadcast_styles import apply_broadcast_styles
from src.ui.broadcast_identity import brand_lockup_html, stadium_data_uri
from src.ui.broadcast_components import (
    banner_html, compact_fixture_html, featured_panel_header_html,
    featured_panel_body_html, future_fixtures_html, strongest_picks_html,
    matchday_toolbar_html,
)
from src.ui.future_fixtures import load_future_fixtures
from pathlib import Path


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="UCL Predictor | Matchday",
    page_icon=":material/sports_soccer:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# APPLICATION SOURCE
# ============================================================

try:

    APP_SOURCE = (
        resolve_app_data_source()
    )

except Exception as error:

    st.error(
        "UCL Predictor could not resolve "
        "an application data source."
    )

    st.code(
        str(error)
    )

    st.stop()


PREDICTIONS_FILE = (
    APP_SOURCE.predictions_file
)

STATUS_FILE = (
    APP_SOURCE.status_file
)


# ============================================================
# CREATOR VIEW
# ============================================================
#
# Local/live development keeps operational diagnostics visible
# to the creator. Public snapshot mode hides data-age, pipeline,
# source and run information from consumers.
# ============================================================

# Creator diagnostics are deliberately opt-in and local-development only.
# Never infer creator privileges from a non-public data-source flag, which
# may still be used by an internet-facing deployment.
CREATOR_MODE = (os.environ.get("UCL_CREATOR_MODE") == "1"
                and not APP_SOURCE.is_public_snapshot)

# ============================================================
# HTML
# ============================================================


def render_html(
    markup: str,
) -> None:

    cleaned = textwrap.dedent(
        markup
    ).strip()

    if hasattr(
        st,
        "html",
    ):

        st.html(
            cleaned
        )

    else:

        st.markdown(
            cleaned,
            unsafe_allow_html=True,
        )


# ============================================================
# CSS
# ============================================================

# ============================================================
# NIGHTFALL — DARK ONLY
# ============================================================
# Legacy sessions may retain a previous light-mode choice. Always reset it.
st.session_state["ui_theme"] = "dark"
CURRENT_THEME = "dark"


# ============================================================
# V2 VISUAL SYSTEM
# ============================================================

apply_v2_styles(
    render_html,
    theme=CURRENT_THEME,
)

# Additional styling is scoped to the two new football screens.
apply_matchday_styles(render_html)
# Nightfall palette/typography overrides the old industrial presentation.
# Existing thermal receipt, probabilities and exports are untouched.
apply_nightfall_styles(render_html, theme=CURRENT_THEME)
# Broadcast theme is the final layer, over the compatible Stage 2 system.
from html import escape as _html_escape
_stadium_uri = stadium_data_uri()
if _stadium_uri:
    render_html('<style>:root{--bp-stadium-image:url("' + _html_escape(_stadium_uri, quote=True) + '")}</style>')
apply_broadcast_styles(render_html, theme=CURRENT_THEME)


# ============================================================
# REALTIME SPORTS API — PUBLIC LIVE SCOREBOARD EMBED
# ============================================================
# Official free widget: no API credentials, server-side polling or scraping.
# This is a separately hosted scoreboard. It does NOT turn score=None below
# into a verified native score and it never modifies the frozen predictions.
# Source: https://www.realtimesportsapi.com/widgets


def render_ucl_live_scoreboard() -> None:
    """Show the provider's free, automatically refreshed Champions League widget."""
    render_html(
        """
        <section class="bp-live-widget-header" aria-label="Champions League live scores">
          <div class="bp-live-widget-kicker">UEFA CHAMPIONS LEAGUE · SCOREBOARD</div>
          <div class="bp-live-widget-heading">MATCHES AS THEY HAPPEN</div>
          <p>Scores from Realtime Sports API. The feed refreshes around every
          60 seconds and may be delayed; this is not an in-play prediction.</p>
        </section>
        <style>
          .bp-live-widget-header{border:1px solid #315579;border-radius:10px 10px 0 0;
            background:linear-gradient(120deg,#0a1f37,#123050);padding:18px 20px 13px;
            margin:12px 0 0;color:#e8f4ff}
          .bp-live-widget-kicker{font-size:10px;letter-spacing:.11em;font-weight:800;
            color:#85beff;margin-bottom:6px}
          .bp-live-widget-heading{font-family:Sora,Manrope,sans-serif;
            font-size:clamp(17px,3vw,23px);font-weight:800;letter-spacing:-.04em}
          .bp-live-widget-header p{color:#b0c8e1;font-size:12px;line-height:1.5;
            margin:8px 0 0}
        </style>
        """
    )
    # Official script-tag widget creates an iframe and refreshes it while visible.
    # The outer iframe is a Streamlit component; content comes solely from the
    # provider's origin and never needs our football-data API key.
    components.html(
        """<!doctype html>
        <html lang="en"><head>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            html,body{margin:0;padding:0;background:#091b30;color:#e8f4ff;
              color-scheme:dark;font-family:system-ui, sans-serif;overflow-x:hidden}
            .rsa-scoreboard{width:100%;max-width:100%;min-height:590px;
              box-sizing:border-box;overflow:hidden}
            .rsa-scoreboard a{display:inline-block;margin:20px;color:#a5d2ff}
          </style>
        </head><body>
          <div class="rsa-scoreboard" data-league="champions-league"
               data-width="100%" data-height="600">
            <a href="https://www.realtimesportsapi.com/embed/scoreboard/champions-league"
               target="_blank" rel="noopener noreferrer">
              Open Champions League scores at Realtime Sports API
            </a>
          </div>
          <script src="https://www.realtimesportsapi.com/widget.js" async></script>
        </body></html>""",
        height=625,
        scrolling=True,
    )
    st.caption(
        "Third-party scoreboard by Realtime Sports API. "
        "Scores may be delayed. Our original pre-match forecasts never update during play. "
        "If the widget does not load on your device, "
        "[open the scoreboard directly](https://www.realtimesportsapi.com/embed/scoreboard/champions-league)."
    )


# ============================================================
# VERIFIED NATIVE MATCH SCORES — OPTIONAL AUTHENTICATED API
# ============================================================
# Only the REST API can populate an individual selected match. The free embed
# above remains fully independent. Requests are server-side and cached across
# Streamlit sessions; no API key is ever included in client markup.
# Free quota after initial trial: 125 calls/month. Ten-minute refresh is a
# deliberately conservative default, NOT 20-second live polling.


def _realtime_api_key() -> str:
    key = (os.environ.get("RSA_API_KEY") or
           os.environ.get("REALTIME_SPORTS_API_KEY") or "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("RSA_API_KEY") or
                   st.secrets.get("REALTIME_SPORTS_API_KEY") or "").strip()
    except Exception:
        return ""


@st.cache_data(ttl=600, show_spinner=False)
def _cached_realtime_scoreboard(api_key: str) -> tuple[list[dict], bool]:
    # Cache failure as well, to avoid retry storms consuming free API quota.
    try:
        return fetch_champions_league_events(api_key), True
    except Exception:
        return [], False


def _matchday_near_kickoff(frame: pd.DataFrame) -> bool:
    """Request the paid-key endpoint only around real matchday play."""
    if frame.empty:
        return False
    now = pd.Timestamp.now(tz="UTC")
    earliest = pd.Timestamp(frame["kickoff_utc"].min())
    latest = pd.Timestamp(frame["kickoff_utc"].max())
    return earliest - pd.Timedelta(minutes=15) <= now <= latest + pd.Timedelta(hours=4)


@st.fragment(run_every="10m")
def render_selected_match_score(
    centre_row: pd.Series, *,
    allow_fetch: bool,
) -> None:
    """Independently refresh only native scores; frozen picks stay unchanged."""
    match_score = None
    connected = False
    key = _realtime_api_key()
    if key and allow_fetch:
        events, connected = _cached_realtime_scoreboard(key)
        if connected:
            native = match_fixture_score(centre_row, events)
            if native is not None:
                match_score = MatchScore(
                    home_goals=native.home_goals,
                    away_goals=native.away_goals,
                    status=native.status,
                    minute=native.minute,
                )

    render_html(scoreboard_html(centre_row, score=match_score))
    if match_score is not None:
        st.caption(
            "Score supplied by Realtime Sports API; delayed and subject to correction. "
            "Checked at most every 10 minutes on the free-plan setting. "
            "Pre-match forecasts are frozen."
        )
    else:
        st.caption(
            "A verified score for this fixture is not available in the custom "
            "Match Centre right now. The independent scoreboard below may "
            "have more recent information."
        )
    with st.container(key="md_native_score_details"):
        pitch_col, forecast_col = st.columns([1.1, 1], gap="medium")
        with pitch_col:
            render_html(empty_pitch_html())
        with forecast_col:
            render_html(forecast_vs_reality_html(
                centre_row, forecasts_available=True, score=match_score,
            ))


# ============================================================
# STARTUP LOADING SCREEN
# ============================================================

show_startup_loading_screen()


# ============================================================
# DATA
# ============================================================


@st.cache_data
def load_predictions(
    path_string: str,
    modified_ns: int,
) -> pd.DataFrame:

    del modified_ns

    path = PREDICTIONS_FILE.__class__(
        path_string
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_probability",
        "draw_probability",
        "away_probability",
        "predicted_result",
        "predicted_selection",
        "max_probability",
        "confidence",
        "confidence_percent",
        "confidence_label",
        "confidence_caps",
        "model_top_two_margin",
        "model_decisiveness_profile",
        "elo_features_available",
        "ucl_form_features_available",
        "domestic_features_available",
        "squad_features_available",
        "overall_features_available",
        "overall_feature_availability",
        "predicted_home_goals",
        "predicted_away_goals",
        "predicted_scoreline",
        "predicted_scoreline_probability",
        "global_modal_scoreline",
        "global_modal_result",
        "score_home_lambda",
        "score_away_lambda",
        "score_total_lambda",
        "score_bridge_max_error_pp",
        "score_bridge_quality",
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Prediction artifact is missing columns: "
            f"{sorted(missing)}"
        )

    dataframe[
        "kickoff_utc"
    ] = pd.to_datetime(
        dataframe[
            "kickoff_utc"
        ],
        errors="raise",
        utc=True,
    )

    return (
        dataframe
        .sort_values(
            [
                "kickoff_utc",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HEALTH
# ============================================================

health = (
    evaluate_app_health(
        source=APP_SOURCE
    )
)

health_class = (
    health_css_class(
        health[
            "severity"
        ]
    )
)

health_finish_display = (
    format_health_finish_time(
        health
    )
)

next_kickoff_display = (
    format_next_kickoff_local(
        health
    )
)


# ============================================================
# BROADCAST BRAND — SHARED WITH LOADER
# ============================================================

render_html(brand_lockup_html())


# ============================================================
# ============================================================
# FORECAST STATUS
# ============================================================

source_label = (
    "PUBLIC"
    if
    APP_SOURCE.is_public_snapshot
    else
    "LIVE"
)

creator_status_html = ""

if CREATOR_MODE:

    creator_status_html = f"""
        <span class="status-pill creator-status-pill">
            UPDATED
            {escape(health["pipeline_age_display"])}
        </span>

        <span class="status-pill creator-status-pill">
            {escape(source_label)}
            DATA
        </span>
    """


if CREATOR_MODE:
    render_html(
        f"""
        <div class="status-row consumer-status-row">

            <span class="status-pill {health_class}">
                                FORECASTS
                {escape(health["state"])}
            </span>

            <span class="status-pill">
                {escape(health["upcoming_fixture_count"])}
                MATCHES READY
            </span>

            <span class="status-pill">
                NEXT
                {escape(next_kickoff_display)}
            </span>

            {creator_status_html}

        </div>
        """
    )


# ============================================================
# BLOCK INVALID STATE
# ============================================================

if CREATOR_MODE and not health[
    "can_serve_predictions"
]:

    warning_text = (
        " / ".join(
            str(warning)
            for warning in health.get(
                "warnings",
                [],
            )
        )
    )

    render_html(
        f"""
        <div class="production-alert red">

            <strong>
                FORECAST SERVICE BLOCKED
            </strong>

            <br><br>

            State:
            {escape(health["state"])}

            <br>

            Reason:
            {escape(health["reason"])}

            <br>

            Source:
            {escape(health["source_mode"])}

            <br>

            Warnings:
            {escape(warning_text or "UNKNOWN")}

        </div>
        """
    )

    if not APP_SOURCE.is_public_snapshot:

        st.code(
            "python -m src.production.run"
        )

    # Keep the Matchday and Evidence Vault pages accessible even when the
    # forecast service cannot issue *new* predictions. Printing remains gated.

FORECASTS_AVAILABLE = bool(health["can_serve_predictions"])


# ============================================================
# AGING WARNING
# ============================================================

if CREATOR_MODE and health[
    "refresh_recommended"
]:

    render_html(
        """
        <div class="production-alert amber">

            <strong>
                REFRESH RECOMMENDED
            </strong>

            <br>

            The current forecast remains usable,
            but production evidence is aging.

        </div>
        """
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

try:

    modified_ns = int(
        PREDICTIONS_FILE
        .stat()
        .st_mtime_ns
    )

    predictions = (
        load_predictions(
            str(
                PREDICTIONS_FILE
            ),
            modified_ns,
        )
    )

except Exception as error:

    st.error(
        "Could not load the selected "
        "prediction artifact."
    )

    st.code(
        str(error)
    )

    st.stop()


# Match Centre sees all fixtures in the published snapshot, including
# already-started matches. This is fixture information, NOT a live score feed.
matchday_predictions = predictions.copy()

# The printer continues to use only upcoming fixtures, and is disabled if
# production health checks cannot safely serve fresh forecasts.
predictions = filter_upcoming_predictions(predictions)
if not FORECASTS_AVAILABLE:
    predictions = predictions.iloc[0:0].copy()


# ============================================================
# V2 SCREEN FLOW STATE
# ============================================================

matchday_date = (
    matchday_predictions["kickoff_utc"].min()
    .tz_convert(LOCAL_TZ)
    .strftime("%d %B %Y")
    if not matchday_predictions.empty
    else "the next matchday"
)

fixture_ids = [
    match_identifier(
        predictions.iloc[
            index
        ]
    )
    for index in range(
        len(
            predictions
        )
    )
]

if (
    "selected_fixture_id"
    not in
    st.session_state
):

    st.session_state[
        "selected_fixture_id"
    ] = None


if (
    "v2_screen"
    not in
    st.session_state
):

    st.session_state[
        "v2_screen"
    ] = "home"


# Migrate an open Stage 6E Evidence session into the new Vault.
if (
    st.session_state[
        "v2_screen"
    ]
    ==
    "evidence"
):

    st.session_state[
        "v2_screen"
    ] = "vault"


if (
    "show_share_forecast"
    not in
    st.session_state
):

    st.session_state[
        "show_share_forecast"
    ] = False



if (
    "vault_filter"
    not in
    st.session_state
):

    st.session_state[
        "vault_filter"
    ] = "all"


if (
    "vault_record_id"
    not in
    st.session_state
):

    st.session_state[
        "vault_record_id"
    ] = None


if (
    "vault_show_receipt"
    not in
    st.session_state
):

    st.session_state[
        "vault_show_receipt"
    ] = False


if (
    st.session_state[
        "selected_fixture_id"
    ]
    is not None
    and
    st.session_state[
        "selected_fixture_id"
    ]
    not in
    fixture_ids
):

    st.session_state[
        "selected_fixture_id"
    ] = None

    if (
        st.session_state[
            "v2_screen"
        ]
        ==
        "printer"
    ):

        st.session_state[
            "v2_screen"
        ] = "fixtures"

    st.session_state[
        "show_share_forecast"
    ] = False


selected_fixture_id = (
    st.session_state[
        "selected_fixture_id"
    ]
)

selected_row = None
selected_index = None
selected_kickoff_local = None
selected_receipt_id = None
presentation = None

if selected_fixture_id is not None:

    selected_index = (
        fixture_ids.index(
            selected_fixture_id
        )
    )

    selected_row = (
        predictions.iloc[
            selected_index
        ]
    )

    selected_kickoff_local = (
        pd.Timestamp(
            selected_row[
                "kickoff_utc"
            ]
        )
        .tz_convert(
            LOCAL_TZ
        )
    )

    selected_receipt_id = (
        match_identifier(
            selected_row
        )
    )

    presentation = (
        build_forecast_presentation(
            row=selected_row,
            kickoff_local=selected_kickoff_local,
            receipt_id=selected_receipt_id,
            available_columns=predictions.columns,
        )
    )


# ============================================================
# SCREEN NAVIGATION — HOME / MATCHES / LIVE / PRINTER / VAULT
# ============================================================

current_screen = st.session_state["v2_screen"]

# Widget callbacks update routing BEFORE Streamlit's normal rerun.
# This avoids rendering a partial Home screen followed by st.rerun(),
# which otherwise leaves transient ghost panels during page transitions.
def _go_to_page(target: str) -> None:
    st.session_state["v2_screen"] = target
    if target == "vault":
        st.session_state["vault_record_id"] = None
        st.session_state["vault_show_receipt"] = False


def _go_to_match(choice_id: str) -> None:
    st.session_state["matchday_selected_id"] = choice_id
    st.session_state["bp_featured_id"] = choice_id
    st.session_state["v2_screen"] = "live"


def _go_to_printer(fixture_id: str | None) -> None:
    if fixture_id is None or fixture_id not in fixture_ids:
        return
    st.session_state["selected_fixture_id"] = fixture_id
    st.session_state["show_share_forecast"] = False
    st.session_state["v2_screen"] = "printer"


def _set_feature_tab(tab: str) -> None:
    st.session_state["bp_feature_tab"] = tab


nav_items = [
    ("HOME", "home", "nav_home"),
    ("FIXTURES", "fixtures", "nav_matches"),
    ("LIVE", "live", "nav_live"),
    ("PRINTER", "printer", "nav_printer"),
    ("VAULT", "vault", "nav_evidence_vault"),
]
with st.container(key="matchday_top_nav"):
    nav_columns = st.columns(5, gap="small")
    for col, (label, target, key) in zip(nav_columns, nav_items):
        with col:
            st.button(
                label,
                key=key,
                use_container_width=True,
                disabled=(target == "printer" and selected_fixture_id is None),
                type="primary" if current_screen == target else "secondary",
                on_click=_go_to_page,
                args=(target,),
            )


# ============================================================
# NEW PAGE — MATCHDAY HOME
# ============================================================

if current_screen == "home":
    render_html(matchday_toolbar_html(matchday_predictions))
    selected_feature_id = st.session_state.get("bp_featured_id")
    featured = None
    if not matchday_predictions.empty:
        for _, _candidate in matchday_predictions.iterrows():
            if match_identifier(_candidate) == selected_feature_id:
                featured = _candidate
                break
        if featured is None:
            # First upcoming match, otherwise first fixture from the snapshot.
            featured = next(
                (row for _, row in matchday_predictions.iterrows()
                 if pd.Timestamp(row["kickoff_utc"]) > pd.Timestamp.now(tz="UTC")),
                matchday_predictions.iloc[0],
            )
        selected_feature_id = match_identifier(featured)
        st.session_state["bp_featured_id"] = selected_feature_id

    render_html(banner_html(matchday_predictions, feature=featured))
    home_can_print = FORECASTS_AVAILABLE and not predictions.empty

    # Exactly one pair of Home actions. Routing callbacks change screens
    # before the normal Streamlit rerun (never render part of the old page).
    with st.container(key="bp_hero_actions"):
        print_col, match_col = st.columns(2, gap="small")
        can_print_feature = home_can_print and selected_feature_id in fixture_ids
        print_target = (selected_feature_id if can_print_feature
                        else (fixture_ids[0] if fixture_ids else None))
        with print_col:
            st.button(
                "PRINT FORECAST",
                key="nf_home_print", type="primary", use_container_width=True,
                disabled=not home_can_print,
                help="Open the printer for the featured fixture when it is still upcoming.",
                on_click=_go_to_printer, args=(print_target,),
            )
        with match_col:
            st.button(
                "OPEN MATCH CENTRE",
                key="nf_home_live", type="secondary", use_container_width=True,
                on_click=_go_to_match, args=(selected_feature_id,),
                disabled=selected_feature_id is None,
            )

    if not matchday_predictions.empty:
        board = matchday_predictions.reset_index(drop=True)
        schedule_file = Path(__file__).resolve().parent / "data" / "app" / "future_fixtures.csv"
        future_rows = load_future_fixtures(
            schedule_file,
            after_matchday=board["kickoff_utc"].max(),
        )
        with st.container(key="bp_dashboard"):
            left, centre, right = st.columns([1.05, 1.45, 1.00], gap="small")
            with left:
                with st.container(key="bp_fixtures"):
                    render_html(f'<div class="bp-panel-head"><h2>Matchday board</h2>'
                                f'<span>{len(board)} fixtures</span></div>'
                                '<div class="bp-list-head"><span>TIME / MATCH</span>'
                                '<span>MODEL LEAN</span></div>')
                    for i, (_, board_row) in enumerate(board.iterrows()):
                        choice_id = match_identifier(board_row)
                        with st.container(key=f"bp_fixture_item_{i}"):
                            fixture_body, fixture_open = st.columns([2.25, 1.25], gap="small")
                            with fixture_body:
                                render_html(compact_fixture_html(
                                    board_row, selected=(choice_id == selected_feature_id)))
                            with fixture_open:
                                st.button(
                                    "VIEW MATCH",
                                    key=f"bp_open_match_{i}",
                                    use_container_width=True,
                                    help=f"Open {board_row['home_team']} vs {board_row['away_team']} in Match Centre",
                                    on_click=_go_to_match, args=(choice_id,),
                                )
            with centre:
                with st.container(key="bp_focus"):
                    render_html(featured_panel_header_html(featured))
                    active_feature_tab = st.session_state.get("bp_feature_tab", "Preview")
                    if active_feature_tab not in ("Preview", "Prediction", "Evidence"):
                        active_feature_tab = "Preview"
                    with st.container(key="bp_feature_tabs"):
                        tab_cols = st.columns(3, gap="small")
                        for tab_col, tab_name in zip(tab_cols, ("Preview", "Prediction", "Evidence")):
                            with tab_col:
                                st.button(
                                    tab_name,
                                    key=f"bp_tab_{tab_name.lower()}",
                                    type="primary" if active_feature_tab == tab_name else "secondary",
                                    use_container_width=True,
                                    on_click=_set_feature_tab, args=(tab_name,),
                                )
                    render_html(featured_panel_body_html(featured, active_feature_tab))
                    st.button("MATCH DETAILS", key="bp_feature_open",
                              use_container_width=True, on_click=_go_to_match,
                              args=(selected_feature_id,))
            with right:
                with st.container(key="bp_sidebar"):
                    render_html(future_fixtures_html(future_rows))
                    render_html(strongest_picks_html(board))
    else:
        st.caption("The matchday board will appear when fixtures are available.")


# ============================================================
# CUSTOM MATCH CENTRE — NO SYNTHETIC SCORES OR EVENTS
# ============================================================

elif current_screen == "live":
    render_html(matchday_toolbar_html(matchday_predictions))
    st.markdown("#### MATCH CENTRE")
    st.caption(
        "Select a fixture to see its verified match score when connected, "
        "alongside the original unchanged pre-match forecast."
    )
    if matchday_predictions.empty:
        st.info("No matchday fixture snapshot is available.")
    else:
        centre_rows = matchday_predictions.reset_index(drop=True)
        centre_ids = [match_identifier(row) for _, row in centre_rows.iterrows()]
        display_labels = {
            match_identifier(row): (
                f"{row['home_team']} vs {row['away_team']} — "
                f"{pd.Timestamp(row['kickoff_utc']).tz_convert(LOCAL_TZ).strftime('%H:%M SAST')}"
            ) for _, row in centre_rows.iterrows()
        }
        if st.session_state.get("matchday_selected_id") not in centre_ids:
            st.session_state["matchday_selected_id"] = centre_ids[0]
        centre_selected = st.selectbox(
            "SELECT A MATCH", centre_ids,
            format_func=lambda value: display_labels.get(value, value),
            index=centre_ids.index(st.session_state["matchday_selected_id"]),
        )
        st.session_state["matchday_selected_id"] = centre_selected
        centre_row = centre_rows.iloc[centre_ids.index(centre_selected)]
        # One authenticated request for the full league, shared through Streamlit's
        # cache. Never scrape the provider iframe for an individual match.
        render_selected_match_score(
            centre_row, allow_fetch=_matchday_near_kickoff(centre_rows),
        )
        can_print = FORECASTS_AVAILABLE and centre_selected in fixture_ids
        st.button("PRINT THIS FORECAST", key="md_print_selected",
                  use_container_width=True, disabled=not can_print,
                  on_click=_go_to_printer, args=(centre_selected,))
        if not can_print:
            st.caption("New receipts are available only for valid pre-kickoff forecasts.")
    # The independently refreshed, no-key public widget still lists other games.
    render_ucl_live_scoreboard()


# ============================================================
# PAGE 01 — CHOOSE MATCH
# ============================================================

elif current_screen == "fixtures":

    render_html(
        f"""
        <div class="v2-page-shell">

            <div class="interaction-kicker">
                01 // CHOOSE YOUR MATCH
            </div>

            <div class="interaction-title">
                WHICH FORECAST SHOULD WE PRINT?
            </div>

            <div class="interaction-copy">
                Champions League matches for
                {escape(matchday_date)}.
                Choose one fixture to continue to the printer.
            </div>

        </div>
        """
    )

    fixture_count = len(
        predictions
    )
    if fixture_count == 0:
        st.info(
            "There are no upcoming printable predictions in this snapshot. "
            "Visit Match Centre for the matchday board or Vault for history."
        )

    # A keyed container scopes the mobile two-column CSS to this
    # picker only; navigation, exports and Evidence Vault retain
    # Streamlit's normal responsive behavior.
    with st.container(key="fixture_picker_grid"):

        for start_index in range(
            0,
            fixture_count,
            2,
        ):

            columns = st.columns(
                2,
                gap="small",
            )

            for offset in range(
                2
            ):

                index = (
                    start_index
                    +
                    offset
                )

                if index >= fixture_count:

                    continue

                fixture_row = (
                    predictions.iloc[
                        index
                    ]
                )

                fixture_id = (
                    fixture_ids[
                        index
                    ]
                )

                fixture_kickoff = (
                    pd.Timestamp(
                        fixture_row[
                            "kickoff_utc"
                        ]
                    )
                    .tz_convert(
                        LOCAL_TZ
                    )
                )

                selected = (
                    fixture_id
                    ==
                    selected_fixture_id
                )

                with columns[
                    offset
                ]:

                    render_html(
                        build_fixture_choice_html(
                            row=fixture_row,
                            kickoff_local=fixture_kickoff,
                            selected=selected,
                        )
                    )

                    button_label = (
                        "OPEN PRINTER"
                        if selected
                        else
                        "PRINT FORECAST"
                    )

                    if st.button(
                        button_label,
                        key=(
                            f"fixture_select_"
                            f"{index}"
                        ),
                        use_container_width=True,
                    ):

                        st.session_state[
                            "selected_fixture_id"
                        ] = fixture_id

                        st.session_state[
                            "v2_screen"
                        ] = "printer"

                        st.session_state[
                            "show_share_forecast"
                        ] = False

                        st.rerun()


    with st.expander(
        "Matchday overview"
    ):

        board = predictions[
            [
                "kickoff_utc",
                "home_team",
                "away_team",
                "predicted_scoreline",
                "predicted_selection",
                "home_probability",
                "draw_probability",
                "away_probability",
                "confidence_percent",
            ]
        ].copy()

        board[
            "KICKOFF"
        ] = (
            board[
                "kickoff_utc"
            ]
            .dt.tz_convert(
                LOCAL_TZ
            )
            .dt.strftime(
                "%H:%M"
            )
        )

        board[
            "HOME"
        ] = (
            board[
                "home_probability"
            ]
            *
            100
        ).round(
            1
        )

        board[
            "DRAW"
        ] = (
            board[
                "draw_probability"
            ]
            *
            100
        ).round(
            1
        )

        board[
            "AWAY"
        ] = (
            board[
                "away_probability"
            ]
            *
            100
        ).round(
            1
        )

        board[
            "EVIDENCE"
        ] = (
            board[
                "confidence_percent"
            ]
            .round(
                1
            )
        )

        board = board.rename(
            columns={
                "home_team":
                    "HOME TEAM",

                "away_team":
                    "AWAY TEAM",

                "predicted_scoreline":
                    "PROJECTED SCORE",

                "predicted_selection":
                    "PREDICTED WINNER",
            }
        )

        board = board[
            [
                "KICKOFF",
                "HOME TEAM",
                "AWAY TEAM",
                "PROJECTED SCORE",
                "PREDICTED WINNER",
                "HOME",
                "DRAW",
                "AWAY",
                "EVIDENCE",
            ]
        ]

        st.dataframe(
            board,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# PAGE 02 — 3D PRINTER
# ============================================================

elif current_screen == "printer":

    if (
        selected_row is None
        or
        presentation is None
    ):

        st.session_state[
            "v2_screen"
        ] = "fixtures"

        st.rerun()

    row = selected_row

    render_html(
        f"""
        <div class="v2-page-shell printer-page-intro">

            <div class="interaction-kicker">
                02 // FORECAST PRINTER
            </div>

            <div class="interaction-title">
                {escape(row["home_team"])}
                VS
                {escape(row["away_team"])}
            </div>

            <div class="interaction-copy">
                The machine is printing this forecast.
                Read the receipt, then inspect the evidence
                or save a copy.
            </div>

        </div>
        """
    )

    render_html(
        presentation.receipt_html
    )


    render_html(
        """
        <div class="prediction-complete">

            <div class="prediction-complete-top">

                <div class="prediction-complete-state">
                    PREDICTION PRINTED
                </div>

                <div class="prediction-complete-copy">
                    Continue to the evidence page,
                    save the receipt, or choose another match.
                </div>

            </div>

        </div>
        """
    )


    action_col_1, action_col_2, action_col_3 = (
        st.columns(
            3
        )
    )

    with action_col_1:

        if st.button(
            "OPEN EVIDENCE VAULT",
            key="open_evidence_vault",
            use_container_width=True,
        ):

            st.session_state[
                "v2_screen"
            ] = "vault"

            st.session_state[
                "vault_record_id"
            ] = None

            st.rerun()


    with action_col_2:

        if st.button(
            "SHARE FORECAST",
            key="toggle_share_forecast",
            use_container_width=True,
        ):

            st.session_state[
                "show_share_forecast"
            ] = (
                not
                st.session_state[
                    "show_share_forecast"
                ]
            )


    with action_col_3:

        if st.button(
            "CHOOSE ANOTHER MATCH",
            key="clear_selected_fixture",
            use_container_width=True,
        ):

            st.session_state[
                "selected_fixture_id"
            ] = None

            st.session_state[
                "v2_screen"
            ] = "fixtures"

            st.session_state[
                "show_share_forecast"
            ] = False

            st.rerun()


    if st.session_state[
        "show_share_forecast"
    ]:

        render_html(
            """
            <div class="share-panel">

                <div class="share-panel-title">
                    SHARE-READY FORECAST
                </div>

                <div class="share-panel-copy">
                    Copy the text below or save the image receipt.
                </div>

            </div>
            """
        )

        st.code(
            presentation.share_text,
            language=None,
        )


    try:

        png_bytes = (
            build_receipt_png(
                row
            )
        )

        pdf_bytes = (
            build_receipt_pdf(
                row
            )
        )

    except Exception as error:

        st.error(
            "The prediction is ready, but the downloadable "
            "receipt could not be generated."
        )

        st.code(
            str(
                error
            )
        )

        png_bytes = None
        pdf_bytes = None


    save_col_1, save_col_2 = (
        st.columns(
            2
        )
    )

    with save_col_1:

        if png_bytes is not None:

            st.download_button(
                label="SHARE / SAVE PNG",
                data=png_bytes,
                file_name=(
                    export_filename(
                        row,
                        extension="png",
                    )
                ),
                mime="image/png",
                use_container_width=True,
            )


    with save_col_2:

        if pdf_bytes is not None:

            st.download_button(
                label="SAVE PDF RECEIPT",
                data=pdf_bytes,
                file_name=(
                    export_filename(
                        row,
                        extension="pdf",
                    )
                ),
                mime="application/pdf",
                use_container_width=True,
            )


# ============================================================
# PAGE 03 — EVIDENCE VAULT
# ============================================================

elif current_screen == "vault":

    vault = (
        load_evidence_vault()
    )

    selected_vault_record = (
        find_evaluation(
            vault.records,
            st.session_state[
                "vault_record_id"
            ],
        )
        if vault.available
        else None
    )


    # ========================================================
    # VAULT RECORD DETAIL
    # ========================================================

    if selected_vault_record is not None:

        render_html(
            """
            <div class="v2-page-shell evidence-page-intro">

                <div class="interaction-kicker">
                    03 // EVIDENCE VAULT
                </div>

                <div class="interaction-title">
                    FORECAST RECORD
                </div>

                <div class="interaction-copy">
                    This is the archived pre-match forecast
                    compared with the verified final result.
                </div>

            </div>
            """
        )

        render_html(
            build_vault_detail_html(
                selected_vault_record
            )
        )

        record_col_1, record_col_2 = (
            st.columns(
                2
            )
        )

        with record_col_1:

            if st.button(
                "← BACK TO VAULT",
                key="vault_back_to_list",
                use_container_width=True,
            ):

                st.session_state[
                    "vault_record_id"
                ] = None

                st.session_state[
                    "vault_show_receipt"
                ] = False

                st.rerun()


        with record_col_2:

            if (
                selected_fixture_id
                is not None
            ):

                if st.button(
                    "RETURN TO CURRENT PRINTER",
                    key="vault_return_to_printer",
                    use_container_width=True,
                ):

                    st.session_state[
                        "v2_screen"
                    ] = "printer"

                    st.rerun()


        receipt_path = (
            receipt_asset_path(
                selected_vault_record
            )
        )

        if receipt_path is not None:

            if st.button(
                (
                    "HIDE ORIGINAL RECEIPT"
                    if
                    st.session_state[
                        "vault_show_receipt"
                    ]
                    else
                    "VIEW ORIGINAL RECEIPT"
                ),
                key="vault_toggle_original_receipt",
                use_container_width=True,
            ):

                st.session_state[
                    "vault_show_receipt"
                ] = (
                    not
                    st.session_state[
                        "vault_show_receipt"
                    ]
                )

                st.rerun()


            if st.session_state[
                "vault_show_receipt"
            ]:

                st.image(
                    str(
                        receipt_path
                    ),
                    caption=(
                        "Original pre-match UCL Predictor receipt"
                    ),
                    use_container_width=True,
                )


        with st.expander(
            "Technical scoring details"
        ):

            st.markdown(
                f"""
**Provenance:**
`{selected_vault_record.provenance}`

**Home / Draw / Away probabilities:**
{selected_vault_record.home_probability * 100:.2f}% /
{selected_vault_record.draw_probability * 100:.2f}% /
{selected_vault_record.away_probability * 100:.2f}%

**Receipt transaction ID:**
`{selected_vault_record.receipt_transaction_id or "N/A"}`

**Model version:**
`{selected_vault_record.model_version or "Historical receipt artifact"}`

**Prediction revision:**
{selected_vault_record.revision or "N/A"}
                """
            )

            if (
                selected_vault_record
                .predicted_scoreline_probability
                is not None
            ):

                st.markdown(
                    f"""
**Chance of projected exact score:**
{selected_vault_record.predicted_scoreline_probability * 100:.2f}%
                    """
                )

            if (
                selected_vault_record.expected_home_goals
                is not None
                and
                selected_vault_record.expected_away_goals
                is not None
            ):

                st.markdown(
                    f"""
**Expected goals:**
{selected_vault_record.expected_home_goals:.2f} -
{selected_vault_record.expected_away_goals:.2f}
                    """
                )

            if (
                selected_vault_record.multiclass_brier
                is not None
            ):

                st.markdown(
                    f"""
**Multiclass Brier score:**
{selected_vault_record.multiclass_brier:.4f} *(lower is better)*

**Log loss:**
{selected_vault_record.log_loss:.4f} *(lower is better)*
                    """
                )


    # ========================================================
    # VAULT INDEX
    # ========================================================

    else:

        render_html(
            """
            <div class="v2-page-shell evidence-page-intro">

                <div class="interaction-kicker">
                    03 // EVIDENCE VAULT
                </div>

                <div class="interaction-title">
                    THE RECEIPTS DON'T DISAPPEAR AFTER KICKOFF.
                </div>

                <div class="interaction-copy">
                    Every live forecast is locked before the
                    match begins. When the result is verified,
                    the original prediction is compared with
                    what actually happened.
                </div>

            </div>
            """
        )


        if not vault.available:

            render_html(
                f"""
                <div class="vault-empty">

                    <div class="vault-empty-title">
                        HISTORY IS STILL BUILDING
                    </div>

                    <div class="vault-empty-copy">
                        {escape(
                            vault.message
                            or
                            "No completed forecasts yet."
                        )}
                    </div>

                </div>
                """
            )

        else:

            summary = vault.summary

            resolved_count = int(
                summary.get(
                    "resolved_fixture_count",
                    0,
                )
            )

            correct_count = int(
                summary.get(
                    "correct_result_picks",
                    0,
                )
            )

            exact_eligible = int(
                summary.get(
                    "exact_score_eligible",
                    0,
                )
            )

            exact_correct = int(
                summary.get(
                    "exact_scores_correct",
                    0,
                )
            )

            historical_count = int(
                summary.get(
                    "historical_receipt_count",
                    0,
                )
            )

            system_count = int(
                summary.get(
                    "system_archive_count",
                    0,
                )
            )

            sample_warning = (
                summary.get(
                    "sample_warning"
                )
            )

            render_html(
                f"""
                <div class="vault-summary-grid">

                    <div class="vault-summary-card">

                        <span>
                            COMPLETED FORECASTS
                        </span>

                        <strong>
                            {resolved_count}
                        </strong>

                        <small>
                            VERIFIED RESULTS
                        </small>

                    </div>

                    <div class="vault-summary-card">

                        <span>
                            RESULT PICKS CORRECT
                        </span>

                        <strong>
                            {correct_count}
                            /
                            {resolved_count}
                        </strong>

                        <small>
                            SO FAR
                        </small>

                    </div>

                    <div class="vault-summary-card">

                        <span>
                            EXACT SCORE HITS
                        </span>

                        <strong>
                            {exact_correct}
                            /
                            {exact_eligible}
                        </strong>

                        <small>
                            HARDER THAN RESULT PICKS
                        </small>

                    </div>

                </div>

                <div class="vault-provenance-summary">

                    <span>
                        {system_count}
                        SYSTEM-ARCHIVED FORWARD TESTS
                    </span>

                    <span>
                        {historical_count}
                        VERIFIED HISTORICAL PRE-KICKOFF RECEIPTS
                    </span>

                </div>
                """
            )


            if sample_warning:

                render_html(
                    """
                    <div class="vault-sample-warning">

                        <strong>
                            EARLY SAMPLE
                        </strong>

                        There are not enough completed matches
                        yet to make reliable claims about the
                        model's long-term performance. This
                        page shows the record transparently
                        while the sample grows.

                    </div>
                    """
                )


            render_html(
                """
                <div class="vault-filter-heading">
                    FILTER RECORDS
                </div>
                """
            )

            filter_1, filter_2, filter_3 = (
                st.columns(
                    3
                )
            )

            with filter_1:

                if st.button(
                    "ALL",
                    key="vault_filter_all",
                    use_container_width=True,
                    type=(
                        "primary"
                        if
                        st.session_state[
                            "vault_filter"
                        ]
                        ==
                        "all"
                        else
                        "secondary"
                    ),
                ):

                    st.session_state[
                        "vault_filter"
                    ] = "all"

                    st.rerun()


            with filter_2:

                if st.button(
                    "CORRECT PICKS",
                    key="vault_filter_correct",
                    use_container_width=True,
                    type=(
                        "primary"
                        if
                        st.session_state[
                            "vault_filter"
                        ]
                        ==
                        "correct"
                        else
                        "secondary"
                    ),
                ):

                    st.session_state[
                        "vault_filter"
                    ] = "correct"

                    st.rerun()


            with filter_3:

                if st.button(
                    "MISSED PICKS",
                    key="vault_filter_missed",
                    use_container_width=True,
                    type=(
                        "primary"
                        if
                        st.session_state[
                            "vault_filter"
                        ]
                        ==
                        "missed"
                        else
                        "secondary"
                    ),
                ):

                    st.session_state[
                        "vault_filter"
                    ] = "missed"

                    st.rerun()


            filtered_records = (
                filter_evaluations(
                    vault.records,
                    st.session_state[
                        "vault_filter"
                    ],
                )
            )


            if not filtered_records:

                render_html(
                    """
                    <div class="vault-empty">

                        <div class="vault-empty-title">
                            NO RECORDS IN THIS FILTER
                        </div>

                    </div>
                    """
                )

            else:

                for start_index in range(
                    0,
                    len(
                        filtered_records
                    ),
                    2,
                ):

                    vault_columns = (
                        st.columns(
                            2
                        )
                    )

                    for offset in range(
                        2
                    ):

                        record_index = (
                            start_index
                            +
                            offset
                        )

                        if (
                            record_index
                            >=
                            len(
                                filtered_records
                            )
                        ):

                            continue

                        evaluation = (
                            filtered_records[
                                record_index
                            ]
                        )

                        with vault_columns[
                            offset
                        ]:

                            render_html(
                                build_vault_card_html(
                                    evaluation
                                )
                            )

                            if st.button(
                                "OPEN RECORD",
                                key=(
                                    "vault_open_"
                                    +
                                    evaluation.record_id
                                ),
                                use_container_width=True,
                            ):

                                st.session_state[
                                    "vault_record_id"
                                ] = (
                                    evaluation.record_id
                                )

                                st.session_state[
                                    "vault_show_receipt"
                                ] = False

                                st.rerun()


            with st.expander(
                "How the Vault stays honest"
            ):

                st.markdown(
                    """
The Vault keeps two provenance classes separate.

**System-archived forward tests** come from the immutable app
history. A live forward-test prediction must be published
before kickoff, and existing prediction IDs or fixture
revisions cannot silently change.

**Verified historical pre-kickoff receipts** are older original
receipt artifacts that existed before kickoff and have separate
creator verification evidence. They stay labeled as historical
receipts instead of being passed off as system-archived data.

After a match finishes, every record is compared with the
verified final result. Each fixture is counted once.
                    """
                )


# ============================================================
# CREATOR SYSTEM STATUS
# ============================================================

if CREATOR_MODE:

    with st.expander(
        "Creator system status"
    ):

        system_status = pd.DataFrame(
            [
                {
                    "Item":
                        "Forecast service",

                    "Value":
                        health[
                            "state"
                        ],
                },

                {
                    "Item":
                        "Pipeline status",

                    "Value":
                        health[
                            "pipeline_status"
                        ],
                },

                {
                    "Item":
                        "Last refresh",

                    "Value":
                        health_finish_display,
                },

                {
                    "Item":
                        "Data age",

                    "Value":
                        health[
                            "pipeline_age_display"
                        ],
                },

                {
                    "Item":
                        "Next kickoff",

                    "Value":
                        next_kickoff_display,
                },

                {
                    "Item":
                        "Data source",

                    "Value":
                        APP_SOURCE.mode,
                },

                {
                    "Item":
                        "Run ID",

                    "Value":
                        (
                            health[
                                "run_id"
                            ]
                            or
                            "UNKNOWN"
                        ),
                },
            ]
        )

        st.dataframe(
            system_status,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

file_modified = (
    datetime.fromtimestamp(
        PREDICTIONS_FILE
        .stat()
        .st_mtime,
        tz=LOCAL_TZ,
    )
)

creator_footer_html = ""

if CREATOR_MODE:

    creator_footer_html = f"""
        <br>

        pipeline:
        {escape(
            health[
                "pipeline_status"
            ]
        )}

        //

        health:
        {escape(
            health[
                "state"
            ]
        )}

        //

        source:
        {escape(
            APP_SOURCE.mode
        )}

        //

        run:
        {escape(
            health[
                "run_id"
            ]
            or
            "UNKNOWN"
        )}

        <br>

        forecast artifact updated

        {escape(
            file_modified.strftime(
                "%d %b %Y %H:%M SAST"
            )
        )}
    """


render_html(
    f"""
    <div class="app-footer">

        UCL PREDICTOR V2

        //

        INDEPENDENT PRE-MATCH FOOTBALL FORECASTS

        {creator_footer_html}

        <br>

        Forecasts show probabilities, not certainties.

    </div>
    """
)
