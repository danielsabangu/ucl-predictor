from __future__ import annotations

import textwrap
from datetime import datetime

import pandas as pd
import streamlit as st

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

from src.ui.startup_loader import (
    show_startup_loading_screen,
)

from src.ui.v2_styles import (
    apply_v2_styles,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="UCL Predictor",
    page_icon="⚽",
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

CREATOR_MODE = (
    not
    APP_SOURCE.is_public_snapshot
)

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
# USER THEME
# ============================================================

if (
    "ui_theme"
    not in
    st.session_state
):

    st.session_state[
        "ui_theme"
    ] = "dark"


CURRENT_THEME = (
    st.session_state[
        "ui_theme"
    ]
)

if CURRENT_THEME not in {
    "dark",
    "light",
}:

    CURRENT_THEME = "dark"

    st.session_state[
        "ui_theme"
    ] = CURRENT_THEME


# ============================================================
# V2 VISUAL SYSTEM
# ============================================================

apply_v2_styles(
    render_html,
    theme=CURRENT_THEME,
)


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
# HEADER
# ============================================================

render_html(
    """
    <div class="main-app-brand">

        <div class="main-app-wordmark-ucl">
            UCL
        </div>

        <div class="main-app-wordmark-predictor">
            PREDICTOR
        </div>

    </div>

    <div class="app-kicker main-app-kicker">
        UEFA CHAMPIONS LEAGUE // MATCH FORECASTS
    </div>

    <div class="app-subtitle main-app-subtitle">
        Pick a Champions League match and print a forecast.
        The predictor weighs team strength, recent form and
        squad evidence, then shows the result as probabilities
        rather than pretending football is certain.
    </div>

    <div class="top-rule"></div>
    """
)


# ============================================================
# DISPLAY THEME
# ============================================================

theme_spacer, theme_control = (
    st.columns(
        [
            5,
            1,
        ]
    )
)

with theme_control:

    next_theme = (
        "light"
        if
        CURRENT_THEME
        ==
        "dark"
        else
        "dark"
    )

    theme_button_label = (
        "☀  LIGHT MODE"
        if
        next_theme
        ==
        "light"
        else
        "◐  DARK MODE"
    )

    if st.button(
        theme_button_label,
        key="toggle_ui_theme",
        use_container_width=True,
    ):

        st.session_state[
            "ui_theme"
        ] = next_theme

        st.rerun()


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


render_html(
    f"""
    <div class="status-row consumer-status-row">

        <span class="status-pill {health_class}">
            ●
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

if not health[
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

    st.stop()


# ============================================================
# AGING WARNING
# ============================================================

if health[
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


predictions = (
    filter_upcoming_predictions(
        predictions
    )
)

if predictions.empty:

    st.error(
        "No upcoming fixtures remain "
        "in this forecast artifact."
    )

    st.stop()


# ============================================================
# V2 SCREEN FLOW STATE
# ============================================================

matchday_date = (
    predictions[
        "kickoff_utc"
    ]
    .min()
    .tz_convert(
        LOCAL_TZ
    )
    .strftime(
        "%d %B %Y"
    )
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
    ] = "fixtures"


if (
    "show_share_forecast"
    not in
    st.session_state
):

    st.session_state[
        "show_share_forecast"
    ] = False


if (
    st.session_state[
        "selected_fixture_id"
    ]
    not in
    fixture_ids
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
# SCREEN NAVIGATION
# ============================================================

current_screen = (
    st.session_state[
        "v2_screen"
    ]
)

nav_1, nav_2, nav_3 = st.columns(
    3
)

with nav_1:

    if st.button(
        "01  MATCHES",
        key="nav_matches",
        use_container_width=True,
        type=(
            "primary"
            if current_screen == "fixtures"
            else
            "secondary"
        ),
    ):

        st.session_state[
            "v2_screen"
        ] = "fixtures"

        st.rerun()


with nav_2:

    if st.button(
        "02  PRINTER",
        key="nav_printer",
        use_container_width=True,
        disabled=(
            selected_fixture_id
            is None
        ),
        type=(
            "primary"
            if current_screen == "printer"
            else
            "secondary"
        ),
    ):

        st.session_state[
            "v2_screen"
        ] = "printer"

        st.rerun()


with nav_3:

    if st.button(
        "03  EVIDENCE",
        key="nav_evidence",
        use_container_width=True,
        disabled=(
            selected_fixture_id
            is None
        ),
        type=(
            "primary"
            if current_screen == "evidence"
            else
            "secondary"
        ),
    ):

        st.session_state[
            "v2_screen"
        ] = "evidence"

        st.rerun()


# ============================================================
# PAGE 01 — CHOOSE MATCH
# ============================================================

if current_screen == "fixtures":

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

    for start_index in range(
        0,
        fixture_count,
        3,
    ):

        columns = st.columns(
            3
        )

        for offset in range(
            3
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
                    ● PREDICTION PRINTED
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
            "VIEW EVIDENCE",
            key="open_evidence_page",
            use_container_width=True,
        ):

            st.session_state[
                "v2_screen"
            ] = "evidence"

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
# PAGE 03 — EVIDENCE
# ============================================================

elif current_screen == "evidence":

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

    home_probability = (
        presentation.home_probability
    )

    draw_probability = (
        presentation.draw_probability
    )

    away_probability = (
        presentation.away_probability
    )

    confidence_percent = (
        presentation.confidence_percent
    )

    top_two_margin = (
        presentation.top_two_margin
    )

    score_distribution_available = (
        presentation
        .score_distribution
        .available
    )

    projected_scoreline = (
        presentation
        .score_distribution
        .projected_scoreline
    )

    global_modal_scoreline = (
        presentation
        .score_distribution
        .global_modal_scoreline
    )

    exact_score_probability = (
        presentation
        .score_distribution
        .exact_score_probability
    )

    home_3plus_probability = (
        presentation
        .score_distribution
        .home_3plus_probability
    )

    away_3plus_probability = (
        presentation
        .score_distribution
        .away_3plus_probability
    )


    if (
        projected_scoreline
        ==
        global_modal_scoreline
    ):

        score_explanation = (
            f"{projected_scoreline} is also the single "
            "most likely exact score overall."
        )

    else:

        score_explanation = (
            f"{global_modal_scoreline} is the single most "
            "likely exact score overall. "
            f"{projected_scoreline} is shown because "
            f"{row['predicted_selection']} is the "
            "predicted winner."
        )


    render_html(
        f"""
        <div class="v2-page-shell evidence-page-intro">

            <div class="interaction-kicker">
                03 // WHY THIS PREDICTION?
            </div>

            <div class="interaction-title">
                {escape(row["home_team"])}
                VS
                {escape(row["away_team"])}
            </div>

            <div class="interaction-copy">
                Here is what the forecast means without the
                machine-learning jargon.
            </div>

        </div>

        <div class="explain-grid">

            <div class="explain-card">

                <div class="explain-label">
                    PREDICTED WINNER
                </div>

                <div class="explain-value">
                    {escape(
                        row[
                            "predicted_selection"
                        ]
                    )}
                </div>

                <div class="explain-copy">
                    This result has a
                    {float(
                        row[
                            "max_probability"
                        ]
                    ) * 100:.1f}%
                    chance according to the model.
                </div>

            </div>

            <div class="explain-card">

                <div class="explain-label">
                    PROJECTED SCORE
                </div>

                <div class="explain-value">
                    {escape(projected_scoreline)}
                </div>

                <div class="explain-copy">
                    Chance of this exact score:
                    {exact_score_probability * 100:.1f}%.
                    {escape(score_explanation)}
                </div>

            </div>

            <div class="explain-card">

                <div class="explain-label">
                    EVIDENCE STRENGTH
                </div>

                <div class="explain-value">
                    {confidence_percent:.0f}%
                </div>

                <div class="explain-copy">
                    {escape(
                        confidence_explanation(
                            row
                        )
                    )}
                </div>

            </div>

        </div>
        """
    )


    evidence_back_1, evidence_back_2 = (
        st.columns(
            2
        )
    )

    with evidence_back_1:

        if st.button(
            "BACK TO PRINTED FORECAST",
            key="evidence_back_to_printer",
            use_container_width=True,
        ):

            st.session_state[
                "v2_screen"
            ] = "printer"

            st.rerun()


    with evidence_back_2:

        if st.button(
            "CHOOSE ANOTHER MATCH",
            key="evidence_choose_another",
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


    with st.expander(
        "See the technical model details"
    ):

        detail_rows = [
            {
                "Item":
                    "Home result probability",

                "Value":
                    percentage(
                        row[
                            "home_probability"
                        ]
                    ),
            },

            {
                "Item":
                    "Draw probability",

                "Value":
                    percentage(
                        row[
                            "draw_probability"
                        ]
                    ),
            },

            {
                "Item":
                    "Away result probability",

                "Value":
                    percentage(
                        row[
                            "away_probability"
                        ]
                    ),
            },

            {
                "Item":
                    "CatBoost result pick",

                "Value":
                    row[
                        "predicted_selection"
                    ],
            },

            {
                "Item":
                    "Top-two result margin",

                "Value":
                    f"{top_two_margin:.2f} pp",
            },

            {
                "Item":
                    "Evidence confidence",

                "Value":
                    f"{confidence_percent:.2f}%",
            },

            {
                "Item":
                    "Confidence label",

                "Value":
                    display_confidence_label(
                        row[
                            "confidence_label"
                        ]
                    ),
            },

            {
                "Item":
                    "Projected exact-score probability",

                "Value":
                    (
                        f"{exact_score_probability * 100:.2f}%"
                    ),
            },

            {
                "Item":
                    "Global modal exact score",

                "Value":
                    global_modal_scoreline,
            },

            {
                "Item":
                    "Home Poisson lambda",

                "Value":
                    (
                        f"{float(
                            row[
                                'score_home_lambda'
                            ]
                        ):.3f}"
                    ),
            },

            {
                "Item":
                    "Away Poisson lambda",

                "Value":
                    (
                        f"{float(
                            row[
                                'score_away_lambda'
                            ]
                        ):.3f}"
                    ),
            },

            {
                "Item":
                    "Score bridge max error",

                "Value":
                    (
                        f"{float(
                            row[
                                'score_bridge_max_error_pp'
                            ]
                        ):.3f} pp"
                    ),
            },

            {
                "Item":
                    "Data coverage",

                "Value":
                    (
                        f"{float(
                            row[
                                'overall_feature_availability'
                            ]
                        ) * 100:.2f}%"
                    ),
            },
        ]

        if score_distribution_available:

            detail_rows.extend(
                [
                    {
                        "Item":
                            f"{row['home_team']} 3+ goals",

                        "Value":
                            (
                                f"{home_3plus_probability * 100:.2f}%"
                            ),
                    },

                    {
                        "Item":
                            f"{row['away_team']} 3+ goals",

                        "Value":
                            (
                                f"{away_3plus_probability * 100:.2f}%"
                            ),
                    },
                ]
            )

        st.dataframe(
            pd.DataFrame(
                detail_rows
            ),
            hide_index=True,
            use_container_width=True,
        )


    with st.expander(
        "How this forecast is made"
    ):

        st.markdown(
            """
The main result forecast asks:
**home win, draw or away win?**

It weighs 290 pre-match signals covering team strength,
Champions League form, domestic form and squad evidence.

**Evidence strength** tells you how complete and trustworthy
the available pre-match evidence is. It is not the same thing
as the predicted result probability.

**Projected score** is also separate. The app uses the result
probabilities to build a range of plausible scores, then shows
a score that agrees with the predicted winner or draw.

That is why the projected score can differ from the single most
likely exact score overall.

For technical users, the frozen CatBoost classifier supplies
the home/draw/away result probabilities and the independent
Poisson bridge supplies the score distribution.

No post-match result information is used as a prediction input.
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

        290 PRE-MATCH SIGNALS

        {creator_footer_html}

        <br>

        Forecasts show probabilities, not certainties.

    </div>
    """
)
