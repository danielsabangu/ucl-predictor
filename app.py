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
        "03  EVIDENCE VAULT",
        key="nav_evidence_vault",
        use_container_width=True,
        type=(
            "primary"
            if current_screen == "vault"
            else
            "secondary"
        ),
    ):

        st.session_state[
            "v2_screen"
        ] = "vault"

        st.session_state[
            "vault_record_id"
        ] = None

        st.session_state[
            "vault_show_receipt"
        ] = False

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

        290 PRE-MATCH SIGNALS

        {creator_footer_html}

        <br>

        Forecasts show probabilities, not certainties.

    </div>
    """
)
