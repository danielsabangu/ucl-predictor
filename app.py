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

from src.production.club_crests import (
    crest_html,
)

from src.production.receipt_exports import (
    build_receipt_pdf,
    build_receipt_png,
    export_filename,
    margin_pp,
)

from src.ui.formatters import (
    LOCAL_TZ,
    confidence_explanation,
    display_confidence_label,
    display_profile,
    escape,
    fixture_label,
    format_health_finish_time,
    format_next_kickoff_local,
    health_css_class,
    match_identifier,
    percentage,
    result_class_name,
    winning_cell_class,
)


from src.ui.v2_components import (
    build_forecast_presentation,
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
# V2 VISUAL SYSTEM
# ============================================================

apply_v2_styles(
    render_html
)


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
    <div class="app-kicker">
        UEFA CHAMPIONS LEAGUE // PRODUCTION MODEL
    </div>

    <h1 class="app-title">
        UCL<br>PREDICTOR
    </h1>

    <div class="app-subtitle">
        A chronological pre-match forecasting system using
        Elo, Champions League form, domestic form and squad
        evidence. Raw result probability, evidence confidence
        and scoreline estimation remain separate.
    </div>

    <div class="top-rule"></div>
    """
)


# ============================================================
# HEALTH HEADER
# ============================================================

source_label = (
    "PUBLIC SNAPSHOT"
    if
    APP_SOURCE.is_public_snapshot
    else
    "LOCAL LIVE"
)

render_html(
    f"""
    <div class="status-row">

        <span class="status-pill {health_class}">
            ●
            {escape(health["state"])}
        </span>

        <span class="status-pill">
            PIPELINE
            {escape(health["pipeline_status"])}
        </span>

        <span class="status-pill">
            {escape(source_label)}
        </span>

        <span class="status-pill">
            ucl_predictor_v1
        </span>

        <span class="status-pill">
            {escape(health["upcoming_fixture_count"])}
            UPCOMING
        </span>

        <span class="status-pill">
            290 FEATURES
        </span>

    </div>


    <div class="health-panel">

        <div class="health-item">

            <div class="health-label">
                Last production refresh
            </div>

            <div class="health-value">
                {escape(health_finish_display)}
            </div>

        </div>


        <div class="health-item">

            <div class="health-label">
                Data age
            </div>

            <div class="health-value">
                {escape(health["pipeline_age_display"])}
            </div>

        </div>


        <div class="health-item">

            <div class="health-label">
                Next kickoff
            </div>

            <div class="health-value">
                {escape(next_kickoff_display)}
            </div>

        </div>


        <div class="health-item">

            <div class="health-label">
                Countdown
            </div>

            <div class="health-value">
                {escape(health["next_kickoff_display"])}
            </div>

        </div>

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
# MATCHDAY INFO
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

render_html(
    f"""
    <div class="status-row">

        <span class="status-pill">
            MATCHDAY
            {escape(matchday_date)}
        </span>

        <span class="status-pill">
            RUN
            {escape(
                health["run_id"]
                or
                "UNKNOWN"
            )}
        </span>

        <span class="status-pill">
            SOURCE
            {escape(APP_SOURCE.mode)}
        </span>

    </div>
    """
)


# ============================================================
# SELECTOR
# ============================================================

options = list(
    range(
        len(predictions)
    )
)

selected_index = (
    st.selectbox(
        "Select forecast",
        options,
        format_func=lambda index: (
            fixture_label(
                predictions.iloc[
                    index
                ]
            )
        ),
    )
)

row = predictions.iloc[
    int(selected_index)
]


# ============================================================
# SELECTED VALUES
# ============================================================

kickoff_utc = pd.Timestamp(
    row[
        "kickoff_utc"
    ]
)

kickoff_local = (
    kickoff_utc
    .tz_convert(
        LOCAL_TZ
    )
)

receipt_id = (
    match_identifier(
        row
    )
)


# ============================================================
# PRESENTATION COMPONENT
# ============================================================

presentation = (
    build_forecast_presentation(
        row=row,
        kickoff_local=kickoff_local,
        receipt_id=receipt_id,
        available_columns=predictions.columns,
    )
)

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

confidence_width = (
    presentation.confidence_width
)

top_two_margin = (
    presentation.top_two_margin
)

home_class = (
    presentation.home_class
)

draw_class = (
    presentation.draw_class
)

away_class = (
    presentation.away_class
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


# ============================================================
# RECEIPT
# ============================================================

render_html(
    presentation.receipt_html
)


# ============================================================
# EXPORTS
# ============================================================

render_html(
    """
    <div class="export-panel">

        <div class="export-title">
            TAKE YOUR FORECAST
        </div>

        <div class="export-note">
            Download the selected forecast as a
            shareable thermal receipt.
        </div>

    </div>
    """
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
        "Could not generate the "
        "forecast exports."
    )

    st.code(
        str(error)
    )

    png_bytes = None
    pdf_bytes = None


download_col_1, download_col_2 = (
    st.columns(
        2
    )
)


with download_col_1:

    if png_bytes is not None:

        st.download_button(
            label="DOWNLOAD PNG",
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


with download_col_2:

    if pdf_bytes is not None:

        st.download_button(
            label="DOWNLOAD PDF",
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
# CONTEXT
# ============================================================

render_html(
    """
    <div class="section-title">
        Forecast context
    </div>
    """
)

col1, col2, col3 = (
    st.columns(
        3
    )
)


with col1:

    render_html(
        f"""
        <div class="dark-card">

            <div class="dark-card-label">
                Result probability
            </div>

            <div class="dark-card-value">

                {float(
                    row[
                        "max_probability"
                    ]
                ) * 100:.2f}%

            </div>

            <div class="dark-card-note">

                Probability assigned to the selected
                H/D/A result by the frozen CatBoost
                classifier.

            </div>

        </div>
        """
    )


with col2:

    render_html(
        f"""
        <div class="dark-card">

            <div class="dark-card-label">
                Evidence confidence
            </div>

            <div class="dark-card-value">
                {confidence_percent:.2f}%
            </div>

            <div class="dark-card-note">

                {escape(
                    confidence_explanation(
                        row
                    )
                )}

            </div>

        </div>
        """
    )


with col3:

    render_html(
        f"""
        <div class="dark-card">

            <div class="dark-card-label">
                Projected score
            </div>

            <div class="dark-card-value">

                {escape(
                    row[
                        "predicted_scoreline"
                    ]
                )}

            </div>

            <div class="dark-card-note">

                Highest-probability exact score consistent
                with the CatBoost result pick.

                Exact-score probability:

                {float(
                    row[
                        "predicted_scoreline_probability"
                    ]
                ) * 100:.2f}%.

            </div>

        </div>
        """
    )


# ============================================================
# MODAL SCORE NOTE
# ============================================================

if (
    str(
        row[
            "global_modal_scoreline"
        ]
    )
    !=
    str(
        row[
            "predicted_scoreline"
        ]
    )
):

    st.info(
        "Score interpretation: the global modal exact score is "
        f"{row['global_modal_scoreline']}. The projected "
        f"{row['predicted_scoreline']} score is the "
        "highest-probability exact score consistent with "
        "CatBoost's "
        f"{result_class_name(row['predicted_result'])} "
        "result class."
    )


# ============================================================
# DETAILS
# ============================================================

with st.expander(
    "Model & evidence detail"
):

    detail_data = pd.DataFrame(
        [
            {
                "Item":
                    "Home probability",

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
                    "Away probability",

                "Value":
                    percentage(
                        row[
                            "away_probability"
                        ]
                    ),
            },

            {
                "Item":
                    "Model pick",

                "Value":
                    row[
                        "predicted_selection"
                    ],
            },

            {
                "Item":
                    "Top-two margin",

                "Value":
                    f"{top_two_margin:.2f} pp",
            },

            {
                "Item":
                    "Probability profile",

                "Value":
                    display_profile(
                        row[
                            "model_decisiveness_profile"
                        ]
                    ),
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
                    "Confidence caps",

                "Value":
                    str(
                        row[
                            "confidence_caps"
                        ]
                    ).replace(
                        "|",
                        " / ",
                    ),
            },

            {
                "Item":
                    "Score estimate",

                "Value":
                    row[
                        "predicted_scoreline"
                    ],
            },

            {
                "Item":
                    "Home Poisson λ",

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
                    "Away Poisson λ",

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
                    "Bridge max error",

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
                    "Application source",

                "Value":
                    APP_SOURCE.mode,
            },
        ]
    )

    extra_score_detail = [
        {
            "Item":
                "Projected exact-score probability",

            "Value":
                f"{exact_score_probability * 100:.2f}%",
        },

        {
            "Item":
                "Global modal score",

            "Value":
                global_modal_scoreline,
        },
    ]

    if score_distribution_available:

        extra_score_detail.extend(
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

    detail_data = pd.concat(
        [
            detail_data,
            pd.DataFrame(
                extra_score_detail
            ),
        ],
        ignore_index=True,
    )

    st.dataframe(
        detail_data,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Methodology"
):

    st.markdown(
        """
**Result probabilities**

The H/D/A distribution comes directly from the frozen
`ucl_predictor_v1` CatBoost production pipeline using exactly
290 features:

- 6 Elo features
- 81 Champions League form features
- 102 domestic-form features
- 101 squad features

**Evidence confidence**

Confidence is separate from probability. It summarizes how
complete and trustworthy the evidence is for the fixture.

Missing live squad-usage evidence lowers confidence rather than
being fabricated.

**Scoreline**

The projected score is produced by an auxiliary independent-Poisson
bridge fitted to the frozen CatBoost H/D/A probabilities.

The projected score is the highest-probability exact score that is
consistent with CatBoost's predicted H/D/A result. It is not always
the global modal exact score.

Where the forecast artifact contains score-distribution diagnostics,
the interface also shows the highest-probability exact-score cells and
each team's probability of scoring 3+ goals.

Exact-score probability is separate from result probability and from
evidence confidence. The score bridge is not a separately trained
historical goals model.

**Production health**

The application validates its prediction source before serving
forecasts.

Failed, stale or expired artifacts are blocked rather than
silently displayed.

**Deployment**

The public deployment consumes a sanitized forecast snapshot.
The training data, production model artifact, raw datasets and
API credentials are not required by the public interface.

**Exports**

PNG and PDF receipts are generated from the already-computed
forecast. Export generation does not alter model probabilities.

**Leakage rule**

All forecasting features are pre-match.

Future match goals, results and scorelines are not prediction
inputs.
        """
    )


# ============================================================
# MATCHDAY BOARD
# ============================================================

render_html(
    """
    <div class="section-title">
        Matchday board
    </div>
    """
)

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
    "kickoff"
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
    "CONF"
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
            "PROJECTED",

        "predicted_selection":
            "PICK",
    }
)


board = board[
    [
        "kickoff",
        "HOME TEAM",
        "AWAY TEAM",
        "PROJECTED",
        "PICK",
        "HOME",
        "DRAW",
        "AWAY",
        "CONF",
    ]
]


st.dataframe(
    board,
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

render_html(
    f"""
    <div
        style="
            border-top:
                1px solid
                #222;

            color:
                #555;

            font-family:
                'Courier New',
                monospace;

            font-size:
                0.62rem;

            line-height:
                1.6;

            margin-top:
                2rem;

            padding-top:
                1rem;

            text-transform:
                uppercase;
        "
    >

        ucl_predictor_v1

        //

        frozen catboost

        //

        290 features

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

        <br>

        For analytical / portfolio use.

        Probability is not certainty.

    </div>
    """
)