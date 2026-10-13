from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import pandas as pd

from src.production.club_crests import (
    crest_html,
)

from src.production.receipt_exports import (
    margin_pp,
)

from src.ui.formatters import (
    display_confidence_label,
    display_profile,
    escape,
    winning_cell_class,
)


# ============================================================
# UCL PREDICTOR V2 — PRESENTATION COMPONENTS
# ============================================================
#
# Presentation only.
#
# This module intentionally does not:
# - load prediction artifacts
# - call Streamlit
# - change CatBoost probabilities
# - change confidence calculations
# - change scoreline calculations
# - perform routing
#
# It converts an already-computed forecast row into reusable
# presentation state and HTML. That separation is what lets V2
# later introduce machine states and progressive reveal without
# coupling animation to prediction logic.
# ============================================================


SCORE_DISTRIBUTION_COLUMNS = {
    "score_home_3plus_probability",
    "score_away_3plus_probability",
    "score_top_1_scoreline",
    "score_top_1_probability",
    "score_top_2_scoreline",
    "score_top_2_probability",
    "score_top_3_scoreline",
    "score_top_3_probability",
    "score_top_4_scoreline",
    "score_top_4_probability",
    "score_top_5_scoreline",
    "score_top_5_probability",
}


@dataclass(
    frozen=True,
)
class ScoreDistributionView:
    available: bool
    projected_scoreline: str
    global_modal_scoreline: str
    exact_score_probability: float
    home_3plus_probability: float | None
    away_3plus_probability: float | None
    html: str


@dataclass(
    frozen=True,
)
class ForecastPresentation:
    home_probability: float
    draw_probability: float
    away_probability: float
    confidence_percent: float
    confidence_width: float
    top_two_margin: float
    home_class: str
    draw_class: str
    away_class: str
    score_distribution: ScoreDistributionView
    receipt_html: str


def build_score_distribution_view(
    *,
    row: pd.Series,
    available_columns: Collection[str],
) -> ScoreDistributionView:

    projected_scoreline = str(
        row[
            "predicted_scoreline"
        ]
    )

    global_modal_scoreline = str(
        row[
            "global_modal_scoreline"
        ]
    )

    exact_score_probability = float(
        row[
            "predicted_scoreline_probability"
        ]
    )

    available = (
        SCORE_DISTRIBUTION_COLUMNS
        <=
        set(
            available_columns
        )
    )

    if not available:

        return ScoreDistributionView(
            available=False,
            projected_scoreline=projected_scoreline,
            global_modal_scoreline=global_modal_scoreline,
            exact_score_probability=exact_score_probability,
            home_3plus_probability=None,
            away_3plus_probability=None,
            html="",
        )

    home_3plus_probability = float(
        row[
            "score_home_3plus_probability"
        ]
    )

    away_3plus_probability = float(
        row[
            "score_away_3plus_probability"
        ]
    )

    score_rows: list[str] = []

    top_probability = max(
        float(
            row[
                "score_top_1_probability"
            ]
        ),
        1e-12,
    )

    for rank in range(
        1,
        6,
    ):

        scoreline = str(
            row[
                f"score_top_{rank}_scoreline"
            ]
        )

        probability = float(
            row[
                f"score_top_{rank}_probability"
            ]
        )

        bar_width = max(
            0.0,
            min(
                100.0,
                (
                    probability
                    /
                    top_probability
                )
                *
                100.0,
            ),
        )

        is_projected = (
            scoreline
            ==
            projected_scoreline
        )

        is_mode = (
            scoreline
            ==
            global_modal_scoreline
        )

        row_class = (
            "score-distribution-row projected"
            if is_projected
            else
            "score-distribution-row"
        )

        tags: list[str] = []

        if is_mode:

            tags.append(
                '<span class="score-distribution-tag">'
                'MODE'
                '</span>'
            )

        if is_projected:

            tags.append(
                '<span class="score-distribution-tag">'
                'PROJECTED'
                '</span>'
            )

        tags_html = ""

        if tags:

            tags_html = (
                '<div class="score-distribution-tags">'
                +
                "".join(
                    tags
                )
                +
                '</div>'
            )

        score_rows.append(
            f"""
            <div class="{row_class}">

                <div class="score-distribution-score">
                    {escape(scoreline)}
                </div>

                <div class="score-distribution-track">

                    <div
                        class="score-distribution-fill"
                        style="
                            width:
                            {bar_width:.2f}%;
                        "
                    >
                    </div>

                </div>

                <div class="score-distribution-probability">
                    {probability * 100:.1f}%
                </div>

                {tags_html}

            </div>
            """
        )

    if (
        projected_scoreline
        ==
        global_modal_scoreline
    ):

        modal_note = (
            "PROJECTED SCORE IS ALSO THE GLOBAL "
            "MODAL EXACT SCORE."
        )

    else:

        modal_note = (
            "MODE "
            f"{global_modal_scoreline} "
            "IS THE HIGHEST-PROBABILITY EXACT SCORE OVERALL. "
            "PROJECTED "
            f"{projected_scoreline} "
            "IS THE HIGHEST-PROBABILITY SCORE CONSISTENT "
            "WITH THE CATBOOST RESULT PICK."
        )

    distribution_html = f"""
    <div class="score-distribution">

        <div class="score-distribution-head">

            <div class="score-distribution-title">
                // SCORE RANGE
            </div>

            <div class="score-distribution-subtitle">
                TOP 5 EXACT-SCORE CELLS
            </div>

        </div>

        {"".join(score_rows)}

        <div class="goal-range-grid">

            <div class="goal-range-cell">

                <div class="goal-range-label">
                    {escape(row["home_team"])}
                    3+ GOALS
                </div>

                <div class="goal-range-value">
                    {home_3plus_probability * 100:.1f}%
                </div>

            </div>

            <div class="goal-range-cell">

                <div class="goal-range-label">
                    {escape(row["away_team"])}
                    3+ GOALS
                </div>

                <div class="goal-range-value">
                    {away_3plus_probability * 100:.1f}%
                </div>

            </div>

        </div>

        <div class="score-distribution-note">
            {escape(modal_note)}
        </div>

    </div>
    """

    return ScoreDistributionView(
        available=True,
        projected_scoreline=projected_scoreline,
        global_modal_scoreline=global_modal_scoreline,
        exact_score_probability=exact_score_probability,
        home_3plus_probability=home_3plus_probability,
        away_3plus_probability=away_3plus_probability,
        html=distribution_html,
    )


def build_forecast_presentation(
    *,
    row: pd.Series,
    kickoff_local: pd.Timestamp,
    receipt_id: str,
    available_columns: Collection[str],
) -> ForecastPresentation:

    home_probability = float(
        row[
            "home_probability"
        ]
    )

    draw_probability = float(
        row[
            "draw_probability"
        ]
    )

    away_probability = float(
        row[
            "away_probability"
        ]
    )

    confidence_percent = float(
        row[
            "confidence_percent"
        ]
    )

    confidence_width = max(
        0.0,
        min(
            100.0,
            confidence_percent,
        ),
    )

    top_two_margin = (
        margin_pp(
            row[
                "model_top_two_margin"
            ]
        )
    )

    home_class = (
        winning_cell_class(
            result=row[
                "predicted_result"
            ],
            cell="H",
        )
    )

    draw_class = (
        winning_cell_class(
            result=row[
                "predicted_result"
            ],
            cell="D",
        )
    )

    away_class = (
        winning_cell_class(
            result=row[
                "predicted_result"
            ],
            cell="A",
        )
    )

    score_distribution = (
        build_score_distribution_view(
            row=row,
            available_columns=available_columns,
        )
    )

    receipt_html = f"""
<div class="terminal-shell">

    <div class="terminal-label-row">

        <div class="terminal-machine-label">
            UCL//POS-290
        </div>


        <div class="terminal-led printer-state">

            <div class="printer-state-printing">

                <span class="printer-dot-printing"></span>

                PRINTING...

            </div>


            <div class="printer-state-ready">

                <span class="printer-dot-ready"></span>

                READY

            </div>

        </div>

    </div>


    <div class="terminal-slot"></div>


    <div class="print-stage">

        <div class="print-feed">

            <div class="receipt print-paper">


                <div class="receipt-brand">
                    UCL//FORECAST
                </div>


                <div class="receipt-brand-sub">
                    Production prediction receipt
                </div>


                <div class="receipt-id">
                    TXN {escape(receipt_id)}
                </div>


                <div class="receipt-rule"></div>


                <div class="fixture-meta">

                    {escape(
                        kickoff_local.strftime(
                            "%A %d %B %Y"
                        )
                    )}

                    <br>

                    {escape(
                        kickoff_local.strftime(
                            "%H:%M"
                        )
                    )}

                    SAST

                    //

                    UEFA CHAMPIONS LEAGUE

                </div>


                <div class="club-lockup">

                    <div class="club-side">

                        {crest_html(
                            row["home_team"],
                            side="home",
                        )}

                        <div class="club-side-name">

                            {escape(
                                row[
                                    "home_team"
                                ]
                            )}

                        </div>

                    </div>


                    <div class="club-vs">
                        VS
                    </div>


                    <div class="club-side">

                        {crest_html(
                            row["away_team"],
                            side="away",
                        )}

                        <div class="club-side-name">

                            {escape(
                                row[
                                    "away_team"
                                ]
                            )}

                        </div>

                    </div>

                </div>


                <div class="score-lockup">

                    {int(
                        row[
                            "predicted_home_goals"
                        ]
                    )}

                    -

                    {int(
                        row[
                            "predicted_away_goals"
                        ]
                    )}

                </div>


                <div class="score-caption">
                    projected score // result-consistent poisson
                </div>


                <div class="score-probability-line">

                    <strong>
                        {score_distribution.exact_score_probability * 100:.1f}%
                    </strong>

                    exact-score probability

                </div>


                <div class="model-pick-box">

                    <div class="model-pick-label">
                        CatBoost result pick
                    </div>

                    <div class="model-pick">

                        {escape(
                            row[
                                "predicted_selection"
                            ]
                        )}

                    </div>

                </div>


                <div class="receipt-rule double"></div>


                <div class="hda">

                    <div class="{home_class}">

                        <div class="hda-label">
                            Home
                        </div>

                        <div class="hda-value">
                            {home_probability * 100:.2f}%
                        </div>

                    </div>


                    <div class="{draw_class}">

                        <div class="hda-label">
                            Draw
                        </div>

                        <div class="hda-value">
                            {draw_probability * 100:.2f}%
                        </div>

                    </div>


                    <div class="{away_class}">

                        <div class="hda-label">
                            Away
                        </div>

                        <div class="hda-value">
                            {away_probability * 100:.2f}%
                        </div>

                    </div>

                </div>


                {score_distribution.html}


                <div class="receipt-rule"></div>


                <div class="confidence-wrap">

                    <div class="confidence-head">

                        <span>
                            Evidence confidence
                        </span>

                        <span>
                            {confidence_percent:.2f}%
                        </span>

                    </div>


                    <div class="confidence-track">

                        <div
                            class="confidence-fill"
                            style="
                                width:
                                {confidence_width:.2f}%;
                            "
                        >
                        </div>

                    </div>


                    <div class="confidence-note">

                        {escape(
                            display_confidence_label(
                                row[
                                    "confidence_label"
                                ]
                            )
                        )}

                        //

                        {escape(
                            display_profile(
                                row[
                                    "model_decisiveness_profile"
                                ]
                            )
                        )}

                        probability profile

                        //

                        top-two margin

                        {top_two_margin:.2f}

                        pp

                    </div>

                </div>


                <div class="receipt-rule"></div>


                <div class="receipt-row">

                    <div class="label">
                        PROJECTED EXACT-SCORE PROB.
                    </div>

                    <div class="value">

                        {float(
                            row[
                                "predicted_scoreline_probability"
                            ]
                        ) * 100:.2f}%

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        EXPECTED GOALS
                    </div>

                    <div class="value">

                        {float(
                            row[
                                "score_home_lambda"
                            ]
                        ):.2f}

                        -

                        {float(
                            row[
                                "score_away_lambda"
                            ]
                        ):.2f}

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        TOTAL GOAL RATE
                    </div>

                    <div class="value">

                        {float(
                            row[
                                "score_total_lambda"
                            ]
                        ):.2f}

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        SCORE BRIDGE
                    </div>

                    <div class="value">

                        {escape(
                            row[
                                "score_bridge_quality"
                            ]
                        )}

                        //

                        {float(
                            row[
                                "score_bridge_max_error_pp"
                            ]
                        ):.3f}

                        PP

                    </div>

                </div>


                <div class="receipt-rule"></div>


                <div class="receipt-row">

                    <div class="label">
                        ELO
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "elo_features_available"
                            ]
                        )}

                        / 6

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        UCL FORM
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "ucl_form_features_available"
                            ]
                        )}

                        / 81

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        DOMESTIC
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "domestic_features_available"
                            ]
                        )}

                        / 102

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        SQUAD
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "squad_features_available"
                            ]
                        )}

                        / 101

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        TOTAL DATA
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "overall_features_available"
                            ]
                        )}

                        / 290

                        //

                        {float(
                            row[
                                "overall_feature_availability"
                            ]
                        ) * 100:.2f}%

                    </div>

                </div>


                <div class="warning-strip"></div>


                <div class="receipt-warning">

                    Probability ≠ confidence.

                    Projected score is the highest-probability
                    exact score consistent with the CatBoost result pick.

                    It is an auxiliary Poisson estimate,
                    not a separately trained goal model.

                </div>


                <div class="barcode"></div>


                <div class="barcode-number">

                    {escape(
                        receipt_id.replace(
                            "-",
                            "",
                        )
                    )}

                </div>


            </div>

        </div>

    </div>

</div>
"""

    return ForecastPresentation(
        home_probability=home_probability,
        draw_probability=draw_probability,
        away_probability=away_probability,
        confidence_percent=confidence_percent,
        confidence_width=confidence_width,
        top_two_margin=top_two_margin,
        home_class=home_class,
        draw_class=draw_class,
        away_class=away_class,
        score_distribution=score_distribution,
        receipt_html=receipt_html,
    )
