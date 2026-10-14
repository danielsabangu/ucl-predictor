from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import pandas as pd

from src.production.club_crests import (
    crest_html,
)

from src.production.receipt_exports import (
    margin_pp,
    SCORE_GRID_LABELS,
    score_grid_for_row,
    scoreline_summary_for_row,
)

from src.ui.formatters import (
    display_confidence_label,
    escape,
    winning_cell_class,
)


# ============================================================
# UCL PREDICTOR V2 — INTERACTIVE PRESENTATION COMPONENTS
# ============================================================
#
# Presentation only.
#
# This module does not:
# - load prediction artifacts
# - call Streamlit
# - change CatBoost probabilities
# - change confidence calculations
# - change scoreline calculations
# - perform application routing
#
# Consumer-facing presentation lives here so the prediction
# engine and the product experience stay cleanly separated.
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
    share_text: str


def build_fixture_choice_html(
    *,
    row: pd.Series,
    kickoff_local: pd.Timestamp,
    selected: bool,
) -> str:

    selected_class = (
        " selected"
        if selected
        else
        ""
    )

    selected_label = (
        "SELECTED // READY TO REPRINT"
        if selected
        else
        "CHOOSE MATCH"
    )

    return f"""
    <div class="fixture-choice-card{selected_class}">

        <div class="fixture-choice-topline">

            <span>
                {escape(
                    kickoff_local.strftime(
                        "%H:%M"
                    )
                )}
                SAST
            </span>

            <span>
                {selected_label}
            </span>

        </div>

        <div class="fixture-choice-teams">

            <div class="fixture-choice-team">

                {crest_html(
                    row["home_team"],
                    side="home",
                )}

                <div class="fixture-choice-name">
                    {escape(row["home_team"])}
                </div>

            </div>

            <div class="fixture-choice-vs">
                VS
            </div>

            <div class="fixture-choice-team">

                {crest_html(
                    row["away_team"],
                    side="away",
                )}

                <div class="fixture-choice-name">
                    {escape(row["away_team"])}
                </div>

            </div>

        </div>

    </div>
    """


def build_idle_machine_html() -> str:

    return """
    <div class="printer-hero printer-hero-idle">

        <div class="printer-3d-scene">

            <div class="printer-floor-shadow"></div>

            <div class="printer-3d-unit">

                <div class="printer-3d-top">
                    <div class="printer-top-ridge"></div>
                </div>

                <div class="printer-3d-side">
                    <div class="printer-side-vent"></div>
                    <div class="printer-side-vent"></div>
                    <div class="printer-side-vent"></div>
                    <div class="printer-side-vent"></div>
                </div>

                <div class="printer-3d-front">

                    <div class="printer-front-header">

                        <div class="printer-engraving">
                            UCL//FORECAST
                            <span>POS-290</span>
                        </div>

                        <div class="printer-led-bank">
                            <span class="printer-led led-off"></span>
                            <span class="printer-led led-off"></span>
                            <span class="printer-led led-off"></span>
                        </div>

                    </div>

                    <div class="printer-control-deck">

                        <div class="printer-control-screen">

                            <div class="printer-control-label">
                                STATUS
                            </div>

                            <div class="printer-control-value idle">
                                STANDBY
                            </div>

                            <div class="printer-control-sub">
                                AWAITING MATCH
                            </div>

                        </div>

                        <div class="printer-control-buttons">

                            <div class="printer-control-button">
                                <span></span>
                            </div>

                            <div class="printer-power-button">
                                <span></span>
                            </div>

                        </div>

                    </div>

                    <div class="printer-mouth-wrap">

                        <div class="printer-mouth-bezel">

                            <div class="printer-mouth">
                                <div class="printer-mouth-line"></div>
                            </div>

                        </div>

                        <div class="printer-mouth-label">
                            FORECAST OUTPUT
                        </div>

                    </div>

                    <div class="printer-lower-panel">

                        <div class="printer-vent-grid">

                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>

                        </div>

                        <div class="printer-serial">
                            UCLP // V2
                        </div>

                    </div>

                </div>

            </div>

        </div>

        <div class="machine-idle-screen">

            <div class="machine-idle-kicker">
                AWAITING MATCH
            </div>

            <div class="machine-idle-title">
                PICK A FIXTURE TO WAKE THE PRINTER
            </div>

            <div class="machine-idle-note">
                Select a match above. The machine will power up,
                print the forecast, then reveal the receipt.
            </div>

        </div>

    </div>
    """


def build_score_distribution_view(
    *,
    row: pd.Series,
    available_columns: Collection[str],
) -> ScoreDistributionView:
    del available_columns  # The public snapshot intentionally omits top-five fields.

    projected_scoreline = str(row["predicted_scoreline"])
    global_modal_scoreline = str(row["global_modal_scoreline"])
    exact_score_probability = float(row["predicted_scoreline_probability"])

    try:
        probability_grid = score_grid_for_row(row)
        scorelines, home_3plus_probability, away_3plus_probability = (
            scoreline_summary_for_row(row)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return ScoreDistributionView(
            available=False,
            projected_scoreline=projected_scoreline,
            global_modal_scoreline=global_modal_scoreline,
            exact_score_probability=exact_score_probability,
            home_3plus_probability=None,
            away_3plus_probability=None,
            html="",
        )

    score_rows = []
    top_probability = max(scorelines[0][1], 1e-12)
    for scoreline, probability in scorelines:
        is_projected = (scoreline == projected_scoreline)
        is_mode = (scoreline == global_modal_scoreline)
        row_class = (
            "score-distribution-row projected" if is_projected
            else "score-distribution-row"
        )
        bar_width = min(100.0, max(0.0, 100.0 * probability / top_probability))
        tags = []
        if is_mode:
            tags.append('<span class="score-distribution-tag">MOST LIKELY</span>')
        if is_projected:
            tags.append('<span class="score-distribution-tag">PROJECTED</span>')
        tags_html = (
            '<div class="score-distribution-tags">'
            + "".join(tags)
            + '</div>'
        ) if tags else ""
        score_rows.append(f"""
        <div class="{row_class}">
            <div class="score-distribution-score">{escape(scoreline)}</div>
            <div class="score-distribution-track">
                <div class="score-distribution-fill" style="width:{bar_width:.2f}%"></div>
            </div>
            <div class="score-distribution-probability">{probability * 100:.1f}%</div>
            {tags_html}
        </div>
        """)

    header_cells = "".join(
        f'<th scope="col">{escape(label)}</th>'
        for label in SCORE_GRID_LABELS
    )
    body_rows: list[str] = []
    for home_goals, home_label in enumerate(SCORE_GRID_LABELS):
        cells: list[str] = []
        for away_goals, away_label in enumerate(SCORE_GRID_LABELS):
            is_projected = (
                home_goals < 6 and away_goals < 6
                and f"{home_goals}-{away_goals}" == projected_scoreline
            )
            cell_class = ' class="projected-score-cell"' if is_projected else ""
            probability = probability_grid[home_goals][away_goals]
            label = (
                "&lt;0.1%" if 0 < probability < .0005
                else f"{probability * 100:.1f}%"
            )
            cells.append(f'<td{cell_class}>{label}</td>')
        body_rows.append(
            '<tr><th scope="row">' + escape(home_label) + '</th>'
            + "".join(cells) + '</tr>'
        )

    matrix_html = (
        '<section class="score-matrix">'
        '<div class="score-distribution-title">ALL POSSIBLE SCORE OUTCOMES</div>'
        '<div class="score-matrix-caption">HOME GOALS DOWN &darr; / '
        'AWAY GOALS ACROSS &rarr;</div>'
        '<div class="score-matrix-scroll"><table class="score-matrix-table">'
        '<thead><tr><th scope="col">H/A</th>'
        + header_cells + '</tr></thead>'
        '<tbody>' + "".join(body_rows) + '</tbody>'
        '</table></div>'
        '<p class="score-matrix-footnote">'
        '6+ means six or more goals (pooled, not an exact score). '
        'All 49 mutually exclusive outcomes sum to 100%. '
        'These are Poisson-bridge estimates, not a player-goalscorer model.'
        '</p></section>'
    )

    if projected_scoreline == global_modal_scoreline:
        modal_note = "PROJECTED SCORE IS ALSO THE MOST LIKELY EXACT SCORE OVERALL."
    else:
        modal_note = (
            f"{global_modal_scoreline} IS THE MOST LIKELY EXACT SCORE OVERALL. "
            f"{projected_scoreline} IS PROJECTED TO MATCH THE MOST LIKELY "
            "HOME / DRAW / AWAY RESULT."
        )

    distribution_html = f"""
    <div class="score-distribution">
        <div class="score-distribution-head">
            <div class="score-distribution-title">MOST LIKELY SCORELINES</div>
            <div class="score-distribution-subtitle">BARS SHOW RELATIVE LIKELIHOOD</div>
        </div>
        {''.join(score_rows)}
        {matrix_html}
        <div class="goal-range-grid">
            <div class="goal-range-cell">
                <div class="goal-range-label">{escape(row['home_team'])} TO SCORE 3+</div>
                <div class="goal-range-value">{home_3plus_probability * 100:.1f}%</div>
            </div>
            <div class="goal-range-cell">
                <div class="goal-range-label">{escape(row['away_team'])} TO SCORE 3+</div>
                <div class="goal-range-value">{away_3plus_probability * 100:.1f}%</div>
            </div>
        </div>
        <div class="score-distribution-note">{escape(modal_note)}</div>
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

    confidence_label = (
        display_confidence_label(
            row[
                "confidence_label"
            ]
        )
    )

    data_coverage = (
        float(
            row[
                "overall_feature_availability"
            ]
        )
        *
        100.0
    )

    predicted_win_chance = (
        float(
            row[
                "max_probability"
            ]
        )
        *
        100.0
    )

    receipt_html = f"""
<div class="printer-hero printer-hero-active">

    <div class="printer-3d-scene">

        <div class="printer-floor-shadow"></div>

        <div class="printer-3d-unit machine-active">

            <div class="printer-3d-top">
                <div class="printer-top-ridge"></div>
            </div>

            <div class="printer-3d-side">
                <div class="printer-side-vent"></div>
                <div class="printer-side-vent"></div>
                <div class="printer-side-vent"></div>
                <div class="printer-side-vent"></div>
            </div>

            <div class="printer-3d-front">

                <div class="printer-front-header">

                    <div class="printer-engraving">
                        UCL//FORECAST
                        <span>POS-290</span>
                    </div>

                    <div class="printer-led-bank">
                        <span class="printer-led led-active"></span>
                        <span class="printer-led led-active delay-1"></span>
                        <span class="printer-led led-ready"></span>
                    </div>

                </div>

                <div class="printer-control-deck">

                    <div class="printer-control-screen">

                        <div class="printer-control-label">
                            STATUS
                        </div>

                        <div class="printer-state">

                            <div class="printer-state-waking">
                                <span class="printer-dot-printing"></span>
                                WAKING
                            </div>

                            <div class="printer-state-printing">
                                <span class="printer-dot-printing"></span>
                                PRINTING
                            </div>

                            <div class="printer-state-ready">
                                <span class="printer-dot-ready"></span>
                                PRINTED
                            </div>

                        </div>

                        <div class="printer-control-sub">
                            FORECAST RECEIPT
                        </div>

                    </div>

                    <div class="printer-control-buttons">

                        <div class="printer-control-button active">
                            <span></span>
                        </div>

                        <div class="printer-power-button on">
                            <span></span>
                        </div>

                    </div>

                </div>

                <div class="printer-mouth-wrap">

                    <div class="printer-mouth-bezel">

                        <div class="printer-mouth active">
                            <div class="printer-mouth-line"></div>
                        </div>

                    </div>

                    <div class="printer-mouth-label">
                        FORECAST OUTPUT
                    </div>

                </div>

                <div class="printer-lower-panel">

                    <div class="printer-vent-grid">

                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>

                    </div>

                    <div class="printer-serial">
                        UCLP // V2
                    </div>

                </div>

            </div>

        </div>

    </div>

    <div class="print-stage printer-output-stage">

        <div class="print-feed">

            <div class="receipt print-paper">

                <div class="receipt-brand">
                    UCL//FORECAST
                </div>

                <div class="receipt-brand-sub">
                    MATCH FORECAST RECEIPT
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
                            {escape(row["home_team"])}
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
                            {escape(row["away_team"])}
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
                    PROJECTED SCORE
                </div>

                <div class="score-probability-line">

                    <strong>
                        {score_distribution.exact_score_probability * 100:.1f}%
                    </strong>

                    CHANCE OF THIS SCORE

                </div>

                <div class="model-pick-box">

                    <div class="model-pick-label">
                        PREDICTED WINNER
                    </div>

                    <div class="model-pick">
                        {escape(row["predicted_selection"])}
                    </div>

                    <div class="model-pick-probability">
                        {predicted_win_chance:.1f}%
                        CHANCE OF THIS RESULT
                    </div>

                </div>

                <div class="receipt-rule double"></div>

                <div class="hda">

                    <div class="{home_class}">

                        <div class="hda-label">
                            HOME
                        </div>

                        <div class="hda-value">
                            {home_probability * 100:.2f}%
                        </div>

                    </div>

                    <div class="{draw_class}">

                        <div class="hda-label">
                            DRAW
                        </div>

                        <div class="hda-value">
                            {draw_probability * 100:.2f}%
                        </div>

                    </div>

                    <div class="{away_class}">

                        <div class="hda-label">
                            AWAY
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
                            EVIDENCE STRENGTH
                        </span>

                        <span>
                            {confidence_percent:.0f}%
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

                        {escape(confidence_label)}

                        //

                        HOW COMPLETE AND TRUSTWORTHY
                        THE MATCH EVIDENCE IS

                    </div>

                </div>

                <div class="receipt-rule"></div>

                <div class="receipt-row">

                    <div class="label">
                        DATA COVERAGE
                    </div>

                    <div class="value">
                        {data_coverage:.0f}%
                    </div>

                </div>

                <div class="receipt-row">

                    <div class="label">
                        MATCH SIGNALS CHECKED
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "overall_features_available"
                            ]
                        )}

                        / 290

                    </div>

                </div>

                <div class="warning-strip"></div>

                <div class="receipt-warning">

                    FORECASTS SHOW PROBABILITIES,
                    NOT CERTAINTIES.

                    THE PROJECTED SCORE IS ONE PLAUSIBLE
                    OUTCOME, NOT A GUARANTEE.

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

    share_text = (
        "UCL Predictor\n"
        f"{row['home_team']} vs {row['away_team']}\n"
        f"Predicted winner: {row['predicted_selection']} "
        f"({predicted_win_chance:.1f}%)\n"
        f"Projected score: {row['predicted_scoreline']} "
        f"({score_distribution.exact_score_probability * 100:.1f}% "
        "chance of this exact score)\n"
        f"Evidence strength: {confidence_percent:.0f}%\n"
        "Forecasts show probabilities, not certainties."
    )

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
        share_text=share_text,
    )
