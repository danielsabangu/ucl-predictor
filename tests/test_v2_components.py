from __future__ import annotations

import pandas as pd

import src.ui.v2_components as components


def sample_row() -> pd.Series:

    return pd.Series(
        {
            "home_team": "Villarreal",
            "away_team": "Napoli",
            "home_probability": 0.2862,
            "draw_probability": 0.2557,
            "away_probability": 0.4582,
            "predicted_result": "A",
            "predicted_selection": "Napoli",
            "predicted_home_goals": 0,
            "predicted_away_goals": 1,
            "predicted_scoreline": "0-1",
            "predicted_scoreline_probability": 0.1067,
            "global_modal_scoreline": "1-1",
            "confidence_percent": 72.0,
            "confidence_label": "high",
            "model_decisiveness_profile": "moderate",
            "model_top_two_margin": 0.1720,
            "score_home_lambda": 1.14,
            "score_away_lambda": 1.51,
            "score_total_lambda": 2.65,
            "score_bridge_quality": "GOOD",
            "score_bridge_max_error_pp": 0.057,
            "elo_features_available": 6,
            "ucl_form_features_available": 81,
            "domestic_features_available": 102,
            "squad_features_available": 101,
            "overall_features_available": 290,
            "overall_feature_availability": 1.0,
            "score_home_3plus_probability": 0.1078,
            "score_away_3plus_probability": 0.1937,
            "score_top_1_scoreline": "1-1",
            "score_top_1_probability": 0.1216,
            "score_top_2_scoreline": "0-1",
            "score_top_2_probability": 0.1067,
            "score_top_3_scoreline": "1-2",
            "score_top_3_probability": 0.0918,
            "score_top_4_scoreline": "0-2",
            "score_top_4_probability": 0.0805,
            "score_top_5_scoreline": "1-0",
            "score_top_5_probability": 0.0805,
        }
    )


def test_score_distribution_distinguishes_mode_and_projected():

    row = sample_row()

    view = (
        components
        .build_score_distribution_view(
            row=row,
            available_columns=row.index,
        )
    )

    assert view.available is True
    assert view.global_modal_scoreline == "1-1"
    assert view.projected_scoreline == "0-1"

    assert (
        view.html.count(
            ">MODE<"
        )
        ==
        1
    )

    assert (
        view.html.count(
            ">PROJECTED<"
        )
        ==
        1
    )

    assert "12.2%" in view.html
    assert "10.7%" in view.html
    assert "Villarreal" in view.html
    assert "Napoli" in view.html


def test_score_distribution_degrades_safely_without_new_columns():

    row = sample_row()

    available_columns = {
        "predicted_scoreline",
        "global_modal_scoreline",
        "predicted_scoreline_probability",
    }

    view = (
        components
        .build_score_distribution_view(
            row=row,
            available_columns=available_columns,
        )
    )

    assert view.available is False
    assert view.html == ""
    assert view.home_3plus_probability is None
    assert view.away_3plus_probability is None


def test_forecast_presentation_keeps_receipt_machine_contract(
    monkeypatch,
):

    monkeypatch.setattr(
        components,
        "crest_html",
        lambda team, side: (
            f'<div data-team="{team}" '
            f'data-side="{side}"></div>'
        ),
    )

    row = sample_row()

    presentation = (
        components
        .build_forecast_presentation(
            row=row,
            kickoff_local=pd.Timestamp(
                "2026-10-21 21:00:00+02:00"
            ),
            receipt_id="TEST-VIL-NAP",
            available_columns=row.index,
        )
    )

    html = presentation.receipt_html

    assert 'class="terminal-shell"' in html
    assert "PRINTING..." in html
    assert "READY" in html
    assert 'class="terminal-slot"' in html
    assert 'class="print-feed"' in html
    assert 'class="receipt print-paper"' in html

    assert "0-1" in html
    assert "Napoli" in html
    assert "45.82%" in html
    assert "10.7%" in html

    assert presentation.home_probability == 0.2862
    assert presentation.draw_probability == 0.2557
    assert presentation.away_probability == 0.4582
