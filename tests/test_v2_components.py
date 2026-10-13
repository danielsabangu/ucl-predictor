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
            "max_probability": 0.4582,
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


def test_score_distribution_uses_consumer_language():

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

    assert "MOST LIKELY SCORELINES" in view.html
    assert "MOST LIKELY" in view.html
    assert "PROJECTED" in view.html
    assert "12.2%" in view.html
    assert "10.7%" in view.html

    assert "MODE" not in view.html
    assert "CATBOOST" not in view.html.upper()


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


def test_idle_machine_waits_for_user_choice():

    html = (
        components
        .build_idle_machine_html()
    )

    assert "AWAITING MATCH" in html
    assert "PICK A FIXTURE TO WAKE THE PRINTER" in html
    assert "STANDBY" in html

    assert 'class="printer-3d-scene"' in html
    assert 'class="printer-3d-unit"' in html
    assert 'class="printer-3d-top"' in html
    assert 'class="printer-3d-side"' in html
    assert 'class="printer-3d-front"' in html
    assert 'class="printer-mouth"' in html

    assert "PRINTING" not in html
    assert "PROJECTED SCORE" not in html


def test_fixture_choice_card_marks_current_selection(
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

    html = (
        components
        .build_fixture_choice_html(
            row=row,
            kickoff_local=pd.Timestamp(
                "2026-10-13 21:00:00+02:00"
            ),
            selected=True,
        )
    )

    assert "Villarreal" in html
    assert "Napoli" in html
    assert "21:00" in html
    assert "SELECTED // READY TO REPRINT" in html
    assert "fixture-choice-card selected" in html


def test_forecast_presentation_keeps_machine_and_uses_plain_language(
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
                "2026-10-13 21:00:00+02:00"
            ),
            receipt_id="TEST-VIL-NAP",
            available_columns=row.index,
        )
    )

    html = presentation.receipt_html

    assert 'class="printer-hero printer-hero-active"' in html
    assert 'class="printer-3d-scene"' in html
    assert 'class="printer-3d-unit machine-active"' in html
    assert 'class="printer-3d-top"' in html
    assert 'class="printer-3d-side"' in html
    assert 'class="printer-3d-front"' in html
    assert 'class="printer-mouth active"' in html
    assert 'class="print-stage printer-output-stage"' in html

    assert "WAKING" in html
    assert "PRINTING" in html
    assert "PRINTED" in html
    assert 'class="print-feed"' in html
    assert 'class="receipt print-paper"' in html

    assert "PREDICTED WINNER" in html
    assert "PROJECTED SCORE" in html
    assert "CHANCE OF THIS SCORE" in html
    assert "EVIDENCE STRENGTH" in html
    assert "MATCH SIGNALS CHECKED" in html
    assert "MOST LIKELY SCORELINES" in html

    assert "CATBOOST RESULT PICK" not in html.upper()
    assert "POISSON BRIDGE" not in html.upper()
    assert "EXPECTED GOALS" not in html.upper()

    assert "0-1" in html
    assert "Napoli" in html
    assert "45.8%" in html
    assert "10.7%" in html

    assert presentation.home_probability == 0.2862
    assert presentation.draw_probability == 0.2557
    assert presentation.away_probability == 0.4582

    assert "Predicted winner: Napoli" in presentation.share_text
    assert "Projected score: 0-1" in presentation.share_text
