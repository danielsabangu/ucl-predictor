from __future__ import annotations

from types import SimpleNamespace
import unittest

import pandas as pd

from src.ui.formatters import (
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


class UiFormatterTests(
    unittest.TestCase
):
    def test_01_escape_html(
        self,
    ) -> None:
        self.assertEqual(
            escape(
                '<script>alert("x")</script>'
            ),
            (
                "&lt;script&gt;"
                "alert(&quot;x&quot;)"
                "&lt;/script&gt;"
            ),
        )

    def test_02_percentage(
        self,
    ) -> None:
        self.assertEqual(
            percentage(
                0.4585
            ),
            "45.85%",
        )

    def test_03_display_confidence_label(
        self,
    ) -> None:
        self.assertEqual(
            display_confidence_label(
                "MODERATE_HIGH"
            ),
            "MODERATE HIGH",
        )

    def test_04_display_profile(
        self,
    ) -> None:
        self.assertEqual(
            display_profile(
                "CLEAR_LEAN"
            ),
            "CLEAR LEAN",
        )

    def test_05_match_identifier(
        self,
    ) -> None:
        row = pd.Series(
            {
                "kickoff_utc": (
                    "2026-10-13T19:00:00+00:00"
                ),
                "home_team": (
                    "Arsenal"
                ),
                "away_team": (
                    "Lille"
                ),
            }
        )

        self.assertEqual(
            match_identifier(
                row
            ),
            "202610131900-ARS-LIL",
        )

    def test_06_fixture_label_uses_sast(
        self,
    ) -> None:
        row = SimpleNamespace(
            kickoff_utc=(
                pd.Timestamp(
                    "2026-10-13T19:00:00+00:00"
                )
            ),
            home_team="Arsenal",
            away_team="Lille",
        )

        self.assertEqual(
            fixture_label(
                row
            ),
            (
                "21:00  •  "
                "Arsenal vs Lille"
            ),
        )

    def test_07_result_class_names(
        self,
    ) -> None:
        self.assertEqual(
            result_class_name(
                "H"
            ),
            "HOME",
        )

        self.assertEqual(
            result_class_name(
                "D"
            ),
            "DRAW",
        )

        self.assertEqual(
            result_class_name(
                "A"
            ),
            "AWAY",
        )

    def test_08_winning_cell_class(
        self,
    ) -> None:
        self.assertEqual(
            winning_cell_class(
                result="H",
                cell="H",
            ),
            "hda-cell winner",
        )

        self.assertEqual(
            winning_cell_class(
                result="H",
                cell="D",
            ),
            "hda-cell",
        )

    def test_09_confidence_explanation_clear_lean(
        self,
    ) -> None:
        row = pd.Series(
            {
                "model_decisiveness_profile": (
                    "CLEAR_LEAN"
                )
            }
        )

        self.assertEqual(
            confidence_explanation(
                row
            ),
            (
                "The outcome distribution "
                "has a clear leader. "
                "Evidence confidence "
                "reflects data quality, not result "
                "probability."
            ),
        )

    def test_10_confidence_explanation_lean(
        self,
    ) -> None:
        row = pd.Series(
            {
                "model_decisiveness_profile": (
                    "LEAN"
                )
            }
        )

        self.assertEqual(
            confidence_explanation(
                row
            ),
            (
                "The model has a lean, but "
                "alternatives remain material. "
                "Evidence confidence "
                "reflects data quality, not result "
                "probability."
            ),
        )

    def test_11_confidence_explanation_tight(
        self,
    ) -> None:
        row = pd.Series(
            {
                "model_decisiveness_profile": (
                    "TIGHT"
                )
            }
        )

        self.assertEqual(
            confidence_explanation(
                row
            ),
            (
                "The outcome distribution "
                "is tight. "
                "Evidence confidence "
                "reflects data quality, not result "
                "probability."
            ),
        )

    def test_12_health_css_class(
        self,
    ) -> None:
        self.assertEqual(
            health_css_class(
                "GREEN"
            ),
            "green",
        )

        self.assertEqual(
            health_css_class(
                "AMBER"
            ),
            "amber",
        )

        self.assertEqual(
            health_css_class(
                "RED"
            ),
            "red",
        )

        self.assertEqual(
            health_css_class(
                "UNKNOWN"
            ),
            "red",
        )

    def test_13_health_finish_time(
        self,
    ) -> None:
        health = {
            "finished_at_utc": (
                "2026-10-13T07:30:00+00:00"
            )
        }

        self.assertEqual(
            format_health_finish_time(
                health
            ),
            "09:30 SAST",
        )

    def test_14_health_finish_time_missing(
        self,
    ) -> None:
        self.assertEqual(
            format_health_finish_time(
                {}
            ),
            "UNKNOWN",
        )

    def test_15_health_finish_time_invalid(
        self,
    ) -> None:
        self.assertEqual(
            format_health_finish_time(
                {
                    "finished_at_utc": (
                        "not-a-date"
                    )
                }
            ),
            "UNKNOWN",
        )

    def test_16_next_kickoff_local(
        self,
    ) -> None:
        health = {
            "next_kickoff_utc": (
                "2026-10-13T16:45:00+00:00"
            )
        }

        self.assertEqual(
            format_next_kickoff_local(
                health
            ),
            "18:45 SAST",
        )

    def test_17_next_kickoff_missing(
        self,
    ) -> None:
        self.assertEqual(
            format_next_kickoff_local(
                {}
            ),
            "NONE",
        )

    def test_18_next_kickoff_invalid(
        self,
    ) -> None:
        self.assertEqual(
            format_next_kickoff_local(
                {
                    "next_kickoff_utc": (
                        "broken"
                    )
                }
            ),
            "UNKNOWN",
        )


if __name__ == "__main__":
    unittest.main()