"""Full-score distribution and sanitized deployment regression checks."""

import io
import math
import unittest

import pandas as pd
from PIL import Image

from src.production.receipt_exports import (
    score_probability_grid,
    scoreline_summary_for_row,
    build_receipt_png,
)
from src.ui.v2_components import build_score_distribution_view


class CompleteScoreGridTests(unittest.TestCase):
    @staticmethod
    def public_row():
        # Mirrors public CSV: deliberately does not include score_top fields.
        return pd.Series({
            "home_team": "Manchester City",
            "away_team": "Paris Saint-Germain",
            "kickoff_utc": "2026-10-14T19:00:00+00:00",
            "home_probability": .5557,
            "draw_probability": .2163,
            "away_probability": .2280,
            "max_probability": .5557,
            "predicted_selection": "Manchester City",
            "predicted_result": "H",
            "predicted_home_goals": 2,
            "predicted_away_goals": 1,
            "predicted_scoreline": "2-1",
            "predicted_scoreline_probability": .0977,
            "global_modal_scoreline": "1-1",
            "score_home_lambda": 1.995,
            "score_away_lambda": 1.210,
            "model_top_two_margin": .3276,
            "confidence_percent": 72.,
            "confidence_label": "MODERATE_HIGH",
            "model_decisiveness_profile": "CLEAR_LEAN",
            "overall_feature_availability": .605,
            "overall_features_available": 175,
        })

    def test_grid_is_full_probability_partition(self):
        for home_rate, away_rate in [(0, 0), (1.995, 1.21), (2.42, 2.585)]:
            grid = score_probability_grid(home_rate, away_rate)
            self.assertEqual((len(grid), len(grid[0])), (7, 7))
            self.assertAlmostEqual(sum(map(sum, grid)), 1.0, places=12)
            self.assertTrue(all(0 <= value <= 1 for row in grid for value in row))

    def test_grid_has_real_exact_score_probability(self):
        grid = score_probability_grid(1.995, 1.21)
        expected = (
            math.exp(-1.995) * 1.995**2 / math.factorial(2)
            * math.exp(-1.21) * 1.21
        )
        self.assertAlmostEqual(grid[2][1], expected, places=12)

    def test_invalid_goal_rates_are_rejected(self):
        for rate in [-1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                score_probability_grid(rate, 1.0)

    def test_top_five_derived_from_public_snapshot(self):
        top_five, home_3plus, away_3plus = scoreline_summary_for_row(
            self.public_row()
        )
        self.assertEqual(len(top_five), 5)
        self.assertEqual(top_five[0][0], "1-1")
        self.assertTrue(0 < home_3plus < 1)
        self.assertTrue(0 < away_3plus < 1)

    def test_on_screen_receipt_has_49_cells(self):
        row = self.public_row()
        view = build_score_distribution_view(
            row=row, available_columns=set(row.index)
        )
        self.assertTrue(view.available)
        self.assertEqual(view.html.count("<td"), 49)
        self.assertEqual(view.html.count("projected-score-cell"), 1)
        self.assertIn("ALL POSSIBLE SCORE OUTCOMES", view.html)

    def test_png_includes_extended_receipt(self):
        png_bytes = build_receipt_png(self.public_row())
        self.assertEqual(png_bytes[:8], b"\x89PNG\r\n\x1a\n")
        with Image.open(io.BytesIO(png_bytes)) as image:
            self.assertEqual(image.size, (900, 2860))


if __name__ == "__main__":
    unittest.main()
