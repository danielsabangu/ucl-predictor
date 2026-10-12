from __future__ import annotations

import math
import unittest

from src.history.evaluation import (
    PredictionEvaluation,
    build_summary,
    latest_revision_per_fixture,
    multiclass_brier_score,
    multiclass_log_loss,
)


class PredictionEvaluationTests(
    unittest.TestCase
):
    def _evaluation(
        self,
        *,
        fixture_id: str,
        revision: int,
        predicted_result: str,
        actual_result: str,
        correct: bool,
        exact: bool,
        brier: float = 0.50,
        log_loss: float = 0.80,
    ):
        return PredictionEvaluation(
            schema_version=1,

            prediction_id=(
                f"pred_{fixture_id}_{revision}"
            ),

            result_id=(
                f"result_{fixture_id}"
            ),

            fixture_id=fixture_id,

            revision=revision,

            home_team="Home FC",
            away_team="Away FC",

            kickoff_utc=(
                "2026-10-20T19:00:00Z"
            ),

            published_at_utc=(
                "2026-10-20T18:00:00Z"
            ),

            model_version=(
                "ucl_predictor_v1"
            ),

            prediction_type=(
                "LIVE_FORWARD_TEST"
            ),

            predicted_result=(
                predicted_result
            ),

            actual_result=(
                actual_result
            ),

            home_probability=0.50,
            draw_probability=0.25,
            away_probability=0.25,

            actual_result_probability=(
                0.50
                if actual_result == "H"
                else 0.25
            ),

            result_pick_correct=correct,

            predicted_scoreline="2-1",
            actual_scoreline=(
                "2-1"
                if exact
                else "1-1"
            ),

            exact_score_correct=exact,

            multiclass_brier=brier,
            log_loss=log_loss,

            evidence_score=0.60,
            evidence_label="MODERATE",
            overall_feature_availability=0.70,
        )

    def test_01_perfect_brier_is_zero(
        self,
    ):
        score = multiclass_brier_score(
            home_probability=1.0,
            draw_probability=0.0,
            away_probability=0.0,
            actual_result="H",
        )

        self.assertEqual(
            score,
            0.0,
        )

    def test_02_uniform_brier_is_two_thirds(
        self,
    ):
        score = multiclass_brier_score(
            home_probability=1 / 3,
            draw_probability=1 / 3,
            away_probability=1 / 3,
            actual_result="H",
        )

        self.assertTrue(
            math.isclose(
                score,
                2 / 3,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )

    def test_03_log_loss_uses_actual_probability(
        self,
    ):
        score = multiclass_log_loss(
            home_probability=0.50,
            draw_probability=0.25,
            away_probability=0.25,
            actual_result="D",
        )

        self.assertTrue(
            math.isclose(
                score,
                -math.log(0.25),
                rel_tol=0,
                abs_tol=1e-12,
            )
        )

    def test_04_latest_revision_only_for_headline(
        self,
    ):
        evaluations = [
            self._evaluation(
                fixture_id="fx1",
                revision=1,
                predicted_result="A",
                actual_result="H",
                correct=False,
                exact=False,
            ),
            self._evaluation(
                fixture_id="fx1",
                revision=2,
                predicted_result="H",
                actual_result="H",
                correct=True,
                exact=True,
            ),
        ]

        latest = (
            latest_revision_per_fixture(
                evaluations
            )
        )

        self.assertEqual(
            len(latest),
            1,
        )

        self.assertEqual(
            latest[0].revision,
            2,
        )

    def test_05_summary_uses_one_fixture_once(
        self,
    ):
        evaluations = [
            self._evaluation(
                fixture_id="fx1",
                revision=1,
                predicted_result="A",
                actual_result="H",
                correct=False,
                exact=False,
            ),
            self._evaluation(
                fixture_id="fx1",
                revision=2,
                predicted_result="H",
                actual_result="H",
                correct=True,
                exact=True,
            ),
            self._evaluation(
                fixture_id="fx2",
                revision=1,
                predicted_result="D",
                actual_result="A",
                correct=False,
                exact=False,
            ),
        ]

        summary = build_summary(
            evaluations
        )

        self.assertEqual(
            summary[
                "all_published_revision_count"
            ],
            3,
        )

        self.assertEqual(
            summary[
                "resolved_fixture_count"
            ],
            2,
        )

        self.assertEqual(
            summary[
                "correct_result_picks"
            ],
            1,
        )

        self.assertEqual(
            summary[
                "result_pick_accuracy"
            ],
            0.5,
        )

    def test_06_small_sample_warning(
        self,
    ):
        evaluations = [
            self._evaluation(
                fixture_id="fx1",
                revision=1,
                predicted_result="H",
                actual_result="H",
                correct=True,
                exact=False,
            )
        ]

        summary = build_summary(
            evaluations
        )

        self.assertEqual(
            summary[
                "sample_warning"
            ],
            (
                "INSUFFICIENT_SAMPLE_FOR_"
                "RELIABLE_PERFORMANCE_CLAIMS"
            ),
        )


if __name__ == "__main__":
    unittest.main()