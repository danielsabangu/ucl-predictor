from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.history.evaluation import (
    multiclass_brier_score,
    multiclass_log_loss,
)
from src.research import evaluate_v2_shadow as evaluator


class V2ShadowEvaluationTests(
    unittest.TestCase
):
    def setUp(
        self,
    ) -> None:
        self.temp_dir = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temp_dir.name
        )

        self.shadow_date = (
            "2026-10-13"
        )

        self.comparison_path = (
            self.root
            / "comparison.csv"
        )

        self.metadata_path = (
            self.root
            / "metadata.json"
        )

        self.results_path = (
            self.root
            / "results.jsonl"
        )

        self.output_root = (
            self.root
            / "evaluations"
        )

        self.output_paths = (
            evaluator.evaluation_output_paths(
                self.shadow_date,
                root=(
                    self.output_root
                ),
            )
        )

        self.comparison = (
            self._build_comparison()
        )

        self.metadata = (
            self._build_metadata()
        )

        self._write_snapshot(
            comparison=(
                self.comparison
            ),
            metadata=(
                self.metadata
            ),
        )

    def tearDown(
        self,
    ) -> None:
        self.temp_dir.cleanup()

    # ========================================================
    # FIXTURES
    # ========================================================

    def _build_comparison(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source_match_id": (
                        "1001"
                    ),
                    "kickoff_utc": (
                        "2026-10-13T18:00:00+00:00"
                    ),
                    "home_team": (
                        "Alpha FC"
                    ),
                    "away_team": (
                        "Beta FC"
                    ),

                    "v1_home_probability": (
                        0.60
                    ),
                    "v1_draw_probability": (
                        0.20
                    ),
                    "v1_away_probability": (
                        0.20
                    ),
                    "v1_predicted_result": (
                        "H"
                    ),

                    "v2_home_probability": (
                        0.70
                    ),
                    "v2_draw_probability": (
                        0.15
                    ),
                    "v2_away_probability": (
                        0.15
                    ),
                    "v2_predicted_result": (
                        "H"
                    ),
                },
                {
                    "source_match_id": (
                        "1002"
                    ),
                    "kickoff_utc": (
                        "2026-10-13T19:00:00+00:00"
                    ),
                    "home_team": (
                        "Gamma FC"
                    ),
                    "away_team": (
                        "Delta FC"
                    ),

                    "v1_home_probability": (
                        0.45
                    ),
                    "v1_draw_probability": (
                        0.20
                    ),
                    "v1_away_probability": (
                        0.35
                    ),
                    "v1_predicted_result": (
                        "H"
                    ),

                    "v2_home_probability": (
                        0.25
                    ),
                    "v2_draw_probability": (
                        0.50
                    ),
                    "v2_away_probability": (
                        0.25
                    ),
                    "v2_predicted_result": (
                        "D"
                    ),
                },
                {
                    "source_match_id": (
                        "1003"
                    ),
                    "kickoff_utc": (
                        "2026-10-13T20:00:00+00:00"
                    ),
                    "home_team": (
                        "Epsilon FC"
                    ),
                    "away_team": (
                        "Zeta FC"
                    ),

                    "v1_home_probability": (
                        0.20
                    ),
                    "v1_draw_probability": (
                        0.20
                    ),
                    "v1_away_probability": (
                        0.60
                    ),
                    "v1_predicted_result": (
                        "A"
                    ),

                    "v2_home_probability": (
                        0.25
                    ),
                    "v2_draw_probability": (
                        0.20
                    ),
                    "v2_away_probability": (
                        0.55
                    ),
                    "v2_predicted_result": (
                        "A"
                    ),
                },
            ]
        )

    def _build_metadata(
        self,
    ) -> dict:
        return {
            "created_at_utc": (
                "2026-10-13T12:00:00+00:00"
            ),

            "status": (
                "shadow_research_only"
            ),

            "production_promoted": (
                False
            ),

            "fixture_count": (
                3
            ),

            "fixture_date_utc": (
                "2026-10-13"
            ),
        }

    def _write_snapshot(
        self,
        *,
        comparison: pd.DataFrame,
        metadata: dict,
    ) -> None:
        comparison.to_csv(
            self.comparison_path,
            index=False,
            encoding="utf-8",
        )

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                metadata,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )

            handle.write(
                "\n"
            )

    def _result(
        self,
        *,
        source_match_id: str,
        fixture_id: str,
        home_team: str,
        away_team: str,
        kickoff_utc: str,
        home_goals: int,
        away_goals: int,
    ) -> SimpleNamespace:
        if home_goals > away_goals:
            actual_result = "H"

        elif home_goals < away_goals:
            actual_result = "A"

        else:
            actual_result = "D"

        return SimpleNamespace(
            result_id=(
                f"result_{source_match_id}"
            ),

            source_match_id=(
                source_match_id
            ),

            fixture_id=(
                fixture_id
            ),

            home_team=(
                home_team
            ),

            away_team=(
                away_team
            ),

            kickoff_utc=(
                kickoff_utc
            ),

            status=(
                "FINISHED"
            ),

            home_goals=(
                home_goals
            ),

            away_goals=(
                away_goals
            ),

            actual_result=(
                actual_result
            ),
        )

    def _complete_results(
        self,
    ) -> list[
        SimpleNamespace
    ]:
        return [
            self._result(
                source_match_id="1001",
                fixture_id="fixture_1001",
                home_team="Alpha FC",
                away_team="Beta FC",
                kickoff_utc=(
                    "2026-10-13T18:00:00+00:00"
                ),
                home_goals=2,
                away_goals=0,
            ),

            self._result(
                source_match_id="1002",
                fixture_id="fixture_1002",
                home_team="Gamma FC",
                away_team="Delta FC",
                kickoff_utc=(
                    "2026-10-13T19:00:00+00:00"
                ),
                home_goals=1,
                away_goals=1,
            ),

            self._result(
                source_match_id="1003",
                fixture_id="fixture_1003",
                home_team="Epsilon FC",
                away_team="Zeta FC",
                kickoff_utc=(
                    "2026-10-13T20:00:00+00:00"
                ),
                home_goals=0,
                away_goals=1,
            ),
        ]

    # ========================================================
    # TESTS
    # ========================================================

    def test_01_existing_metric_definitions_are_reused(
        self,
    ) -> None:
        row = pd.Series(
            {
                "actual_result": (
                    "H"
                ),

                "v1_home_probability": (
                    0.60
                ),
                "v1_draw_probability": (
                    0.20
                ),
                "v1_away_probability": (
                    0.20
                ),
                "v1_predicted_result": (
                    "H"
                ),
            }
        )

        metrics = (
            evaluator.model_fixture_metrics(
                row,
                prefix="v1",
            )
        )

        expected_brier = (
            multiclass_brier_score(
                home_probability=0.60,
                draw_probability=0.20,
                away_probability=0.20,
                actual_result="H",
            )
        )

        expected_log_loss = (
            multiclass_log_loss(
                home_probability=0.60,
                draw_probability=0.20,
                away_probability=0.20,
                actual_result="H",
            )
        )

        self.assertAlmostEqual(
            metrics[
                "multiclass_brier"
            ],
            expected_brier,
            places=15,
        )

        self.assertAlmostEqual(
            metrics[
                "log_loss"
            ],
            expected_log_loss,
            places=15,
        )

    def test_02_shadow_snapshot_must_be_prematch(
        self,
    ) -> None:
        metadata = dict(
            self.metadata
        )

        metadata[
            "created_at_utc"
        ] = (
            "2026-10-13T18:00:00+00:00"
        )

        self._write_snapshot(
            comparison=(
                self.comparison
            ),
            metadata=(
                metadata
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            evaluator.load_shadow_snapshot(
                shadow_date=(
                    self.shadow_date
                ),
                comparison_path=(
                    self.comparison_path
                ),
                metadata_path=(
                    self.metadata_path
                ),
            )

    def test_03_shadow_probability_contract_is_enforced(
        self,
    ) -> None:
        comparison = (
            self.comparison.copy()
        )

        comparison.loc[
            0,
            "v1_home_probability",
        ] = 0.90

        self._write_snapshot(
            comparison=(
                comparison
            ),
            metadata=(
                self.metadata
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            evaluator.load_shadow_snapshot(
                shadow_date=(
                    self.shadow_date
                ),
                comparison_path=(
                    self.comparison_path
                ),
                metadata_path=(
                    self.metadata_path
                ),
            )

    def test_04_predicted_result_must_match_argmax(
        self,
    ) -> None:
        comparison = (
            self.comparison.copy()
        )

        comparison.loc[
            0,
            "v1_predicted_result",
        ] = "A"

        self._write_snapshot(
            comparison=(
                comparison
            ),
            metadata=(
                self.metadata
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            evaluator.load_shadow_snapshot(
                shadow_date=(
                    self.shadow_date
                ),
                comparison_path=(
                    self.comparison_path
                ),
                metadata_path=(
                    self.metadata_path
                ),
            )

    def test_05_partial_results_never_calculate_metrics(
        self,
    ) -> None:
        partial_results = (
            self._complete_results()[
                :2
            ]
        )

        with patch.object(
            evaluator,
            "load_results",
            return_value=(
                partial_results
            ),
        ):
            run = (
                evaluator.evaluate_shadow_set(
                    shadow_date=(
                        self.shadow_date
                    ),

                    comparison_path=(
                        self.comparison_path
                    ),

                    metadata_path=(
                        self.metadata_path
                    ),

                    results_path=(
                        self.results_path
                    ),

                    output_paths=(
                        self.output_paths
                    ),

                    write_artifacts=True,
                )
            )

        self.assertEqual(
            run.status,
            "WAITING_FOR_RESULTS",
        )

        self.assertEqual(
            run.fixture_count,
            3,
        )

        self.assertEqual(
            run.resolved_count,
            2,
        )

        self.assertEqual(
            run.missing_source_match_ids,
            (
                "1003",
            ),
        )

        self.assertIsNone(
            run.evaluation
        )

        self.assertIsNone(
            run.summary
        )

        self.assertFalse(
            self.output_paths
            .fixture_evaluations
            .exists()
        )

        self.assertFalse(
            self.output_paths
            .summary
            .exists()
        )

    def test_06_complete_results_unlock_evaluation(
        self,
    ) -> None:
        with patch.object(
            evaluator,
            "load_results",
            return_value=(
                self._complete_results()
            ),
        ):
            run = (
                evaluator.evaluate_shadow_set(
                    shadow_date=(
                        self.shadow_date
                    ),

                    comparison_path=(
                        self.comparison_path
                    ),

                    metadata_path=(
                        self.metadata_path
                    ),

                    results_path=(
                        self.results_path
                    ),

                    output_paths=(
                        self.output_paths
                    ),

                    write_artifacts=False,
                )
            )

        self.assertEqual(
            run.status,
            "COMPLETE",
        )

        self.assertEqual(
            run.fixture_count,
            3,
        )

        self.assertEqual(
            run.resolved_count,
            3,
        )

        self.assertEqual(
            run.missing_source_match_ids,
            tuple(),
        )

        self.assertIsNotNone(
            run.evaluation
        )

        self.assertIsNotNone(
            run.summary
        )

        assert (
            run.evaluation
            is not None
        )

        assert (
            run.summary
            is not None
        )

        self.assertEqual(
            len(
                run.evaluation
            ),
            3,
        )

        self.assertEqual(
            run.summary[
                "status"
            ],
            "COMPLETE",
        )

        self.assertTrue(
            run.summary[
                "all_shadow_fixtures_resolved"
            ]
        )

        self.assertFalse(
            run.summary[
                "production_promoted"
            ]
        )

        self.assertFalse(
            run.summary[
                "promotion_decision_made"
            ]
        )

    def test_07_locked_primary_metrics_favour_v2_in_synthetic_set(
        self,
    ) -> None:
        with patch.object(
            evaluator,
            "load_results",
            return_value=(
                self._complete_results()
            ),
        ):
            run = (
                evaluator.evaluate_shadow_set(
                    shadow_date=(
                        self.shadow_date
                    ),

                    comparison_path=(
                        self.comparison_path
                    ),

                    metadata_path=(
                        self.metadata_path
                    ),

                    results_path=(
                        self.results_path
                    ),

                    output_paths=(
                        self.output_paths
                    ),

                    write_artifacts=False,
                )
            )

        assert (
            run.summary
            is not None
        )

        summary = run.summary

        self.assertEqual(
            summary[
                "primary_metrics"
            ],
            [
                "mean_log_loss",
                "mean_multiclass_brier",
            ],
        )

        self.assertLess(
            summary[
                "v2"
            ][
                "mean_log_loss"
            ],
            summary[
                "v1"
            ][
                "mean_log_loss"
            ],
        )

        self.assertLess(
            summary[
                "v2"
            ][
                "mean_multiclass_brier"
            ],
            summary[
                "v1"
            ][
                "mean_multiclass_brier"
            ],
        )

        self.assertLess(
            summary[
                "v2_minus_v1"
            ][
                "mean_log_loss_v2_minus_v1"
            ],
            0.0,
        )

        self.assertLess(
            summary[
                "v2_minus_v1"
            ][
                "mean_multiclass_brier_v2_minus_v1"
            ],
            0.0,
        )

    def test_08_classification_metrics_are_locked(
        self,
    ) -> None:
        actual = [
            "H",
            "D",
            "A",
        ]

        v1_predicted = [
            "H",
            "H",
            "A",
        ]

        v2_predicted = [
            "H",
            "D",
            "A",
        ]

        v1 = (
            evaluator.classification_metrics(
                actual=(
                    actual
                ),
                predicted=(
                    v1_predicted
                ),
            )
        )

        v2 = (
            evaluator.classification_metrics(
                actual=(
                    actual
                ),
                predicted=(
                    v2_predicted
                ),
            )
        )

        self.assertAlmostEqual(
            v1[
                "accuracy"
            ],
            2.0 / 3.0,
            places=15,
        )

        self.assertAlmostEqual(
            v1[
                "macro_f1"
            ],
            (
                (
                    2.0 / 3.0
                )
                +
                0.0
                +
                1.0
            )
            /
            3.0,
            places=15,
        )

        self.assertEqual(
            v1[
                "draw_recall"
            ],
            0.0,
        )

        self.assertEqual(
            v2[
                "accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            v2[
                "macro_f1"
            ],
            1.0,
        )

        self.assertEqual(
            v2[
                "draw_recall"
            ],
            1.0,
        )

    def test_09_draw_recall_is_null_without_actual_draws(
        self,
    ) -> None:
        metrics = (
            evaluator.classification_metrics(
                actual=[
                    "H",
                    "A",
                ],
                predicted=[
                    "H",
                    "A",
                ],
            )
        )

        self.assertIsNone(
            metrics[
                "draw_recall"
            ]
        )

    def test_10_result_identity_mismatch_is_rejected(
        self,
    ) -> None:
        results = (
            self._complete_results()
        )

        results[
            0
        ] = self._result(
            source_match_id="1001",
            fixture_id="fixture_1001",
            home_team="WRONG TEAM",
            away_team="Beta FC",
            kickoff_utc=(
                "2026-10-13T18:00:00+00:00"
            ),
            home_goals=2,
            away_goals=0,
        )

        comparison, _ = (
            evaluator.load_shadow_snapshot(
                shadow_date=(
                    self.shadow_date
                ),
                comparison_path=(
                    self.comparison_path
                ),
                metadata_path=(
                    self.metadata_path
                ),
            )
        )

        with self.assertRaises(
            ValueError
        ):
            evaluator.match_verified_results(
                comparison=(
                    comparison
                ),
                result_records=(
                    results
                ),
            )

    def test_11_score_result_disagreement_is_rejected(
        self,
    ) -> None:
        bad_result = (
            self._result(
                source_match_id="1001",
                fixture_id="fixture_1001",
                home_team="Alpha FC",
                away_team="Beta FC",
                kickoff_utc=(
                    "2026-10-13T18:00:00+00:00"
                ),
                home_goals=2,
                away_goals=0,
            )
        )

        bad_result.actual_result = (
            "A"
        )

        comparison, _ = (
            evaluator.load_shadow_snapshot(
                shadow_date=(
                    self.shadow_date
                ),
                comparison_path=(
                    self.comparison_path
                ),
                metadata_path=(
                    self.metadata_path
                ),
            )
        )

        with self.assertRaises(
            ValueError
        ):
            evaluator.match_verified_results(
                comparison=(
                    comparison
                ),
                result_records=[
                    bad_result,
                ],
            )

    def test_12_written_evaluation_is_immutable(
        self,
    ) -> None:
        complete_results = (
            self._complete_results()
        )

        with patch.object(
            evaluator,
            "load_results",
            return_value=(
                complete_results
            ),
        ):
            first = (
                evaluator.evaluate_shadow_set(
                    shadow_date=(
                        self.shadow_date
                    ),

                    comparison_path=(
                        self.comparison_path
                    ),

                    metadata_path=(
                        self.metadata_path
                    ),

                    results_path=(
                        self.results_path
                    ),

                    output_paths=(
                        self.output_paths
                    ),

                    write_artifacts=True,
                )
            )

        self.assertEqual(
            first.status,
            "COMPLETE",
        )

        self.assertTrue(
            self.output_paths
            .fixture_evaluations
            .exists()
        )

        self.assertTrue(
            self.output_paths
            .summary
            .exists()
        )

        with patch.object(
            evaluator,
            "load_results",
            return_value=(
                complete_results
            ),
        ):
            with self.assertRaises(
                FileExistsError
            ):
                evaluator.evaluate_shadow_set(
                    shadow_date=(
                        self.shadow_date
                    ),

                    comparison_path=(
                        self.comparison_path
                    ),

                    metadata_path=(
                        self.metadata_path
                    ),

                    results_path=(
                        self.results_path
                    ),

                    output_paths=(
                        self.output_paths
                    ),

                    write_artifacts=True,
                )


if __name__ == "__main__":
    unittest.main()