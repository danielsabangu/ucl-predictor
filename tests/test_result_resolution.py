from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from src.history.archive import append_records
from src.history.resolver import (
    resolve_finished_results,
)
from src.history.results_archive import (
    ImmutableResultConflict,
    load_results,
)
from src.history.schemas import (
    SCHEMA_VERSION,
    PredictionRecord,
)


class ResultResolutionTests(
    unittest.TestCase
):
    def setUp(self):
        self.temp_dir = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temp_dir.name
        )

        self.predictions_path = (
            self.root
            / "predictions.jsonl"
        )

        self.results_path = (
            self.root
            / "results.jsonl"
        )

        self.fixtures_path = (
            self.root
            / "fixtures.csv"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _prediction(
        self,
        *,
        fixture_id: str = "fx_test",
        prediction_id: str = "pred_test",
        home_team: str = "Home FC",
        away_team: str = "Away FC",
        kickoff_utc: str = (
            "2026-10-20T19:00:00Z"
        ),
    ) -> PredictionRecord:
        return PredictionRecord(
            schema_version=SCHEMA_VERSION,

            prediction_id=prediction_id,
            fixture_id=fixture_id,
            fixture_id_source="test",

            revision=1,
            prediction_type=(
                "LIVE_FORWARD_TEST"
            ),
            prediction_status="PUBLISHED",
            record_origin="TEST",

            home_team=home_team,
            away_team=away_team,
            kickoff_utc=kickoff_utc,
            published_at_utc=(
                "2026-10-20T18:00:00Z"
            ),

            model_version=(
                "ucl_predictor_v1"
            ),
            source_run_id="RUN001",

            data_cutoff_utc=None,
            dataset_version=None,

            home_probability=0.50,
            draw_probability=0.25,
            away_probability=0.25,

            predicted_result="H",
            predicted_selection=home_team,

            max_probability=0.50,

            evidence_score=0.60,
            evidence_label="MODERATE",
            evidence_caps=None,

            model_top_two_margin=None,
            model_decisiveness_profile=None,

            elo_features_available=None,
            ucl_form_features_available=None,
            domestic_features_available=None,
            squad_features_available=None,

            overall_features_available=None,
            overall_feature_availability=None,

            predicted_home_goals=2,
            predicted_away_goals=1,
            predicted_scoreline="2-1",
            predicted_scoreline_probability=0.10,

            global_modal_scoreline=None,
            global_modal_result=None,

            score_home_lambda=None,
            score_away_lambda=None,
            score_total_lambda=None,

            score_bridge_max_error_pp=None,
            score_bridge_quality=None,

            source_snapshot_hash=(
                "a" * 64
            ),
        )

    def _write_fixture(
        self,
        *,
        source_match_id: str = "1001",
        status: str = "FINISHED",
        home_team: str = "Home FC",
        away_team: str = "Away FC",
        home_goals: str = "2",
        away_goals: str = "1",
        result: str = "H",
        kickoff_utc: str = (
            "2026-10-20T19:00:00Z"
        ),
        last_updated: str = (
            "2026-10-20T21:00:00Z"
        ),
    ):
        fields = [
            "source_match_id",
            "date_utc",
            "competition",
            "stage",
            "matchday",
            "status",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "venue",
            "last_updated",
            "source",
        ]

        row = {
            "source_match_id": (
                source_match_id
            ),
            "date_utc": kickoff_utc,
            "competition": "UCL",
            "stage": "LEAGUE_STAGE",
            "matchday": "1",
            "status": status,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
            "venue": "Test Stadium",
            "last_updated": (
                last_updated
            ),
            "source": "test_provider",
        }

        with self.fixtures_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerow(row)

    def test_01_finished_result_is_imported(
        self,
    ):
        append_records(
            self.predictions_path,
            [self._prediction()],
        )

        self._write_fixture()

        summary = (
            resolve_finished_results(
                prediction_archive=(
                    self.predictions_path
                ),
                fixture_source=(
                    self.fixtures_path
                ),
                result_archive=(
                    self.results_path
                ),
            )
        )

        self.assertEqual(
            summary.imported,
            1,
        )

        records = load_results(
            self.results_path
        )

        self.assertEqual(
            len(records),
            1,
        )

        self.assertEqual(
            records[0].actual_result,
            "H",
        )

        self.assertEqual(
            records[0].home_goals,
            2,
        )

        self.assertEqual(
            records[0].away_goals,
            1,
        )

    def test_02_resolution_is_idempotent(
        self,
    ):
        append_records(
            self.predictions_path,
            [self._prediction()],
        )

        self._write_fixture()

        first = resolve_finished_results(
            prediction_archive=(
                self.predictions_path
            ),
            fixture_source=(
                self.fixtures_path
            ),
            result_archive=(
                self.results_path
            ),
        )

        second = resolve_finished_results(
            prediction_archive=(
                self.predictions_path
            ),
            fixture_source=(
                self.fixtures_path
            ),
            result_archive=(
                self.results_path
            ),
        )

        self.assertEqual(
            first.imported,
            1,
        )

        self.assertEqual(
            second.imported,
            0,
        )

        self.assertEqual(
            second.skipped,
            1,
        )

    def test_03_unfinished_fixture_is_ignored(
        self,
    ):
        append_records(
            self.predictions_path,
            [self._prediction()],
        )

        self._write_fixture(
            status="TIMED",
            home_goals="",
            away_goals="",
            result="",
        )

        summary = (
            resolve_finished_results(
                prediction_archive=(
                    self.predictions_path
                ),
                fixture_source=(
                    self.fixtures_path
                ),
                result_archive=(
                    self.results_path
                ),
            )
        )

        self.assertEqual(
            summary.imported,
            0,
        )

        self.assertEqual(
            summary.finished_seen,
            0,
        )

    def test_04_explicit_team_alias_matches(
        self,
    ):
        prediction = self._prediction(
            fixture_id="fx_shakhtar",
            prediction_id=(
                "pred_shakhtar"
            ),
            home_team="PSV Eindhoven",
            away_team="Shakhtar",
        )

        append_records(
            self.predictions_path,
            [prediction],
        )

        self._write_fixture(
            home_team="PSV Eindhoven",
            away_team=(
                "Shakhtar Donetsk"
            ),
            home_goals="1",
            away_goals="1",
            result="D",
        )

        summary = (
            resolve_finished_results(
                prediction_archive=(
                    self.predictions_path
                ),
                fixture_source=(
                    self.fixtures_path
                ),
                result_archive=(
                    self.results_path
                ),
            )
        )

        self.assertEqual(
            summary.matched_finished,
            1,
        )

        self.assertEqual(
            summary.imported,
            1,
        )

    def test_05_unmatched_finished_is_not_failure(
        self,
    ):
        append_records(
            self.predictions_path,
            [self._prediction()],
        )

        self._write_fixture(
            home_team="Other FC",
            away_team="Different FC",
        )

        summary = (
            resolve_finished_results(
                prediction_archive=(
                    self.predictions_path
                ),
                fixture_source=(
                    self.fixtures_path
                ),
                result_archive=(
                    self.results_path
                ),
            )
        )

        self.assertEqual(
            summary.finished_seen,
            1,
        )

        self.assertEqual(
            summary.matched_finished,
            0,
        )

        self.assertEqual(
            summary.unmatched_finished,
            1,
        )

        self.assertEqual(
            summary.imported,
            0,
        )

    def test_06_wrong_provider_result_rejected(
        self,
    ):
        append_records(
            self.predictions_path,
            [self._prediction()],
        )

        self._write_fixture(
            home_goals="2",
            away_goals="1",
            result="A",
        )

        with self.assertRaises(
            ValueError
        ):
            resolve_finished_results(
                prediction_archive=(
                    self.predictions_path
                ),
                fixture_source=(
                    self.fixtures_path
                ),
                result_archive=(
                    self.results_path
                ),
            )

    def test_07_existing_result_cannot_change(
        self,
    ):
        append_records(
            self.predictions_path,
            [self._prediction()],
        )

        self._write_fixture(
            home_goals="2",
            away_goals="1",
            result="H",
        )

        resolve_finished_results(
            prediction_archive=(
                self.predictions_path
            ),
            fixture_source=(
                self.fixtures_path
            ),
            result_archive=(
                self.results_path
            ),
        )

        self._write_fixture(
            home_goals="3",
            away_goals="1",
            result="H",
        )

        with self.assertRaises(
            ImmutableResultConflict
        ):
            resolve_finished_results(
                prediction_archive=(
                    self.predictions_path
                ),
                fixture_source=(
                    self.fixtures_path
                ),
                result_archive=(
                    self.results_path
                ),
            )



    def test_08_provider_metadata_refresh_is_idempotent(
        self,
    ):
        append_records(
            self.predictions_path,
            [self._prediction()],
        )

        self._write_fixture(
            home_goals="2",
            away_goals="1",
            result="H",
            last_updated=(
                "2026-10-20T21:00:00Z"
            ),
        )

        first = resolve_finished_results(
            prediction_archive=(
                self.predictions_path
            ),
            fixture_source=(
                self.fixtures_path
            ),
            result_archive=(
                self.results_path
            ),
        )

        self.assertEqual(
            first.imported,
            1,
        )

        stored_before = load_results(
            self.results_path
        )[0]

        self._write_fixture(
            home_goals="2",
            away_goals="1",
            result="H",
            last_updated=(
                "2026-10-21T09:30:00Z"
            ),
        )

        second = resolve_finished_results(
            prediction_archive=(
                self.predictions_path
            ),
            fixture_source=(
                self.fixtures_path
            ),
            result_archive=(
                self.results_path
            ),
        )

        self.assertEqual(
            second.imported,
            0,
        )

        self.assertEqual(
            second.skipped,
            1,
        )

        stored_after = load_results(
            self.results_path
        )[0]

        self.assertEqual(
            stored_after,
            stored_before,
        )

        self.assertEqual(
            stored_after.provider_last_updated_utc,
            "2026-10-20T21:00:00Z",
        )


if __name__ == "__main__":
    unittest.main()