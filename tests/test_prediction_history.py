from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.history.archive import (
    ImmutableArchiveConflict,
    load_records,
)
from src.history.snapshot_import import (
    import_public_snapshot,
)


class PredictionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.root = Path(
            self.temp_dir.name
        )

        self.app_dir = (
            self.root
            / "data"
            / "app"
        )

        self.app_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.archive_path = (
            self.app_dir
            / "history"
            / "predictions.jsonl"
        )

        self.row = {
            "kickoff_utc": "2026-10-20T19:00:00+00:00",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "home_probability": "0.50",
            "draw_probability": "0.25",
            "away_probability": "0.25",
            "predicted_result": "H",
            "predicted_selection": "Home FC",
            "max_probability": "0.50",
            "confidence": "0.61",
            "confidence_label": "MODERATE",
            "confidence_caps": "",
            "overall_features_available": "200",
            "overall_feature_availability": "0.6896551724",
            "predicted_home_goals": "2",
            "predicted_away_goals": "1",
            "predicted_scoreline": "2-1",
            "predicted_scoreline_probability": "0.10",
        }

        self._write_snapshot(
            run_id="RUN001",
            created_at_utc=(
                "2026-10-20T18:00:00+00:00"
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_prediction_csv(self):
        path = (
            self.app_dir
            / "current_predictions.csv"
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    self.row.keys()
                ),
            )

            writer.writeheader()
            writer.writerow(self.row)

    def _write_snapshot(
        self,
        *,
        run_id: str,
        created_at_utc: str,
    ):
        self._write_prediction_csv()

        metadata = {
            "created_at_utc": created_at_utc,
            "model_version": "ucl_predictor_v1",
            "fixture_count": 1,
            "prediction_columns": len(
                self.row
            ),
            "source_run_id": run_id,
            "probability_model": (
                "Frozen CatBoost"
            ),
        }

        status = {
            "status": "SUCCESS",
            "production_ready": True,
            "run_id": run_id,
        }

        with (
            self.app_dir
            / "snapshot_metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metadata,
                handle,
            )

        with (
            self.app_dir
            / "pipeline_status.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                status,
                handle,
            )

    def test_01_first_import_creates_record(self):
        result = import_public_snapshot(
            self.app_dir,
            self.archive_path,
        )

        self.assertEqual(
            result.imported,
            1,
        )

        self.assertEqual(
            result.skipped,
            0,
        )

        records = load_records(
            self.archive_path
        )

        self.assertEqual(
            len(records),
            1,
        )

        self.assertEqual(
            records[0].revision,
            1,
        )

        self.assertEqual(
            records[0].prediction_type,
            "LIVE_FORWARD_TEST",
        )

        self.assertEqual(
            records[0].prediction_status,
            "RECORDED_PREMATCH",
        )

        self.assertEqual(
            records[0].record_origin,
            "PREMATCH_SNAPSHOT_IMPORT",
        )

        self.assertEqual(
            records[0].published_at_utc,
            None,
        )

        self.assertEqual(
            records[0].recorded_at_utc,
            "2026-10-20T18:00:00Z",
        )

    def test_02_same_run_is_idempotent(self):
        first = import_public_snapshot(
            self.app_dir,
            self.archive_path,
        )

        second = import_public_snapshot(
            self.app_dir,
            self.archive_path,
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

        records = load_records(
            self.archive_path
        )

        self.assertEqual(
            len(records),
            1,
        )

    def test_03_new_run_becomes_new_revision(self):
        import_public_snapshot(
            self.app_dir,
            self.archive_path,
        )

        self._write_snapshot(
            run_id="RUN002",
            created_at_utc=(
                "2026-10-20T18:10:00+00:00"
            ),
        )

        second = import_public_snapshot(
            self.app_dir,
            self.archive_path,
        )

        self.assertEqual(
            second.imported,
            1,
        )

        records = load_records(
            self.archive_path
        )

        self.assertEqual(
            len(records),
            2,
        )

        self.assertEqual(
            records[0].revision,
            1,
        )

        self.assertEqual(
            records[1].revision,
            2,
        )

        self.assertNotEqual(
            records[0].prediction_id,
            records[1].prediction_id,
        )

    def test_04_same_prediction_id_cannot_change(self):
        import_public_snapshot(
            self.app_dir,
            self.archive_path,
        )

        self.row[
            "home_probability"
        ] = "0.55"

        self.row[
            "draw_probability"
        ] = "0.20"

        self.row[
            "away_probability"
        ] = "0.25"

        self.row[
            "max_probability"
        ] = "0.55"

        self._write_snapshot(
            run_id="RUN001",
            created_at_utc=(
                "2026-10-20T18:00:00+00:00"
            ),
        )

        with self.assertRaises(
            ImmutableArchiveConflict
        ):
            import_public_snapshot(
                self.app_dir,
                self.archive_path,
            )

    def test_05_post_kickoff_snapshot_rejected(self):
        self._write_snapshot(
            run_id="RUN003",
            created_at_utc=(
                "2026-10-20T20:00:00+00:00"
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            import_public_snapshot(
                self.app_dir,
                self.archive_path,
            )

    def test_06_invalid_probability_sum_rejected(self):
        self.row[
            "home_probability"
        ] = "0.50"

        self.row[
            "draw_probability"
        ] = "0.20"

        self.row[
            "away_probability"
        ] = "0.20"

        self.row[
            "max_probability"
        ] = "0.50"

        self._write_snapshot(
            run_id="RUN004",
            created_at_utc=(
                "2026-10-20T18:00:00+00:00"
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            import_public_snapshot(
                self.app_dir,
                self.archive_path,
            )


if __name__ == "__main__":
    unittest.main()