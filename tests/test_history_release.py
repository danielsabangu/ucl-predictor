from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.history.archive import load_records
from src.history.evaluation import (
    build_evaluations,
    build_summary,
)
from src.history.results_archive import load_results
from src.production import release
from src.production.run_pipeline import (
    PIPELINE_STEPS,
)


ROOT = Path(__file__).resolve().parents[1]

HISTORY_DIR = (
    ROOT
    / "data"
    / "app"
    / "history"
)


class HistoryReleaseTests(unittest.TestCase):
    def test_01_public_history_files_exist(self):
        expected = [
            "predictions.jsonl",
            "results.jsonl",
            "evaluations.jsonl",
            "summary.json",
        ]

        for filename in expected:
            self.assertTrue(
                (
                    HISTORY_DIR
                    / filename
                ).exists(),
                filename,
            )

    def test_02_results_reference_archived_predictions(
        self,
    ):
        predictions = load_records(
            HISTORY_DIR
            / "predictions.jsonl"
        )

        results = load_results(
            HISTORY_DIR
            / "results.jsonl"
        )

        fixture_ids = {
            prediction.fixture_id
            for prediction in predictions
        }

        self.assertTrue(
            all(
                result.fixture_id
                in fixture_ids
                for result in results
            )
        )

    def test_03_evaluations_are_reproducible(
        self,
    ):
        expected = [
            evaluation.to_dict()
            for evaluation in build_evaluations(
                predictions_path=(
                    HISTORY_DIR
                    / "predictions.jsonl"
                ),
                results_path=(
                    HISTORY_DIR
                    / "results.jsonl"
                ),
            )
        ]

        actual = []

        with (
            HISTORY_DIR
            / "evaluations.jsonl"
        ).open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                if line.strip():
                    actual.append(
                        json.loads(line)
                    )

        self.assertEqual(
            actual,
            expected,
        )

    def test_04_summary_is_reproducible(
        self,
    ):
        evaluations = build_evaluations(
            predictions_path=(
                HISTORY_DIR
                / "predictions.jsonl"
            ),
            results_path=(
                HISTORY_DIR
                / "results.jsonl"
            ),
        )

        expected = build_summary(
            evaluations
        )

        with (
            HISTORY_DIR
            / "summary.json"
        ).open(
            "r",
            encoding="utf-8",
        ) as handle:
            actual = json.load(handle)

        self.assertEqual(
            actual,
            expected,
        )

    def test_05_release_allowlist_contains_history(
        self,
    ):
        expected_paths = {
            "data/app/history/predictions.jsonl",
            "data/app/history/results.jsonl",
            "data/app/history/evaluations.jsonl",
            "data/app/history/summary.json",
        }

        self.assertTrue(
            expected_paths.issubset(
                release.ALLOWED_STAGED_PATHS
            )
        )

    def test_06_fixture_identity_follows_fixture_refresh(
        self,
    ):
        modules = [
            step.module
            for step in PIPELINE_STEPS
        ]

        fixture_index = modules.index(
            "src.data.collect_fixtures"
        )

        identity_index = modules.index(
            "src.utils.resolve_current_ucl_clubs"
        )

        self.assertEqual(
            identity_index,
            fixture_index + 1,
        )


if __name__ == "__main__":
    unittest.main()