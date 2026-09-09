from __future__ import annotations

import json
import unittest
from datetime import timedelta

import pandas as pd

from src.production.app_health import (
    evaluate_app_health,
    filter_upcoming_predictions,
)

from src.production.app_sources import (
    PUBLIC_METADATA_FILE,
    PUBLIC_PREDICTIONS_FILE,
    PUBLIC_STATUS_FILE,
    public_source,
)

from src.production.club_crests import (
    get_crest_path,
    load_crest_manifest,
)

from src.production.receipt_exports import (
    build_receipt_pdf,
    build_receipt_png,
)


# ============================================================
# EXPECTED PUBLIC CONTRACT
# ============================================================

EXPECTED_PUBLIC_COLUMNS = {
    "kickoff_utc",
    "home_team",
    "away_team",

    "home_probability",
    "draw_probability",
    "away_probability",

    "predicted_result",
    "predicted_selection",
    "max_probability",

    "confidence",
    "confidence_percent",
    "confidence_label",
    "confidence_caps",

    "model_top_two_margin",
    "model_decisiveness_profile",

    "elo_features_available",
    "ucl_form_features_available",
    "domestic_features_available",
    "squad_features_available",

    "overall_features_available",
    "overall_feature_availability",

    "predicted_home_goals",
    "predicted_away_goals",
    "predicted_scoreline",

    "predicted_scoreline_probability",

    "global_modal_scoreline",
    "global_modal_result",

    "score_home_lambda",
    "score_away_lambda",
    "score_total_lambda",

    "score_bridge_max_error_pp",
    "score_bridge_quality",
}


# ============================================================
# MODEL FEATURE ARCHITECTURE
# ============================================================

ELO_FEATURE_COUNT = 6

UCL_FORM_FEATURE_COUNT = 81

DOMESTIC_FEATURE_COUNT = 102

SQUAD_FEATURE_COUNT = 101

PRODUCTION_FEATURE_COUNT = (
    ELO_FEATURE_COUNT
    +
    UCL_FORM_FEATURE_COUNT
    +
    DOMESTIC_FEATURE_COUNT
    +
    SQUAD_FEATURE_COUNT
)


# ============================================================
# TESTS
# ============================================================


class PublicDeploymentTests(
    unittest.TestCase
):

    @classmethod
    def setUpClass(
        cls,
    ) -> None:

        if not PUBLIC_PREDICTIONS_FILE.exists():

            raise RuntimeError(
                "Public prediction snapshot is missing:\n"
                f"{PUBLIC_PREDICTIONS_FILE}"
            )

        if not PUBLIC_STATUS_FILE.exists():

            raise RuntimeError(
                "Public pipeline status is missing:\n"
                f"{PUBLIC_STATUS_FILE}"
            )

        if not PUBLIC_METADATA_FILE.exists():

            raise RuntimeError(
                "Public snapshot metadata is missing:\n"
                f"{PUBLIC_METADATA_FILE}"
            )

        cls.predictions = pd.read_csv(
            PUBLIC_PREDICTIONS_FILE,
            low_memory=False,
        )

        cls.predictions[
            "kickoff_utc"
        ] = pd.to_datetime(
            cls.predictions[
                "kickoff_utc"
            ],
            utc=True,
            errors="raise",
        )

        with PUBLIC_STATUS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            cls.status = json.load(
                file
            )

        with PUBLIC_METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            cls.metadata = json.load(
                file
            )

        cls.source = (
            public_source()
        )


    # ========================================================
    # FILE CONTRACT
    # ========================================================

    def test_01_public_files_exist(
        self,
    ) -> None:

        self.assertTrue(
            PUBLIC_PREDICTIONS_FILE.exists()
        )

        self.assertTrue(
            PUBLIC_STATUS_FILE.exists()
        )

        self.assertTrue(
            PUBLIC_METADATA_FILE.exists()
        )


    def test_02_public_snapshot_not_empty(
        self,
    ) -> None:

        self.assertGreater(
            len(
                self.predictions
            ),
            0,
        )


    def test_03_public_columns_exact(
        self,
    ) -> None:

        actual = set(
            self.predictions.columns
        )

        self.assertEqual(
            actual,
            EXPECTED_PUBLIC_COLUMNS,
            msg=(
                "\nPublic deployment column contract changed.\n"
                f"Missing: "
                f"{sorted(EXPECTED_PUBLIC_COLUMNS - actual)}\n"
                f"Unexpected: "
                f"{sorted(actual - EXPECTED_PUBLIC_COLUMNS)}"
            ),
        )


    # ========================================================
    # PROBABILITY INTEGRITY
    # ========================================================

    def test_04_probabilities_are_finite(
        self,
    ) -> None:

        probabilities = self.predictions[
            [
                "home_probability",
                "draw_probability",
                "away_probability",
            ]
        ].astype(
            float
        )

        self.assertFalse(
            probabilities.isna().any().any()
        )

        finite = (
            probabilities
            .map(
                lambda value:
                    pd.notna(
                        value
                    )
                    and
                    float("-inf")
                    <
                    float(
                        value
                    )
                    <
                    float("inf")
            )
        )

        self.assertTrue(
            finite.all().all()
        )


    def test_05_probabilities_inside_zero_one(
        self,
    ) -> None:

        probabilities = self.predictions[
            [
                "home_probability",
                "draw_probability",
                "away_probability",
            ]
        ].astype(
            float
        )

        self.assertTrue(
            (
                probabilities
                >=
                0.0
            )
            .all()
            .all()
        )

        self.assertTrue(
            (
                probabilities
                <=
                1.0
            )
            .all()
            .all()
        )


    def test_06_probabilities_sum_to_one(
        self,
    ) -> None:

        probabilities = self.predictions[
            [
                "home_probability",
                "draw_probability",
                "away_probability",
            ]
        ].astype(
            float
        )

        sums = (
            probabilities
            .sum(
                axis=1
            )
        )

        maximum_error = float(
            (
                sums
                -
                1.0
            )
            .abs()
            .max()
        )

        self.assertLessEqual(
            maximum_error,
            1e-6,
            msg=(
                "H/D/A probability sum error "
                f"is {maximum_error}"
            ),
        )


    def test_07_model_pick_matches_max_probability(
        self,
    ) -> None:

        mapping = {
            "home_probability": "H",
            "draw_probability": "D",
            "away_probability": "A",
        }

        for _, row in self.predictions.iterrows():

            probability_values = {
                column: float(
                    row[
                        column
                    ]
                )
                for column in mapping
            }

            maximum_column = max(
                probability_values,
                key=(
                    probability_values.get
                ),
            )

            expected_result = mapping[
                maximum_column
            ]

            self.assertEqual(
                str(
                    row[
                        "predicted_result"
                    ]
                ),
                expected_result,
                msg=(
                    f"{row['home_team']} vs "
                    f"{row['away_team']} has a "
                    "predicted result inconsistent "
                    "with its maximum H/D/A probability."
                ),
            )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    def test_08_confidence_is_valid(
        self,
    ) -> None:

        confidence = self.predictions[
            "confidence_percent"
        ].astype(
            float
        )

        self.assertFalse(
            confidence.isna().any()
        )

        self.assertTrue(
            (
                confidence
                >=
                0.0
            ).all()
        )

        self.assertTrue(
            (
                confidence
                <=
                100.0
            ).all()
        )


    # ========================================================
    # FEATURE ARCHITECTURE
    # ========================================================

    def test_09_production_architecture_is_290(
        self,
    ) -> None:

        self.assertEqual(
            PRODUCTION_FEATURE_COUNT,
            290,
        )


    def test_10_feature_availability_never_exceeds_family_size(
        self,
    ) -> None:

        checks = [
            (
                "elo_features_available",
                ELO_FEATURE_COUNT,
            ),

            (
                "ucl_form_features_available",
                UCL_FORM_FEATURE_COUNT,
            ),

            (
                "domestic_features_available",
                DOMESTIC_FEATURE_COUNT,
            ),

            (
                "squad_features_available",
                SQUAD_FEATURE_COUNT,
            ),

            (
                "overall_features_available",
                PRODUCTION_FEATURE_COUNT,
            ),
        ]

        for column, maximum in checks:

            values = pd.to_numeric(
                self.predictions[
                    column
                ],
                errors="raise",
            )

            self.assertTrue(
                (
                    values
                    >=
                    0
                ).all(),
                msg=(
                    f"{column} contains "
                    "negative availability."
                ),
            )

            self.assertTrue(
                (
                    values
                    <=
                    maximum
                ).all(),
                msg=(
                    f"{column} exceeds "
                    f"maximum {maximum}."
                ),
            )


    def test_11_overall_feature_ratio_matches_count(
        self,
    ) -> None:

        expected = (
            self.predictions[
                "overall_features_available"
            ].astype(
                float
            )
            /
            PRODUCTION_FEATURE_COUNT
        )

        actual = self.predictions[
            "overall_feature_availability"
        ].astype(
            float
        )

        error = (
            expected
            -
            actual
        ).abs()

        self.assertLessEqual(
            float(
                error.max()
            ),
            1e-6,
        )


    # ========================================================
    # SCORELINE
    # ========================================================

    def test_12_scorelines_exist(
        self,
    ) -> None:

        self.assertFalse(
            self.predictions[
                "predicted_scoreline"
            ]
            .isna()
            .any()
        )

        self.assertFalse(
            self.predictions[
                "predicted_home_goals"
            ]
            .isna()
            .any()
        )

        self.assertFalse(
            self.predictions[
                "predicted_away_goals"
            ]
            .isna()
            .any()
        )


    def test_13_scoreline_matches_goal_columns(
        self,
    ) -> None:

        for _, row in self.predictions.iterrows():

            expected = (
                f"{int(row['predicted_home_goals'])}"
                "-"
                f"{int(row['predicted_away_goals'])}"
            )

            self.assertEqual(
                str(
                    row[
                        "predicted_scoreline"
                    ]
                ),
                expected,
            )


    # ========================================================
    # PUBLIC STATUS
    # ========================================================

    def test_14_public_status_success(
        self,
    ) -> None:

        self.assertEqual(
            str(
                self.status.get(
                    "status"
                )
            ).upper(),
            "SUCCESS",
        )

        self.assertTrue(
            bool(
                self.status.get(
                    "production_ready"
                )
            )
        )

        self.assertTrue(
            bool(
                self.status.get(
                    "public_snapshot"
                )
            )
        )


    def test_15_public_metadata_matches_snapshot(
        self,
    ) -> None:

        self.assertEqual(
            int(
                self.metadata[
                    "fixture_count"
                ]
            ),
            len(
                self.predictions
            ),
        )

        self.assertEqual(
            int(
                self.metadata[
                    "prediction_columns"
                ]
            ),
            len(
                self.predictions.columns
            ),
        )

        self.assertEqual(
            self.metadata[
                "model_version"
            ],
            "ucl_predictor_v1",
        )


    # ========================================================
    # HEALTH
    # ========================================================

    def test_16_public_health_is_healthy_when_fresh(
        self,
    ) -> None:

        finished_at = pd.Timestamp(
            self.status[
                "finished_at_utc"
            ]
        )

        if finished_at.tzinfo is None:

            finished_at = (
                finished_at
                .tz_localize(
                    "UTC"
                )
            )

        else:

            finished_at = (
                finished_at
                .tz_convert(
                    "UTC"
                )
            )

        test_now = (
            finished_at
            +
            pd.Timedelta(
                hours=2
            )
        )

        health = (
            evaluate_app_health(
                now=test_now,
                source=self.source,
            )
        )

        self.assertEqual(
            health[
                "state"
            ],
            "HEALTHY",
        )

        self.assertEqual(
            health[
                "severity"
            ],
            "GREEN",
        )

        self.assertTrue(
            health[
                "can_serve_predictions"
            ]
        )

        self.assertTrue(
            health[
                "is_public_snapshot"
            ]
        )

        self.assertEqual(
            health[
                "source_mode"
            ],
            "PUBLIC",
        )


    def test_17_stale_health_blocks_predictions(
        self,
    ) -> None:

        finished_at = pd.Timestamp(
            self.status[
                "finished_at_utc"
            ]
        )

        if finished_at.tzinfo is None:

            finished_at = (
                finished_at
                .tz_localize(
                    "UTC"
                )
            )

        else:

            finished_at = (
                finished_at
                .tz_convert(
                    "UTC"
                )
            )

        # Use a time safely past the 24-hour maximum,
        # while still before the first fixture if possible.
        stale_now = (
            finished_at
            +
            pd.Timedelta(
                hours=25
            )
        )

        first_kickoff = (
            self.predictions[
                "kickoff_utc"
            ]
            .min()
        )

        if stale_now >= first_kickoff:

            # The health state may become EXPIRED first if
            # every fixture has already kicked off.
            health = (
                evaluate_app_health(
                    now=stale_now,
                    source=self.source,
                )
            )

            self.assertFalse(
                health[
                    "can_serve_predictions"
                ]
            )

            self.assertIn(
                health[
                    "state"
                ],
                {
                    "STALE",
                    "EXPIRED",
                },
            )

        else:

            health = (
                evaluate_app_health(
                    now=stale_now,
                    source=self.source,
                )
            )

            self.assertEqual(
                health[
                    "state"
                ],
                "STALE",
            )

            self.assertFalse(
                health[
                    "can_serve_predictions"
                ]
            )


    # ========================================================
    # UPCOMING FILTER
    # ========================================================

    def test_18_upcoming_filter_removes_started_fixture(
        self,
    ) -> None:

        ordered = (
            self.predictions
            .sort_values(
                "kickoff_utc"
            )
            .reset_index(
                drop=True
            )
        )

        first_kickoff = pd.Timestamp(
            ordered.iloc[
                0
            ][
                "kickoff_utc"
            ]
        )

        test_now = (
            first_kickoff
            +
            pd.Timedelta(
                seconds=1
            )
        )

        filtered = (
            filter_upcoming_predictions(
                ordered,
                now=test_now,
            )
        )

        self.assertLess(
            len(
                filtered
            ),
            len(
                ordered
            ),
        )

        self.assertTrue(
            (
                filtered[
                    "kickoff_utc"
                ]
                >
                test_now
            ).all()
        )


    # ========================================================
    # CRESTS
    # ========================================================

    def test_19_crest_manifest_exists(
        self,
    ) -> None:

        manifest = (
            load_crest_manifest()
        )

        self.assertGreater(
            len(
                manifest
            ),
            0,
        )


    def test_20_all_published_fixture_teams_have_crests(
        self,
    ) -> None:

        teams = sorted(
            set(
                self.predictions[
                    "home_team"
                ].astype(
                    str
                )
            )
            |
            set(
                self.predictions[
                    "away_team"
                ].astype(
                    str
                )
            )
        )

        missing = []

        for team in teams:

            crest_path = (
                get_crest_path(
                    team
                )
            )

            if (
                crest_path is None
                or
                not crest_path.exists()
            ):

                missing.append(
                    team
                )

        self.assertEqual(
            missing,
            [],
            msg=(
                "Published fixtures are missing "
                f"club crests: {missing}"
            ),
        )


    # ========================================================
    # EXPORTS
    # ========================================================

    def test_21_png_export_signature(
        self,
    ) -> None:

        row = (
            self.predictions
            .sort_values(
                "kickoff_utc"
            )
            .iloc[
                0
            ]
        )

        png = (
            build_receipt_png(
                row
            )
        )

        self.assertGreater(
            len(
                png
            ),
            10_000,
        )

        self.assertTrue(
            png.startswith(
                b"\x89PNG\r\n\x1a\n"
            )
        )


    def test_22_pdf_export_signature(
        self,
    ) -> None:

        row = (
            self.predictions
            .sort_values(
                "kickoff_utc"
            )
            .iloc[
                0
            ]
        )

        pdf = (
            build_receipt_pdf(
                row
            )
        )

        self.assertGreater(
            len(
                pdf
            ),
            10_000,
        )

        self.assertTrue(
            pdf.startswith(
                b"%PDF"
            )
        )


    # ========================================================
    # PUBLIC DATA SAFETY
    # ========================================================

    def test_23_public_snapshot_has_no_obvious_sensitive_content(
        self,
    ) -> None:

        files = [
            PUBLIC_PREDICTIONS_FILE,
            PUBLIC_STATUS_FILE,
            PUBLIC_METADATA_FILE,
        ]

        forbidden_fragments = [
            "football_data_api_key",
            "api_football_key",
            "thesportsdb_api_key",
            "x-auth-token",
            "models/trained",
            "data/raw",
            "data/interim",
            "transfermarkt",
            ".env",
        ]

        for path in files:

            text = path.read_text(
                encoding="utf-8",
            ).lower()

            for fragment in forbidden_fragments:

                self.assertNotIn(
                    fragment,
                    text,
                    msg=(
                        f"Sensitive/internal fragment "
                        f"{fragment!r} found in "
                        f"{path.name}."
                    ),
                )


    def test_24_public_status_does_not_expose_local_paths(
        self,
    ) -> None:

        text = PUBLIC_STATUS_FILE.read_text(
            encoding="utf-8",
        )

        self.assertNotIn(
            "F:\\",
            text,
        )

        self.assertNotIn(
            "/home/",
            text,
        )


    def test_25_public_metadata_does_not_expose_local_paths(
        self,
    ) -> None:

        text = PUBLIC_METADATA_FILE.read_text(
            encoding="utf-8",
        )

        self.assertNotIn(
            "F:\\",
            text,
        )

        self.assertNotIn(
            "/home/",
            text,
        )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )