from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import predict_live as v1
from src.ml import predict_v2_shadow as shadow


class V2ShadowInferenceTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        (
            cls.v1_model,
            cls.v1_features,
        ) = v1.load_production_model()

        (
            cls.v2_model,
            cls.v2_features,
            cls.v2_metadata,
        ) = shadow.load_v2_candidate(
            cls.v1_features
        )

        cls.uefa = (
            shadow.load_live_uefa_features()
        )

    def test_01_locked_feature_contract(
        self,
    ) -> None:
        self.assertEqual(
            shadow.EXPECTED_V1_FEATURES,
            290,
        )

        self.assertEqual(
            shadow.EXPECTED_V2_FEATURES,
            292,
        )

        self.assertEqual(
            shadow.LOCKED_UEFA_FEATURES,
            [
                "uefa_coefficient_diff",
                "uefa_strength_percentile_diff",
            ],
        )

        self.assertEqual(
            len(
                self.v1_features
            ),
            290,
        )

        self.assertEqual(
            len(
                self.v2_features
            ),
            292,
        )

        self.assertEqual(
            self.v2_features[
                :290
            ],
            self.v1_features,
        )

        self.assertEqual(
            self.v2_features[
                290:
            ],
            shadow.LOCKED_UEFA_FEATURES,
        )

    def test_02_v2_remains_unpromoted(
        self,
    ) -> None:
        self.assertEqual(
            self.v2_metadata[
                "model_version"
            ],
            "ucl_predictor_v2_candidate",
        )

        self.assertEqual(
            self.v2_metadata[
                "status"
            ],
            "candidate_not_promoted",
        )

        self.assertEqual(
            self.v2_metadata[
                "selected_uefa_features"
            ],
            shadow.LOCKED_UEFA_FEATURES,
        )

        self.assertNotEqual(
            shadow.V1_MODEL_FILE.resolve(),
            shadow.V2_MODEL_FILE.resolve(),
        )

    def test_03_live_uefa_layer_is_complete(
        self,
    ) -> None:
        self.assertEqual(
            len(
                self.uefa
            ),
            144,
        )

        self.assertFalse(
            self.uefa[
                "source_match_id_normalized"
            ].duplicated().any()
        )

        self.assertTrue(
            (
                self.uefa[
                    "season"
                ]
                ==
                2026
            ).all()
        )

        self.assertTrue(
            (
                self.uefa[
                    "ranking_year"
                ]
                ==
                2026
            ).all()
        )

        for column in (
            shadow.LOCKED_UEFA_FEATURES
        ):
            values = (
                self.uefa[
                    column
                ]
                .astype(float)
                .to_numpy()
            )

            self.assertTrue(
                np.isfinite(
                    values
                ).all()
            )

        for column in [
            "home_uefa_resolution_method",
            "away_uefa_resolution_method",
        ]:
            contains_fuzzy = (
                self.uefa[
                    column
                ]
                .astype(str)
                .str.upper()
                .str.contains(
                    "FUZZY",
                    regex=False,
                )
            )

            self.assertFalse(
                contains_fuzzy.any()
            )

    def test_04_source_match_id_normalization(
        self,
    ) -> None:
        self.assertEqual(
            shadow.normalize_source_match_id(
                575341
            ),
            "575341",
        )

        self.assertEqual(
            shadow.normalize_source_match_id(
                575341.0
            ),
            "575341",
        )

        self.assertEqual(
            shadow.normalize_source_match_id(
                "575341"
            ),
            "575341",
        )

        self.assertEqual(
            shadow.normalize_source_match_id(
                " 575341 "
            ),
            "575341",
        )

        with self.assertRaises(
            RuntimeError
        ):
            shadow.normalize_source_match_id(
                np.nan
            )

    def test_05_exact_fixture_identity_lookup(
        self,
    ) -> None:
        fixtures = (
            self.uefa[
                [
                    "source_match_id",
                    "date_utc",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(
                3
            )
            .copy()
        )

        lookup = (
            shadow.build_uefa_lookup(
                fixtures,
                self.uefa,
            )
        )

        self.assertEqual(
            len(
                lookup
            ),
            3,
        )

        for fixture in (
            fixtures.itertuples(
                index=False
            )
        ):
            key = (
                shadow.normalize_source_match_id(
                    fixture.source_match_id
                )
            )

            self.assertIn(
                key,
                lookup,
            )

    def test_06_team_identity_mismatch_is_rejected(
        self,
    ) -> None:
        fixtures = (
            self.uefa[
                [
                    "source_match_id",
                    "date_utc",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(
                1
            )
            .copy()
        )

        fixtures.loc[
            fixtures.index[
                0
            ],
            "home_team",
        ] = "INTENTIONAL_TEST_MISMATCH"

        with self.assertRaises(
            RuntimeError
        ):
            shadow.build_uefa_lookup(
                fixtures,
                self.uefa,
            )

    def test_07_kickoff_mismatch_is_rejected(
        self,
    ) -> None:
        fixtures = (
            self.uefa[
                [
                    "source_match_id",
                    "date_utc",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(
                1
            )
            .copy()
        )

        kickoff = pd.to_datetime(
            fixtures.iloc[
                0
            ][
                "date_utc"
            ],
            utc=True,
        )

        fixtures.loc[
            fixtures.index[
                0
            ],
            "date_utc",
        ] = (
            kickoff
            +
            pd.Timedelta(
                minutes=1
            )
        )

        with self.assertRaises(
            RuntimeError
        ):
            shadow.build_uefa_lookup(
                fixtures,
                self.uefa,
            )

    def test_08_legacy_shadow_snapshot_is_frozen(
        self,
    ) -> None:
        for path in [
            shadow.LEGACY_SHADOW_COMPARISON_FILE,
            shadow.LEGACY_SHADOW_FEATURE_MATRIX_FILE,
            shadow.LEGACY_SHADOW_METADATA_FILE,
        ]:
            self.assertTrue(
                path.exists(),
                msg=(
                    f"Missing frozen legacy "
                    f"shadow artifact: {path}"
                ),
            )

        comparison = pd.read_csv(
            shadow.LEGACY_SHADOW_COMPARISON_FILE,
            low_memory=False,
        )

        feature_matrix = pd.read_csv(
            shadow.LEGACY_SHADOW_FEATURE_MATRIX_FILE,
            low_memory=False,
        )

        with (
            shadow.LEGACY_SHADOW_METADATA_FILE.open(
                "r",
                encoding="utf-8",
            )
        ) as file:
            metadata = json.load(
                file
            )

        self.assertEqual(
            len(
                comparison
            ),
            len(
                feature_matrix
            ),
        )

        self.assertGreater(
            len(
                comparison
            ),
            0,
        )

        self.assertEqual(
            len(
                feature_matrix.columns
            ),
            292,
        )

        self.assertEqual(
            list(
                feature_matrix.columns[
                    -2:
                ]
            ),
            shadow.LOCKED_UEFA_FEATURES,
        )

        for prefix in [
            "v1",
            "v2",
        ]:
            sums = (
                comparison[
                    f"{prefix}_home_probability"
                ]
                +
                comparison[
                    f"{prefix}_draw_probability"
                ]
                +
                comparison[
                    f"{prefix}_away_probability"
                ]
            )

            self.assertTrue(
                np.allclose(
                    sums.to_numpy(
                        dtype=float
                    ),
                    1.0,
                    atol=1e-6,
                    rtol=0.0,
                )
            )

        self.assertEqual(
            metadata[
                "status"
            ],
            "shadow_research_only",
        )

        self.assertFalse(
            metadata[
                "production_promoted"
            ]
        )

        self.assertFalse(
            metadata[
                "public_predictions_modified"
            ]
        )

        self.assertFalse(
            metadata[
                "prediction_history_modified"
            ]
        )

        self.assertTrue(
            metadata[
                "protected_production_files"
            ][
                "unchanged"
            ]
        )

        self.assertFalse(
            metadata[
                "rules"
            ][
                "v2_promoted"
            ]
        )

        self.assertTrue(
            metadata[
                "rules"
            ][
                "same_290_live_features_used_for_both_models"
            ]
        )

        self.assertTrue(
            metadata[
                "rules"
            ][
                "v2_only_adds_locked_uefa_features"
            ]
        )

    def test_09_fixture_date_contract_requires_one_utc_date(
        self,
    ) -> None:
        one_date = pd.DataFrame(
            {
                "date_utc": [
                    "2026-10-20T16:45:00Z",
                    "2026-10-20T19:00:00Z",
                ],
            }
        )

        self.assertEqual(
            shadow.fixture_date_utc_from_frame(
                one_date
            ),
            "2026-10-20",
        )

        mixed_dates = pd.DataFrame(
            {
                "date_utc": [
                    "2026-10-20T19:00:00Z",
                    "2026-10-21T19:00:00Z",
                ],
            }
        )

        with self.assertRaises(
            RuntimeError
        ):
            shadow.fixture_date_utc_from_frame(
                mixed_dates
            )

    def test_10_archive_paths_are_date_scoped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(
                temp
            )

            paths = (
                shadow.build_shadow_archive_paths(
                    "2026-10-20",
                    archive_root=root,
                )
            )

            expected_dir = (
                root
                / "2026-10-20"
            )

            self.assertEqual(
                paths[
                    "directory"
                ],
                expected_dir,
            )

            self.assertEqual(
                paths[
                    "comparison"
                ],
                expected_dir
                / "comparison.csv",
            )

            self.assertEqual(
                paths[
                    "feature_matrix"
                ],
                expected_dir
                / "features.csv",
            )

            self.assertEqual(
                paths[
                    "metadata"
                ],
                expected_dir
                / "metadata.json",
            )

    def test_11_existing_archive_refuses_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(
                temp
            )

            archive_dir = (
                root
                / "2026-10-20"
            )

            archive_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            reason = (
                shadow.shadow_date_frozen_reason(
                    "2026-10-20",
                    archive_root=root,
                    include_legacy=False,
                )
            )

            self.assertIsNotNone(
                reason
            )

            with self.assertRaises(
                RuntimeError
            ):
                shadow.ensure_shadow_date_is_not_frozen(
                    "2026-10-20",
                    archive_root=root,
                    include_legacy=False,
                )

    def test_12_first_shadow_date_is_protected_by_legacy_guard(
        self,
    ) -> None:
        fixture_date = (
            shadow.load_legacy_shadow_fixture_date()
        )

        self.assertIsNotNone(
            fixture_date
        )

        reason = (
            shadow.shadow_date_frozen_reason(
                fixture_date
            )
        )

        self.assertIsNotNone(
            reason
        )

        self.assertIn(
            "frozen legacy shadow snapshot",
            str(
                reason
            ).lower(),
        )

        with self.assertRaises(
            RuntimeError
        ):
            shadow.ensure_shadow_date_is_not_frozen(
                fixture_date
            )


if __name__ == "__main__":
    unittest.main()
