from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.ml import train_production_model as v1
from src.ml import train_v2_candidate as v2


class V2CandidateContractTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        (
            cls.dataframe,
            cls.master_features,
            cls.control_features,
            cls.transfer_features,
            cls.confirmation,
        ) = v2.prepare_v2_training_data()

        cls.candidate_features = [
            *cls.control_features,
            *v2.SELECTED_UEFA_FEATURES,
        ]

    def test_01_locked_feature_counts(
        self,
    ) -> None:
        self.assertEqual(
            len(
                self.master_features
            ),
            411,
        )

        self.assertEqual(
            len(
                self.transfer_features
            ),
            121,
        )

        self.assertEqual(
            len(
                self.control_features
            ),
            v2.EXPECTED_V1_FEATURES,
        )

        self.assertEqual(
            len(
                self.candidate_features
            ),
            v2.EXPECTED_V2_FEATURES,
        )

        self.assertEqual(
            len(
                self.dataframe
            ),
            v2.EXPECTED_ROWS,
        )

    def test_02_locked_uefa_subset_is_exact(
        self,
    ) -> None:
        self.assertEqual(
            v2.SELECTED_UEFA_FEATURES,
            [
                "uefa_coefficient_diff",
                "uefa_strength_percentile_diff",
            ],
        )

        self.assertEqual(
            self.candidate_features[
                -2:
            ],
            v2.SELECTED_UEFA_FEATURES,
        )

    def test_03_v1_feature_order_is_preserved(
        self,
    ) -> None:
        self.assertEqual(
            self.candidate_features[
                :v2.EXPECTED_V1_FEATURES
            ],
            self.control_features,
        )

        self.assertEqual(
            self.candidate_features[
                v2.EXPECTED_V1_FEATURES:
            ],
            v2.SELECTED_UEFA_FEATURES,
        )

        self.assertEqual(
            len(
                self.candidate_features
            ),
            len(
                set(
                    self.candidate_features
                )
            ),
        )

    def test_04_historical_uefa_features_are_complete(
        self,
    ) -> None:
        for column in (
            v2.SELECTED_UEFA_FEATURES
        ):
            self.assertIn(
                column,
                self.dataframe.columns,
            )

            self.assertEqual(
                int(
                    self.dataframe[
                        column
                    ]
                    .isna()
                    .sum()
                ),
                0,
            )

            values = (
                self.dataframe[
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

    def test_05_locked_confirmation_is_read_only_evidence(
        self,
    ) -> None:
        self.assertEqual(
            self.confirmation[
                "experiment"
            ],
            "uefa_diffs_2025_confirmation",
        )

        self.assertEqual(
            self.confirmation[
                "selection_source"
            ],
            "2018_2024_rolling_ablation",
        )

        self.assertEqual(
            self.confirmation[
                "comparison_season"
            ],
            2025,
        )

        self.assertTrue(
            self.confirmation[
                "comparison_season_already_consumed"
            ]
        )

        self.assertTrue(
            self.confirmation[
                "no_further_uefa_selection_from_result"
            ]
        )

        self.assertEqual(
            self.confirmation[
                "selected_uefa_features"
            ],
            v2.SELECTED_UEFA_FEATURES,
        )

        control = self.confirmation[
            "control_metrics"
        ]

        candidate = self.confirmation[
            "candidate_metrics"
        ]

        self.assertLess(
            float(
                candidate[
                    "log_loss"
                ]
            ),
            float(
                control[
                    "log_loss"
                ]
            ),
        )

        self.assertLess(
            float(
                candidate[
                    "brier_score"
                ]
            ),
            float(
                control[
                    "brier_score"
                ]
            ),
        )

    def test_06_v2_uses_exact_frozen_model_spec(
        self,
    ) -> None:
        model = (
            v1.build_production_model()
        )

        v2.validate_locked_model_spec(
            model
        )

        imputer = model.named_steps[
            "imputer"
        ]

        estimator = model.named_steps[
            "model"
        ]

        self.assertEqual(
            imputer.strategy,
            "median",
        )

        self.assertTrue(
            imputer.add_indicator
        )

        params = (
            estimator.get_params()
        )

        self.assertEqual(
            params[
                "iterations"
            ],
            700,
        )

        self.assertEqual(
            params[
                "learning_rate"
            ],
            0.025,
        )

        self.assertEqual(
            params[
                "depth"
            ],
            5,
        )

        self.assertEqual(
            params[
                "loss_function"
            ],
            "MultiClass",
        )

        self.assertEqual(
            params[
                "l2_leaf_reg"
            ],
            5.0,
        )

        self.assertEqual(
            params[
                "random_seed"
            ],
            42,
        )

        self.assertFalse(
            params[
                "allow_writing_files"
            ]
        )

    def test_07_v2_artifacts_cannot_overwrite_v1(
        self,
    ) -> None:
        self.assertNotEqual(
            v2.V2_MODEL_FILE.resolve(),
            v1.PRODUCTION_MODEL_FILE.resolve(),
        )

        self.assertNotEqual(
            v2.V2_METADATA_FILE.resolve(),
            v1.PRODUCTION_METADATA_FILE.resolve(),
        )

        self.assertEqual(
            v2.MODEL_VERSION,
            "ucl_predictor_v2_candidate",
        )

    def test_08_identity_mismatch_is_rejected(
        self,
    ) -> None:
        base = (
            self.dataframe[
                [
                    column
                    for column
                    in self.dataframe.columns
                    if column
                    not in v2.SELECTED_UEFA_FEATURES
                ]
            ]
            .copy()
        )

        uefa = (
            v2.load_historical_uefa_features()
        )

        tampered = uefa.copy()

        tampered.loc[
            tampered.index[
                0
            ],
            "home_team",
        ] = "INTENTIONAL_TEST_MISMATCH"

        with self.assertRaises(
            RuntimeError
        ):
            v2.merge_v2_features(
                base,
                tampered,
            )


if __name__ == "__main__":
    unittest.main()