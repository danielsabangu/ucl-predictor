from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "research"
    / "v2_promotion_policy.json"
)


EXPECTED_V2_MODEL_SHA256 = (
    "ad155ed3bcc782eee38c84be59ce0af4"
    "ecd9cd8f1b71d8ac7ac0759b5ade3f26"
)

EXPECTED_V2_METADATA_SHA256 = (
    "7e9c4544167ee0db911bce3b9428ebf"
    "ad587b5b9358b92ec83a115a7f4d03312"
)

EXPECTED_NEW_FEATURES = [
    "uefa_coefficient_diff",
    "uefa_strength_percentile_diff",
]


class V2PromotionPolicyTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        if not POLICY_PATH.exists():
            raise FileNotFoundError(
                "Missing preregistered V2 promotion "
                f"policy: {POLICY_PATH}"
            )

        with POLICY_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            cls.policy = json.load(
                handle
            )

    def test_01_policy_identity_is_preregistered(
        self,
    ) -> None:
        self.assertEqual(
            self.policy[
                "schema_version"
            ],
            1,
        )

        self.assertEqual(
            self.policy[
                "policy_name"
            ],
            "V2_FORWARD_PROMOTION_POLICY",
        )

        self.assertEqual(
            self.policy[
                "status"
            ],
            "PREREGISTERED",
        )

    def test_02_candidate_identity_is_frozen(
        self,
    ) -> None:
        candidate = (
            self.policy[
                "candidate"
            ]
        )

        self.assertEqual(
            candidate[
                "name"
            ],
            "ucl_predictor_v2_candidate",
        )

        self.assertEqual(
            candidate[
                "feature_count"
            ],
            292,
        )

        self.assertEqual(
            candidate[
                "v1_feature_count"
            ],
            290,
        )

        self.assertFalse(
            candidate[
                "production_promoted"
            ]
        )

        self.assertEqual(
            candidate[
                "new_probability_features"
            ],
            EXPECTED_NEW_FEATURES,
        )

        self.assertEqual(
            candidate[
                "candidate_model_sha256"
            ],
            EXPECTED_V2_MODEL_SHA256,
        )

        self.assertEqual(
            candidate[
                "candidate_metadata_sha256"
            ],
            EXPECTED_V2_METADATA_SHA256,
        )

    def test_03_forward_sample_thresholds_are_locked(
        self,
    ) -> None:
        evidence = (
            self.policy[
                "forward_evidence"
            ]
        )

        self.assertEqual(
            evidence[
                "minimum_fixtures_for_reliable_claims"
            ],
            30,
        )

        self.assertEqual(
            evidence[
                "minimum_fixtures_for_promotion_review"
            ],
            72,
        )

        self.assertEqual(
            evidence[
                "minimum_distinct_shadow_dates_for_promotion_review"
            ],
            8,
        )

        self.assertFalse(
            evidence[
                "partial_shadow_sets_may_be_scored"
            ]
        )

        self.assertFalse(
            evidence[
                "historical_2025_2026_confirmation_counts_toward_forward_sample"
            ]
        )

        self.assertTrue(
            evidence[
                "only_immutable_prematch_shadow_predictions_count"
            ]
        )

        self.assertTrue(
            evidence[
                "only_verified_finished_results_count"
            ]
        )

        self.assertTrue(
            evidence[
                "same_fixture_population_required_for_v1_and_v2"
            ]
        )

    def test_04_primary_metric_gate_is_locked(
        self,
    ) -> None:
        metrics = (
            self.policy[
                "primary_metrics"
            ]
        )

        self.assertEqual(
            set(
                metrics
            ),
            {
                "mean_log_loss",
                "mean_multiclass_brier",
            },
        )

        for metric_name in (
            "mean_log_loss",
            "mean_multiclass_brier",
        ):
            self.assertEqual(
                metrics[
                    metric_name
                ][
                    "direction"
                ],
                "LOWER_IS_BETTER",
            )

            self.assertEqual(
                metrics[
                    metric_name
                ][
                    "promotion_requirement"
                ],
                "V2_STRICTLY_BETTER_THAN_V1",
            )

        rules = (
            self.policy[
                "promotion_rules"
            ]
        )

        self.assertTrue(
            rules[
                "both_primary_metrics_must_improve"
            ]
        )

        self.assertFalse(
            rules[
                "promotion_is_automatic"
            ]
        )

        self.assertTrue(
            rules[
                "minimum_sample_only_grants_review_eligibility"
            ]
        )

    def test_05_candidate_changes_reset_forward_evidence(
        self,
    ) -> None:
        reset_policy = (
            self.policy[
                "candidate_change_policy"
            ]
        )

        required_reset_rules = {
            "feature_set_change_resets_forward_evidence",
            "model_parameter_change_resets_forward_evidence",
            "model_retraining_resets_forward_evidence",
            "calibration_change_resets_forward_evidence",
            "probability_postprocessing_change_resets_forward_evidence",
            "candidate_artifact_hash_change_resets_forward_evidence",
        }

        self.assertEqual(
            set(
                reset_policy
            ),
            required_reset_rules,
        )

        for rule_name in (
            required_reset_rules
        ):
            self.assertTrue(
                reset_policy[
                    rule_name
                ]
            )

    def test_06_decision_states_are_locked(
        self,
    ) -> None:
        states = (
            self.policy[
                "decision_states"
            ]
        )

        self.assertEqual(
            states[
                "under_30_fixtures"
            ],
            "INSUFFICIENT_FORWARD_SAMPLE",
        )

        self.assertEqual(
            states[
                "30_to_71_fixtures"
            ],
            "CONTINUE_SHADOW_EVALUATION",
        )

        self.assertEqual(
            states[
                "72_plus_but_under_8_dates"
            ],
            "CONTINUE_SHADOW_EVALUATION",
        )

        self.assertEqual(
            states[
                "72_plus_and_8_plus_dates_primary_metrics_not_both_better"
            ],
            "DO_NOT_PROMOTE",
        )

        self.assertEqual(
            states[
                "72_plus_and_8_plus_dates_primary_metrics_both_better"
            ],
            "ELIGIBLE_FOR_PROMOTION_REVIEW",
        )

        rules = (
            self.policy[
                "promotion_rules"
            ]
        )

        self.assertTrue(
            rules[
                "production_v1_remains_frozen_until_explicit_promotion"
            ]
        )

        self.assertTrue(
            rules[
                "no_metric_or_threshold_retuning_after_forward_results_are_seen"
            ]
        )


if __name__ == "__main__":
    unittest.main()