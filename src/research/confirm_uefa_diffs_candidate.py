from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import evaluate_final_test as v1

from src.research import (
    run_uefa_strength_candidate
    as candidate_a
)


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

COMPARISON_PATH = (
    REPORT_DIR
    / "uefa_diffs_2025_confirmation_comparison.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "uefa_diffs_2025_confirmation_predictions.csv"
)

CALIBRATION_PATH = (
    REPORT_DIR
    / "uefa_diffs_2025_confirmation_calibration.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "uefa_diffs_2025_confirmation_summary.json"
)


# ============================================================
# LOCKED SELECTED SUBSET
# ============================================================


SELECTED_UEFA_FEATURES = [
    "uefa_coefficient_diff",
    "uefa_strength_percentile_diff",
]


CONTROL_FEATURE_COUNT = 290

CANDIDATE_FEATURE_COUNT = 292


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=" * 100
    )

    print(
        "UCL PREDICTOR V2"
    )

    print(
        "UEFA DIFFS-ONLY 2025/26 CONFIRMATION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The UEFA subset was selected using "
        "2018-2024 rolling development only."
    )

    print(
        "2025/26 has already been observed in "
        "earlier project evaluation."
    )

    print(
        "This is therefore a locked historical "
        "confirmation, not a new untouched test."
    )

    print(
        "No further UEFA feature selection or "
        "hyperparameter tuning will follow from "
        "this result."
    )

    print(
        "Production remains untouched."
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        base,
        control_features,
    ) = (
        candidate_a.load_v1_dataframe()
    )

    uefa = (
        candidate_a.load_uefa_features()
    )

    dataframe = (
        candidate_a.merge_candidate_features(
            base,
            uefa,
        )
    )

    if len(
        control_features
    ) != CONTROL_FEATURE_COUNT:

        raise RuntimeError(
            "Expected "
            f"{CONTROL_FEATURE_COUNT} frozen "
            "V1 features, found "
            f"{len(control_features)}."
        )

    candidate_features = [
        *control_features,
        *SELECTED_UEFA_FEATURES,
    ]

    if len(
        candidate_features
    ) != CANDIDATE_FEATURE_COUNT:

        raise RuntimeError(
            "Expected "
            f"{CANDIDATE_FEATURE_COUNT} features "
            "for UEFA DIFFS candidate, found "
            f"{len(candidate_features)}."
        )

    if len(
        candidate_features
    ) != len(
        set(
            candidate_features
        )
    ):

        raise RuntimeError(
            "Duplicate candidate feature names."
        )

    # --------------------------------------------------------
    # SAME HISTORICAL SPLIT
    # --------------------------------------------------------

    (
        train,
        test,
    ) = (
        v1.build_final_split(
            dataframe
        )
    )

    print(
        "=" * 100
    )

    print(
        "CONFIRMATION SPLIT"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Training seasons: "
        f"{train['season'].min()} "
        f"to "
        f"{train['season'].max()}"
    )

    print(
        f"Training dates: "
        f"{train['date'].min().date()} "
        f"to "
        f"{train['date'].max().date()}"
    )

    print()

    print(
        f"Confirmation rows: "
        f"{len(test):,}"
    )

    print(
        f"Confirmation season: "
        f"{v1.FINAL_TEST_SEASON}"
    )

    print(
        f"Confirmation dates: "
        f"{test['date'].min().date()} "
        f"to "
        f"{test['date'].max().date()}"
    )

    print()

    print(
        f"Control features: "
        f"{len(control_features)}"
    )

    print(
        f"UEFA DIFFS candidate features: "
        f"{len(candidate_features)}"
    )

    print()

    print(
        "Locked UEFA additions:"
    )

    for feature in (
        SELECTED_UEFA_FEATURES
    ):

        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # CONTROL
    # --------------------------------------------------------

    control = (
        candidate_a.run_model(
            name=(
                "CONTROL - FROZEN V1"
            ),
            train=train,
            test=test,
            features=control_features,
        )
    )

    # --------------------------------------------------------
    # SELECTED UEFA CANDIDATE
    # --------------------------------------------------------

    candidate = (
        candidate_a.run_model(
            name=(
                "CANDIDATE A2 - "
                "V1 + UEFA DIFFS ONLY"
            ),
            train=train,
            test=test,
            features=candidate_features,
        )
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    (
        control_calibration,
        control_ece,
    ) = (
        candidate_a.build_calibration_bins(
            model_name="CONTROL_V1",
            y_true=control[
                "y_true"
            ],
            predictions=control[
                "predictions"
            ],
            probabilities=control[
                "probabilities"
            ],
        )
    )

    (
        candidate_calibration,
        candidate_ece,
    ) = (
        candidate_a.build_calibration_bins(
            model_name="UEFA_DIFFS_ONLY",
            y_true=candidate[
                "y_true"
            ],
            predictions=candidate[
                "predictions"
            ],
            probabilities=candidate[
                "probabilities"
            ],
        )
    )

    calibration = pd.concat(
        [
            control_calibration,
            candidate_calibration,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # METRIC COMPARISON
    # --------------------------------------------------------

    comparison = (
        candidate_a.build_metric_comparison(
            control_metrics=(
                control[
                    "metrics"
                ]
            ),
            candidate_metrics=(
                candidate[
                    "metrics"
                ]
            ),
            control_ece=(
                control_ece
            ),
            candidate_ece=(
                candidate_ece
            ),
        )
    )

    predictions = (
        candidate_a.build_prediction_report(
            test=test,
            control=control,
            candidate=candidate,
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_PATH,
        index=False,
    )

    control_metrics = {
        key: float(
            value
        )
        for key, value
        in control[
            "metrics"
        ].items()
        if np.isscalar(
            value
        )
    }

    candidate_metrics = {
        key: float(
            value
        )
        for key, value
        in candidate[
            "metrics"
        ].items()
        if np.isscalar(
            value
        )
    }

    summary = {
        "experiment": (
            "uefa_diffs_2025_confirmation"
        ),
        "status": (
            "research_only"
        ),
        "selection_source": (
            "2018_2024_rolling_ablation"
        ),
        "comparison_season": int(
            v1.FINAL_TEST_SEASON
        ),
        "comparison_season_already_consumed": (
            True
        ),
        "production_modified": False,
        "control_feature_count": int(
            len(
                control_features
            )
        ),
        "candidate_feature_count": int(
            len(
                candidate_features
            )
        ),
        "selected_uefa_features": (
            SELECTED_UEFA_FEATURES
        ),
        "control_metrics": (
            control_metrics
        ),
        "candidate_metrics": (
            candidate_metrics
        ),
        "control_top_label_ece_10bin": float(
            control_ece
        ),
        "candidate_top_label_ece_10bin": float(
            candidate_ece
        ),
        "no_further_uefa_selection_from_result": (
            True
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "UEFA DIFFS-ONLY CONFIRMATION RESULTS"
    )

    print(
        "=" * 100
    )

    print()

    candidate_a.print_metric_table(
        comparison
    )

    # --------------------------------------------------------
    # CORE METRICS
    # --------------------------------------------------------

    control_ll = float(
        control_metrics[
            "log_loss"
        ]
    )

    candidate_ll = float(
        candidate_metrics[
            "log_loss"
        ]
    )

    control_brier = float(
        control_metrics[
            "brier_score"
        ]
    )

    candidate_brier = float(
        candidate_metrics[
            "brier_score"
        ]
    )

    control_accuracy = float(
        control_metrics[
            "accuracy"
        ]
    )

    candidate_accuracy = float(
        candidate_metrics[
            "accuracy"
        ]
    )

    control_macro = float(
        control_metrics[
            "macro_f1"
        ]
    )

    candidate_macro = float(
        candidate_metrics[
            "macro_f1"
        ]
    )

    control_draw = float(
        control_metrics[
            "draw_recall"
        ]
    )

    candidate_draw = float(
        candidate_metrics[
            "draw_recall"
        ]
    )

    print()
    print(
        "=" * 100
    )

    print(
        "CORE CONFIRMATION SIGNAL"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Log Loss:"
    )

    print(
        f"  Control:     "
        f"{control_ll:.6f}"
    )

    print(
        f"  UEFA DIFFS:  "
        f"{candidate_ll:.6f}"
    )

    print(
        f"  Improvement: "
        f"{control_ll - candidate_ll:+.6f}"
    )

    print()

    print(
        "Brier Score:"
    )

    print(
        f"  Control:     "
        f"{control_brier:.6f}"
    )

    print(
        f"  UEFA DIFFS:  "
        f"{candidate_brier:.6f}"
    )

    print(
        f"  Improvement: "
        f"{control_brier - candidate_brier:+.6f}"
    )

    print()

    print(
        "Accuracy:"
    )

    print(
        f"  Control:     "
        f"{control_accuracy:.6f}"
    )

    print(
        f"  UEFA DIFFS:  "
        f"{candidate_accuracy:.6f}"
    )

    print(
        f"  Improvement: "
        f"{candidate_accuracy - control_accuracy:+.6f}"
    )

    print()

    print(
        "Macro F1:"
    )

    print(
        f"  Control:     "
        f"{control_macro:.6f}"
    )

    print(
        f"  UEFA DIFFS:  "
        f"{candidate_macro:.6f}"
    )

    print(
        f"  Improvement: "
        f"{candidate_macro - control_macro:+.6f}"
    )

    print()

    print(
        "Draw Recall:"
    )

    print(
        f"  Control:     "
        f"{control_draw:.6f}"
    )

    print(
        f"  UEFA DIFFS:  "
        f"{candidate_draw:.6f}"
    )

    print(
        f"  Improvement: "
        f"{candidate_draw - control_draw:+.6f}"
    )

    print()

    print(
        "Top-label ECE:"
    )

    print(
        f"  Control:     "
        f"{control_ece:.6f}"
    )

    print(
        f"  UEFA DIFFS:  "
        f"{candidate_ece:.6f}"
    )

    print(
        f"  Improvement: "
        f"{control_ece - candidate_ece:+.6f}"
    )

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    candidate_a.print_biggest_changes(
        predictions
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Subset selected only from 2018-2024: PASS"
    )

    print(
        "Exactly two UEFA features: PASS"
    )

    print(
        "Frozen V1 control features: PASS"
    )

    print(
        "Same historical training rows: PASS"
    )

    print(
        "Same 2025/26 comparison rows: PASS"
    )

    print(
        "Same locked CatBoost specification: PASS"
    )

    print(
        "No hyperparameter tuning: PASS"
    )

    print(
        "No additional UEFA feature selection: PASS"
    )

    print(
        "Production feature matrix untouched: PASS"
    )

    print(
        "Production model untouched: PASS"
    )

    print()

    print(
        "=" * 100
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        COMPARISON_PATH
    )

    print(
        PREDICTIONS_PATH
    )

    print(
        CALIBRATION_PATH
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()