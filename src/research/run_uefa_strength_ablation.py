from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import run_boosted_backtest as rb
from src.research import (
    run_uefa_strength_rolling_backtest
    as uefa_rolling
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

FOLD_METRICS_PATH = (
    REPORT_DIR
    / "uefa_strength_ablation_fold_metrics.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "uefa_strength_ablation_summary.csv"
)

OOF_PATH = (
    REPORT_DIR
    / "uefa_strength_ablation_oof_predictions.csv"
)

METADATA_PATH = (
    REPORT_DIR
    / "uefa_strength_ablation_summary.json"
)


# ============================================================
# FEATURE SUBSETS
# ============================================================


RAW_FEATURES = [
    "home_uefa_coefficient",
    "away_uefa_coefficient",
    "uefa_coefficient_diff",
]


PERCENTILE_FEATURES = [
    "home_uefa_strength_percentile",
    "away_uefa_strength_percentile",
    "uefa_strength_percentile_diff",
]


DIFFERENCE_FEATURES = [
    "uefa_coefficient_diff",
    "uefa_strength_percentile_diff",
]


ALL_SIX_FEATURES = [
    *RAW_FEATURES,
    *PERCENTILE_FEATURES,
]


SCENARIOS = {
    "CONTROL_V1": [],
    "UEFA_RAW_TRIO": (
        RAW_FEATURES
    ),
    "UEFA_PERCENTILE_TRIO": (
        PERCENTILE_FEATURES
    ),
    "UEFA_DIFFS_ONLY": (
        DIFFERENCE_FEATURES
    ),
    "UEFA_ALL_SIX": (
        ALL_SIX_FEATURES
    ),
}


# ============================================================
# METRICS
# ============================================================


METRICS = [
    "log_loss",
    "brier_score",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "home_recall",
    "draw_recall",
    "away_recall",
]


LOWER_IS_BETTER = {
    "log_loss",
    "brier_score",
}


def improvement(
    *,
    metric: str,
    control: float,
    candidate: float,
) -> float:

    if metric in LOWER_IS_BETTER:

        return (
            control
            -
            candidate
        )

    return (
        candidate
        -
        control
    )


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:

    return float(
        np.average(
            values.to_numpy(
                dtype=float
            ),
            weights=weights.to_numpy(
                dtype=float
            ),
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=" * 110
    )

    print(
        "UCL PREDICTOR V2"
    )

    print(
        "UEFA STRENGTH FEATURE ABLATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Determine whether raw UEFA coefficients, "
        "era-relative percentiles, or only their "
        "differences provide the useful signal."
    )

    print()

    print(
        "Development seasons only: 2018-2024"
    )

    print(
        "2025/26 will not be evaluated."
    )

    print(
        "No hyperparameter tuning."
    )

    print(
        "Production model untouched."
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        base,
        control_features,
    ) = (
        uefa_rolling.load_control_data()
    )

    uefa = (
        uefa_rolling.load_uefa_features()
    )

    dataframe = (
        uefa_rolling.merge_features(
            base=base,
            uefa=uefa,
        )
    )

    if len(
        control_features
    ) != 290:

        raise RuntimeError(
            "Expected 290 frozen V1 features, "
            f"found {len(control_features)}."
        )

    # --------------------------------------------------------
    # EXCLUDE 2025
    # --------------------------------------------------------

    final_2025 = dataframe[
        dataframe[
            "season"
        ]
        ==
        rb.FINAL_TEST_SEASON
    ].copy()

    backtest_data = dataframe[
        dataframe[
            "season"
        ]
        <
        rb.FINAL_TEST_SEASON
    ].copy()

    if len(
        final_2025
    ) != 189:

        raise RuntimeError(
            "Expected 189 excluded 2025/26 "
            f"rows, found {len(final_2025)}."
        )

    validation_seasons = list(
        range(
            rb.FIRST_VALIDATION_SEASON,
            rb.LAST_VALIDATION_SEASON
            + 1,
        )
    )

    print(
        "Validation seasons:"
    )

    print(
        "  "
        +
        ", ".join(
            str(
                value
            )
            for value
            in validation_seasons
        )
    )

    print()

    print(
        "Scenarios:"
    )

    for (
        scenario,
        additions,
    ) in SCENARIOS.items():

        print(
            f"  {scenario}: "
            f"{290 + len(additions)} features"
        )

    # --------------------------------------------------------
    # RUN FOLDS
    # --------------------------------------------------------

    fold_rows = []

    oof_frames = []

    for validation_season in (
        validation_seasons
    ):

        train = backtest_data[
            backtest_data[
                "season"
            ]
            <
            validation_season
        ].copy()

        validation = backtest_data[
            backtest_data[
                "season"
            ]
            ==
            validation_season
        ].copy()

        if train.empty:

            raise RuntimeError(
                f"Empty training fold for "
                f"{validation_season}."
            )

        if validation.empty:

            raise RuntimeError(
                f"Empty validation fold for "
                f"{validation_season}."
            )

        if (
            train[
                "date"
            ].max()
            >=
            validation[
                "date"
            ].min()
        ):

            raise RuntimeError(
                "Chronological overlap in "
                f"{validation_season}."
            )

        print()
        print(
            "=" * 110
        )

        print(
            f"VALIDATION SEASON "
            f"{validation_season}"
        )

        print(
            "=" * 110
        )

        print()

        print(
            f"Training rows: "
            f"{len(train):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        for (
            scenario,
            additions,
        ) in SCENARIOS.items():

            features = [
                *control_features,
                *additions,
            ]

            if len(
                features
            ) != len(
                set(
                    features
                )
            ):

                raise RuntimeError(
                    f"Duplicate features in "
                    f"{scenario}."
                )

            print()

            print(
                f"Training {scenario} "
                f"({len(features)} features)..."
            )

            result = (
                uefa_rolling.fit_and_evaluate(
                    train=train,
                    validation=validation,
                    features=features,
                )
            )

            metrics = (
                result[
                    "metrics"
                ]
            )

            row = {
                "validation_season": (
                    validation_season
                ),
                "scenario": (
                    scenario
                ),
                "feature_count": (
                    len(
                        features
                    )
                ),
                "training_rows": (
                    len(
                        train
                    )
                ),
                "validation_rows": (
                    len(
                        validation
                    )
                ),
            }

            for metric in METRICS:

                row[
                    metric
                ] = float(
                    metrics[
                        metric
                    ]
                )

            fold_rows.append(
                row
            )

            probabilities = (
                result[
                    "probabilities"
                ]
            )

            predictions = (
                result[
                    "predictions"
                ]
            )

            y_true = (
                result[
                    "y_true"
                ]
            )

            prediction_frame = pd.DataFrame(
                {
                    "validation_season": (
                        validation_season
                    ),
                    "scenario": (
                        scenario
                    ),
                    "match_id": (
                        validation[
                            "match_id"
                        ].to_numpy()
                    ),
                    "date": (
                        validation[
                            "date"
                        ].to_numpy()
                    ),
                    "home_team": (
                        validation[
                            "home_team"
                        ].to_numpy()
                    ),
                    "away_team": (
                        validation[
                            "away_team"
                        ].to_numpy()
                    ),
                    "actual_result": (
                        validation[
                            "result"
                        ].to_numpy()
                    ),
                    "actual_label": (
                        y_true
                    ),
                    "prediction": [
                        rb.TARGET_NAMES[
                            int(
                                value
                            )
                        ]
                        for value
                        in predictions
                    ],
                    "prob_H": (
                        probabilities[
                            :,
                            0
                        ]
                    ),
                    "prob_D": (
                        probabilities[
                            :,
                            1
                        ]
                    ),
                    "prob_A": (
                        probabilities[
                            :,
                            2
                        ]
                    ),
                }
            )

            oof_frames.append(
                prediction_frame
            )

            print(
                f"  Log Loss:    "
                f"{metrics['log_loss']:.6f}"
            )

            print(
                f"  Brier:       "
                f"{metrics['brier_score']:.6f}"
            )

            print(
                f"  Accuracy:    "
                f"{metrics['accuracy']:.6f}"
            )

            print(
                f"  Macro F1:    "
                f"{metrics['macro_f1']:.6f}"
            )

            print(
                f"  Draw Recall: "
                f"{metrics['draw_recall']:.6f}"
            )

    # --------------------------------------------------------
    # FOLD TABLE
    # --------------------------------------------------------

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    control_folds = (
        fold_metrics[
            fold_metrics[
                "scenario"
            ]
            ==
            "CONTROL_V1"
        ]
        .set_index(
            "validation_season"
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = []

    for (
        scenario,
        additions,
    ) in SCENARIOS.items():

        scenario_folds = (
            fold_metrics[
                fold_metrics[
                    "scenario"
                ]
                ==
                scenario
            ]
            .copy()
            .set_index(
                "validation_season"
            )
        )

        scenario_oof = (
            oof[
                oof[
                    "scenario"
                ]
                ==
                scenario
            ]
            .copy()
        )

        probabilities = (
            scenario_oof[
                [
                    "prob_H",
                    "prob_D",
                    "prob_A",
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_true = (
            scenario_oof[
                "actual_label"
            ]
            .to_numpy(
                dtype=int
            )
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        (
            _,
            ece,
        ) = (
            uefa_rolling.build_calibration_bins(
                model_name=(
                    scenario
                ),
                y_true=y_true,
                predictions=predictions,
                probabilities=probabilities,
            )
        )

        summary_row = {
            "scenario": (
                scenario
            ),
            "added_feature_count": (
                len(
                    additions
                )
            ),
            "total_feature_count": (
                290
                +
                len(
                    additions
                )
            ),
            "oof_rows": (
                len(
                    scenario_oof
                )
            ),
            "top_label_ece_10bin": (
                float(
                    ece
                )
            ),
        }

        for metric in METRICS:

            weighted_value = (
                weighted_mean(
                    scenario_folds[
                        metric
                    ],
                    scenario_folds[
                        "validation_rows"
                    ],
                )
            )

            summary_row[
                f"weighted_{metric}"
            ] = (
                weighted_value
            )

            if scenario == (
                "CONTROL_V1"
            ):

                summary_row[
                    f"{metric}_improvement_vs_control"
                ] = 0.0

                summary_row[
                    f"{metric}_better_folds_vs_control"
                ] = 0

                continue

            improvements = []

            for validation_season in (
                validation_seasons
            ):

                control_value = float(
                    control_folds.loc[
                        validation_season,
                        metric,
                    ]
                )

                candidate_value = float(
                    scenario_folds.loc[
                        validation_season,
                        metric,
                    ]
                )

                improvements.append(
                    improvement(
                        metric=metric,
                        control=(
                            control_value
                        ),
                        candidate=(
                            candidate_value
                        ),
                    )
                )

            improvements = np.asarray(
                improvements,
                dtype=float,
            )

            control_weighted = (
                weighted_mean(
                    control_folds[
                        metric
                    ],
                    control_folds[
                        "validation_rows"
                    ],
                )
            )

            summary_row[
                f"{metric}_improvement_vs_control"
            ] = (
                improvement(
                    metric=metric,
                    control=(
                        control_weighted
                    ),
                    candidate=(
                        weighted_value
                    ),
                )
            )

            summary_row[
                f"{metric}_better_folds_vs_control"
            ] = int(
                (
                    improvements
                    > 0
                ).sum()
            )

        summary_rows.append(
            summary_row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    control_ece = float(
        summary.loc[
            summary[
                "scenario"
            ]
            ==
            "CONTROL_V1",
            "top_label_ece_10bin",
        ].iloc[
            0
        ]
    )

    summary[
        "ece_improvement_vs_control"
    ] = (
        control_ece
        -
        summary[
            "top_label_ece_10bin"
        ]
    )

    # --------------------------------------------------------
    # SCREENING RULE
    # --------------------------------------------------------

    summary[
        "passes_probability_screen"
    ] = (
        (
            summary[
                "log_loss_improvement_vs_control"
            ]
            > 0
        )
        &
        (
            summary[
                "brier_score_improvement_vs_control"
            ]
            >= 0
        )
        &
        (
            summary[
                "log_loss_better_folds_vs_control"
            ]
            >= 4
        )
    )

    summary.loc[
        summary[
            "scenario"
        ]
        ==
        "CONTROL_V1",
        "passes_probability_screen",
    ] = False

    candidate_summary = (
        summary[
            summary[
                "scenario"
            ]
            !=
            "CONTROL_V1"
        ]
        .copy()
    )

    passing = (
        candidate_summary[
            candidate_summary[
                "passes_probability_screen"
            ]
        ]
        .copy()
    )

    if passing.empty:

        decision = (
            "PARK_UEFA_FEATURE_FAMILY"
        )

        selected_scenario = None

    else:

        selected = (
            passing.sort_values(
                [
                    "weighted_log_loss",
                    "weighted_brier_score",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .iloc[
                0
            ]
        )

        decision = (
            "KEEP_UEFA_SUBSET_FOR_FURTHER_RESEARCH"
        )

        selected_scenario = str(
            selected[
                "scenario"
            ]
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    oof.to_csv(
        OOF_PATH,
        index=False,
    )

    metadata = {
        "experiment": (
            "uefa_strength_feature_ablation"
        ),
        "status": (
            "research_only"
        ),
        "production_modified": False,
        "final_2025_used": False,
        "validation_seasons": (
            validation_seasons
        ),
        "scenarios": {
            key: value
            for key, value
            in SCENARIOS.items()
        },
        "primary_metric": (
            "weighted_log_loss"
        ),
        "secondary_metric": (
            "weighted_brier_score"
        ),
        "screening_rule": {
            "log_loss_improvement_gt": 0,
            "brier_improvement_gte": 0,
            "minimum_log_loss_better_folds": 4,
        },
        "decision": (
            decision
        ),
        "selected_scenario": (
            selected_scenario
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "UEFA FEATURE ABLATION SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    display_columns = [
        "scenario",
        "total_feature_count",
        "weighted_log_loss",
        "log_loss_improvement_vs_control",
        "log_loss_better_folds_vs_control",
        "weighted_brier_score",
        "brier_score_improvement_vs_control",
        "weighted_accuracy",
        "weighted_macro_f1",
        "weighted_draw_recall",
        "top_label_ece_10bin",
        "ece_improvement_vs_control",
        "passes_probability_screen",
    ]

    display = (
        summary[
            display_columns
        ]
        .copy()
    )

    numeric_columns = [
        "weighted_log_loss",
        "log_loss_improvement_vs_control",
        "weighted_brier_score",
        "brier_score_improvement_vs_control",
        "weighted_accuracy",
        "weighted_macro_f1",
        "weighted_draw_recall",
        "top_label_ece_10bin",
        "ece_improvement_vs_control",
    ]

    for column in numeric_columns:

        display[
            column
        ] = display[
            column
        ].map(
            lambda value: (
                f"{value:.6f}"
            )
        )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "DECISION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Decision: "
        f"{decision}"
    )

    if selected_scenario is None:

        print(
            "Selected UEFA subset: None"
        )

        print()

        print(
            "The UEFA feature family does not "
            "currently pass the predefined "
            "rolling probability-quality screen."
        )

    else:

        print(
            f"Selected UEFA subset: "
            f"{selected_scenario}"
        )

        print()

        print(
            "This subset may proceed to the "
            "next research stage."
        )

    print()
    print(
        "=" * 110
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "2025/26 excluded: PASS"
    )

    print(
        "Validation seasons 2018-2024: PASS"
    )

    print(
        "Frozen V1 control: PASS"
    )

    print(
        "Locked CatBoost specification: PASS"
    )

    print(
        "No hyperparameter tuning: PASS"
    )

    print(
        "No production changes: PASS"
    )

    print()

    print(
        "=" * 110
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 110
    )

    print()

    print(
        FOLD_METRICS_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        OOF_PATH
    )

    print(
        METADATA_PATH
    )


if __name__ == "__main__":
    main()