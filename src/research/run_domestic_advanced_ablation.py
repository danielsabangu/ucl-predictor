from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research import (
    run_domestic_advanced_common_coverage_backtest
    as c1
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

FOLD_OUTPUT_PATH = (
    REPORT_DIR
    / "domestic_advanced_ablation_fold_metrics.csv"
)

SUMMARY_OUTPUT_PATH = (
    REPORT_DIR
    / "domestic_advanced_ablation_summary.csv"
)

METADATA_OUTPUT_PATH = (
    REPORT_DIR
    / "domestic_advanced_ablation_summary.json"
)


# ============================================================
# EXPERIMENT CONTRACT
# ============================================================


EXPECTED_DEVELOPMENT_MATCHES = 383

EXPECTED_CONFIRMATION_MATCHES = 78

CONTROL_FEATURE_COUNT = 290


LAST5_FEATURES = [
    "shots_for_diff_last5",
    "shots_against_diff_last5",
    "sot_for_diff_last5",
    "sot_against_diff_last5",
    "corners_for_diff_last5",
    "corners_against_diff_last5",
]


LAST10_FEATURES = [
    "shots_for_diff_last10",
    "shots_against_diff_last10",
    "sot_for_diff_last10",
    "sot_against_diff_last10",
    "corners_for_diff_last10",
    "corners_against_diff_last10",
]


ALL12_FEATURES = [
    *LAST5_FEATURES,
    *LAST10_FEATURES,
]


SCENARIOS = {
    "CONTROL": [],
    "LAST5_SIX": LAST5_FEATURES,
    "LAST10_SIX": LAST10_FEATURES,
    "ALL12": ALL12_FEATURES,
}


PRIMARY_METRICS = [
    "log_loss",
    "brier_score",
]


# ============================================================
# HELPERS
# ============================================================


def get_metric(
    result: dict,
    metric: str,
) -> float:
    return float(
        result[
            "metrics"
        ][
            metric
        ]
    )


def scenario_screen(
    summary: pd.DataFrame,
    scenario: str,
) -> bool:
    subset = summary[
        summary[
            "scenario"
        ]
        ==
        scenario
    ]

    log_loss = (
        subset[
            subset[
                "metric"
            ]
            ==
            "log_loss"
        ]
        .iloc[
            0
        ]
    )

    brier = (
        subset[
            subset[
                "metric"
            ]
            ==
            "brier_score"
        ]
        .iloc[
            0
        ]
    )

    return bool(
        (
            log_loss[
                "candidate_improvement"
            ]
            >
            0
        )
        and
        (
            brier[
                "candidate_improvement"
            ]
            >=
            0
        )
        and
        (
            int(
                log_loss[
                    "candidate_better_folds"
                ]
            )
            >=
            4
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
        "UCL PREDICTOR V2 - CANDIDATE C ABLATION"
    )

    print(
        "DOMESTIC ADVANCED PERFORMANCE WINDOW TEST"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Test whether the complete Candidate C failure "
        "is caused by mixing short-term and longer-term "
        "advanced-performance windows."
    )

    print()

    print(
        "Scenarios:"
    )

    print(
        "  CONTROL    = frozen V1 290"
    )

    print(
        "  LAST5_SIX  = V1 + six last-5 features"
    )

    print(
        "  LAST10_SIX = V1 + six last-10 features"
    )

    print(
        "  ALL12      = V1 + all twelve features"
    )

    print()

    print(
        "Rules:"
    )

    print(
        "  identical eligible matches"
    )

    print(
        "  2018-2024 rolling development only"
    )

    print(
        "  2025/26 remains locked"
    )

    print(
        "  exact locked CatBoost specification"
    )

    print(
        "  no hyperparameter tuning"
    )

    print(
        "  no additional feature-family search"
    )

    print(
        "  production untouched"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()

    print(
        "Loading frozen V1 data..."
    )

    (
        v1_data,
        control_features,
    ) = c1.load_v1_data()

    if len(
        control_features
    ) != CONTROL_FEATURE_COUNT:

        raise RuntimeError(
            "Frozen V1 feature count changed."
        )

    print(
        "Loading Candidate C feature layer..."
    )

    advanced = (
        c1.load_advanced_features()
    )

    print(
        "Merging Candidate C layer..."
    )

    dataframe = (
        c1.merge_candidate_layer(
            v1_data=v1_data,
            advanced=advanced,
        )
    )

    eligible = (
        c1.select_eligible_matches(
            dataframe
        )
    )

    development = eligible[
        eligible[
            "season"
        ].between(
            c1.FIRST_VALIDATION_SEASON,
            c1.LAST_VALIDATION_SEASON,
        )
    ].copy()

    confirmation = eligible[
        eligible[
            "season"
        ]
        ==
        c1.LOCKED_CONFIRMATION_SEASON
    ].copy()

    if len(
        development
    ) != EXPECTED_DEVELOPMENT_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_DEVELOPMENT_MATCHES} "
            "development matches, found "
            f"{len(development)}."
        )

    if len(
        confirmation
    ) != EXPECTED_CONFIRMATION_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_CONFIRMATION_MATCHES} "
            "locked confirmation matches, found "
            f"{len(confirmation)}."
        )

    # --------------------------------------------------------
    # COLLISION CHECK
    # --------------------------------------------------------

    collisions = [
        feature
        for feature
        in ALL12_FEATURES
        if feature
        in control_features
    ]

    if collisions:

        raise RuntimeError(
            "Candidate C collision detected: "
            f"{collisions}"
        )

    # --------------------------------------------------------
    # FEATURE SETS
    # --------------------------------------------------------

    feature_sets = {
        scenario: [
            *control_features,
            *additions,
        ]
        for (
            scenario,
            additions,
        ) in SCENARIOS.items()
    }

    expected_counts = {
        "CONTROL": 290,
        "LAST5_SIX": 296,
        "LAST10_SIX": 296,
        "ALL12": 302,
    }

    for (
        scenario,
        features,
    ) in feature_sets.items():

        if len(
            features
        ) != expected_counts[
            scenario
        ]:

            raise RuntimeError(
                f"{scenario} expected "
                f"{expected_counts[scenario]} "
                f"features, found "
                f"{len(features)}."
            )

    print()

    print(
        "=" * 110
    )

    print(
        "DATASET"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Eligible historical matches: "
        f"{len(eligible):,}"
    )

    print(
        f"Development matches 2018-2024: "
        f"{len(development):,}"
    )

    print(
        f"Locked 2025/26 matches: "
        f"{len(confirmation):,}"
    )

    print()

    for (
        scenario,
        features,
    ) in feature_sets.items():

        print(
            f"{scenario:<12}: "
            f"{len(features)} features"
        )

    # --------------------------------------------------------
    # ROLLING BACKTEST
    # --------------------------------------------------------

    fold_rows = []

    for validation_season in range(
        c1.FIRST_VALIDATION_SEASON,
        c1.LAST_VALIDATION_SEASON + 1,
    ):

        train = eligible[
            eligible[
                "season"
            ]
            <
            validation_season
        ].copy()

        validation = eligible[
            eligible[
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
                "Chronological overlap detected "
                f"for {validation_season}."
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

        print()

        results = {}

        for scenario in SCENARIOS:

            print(
                f"Training {scenario}..."
            )

            results[
                scenario
            ] = (
                c1.fit_and_evaluate(
                    train=train,
                    validation=validation,
                    features=feature_sets[
                        scenario
                    ],
                )
            )

        control = results[
            "CONTROL"
        ]

        for scenario in SCENARIOS:

            result = results[
                scenario
            ]

            if not np.array_equal(
                control[
                    "y_true"
                ],
                result[
                    "y_true"
                ],
            ):

                raise RuntimeError(
                    "Target mismatch between "
                    f"CONTROL and {scenario}."
                )

            row = {
                "validation_season": (
                    validation_season
                ),
                "scenario": (
                    scenario
                ),
                "training_rows": int(
                    len(
                        train
                    )
                ),
                "validation_rows": int(
                    len(
                        validation
                    )
                ),
                "feature_count": int(
                    len(
                        feature_sets[
                            scenario
                        ]
                    )
                ),
            }

            for metric in c1.METRICS:

                control_value = (
                    get_metric(
                        control,
                        metric,
                    )
                )

                scenario_value = (
                    get_metric(
                        result,
                        metric,
                    )
                )

                if scenario == "CONTROL":

                    metric_improvement = 0.0

                else:

                    metric_improvement = (
                        c1.improvement(
                            metric=metric,
                            control=control_value,
                            candidate=scenario_value,
                        )
                    )

                row[
                    f"control_{metric}"
                ] = (
                    control_value
                )

                row[
                    f"scenario_{metric}"
                ] = (
                    scenario_value
                )

                row[
                    f"{metric}_improvement"
                ] = (
                    metric_improvement
                )

            fold_rows.append(
                row
            )

        print()

        print(
            "Fold Log Loss:"
        )

        for scenario in SCENARIOS:

            result = results[
                scenario
            ]

            value = get_metric(
                result,
                "log_loss",
            )

            if scenario == "CONTROL":

                print(
                    f"  {scenario:<12} "
                    f"{value:.6f}"
                )

            else:

                delta = (
                    c1.improvement(
                        metric="log_loss",
                        control=get_metric(
                            control,
                            "log_loss",
                        ),
                        candidate=value,
                    )
                )

                print(
                    f"  {scenario:<12} "
                    f"{value:.6f} "
                    f"({delta:+.6f})"
                )

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    summary_rows = []

    for scenario in SCENARIOS:

        scenario_folds = fold_metrics[
            fold_metrics[
                "scenario"
            ]
            ==
            scenario
        ].copy()

        if len(
            scenario_folds
        ) != 7:

            raise RuntimeError(
                f"{scenario} expected 7 folds, "
                f"found {len(scenario_folds)}."
            )

        for metric in c1.METRICS:

            control_weighted = (
                c1.weighted_mean(
                    scenario_folds[
                        f"control_{metric}"
                    ],
                    scenario_folds[
                        "validation_rows"
                    ],
                )
            )

            scenario_weighted = (
                c1.weighted_mean(
                    scenario_folds[
                        f"scenario_{metric}"
                    ],
                    scenario_folds[
                        "validation_rows"
                    ],
                )
            )

            if scenario == "CONTROL":

                metric_improvement = 0.0

                better_folds = 0

                tied_folds = 7

                worse_folds = 0

            else:

                metric_improvement = (
                    c1.improvement(
                        metric=metric,
                        control=control_weighted,
                        candidate=scenario_weighted,
                    )
                )

                improvements = (
                    scenario_folds[
                        f"{metric}_improvement"
                    ]
                )

                better_folds = int(
                    (
                        improvements
                        >
                        0
                    ).sum()
                )

                tied_folds = int(
                    np.isclose(
                        improvements,
                        0.0,
                        atol=1e-12,
                    ).sum()
                )

                worse_folds = int(
                    7
                    -
                    better_folds
                    -
                    tied_folds
                )

            summary_rows.append(
                {
                    "scenario": (
                        scenario
                    ),
                    "feature_count": int(
                        len(
                            feature_sets[
                                scenario
                            ]
                        )
                    ),
                    "added_feature_count": int(
                        len(
                            SCENARIOS[
                                scenario
                            ]
                        )
                    ),
                    "metric": (
                        metric
                    ),
                    "better_direction": (
                        "lower"
                        if metric
                        in c1.LOWER_IS_BETTER
                        else "higher"
                    ),
                    "weighted_control": (
                        control_weighted
                    ),
                    "weighted_scenario": (
                        scenario_weighted
                    ),
                    "candidate_improvement": (
                        metric_improvement
                    ),
                    "candidate_better_folds": (
                        better_folds
                    ),
                    "tied_folds": (
                        tied_folds
                    ),
                    "candidate_worse_folds": (
                        worse_folds
                    ),
                    "total_folds": 7,
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # SCREEN CANDIDATES
    # --------------------------------------------------------

    candidate_scenarios = [
        "LAST5_SIX",
        "LAST10_SIX",
        "ALL12",
    ]

    screen_results = {
        scenario: (
            scenario_screen(
                summary,
                scenario,
            )
        )
        for scenario
        in candidate_scenarios
    }

    passing = [
        scenario
        for scenario
        in candidate_scenarios
        if screen_results[
            scenario
        ]
    ]

    selected_scenario = None

    if passing:

        ranking = []

        for scenario in passing:

            scenario_summary = summary[
                summary[
                    "scenario"
                ]
                ==
                scenario
            ]

            log_loss_row = (
                scenario_summary[
                    scenario_summary[
                        "metric"
                    ]
                    ==
                    "log_loss"
                ]
                .iloc[
                    0
                ]
            )

            brier_row = (
                scenario_summary[
                    scenario_summary[
                        "metric"
                    ]
                    ==
                    "brier_score"
                ]
                .iloc[
                    0
                ]
            )

            ranking.append(
                (
                    scenario,
                    float(
                        log_loss_row[
                            "candidate_improvement"
                        ]
                    ),
                    float(
                        brier_row[
                            "candidate_improvement"
                        ]
                    ),
                )
            )

        ranking.sort(
            key=lambda item: (
                item[
                    1
                ],
                item[
                    2
                ],
            ),
            reverse=True,
        )

        selected_scenario = (
            ranking[
                0
            ][
                0
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
        FOLD_OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    metadata = {
        "experiment": (
            "candidate_c_domestic_advanced_window_ablation"
        ),
        "status": (
            "research_only"
        ),
        "selection_window": (
            "2018_2024"
        ),
        "confirmation_2025_used": False,
        "locked_confirmation_matches": int(
            len(
                confirmation
            )
        ),
        "development_matches": int(
            len(
                development
            )
        ),
        "control_feature_count": int(
            len(
                control_features
            )
        ),
        "scenarios": {
            scenario: {
                "added_features": (
                    SCENARIOS[
                        scenario
                    ]
                ),
                "total_feature_count": int(
                    len(
                        feature_sets[
                            scenario
                        ]
                    )
                ),
                "passes_signal_screen": bool(
                    screen_results.get(
                        scenario,
                        False,
                    )
                ),
            }
            for scenario
            in SCENARIOS
        },
        "signal_screen": {
            "weighted_log_loss_improves": True,
            "weighted_brier_not_worse": True,
            "minimum_log_loss_better_folds": 4,
        },
        "selected_scenario": (
            selected_scenario
        ),
        "production_modified": False,
        "hyperparameters_tuned": False,
        "additional_posthoc_feature_search": False,
    }

    with METADATA_OUTPUT_PATH.open(
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
        "ABLATION SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    display_metrics = [
        "log_loss",
        "brier_score",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "draw_recall",
        "ece",
    ]

    display = summary[
        summary[
            "metric"
        ].isin(
            display_metrics
        )
    ].copy()

    display[
        "weighted_control"
    ] = display[
        "weighted_control"
    ].map(
        lambda value: (
            f"{value:.6f}"
        )
    )

    display[
        "weighted_scenario"
    ] = display[
        "weighted_scenario"
    ].map(
        lambda value: (
            f"{value:.6f}"
        )
    )

    display[
        "candidate_improvement"
    ] = display[
        "candidate_improvement"
    ].map(
        lambda value: (
            f"{value:+.6f}"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PRIMARY SCREEN
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "PRIMARY SIGNAL SCREEN"
    )

    print(
        "=" * 110
    )

    print()

    for scenario in candidate_scenarios:

        scenario_summary = summary[
            summary[
                "scenario"
            ]
            ==
            scenario
        ]

        log_loss_row = (
            scenario_summary[
                scenario_summary[
                    "metric"
                ]
                ==
                "log_loss"
            ]
            .iloc[
                0
            ]
        )

        brier_row = (
            scenario_summary[
                scenario_summary[
                    "metric"
                ]
                ==
                "brier_score"
            ]
            .iloc[
                0
            ]
        )

        print(
            f"{scenario}:"
        )

        print(
            f"  Log Loss improvement: "
            f"{log_loss_row['candidate_improvement']:+.6f}"
        )

        print(
            f"  Log Loss better folds: "
            f"{int(log_loss_row['candidate_better_folds'])}/7"
        )

        print(
            f"  Brier improvement: "
            f"{brier_row['candidate_improvement']:+.6f}"
        )

        print(
            f"  Signal screen: "
            f"{'PASS' if screen_results[scenario] else 'FAIL'}"
        )

        print()

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    print(
        "=" * 110
    )

    print(
        "CANDIDATE C DEVELOPMENT DECISION"
    )

    print(
        "=" * 110
    )

    print()

    if selected_scenario is None:

        print(
            "Decision: REJECT_ADVANCED_PERFORMANCE_FAMILY"
        )

        print()

        print(
            "Neither last-5, last-10 nor all-12 "
            "passes the pre-2025 signal screen."
        )

        print(
            "Do not evaluate Candidate C on 2025/26."
        )

    else:

        print(
            "Decision: RETAIN_ADVANCED_PERFORMANCE_SUBSET"
        )

        print()

        print(
            f"Selected development subset: "
            f"{selected_scenario}"
        )

        print()

        print(
            "Only this locked subset may proceed "
            "to one 2025/26 confirmation."
        )

        print(
            "No further Candidate C feature selection "
            "before that confirmation."
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

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
        "Frozen V1 control = 290 features: PASS"
    )

    print(
        "Candidate C additions new to V1: PASS"
    )

    print(
        "All scenarios use identical rows: PASS"
    )

    print(
        "All matches have >=10 prior domestic matches: PASS"
    )

    print(
        "2018-2024 only used for subset selection: PASS"
    )

    print(
        "2025/26 evaluated: NO"
    )

    print(
        "CatBoost hyperparameters tuned: NO"
    )

    print(
        "Production model modified: NO"
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
        FOLD_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        METADATA_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()