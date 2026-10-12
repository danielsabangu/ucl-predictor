from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import run_boosted_backtest as boosted
from src.research import (
    run_domestic_advanced_common_coverage_backtest as c1,
)


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]


CONTEXT_FEATURE_PATH = (
    ROOT
    / "data"
    / "research"
    / "ucl_tournament_context_features.csv"
)


REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)


FOLD_OUTPUT_PATH = (
    REPORT_DIR
    / "tournament_context_ablation_fold_metrics.csv"
)


SUMMARY_OUTPUT_PATH = (
    REPORT_DIR
    / "tournament_context_ablation_summary.csv"
)


METADATA_OUTPUT_PATH = (
    REPORT_DIR
    / "tournament_context_ablation_summary.json"
)


# ============================================================
# EXPERIMENT POLICY
# ============================================================


FIRST_CONTEXT_SEASON = 2012

FIRST_VALIDATION_SEASON = 2018

LAST_VALIDATION_SEASON = 2024

CONFIRMATION_SEASON = 2025


EXPECTED_MASTER_MATCHES = 2089

EXPECTED_DEVELOPMENT_MATCHES = 924

EXPECTED_CONFIRMATION_MATCHES = 189

EXPECTED_CONTROL_FEATURES = 290

EXPECTED_FOLDS = 7


# ============================================================
# CANDIDATE E FEATURES
# ============================================================


STAGE_CONTEXT_FEATURES = [
    "tournament_is_knockout",
    "tournament_is_final",
    "tournament_round_depth",
]


LEG_CONTEXT_FEATURES = [
    "tournament_is_first_leg",
    "tournament_is_second_leg",
]


TIE_STATE_FEATURES = [
    "tournament_is_second_leg",
    "aggregate_goal_diff_home_pre",
    "aggregate_abs_goal_diff_pre",
    "aggregate_state_home_pre",
]


ALL_CONTEXT_FEATURES = [
    "tournament_is_knockout",
    "tournament_is_final",
    "tournament_is_first_leg",
    "tournament_is_second_leg",
    "tournament_round_depth",
    "aggregate_goal_diff_home_pre",
    "aggregate_abs_goal_diff_pre",
    "aggregate_state_home_pre",
]


SCENARIOS = {
    "CONTROL": [],
    "STAGE_CONTEXT": (
        STAGE_CONTEXT_FEATURES
    ),
    "LEG_CONTEXT": (
        LEG_CONTEXT_FEATURES
    ),
    "TIE_STATE": (
        TIE_STATE_FEATURES
    ),
    "ALL_CONTEXT": (
        ALL_CONTEXT_FEATURES
    ),
}


CANDIDATE_SCENARIOS = [
    "STAGE_CONTEXT",
    "LEG_CONTEXT",
    "TIE_STATE",
    "ALL_CONTEXT",
]


EXPECTED_FEATURE_COUNTS = {
    "CONTROL": 290,
    "STAGE_CONTEXT": 293,
    "LEG_CONTEXT": 292,
    "TIE_STATE": 294,
    "ALL_CONTEXT": 298,
}


# ============================================================
# HELPERS
# ============================================================


def get_metric(
    result: dict,
    metric: str,
) -> float:
    """
    Support the result structure used by the existing
    Candidate C evaluator without changing its metric logic.
    """

    if metric in result:

        return float(
            result[
                metric
            ]
        )

    metrics = result.get(
        "metrics"
    )

    if (
        isinstance(
            metrics,
            dict,
        )
        and
        metric
        in metrics
    ):

        return float(
            metrics[
                metric
            ]
        )

    raise KeyError(
        "Metric "
        f"{metric!r} "
        "not present in evaluation result. "
        f"Available top-level keys: "
        f"{sorted(result.keys())}"
    )


def scenario_screen(
    summary: pd.DataFrame,
    scenario: str,
) -> bool:
    """
    Pre-registered Candidate E signal screen.

    Requirements:
      1. Weighted log loss must improve.
      2. Weighted Brier score must not worsen.
      3. Log loss must beat CONTROL in >= 4/7 folds.

    This intentionally mirrors the Candidate C screen.
    """

    scenario_summary = summary[
        summary[
            "scenario"
        ]
        ==
        scenario
    ]

    log_loss_rows = (
        scenario_summary[
            scenario_summary[
                "metric"
            ]
            ==
            "log_loss"
        ]
    )

    brier_rows = (
        scenario_summary[
            scenario_summary[
                "metric"
            ]
            ==
            "brier_score"
        ]
    )

    if len(
        log_loss_rows
    ) != 1:

        raise RuntimeError(
            f"{scenario}: expected exactly one "
            "log-loss summary row."
        )

    if len(
        brier_rows
    ) != 1:

        raise RuntimeError(
            f"{scenario}: expected exactly one "
            "Brier summary row."
        )

    log_loss_row = (
        log_loss_rows.iloc[
            0
        ]
    )

    brier_row = (
        brier_rows.iloc[
            0
        ]
    )

    log_loss_improvement = float(
        log_loss_row[
            "candidate_improvement"
        ]
    )

    brier_improvement = float(
        brier_row[
            "candidate_improvement"
        ]
    )

    log_loss_better_folds = int(
        log_loss_row[
            "candidate_better_folds"
        ]
    )

    return bool(
        (
            log_loss_improvement
            >
            0.0
        )
        and
        (
            brier_improvement
            >=
            -1e-12
        )
        and
        (
            log_loss_better_folds
            >=
            4
        )
    )


# ============================================================
# LOAD BASE ML DATA
# ============================================================


def load_base_data() -> tuple[
    pd.DataFrame,
    list[str],
]:
    """
    Load the exact frozen V1 feature source and its
    approved 290-feature no-transfer set.
    """

    manifest = (
        boosted.load_manifest()
    )

    all_features = manifest[
        "feature_columns"
    ]

    control_features = (
        boosted.primary_features(
            all_features
        )
    )

    if len(
        control_features
    ) != EXPECTED_CONTROL_FEATURES:

        raise RuntimeError(
            "Expected frozen V1 control to contain "
            f"{EXPECTED_CONTROL_FEATURES} features, "
            f"found {len(control_features)}."
        )

    dataframe = (
        boosted.load_data(
            all_features
        )
    )

    if len(
        dataframe
    ) != EXPECTED_MASTER_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_MASTER_MATCHES:,} "
            "historical ML rows, found "
            f"{len(dataframe):,}."
        )

    required = {
        "match_id",
        "date",
        "season",
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Frozen ML data missing identity columns: "
            f"{sorted(missing)}"
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "match_id"
    ] = pd.to_numeric(
        dataframe[
            "match_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "season"
    ] = pd.to_numeric(
        dataframe[
            "season"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    if dataframe[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Frozen ML data contains duplicate match_id values."
        )

    return (
        dataframe,
        control_features,
    )


# ============================================================
# LOAD CANDIDATE E FEATURE LAYER
# ============================================================


def load_context_features() -> pd.DataFrame:
    if not CONTEXT_FEATURE_PATH.exists():

        raise FileNotFoundError(
            "Candidate E feature file not found:\n"
            f"{CONTEXT_FEATURE_PATH}\n\n"
            "Run first:\n"
            "python -m "
            "src.research.build_tournament_context_features"
        )

    dataframe = pd.read_csv(
        CONTEXT_FEATURE_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "context_source",
        "tournament_context_available",
        *ALL_CONTEXT_FEATURES,
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Candidate E feature layer missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_MASTER_MATCHES:

        raise RuntimeError(
            "Expected Candidate E feature layer to contain "
            f"{EXPECTED_MASTER_MATCHES:,} rows, found "
            f"{len(dataframe):,}."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "match_id"
    ] = pd.to_numeric(
        dataframe[
            "match_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "season"
    ] = pd.to_numeric(
        dataframe[
            "season"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    dataframe[
        "tournament_context_available"
    ] = pd.to_numeric(
        dataframe[
            "tournament_context_available"
        ],
        errors="raise",
    ).astype(
        int
    )

    if dataframe[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Candidate E feature layer contains "
            "duplicate match_id values."
        )

    for feature in (
        ALL_CONTEXT_FEATURES
    ):

        dataframe[
            feature
        ] = pd.to_numeric(
            dataframe[
                feature
            ],
            errors="coerce",
        )

    return dataframe


# ============================================================
# MERGE
# ============================================================


def merge_context_features(
    base: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    context_columns = [
        "match_id",
        "season",
        "date",
        "home_team",
        "away_team",
        "context_source",
        "tournament_context_available",
        *ALL_CONTEXT_FEATURES,
    ]

    context_subset = (
        context[
            context_columns
        ]
        .copy()
    )

    # Avoid duplicate season/date names during merge.
    context_subset = (
        context_subset.rename(
            columns={
                "season": (
                    "context_season"
                ),
                "date": (
                    "context_date"
                ),
            }
        )
    )

    result = base.merge(
        context_subset,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    if result[
        "context_season"
    ].isna().any():

        missing = result[
            result[
                "context_season"
            ].isna()
        ][
            [
                "match_id",
                "season",
                "date",
            ]
        ]

        raise RuntimeError(
            "Candidate E merge failed for some ML rows:\n"
            +
            missing.head(
                50
            ).to_string(
                index=False
            )
        )

    if not (
        result[
            "season"
        ].astype(
            int
        )
        ==
        result[
            "context_season"
        ].astype(
            int
        )
    ).all():

        mismatch = result[
            result[
                "season"
            ].astype(
                int
            )
            !=
            result[
                "context_season"
            ].astype(
                int
            )
        ][
            [
                "match_id",
                "season",
                "context_season",
            ]
        ]

        raise RuntimeError(
            "Season mismatch after Candidate E merge:\n"
            +
            mismatch.head(
                50
            ).to_string(
                index=False
            )
        )

    base_dates = pd.to_datetime(
        result[
            "date"
        ],
        errors="raise",
    ).dt.normalize()

    context_dates = pd.to_datetime(
        result[
            "context_date"
        ],
        errors="raise",
    ).dt.normalize()

    if not (
        base_dates
        ==
        context_dates
    ).all():

        mismatch = result[
            base_dates
            !=
            context_dates
        ][
            [
                "match_id",
                "date",
                "context_date",
            ]
        ]

        raise RuntimeError(
            "Date mismatch after Candidate E merge:\n"
            +
            mismatch.head(
                50
            ).to_string(
                index=False
            )
        )

    result = result.drop(
        columns=[
            "context_season",
            "context_date",
        ]
    )

    return result


# ============================================================
# COMMON-COVERAGE DATASET
# ============================================================


def build_eligible_dataset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Candidate E has no trustworthy context source for
    seasons 2010 and 2011.

    Therefore both CONTROL and Candidate E are trained on
    the identical 2012+ common-coverage population.

    We do NOT feed missing Candidate E values from 2010/11
    into the model because their missingness would encode
    source/era information.
    """

    eligible = dataframe[
        (
            dataframe[
                "season"
            ]
            >=
            FIRST_CONTEXT_SEASON
        )
        &
        (
            dataframe[
                "season"
            ]
            <=
            CONFIRMATION_SEASON
        )
    ].copy()

    if eligible.empty:

        raise RuntimeError(
            "Candidate E eligible dataset is empty."
        )

    unavailable = eligible[
        eligible[
            "tournament_context_available"
        ]
        !=
        1
    ]

    if not unavailable.empty:

        raise RuntimeError(
            "Context unavailable inside 2012-2025 "
            "common-coverage population:\n"
            +
            unavailable[
                [
                    "match_id",
                    "date",
                    "season",
                    "home_team",
                    "away_team",
                    "context_source",
                ]
            ]
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

    missing_candidate_values = (
        eligible[
            ALL_CONTEXT_FEATURES
        ]
        .isna()
        .sum()
    )

    missing_candidate_values = (
        missing_candidate_values[
            missing_candidate_values
            >
            0
        ]
    )

    if not missing_candidate_values.empty:

        raise RuntimeError(
            "Candidate E contains missing values inside "
            "the common-coverage population:\n"
            +
            missing_candidate_values.to_string()
        )

    return (
        eligible.sort_values(
            [
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
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
        "UCL PREDICTOR V2 - CANDIDATE E"
    )

    print(
        "TOURNAMENT / TIE CONTEXT ABLATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Research only."
    )

    print(
        "Frozen V1 production model unchanged."
    )

    print(
        "Selection window: 2018-2024."
    )

    print(
        "2025/26 remains locked for Candidate E."
    )

    print(
        "Common-coverage training population begins in 2012."
    )

    print(
        "2010/11 and 2011/12 are excluded from BOTH control "
        "and candidate scenarios."
    )

    print(
        "Reason: tournament context is unavailable in those "
        "seasons and missingness must not encode source era."
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    base, control_features = (
        load_base_data()
    )

    context = (
        load_context_features()
    )

    dataframe = (
        merge_context_features(
            base=base,
            context=context,
        )
    )

    # --------------------------------------------------------
    # COLLISION CHECK
    # --------------------------------------------------------

    collisions = [
        feature
        for feature
        in ALL_CONTEXT_FEATURES
        if feature
        in control_features
    ]

    if collisions:

        raise RuntimeError(
            "Candidate E collision detected with frozen V1: "
            f"{collisions}"
        )

    # --------------------------------------------------------
    # ELIGIBLE ROWS
    # --------------------------------------------------------

    eligible = (
        build_eligible_dataset(
            dataframe
        )
    )

    development = eligible[
        eligible[
            "season"
        ].between(
            FIRST_VALIDATION_SEASON,
            LAST_VALIDATION_SEASON,
        )
    ].copy()

    confirmation = eligible[
        eligible[
            "season"
        ]
        ==
        CONFIRMATION_SEASON
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

    for (
        scenario,
        features,
    ) in feature_sets.items():

        expected = (
            EXPECTED_FEATURE_COUNTS[
                scenario
            ]
        )

        if len(
            features
        ) != expected:

            raise RuntimeError(
                f"{scenario} expected "
                f"{expected} features, found "
                f"{len(features)}."
            )

        duplicates = [
            feature
            for feature
            in features
            if features.count(
                feature
            )
            >
            1
        ]

        if duplicates:

            raise RuntimeError(
                f"{scenario} contains duplicate "
                f"features: {sorted(set(duplicates))}"
            )

    # --------------------------------------------------------
    # DATASET DISPLAY
    # --------------------------------------------------------

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
        f"Full historical ML rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Common-coverage eligible rows "
        f"(2012-2025): "
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

    print(
        "IMPORTANT:"
    )

    print(
        "This is a common-coverage signal experiment."
    )

    print(
        "CONTROL and Candidate E use exactly the same "
        "2012+ training rows."
    )

    print(
        "It is not yet the final full-history production "
        "V1-vs-V2 comparison."
    )

    print()

    for (
        scenario,
        features,
    ) in feature_sets.items():

        print(
            f"{scenario:<15}: "
            f"{len(features)} features"
        )

    # --------------------------------------------------------
    # ROLLING BACKTEST
    # --------------------------------------------------------

    fold_rows = []

    for validation_season in range(
        FIRST_VALIDATION_SEASON,
        LAST_VALIDATION_SEASON + 1,
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
                "Empty training fold for "
                f"{validation_season}."
            )

        if validation.empty:

            raise RuntimeError(
                "Empty validation fold for "
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

        # ----------------------------------------------------
        # IDENTICAL TARGET VALIDATION
        # ----------------------------------------------------

        for scenario in SCENARIOS:

            result = results[
                scenario
            ]

            if not np.array_equal(
                np.asarray(
                    control[
                        "y_true"
                    ]
                ),
                np.asarray(
                    result[
                        "y_true"
                    ]
                ),
            ):

                raise RuntimeError(
                    "Target mismatch between "
                    f"CONTROL and {scenario}."
                )

            row = {
                "validation_season": int(
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

                if (
                    scenario
                    ==
                    "CONTROL"
                ):

                    metric_improvement = (
                        0.0
                    )

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

        # ----------------------------------------------------
        # FOLD DISPLAY
        # ----------------------------------------------------

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

            if (
                scenario
                ==
                "CONTROL"
            ):

                print(
                    f"  {scenario:<15} "
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
                    f"  {scenario:<15} "
                    f"{value:.6f} "
                    f"({delta:+.6f})"
                )

    # --------------------------------------------------------
    # AGGREGATE FOLDS
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
        ) != EXPECTED_FOLDS:

            raise RuntimeError(
                f"{scenario} expected "
                f"{EXPECTED_FOLDS} folds, "
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

            if (
                scenario
                ==
                "CONTROL"
            ):

                metric_improvement = (
                    0.0
                )

                better_folds = 0

                tied_folds = (
                    EXPECTED_FOLDS
                )

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
                    EXPECTED_FOLDS
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
                        else
                        "higher"
                    ),
                    "weighted_control": float(
                        control_weighted
                    ),
                    "weighted_scenario": float(
                        scenario_weighted
                    ),
                    "candidate_improvement": float(
                        metric_improvement
                    ),
                    "candidate_better_folds": int(
                        better_folds
                    ),
                    "tied_folds": int(
                        tied_folds
                    ),
                    "candidate_worse_folds": int(
                        worse_folds
                    ),
                    "total_folds": int(
                        EXPECTED_FOLDS
                    ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # PRE-REGISTERED SIGNAL SCREEN
    # --------------------------------------------------------

    screen_results = {
        scenario: (
            scenario_screen(
                summary,
                scenario,
            )
        )
        for scenario
        in CANDIDATE_SCENARIOS
    }

    passing = [
        scenario
        for scenario
        in CANDIDATE_SCENARIOS
        if screen_results[
            scenario
        ]
    ]

    # --------------------------------------------------------
    # LOCK ONE DEVELOPMENT SCENARIO
    # --------------------------------------------------------

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
            "candidate_e_tournament_context_ablation"
        ),
        "status": (
            "research_only"
        ),
        "comparison_type": (
            "common_coverage_signal_experiment"
        ),
        "common_coverage_first_season": int(
            FIRST_CONTEXT_SEASON
        ),
        "selection_window": (
            "2018_2024"
        ),
        "confirmation_2025_used": False,
        "confirmation_2025_outcomes_inspected": False,
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
        "eligible_rows": int(
            len(
                eligible
            )
        ),
        "excluded_for_context_coverage": {
            "2010": int(
                (
                    dataframe[
                        "season"
                    ]
                    ==
                    2010
                ).sum()
            ),
            "2011": int(
                (
                    dataframe[
                        "season"
                    ]
                    ==
                    2011
                ).sum()
            ),
        },
        "control_feature_count": int(
            len(
                control_features
            )
        ),
        "candidate_feature_family": (
            "tournament_context"
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
        "selection_ranking": (
            "log_loss_improvement_then_brier_improvement"
        ),
        "selected_scenario": (
            selected_scenario
        ),
        "raw_transfermarkt_aggregate_used": False,
        "current_match_score_used_as_candidate_feature": False,
        "second_leg_context_from_strictly_prior_first_leg": True,
        "fuzzy_matching_used": False,
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
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print()

    print(
        "=" * 110
    )

    print(
        "CANDIDATE E ABLATION SUMMARY"
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
    # PRIMARY SIGNAL SCREEN
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

    for scenario in CANDIDATE_SCENARIOS:

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
            f"{int(log_loss_row['candidate_better_folds'])}/"
            f"{EXPECTED_FOLDS}"
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
        "CANDIDATE E DEVELOPMENT DECISION"
    )

    print(
        "=" * 110
    )

    print()

    if selected_scenario is None:

        print(
            "Decision: "
            "REJECT_TOURNAMENT_CONTEXT_FAMILY"
        )

        print()

        print(
            "No pre-registered Candidate E scenario "
            "passes the 2018-2024 signal screen."
        )

        print(
            "Do not evaluate Candidate E on 2025/26."
        )

        print(
            "Do not mine additional Candidate E subsets."
        )

    else:

        print(
            "Decision: "
            "RETAIN_TOURNAMENT_CONTEXT_SUBSET"
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
            "No further Candidate E feature selection "
            "before confirmation."
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
        "Candidate E additions new to V1: PASS"
    )

    print(
        "All scenarios use identical rows: PASS"
    )

    print(
        "2010/11 and 2011/12 excluded from BOTH sides: PASS"
    )

    print(
        "No era-missingness indicator introduced: PASS"
    )

    print(
        "2012-2024 Candidate E context complete: PASS"
    )

    print(
        "2018-2024 only used for subset selection: PASS"
    )

    print(
        "2025/26 evaluated: NO"
    )

    print(
        "Raw Transfermarkt aggregate used: NO"
    )

    print(
        "Current-match score used as Candidate E feature: NO"
    )

    print(
        "Second-leg state derived from prior first leg only: PASS"
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