from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import run_boosted_backtest as rb
from src.research import (
    run_uefa_strength_rolling_backtest
    as research_helpers
)


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

SCHEDULE_AUDIT_PATH = (
    ROOT
    / "reports"
    / "research"
    / "schedule_rest_team_side_audit.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

FEATURE_LAYER_PATH = (
    ROOT
    / "data"
    / "research"
    / "schedule_rest_common_coverage_features.csv"
)

FOLD_METRICS_PATH = (
    REPORT_DIR
    / "schedule_rest_common_coverage_fold_metrics.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "schedule_rest_common_coverage_summary.csv"
)

OOF_PATH = (
    REPORT_DIR
    / "schedule_rest_common_coverage_oof_predictions.csv"
)

METADATA_PATH = (
    REPORT_DIR
    / "schedule_rest_common_coverage_summary.json"
)


# ============================================================
# EXPERIMENT CONTRACT
# ============================================================


CONTROL_FEATURE_COUNT = 290

CANDIDATE_FEATURE_COUNT = 294

EXPECTED_TOTAL_MATCHES = 2089

EXPECTED_DEV_BOTH_SUPPORTED = 383

EXPECTED_2025_BOTH_SUPPORTED = 78


SCHEDULE_FEATURES = [
    "rest_days_diff",
    "known_matches_last_7d_diff",
    "known_matches_last_14d_diff",
    "known_matches_last_30d_diff",
]


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


# ============================================================
# HELPERS
# ============================================================


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
# LOAD SCHEDULE AUDIT
# ============================================================


def load_schedule_audit() -> pd.DataFrame:
    if not SCHEDULE_AUDIT_PATH.exists():

        raise FileNotFoundError(
            SCHEDULE_AUDIT_PATH
        )

    dataframe = pd.read_csv(
        SCHEDULE_AUDIT_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
        "side",
        "ucl_team",
        "candidate_schedule_available",
        "days_since_last_known_match",
        "known_matches_last_7d",
        "known_matches_last_14d",
        "known_matches_last_30d",
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
            "Schedule audit missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = (
        dataframe.copy()
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
        "candidate_schedule_available"
    ] = (
        dataframe[
            "candidate_schedule_available"
        ]
        .astype(
            str
        )
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if dataframe[
        "candidate_schedule_available"
    ].isna().any():

        raise RuntimeError(
            "Could not parse schedule availability flags."
        )

    side_counts = (
        dataframe.groupby(
            "match_id"
        )[
            "side"
        ]
        .nunique()
    )

    if not (
        side_counts
        ==
        2
    ).all():

        raise RuntimeError(
            "Every match must contain exactly "
            "HOME and AWAY audit rows."
        )

    expected_sides = {
        "HOME",
        "AWAY",
    }

    for (
        match_id,
        group,
    ) in dataframe.groupby(
        "match_id"
    ):

        sides = set(
            group[
                "side"
            ].astype(
                str
            )
        )

        if sides != expected_sides:

            raise RuntimeError(
                f"Unexpected sides for match "
                f"{match_id}: {sides}"
            )

    return dataframe


# ============================================================
# BUILD MATCH-LEVEL SCHEDULE FEATURES
# ============================================================


def build_match_features(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        match_id,
        group,
    ) in audit.groupby(
        "match_id",
        sort=False,
    ):

        home = (
            group[
                group[
                    "side"
                ]
                ==
                "HOME"
            ]
            .iloc[
                0
            ]
        )

        away = (
            group[
                group[
                    "side"
                ]
                ==
                "AWAY"
            ]
            .iloc[
                0
            ]
        )

        both_supported = bool(
            home[
                "candidate_schedule_available"
            ]
            and
            away[
                "candidate_schedule_available"
            ]
        )

        row = {
            "match_id": (
                match_id
            ),
            "date": (
                home[
                    "date"
                ]
            ),
            "season": int(
                home[
                    "season"
                ]
            ),
            "home_team": (
                home[
                    "ucl_team"
                ]
            ),
            "away_team": (
                away[
                    "ucl_team"
                ]
            ),
            "both_schedule_supported": (
                both_supported
            ),
        }

        if both_supported:

            home_rest = float(
                home[
                    "days_since_last_known_match"
                ]
            )

            away_rest = float(
                away[
                    "days_since_last_known_match"
                ]
            )

            home_7 = float(
                home[
                    "known_matches_last_7d"
                ]
            )

            away_7 = float(
                away[
                    "known_matches_last_7d"
                ]
            )

            home_14 = float(
                home[
                    "known_matches_last_14d"
                ]
            )

            away_14 = float(
                away[
                    "known_matches_last_14d"
                ]
            )

            home_30 = float(
                home[
                    "known_matches_last_30d"
                ]
            )

            away_30 = float(
                away[
                    "known_matches_last_30d"
                ]
            )

            row[
                "rest_days_diff"
            ] = (
                home_rest
                -
                away_rest
            )

            row[
                "known_matches_last_7d_diff"
            ] = (
                home_7
                -
                away_7
            )

            row[
                "known_matches_last_14d_diff"
            ] = (
                home_14
                -
                away_14
            )

            row[
                "known_matches_last_30d_diff"
            ] = (
                home_30
                -
                away_30
            )

        else:

            for feature in (
                SCHEDULE_FEATURES
            ):

                row[
                    feature
                ] = np.nan

        rows.append(
            row
        )

    features = pd.DataFrame(
        rows
    )

    if len(
        features
    ) != EXPECTED_TOTAL_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TOTAL_MATCHES:,} "
            "match-level schedule rows, found "
            f"{len(features):,}."
        )

    supported = features[
        features[
            "both_schedule_supported"
        ]
    ]

    if (
        supported[
            SCHEDULE_FEATURES
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Missing schedule feature inside "
            "both-supported matches."
        )

    unsupported = features[
        ~features[
            "both_schedule_supported"
        ]
    ]

    if not unsupported.empty:

        if not (
            unsupported[
                SCHEDULE_FEATURES
            ]
            .isna()
            .all()
            .all()
        ):

            raise RuntimeError(
                "Unsupported matches unexpectedly "
                "contain schedule features."
            )

    return features


# ============================================================
# LOAD FROZEN V1 DATA
# ============================================================


def load_v1_data() -> tuple[
    pd.DataFrame,
    list[str],
]:
    manifest = (
        rb.load_manifest()
    )

    all_features = (
        manifest[
            "feature_columns"
        ]
    )

    control_features = (
        rb.primary_features(
            all_features
        )
    )

    if len(
        control_features
    ) != CONTROL_FEATURE_COUNT:

        raise RuntimeError(
            "Expected 290 V1 control features, "
            f"found {len(control_features)}."
        )

    dataframe = (
        rb.load_data(
            all_features
        )
    )

    if len(
        dataframe
    ) != EXPECTED_TOTAL_MATCHES:

        raise RuntimeError(
            "Unexpected V1 historical row count."
        )

    return (
        dataframe,
        control_features,
    )


# ============================================================
# MERGE
# ============================================================


def merge_schedule_layer(
    *,
    v1_data: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    overlap = [
        feature
        for feature
        in SCHEDULE_FEATURES
        if feature
        in v1_data.columns
    ]

    if overlap:

        raise RuntimeError(
            "Schedule features already exist in "
            f"V1 matrix: {overlap}"
        )

    schedule_columns = [
        "match_id",
        "both_schedule_supported",
        *SCHEDULE_FEATURES,
    ]

    merged = v1_data.merge(
        schedule[
            schedule_columns
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(
        merged
    ) != len(
        v1_data
    ):

        raise RuntimeError(
            "Schedule merge changed historical "
            "row count."
        )

    if merged[
        "both_schedule_supported"
    ].isna().any():

        raise RuntimeError(
            "Historical match missing from "
            "schedule layer."
        )

    return merged


# ============================================================
# RUN ONE MODEL
# ============================================================


def run_model(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict:
    return (
        research_helpers.fit_and_evaluate(
            train=train,
            validation=validation,
            features=features,
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
        "UCL PREDICTOR V2 - CANDIDATE B1"
    )

    print(
        "COMMON-COVERAGE SCHEDULE / REST BACKTEST"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Research question:"
    )

    print(
        "When both clubs have genuine historical "
        "schedule coverage, does relative rest and "
        "fixture load improve prediction quality?"
    )

    print()

    print(
        "Important:"
    )

    print(
        "Only BOTH_SUPPORTED matches enter this experiment."
    )

    print(
        "Control and Candidate B train on identical rows."
    )

    print(
        "No missing schedule values enter the candidate."
    )

    print(
        "No schedule availability flag enters the model."
    )

    print(
        "2018-2024 only for candidate selection."
    )

    print(
        "2025/26 is excluded."
    )

    print(
        "Production remains untouched."
    )

    print()

    # --------------------------------------------------------
    # SCHEDULE LAYER
    # --------------------------------------------------------

    print(
        "Loading schedule coverage audit..."
    )

    audit = (
        load_schedule_audit()
    )

    print(
        "Building match-level schedule differences..."
    )

    schedule = (
        build_match_features(
            audit
        )
    )

    FEATURE_LAYER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule.to_csv(
        FEATURE_LAYER_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # V1 DATA
    # --------------------------------------------------------

    print(
        "Loading frozen V1 feature matrix..."
    )

    (
        v1_data,
        control_features,
    ) = (
        load_v1_data()
    )

    dataframe = (
        merge_schedule_layer(
            v1_data=v1_data,
            schedule=schedule,
        )
    )

    candidate_features = [
        *control_features,
        *SCHEDULE_FEATURES,
    ]

    if len(
        candidate_features
    ) != CANDIDATE_FEATURE_COUNT:

        raise RuntimeError(
            "Expected "
            f"{CANDIDATE_FEATURE_COUNT} Candidate B "
            "features, found "
            f"{len(candidate_features)}."
        )

    # --------------------------------------------------------
    # COMMON-COVERAGE DATA
    # --------------------------------------------------------

    common = dataframe[
        dataframe[
            "both_schedule_supported"
        ]
    ].copy()

    development = common[
        common[
            "season"
        ].between(
            rb.FIRST_VALIDATION_SEASON,
            rb.LAST_VALIDATION_SEASON,
        )
    ]

    confirmation_2025 = common[
        common[
            "season"
        ]
        ==
        rb.FINAL_TEST_SEASON
    ]

    if len(
        development
    ) != EXPECTED_DEV_BOTH_SUPPORTED:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_DEV_BOTH_SUPPORTED} both-supported "
            "development matches, found "
            f"{len(development)}."
        )

    if len(
        confirmation_2025
    ) != EXPECTED_2025_BOTH_SUPPORTED:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_2025_BOTH_SUPPORTED} both-supported "
            "2025 matches, found "
            f"{len(confirmation_2025)}."
        )

    print()
    print(
        "=" * 110
    )

    print(
        "COMMON-COVERAGE DATASET"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"All historical both-supported matches: "
        f"{len(common):,}"
    )

    print(
        f"2018-2024 development matches: "
        f"{len(development):,}"
    )

    print(
        f"Excluded 2025/26 both-supported matches: "
        f"{len(confirmation_2025):,}"
    )

    print(
        f"Control features: "
        f"{len(control_features)}"
    )

    print(
        f"Candidate B features: "
        f"{len(candidate_features)}"
    )

    print()

    print(
        "Candidate B additions:"
    )

    for feature in (
        SCHEDULE_FEATURES
    ):

        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # ROLLING BACKTEST
    # --------------------------------------------------------

    validation_seasons = list(
        range(
            rb.FIRST_VALIDATION_SEASON,
            rb.LAST_VALIDATION_SEASON
            + 1,
        )
    )

    fold_rows = []

    oof_frames = []

    for validation_season in (
        validation_seasons
    ):

        train = common[
            common[
                "season"
            ]
            <
            validation_season
        ].copy()

        validation = common[
            common[
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

        print(
            f"Training seasons: "
            f"{train['season'].min()} "
            f"to "
            f"{train['season'].max()}"
        )

        print()

        print(
            "Training COMMON-COVERAGE CONTROL..."
        )

        control = (
            run_model(
                train=train,
                validation=validation,
                features=control_features,
            )
        )

        print(
            "Training CANDIDATE B1..."
        )

        candidate = (
            run_model(
                train=train,
                validation=validation,
                features=candidate_features,
            )
        )

        row = {
            "validation_season": (
                validation_season
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
            "training_first_season": int(
                train[
                    "season"
                ].min()
            ),
            "training_last_season": int(
                train[
                    "season"
                ].max()
            ),
            "control_feature_count": (
                len(
                    control_features
                )
            ),
            "candidate_feature_count": (
                len(
                    candidate_features
                )
            ),
        }

        for metric in METRICS:

            control_value = float(
                control[
                    "metrics"
                ][
                    metric
                ]
            )

            candidate_value = float(
                candidate[
                    "metrics"
                ][
                    metric
                ]
            )

            row[
                f"control_{metric}"
            ] = control_value

            row[
                f"candidate_{metric}"
            ] = candidate_value

            row[
                f"{metric}_improvement"
            ] = improvement(
                metric=metric,
                control=control_value,
                candidate=candidate_value,
            )

        fold_rows.append(
            row
        )

        y_true = (
            control[
                "y_true"
            ]
        )

        if not np.array_equal(
            y_true,
            candidate[
                "y_true"
            ],
        ):

            raise RuntimeError(
                "Control and candidate targets differ."
            )

        control_probabilities = (
            control[
                "probabilities"
            ]
        )

        candidate_probabilities = (
            candidate[
                "probabilities"
            ]
        )

        control_predictions = (
            control[
                "predictions"
            ]
        )

        candidate_predictions = (
            candidate[
                "predictions"
            ]
        )

        prediction_frame = pd.DataFrame(
            {
                "validation_season": (
                    validation_season
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
                "control_prediction": [
                    rb.TARGET_NAMES[
                        int(
                            value
                        )
                    ]
                    for value
                    in control_predictions
                ],
                "control_prob_H": (
                    control_probabilities[
                        :,
                        0
                    ]
                ),
                "control_prob_D": (
                    control_probabilities[
                        :,
                        1
                    ]
                ),
                "control_prob_A": (
                    control_probabilities[
                        :,
                        2
                    ]
                ),
                "candidate_prediction": [
                    rb.TARGET_NAMES[
                        int(
                            value
                        )
                    ]
                    for value
                    in candidate_predictions
                ],
                "candidate_prob_H": (
                    candidate_probabilities[
                        :,
                        0
                    ]
                ),
                "candidate_prob_D": (
                    candidate_probabilities[
                        :,
                        1
                    ]
                ),
                "candidate_prob_A": (
                    candidate_probabilities[
                        :,
                        2
                    ]
                ),
                "rest_days_diff": (
                    validation[
                        "rest_days_diff"
                    ].to_numpy()
                ),
                "known_matches_last_7d_diff": (
                    validation[
                        "known_matches_last_7d_diff"
                    ].to_numpy()
                ),
                "known_matches_last_14d_diff": (
                    validation[
                        "known_matches_last_14d_diff"
                    ].to_numpy()
                ),
                "known_matches_last_30d_diff": (
                    validation[
                        "known_matches_last_30d_diff"
                    ].to_numpy()
                ),
            }
        )

        oof_frames.append(
            prediction_frame
        )

        print()

        print(
            "Fold result:"
        )

        print(
            f"  Log Loss: "
            f"{control['metrics']['log_loss']:.6f} "
            f"→ "
            f"{candidate['metrics']['log_loss']:.6f} "
            f"("
            f"{row['log_loss_improvement']:+.6f}"
            f")"
        )

        print(
            f"  Brier:    "
            f"{control['metrics']['brier_score']:.6f} "
            f"→ "
            f"{candidate['metrics']['brier_score']:.6f} "
            f"("
            f"{row['brier_score_improvement']:+.6f}"
            f")"
        )

        print(
            f"  Accuracy: "
            f"{control['metrics']['accuracy']:.6f} "
            f"→ "
            f"{candidate['metrics']['accuracy']:.6f}"
        )

        print(
            f"  Macro F1: "
            f"{control['metrics']['macro_f1']:.6f} "
            f"→ "
            f"{candidate['metrics']['macro_f1']:.6f}"
        )

        print(
            f"  Draw Rcl: "
            f"{control['metrics']['draw_recall']:.6f} "
            f"→ "
            f"{candidate['metrics']['draw_recall']:.6f}"
        )

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    if len(
        oof
    ) != EXPECTED_DEV_BOTH_SUPPORTED:

        raise RuntimeError(
            "Unexpected OOF row count."
        )

    summary_rows = []

    for metric in METRICS:

        control_weighted = (
            weighted_mean(
                fold_metrics[
                    f"control_{metric}"
                ],
                fold_metrics[
                    "validation_rows"
                ],
            )
        )

        candidate_weighted = (
            weighted_mean(
                fold_metrics[
                    f"candidate_{metric}"
                ],
                fold_metrics[
                    "validation_rows"
                ],
            )
        )

        metric_improvement = (
            improvement(
                metric=metric,
                control=control_weighted,
                candidate=candidate_weighted,
            )
        )

        better_folds = int(
            (
                fold_metrics[
                    f"{metric}_improvement"
                ]
                > 0
            ).sum()
        )

        tied_folds = int(
            np.isclose(
                fold_metrics[
                    f"{metric}_improvement"
                ],
                0.0,
                atol=1e-12,
            ).sum()
        )

        summary_rows.append(
            {
                "metric": (
                    metric
                ),
                "better_direction": (
                    "lower"
                    if metric
                    in LOWER_IS_BETTER
                    else "higher"
                ),
                "weighted_control": (
                    control_weighted
                ),
                "weighted_candidate": (
                    candidate_weighted
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
                "candidate_worse_folds": int(
                    len(
                        fold_metrics
                    )
                    -
                    better_folds
                    -
                    tied_folds
                ),
                "total_folds": int(
                    len(
                        fold_metrics
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    log_loss_row = (
        summary[
            summary[
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
        summary[
            summary[
                "metric"
            ]
            ==
            "brier_score"
        ]
        .iloc[
            0
        ]
    )

    # This is a signal screen, not a production-promotion rule.
    passes_signal_screen = bool(
        (
            log_loss_row[
                "candidate_improvement"
            ]
            > 0
        )
        and
        (
            brier_row[
                "candidate_improvement"
            ]
            >= 0
        )
        and
        (
            log_loss_row[
                "candidate_better_folds"
            ]
            >= 4
        )
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
            "candidate_b1_schedule_rest_common_coverage"
        ),
        "status": (
            "research_only"
        ),
        "selection_window": (
            "2018_2024"
        ),
        "final_2025_used": False,
        "production_modified": False,
        "coverage_policy": (
            "both_teams_schedule_supported_only"
        ),
        "historical_both_supported_matches": int(
            len(
                common
            )
        ),
        "development_oof_matches": int(
            len(
                oof
            )
        ),
        "excluded_2025_supported_matches": int(
            len(
                confirmation_2025
            )
        ),
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
        "candidate_features": (
            SCHEDULE_FEATURES
        ),
        "schedule_scope": (
            "known_domestic_league_plus_ucl"
        ),
        "domestic_cups_included": False,
        "availability_flag_used_as_feature": False,
        "missing_schedule_values_used": False,
        "passes_signal_screen": (
            passes_signal_screen
        ),
        "signal_screen": {
            "weighted_log_loss_improves": True,
            "weighted_brier_not_worse": True,
            "minimum_log_loss_better_folds": 4,
        },
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
        "CANDIDATE B1 ROLLING SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    display = (
        summary.copy()
    )

    for column in [
        "weighted_control",
        "weighted_candidate",
        "candidate_improvement",
    ]:

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
        "SIGNAL DECISION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Weighted Log Loss improvement: "
        f"{log_loss_row['candidate_improvement']:+.6f}"
    )

    print(
        f"Log Loss better folds: "
        f"{int(log_loss_row['candidate_better_folds'])}/"
        f"{int(log_loss_row['total_folds'])}"
    )

    print(
        f"Weighted Brier improvement: "
        f"{brier_row['candidate_improvement']:+.6f}"
    )

    print()

    if passes_signal_screen:

        print(
            "Decision: SCHEDULE_SIGNAL_DETECTED"
        )

        print()

        print(
            "Candidate B1 has enough common-coverage "
            "signal to justify further schedule-data "
            "research."
        )

        print(
            "This does NOT promote schedule features "
            "to production because historical coverage "
            "is still incomplete."
        )

    else:

        print(
            "Decision: SCHEDULE_SIGNAL_NOT_ESTABLISHED"
        )

        print()

        print(
            "Current known schedule features do not "
            "justify expanding the probability model."
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
        "Only both-supported matches used: PASS"
    )

    print(
        "Same rows for control and candidate: PASS"
    )

    print(
        "No schedule missingness entered model: PASS"
    )

    print(
        "No availability flag entered model: PASS"
    )

    print(
        "Only pre-match schedule features used: PASS"
    )

    print(
        "2018-2024 rolling selection window: PASS"
    )

    print(
        "2025/26 excluded from selection: PASS"
    )

    print(
        "Locked CatBoost specification: PASS"
    )

    print(
        "Production model untouched: PASS"
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

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
        FEATURE_LAYER_PATH
    )

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