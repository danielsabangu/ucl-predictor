from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.ml import run_boosted_backtest as rb


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

ADVANCED_FEATURE_PATH = (
    ROOT
    / "data"
    / "research"
    / "ucl_domestic_advanced_features.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

FOLD_METRICS_PATH = (
    REPORT_DIR
    / "domestic_advanced_common_coverage_fold_metrics.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "domestic_advanced_common_coverage_summary.csv"
)

OOF_PATH = (
    REPORT_DIR
    / "domestic_advanced_common_coverage_oof_predictions.csv"
)

METADATA_PATH = (
    REPORT_DIR
    / "domestic_advanced_common_coverage_summary.json"
)


# ============================================================
# EXPERIMENT CONTRACT
# ============================================================


EXPECTED_TOTAL_MATCHES = 2089

EXPECTED_CONTROL_FEATURES = 290

EXPECTED_CANDIDATE_FEATURES = 302

EXPECTED_DEVELOPMENT_MATCHES = 383

EXPECTED_2025_CONFIRMATION_MATCHES = 78


FIRST_VALIDATION_SEASON = 2018

LAST_VALIDATION_SEASON = 2024

LOCKED_CONFIRMATION_SEASON = 2025


TARGET_NAMES = {
    0: "H",
    1: "D",
    2: "A",
}


CANDIDATE_FEATURES = [
    "shots_for_diff_last5",
    "shots_against_diff_last5",
    "sot_for_diff_last5",
    "sot_against_diff_last5",
    "corners_for_diff_last5",
    "corners_against_diff_last5",

    "shots_for_diff_last10",
    "shots_against_diff_last10",
    "sot_for_diff_last10",
    "sot_against_diff_last10",
    "corners_for_diff_last10",
    "corners_against_diff_last10",
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
    "ece",
]


LOWER_IS_BETTER = {
    "log_loss",
    "brier_score",
    "ece",
}


# ============================================================
# HELPERS
# ============================================================


def parse_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):

        return series.astype(
            bool
        )

    parsed = (
        series.astype(
            str
        )
        .str.strip()
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

    if parsed.isna().any():

        bad = (
            series[
                parsed.isna()
            ]
            .astype(
                str
            )
            .unique()
        )

        raise RuntimeError(
            "Could not parse boolean values: "
            f"{bad.tolist()}"
        )

    return parsed.astype(
        bool
    )


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


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.eye(
        3,
        dtype=float,
    )[
        y_true.astype(
            int
        )
    ]

    squared_error = (
        probabilities
        -
        one_hot
    ) ** 2

    return float(
        squared_error.sum(
            axis=1
        ).mean()
    )


def top_label_ece(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    correct = (
        predictions
        ==
        y_true
    ).astype(
        float
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    total = len(
        y_true
    )

    if total == 0:

        return np.nan

    ece = 0.0

    for index in range(
        bins
    ):

        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if index == 0:

            mask = (
                confidence
                >=
                lower
            ) & (
                confidence
                <=
                upper
            )

        else:

            mask = (
                confidence
                >
                lower
            ) & (
                confidence
                <=
                upper
            )

        count = int(
            mask.sum()
        )

        if count == 0:

            continue

        bin_accuracy = float(
            correct[
                mask
            ].mean()
        )

        bin_confidence = float(
            confidence[
                mask
            ].mean()
        )

        ece += (
            count
            /
            total
        ) * abs(
            bin_accuracy
            -
            bin_confidence
        )

    return float(
        ece
    )


# ============================================================
# LOCKED MODEL
# ============================================================


def build_locked_pipeline() -> Pipeline:
    """
    Exact CatBoost specification used by the frozen V1
    evaluation/backtest policy.

    No Candidate C hyperparameter tuning.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                CatBoostClassifier(
                    iterations=700,
                    learning_rate=0.025,
                    depth=5,
                    loss_function="MultiClass",
                    l2_leaf_reg=5.0,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


# ============================================================
# MODEL EVALUATION
# ============================================================


def fit_and_evaluate(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict:
    X_train = train[
        features
    ]

    X_validation = validation[
        features
    ]

    y_train = (
        pd.to_numeric(
            train[
                "target"
            ],
            errors="raise",
        )
        .astype(
            int
        )
        .to_numpy()
    )

    y_validation = (
        pd.to_numeric(
            validation[
                "target"
            ],
            errors="raise",
        )
        .astype(
            int
        )
        .to_numpy()
    )

    model = (
        build_locked_pipeline()
    )

    model.fit(
        X_train,
        y_train,
    )

    catboost_classes = (
        model.named_steps[
            "model"
        ].classes_
    )

    expected_classes = np.array(
        [
            0,
            1,
            2,
        ]
    )

    if not np.array_equal(
        catboost_classes,
        expected_classes,
    ):

        raise RuntimeError(
            "Unexpected CatBoost class order: "
            f"{catboost_classes}"
        )

    probabilities = (
        model.predict_proba(
            X_validation
        )
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    recalls = recall_score(
        y_validation,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
        average=None,
        zero_division=0,
    )

    metrics = {
        "log_loss": float(
            log_loss(
                y_validation,
                probabilities,
                labels=[
                    0,
                    1,
                    2,
                ],
            )
        ),
        "brier_score": (
            multiclass_brier_score(
                y_validation,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "home_recall": float(
            recalls[
                0
            ]
        ),
        "draw_recall": float(
            recalls[
                1
            ]
        ),
        "away_recall": float(
            recalls[
                2
            ]
        ),
        "ece": (
            top_label_ece(
                y_true=y_validation,
                probabilities=probabilities,
            )
        ),
    }

    return {
        "metrics": metrics,
        "y_true": (
            y_validation
        ),
        "probabilities": (
            probabilities
        ),
        "predictions": (
            predictions
        ),
    }


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
    ) != EXPECTED_CONTROL_FEATURES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_CONTROL_FEATURES} "
            "frozen V1 features, found "
            f"{len(control_features)}."
        )

    collisions = [
        feature
        for feature
        in CANDIDATE_FEATURES
        if feature
        in control_features
    ]

    if collisions:

        raise RuntimeError(
            "Candidate C feature collision "
            f"with frozen V1: {collisions}"
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
            "Expected "
            f"{EXPECTED_TOTAL_MATCHES:,} "
            "V1 historical rows, found "
            f"{len(dataframe):,}."
        )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "result",
        "target",
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
            "V1 data missing columns: "
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

    return (
        dataframe,
        control_features,
    )


# ============================================================
# LOAD CANDIDATE C FEATURE LAYER
# ============================================================


def load_advanced_features() -> pd.DataFrame:
    if not ADVANCED_FEATURE_PATH.exists():

        raise FileNotFoundError(
            ADVANCED_FEATURE_PATH
        )

    dataframe = pd.read_csv(
        ADVANCED_FEATURE_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "home_advanced_history_matches",
        "away_advanced_history_matches",
        "both_advanced_history_available",
        *CANDIDATE_FEATURES,
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
            "Candidate C feature layer missing "
            f"columns: {sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_TOTAL_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TOTAL_MATCHES:,} "
            "Candidate C rows, found "
            f"{len(dataframe):,}."
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
        "home_advanced_history_matches"
    ] = pd.to_numeric(
        dataframe[
            "home_advanced_history_matches"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "away_advanced_history_matches"
    ] = pd.to_numeric(
        dataframe[
            "away_advanced_history_matches"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "both_advanced_history_available"
    ] = parse_bool_series(
        dataframe[
            "both_advanced_history_available"
        ]
    )

    for feature in CANDIDATE_FEATURES:

        dataframe[
            feature
        ] = pd.to_numeric(
            dataframe[
                feature
            ],
            errors="coerce",
        )

    if dataframe[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate match_id values in "
            "Candidate C feature layer."
        )

    return dataframe


# ============================================================
# MERGE
# ============================================================


def merge_candidate_layer(
    *,
    v1_data: pd.DataFrame,
    advanced: pd.DataFrame,
) -> pd.DataFrame:
    candidate_columns = [
        "match_id",
        "home_advanced_history_matches",
        "away_advanced_history_matches",
        "both_advanced_history_available",
        *CANDIDATE_FEATURES,
    ]

    merged = v1_data.merge(
        advanced[
            candidate_columns
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
            "Candidate C merge changed "
            "historical row count."
        )

    if merged[
        "both_advanced_history_available"
    ].isna().any():

        missing_ids = (
            merged.loc[
                merged[
                    "both_advanced_history_available"
                ].isna(),
                "match_id",
            ]
            .head(
                20
            )
            .tolist()
        )

        raise RuntimeError(
            "V1 matches missing from Candidate C "
            f"layer: {missing_ids}"
        )

    return merged


# ============================================================
# COMMON-COVERAGE ELIGIBILITY
# ============================================================


def select_eligible_matches(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Candidate C's experiment requires:

    1. both clubs have historical advanced-stat data;
    2. both clubs have at least 10 prior domestic matches.

    The >=10 rule means every last-5 and last-10 feature has
    the literal window size stated by its name.
    """

    eligible = dataframe[
        dataframe[
            "both_advanced_history_available"
        ]
        &
        (
            dataframe[
                "home_advanced_history_matches"
            ]
            >=
            10
        )
        &
        (
            dataframe[
                "away_advanced_history_matches"
            ]
            >=
            10
        )
    ].copy()

    if eligible.empty:

        raise RuntimeError(
            "No Candidate C eligible matches."
        )

    missing_candidate = (
        eligible[
            CANDIDATE_FEATURES
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if missing_candidate.any():

        examples = (
            eligible.loc[
                missing_candidate,
                [
                    "match_id",
                    "date",
                    "season",
                    "home_team",
                    "away_team",
                ],
            ]
            .head(
                20
            )
        )

        raise RuntimeError(
            "Eligible Candidate C matches "
            "contain missing values:\n"
            +
            examples.to_string(
                index=False
            )
        )

    return eligible


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=" * 110
    )

    print(
        "UCL PREDICTOR V2 - CANDIDATE C1"
    )

    print(
        "DOMESTIC ADVANCED PERFORMANCE "
        "COMMON-COVERAGE BACKTEST"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Research question:"
    )

    print(
        "When both clubs have sufficient historical "
        "domestic advanced-stat coverage, do rolling "
        "shots, shots-on-target and corners improve "
        "UCL prediction quality?"
    )

    print()

    print(
        "Experiment rules:"
    )

    print(
        "  frozen V1 = 290 features"
    )

    print(
        "  Candidate C1 = V1 + 12 advanced features"
    )

    print(
        "  control and candidate use identical rows"
    )

    print(
        "  both teams require >=10 prior domestic matches"
    )

    print(
        "  2018-2024 only for feature-family selection"
    )

    print(
        "  2025/26 remains locked"
    )

    print(
        "  exact frozen CatBoost specification"
    )

    print(
        "  no production changes"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "Loading frozen V1 data..."
    )

    (
        v1_data,
        control_features,
    ) = load_v1_data()

    print(
        "Loading Candidate C feature layer..."
    )

    advanced = (
        load_advanced_features()
    )

    print(
        "Merging research feature layer..."
    )

    dataframe = (
        merge_candidate_layer(
            v1_data=v1_data,
            advanced=advanced,
        )
    )

    candidate_features = [
        *control_features,
        *CANDIDATE_FEATURES,
    ]

    if len(
        candidate_features
    ) != EXPECTED_CANDIDATE_FEATURES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_CANDIDATE_FEATURES} "
            "Candidate C features, found "
            f"{len(candidate_features)}."
        )

    # --------------------------------------------------------
    # ELIGIBLE COMMON COVERAGE
    # --------------------------------------------------------

    eligible = (
        select_eligible_matches(
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

    confirmation_2025 = eligible[
        eligible[
            "season"
        ]
        ==
        LOCKED_CONFIRMATION_SEASON
    ].copy()

    if len(
        development
    ) != EXPECTED_DEVELOPMENT_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_DEVELOPMENT_MATCHES} "
            "2018-2024 Candidate C matches, "
            f"found {len(development)}."
        )

    if len(
        confirmation_2025
    ) != EXPECTED_2025_CONFIRMATION_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_2025_CONFIRMATION_MATCHES} "
            "locked 2025 matches, found "
            f"{len(confirmation_2025)}."
        )

    print()

    print(
        "=" * 110
    )

    print(
        "ELIGIBLE COMMON-COVERAGE DATA"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"All eligible historical matches: "
        f"{len(eligible):,}"
    )

    print(
        f"2018-2024 development matches: "
        f"{len(development):,}"
    )

    print(
        f"Locked 2025/26 matches: "
        f"{len(confirmation_2025):,}"
    )

    print(
        f"Control feature count: "
        f"{len(control_features)}"
    )

    print(
        f"Candidate feature count: "
        f"{len(candidate_features)}"
    )

    print()

    print(
        "Candidate C additions:"
    )

    for feature in CANDIDATE_FEATURES:

        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # ROLLING BACKTEST
    # --------------------------------------------------------

    fold_rows = []

    oof_frames = []

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
                "Empty training set for "
                f"{validation_season}."
            )

        if validation.empty:

            raise RuntimeError(
                "Empty validation set for "
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
                f"for season {validation_season}."
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
            fit_and_evaluate(
                train=train,
                validation=validation,
                features=control_features,
            )
        )

        print(
            "Training CANDIDATE C1..."
        )

        candidate = (
            fit_and_evaluate(
                train=train,
                validation=validation,
                features=candidate_features,
            )
        )

        if not np.array_equal(
            control[
                "y_true"
            ],
            candidate[
                "y_true"
            ],
        ):

            raise RuntimeError(
                "Control and Candidate C "
                "targets differ."
            )

        row = {
            "validation_season": (
                validation_season
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
                    control[
                        "y_true"
                    ]
                ),

                "control_prediction": [
                    TARGET_NAMES[
                        int(
                            value
                        )
                    ]
                    for value
                    in control[
                        "predictions"
                    ]
                ],

                "control_prob_H": (
                    control[
                        "probabilities"
                    ][
                        :,
                        0
                    ]
                ),

                "control_prob_D": (
                    control[
                        "probabilities"
                    ][
                        :,
                        1
                    ]
                ),

                "control_prob_A": (
                    control[
                        "probabilities"
                    ][
                        :,
                        2
                    ]
                ),

                "candidate_prediction": [
                    TARGET_NAMES[
                        int(
                            value
                        )
                    ]
                    for value
                    in candidate[
                        "predictions"
                    ]
                ],

                "candidate_prob_H": (
                    candidate[
                        "probabilities"
                    ][
                        :,
                        0
                    ]
                ),

                "candidate_prob_D": (
                    candidate[
                        "probabilities"
                    ][
                        :,
                        1
                    ]
                ),

                "candidate_prob_A": (
                    candidate[
                        "probabilities"
                    ][
                        :,
                        2
                    ]
                ),
            }
        )

        for feature in CANDIDATE_FEATURES:

            prediction_frame[
                feature
            ] = validation[
                feature
            ].to_numpy()

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

        print(
            f"  ECE:      "
            f"{control['metrics']['ece']:.6f} "
            f"→ "
            f"{candidate['metrics']['ece']:.6f} "
            f"("
            f"{row['ece_improvement']:+.6f}"
            f")"
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
    ) != EXPECTED_DEVELOPMENT_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_DEVELOPMENT_MATCHES} "
            "OOF rows, found "
            f"{len(oof)}."
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

        improvements = (
            fold_metrics[
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

    ece_row = (
        summary[
            summary[
                "metric"
            ]
            ==
            "ece"
        ]
        .iloc[
            0
        ]
    )

    # --------------------------------------------------------
    # PRE-REGISTERED SIGNAL SCREEN
    # --------------------------------------------------------

    passes_signal_screen = bool(
        (
            log_loss_row[
                "candidate_improvement"
            ]
            >
            0
        )
        and
        (
            brier_row[
                "candidate_improvement"
            ]
            >=
            0
        )
        and
        (
            log_loss_row[
                "candidate_better_folds"
            ]
            >=
            4
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
            "candidate_c1_domestic_advanced_common_coverage"
        ),
        "status": (
            "research_only"
        ),
        "selection_window": (
            "2018_2024"
        ),
        "locked_confirmation_season": (
            LOCKED_CONFIRMATION_SEASON
        ),
        "confirmation_2025_used": False,
        "production_modified": False,

        "coverage_policy": (
            "both_teams_advanced_history_and_minimum_10_prior_matches"
        ),

        "eligible_historical_matches": int(
            len(
                eligible
            )
        ),

        "development_oof_matches": int(
            len(
                oof
            )
        ),

        "locked_2025_matches": int(
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
            CANDIDATE_FEATURES
        ),

        "advanced_stats_source": (
            "historical_big_five_domestic_league_files"
        ),

        "advanced_stats": [
            "shots",
            "shots_on_target",
            "corners",
        ],

        "rolling_windows": [
            5,
            10,
        ],

        "current_ucl_match_stats_used": False,
        "future_matches_used": False,
        "fuzzy_matching": False,
        "missing_candidate_values_used": False,
        "minimum_history_matches_per_team": 10,

        "locked_model": {
            "imputer": (
                "SimpleImputer(strategy=median, add_indicator=True)"
            ),
            "model": "CatBoostClassifier",
            "iterations": 700,
            "learning_rate": 0.025,
            "depth": 5,
            "loss_function": "MultiClass",
            "l2_leaf_reg": 5.0,
            "random_seed": 42,
        },

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
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "CANDIDATE C1 ROLLING SUMMARY"
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

    # --------------------------------------------------------
    # SIGNAL DECISION
    # --------------------------------------------------------

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

    print(
        f"Weighted ECE improvement: "
        f"{ece_row['candidate_improvement']:+.6f}"
    )

    print()

    if passes_signal_screen:

        print(
            "Decision: ADVANCED_PERFORMANCE_SIGNAL_DETECTED"
        )

        print()

        print(
            "Candidate C1 has sufficient development-window "
            "evidence to retain the advanced-performance "
            "feature family for further V2 research."
        )

        print()

        print(
            "2025/26 has NOT been evaluated."
        )

        print(
            "Production remains frozen V1."
        )

    else:

        print(
            "Decision: ADVANCED_PERFORMANCE_SIGNAL_NOT_ESTABLISHED"
        )

        print()

        print(
            "The complete 12-feature family does not "
            "currently justify progression."
        )

        print()

        print(
            "2025/26 remains unused."
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
        "Frozen V1 control features = 290: PASS"
    )

    print(
        "Candidate C features are new to V1: PASS"
    )

    print(
        "Control and candidate use identical rows: PASS"
    )

    print(
        "Both teams have >=10 prior domestic matches: PASS"
    )

    print(
        "Candidate C feature missingness used by model: NO"
    )

    print(
        "Current UCL match statistics used: NO"
    )

    print(
        "Future fixtures used: NO"
    )

    print(
        "2018-2024 selection only: PASS"
    )

    print(
        "2025/26 evaluated: NO"
    )

    print(
        "CatBoost hyperparameters tuned: NO"
    )

    print(
        "Production feature matrix modified: NO"
    )

    print(
        "Production model modified: NO"
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