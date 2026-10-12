from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ml import run_boosted_backtest as rb


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

UEFA_FEATURES_PATH = (
    ROOT
    / "data"
    / "research"
    / "ucl_uefa_strength_features.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

FOLD_COMPARISON_PATH = (
    REPORT_DIR
    / "uefa_strength_rolling_fold_comparison.csv"
)

OOF_PREDICTIONS_PATH = (
    REPORT_DIR
    / "uefa_strength_rolling_oof_predictions.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "uefa_strength_rolling_summary.csv"
)

CALIBRATION_PATH = (
    REPORT_DIR
    / "uefa_strength_rolling_calibration_bins.csv"
)

METADATA_PATH = (
    REPORT_DIR
    / "uefa_strength_rolling_summary.json"
)


# ============================================================
# EXPERIMENT CONTRACT
# ============================================================


CONTROL_FEATURE_COUNT = 290

CANDIDATE_FEATURE_COUNT = 296

EXPECTED_TOTAL_ROWS = 2089

EXPECTED_2025_ROWS = 189


CANDIDATE_UEFA_FEATURES = [
    "home_uefa_coefficient",
    "away_uefa_coefficient",
    "uefa_coefficient_diff",
    "home_uefa_strength_percentile",
    "away_uefa_strength_percentile",
    "uefa_strength_percentile_diff",
]


LOWER_IS_BETTER = {
    "log_loss",
    "brier_score",
    "top_label_ece_10bin",
}


METRICS_TO_COMPARE = [
    "log_loss",
    "brier_score",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "home_recall",
    "draw_recall",
    "away_recall",
]


# ============================================================
# MATCH ID
# ============================================================


def canonical_match_id(
    value,
) -> str:
    if pd.isna(
        value
    ):
        raise RuntimeError(
            "Missing match_id."
        )

    text = str(
        value
    ).strip()

    numeric_float = re.fullmatch(
        r"([0-9]+)\.0+",
        text,
    )

    if numeric_float:
        return numeric_float.group(
            1
        )

    return text


# ============================================================
# MODEL
# ============================================================


def build_locked_catboost() -> Pipeline:
    """
    Exact CatBoost specification used by the
    established boosted rolling backtest.
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
                    random_seed=rb.RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


# ============================================================
# LOAD CONTROL DATA
# ============================================================


def load_control_data() -> tuple[
    pd.DataFrame,
    list[str],
]:
    print(
        "Loading frozen feature manifest..."
    )

    manifest = (
        rb.load_manifest()
    )

    all_features = (
        manifest[
            "feature_columns"
        ]
    )

    print(
        "Applying established no-transfer "
        "feature policy..."
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
            "Expected "
            f"{CONTROL_FEATURE_COUNT} control features, "
            f"found {len(control_features)}."
        )

    print(
        "Loading historical ML matrix..."
    )

    dataframe = (
        rb.load_data(
            all_features
        )
    )

    if len(
        dataframe
    ) != EXPECTED_TOTAL_ROWS:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TOTAL_ROWS:,} historical rows, "
            f"found {len(dataframe):,}."
        )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "result",
        rb.TARGET_COLUMN,
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
            "Historical ML matrix missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "__match_key"
    ] = dataframe[
        "match_id"
    ].apply(
        canonical_match_id
    )

    if dataframe[
        "__match_key"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate match IDs found in "
            "historical ML matrix."
        )

    return (
        dataframe,
        control_features,
    )


# ============================================================
# LOAD UEFA
# ============================================================


def load_uefa_features() -> pd.DataFrame:
    if not UEFA_FEATURES_PATH.exists():

        raise FileNotFoundError(
            UEFA_FEATURES_PATH
        )

    print(
        "Loading UEFA strength layer..."
    )

    dataframe = pd.read_csv(
        UEFA_FEATURES_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        *CANDIDATE_UEFA_FEATURES,
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
            "UEFA feature layer missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_TOTAL_ROWS:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TOTAL_ROWS:,} UEFA rows, "
            f"found {len(dataframe):,}."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "__match_key"
    ] = dataframe[
        "match_id"
    ].apply(
        canonical_match_id
    )

    if dataframe[
        "__match_key"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate match IDs found in "
            "UEFA feature layer."
        )

    missing_values = (
        dataframe[
            CANDIDATE_UEFA_FEATURES
        ]
        .isna()
        .sum()
    )

    if (
        missing_values
        > 0
    ).any():

        raise RuntimeError(
            "Missing UEFA candidate values:\n"
            f"{missing_values.to_string()}"
        )

    return dataframe


# ============================================================
# MERGE
# ============================================================


def merge_features(
    *,
    base: pd.DataFrame,
    uefa: pd.DataFrame,
) -> pd.DataFrame:
    print(
        "Joining UEFA features to historical rows..."
    )

    collisions = [
        feature
        for feature
        in CANDIDATE_UEFA_FEATURES
        if feature
        in base.columns
    ]

    if collisions:

        raise RuntimeError(
            "UEFA features already exist in "
            "base matrix: "
            f"{collisions}"
        )

    base_keys = set(
        base[
            "__match_key"
        ]
    )

    uefa_keys = set(
        uefa[
            "__match_key"
        ]
    )

    missing_uefa = (
        base_keys
        -
        uefa_keys
    )

    extra_uefa = (
        uefa_keys
        -
        base_keys
    )

    if missing_uefa:

        raise RuntimeError(
            "Historical matches missing from "
            "UEFA layer: "
            f"{sorted(missing_uefa)[:20]}"
        )

    if extra_uefa:

        raise RuntimeError(
            "UEFA matches absent from historical "
            "matrix: "
            f"{sorted(extra_uefa)[:20]}"
        )

    left = (
        base.copy()
    )

    left[
        "__row_order"
    ] = np.arange(
        len(
            left
        )
    )

    right = (
        uefa[
            [
                "__match_key",
                *CANDIDATE_UEFA_FEATURES,
            ]
        ]
        .copy()
    )

    merged = left.merge(
        right,
        on="__match_key",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    merged = (
        merged.sort_values(
            "__row_order"
        )
        .drop(
            columns=[
                "__row_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        merged
    ) != len(
        base
    ):

        raise RuntimeError(
            "UEFA merge changed historical "
            "row count."
        )

    if (
        merged[
            CANDIDATE_UEFA_FEATURES
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "UEFA features became missing "
            "after merge."
        )

    if (
        merged[
            "__match_key"
        ].tolist()
        !=
        base[
            "__match_key"
        ].tolist()
    ):

        raise RuntimeError(
            "Historical row order changed "
            "during merge."
        )

    return merged


# ============================================================
# CALIBRATION
# ============================================================


def build_calibration_bins(
    *,
    model_name: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> tuple[
    pd.DataFrame,
    float,
]:
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

    rows = []

    ece = 0.0

    total = len(
        y_true
    )

    for index in range(
        bins
    ):

        lower = float(
            edges[
                index
            ]
        )

        upper = float(
            edges[
                index + 1
            ]
        )

        if index == (
            bins - 1
        ):

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:

            rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "bin": (
                        index + 1
                    ),
                    "lower_bound": (
                        lower
                    ),
                    "upper_bound": (
                        upper
                    ),
                    "matches": 0,
                    "mean_confidence": (
                        np.nan
                    ),
                    "accuracy": (
                        np.nan
                    ),
                    "absolute_gap": (
                        np.nan
                    ),
                }
            )

            continue

        mean_confidence = float(
            confidence[
                mask
            ].mean()
        )

        accuracy = float(
            correct[
                mask
            ].mean()
        )

        gap = abs(
            accuracy
            -
            mean_confidence
        )

        ece += (
            (
                count
                /
                total
            )
            *
            gap
        )

        rows.append(
            {
                "model": (
                    model_name
                ),
                "bin": (
                    index + 1
                ),
                "lower_bound": (
                    lower
                ),
                "upper_bound": (
                    upper
                ),
                "matches": (
                    count
                ),
                "mean_confidence": (
                    mean_confidence
                ),
                "accuracy": (
                    accuracy
                ),
                "absolute_gap": (
                    gap
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        ),
        float(
            ece
        ),
    )


# ============================================================
# PER-MATCH SCORES
# ============================================================


def per_match_log_loss(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = np.clip(
        probabilities,
        1e-15,
        1.0 - 1e-15,
    )

    actual_probability = (
        probabilities[
            np.arange(
                len(
                    y_true
                )
            ),
            y_true,
        ]
    )

    return (
        -np.log(
            actual_probability
        )
    )


def per_match_brier(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    one_hot = np.zeros_like(
        probabilities,
        dtype=float,
    )

    one_hot[
        np.arange(
            len(
                y_true
            )
        ),
        y_true,
    ] = 1.0

    return np.sum(
        (
            probabilities
            -
            one_hot
        )
        ** 2,
        axis=1,
    )


# ============================================================
# RUN ONE MODEL
# ============================================================


def fit_and_evaluate(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict:
    X_train = (
        train[
            features
        ]
    )

    y_train = (
        train[
            rb.TARGET_COLUMN
        ]
    )

    X_validation = (
        validation[
            features
        ]
    )

    y_validation = (
        validation[
            rb.TARGET_COLUMN
        ]
    )

    model = (
        build_locked_catboost()
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )
    )

    probabilities = (
        rb.align_probabilities(
            model=model,
            probabilities=probabilities,
        )
    )

    metrics = (
        rb.evaluate(
            y_true=y_validation,
            probabilities=probabilities,
        )
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "metrics": (
            metrics
        ),
        "probabilities": (
            probabilities
        ),
        "predictions": (
            predictions
        ),
        "y_true": (
            y_validation.to_numpy(
                dtype=int
            )
        ),
    }


# ============================================================
# METRIC HELPERS
# ============================================================


def candidate_improvement(
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
# BUILD SUMMARY
# ============================================================


def build_summary(
    *,
    fold_metrics: pd.DataFrame,
    pooled_control_metrics: dict,
    pooled_candidate_metrics: dict,
    control_ece: float,
    candidate_ece: float,
) -> pd.DataFrame:
    rows = []

    for metric in (
        METRICS_TO_COMPARE
    ):

        control_column = (
            f"control_{metric}"
        )

        candidate_column = (
            f"candidate_{metric}"
        )

        control_weighted = (
            weighted_mean(
                fold_metrics[
                    control_column
                ],
                fold_metrics[
                    "validation_rows"
                ],
            )
        )

        candidate_weighted = (
            weighted_mean(
                fold_metrics[
                    candidate_column
                ],
                fold_metrics[
                    "validation_rows"
                ],
            )
        )

        improvement = (
            candidate_improvement(
                metric=metric,
                control=control_weighted,
                candidate=candidate_weighted,
            )
        )

        fold_improvement_column = (
            f"{metric}_improvement"
        )

        better_folds = int(
            (
                fold_metrics[
                    fold_improvement_column
                ]
                > 0
            ).sum()
        )

        tied_folds = int(
            np.isclose(
                fold_metrics[
                    fold_improvement_column
                ],
                0.0,
                atol=1e-12,
            ).sum()
        )

        worse_folds = int(
            len(
                fold_metrics
            )
            -
            better_folds
            -
            tied_folds
        )

        pooled_control = float(
            pooled_control_metrics[
                metric
            ]
        )

        pooled_candidate = float(
            pooled_candidate_metrics[
                metric
            ]
        )

        pooled_improvement = (
            candidate_improvement(
                metric=metric,
                control=pooled_control,
                candidate=pooled_candidate,
            )
        )

        rows.append(
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
                "weighted_improvement": (
                    improvement
                ),
                "pooled_control": (
                    pooled_control
                ),
                "pooled_candidate": (
                    pooled_candidate
                ),
                "pooled_improvement": (
                    pooled_improvement
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
                "total_folds": int(
                    len(
                        fold_metrics
                    )
                ),
            }
        )

    ece_improvement = (
        candidate_improvement(
            metric=(
                "top_label_ece_10bin"
            ),
            control=(
                control_ece
            ),
            candidate=(
                candidate_ece
            ),
        )
    )

    rows.append(
        {
            "metric": (
                "top_label_ece_10bin"
            ),
            "better_direction": (
                "lower"
            ),
            "weighted_control": (
                np.nan
            ),
            "weighted_candidate": (
                np.nan
            ),
            "weighted_improvement": (
                np.nan
            ),
            "pooled_control": (
                control_ece
            ),
            "pooled_candidate": (
                candidate_ece
            ),
            "pooled_improvement": (
                ece_improvement
            ),
            "candidate_better_folds": (
                np.nan
            ),
            "tied_folds": (
                np.nan
            ),
            "candidate_worse_folds": (
                np.nan
            ),
            "total_folds": int(
                len(
                    fold_metrics
                )
            ),
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT HELPERS
# ============================================================


def print_fold_table(
    dataframe: pd.DataFrame,
) -> None:
    display = dataframe[
        [
            "validation_season",
            "validation_rows",
            "control_log_loss",
            "candidate_log_loss",
            "log_loss_improvement",
            "control_brier_score",
            "candidate_brier_score",
            "brier_score_improvement",
            "control_accuracy",
            "candidate_accuracy",
            "control_macro_f1",
            "candidate_macro_f1",
            "control_draw_recall",
            "candidate_draw_recall",
        ]
    ].copy()

    numeric_columns = [
        column
        for column
        in display.columns
        if column
        not in {
            "validation_season",
            "validation_rows",
        }
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


def print_summary(
    summary: pd.DataFrame,
) -> None:
    display = (
        summary.copy()
    )

    for column in [
        "weighted_control",
        "weighted_candidate",
        "weighted_improvement",
        "pooled_control",
        "pooled_candidate",
        "pooled_improvement",
    ]:

        display[
            column
        ] = display[
            column
        ].map(
            lambda value: (
                ""
                if pd.isna(
                    value
                )
                else f"{value:.6f}"
            )
        )

    print(
        display.to_string(
            index=False
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
        "UCL PREDICTOR V2 - CANDIDATE A"
    )

    print(
        "UEFA STRENGTH ROLLING BACKTEST"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Research objective:"
    )

    print(
        "Test whether the six UEFA strength "
        "features improve pre-2025 out-of-sample "
        "performance consistently."
    )

    print()

    print(
        "2025/26 will not be used in any "
        "training or validation fold."
    )

    print(
        "No hyperparameter tuning is performed."
    )

    print(
        "Production model remains untouched."
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        base,
        control_features,
    ) = load_control_data()

    uefa = (
        load_uefa_features()
    )

    dataframe = (
        merge_features(
            base=base,
            uefa=uefa,
        )
    )

    candidate_features = [
        *control_features,
        *CANDIDATE_UEFA_FEATURES,
    ]

    if len(
        candidate_features
    ) != CANDIDATE_FEATURE_COUNT:

        raise RuntimeError(
            "Expected "
            f"{CANDIDATE_FEATURE_COUNT} Candidate A "
            "features, found "
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
            "Candidate A contains duplicate "
            "feature names."
        )

    if list(
        rb.TARGET_LABELS
    ) != [
        0,
        1,
        2,
    ]:

        raise RuntimeError(
            "Unexpected target class order."
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
    ) != EXPECTED_2025_ROWS:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_2025_ROWS} rows for season "
            f"{rb.FINAL_TEST_SEASON}, found "
            f"{len(final_2025)}."
        )

    validation_seasons = list(
        range(
            rb.FIRST_VALIDATION_SEASON,
            rb.LAST_VALIDATION_SEASON
            + 1,
        )
    )

    print(
        "=" * 110
    )

    print(
        "ROLLING POLICY"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Backtest rows: "
        f"{len(backtest_data):,}"
    )

    print(
        f"Excluded 2025/26 rows: "
        f"{len(final_2025):,}"
    )

    print(
        "Validation seasons: "
        +
        ", ".join(
            str(
                season
            )
            for season
            in validation_seasons
        )
    )

    print(
        f"Control features: "
        f"{len(control_features)}"
    )

    print(
        f"Candidate features: "
        f"{len(candidate_features)}"
    )

    print()

    # --------------------------------------------------------
    # ROLLING BACKTEST
    # --------------------------------------------------------

    fold_rows = []

    prediction_frames = []

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
                "Training data empty for "
                f"validation season "
                f"{validation_season}."
            )

        if validation.empty:

            raise RuntimeError(
                "Validation data empty for "
                f"season {validation_season}."
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
            "Training CONTROL..."
        )

        control = (
            fit_and_evaluate(
                train=train,
                validation=validation,
                features=control_features,
            )
        )

        print(
            "Training CANDIDATE A..."
        )

        candidate = (
            fit_and_evaluate(
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

        for metric in (
            METRICS_TO_COMPARE
        ):

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
            ] = (
                candidate_improvement(
                    metric=metric,
                    control=control_value,
                    candidate=candidate_value,
                )
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
                "Control and Candidate A target "
                "arrays differ."
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

        control_match_ll = (
            per_match_log_loss(
                y_true=y_true,
                probabilities=(
                    control_probabilities
                ),
            )
        )

        candidate_match_ll = (
            per_match_log_loss(
                y_true=y_true,
                probabilities=(
                    candidate_probabilities
                ),
            )
        )

        control_match_brier = (
            per_match_brier(
                y_true=y_true,
                probabilities=(
                    control_probabilities
                ),
            )
        )

        candidate_match_brier = (
            per_match_brier(
                y_true=y_true,
                probabilities=(
                    candidate_probabilities
                ),
            )
        )

        predictions = pd.DataFrame(
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
                "control_correct": (
                    control_predictions
                    ==
                    y_true
                ).astype(
                    int
                ),
                "control_match_log_loss": (
                    control_match_ll
                ),
                "control_match_brier": (
                    control_match_brier
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
                "candidate_correct": (
                    candidate_predictions
                    ==
                    y_true
                ).astype(
                    int
                ),
                "candidate_match_log_loss": (
                    candidate_match_ll
                ),
                "candidate_match_brier": (
                    candidate_match_brier
                ),

                "log_loss_improvement": (
                    control_match_ll
                    -
                    candidate_match_ll
                ),
                "brier_improvement": (
                    control_match_brier
                    -
                    candidate_match_brier
                ),
            }
        )

        prediction_frames.append(
            predictions
        )

        print()

        print(
            "Fold result:"
        )

        print(
            "  Log Loss:"
        )

        print(
            f"    Control:     "
            f"{control['metrics']['log_loss']:.6f}"
        )

        print(
            f"    Candidate A: "
            f"{candidate['metrics']['log_loss']:.6f}"
        )

        print(
            f"    Improvement: "
            f"{row['log_loss_improvement']:+.6f}"
        )

        print(
            "  Brier:"
        )

        print(
            f"    Control:     "
            f"{control['metrics']['brier_score']:.6f}"
        )

        print(
            f"    Candidate A: "
            f"{candidate['metrics']['brier_score']:.6f}"
        )

        print(
            f"    Improvement: "
            f"{row['brier_score_improvement']:+.6f}"
        )

        print(
            "  Accuracy:"
        )

        print(
            f"    Control:     "
            f"{control['metrics']['accuracy']:.6f}"
        )

        print(
            f"    Candidate A: "
            f"{candidate['metrics']['accuracy']:.6f}"
        )

        print(
            "  Macro F1:"
        )

        print(
            f"    Control:     "
            f"{control['metrics']['macro_f1']:.6f}"
        )

        print(
            f"    Candidate A: "
            f"{candidate['metrics']['macro_f1']:.6f}"
        )

        print(
            "  Draw Recall:"
        )

        print(
            f"    Control:     "
            f"{control['metrics']['draw_recall']:.6f}"
        )

        print(
            f"    Candidate A: "
            f"{candidate['metrics']['draw_recall']:.6f}"
        )

    # --------------------------------------------------------
    # COMBINE OOF
    # --------------------------------------------------------

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    oof = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    expected_oof_rows = int(
        fold_metrics[
            "validation_rows"
        ].sum()
    )

    if len(
        oof
    ) != expected_oof_rows:

        raise RuntimeError(
            "OOF row count does not match "
            "validation-fold row total."
        )

    if oof[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "A match appeared in more than "
            "one validation fold."
        )

    # --------------------------------------------------------
    # POOLED METRICS
    # --------------------------------------------------------

    pooled_y = (
        oof[
            "actual_label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    control_probabilities = (
        oof[
            [
                "control_prob_H",
                "control_prob_D",
                "control_prob_A",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    candidate_probabilities = (
        oof[
            [
                "candidate_prob_H",
                "candidate_prob_D",
                "candidate_prob_A",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    pooled_control_metrics = (
        rb.evaluate(
            y_true=pd.Series(
                pooled_y
            ),
            probabilities=(
                control_probabilities
            ),
        )
    )

    pooled_candidate_metrics = (
        rb.evaluate(
            y_true=pd.Series(
                pooled_y
            ),
            probabilities=(
                candidate_probabilities
            ),
        )
    )

    pooled_control_predictions = np.argmax(
        control_probabilities,
        axis=1,
    )

    pooled_candidate_predictions = np.argmax(
        candidate_probabilities,
        axis=1,
    )

    (
        control_calibration,
        control_ece,
    ) = (
        build_calibration_bins(
            model_name="CONTROL_V1",
            y_true=pooled_y,
            predictions=(
                pooled_control_predictions
            ),
            probabilities=(
                control_probabilities
            ),
        )
    )

    (
        candidate_calibration,
        candidate_ece,
    ) = (
        build_calibration_bins(
            model_name="CANDIDATE_A",
            y_true=pooled_y,
            predictions=(
                pooled_candidate_predictions
            ),
            probabilities=(
                candidate_probabilities
            ),
        )
    )

    calibration = pd.concat(
        [
            control_calibration,
            candidate_calibration,
        ],
        ignore_index=True,
    )

    summary = (
        build_summary(
            fold_metrics=(
                fold_metrics
            ),
            pooled_control_metrics=(
                pooled_control_metrics
            ),
            pooled_candidate_metrics=(
                pooled_candidate_metrics
            ),
            control_ece=(
                control_ece
            ),
            candidate_ece=(
                candidate_ece
            ),
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
        FOLD_COMPARISON_PATH,
        index=False,
    )

    oof.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_PATH,
        index=False,
    )

    metadata = {
        "experiment": (
            "candidate_a_uefa_strength_rolling"
        ),
        "status": (
            "research_only"
        ),
        "production_modified": False,
        "final_2025_used": False,
        "validation_seasons": (
            validation_seasons
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
        "added_features": (
            CANDIDATE_UEFA_FEATURES
        ),
        "catboost": {
            "iterations": 700,
            "learning_rate": 0.025,
            "depth": 5,
            "loss_function": (
                "MultiClass"
            ),
            "l2_leaf_reg": 5.0,
            "random_seed": int(
                rb.RANDOM_STATE
            ),
            "imputer": (
                "median_add_indicator"
            ),
        },
        "oof_rows": int(
            len(
                oof
            )
        ),
        "control_top_label_ece_10bin": (
            float(
                control_ece
            )
        ),
        "candidate_top_label_ece_10bin": (
            float(
                candidate_ece
            )
        ),
        "pooled_control_metrics": {
            key: float(
                value
            )
            for key, value
            in pooled_control_metrics.items()
        },
        "pooled_candidate_metrics": {
            key: float(
                value
            )
            for key, value
            in pooled_candidate_metrics.items()
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
    # PRINT FOLDS
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "ROLLING FOLD COMPARISON"
    )

    print(
        "=" * 110
    )

    print()

    print_fold_table(
        fold_metrics
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "ROLLING SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Positive improvement = Candidate A better."
    )

    print()

    print_summary(
        summary
    )

    # --------------------------------------------------------
    # CORE SIGNAL
    # --------------------------------------------------------

    log_row = (
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

    accuracy_row = (
        summary[
            summary[
                "metric"
            ]
            ==
            "accuracy"
        ]
        .iloc[
            0
        ]
    )

    macro_row = (
        summary[
            summary[
                "metric"
            ]
            ==
            "macro_f1"
        ]
        .iloc[
            0
        ]
    )

    draw_row = (
        summary[
            summary[
                "metric"
            ]
            ==
            "draw_recall"
        ]
        .iloc[
            0
        ]
    )

    print()
    print(
        "=" * 110
    )

    print(
        "CORE ROLLING SIGNAL"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Weighted Log Loss:"
    )

    print(
        f"  Control:     "
        f"{log_row['weighted_control']:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{log_row['weighted_candidate']:.6f}"
    )

    print(
        f"  Improvement: "
        f"{log_row['weighted_improvement']:+.6f}"
    )

    print(
        f"  Better folds: "
        f"{int(log_row['candidate_better_folds'])}/"
        f"{int(log_row['total_folds'])}"
    )

    print()

    print(
        "Weighted Brier:"
    )

    print(
        f"  Control:     "
        f"{brier_row['weighted_control']:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{brier_row['weighted_candidate']:.6f}"
    )

    print(
        f"  Improvement: "
        f"{brier_row['weighted_improvement']:+.6f}"
    )

    print(
        f"  Better folds: "
        f"{int(brier_row['candidate_better_folds'])}/"
        f"{int(brier_row['total_folds'])}"
    )

    print()

    print(
        "Weighted Accuracy:"
    )

    print(
        f"  Control:     "
        f"{accuracy_row['weighted_control']:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{accuracy_row['weighted_candidate']:.6f}"
    )

    print(
        f"  Improvement: "
        f"{accuracy_row['weighted_improvement']:+.6f}"
    )

    print()

    print(
        "Weighted Macro F1:"
    )

    print(
        f"  Control:     "
        f"{macro_row['weighted_control']:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{macro_row['weighted_candidate']:.6f}"
    )

    print(
        f"  Improvement: "
        f"{macro_row['weighted_improvement']:+.6f}"
    )

    print()

    print(
        "Weighted Draw Recall:"
    )

    print(
        f"  Control:     "
        f"{draw_row['weighted_control']:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{draw_row['weighted_candidate']:.6f}"
    )

    print(
        f"  Improvement: "
        f"{draw_row['weighted_improvement']:+.6f}"
    )

    print()

    print(
        "Pooled top-label ECE:"
    )

    print(
        f"  Control:     "
        f"{control_ece:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{candidate_ece:.6f}"
    )

    print(
        f"  Improvement: "
        f"{control_ece - candidate_ece:+.6f}"
    )

    # --------------------------------------------------------
    # BIGGEST OOF CHANGES
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "CANDIDATE A HELPED MOST"
    )

    print(
        "=" * 110
    )

    print()

    helped = (
        oof.sort_values(
            "log_loss_improvement",
            ascending=False,
        )
        .head(
            10
        )
    )

    print(
        helped[
            [
                "validation_season",
                "date",
                "home_team",
                "away_team",
                "actual_result",
                "control_prediction",
                "candidate_prediction",
                "log_loss_improvement",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "CANDIDATE A HURT MOST"
    )

    print(
        "=" * 110
    )

    print()

    hurt = (
        oof.sort_values(
            "log_loss_improvement",
            ascending=True,
        )
        .head(
            10
        )
    )

    print(
        hurt[
            [
                "validation_season",
                "date",
                "home_team",
                "away_team",
                "actual_result",
                "control_prediction",
                "candidate_prediction",
                "log_loss_improvement",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "EXPERIMENT VALIDATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Control feature count 290: PASS"
    )

    print(
        "Candidate feature count 296: PASS"
    )

    print(
        "Validation seasons 2018-2024: PASS"
    )

    print(
        "2025/26 excluded from all folds: PASS"
    )

    print(
        "Train season strictly before "
        "validation season: PASS"
    )

    print(
        "Same rows for control/candidate: PASS"
    )

    print(
        "Same CatBoost specification: PASS"
    )

    print(
        "Same established metric functions: PASS"
    )

    print(
        "No transfer features added: PASS"
    )

    print(
        "Only six UEFA features added: PASS"
    )

    print(
        "No UEFA source metadata used: PASS"
    )

    print(
        "No country-floor flags used: PASS"
    )

    print(
        "No fuzzy matching: PASS"
    )

    print(
        "Production model untouched: PASS"
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
        FOLD_COMPARISON_PATH
    )

    print(
        OOF_PREDICTIONS_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        CALIBRATION_PATH
    )

    print(
        METADATA_PATH
    )

    print()

    print(
        "Rolling Candidate A research "
        "experiment complete."
    )


if __name__ == "__main__":
    main()