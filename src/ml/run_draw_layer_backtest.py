from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_METADATA_DIR,
    REPORTS_DIR,
)


# ============================================================
# PATHS
# ============================================================

FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "ml_features.csv"
)

FEATURE_MANIFEST_FILE = (
    MODEL_METADATA_DIR
    / "ml_feature_manifest.json"
)

BOOSTED_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "model_reports"
    / "boosted_rolling_backtest_predictions.csv"
)

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)

MODEL_REPORT_DIR = (
    REPORTS_DIR
    / "model_reports"
)

DRAW_MODEL_FOLD_METRICS_FILE = (
    METRICS_DIR
    / "draw_layer_fold_metrics.csv"
)

DRAW_MODEL_SUMMARY_FILE = (
    METRICS_DIR
    / "draw_layer_summary.csv"
)

DRAW_FUSION_SEARCH_FILE = (
    METRICS_DIR
    / "draw_fusion_search.csv"
)

DRAW_META_BACKTEST_FILE = (
    METRICS_DIR
    / "draw_fusion_meta_backtest.csv"
)

DRAW_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "draw_layer_oof_predictions.csv"
)

DRAW_METADATA_FILE = (
    MODEL_METADATA_DIR
    / "draw_layer_results.json"
)


# ============================================================
# TEMPORAL POLICY
# ============================================================

FIRST_VALIDATION_SEASON = 2018

LAST_VALIDATION_SEASON = 2024

FINAL_TEST_SEASON = 2025


# ============================================================
# TARGETS
# ============================================================

TARGET_COLUMN = "target"

DRAW_TARGET_COLUMN = "is_draw"

TARGET_LABELS = [
    0,
    1,
    2,
]


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

EPSILON = 1e-12

ALPHA_STEP = 0.05


# ============================================================
# DRAW MODEL CONFIGURATIONS
# ============================================================

DRAW_MODEL_CONFIGS = {
    "draw_catboost_plain": {
        "class_weights": None,
    },

    "draw_catboost_1_5x": {
        "class_weights": [
            1.0,
            1.5,
        ],
    },

    "draw_catboost_2x": {
        "class_weights": [
            1.0,
            2.0,
        ],
    },

    "draw_catboost_2_5x": {
        "class_weights": [
            1.0,
            2.5,
        ],
    },

    "draw_catboost_3x": {
        "class_weights": [
            1.0,
            3.0,
        ],
    },
}


# ============================================================
# HELPERS
# ============================================================


def require_file(
    path,
    description: str,
) -> None:
    """
    Fail early if a required file does not exist.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )

    if path.stat().st_size == 0:

        raise RuntimeError(
            f"{description} is empty:\n"
            f"{path}"
        )


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Normalize multiclass probabilities.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        None,
    )

    row_sums = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    if (
        row_sums
        <=
        0
    ).any():

        raise RuntimeError(
            "Invalid zero-sum probability row."
        )

    normalized = (
        probabilities
        /
        row_sums
    )

    if not np.allclose(
        normalized.sum(
            axis=1
        ),
        1.0,
        atol=1e-10,
    ):

        raise RuntimeError(
            "Probability normalization failed."
        )

    return normalized


def determine_feature_family(
    feature: str,
) -> str:
    """
    Existing project feature-family rules.
    """

    lowered = feature.lower()

    if "transfer" in lowered:

        return "transfer"

    if "squad" in lowered:

        return "squad"

    if (
        "domestic" in lowered
        or
        "known_match" in lowered
    ):

        return "domestic"

    if "elo" in lowered:

        return "elo"

    return "ucl_form"


# ============================================================
# LOAD MANIFEST
# ============================================================


def load_manifest() -> dict:
    """
    Load approved feature manifest.
    """

    require_file(
        FEATURE_MANIFEST_FILE,
        "ML feature manifest",
    )

    with FEATURE_MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(
            file
        )

    features = manifest.get(
        "feature_columns"
    )

    if not features:

        raise RuntimeError(
            "Feature manifest contains no features."
        )

    return manifest


# ============================================================
# PRIMARY FEATURE SET
# ============================================================


def build_primary_features(
    all_features: list[str],
) -> list[str]:
    """
    Match current primary CatBoost feature policy:

        exclude transfer features
    """

    features = [
        feature
        for feature in all_features
        if determine_feature_family(
            feature
        )
        !=
        "transfer"
    ]

    if len(
        features
    ) != 290:

        print(
            "\nWARNING:"
        )

        print(
            f"Expected 290 primary features, "
            f"found {len(features)}."
        )

    return features


# ============================================================
# LOAD ML DATA
# ============================================================


def load_ml_data(
    all_features: list[str],
) -> pd.DataFrame:
    """
    Load chronological ML matrix.
    """

    require_file(
        FEATURE_FILE,
        "ML feature matrix",
    )

    dataframe = pd.read_csv(
        FEATURE_FILE,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "result",
        TARGET_COLUMN,
    }

    required.update(
        all_features
    )

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"ML dataset missing columns: "
            f"{sorted(missing)}"
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
        TARGET_COLUMN
    ] = pd.to_numeric(
        dataframe[
            TARGET_COLUMN
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        DRAW_TARGET_COLUMN
    ] = (
        dataframe[
            TARGET_COLUMN
        ]
        ==
        1
    ).astype(
        int
    )

    dataframe = dataframe.sort_values(
        by=[
            "date",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    return dataframe


# ============================================================
# LOAD BASE CATBOOST OOF
# ============================================================


def load_base_catboost_oof() -> pd.DataFrame:
    """
    Load existing three-way CatBoost OOF predictions.
    """

    require_file(
        BOOSTED_PREDICTIONS_FILE,
        "Boosted rolling predictions",
    )

    dataframe = pd.read_csv(
        BOOSTED_PREDICTIONS_FILE,
        low_memory=False,
    )

    required = {
        "validation_season",
        "model",
        "match_id",
        "date",
        "home_team",
        "away_team",
        "actual_result",
        "prob_H",
        "prob_D",
        "prob_A",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Boosted prediction file "
            f"missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe[
        dataframe[
            "model"
        ]
        ==
        "catboost"
    ].copy()

    if len(
        dataframe
    ) != 924:

        raise RuntimeError(
            f"Expected 924 CatBoost OOF rows, "
            f"found {len(dataframe):,}."
        )

    probabilities = normalize_probabilities(
        dataframe[
            [
                "prob_H",
                "prob_D",
                "prob_A",
            ]
        ].to_numpy(
            dtype=float
        )
    )

    dataframe[
        "base_prob_H"
    ] = probabilities[
        :,
        0
    ]

    dataframe[
        "base_prob_D"
    ] = probabilities[
        :,
        1
    ]

    dataframe[
        "base_prob_A"
    ] = probabilities[
        :,
        2
    ]

    dataframe = dataframe[
        [
            "validation_season",
            "match_id",
            "base_prob_H",
            "base_prob_D",
            "base_prob_A",
        ]
    ]

    return dataframe


# ============================================================
# FOLD
# ============================================================


def build_fold(
    *,
    dataframe: pd.DataFrame,
    validation_season: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Expanding chronological fold.
    """

    train = dataframe[
        dataframe[
            "season"
        ]
        <
        validation_season
    ].copy()

    validation = dataframe[
        dataframe[
            "season"
        ]
        ==
        validation_season
    ].copy()

    if train.empty:

        raise RuntimeError(
            f"Training fold empty for "
            f"{validation_season}."
        )

    if validation.empty:

        raise RuntimeError(
            f"Validation fold empty for "
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
            "Chronological overlap detected."
        )

    return (
        train,
        validation,
    )


# ============================================================
# DRAW MODEL
# ============================================================


def build_draw_model(
    *,
    class_weights,
) -> Pipeline:
    """
    Binary CatBoost draw detector.
    """

    model_args = {
        "iterations": 700,

        "learning_rate": 0.025,

        "depth": 5,

        "loss_function": (
            "Logloss"
        ),

        "eval_metric": (
            "Logloss"
        ),

        "l2_leaf_reg": 5.0,

        "random_seed": (
            RANDOM_STATE
        ),

        "verbose": False,

        "allow_writing_files": False,
    }

    if class_weights is not None:

        model_args[
            "class_weights"
        ] = class_weights

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
                    **model_args
                ),
            ),
        ]
    )


# ============================================================
# BINARY DRAW METRICS
# ============================================================


def evaluate_draw_model(
    *,
    y_true: np.ndarray,
    draw_probabilities: np.ndarray,
) -> dict:
    """
    Evaluate raw binary draw discrimination.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        draw_probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0
        -
        EPSILON,
    )

    predictions = (
        probabilities
        >=
        0.5
    ).astype(
        int
    )

    binary_probability_matrix = np.column_stack(
        [
            1.0
            -
            probabilities,
            probabilities,
        ]
    )

    try:

        roc_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    except ValueError:

        roc_auc = np.nan

    try:

        average_precision = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    except ValueError:

        average_precision = np.nan

    return {
        "binary_log_loss": float(
            log_loss(
                y_true,
                binary_probability_matrix,
                labels=[
                    0,
                    1,
                ],
            )
        ),

        "roc_auc": (
            roc_auc
        ),

        "average_precision": (
            average_precision
        ),

        "predicted_draws_at_0_5": int(
            predictions.sum()
        ),
    }


# ============================================================
# RUN DRAW ROLLING BACKTEST
# ============================================================


def run_draw_oof(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Produce rolling OOF draw probabilities for all candidate
    binary CatBoost configurations.
    """

    fold_metric_rows = []

    prediction_rows = []

    validation_seasons = list(
        range(
            FIRST_VALIDATION_SEASON,
            LAST_VALIDATION_SEASON
            +
            1,
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ROLLING BINARY DRAW MODELS"
    )

    print(
        "=" * 75
    )

    for validation_season in validation_seasons:

        (
            train,
            validation,
        ) = build_fold(
            dataframe=dataframe,
            validation_season=validation_season,
        )

        print(
            "\n"
            + "-" * 75
        )

        print(
            f"VALIDATION SEASON "
            f"{validation_season}"
        )

        print(
            "-" * 75
        )

        print(
            f"Training rows: "
            f"{len(train):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        print(
            f"Training draw rate: "
            f"{train[DRAW_TARGET_COLUMN].mean():.4f}"
        )

        print(
            f"Validation draw rate: "
            f"{validation[DRAW_TARGET_COLUMN].mean():.4f}"
        )

        X_train = train[
            feature_columns
        ]

        y_train = train[
            DRAW_TARGET_COLUMN
        ]

        X_validation = validation[
            feature_columns
        ]

        y_validation = validation[
            DRAW_TARGET_COLUMN
        ]

        for (
            model_name,
            config,
        ) in DRAW_MODEL_CONFIGS.items():

            print(
                f"\n  {model_name}"
            )

            model = build_draw_model(
                class_weights=config[
                    "class_weights"
                ]
            )

            model.fit(
                X_train,
                y_train,
            )

            probabilities = model.predict_proba(
                X_validation
            )

            classes = np.asarray(
                model.classes_,
                dtype=int,
            )

            positive_indices = np.where(
                classes
                ==
                1
            )[0]

            if len(
                positive_indices
            ) != 1:

                raise RuntimeError(
                    "Could not identify draw probability "
                    "column."
                )

            positive_index = int(
                positive_indices[
                    0
                ]
            )

            draw_probabilities = probabilities[
                :,
                positive_index
            ]

            metrics = evaluate_draw_model(
                y_true=y_validation.to_numpy(
                    dtype=int
                ),
                draw_probabilities=draw_probabilities,
            )

            fold_metric_rows.append(
                {
                    "validation_season": (
                        validation_season
                    ),

                    "model": (
                        model_name
                    ),

                    "training_rows": len(
                        train
                    ),

                    "validation_rows": len(
                        validation
                    ),

                    "training_draw_rate": float(
                        y_train.mean()
                    ),

                    "validation_draw_rate": float(
                        y_validation.mean()
                    ),

                    **metrics,
                }
            )

            for row_index, (
                _,
                match,
            ) in enumerate(
                validation.iterrows()
            ):

                prediction_rows.append(
                    {
                        "validation_season": (
                            validation_season
                        ),

                        "model": (
                            model_name
                        ),

                        "match_id": (
                            match[
                                "match_id"
                            ]
                        ),

                        "date": (
                            match[
                                "date"
                            ]
                        ),

                        "home_team": (
                            match[
                                "home_team"
                            ]
                        ),

                        "away_team": (
                            match[
                                "away_team"
                            ]
                        ),

                        "actual_result": (
                            match[
                                "result"
                            ]
                        ),

                        "target": (
                            match[
                                TARGET_COLUMN
                            ]
                        ),

                        "is_draw": (
                            match[
                                DRAW_TARGET_COLUMN
                            ]
                        ),

                        "draw_probability": float(
                            draw_probabilities[
                                row_index
                            ]
                        ),
                    }
                )

            print(
                f"    Binary Log Loss: "
                f"{metrics['binary_log_loss']:.4f}"
            )

            print(
                f"    ROC AUC:         "
                f"{metrics['roc_auc']:.4f}"
            )

            print(
                f"    Avg Precision:   "
                f"{metrics['average_precision']:.4f}"
            )

    return (
        pd.DataFrame(
            fold_metric_rows
        ),
        pd.DataFrame(
            prediction_rows
        ),
    )


# ============================================================
# DRAW MODEL SUMMARY
# ============================================================


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Validation-row weighted mean.
    """

    return float(
        np.average(
            values.astype(
                float
            ),
            weights=weights.astype(
                float
            ),
        )
    )


def summarize_draw_models(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate draw-model quality.
    """

    rows = []

    for (
        model_name,
        group,
    ) in fold_metrics.groupby(
        "model"
    ):

        weights = group[
            "validation_rows"
        ]

        rows.append(
            {
                "model": (
                    model_name
                ),

                "folds": int(
                    len(
                        group
                    )
                ),

                "validation_matches": int(
                    weights.sum()
                ),

                "weighted_binary_log_loss": (
                    weighted_mean(
                        group[
                            "binary_log_loss"
                        ],
                        weights,
                    )
                ),

                "weighted_roc_auc": (
                    weighted_mean(
                        group[
                            "roc_auc"
                        ],
                        weights,
                    )
                ),

                "weighted_average_precision": (
                    weighted_mean(
                        group[
                            "average_precision"
                        ],
                        weights,
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        by=[
            "weighted_binary_log_loss",
            "weighted_roc_auc",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    summary[
        "rank"
    ] = range(
        1,
        len(
            summary
        )
        +
        1,
    )

    return summary


# ============================================================
# MERGE WITH BASE CATBOOST
# ============================================================


def merge_with_base_predictions(
    *,
    draw_predictions: pd.DataFrame,
    base_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge binary draw probabilities with existing
    three-class CatBoost OOF probabilities.
    """

    merged = draw_predictions.merge(
        base_predictions,
        on=[
            "validation_season",
            "match_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    expected_rows = (
        924
        *
        len(
            DRAW_MODEL_CONFIGS
        )
    )

    if len(
        merged
    ) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows:,} merged rows, "
            f"found {len(merged):,}."
        )

    return merged


# ============================================================
# MULTICLASS BRIER
# ============================================================


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Multiclass Brier score.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

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

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    -
                    one_hot
                )
                ** 2,
                axis=1,
            )
        )
    )


# ============================================================
# DRAW FUSION
# ============================================================


def fuse_draw_probability(
    *,
    base_probabilities: np.ndarray,
    draw_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Blend base CatBoost draw probability with dedicated
    binary draw probability.

    alpha = 0:
        pure three-class CatBoost

    alpha = 1:
        dedicated binary draw probability fully controls P(D)

    H/A share the remaining probability using CatBoost's
    original H:A ratio.
    """

    if not (
        0.0
        <=
        alpha
        <=
        1.0
    ):

        raise ValueError(
            "alpha must be within [0,1]."
        )

    base = normalize_probabilities(
        base_probabilities
    )

    draw_probabilities = np.asarray(
        draw_probabilities,
        dtype=float,
    )

    draw_probabilities = np.clip(
        draw_probabilities,
        EPSILON,
        1.0
        -
        EPSILON,
    )

    new_draw = (
        (
            1.0
            -
            alpha
        )
        *
        base[
            :,
            1
        ]
        +
        alpha
        *
        draw_probabilities
    )

    non_draw_mass = (
        1.0
        -
        new_draw
    )

    home_away_sum = (
        base[
            :,
            0
        ]
        +
        base[
            :,
            2
        ]
    )

    home_ratio = np.divide(
        base[
            :,
            0
        ],
        home_away_sum,
        out=np.full(
            len(
                base
            ),
            0.5,
            dtype=float,
        ),
        where=(
            home_away_sum
            >
            EPSILON
        ),
    )

    away_ratio = (
        1.0
        -
        home_ratio
    )

    fused = np.column_stack(
        [
            non_draw_mass
            *
            home_ratio,

            new_draw,

            non_draw_mass
            *
            away_ratio,
        ]
    )

    return normalize_probabilities(
        fused
    )


# ============================================================
# MULTICLASS EVALUATION
# ============================================================


def evaluate_multiclass(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Evaluate fused H/D/A probabilities.
    """

    probabilities = normalize_probabilities(
        probabilities
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    (
        precision,
        recall,
        class_f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=TARGET_LABELS,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=TARGET_LABELS,
            )
        ),

        "brier_score": (
            multiclass_brier_score(
                y_true,
                probabilities,
            )
        ),

        "draw_precision": float(
            precision[
                1
            ]
        ),

        "draw_recall": float(
            recall[
                1
            ]
        ),

        "draw_f1": float(
            class_f1[
                1
            ]
        ),

        "predicted_draw": int(
            (
                predictions
                ==
                1
            ).sum()
        ),

        "actual_draw": int(
            support[
                1
            ]
        ),
    }


# ============================================================
# SEARCH DRAW MODEL + ALPHA
# ============================================================


def search_fusion(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Search draw-model configuration and fusion strength.
    """

    rows = []

    alpha_values = np.arange(
        0.0,
        1.0
        +
        ALPHA_STEP
        /
        2,
        ALPHA_STEP,
    )

    for model_name in DRAW_MODEL_CONFIGS:

        model_rows = dataframe[
            dataframe[
                "model"
            ]
            ==
            model_name
        ].copy()

        base_probabilities = model_rows[
            [
                "base_prob_H",
                "base_prob_D",
                "base_prob_A",
            ]
        ].to_numpy(
            dtype=float
        )

        draw_probabilities = model_rows[
            "draw_probability"
        ].to_numpy(
            dtype=float
        )

        y_true = model_rows[
            "target"
        ].to_numpy(
            dtype=int
        )

        for alpha in alpha_values:

            fused = fuse_draw_probability(
                base_probabilities=base_probabilities,
                draw_probabilities=draw_probabilities,
                alpha=float(
                    alpha
                ),
            )

            metrics = evaluate_multiclass(
                y_true=y_true,
                probabilities=fused,
            )

            rows.append(
                {
                    "draw_model": (
                        model_name
                    ),

                    "alpha": float(
                        alpha
                    ),

                    **metrics,
                }
            )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        by=[
            "log_loss",
            "brier_score",
            "accuracy",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    results[
        "rank"
    ] = range(
        1,
        len(
            results
        )
        +
        1,
    )

    return results


# ============================================================
# META FUSION BACKTEST
# ============================================================


def run_fusion_meta_backtest(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Leave one OOF season out.

    Choose:
        draw model configuration
        fusion alpha

    using the OTHER six seasons, then evaluate on the
    held-out season.

    This is the unbiased draw-layer decision test.
    """

    seasons = sorted(
        dataframe[
            "validation_season"
        ].unique()
    )

    rows = []

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LEAVE-ONE-SEASON-OUT DRAW FUSION META-BACKTEST"
    )

    print(
        "=" * 75
    )

    for held_out_season in seasons:

        meta_train = dataframe[
            dataframe[
                "validation_season"
            ]
            !=
            held_out_season
        ].copy()

        held_out = dataframe[
            dataframe[
                "validation_season"
            ]
            ==
            held_out_season
        ].copy()

        search = search_fusion(
            meta_train
        )

        best = search.iloc[
            0
        ]

        best_model = best[
            "draw_model"
        ]

        best_alpha = float(
            best[
                "alpha"
            ]
        )

        held_model = held_out[
            held_out[
                "model"
            ]
            ==
            best_model
        ].copy()

        base_probabilities = held_model[
            [
                "base_prob_H",
                "base_prob_D",
                "base_prob_A",
            ]
        ].to_numpy(
            dtype=float
        )

        draw_probabilities = held_model[
            "draw_probability"
        ].to_numpy(
            dtype=float
        )

        fused = fuse_draw_probability(
            base_probabilities=base_probabilities,
            draw_probabilities=draw_probabilities,
            alpha=best_alpha,
        )

        metrics = evaluate_multiclass(
            y_true=held_model[
                "target"
            ].to_numpy(
                dtype=int
            ),
            probabilities=fused,
        )

        rows.append(
            {
                "held_out_season": int(
                    held_out_season
                ),

                "rows": int(
                    len(
                        held_model
                    )
                ),

                "draw_model": (
                    best_model
                ),

                "alpha": (
                    best_alpha
                ),

                **metrics,
            }
        )

        print(
            f"\nSeason {held_out_season}:"
        )

        print(
            f"  Draw model: "
            f"{best_model}"
        )

        print(
            f"  Alpha:      "
            f"{best_alpha:.2f}"
        )

        print(
            f"  Log Loss:   "
            f"{metrics['log_loss']:.4f}"
        )

        print(
            f"  Brier:      "
            f"{metrics['brier_score']:.4f}"
        )

        print(
            f"  Accuracy:   "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Draw Recall:"
            f" {metrics['draw_recall']:.4f}"
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================


def run_draw_layer_backtest() -> None:
    """
    Build and evaluate a dedicated draw probability layer.

    2025/26 remains untouched.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - DEDICATED DRAW LAYER BACKTEST"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # FEATURES / DATA
    # --------------------------------------------------------

    manifest = load_manifest()

    all_features = manifest[
        "feature_columns"
    ]

    features = build_primary_features(
        all_features
    )

    dataframe = load_ml_data(
        all_features
    )

    backtest_data = dataframe[
        dataframe[
            "season"
        ]
        <
        FINAL_TEST_SEASON
    ].copy()

    final_test = dataframe[
        dataframe[
            "season"
        ]
        ==
        FINAL_TEST_SEASON
    ].copy()

    print(
        f"\nPrimary features: "
        f"{len(features):,}"
    )

    print(
        f"Backtest rows: "
        f"{len(backtest_data):,}"
    )

    print(
        f"Final untouched rows: "
        f"{len(final_test):,}"
    )

    print(
        f"Overall historical draw rate: "
        f"{backtest_data[DRAW_TARGET_COLUMN].mean():.4f}"
    )

    print(
        "2025/26 excluded from all model development ✓"
    )

    # --------------------------------------------------------
    # DRAW OOF
    # --------------------------------------------------------

    (
        fold_metrics,
        draw_predictions,
    ) = run_draw_oof(
        dataframe=backtest_data,
        feature_columns=features,
    )

    draw_summary = summarize_draw_models(
        fold_metrics
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BINARY DRAW MODEL SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        "\n"
        + draw_summary[
            [
                "rank",
                "model",
                "folds",
                "validation_matches",
                "weighted_binary_log_loss",
                "weighted_roc_auc",
                "weighted_average_precision",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # MERGE BASE CATBOOST
    # --------------------------------------------------------

    base_predictions = (
        load_base_catboost_oof()
    )

    merged = merge_with_base_predictions(
        draw_predictions=draw_predictions,
        base_predictions=base_predictions,
    )

    # --------------------------------------------------------
    # META FUSION
    # --------------------------------------------------------

    meta_backtest = (
        run_fusion_meta_backtest(
            merged
        )
    )

    meta_weights = meta_backtest[
        "rows"
    ]

    meta_log_loss = weighted_mean(
        meta_backtest[
            "log_loss"
        ],
        meta_weights,
    )

    meta_brier = weighted_mean(
        meta_backtest[
            "brier_score"
        ],
        meta_weights,
    )

    meta_accuracy = weighted_mean(
        meta_backtest[
            "accuracy"
        ],
        meta_weights,
    )

    meta_macro_f1 = weighted_mean(
        meta_backtest[
            "macro_f1"
        ],
        meta_weights,
    )

    meta_draw_recall = weighted_mean(
        meta_backtest[
            "draw_recall"
        ],
        meta_weights,
    )

    meta_draw_f1 = weighted_mean(
        meta_backtest[
            "draw_f1"
        ],
        meta_weights,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DRAW FUSION META-BACKTEST SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nWeighted Log Loss: "
        f"{meta_log_loss:.4f}"
    )

    print(
        f"Weighted Brier: "
        f"{meta_brier:.4f}"
    )

    print(
        f"Weighted Accuracy: "
        f"{meta_accuracy:.4f}"
    )

    print(
        f"Weighted Macro F1: "
        f"{meta_macro_f1:.4f}"
    )

    print(
        f"Weighted Draw Recall: "
        f"{meta_draw_recall:.4f}"
    )

    print(
        f"Weighted Draw F1: "
        f"{meta_draw_f1:.4f}"
    )

    # --------------------------------------------------------
    # FINAL OOF SEARCH
    # --------------------------------------------------------

    final_search = search_fusion(
        merged
    )

    final_best = final_search.iloc[
        0
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL OOF DRAW FUSION SEARCH"
    )

    print(
        "=" * 75
    )

    print(
        f"\nBest draw model: "
        f"{final_best['draw_model']}"
    )

    print(
        f"Best alpha: "
        f"{final_best['alpha']:.2f}"
    )

    print(
        f"OOF Log Loss: "
        f"{final_best['log_loss']:.4f}"
    )

    print(
        f"OOF Brier: "
        f"{final_best['brier_score']:.4f}"
    )

    print(
        f"OOF Accuracy: "
        f"{final_best['accuracy']:.4f}"
    )

    print(
        f"OOF Macro F1: "
        f"{final_best['macro_f1']:.4f}"
    )

    print(
        f"OOF Draw Recall: "
        f"{final_best['draw_recall']:.4f}"
    )

    print(
        f"OOF Draw F1: "
        f"{final_best['draw_f1']:.4f}"
    )

    print(
        "\nTop 15 fusion combinations:"
    )

    print(
        final_search[
            [
                "rank",
                "draw_model",
                "alpha",
                "log_loss",
                "brier_score",
                "accuracy",
                "macro_f1",
                "draw_precision",
                "draw_recall",
                "draw_f1",
                "predicted_draw",
            ]
        ]
        .head(
            15
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        DRAW_MODEL_FOLD_METRICS_FILE,
        index=False,
        encoding="utf-8",
    )

    draw_summary.to_csv(
        DRAW_MODEL_SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    final_search.to_csv(
        DRAW_FUSION_SEARCH_FILE,
        index=False,
        encoding="utf-8",
    )

    meta_backtest.to_csv(
        DRAW_META_BACKTEST_FILE,
        index=False,
        encoding="utf-8",
    )

    output_predictions = (
        draw_predictions.copy()
    )

    output_predictions[
        "date"
    ] = pd.to_datetime(
        output_predictions[
            "date"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    output_predictions.to_csv(
        DRAW_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "base_model": (
            "catboost_without_transfer"
        ),

        "base_model_oof_log_loss": (
            0.9166
        ),

        "feature_count": len(
            features
        ),

        "draw_model_configs": (
            DRAW_MODEL_CONFIGS
        ),

        "fusion_method": (
            "blend binary draw probability with base P(D), "
            "then redistribute remaining mass using base H:A ratio"
        ),

        "alpha_step": (
            ALPHA_STEP
        ),

        "meta_backtest": {
            "weighted_log_loss": (
                meta_log_loss
            ),

            "weighted_brier": (
                meta_brier
            ),

            "weighted_accuracy": (
                meta_accuracy
            ),

            "weighted_macro_f1": (
                meta_macro_f1
            ),

            "weighted_draw_recall": (
                meta_draw_recall
            ),

            "weighted_draw_f1": (
                meta_draw_f1
            ),
        },

        "final_oof_choice": {
            "draw_model": (
                final_best[
                    "draw_model"
                ]
            ),

            "alpha": float(
                final_best[
                    "alpha"
                ]
            ),

            "log_loss": float(
                final_best[
                    "log_loss"
                ]
            ),

            "brier": float(
                final_best[
                    "brier_score"
                ]
            ),

            "draw_recall": float(
                final_best[
                    "draw_recall"
                ]
            ),

            "draw_f1": float(
                final_best[
                    "draw_f1"
                ]
            ),
        },

        "final_test_season": (
            FINAL_TEST_SEASON
        ),

        "final_test_evaluated": False,
    }

    with DRAW_METADATA_FILE.open(
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
    # FINAL
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DRAW LAYER BACKTEST COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nBase CatBoost benchmark:"
    )

    print(
        "  Log Loss:    0.9166"
    )

    print(
        "  Brier:       0.5390"
    )

    print(
        "  Draw Recall: 0.0164"
    )

    print(
        "\nDraw-layer meta-backtest:"
    )

    print(
        f"  Log Loss:    "
        f"{meta_log_loss:.4f}"
    )

    print(
        f"  Brier:       "
        f"{meta_brier:.4f}"
    )

    print(
        f"  Draw Recall: "
        f"{meta_draw_recall:.4f}"
    )

    print(
        "\n2025/26 remains untouched ✓"
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    run_draw_layer_backtest()