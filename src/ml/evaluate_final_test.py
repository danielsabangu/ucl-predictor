from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_METADATA_DIR,
    TRAINED_MODELS_DIR,
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

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)

MODEL_REPORT_DIR = (
    REPORTS_DIR
    / "model_reports"
)

FINAL_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "final_catboost.joblib"
)

FINAL_METRICS_FILE = (
    METRICS_DIR
    / "final_test_metrics.csv"
)

FINAL_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "final_test_predictions.csv"
)

FINAL_METADATA_FILE = (
    MODEL_METADATA_DIR
    / "final_model_evaluation.json"
)


# ============================================================
# FINAL LOCKED MODEL POLICY
# ============================================================

FINAL_TEST_SEASON = 2025

RANDOM_STATE = 42

EXPECTED_TEST_ROWS = 189

EXPECTED_FEATURE_COUNT = 290


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "target"

TARGET_LABELS = [
    0,
    1,
    2,
]

TARGET_NAMES = {
    0: "H",
    1: "D",
    2: "A",
}


# ============================================================
# HELPERS
# ============================================================


def require_file(
    path,
    description: str,
) -> None:
    """
    Fail early if a required artifact is unavailable.
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
    Normalize probability rows defensively.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
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


# ============================================================
# FEATURE FAMILY
# ============================================================


def determine_feature_family(
    feature: str,
) -> str:
    """
    Match the feature-family rules used during development.
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
# LOAD FEATURE MANIFEST
# ============================================================


def load_manifest() -> dict:
    """
    Load exact leakage-audited model feature manifest.
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
            "Feature manifest contains no "
            "approved features."
        )

    return manifest


# ============================================================
# LOCKED PRIMARY FEATURES
# ============================================================


def build_final_feature_set(
    all_features: list[str],
) -> list[str]:
    """
    Final model feature policy established during rolling
    development:

        Elo
        UCL form
        domestic form
        squad/player features

    Transfer features remain excluded.
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

    transfer_features = [
        feature
        for feature in all_features
        if determine_feature_family(
            feature
        )
        ==
        "transfer"
    ]

    print(
        f"\nApproved manifest features: "
        f"{len(all_features):,}"
    )

    print(
        f"Transfer features excluded: "
        f"{len(transfer_features):,}"
    )

    print(
        f"Final CatBoost features: "
        f"{len(features):,}"
    )

    if len(
        features
    ) != EXPECTED_FEATURE_COUNT:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT} final features, "
            f"found {len(features)}."
        )

    return features


# ============================================================
# LOAD ML DATA
# ============================================================


def load_data(
    all_features: list[str],
) -> pd.DataFrame:
    """
    Load final chronological ML matrix.
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
            f"ML feature matrix missing columns: "
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
# FINAL SPLIT
# ============================================================


def build_final_split(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Development data:
        every season before 2025/26

    Final untouched test:
        2025/26

    There is no further model selection after this point.
    """

    train = dataframe[
        dataframe[
            "season"
        ]
        <
        FINAL_TEST_SEASON
    ].copy()

    test = dataframe[
        dataframe[
            "season"
        ]
        ==
        FINAL_TEST_SEASON
    ].copy()

    if train.empty:

        raise RuntimeError(
            "Final training set is empty."
        )

    if test.empty:

        raise RuntimeError(
            "Final test set is empty."
        )

    if len(
        test
    ) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_TEST_ROWS} "
            f"2025/26 matches, found "
            f"{len(test)}."
        )

    if (
        train[
            "date"
        ].max()
        >=
        test[
            "date"
        ].min()
    ):

        raise RuntimeError(
            "Final train/test chronology overlap."
        )

    return (
        train,
        test,
    )


# ============================================================
# FINAL MODEL
# ============================================================


def build_final_model() -> Pipeline:
    """
    Locked CatBoost model.

    Same configuration that won the rolling boosted
    backtest.

    No post-test tuning is permitted.
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

                    random_seed=RANDOM_STATE,

                    verbose=False,

                    allow_writing_files=False,
                ),
            ),
        ]
    )


# ============================================================
# BRIER SCORE
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

    squared_error = (
        probabilities
        -
        one_hot
    ) ** 2

    return float(
        np.mean(
            np.sum(
                squared_error,
                axis=1,
            )
        )
    )


# ============================================================
# EVALUATE
# ============================================================


def evaluate_final_test(
    *,
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[
    dict,
    pd.DataFrame,
    np.ndarray,
]:
    """
    Perform the one-time final test evaluation.
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

    confusion = confusion_matrix(
        y_true,
        predictions,
        labels=TARGET_LABELS,
    )

    metrics = {
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

        "weighted_f1": float(
            f1_score(
                y_true,
                predictions,
                average="weighted",
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
                y_true.to_numpy(
                    dtype=int
                ),
                probabilities,
            )
        ),

        "home_precision": float(
            precision[
                0
            ]
        ),

        "home_recall": float(
            recall[
                0
            ]
        ),

        "home_f1": float(
            class_f1[
                0
            ]
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

        "away_precision": float(
            precision[
                2
            ]
        ),

        "away_recall": float(
            recall[
                2
            ]
        ),

        "away_f1": float(
            class_f1[
                2
            ]
        ),

        "actual_home": int(
            support[
                0
            ]
        ),

        "actual_draw": int(
            support[
                1
            ]
        ),

        "actual_away": int(
            support[
                2
            ]
        ),

        "predicted_home": int(
            (
                predictions
                ==
                0
            ).sum()
        ),

        "predicted_draw": int(
            (
                predictions
                ==
                1
            ).sum()
        ),

        "predicted_away": int(
            (
                predictions
                ==
                2
            ).sum()
        ),
    }

    confusion_frame = pd.DataFrame(
        confusion,
        index=[
            "actual_H",
            "actual_D",
            "actual_A",
        ],
        columns=[
            "pred_H",
            "pred_D",
            "pred_A",
        ],
    )

    return (
        metrics,
        confusion_frame,
        predictions,
    )


# ============================================================
# PRIOR BASELINE
# ============================================================


def evaluate_prior_baseline(
    *,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Historical class-prior benchmark using training data only.
    """

    distribution = (
        y_train
        .value_counts(
            normalize=True
        )
    )

    prior = np.array(
        [
            distribution.get(
                0,
                0.0,
            ),

            distribution.get(
                1,
                0.0,
            ),

            distribution.get(
                2,
                0.0,
            ),
        ],
        dtype=float,
    )

    probabilities = np.tile(
        prior,
        (
            len(
                y_test
            ),
            1,
        ),
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "log_loss": float(
            log_loss(
                y_test,
                probabilities,
                labels=TARGET_LABELS,
            )
        ),

        "brier_score": (
            multiclass_brier_score(
                y_test.to_numpy(
                    dtype=int
                ),
                probabilities,
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================


def evaluate_final_model() -> None:
    """
    Perform one-time 2025/26 final test evaluation.

    The model specification was fixed before this script was
    run.

    Results from this test must not be used for additional
    hyperparameter or feature tuning.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - ONE-TIME FINAL TEST EVALUATION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = load_manifest()

    all_features = manifest[
        "feature_columns"
    ]

    feature_columns = build_final_feature_set(
        all_features
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    dataframe = load_data(
        all_features
    )

    (
        train,
        test,
    ) = build_final_split(
        dataframe
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL CHRONOLOGICAL SPLIT"
    )

    print(
        "=" * 75
    )

    print(
        f"\nTraining rows: "
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

    print(
        f"\nFinal test rows: "
        f"{len(test):,}"
    )

    print(
        f"Final test season: "
        f"{FINAL_TEST_SEASON}"
    )

    print(
        f"Final test dates: "
        f"{test['date'].min().date()} "
        f"to "
        f"{test['date'].max().date()}"
    )

    # --------------------------------------------------------
    # FEATURE AVAILABILITY
    # --------------------------------------------------------

    train_availability = (
        1
        -
        train[
            feature_columns
        ].isna().mean().mean()
    )

    test_availability = (
        1
        -
        test[
            feature_columns
        ].isna().mean().mean()
    )

    print(
        "\nFeature availability:"
    )

    print(
        f"  Train: "
        f"{train_availability * 100:.2f}%"
    )

    print(
        f"  Test:  "
        f"{test_availability * 100:.2f}%"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    X_train = train[
        feature_columns
    ]

    y_train = train[
        TARGET_COLUMN
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        TARGET_COLUMN
    ]

    print(
        "\nTraining locked CatBoost model..."
    )

    model = build_final_model()

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print(
        "\nOpening untouched 2025/26 test set..."
    )

    probabilities = model.predict_proba(
        X_test
    )

    probabilities = normalize_probabilities(
        probabilities
    )

    (
        metrics,
        confusion,
        predictions,
    ) = evaluate_final_test(
        y_true=y_test,
        probabilities=probabilities,
    )

    prior_metrics = evaluate_prior_baseline(
        y_train=y_train,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL 2025/26 TEST RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"\nLog Loss:     "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Brier Score:  "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"Accuracy:     "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1:     "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:  "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        "\nClass performance:"
    )

    print(
        f"  Home Recall: "
        f"{metrics['home_recall']:.4f}"
    )

    print(
        f"  Draw Recall: "
        f"{metrics['draw_recall']:.4f}"
    )

    print(
        f"  Away Recall: "
        f"{metrics['away_recall']:.4f}"
    )

    print(
        f"\n  Home F1: "
        f"{metrics['home_f1']:.4f}"
    )

    print(
        f"  Draw F1: "
        f"{metrics['draw_f1']:.4f}"
    )

    print(
        f"  Away F1: "
        f"{metrics['away_f1']:.4f}"
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        f"  Actual H/D/A:    "
        f"{metrics['actual_home']} / "
        f"{metrics['actual_draw']} / "
        f"{metrics['actual_away']}"
    )

    print(
        f"  Predicted H/D/A: "
        f"{metrics['predicted_home']} / "
        f"{metrics['predicted_draw']} / "
        f"{metrics['predicted_away']}"
    )

    print(
        "\n"
        + "-" * 75
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "-" * 75
    )

    print(
        confusion.to_string()
    )

    # --------------------------------------------------------
    # PRIOR COMPARISON
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL TEST VS PRIOR BASELINE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nCatBoost Log Loss: "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Prior Log Loss:    "
        f"{prior_metrics['log_loss']:.4f}"
    )

    print(
        f"Improvement:       "
        f"{prior_metrics['log_loss'] - metrics['log_loss']:+.4f}"
    )

    print(
        f"\nCatBoost Brier: "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"Prior Brier:    "
        f"{prior_metrics['brier_score']:.4f}"
    )

    print(
        f"\nCatBoost Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Prior Accuracy:    "
        f"{prior_metrics['accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE DIRECTORIES
    # --------------------------------------------------------

    TRAINED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    joblib.dump(
        model,
        FINAL_MODEL_FILE,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_row = {
        "model": (
            "catboost_without_transfer"
        ),

        "test_season": (
            FINAL_TEST_SEASON
        ),

        "training_rows": (
            len(
                train
            )
        ),

        "test_rows": (
            len(
                test
            )
        ),

        "feature_count": (
            len(
                feature_columns
            )
        ),

        **metrics,

        "prior_log_loss": (
            prior_metrics[
                "log_loss"
            ]
        ),

        "prior_brier_score": (
            prior_metrics[
                "brier_score"
            ]
        ),

        "prior_accuracy": (
            prior_metrics[
                "accuracy"
            ]
        ),
    }

    pd.DataFrame(
        [
            metrics_row
        ]
    ).to_csv(
        FINAL_METRICS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_output = test[
        [
            "match_id",
            "date",
            "season",
            "home_team",
            "away_team",
            "result",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_output[
        "prob_H"
    ] = probabilities[
        :,
        0
    ]

    prediction_output[
        "prob_D"
    ] = probabilities[
        :,
        1
    ]

    prediction_output[
        "prob_A"
    ] = probabilities[
        :,
        2
    ]

    prediction_output[
        "prediction"
    ] = [
        TARGET_NAMES[
            int(
                value
            )
        ]
        for value in predictions
    ]

    prediction_output[
        "correct"
    ] = (
        prediction_output[
            "prediction"
        ]
        ==
        prediction_output[
            "result"
        ]
    ).astype(
        int
    )

    prediction_output[
        "max_probability"
    ] = probabilities.max(
        axis=1
    )

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    prediction_output[
        "probability_margin"
    ] = (
        sorted_probabilities[
            :,
            -1
        ]
        -
        sorted_probabilities[
            :,
            -2
        ]
    )

    prediction_output[
        "date"
    ] = prediction_output[
        "date"
    ].dt.strftime(
        "%Y-%m-%d"
    )

    prediction_output.to_csv(
        FINAL_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "model": (
            "CatBoostClassifier"
        ),

        "model_role": (
            "primary H/D/A probability engine"
        ),

        "feature_policy": {
            "master_features": len(
                all_features
            ),

            "final_features": len(
                feature_columns
            ),

            "transfer_features_excluded": True,
        },

        "model_parameters": {
            "iterations": 700,

            "learning_rate": 0.025,

            "depth": 5,

            "loss_function": (
                "MultiClass"
            ),

            "l2_leaf_reg": 5.0,

            "random_seed": (
                RANDOM_STATE
            ),
        },

        "development_decisions": {
            "ensemble_rejected": True,

            "temperature_calibration_rejected": True,

            "dedicated_draw_layer_rejected": True,

            "raw_catboost_probabilities": True,
        },

        "training": {
            "rows": len(
                train
            ),

            "first_season": int(
                train[
                    "season"
                ].min()
            ),

            "last_season": int(
                train[
                    "season"
                ].max()
            ),

            "first_date": (
                train[
                    "date"
                ]
                .min()
                .date()
                .isoformat()
            ),

            "last_date": (
                train[
                    "date"
                ]
                .max()
                .date()
                .isoformat()
            ),
        },

        "final_test": {
            "season": (
                FINAL_TEST_SEASON
            ),

            "rows": len(
                test
            ),

            "first_date": (
                test[
                    "date"
                ]
                .min()
                .date()
                .isoformat()
            ),

            "last_date": (
                test[
                    "date"
                ]
                .max()
                .date()
                .isoformat()
            ),

            "log_loss": (
                metrics[
                    "log_loss"
                ]
            ),

            "brier_score": (
                metrics[
                    "brier_score"
                ]
            ),

            "accuracy": (
                metrics[
                    "accuracy"
                ]
            ),

            "macro_f1": (
                metrics[
                    "macro_f1"
                ]
            ),

            "draw_recall": (
                metrics[
                    "draw_recall"
                ]
            ),
        },

        "important_rule": (
            "2025/26 is now consumed as the final held-out "
            "test set. Its results must not be used for "
            "further feature or hyperparameter tuning."
        ),
    }

    with FINAL_METADATA_FILE.open(
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
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL MODEL EVALUATION COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nFinal trained model:"
    )

    print(
        FINAL_MODEL_FILE
    )

    print(
        "\nFinal test metrics:"
    )

    print(
        FINAL_METRICS_FILE
    )

    print(
        "\nFinal test predictions:"
    )

    print(
        FINAL_PREDICTIONS_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "2025/26 has now been consumed as the "
        "one-time final test set."
    )

    print(
        "Do not tune the model using these results."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    evaluate_final_model()