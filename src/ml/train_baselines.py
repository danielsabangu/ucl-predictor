from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

BASELINE_METRICS_FILE = (
    METRICS_DIR
    / "baseline_validation_metrics.csv"
)

BASELINE_JSON_FILE = (
    MODEL_METADATA_DIR
    / "baseline_validation_metrics.json"
)

VALIDATION_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "baseline_validation_predictions.csv"
)

SPLIT_REPORT_FILE = (
    METRICS_DIR
    / "chronological_split_report.csv"
)


# ============================================================
# SPLIT POLICY
# ============================================================

TRAIN_MAX_SEASON = 2023

VALIDATION_SEASON = 2024

FINAL_TEST_SEASON = 2025


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
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD MANIFEST
# ============================================================


def load_feature_manifest() -> dict:
    """
    Load the exact approved model feature list.
    """

    if not FEATURE_MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Feature manifest not found:\n"
            f"{FEATURE_MANIFEST_FILE}"
        )

    with FEATURE_MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(
            file
        )

    if (
        "feature_columns"
        not in manifest
    ):

        raise RuntimeError(
            "Feature manifest does not contain "
            "'feature_columns'."
        )

    features = manifest[
        "feature_columns"
    ]

    if not features:

        raise RuntimeError(
            "Feature manifest contains "
            "no model features."
        )

    return manifest


# ============================================================
# LOAD DATA
# ============================================================


def load_ml_data(
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Load chronological ML matrix.
    """

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"ML feature dataset not found:\n"
            f"{FEATURE_FILE}"
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
        feature_columns
    )

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "ML dataset missing required columns: "
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
# CHRONOLOGICAL SPLIT
# ============================================================


def chronological_split(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Fixed season-based chronological split.

    Train:
        through 2023/24

    Validation:
        2024/25

    Final test:
        2025/26

    The test set is intentionally not scored by this script.
    """

    train = dataframe[
        dataframe[
            "season"
        ]
        <=
        TRAIN_MAX_SEASON
    ].copy()

    validation = dataframe[
        dataframe[
            "season"
        ]
        ==
        VALIDATION_SEASON
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
            "Training split is empty."
        )

    if validation.empty:
        raise RuntimeError(
            "Validation split is empty."
        )

    if test.empty:
        raise RuntimeError(
            "Final test split is empty."
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
            "Train/validation chronology overlap."
        )

    if (
        validation[
            "date"
        ].max()
        >=
        test[
            "date"
        ].min()
    ):

        raise RuntimeError(
            "Validation/test chronology overlap."
        )

    return (
        train,
        validation,
        test,
    )


# ============================================================
# SPLIT AUDIT
# ============================================================


def validate_split_classes(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    """
    Ensure all H/D/A labels exist in each split.
    """

    classes = set(
        dataframe[
            TARGET_COLUMN
        ].unique()
    )

    expected = set(
        TARGET_LABELS
    )

    if classes != expected:

        raise RuntimeError(
            f"{split_name} split does not contain "
            f"all target classes: {classes}"
        )


def build_split_report(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build human-readable chronological split report.
    """

    rows = []

    for (
        name,
        dataframe,
    ) in [
        (
            "train",
            train,
        ),
        (
            "validation",
            validation,
        ),
        (
            "final_test",
            test,
        ),
    ]:

        target_counts = (
            dataframe[
                "result"
            ]
            .value_counts()
        )

        rows.append(
            {
                "split": name,

                "rows": len(
                    dataframe
                ),

                "first_date": (
                    dataframe[
                        "date"
                    ]
                    .min()
                    .date()
                    .isoformat()
                ),

                "last_date": (
                    dataframe[
                        "date"
                    ]
                    .max()
                    .date()
                    .isoformat()
                ),

                "first_season": int(
                    dataframe[
                        "season"
                    ].min()
                ),

                "last_season": int(
                    dataframe[
                        "season"
                    ].max()
                ),

                "home_wins": int(
                    target_counts.get(
                        "H",
                        0,
                    )
                ),

                "draws": int(
                    target_counts.get(
                        "D",
                        0,
                    )
                ),

                "away_wins": int(
                    target_counts.get(
                        "A",
                        0,
                    )
                ),

                "feature_availability_percent": (
                    np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PREPROCESSING
# ============================================================


def make_scaled_preprocessor(
    feature_columns: list[str],
) -> ColumnTransformer:
    """
    Median imputation + missing indicators + scaling.

    Statistics are learned from training only because this
    transformer lives inside the sklearn Pipeline.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_columns,
            ),
        ],
        remainder="drop",
    )


def make_tree_preprocessor(
    feature_columns: list[str],
) -> ColumnTransformer:
    """
    Median imputation + explicit missing indicators.

    Tree models do not require feature scaling.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_columns,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# MODELS
# ============================================================


def build_models(
    feature_columns: list[str],
) -> dict[str, Pipeline]:
    """
    Establish diverse baseline models.

    These are intentionally not heavily tuned yet.
    """

    scaled = make_scaled_preprocessor(
        feature_columns
    )

    tree = make_tree_preprocessor(
        feature_columns
    )

    models = {
        # ----------------------------------------------------
        # PRIOR-ONLY CONTROL
        # ----------------------------------------------------

        "dummy_prior": Pipeline(
            steps=[
                (
                    "model",
                    DummyClassifier(
                        strategy="prior",
                    ),
                ),
            ]
        ),

        # ----------------------------------------------------
        # LINEAR
        # ----------------------------------------------------

        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    scaled,
                ),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "logistic_regression_balanced": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_scaled_preprocessor(
                        feature_columns
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    tree,
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        max_depth=None,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # ----------------------------------------------------
        # EXTRA TREES
        # ----------------------------------------------------

        "extra_trees": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_tree_preprocessor(
                        feature_columns
                    ),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_depth=None,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


# ============================================================
# MULTICLASS BRIER SCORE
# ============================================================


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Multiclass Brier score.

    Lower is better.

    Computes mean squared probability error across
    the three mutually exclusive result classes.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
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
# PROBABILITY CLASS ORDER
# ============================================================


def align_probabilities(
    *,
    model: Pipeline,
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Guarantee probability columns are ordered:

        H=0, D=1, A=2
    """

    classes = np.asarray(
        model.classes_,
        dtype=int,
    )

    aligned = np.zeros(
        (
            len(
                probabilities
            ),
            len(
                TARGET_LABELS
            ),
        ),
        dtype=float,
    )

    for (
        source_index,
        class_label,
    ) in enumerate(
        classes
    ):

        destination_index = (
            TARGET_LABELS.index(
                int(
                    class_label
                )
            )
        )

        aligned[
            :,
            destination_index,
        ] = probabilities[
            :,
            source_index,
        ]

    return aligned


# ============================================================
# EVALUATION
# ============================================================


def evaluate_model(
    *,
    model_name: str,
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    dict,
    pd.DataFrame,
    np.ndarray,
]:
    """
    Evaluate one model on validation data only.
    """

    probabilities = model.predict_proba(
        X_validation
    )

    probabilities = align_probabilities(
        model=model,
        probabilities=probabilities,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    validation_log_loss = log_loss(
        y_validation,
        probabilities,
        labels=TARGET_LABELS,
    )

    brier = multiclass_brier_score(
        y_validation.to_numpy(),
        probabilities,
    )

    confusion = confusion_matrix(
        y_validation,
        predictions,
        labels=TARGET_LABELS,
    )

    metrics = {
        "model": model_name,

        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "log_loss": float(
            validation_log_loss
        ),

        "brier_score": float(
            brier
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
        probabilities,
    )


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================


def add_prediction_columns(
    *,
    prediction_frame: pd.DataFrame,
    model_name: str,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Add validation probabilities/predictions for one model.
    """

    output = prediction_frame.copy()

    predicted_class = np.argmax(
        probabilities,
        axis=1,
    )

    output[
        f"{model_name}_prob_H"
    ] = probabilities[
        :,
        0,
    ]

    output[
        f"{model_name}_prob_D"
    ] = probabilities[
        :,
        1,
    ]

    output[
        f"{model_name}_prob_A"
    ] = probabilities[
        :,
        2,
    ]

    output[
        f"{model_name}_prediction"
    ] = [
        TARGET_NAMES[
            int(
                value
            )
        ]
        for value in predicted_class
    ]

    return output


# ============================================================
# MAIN
# ============================================================


def train_baselines() -> None:
    """
    Train first validation-only baseline models.

    IMPORTANT:
    The 2025/26 final test set is intentionally not scored.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - CHRONOLOGICAL BASELINE MODELS"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    manifest = load_feature_manifest()

    feature_columns = manifest[
        "feature_columns"
    ]

    print(
        f"\nApproved model features: "
        f"{len(feature_columns):,}"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    dataframe = load_ml_data(
        feature_columns
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train,
        validation,
        final_test,
    ) = chronological_split(
        dataframe
    )

    validate_split_classes(
        train,
        "Training",
    )

    validate_split_classes(
        validation,
        "Validation",
    )

    validate_split_classes(
        final_test,
        "Final test",
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CHRONOLOGICAL SPLIT"
    )

    print(
        "=" * 75
    )

    print(
        f"\nTraining:"
    )

    print(
        f"  rows: "
        f"{len(train):,}"
    )

    print(
        f"  seasons: "
        f"{train['season'].min()} "
        f"to "
        f"{train['season'].max()}"
    )

    print(
        f"  dates: "
        f"{train['date'].min().date()} "
        f"to "
        f"{train['date'].max().date()}"
    )

    print(
        "\nValidation:"
    )

    print(
        f"  rows: "
        f"{len(validation):,}"
    )

    print(
        f"  season: "
        f"{VALIDATION_SEASON}"
    )

    print(
        f"  dates: "
        f"{validation['date'].min().date()} "
        f"to "
        f"{validation['date'].max().date()}"
    )

    print(
        "\nFinal test:"
    )

    print(
        f"  rows: "
        f"{len(final_test):,}"
    )

    print(
        f"  season: "
        f"{FINAL_TEST_SEASON}"
    )

    print(
        f"  dates: "
        f"{final_test['date'].min().date()} "
        f"to "
        f"{final_test['date'].max().date()}"
    )

    print(
        "\nFINAL TEST LABELS WILL NOT BE "
        "EVALUATED IN THIS SCRIPT ✓"
    )

    # --------------------------------------------------------
    # SPLIT REPORT
    # --------------------------------------------------------

    split_report = build_split_report(
        train=train,
        validation=validation,
        test=final_test,
    )

    for (
        split_name,
        split_dataframe,
    ) in [
        (
            "train",
            train,
        ),
        (
            "validation",
            validation,
        ),
        (
            "final_test",
            final_test,
        ),
    ]:

        availability = (
            1
            -
            split_dataframe[
                feature_columns
            ].isna().mean().mean()
        )

        split_report.loc[
            split_report[
                "split"
            ]
            ==
            split_name,
            "feature_availability_percent",
        ] = (
            availability
            *
            100
        )

    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X_train = train[
        feature_columns
    ]

    y_train = train[
        TARGET_COLUMN
    ]

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = build_models(
        feature_columns
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRAINING BASELINE MODELS"
    )

    print(
        "=" * 75
    )

    metrics_rows = []

    confusion_matrices = {}

    prediction_frame = validation[
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
    # FIT
    # --------------------------------------------------------

    for (
        model_name,
        model,
    ) in models.items():

        print(
            f"\nTraining: "
            f"{model_name}"
        )

        model.fit(
            X_train,
            y_train,
        )

        (
            metrics,
            confusion,
            probabilities,
        ) = evaluate_model(
            model_name=model_name,
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        metrics_rows.append(
            metrics
        )

        confusion_matrices[
            model_name
        ] = confusion

        prediction_frame = add_prediction_columns(
            prediction_frame=prediction_frame,
            model_name=model_name,
            probabilities=probabilities,
        )

        model_file = (
            TRAINED_MODELS_DIR
            /
            f"baseline_{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_file,
        )

        print(
            f"  Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"  Log Loss: "
            f"{metrics['log_loss']:.4f}"
        )

        print(
            f"  Brier:    "
            f"{metrics['brier_score']:.4f}"
        )

    # --------------------------------------------------------
    # METRICS TABLE
    # --------------------------------------------------------

    metrics_frame = pd.DataFrame(
        metrics_rows
    )

    metrics_frame = metrics_frame.sort_values(
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

    metrics_frame[
        "validation_rank"
    ] = range(
        1,
        len(
            metrics_frame
        )
        + 1,
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 75
    )

    display_columns = [
        "validation_rank",
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "brier_score",
        "predicted_home",
        "predicted_draw",
        "predicted_away",
    ]

    print(
        "\n"
        + metrics_frame[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    for model_name in metrics_frame[
        "model"
    ]:

        print(
            "\n"
            + "-" * 75
        )

        print(
            f"CONFUSION MATRIX: "
            f"{model_name}"
        )

        print(
            "-" * 75
        )

        print(
            confusion_matrices[
                model_name
            ].to_string()
        )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    metrics_frame.to_csv(
        BASELINE_METRICS_FILE,
        index=False,
        encoding="utf-8",
    )

    split_report.to_csv(
        SPLIT_REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    prediction_output = (
        prediction_frame.copy()
    )

    prediction_output[
        "date"
    ] = prediction_output[
        "date"
    ].dt.strftime(
        "%Y-%m-%d"
    )

    prediction_output.to_csv(
        VALIDATION_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # JSON METADATA
    # --------------------------------------------------------

    best_model = metrics_frame.iloc[
        0
    ]

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "feature_count": len(
            feature_columns
        ),

        "split_policy": {
            "train_max_season": (
                TRAIN_MAX_SEASON
            ),

            "validation_season": (
                VALIDATION_SEASON
            ),

            "final_test_season": (
                FINAL_TEST_SEASON
            ),

            "final_test_evaluated": False,
        },

        "selection_metric": (
            "validation log loss"
        ),

        "secondary_metrics": [
            "multiclass Brier score",
            "macro F1",
            "accuracy",
        ],

        "validation_results": (
            metrics_frame.to_dict(
                orient="records"
            )
        ),

        "best_validation_model": (
            best_model[
                "model"
            ]
        ),

        "best_validation_log_loss": float(
            best_model[
                "log_loss"
            ]
        ),

        "modeling_rules": {
            "imputation_fit_on_training_only": True,

            "scaling_fit_on_training_only": True,

            "validation_used_for_model_selection": True,

            "final_test_used_for_model_selection": False,

            "final_test_remains_untouched": True,
        },
    }

    with BASELINE_JSON_FILE.open(
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
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BASELINE MODEL SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nBest validation model: "
        f"{best_model['model']}"
    )

    print(
        f"Validation Log Loss: "
        f"{best_model['log_loss']:.4f}"
    )

    print(
        f"Validation Brier Score: "
        f"{best_model['brier_score']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{best_model['accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_model['macro_f1']:.4f}"
    )

    print(
        "\nMetrics:"
    )

    print(
        BASELINE_METRICS_FILE
    )

    print(
        "\nValidation predictions:"
    )

    print(
        VALIDATION_PREDICTIONS_FILE
    )

    print(
        "\nSplit report:"
    )

    print(
        SPLIT_REPORT_FILE
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BASELINE TRAINING COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\n2025/26 remains the untouched final test set ✓"
    )

    print(
        "Do not select or tune models using "
        "2025/26 performance."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    train_baselines()