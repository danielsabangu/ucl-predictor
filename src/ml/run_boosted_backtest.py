from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)

MODEL_REPORT_DIR = (
    REPORTS_DIR
    / "model_reports"
)

FOLD_METRICS_FILE = (
    METRICS_DIR
    / "boosted_rolling_backtest_fold_metrics.csv"
)

SUMMARY_FILE = (
    METRICS_DIR
    / "boosted_rolling_backtest_summary.csv"
)

PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "boosted_rolling_backtest_predictions.csv"
)

METADATA_FILE = (
    MODEL_METADATA_DIR
    / "boosted_rolling_backtest_results.json"
)


# ============================================================
# BACKTEST POLICY
# ============================================================

FIRST_VALIDATION_SEASON = 2018

LAST_VALIDATION_SEASON = 2024

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


def load_manifest() -> dict:
    """
    Load approved feature manifest.
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

    features = manifest.get(
        "feature_columns"
    )

    if not features:

        raise RuntimeError(
            "Feature manifest contains no features."
        )

    return manifest


# ============================================================
# FEATURE FAMILY
# ============================================================


def determine_feature_family(
    feature: str,
) -> str:
    """
    Match existing feature-family rules.
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


def primary_features(
    all_features: list[str],
) -> list[str]:
    """
    Primary rolling-backtest feature set.

    Transfer features are excluded because the previous
    seven-season rolling test showed better probability
    performance without them.
    """

    return [
        feature
        for feature in all_features
        if determine_feature_family(
            feature
        )
        !=
        "transfer"
    ]


# ============================================================
# LOAD DATA
# ============================================================


def load_data(
    all_features: list[str],
) -> pd.DataFrame:
    """
    Load chronological ML dataset.
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
# BUILD FOLD
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
    Expanding chronological training window.
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
            f"No training data for "
            f"{validation_season}."
        )

    if validation.empty:

        raise RuntimeError(
            f"No validation data for "
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
            f"Chronological overlap detected "
            f"for {validation_season}."
        )

    return (
        train,
        validation,
    )


# ============================================================
# MODELS
# ============================================================


def build_models() -> dict:
    """
    Boosted model candidates.

    Hyperparameters are intentionally reasonable defaults,
    not final tuned values.
    """

    models = {
        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        "xgboost": Pipeline(
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
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=3,

                        n_estimators=700,
                        learning_rate=0.025,

                        max_depth=4,
                        min_child_weight=5,

                        subsample=0.80,
                        colsample_bytree=0.70,

                        reg_alpha=0.25,
                        reg_lambda=2.0,

                        eval_metric="mlogloss",

                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # ----------------------------------------------------
        # CATBOOST
        # ----------------------------------------------------

        "catboost": Pipeline(
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
        ),

        # ----------------------------------------------------
        # SKLEARN HISTOGRAM GRADIENT BOOSTING
        # ----------------------------------------------------

        "hist_gradient_boosting": Pipeline(
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
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,

                        max_iter=350,

                        max_leaf_nodes=15,

                        max_depth=None,

                        min_samples_leaf=20,

                        l2_regularization=1.0,

                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


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
# ALIGN PROBABILITIES
# ============================================================


def align_probabilities(
    *,
    model: Pipeline,
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Ensure H/D/A probability order.
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
            3,
        ),
        dtype=float,
    )

    for (
        source_index,
        class_label,
    ) in enumerate(
        classes
    ):

        aligned[
            :,
            int(
                class_label
            ),
        ] = probabilities[
            :,
            source_index,
        ]

    return aligned


# ============================================================
# EVALUATE
# ============================================================


def evaluate(
    *,
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:
    """
    Calculate prediction and probability metrics.
    """

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    model_log_loss = log_loss(
        y_true,
        probabilities,
        labels=TARGET_LABELS,
    )

    brier = multiclass_brier_score(
        y_true.to_numpy(),
        probabilities,
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
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "log_loss": float(
            model_log_loss
        ),

        "brier_score": float(
            brier
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
    }


# ============================================================
# WEIGHTED MEAN
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


# ============================================================
# SUMMARY
# ============================================================


def build_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate all seven rolling folds.
    """

    rows = []

    for (
        model_name,
        group,
    ) in metrics.groupby(
        "model"
    ):

        weights = group[
            "validation_rows"
        ]

        rows.append(
            {
                "model": model_name,

                "feature_count": int(
                    group[
                        "feature_count"
                    ].iloc[
                        0
                    ]
                ),

                "folds": int(
                    len(
                        group
                    )
                ),

                "validation_matches": int(
                    group[
                        "validation_rows"
                    ].sum()
                ),

                "weighted_accuracy": weighted_mean(
                    group[
                        "accuracy"
                    ],
                    weights,
                ),

                "weighted_macro_f1": weighted_mean(
                    group[
                        "macro_f1"
                    ],
                    weights,
                ),

                "weighted_log_loss": weighted_mean(
                    group[
                        "log_loss"
                    ],
                    weights,
                ),

                "log_loss_std": float(
                    group[
                        "log_loss"
                    ].std(
                        ddof=0
                    )
                ),

                "weighted_brier": weighted_mean(
                    group[
                        "brier_score"
                    ],
                    weights,
                ),

                "weighted_draw_recall": weighted_mean(
                    group[
                        "draw_recall"
                    ],
                    weights,
                ),

                "weighted_draw_f1": weighted_mean(
                    group[
                        "draw_f1"
                    ],
                    weights,
                ),

                "best_fold_log_loss": float(
                    group[
                        "log_loss"
                    ].min()
                ),

                "worst_fold_log_loss": float(
                    group[
                        "log_loss"
                    ].max()
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        by=[
            "weighted_log_loss",
            "weighted_brier",
            "weighted_accuracy",
        ],
        ascending=[
            True,
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
        + 1,
    )

    return summary


# ============================================================
# MAIN
# ============================================================


def run_boosted_backtest() -> None:
    """
    Rolling boosted-model comparison.

    Uses the 290-feature no-transfer set established by the
    previous rolling Random Forest experiment.

    2025/26 remains untouched.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - BOOSTED MODEL ROLLING BACKTEST"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # FEATURE SET
    # --------------------------------------------------------

    manifest = load_manifest()

    all_features = manifest[
        "feature_columns"
    ]

    features = primary_features(
        all_features
    )

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
        f"\nMaster features: "
        f"{len(all_features):,}"
    )

    print(
        f"Transfer features excluded: "
        f"{len(transfer_features):,}"
    )

    print(
        f"Primary model features: "
        f"{len(features):,}"
    )

    if len(
        features
    ) != 290:

        print(
            "\nWARNING:"
        )

        print(
            "Expected 290 no-transfer features based "
            "on the previous experiment."
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    dataframe = load_data(
        all_features
    )

    final_test = dataframe[
        dataframe[
            "season"
        ]
        ==
        FINAL_TEST_SEASON
    ]

    backtest_data = dataframe[
        dataframe[
            "season"
        ]
        <
        FINAL_TEST_SEASON
    ].copy()

    if len(
        final_test
    ) != 189:

        raise RuntimeError(
            "Unexpected final test row count."
        )

    print(
        f"\nBacktest rows: "
        f"{len(backtest_data):,}"
    )

    print(
        f"Final untouched test rows: "
        f"{len(final_test):,}"
    )

    print(
        "2025/26 excluded from all folds ✓"
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    validation_seasons = list(
        range(
            FIRST_VALIDATION_SEASON,
            LAST_VALIDATION_SEASON
            +
            1,
        )
    )

    print(
        "\nValidation seasons:"
    )

    print(
        "  "
        +
        ", ".join(
            str(
                season
            )
            for season in validation_seasons
        )
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

    metric_rows = []

    prediction_rows = []

    # --------------------------------------------------------
    # FOLDS
    # --------------------------------------------------------

    for validation_season in validation_seasons:

        (
            train,
            validation,
        ) = build_fold(
            dataframe=backtest_data,
            validation_season=validation_season,
        )

        print(
            "\n"
            + "=" * 75
        )

        print(
            f"VALIDATION SEASON {validation_season}"
        )

        print(
            "=" * 75
        )

        print(
            f"\nTraining rows: "
            f"{len(train):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        X_train = train[
            features
        ]

        y_train = train[
            TARGET_COLUMN
        ]

        X_validation = validation[
            features
        ]

        y_validation = validation[
            TARGET_COLUMN
        ]

        models = build_models()

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

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

            probabilities = model.predict_proba(
                X_validation
            )

            probabilities = align_probabilities(
                model=model,
                probabilities=probabilities,
            )

            metrics = evaluate(
                y_true=y_validation,
                probabilities=probabilities,
            )

            metric_row = {
                "validation_season": (
                    validation_season
                ),

                "model": (
                    model_name
                ),

                "feature_set": (
                    "without_transfer"
                ),

                "feature_count": len(
                    features
                ),

                "training_rows": len(
                    train
                ),

                "validation_rows": len(
                    validation
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
            }

            metric_row.update(
                metrics
            )

            metric_rows.append(
                metric_row
            )

            predictions = np.argmax(
                probabilities,
                axis=1,
            )

            for index, (
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

                        "model": model_name,

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

                        "prediction": TARGET_NAMES[
                            int(
                                predictions[
                                    index
                                ]
                            )
                        ],

                        "prob_H": float(
                            probabilities[
                                index,
                                0,
                            ]
                        ),

                        "prob_D": float(
                            probabilities[
                                index,
                                1,
                            ]
                        ),

                        "prob_A": float(
                            probabilities[
                                index,
                                2,
                            ]
                        ),
                    }
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
                f"  Macro F1:   "
                f"{metrics['macro_f1']:.4f}"
            )

            print(
                f"  Draw Recall:"
                f" {metrics['draw_recall']:.4f}"
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    summary = build_summary(
        fold_metrics
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BOOSTED MODEL ROLLING SUMMARY"
    )

    print(
        "=" * 75
    )

    display_columns = [
        "rank",
        "model",
        "feature_count",
        "folds",
        "validation_matches",
        "weighted_accuracy",
        "weighted_macro_f1",
        "weighted_log_loss",
        "log_loss_std",
        "weighted_brier",
        "weighted_draw_recall",
        "weighted_draw_f1",
    ]

    print(
        "\n"
        + summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # SEASON RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SEASON-BY-SEASON LOG LOSS"
    )

    print(
        "=" * 75
    )

    pivot = fold_metrics.pivot(
        index="validation_season",
        columns="model",
        values="log_loss",
    )

    print(
        "\n"
        + pivot.to_string(
            float_format=lambda value: (
                f"{value:.4f}"
            )
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    fold_metrics.to_csv(
        FOLD_METRICS_FILE,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    predictions[
        "date"
    ] = pd.to_datetime(
        predictions[
            "date"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    best = summary.iloc[
        0
    ]

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "feature_set": (
            "without_transfer"
        ),

        "feature_count": len(
            features
        ),

        "excluded_transfer_features": len(
            transfer_features
        ),

        "first_validation_season": (
            FIRST_VALIDATION_SEASON
        ),

        "last_validation_season": (
            LAST_VALIDATION_SEASON
        ),

        "final_test_season": (
            FINAL_TEST_SEASON
        ),

        "final_test_evaluated": False,

        "models": [
            "xgboost",
            "catboost",
            "hist_gradient_boosting",
        ],

        "summary": (
            summary.to_dict(
                orient="records"
            )
        ),

        "best_model": (
            best[
                "model"
            ]
        ),

        "best_weighted_log_loss": float(
            best[
                "weighted_log_loss"
            ]
        ),

        "best_weighted_brier": float(
            best[
                "weighted_brier"
            ]
        ),
    }

    with METADATA_FILE.open(
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
        "BOOSTED MODEL BACKTEST COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        f"\nBest boosted model: "
        f"{best['model']}"
    )

    print(
        f"Weighted Log Loss: "
        f"{best['weighted_log_loss']:.4f}"
    )

    print(
        f"Weighted Brier: "
        f"{best['weighted_brier']:.4f}"
    )

    print(
        f"Weighted Accuracy: "
        f"{best['weighted_accuracy']:.4f}"
    )

    print(
        f"Weighted Draw Recall: "
        f"{best['weighted_draw_recall']:.4f}"
    )

    print(
        "\nRandom Forest benchmark to beat:"
    )

    print(
        "  Weighted Log Loss: 0.9228"
    )

    print(
        "  Weighted Brier:    0.5428"
    )

    print(
        "\n2025/26 remains untouched ✓"
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    run_boosted_backtest()