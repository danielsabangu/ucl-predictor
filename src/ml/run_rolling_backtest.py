from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
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

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)

MODEL_REPORT_DIR = (
    REPORTS_DIR
    / "model_reports"
)

ROLLING_FOLD_METRICS_FILE = (
    METRICS_DIR
    / "rolling_backtest_fold_metrics.csv"
)

ROLLING_SUMMARY_FILE = (
    METRICS_DIR
    / "rolling_backtest_summary.csv"
)

ROLLING_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "rolling_backtest_predictions.csv"
)

ROLLING_JSON_FILE = (
    MODEL_METADATA_DIR
    / "rolling_backtest_results.json"
)


# ============================================================
# BACKTEST POLICY
# ============================================================

FIRST_VALIDATION_SEASON = 2018
LAST_VALIDATION_SEASON = 2024

FINAL_TEST_SEASON = 2025

MINIMUM_TRAIN_SEASONS = 5


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
# MODEL SETTINGS
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 500


# ============================================================
# LOAD MANIFEST
# ============================================================


def load_manifest() -> dict:
    """
    Load exact approved feature list.
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
# LOAD DATA
# ============================================================


def load_data(
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Load chronological model dataset.
    """

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"ML feature file not found:\n"
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
            f"ML data missing columns: "
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
# FEATURE FAMILY
# ============================================================


def determine_feature_family(
    feature: str,
) -> str:
    """
    Match feature-family rules used previously.
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


def build_family_map(
    feature_columns: list[str],
) -> dict[str, list[str]]:
    """
    Split approved features into families.
    """

    families = {
        "elo": [],
        "ucl_form": [],
        "domestic": [],
        "squad": [],
        "transfer": [],
    }

    for feature in feature_columns:

        family = determine_feature_family(
            feature
        )

        families[
            family
        ].append(
            feature
        )

    return families


# ============================================================
# SCENARIOS
# ============================================================


def build_scenarios(
    *,
    all_features: list[str],
    families: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Focus the rolling backtest on the most informative
    ablation comparisons.

    We intentionally do not run all 12 previous scenarios.
    """

    all_set = set(
        all_features
    )

    family_sets = {
        name: set(
            features
        )
        for (
            name,
            features,
        ) in families.items()
    }

    scenarios = {
        "all_features": (
            all_features
        ),

        "without_transfer": sorted(
            all_set
            -
            family_sets[
                "transfer"
            ]
        ),

        "without_domestic": sorted(
            all_set
            -
            family_sets[
                "domestic"
            ]
        ),

        "without_squad": sorted(
            all_set
            -
            family_sets[
                "squad"
            ]
        ),

        "without_elo": sorted(
            all_set
            -
            family_sets[
                "elo"
            ]
        ),

        "elo_plus_ucl_form": sorted(
            family_sets[
                "elo"
            ]
            |
            family_sets[
                "ucl_form"
            ]
        ),
    }

    return scenarios


# ============================================================
# PREPROCESSOR
# ============================================================


def build_preprocessor(
    feature_columns: list[str],
) -> ColumnTransformer:
    """
    Training-only median imputation + missing indicators.
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
# MODEL
# ============================================================


def build_model(
    feature_columns: list[str],
) -> Pipeline:
    """
    Random Forest configuration matching the successful
    baseline/ablation model.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    feature_columns
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=None,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


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
    Align class probabilities to H/D/A = 0/1/2.
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
# FOLD CREATION
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
    Expanding-window chronological fold.

    Train:
        every season strictly before validation season

    Validate:
        exactly validation season
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
            f"Training data empty for "
            f"validation season "
            f"{validation_season}."
        )

    if validation.empty:

        raise RuntimeError(
            f"Validation season "
            f"{validation_season} missing."
        )

    training_seasons = sorted(
        train[
            "season"
        ].unique()
    )

    if (
        len(
            training_seasons
        )
        <
        MINIMUM_TRAIN_SEASONS
    ):

        raise RuntimeError(
            f"Validation season "
            f"{validation_season} has only "
            f"{len(training_seasons)} "
            f"training seasons."
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
            f"Chronology overlap for "
            f"validation season "
            f"{validation_season}."
        )

    return (
        train,
        validation,
    )


# ============================================================
# PRIOR BASELINE
# ============================================================


def prior_probabilities(
    *,
    y_train: pd.Series,
    validation_rows: int,
) -> np.ndarray:
    """
    Training-set class-prior probabilities for comparison.
    """

    counts = (
        y_train
        .value_counts(
            normalize=True
        )
    )

    prior = np.array(
        [
            counts.get(
                0,
                0.0,
            ),
            counts.get(
                1,
                0.0,
            ),
            counts.get(
                2,
                0.0,
            ),
        ],
        dtype=float,
    )

    return np.tile(
        prior,
        (
            validation_rows,
            1,
        ),
    )


# ============================================================
# EVALUATE PROBABILITIES
# ============================================================


def evaluate_probabilities(
    *,
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:
    """
    Return full probability/classification metrics.
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

    fold_log_loss = log_loss(
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
            fold_log_loss
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


# ============================================================
# FEATURE AVAILABILITY
# ============================================================


def feature_availability(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> float:
    """
    Percentage of non-missing feature cells.
    """

    return float(
        (
            1
            -
            dataframe[
                feature_columns
            ].isna().mean().mean()
        )
        *
        100
    )


# ============================================================
# WEIGHTED MEAN
# ============================================================


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Weighted average by validation row count.
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
# BUILD SUMMARY
# ============================================================


def build_summary(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate metrics across rolling seasons.
    """

    rows = []

    for (
        scenario,
        group,
    ) in fold_metrics.groupby(
        "scenario"
    ):

        weights = group[
            "validation_rows"
        ]

        rows.append(
            {
                "scenario": scenario,

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

                "mean_accuracy": float(
                    group[
                        "accuracy"
                    ].mean()
                ),

                "weighted_accuracy": weighted_mean(
                    group[
                        "accuracy"
                    ],
                    weights,
                ),

                "mean_macro_f1": float(
                    group[
                        "macro_f1"
                    ].mean()
                ),

                "weighted_macro_f1": weighted_mean(
                    group[
                        "macro_f1"
                    ],
                    weights,
                ),

                "mean_log_loss": float(
                    group[
                        "log_loss"
                    ].mean()
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

                "mean_brier": float(
                    group[
                        "brier_score"
                    ].mean()
                ),

                "weighted_brier": weighted_mean(
                    group[
                        "brier_score"
                    ],
                    weights,
                ),

                "mean_draw_recall": float(
                    group[
                        "draw_recall"
                    ].mean()
                ),

                "weighted_draw_recall": weighted_mean(
                    group[
                        "draw_recall"
                    ],
                    weights,
                ),

                "mean_validation_feature_availability": float(
                    group[
                        "validation_feature_availability"
                    ].mean()
                ),

                "mean_log_loss_improvement_vs_prior": float(
                    group[
                        "log_loss_improvement_vs_prior"
                    ].mean()
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


def run_rolling_backtest() -> None:
    """
    Expanding-window multi-season Random Forest backtest.

    2025/26 remains completely untouched.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - MULTI-SEASON ROLLING BACKTEST"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    manifest = load_manifest()

    all_features = manifest[
        "feature_columns"
    ]

    dataframe = load_data(
        all_features
    )

    if (
        dataframe[
            "season"
        ].max()
        <
        FINAL_TEST_SEASON
    ):

        raise RuntimeError(
            "Expected final test season not found."
        )

    # --------------------------------------------------------
    # FINAL TEST PROTECTION
    # --------------------------------------------------------

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
        f"\nApproved features: "
        f"{len(all_features):,}"
    )

    print(
        f"Backtest rows available: "
        f"{len(backtest_data):,}"
    )

    print(
        f"Final untouched 2025/26 rows: "
        f"{len(final_test):,}"
    )

    print(
        "Final test excluded from all training/"
        "validation folds ✓"
    )

    # --------------------------------------------------------
    # FEATURE FAMILIES
    # --------------------------------------------------------

    families = build_family_map(
        all_features
    )

    scenarios = build_scenarios(
        all_features=all_features,
        families=families,
    )

    print(
        "\nBacktest scenarios:"
    )

    for (
        scenario,
        features,
    ) in scenarios.items():

        print(
            f"  {scenario:<22} "
            f"{len(features):>4} features"
        )

    validation_seasons = list(
        range(
            FIRST_VALIDATION_SEASON,
            LAST_VALIDATION_SEASON
            +
            1,
        )
    )

    print(
        "\nRolling validation seasons:"
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

    # --------------------------------------------------------
    # DIRECTORIES
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

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    metrics_rows = []

    prediction_rows = []

    # --------------------------------------------------------
    # FOLDS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RUNNING EXPANDING-WINDOW FOLDS"
    )

    print(
        "=" * 75
    )

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
            + "-" * 75
        )

        print(
            f"VALIDATION SEASON: "
            f"{validation_season}"
        )

        print(
            "-" * 75
        )

        print(
            f"Training rows:   "
            f"{len(train):,}"
        )

        print(
            f"Training seasons:"
            f" {train['season'].min()}-"
            f"{train['season'].max()}"
        )

        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        print(
            f"Validation dates:"
            f" {validation['date'].min().date()} "
            f"to "
            f"{validation['date'].max().date()}"
        )

        y_train = train[
            TARGET_COLUMN
        ]

        y_validation = validation[
            TARGET_COLUMN
        ]

        # ----------------------------------------------------
        # PRIOR
        # ----------------------------------------------------

        prior_probs = prior_probabilities(
            y_train=y_train,
            validation_rows=len(
                validation
            ),
        )

        prior_metrics = evaluate_probabilities(
            y_true=y_validation,
            probabilities=prior_probs,
        )

        print(
            f"Prior Log Loss: "
            f"{prior_metrics['log_loss']:.4f}"
        )

        # ----------------------------------------------------
        # SCENARIOS
        # ----------------------------------------------------

        for (
            scenario_name,
            feature_columns,
        ) in scenarios.items():

            print(
                f"\n  {scenario_name}"
            )

            X_train = train[
                feature_columns
            ]

            X_validation = validation[
                feature_columns
            ]

            model = build_model(
                feature_columns
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

            metrics = evaluate_probabilities(
                y_true=y_validation,
                probabilities=probabilities,
            )

            train_availability = (
                feature_availability(
                    train,
                    feature_columns,
                )
            )

            validation_availability = (
                feature_availability(
                    validation,
                    feature_columns,
                )
            )

            log_loss_improvement = (
                prior_metrics[
                    "log_loss"
                ]
                -
                metrics[
                    "log_loss"
                ]
            )

            metric_row = {
                "validation_season": (
                    validation_season
                ),

                "scenario": (
                    scenario_name
                ),

                "feature_count": len(
                    feature_columns
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

                "validation_first_date": (
                    validation[
                        "date"
                    ]
                    .min()
                    .date()
                    .isoformat()
                ),

                "validation_last_date": (
                    validation[
                        "date"
                    ]
                    .max()
                    .date()
                    .isoformat()
                ),

                "train_feature_availability": (
                    train_availability
                ),

                "validation_feature_availability": (
                    validation_availability
                ),

                "prior_log_loss": (
                    prior_metrics[
                        "log_loss"
                    ]
                ),

                "log_loss_improvement_vs_prior": (
                    log_loss_improvement
                ),
            }

            metric_row.update(
                metrics
            )

            metrics_rows.append(
                metric_row
            )

            predictions = np.argmax(
                probabilities,
                axis=1,
            )

            for row_index, (
                validation_index,
                match_row,
            ) in enumerate(
                validation.iterrows()
            ):

                prediction_rows.append(
                    {
                        "validation_season": (
                            validation_season
                        ),

                        "scenario": (
                            scenario_name
                        ),

                        "match_id": (
                            match_row[
                                "match_id"
                            ]
                        ),

                        "date": (
                            match_row[
                                "date"
                            ]
                        ),

                        "home_team": (
                            match_row[
                                "home_team"
                            ]
                        ),

                        "away_team": (
                            match_row[
                                "away_team"
                            ]
                        ),

                        "actual_result": (
                            match_row[
                                "result"
                            ]
                        ),

                        "prediction": TARGET_NAMES[
                            int(
                                predictions[
                                    row_index
                                ]
                            )
                        ],

                        "prob_H": float(
                            probabilities[
                                row_index,
                                0,
                            ]
                        ),

                        "prob_D": float(
                            probabilities[
                                row_index,
                                1,
                            ]
                        ),

                        "prob_A": float(
                            probabilities[
                                row_index,
                                2,
                            ]
                        ),
                    }
                )

            print(
                f"    Log Loss: "
                f"{metrics['log_loss']:.4f}"
            )

            print(
                f"    Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"    Macro F1: "
                f"{metrics['macro_f1']:.4f}"
            )

            print(
                f"    Draw Recall: "
                f"{metrics['draw_recall']:.4f}"
            )

            print(
                f"    vs Prior: "
                f"{log_loss_improvement:+.4f}"
            )

    # --------------------------------------------------------
    # FOLD RESULTS
    # --------------------------------------------------------

    fold_metrics = pd.DataFrame(
        metrics_rows
    )

    fold_metrics = fold_metrics.sort_values(
        by=[
            "validation_season",
            "scenario",
        ]
    ).reset_index(
        drop=True
    )

    summary = build_summary(
        fold_metrics
    )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ROLLING BACKTEST SUMMARY"
    )

    print(
        "=" * 75
    )

    display_columns = [
        "rank",
        "scenario",
        "feature_count",
        "folds",
        "validation_matches",
        "weighted_accuracy",
        "weighted_macro_f1",
        "weighted_log_loss",
        "log_loss_std",
        "weighted_brier",
        "weighted_draw_recall",
        "mean_log_loss_improvement_vs_prior",
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
    # SEASON-BY-SEASON ALL FEATURES
    # --------------------------------------------------------

    all_features_folds = fold_metrics[
        fold_metrics[
            "scenario"
        ]
        ==
        "all_features"
    ].copy()

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ALL-FEATURES SEASON-BY-SEASON"
    )

    print(
        "=" * 75
    )

    season_columns = [
        "validation_season",
        "training_rows",
        "validation_rows",
        "validation_feature_availability",
        "accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
        "draw_recall",
        "prior_log_loss",
        "log_loss_improvement_vs_prior",
    ]

    print(
        "\n"
        + all_features_folds[
            season_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # FAMILY DELTA TABLE
    # --------------------------------------------------------

    pivot = fold_metrics.pivot(
        index="validation_season",
        columns="scenario",
        values="log_loss",
    )

    if "all_features" in pivot.columns:

        delta_rows = []

        for scenario in [
            "without_transfer",
            "without_domestic",
            "without_squad",
            "without_elo",
            "elo_plus_ucl_form",
        ]:

            if scenario not in pivot.columns:
                continue

            delta = (
                pivot[
                    scenario
                ]
                -
                pivot[
                    "all_features"
                ]
            )

            delta_rows.append(
                {
                    "scenario": scenario,

                    "mean_log_loss_delta_vs_all": float(
                        delta.mean()
                    ),

                    "median_log_loss_delta_vs_all": float(
                        delta.median()
                    ),

                    "seasons_all_features_better": int(
                        (
                            delta
                            >
                            0
                        ).sum()
                    ),

                    "seasons_scenario_better": int(
                        (
                            delta
                            <
                            0
                        ).sum()
                    ),

                    "ties": int(
                        (
                            delta
                            ==
                            0
                        ).sum()
                    ),
                }
            )

        delta_frame = pd.DataFrame(
            delta_rows
        ).sort_values(
            by="mean_log_loss_delta_vs_all",
            ascending=False,
        )

        print(
            "\n"
            + "=" * 75
        )

        print(
            "FEATURE-FAMILY STABILITY VS ALL FEATURES"
        )

        print(
            "=" * 75
        )

        print(
            "\nPositive delta means "
            "all_features had lower Log Loss.\n"
        )

        print(
            delta_frame.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.4f}"
                ),
            )
        )

    else:

        delta_frame = pd.DataFrame()

    # --------------------------------------------------------
    # SAVE FOLD METRICS
    # --------------------------------------------------------

    fold_metrics.to_csv(
        ROLLING_FOLD_METRICS_FILE,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        ROLLING_SUMMARY_FILE,
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
        ROLLING_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # JSON METADATA
    # --------------------------------------------------------

    best = summary.iloc[
        0
    ]

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "backtest_policy": {
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

            "split_type": (
                "expanding_window_by_season"
            ),
        },

        "random_forest_parameters": {
            "n_estimators": (
                N_ESTIMATORS
            ),

            "min_samples_leaf": 3,

            "max_features": "sqrt",

            "class_weight": (
                "balanced_subsample"
            ),

            "random_state": (
                RANDOM_STATE
            ),
        },

        "scenarios": {
            scenario: features
            for (
                scenario,
                features,
            ) in scenarios.items()
        },

        "summary": (
            summary.to_dict(
                orient="records"
            )
        ),

        "best_scenario": (
            best[
                "scenario"
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

        "final_test_protection": (
            "2025/26 was excluded from every "
            "training and validation fold."
        ),
    }

    with ROLLING_JSON_FILE.open(
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
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ROLLING BACKTEST COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        f"\nBest scenario: "
        f"{best['scenario']}"
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
        f"Weighted Macro F1: "
        f"{best['weighted_macro_f1']:.4f}"
    )

    print(
        f"Weighted Draw Recall: "
        f"{best['weighted_draw_recall']:.4f}"
    )

    print(
        "\nFold metrics:"
    )

    print(
        ROLLING_FOLD_METRICS_FILE
    )

    print(
        "\nSummary:"
    )

    print(
        ROLLING_SUMMARY_FILE
    )

    print(
        "\nPredictions:"
    )

    print(
        ROLLING_PREDICTIONS_FILE
    )

    print(
        "\n2025/26 remains untouched ✓"
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    run_rolling_backtest()