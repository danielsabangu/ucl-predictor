from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

ABLATION_METRICS_FILE = (
    METRICS_DIR
    / "feature_family_ablation_metrics.csv"
)

ABLATION_JSON_FILE = (
    MODEL_METADATA_DIR
    / "feature_family_ablation_metrics.json"
)

ABLATION_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "feature_family_ablation_predictions.csv"
)

BEST_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "ablation_best_random_forest.joblib"
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
# MODEL
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 700


# ============================================================
# LOAD MANIFEST
# ============================================================


def load_manifest() -> dict:
    """
    Load the exact approved feature manifest.
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

    feature_columns = manifest.get(
        "feature_columns"
    )

    if not feature_columns:

        raise RuntimeError(
            "Feature manifest contains no "
            "approved model features."
        )

    return manifest


# ============================================================
# LOAD ML DATA
# ============================================================


def load_ml_data(
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Load chronological ML matrix.
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
            "ML feature matrix missing columns: "
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
    Replicate the same family classification used during
    master feature construction.
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
    Group approved features by family.
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
# ABLATION DEFINITIONS
# ============================================================


def build_ablation_scenarios(
    *,
    all_features: list[str],
    families: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Build controlled feature-family experiments.

    No final-test information is used.
    """

    family_sets = {
        name: set(
            features
        )
        for (
            name,
            features,
        ) in families.items()
    }

    all_set = set(
        all_features
    )

    scenarios = {
        # ----------------------------------------------------
        # FULL MODEL
        # ----------------------------------------------------

        "all_features": (
            all_features
        ),

        # ----------------------------------------------------
        # LEAVE-ONE-FAMILY-OUT
        # ----------------------------------------------------

        "without_transfer": sorted(
            all_set
            -
            family_sets[
                "transfer"
            ]
        ),

        "without_squad": sorted(
            all_set
            -
            family_sets[
                "squad"
            ]
        ),

        "without_domestic": sorted(
            all_set
            -
            family_sets[
                "domestic"
            ]
        ),

        "without_ucl_form": sorted(
            all_set
            -
            family_sets[
                "ucl_form"
            ]
        ),

        "without_elo": sorted(
            all_set
            -
            family_sets[
                "elo"
            ]
        ),

        # ----------------------------------------------------
        # FAMILY COMBINATIONS
        # ----------------------------------------------------

        "elo_only": sorted(
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

        "performance_core": sorted(
            family_sets[
                "elo"
            ]
            |
            family_sets[
                "ucl_form"
            ]
            |
            family_sets[
                "domestic"
            ]
        ),

        "squad_plus_transfer": sorted(
            family_sets[
                "squad"
            ]
            |
            family_sets[
                "transfer"
            ]
        ),

        "performance_plus_squad": sorted(
            family_sets[
                "elo"
            ]
            |
            family_sets[
                "ucl_form"
            ]
            |
            family_sets[
                "domestic"
            ]
            |
            family_sets[
                "squad"
            ]
        ),

        "performance_plus_transfer": sorted(
            family_sets[
                "elo"
            ]
            |
            family_sets[
                "ucl_form"
            ]
            |
            family_sets[
                "domestic"
            ]
            |
            family_sets[
                "transfer"
            ]
        ),
    }

    return scenarios


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
    Same split as baseline modeling.

    Train:
        <= 2023

    Validation:
        2024

    Final test:
        2025

    Final test remains untouched.
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

    final_test = dataframe[
        dataframe[
            "season"
        ]
        ==
        FINAL_TEST_SEASON
    ].copy()

    if (
        train.empty
        or validation.empty
        or final_test.empty
    ):

        raise RuntimeError(
            "One or more chronological splits "
            "are empty."
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
        final_test[
            "date"
        ].min()
    ):

        raise RuntimeError(
            "Validation/test chronology overlap."
        )

    return (
        train,
        validation,
        final_test,
    )


# ============================================================
# PREPROCESSOR
# ============================================================


def build_preprocessor(
    feature_columns: list[str],
) -> ColumnTransformer:
    """
    Median imputation plus missing indicators.

    Fitted inside the training pipeline only.
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
# RANDOM FOREST
# ============================================================


def build_random_forest(
    feature_columns: list[str],
) -> Pipeline:
    """
    Same primary Random Forest configuration as the
    successful baseline.
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
    Multiclass Brier probability score.
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
# PROBABILITY ALIGNMENT
# ============================================================


def align_probabilities(
    *,
    model: Pipeline,
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Force class probability order:

        H = 0
        D = 1
        A = 2
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
    scenario_name: str,
    model: Pipeline,
    feature_count: int,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    dict,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Evaluate one feature-family scenario.
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

    (
        precision,
        recall,
        class_f1,
        support,
    ) = precision_recall_fscore_support(
        y_validation,
        predictions,
        labels=TARGET_LABELS,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_validation,
        predictions,
        labels=TARGET_LABELS,
    )

    metrics = {
        "scenario": scenario_name,

        "feature_count": int(
            feature_count
        ),

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
        probabilities,
        predictions,
        confusion_frame,
    )


# ============================================================
# MAIN
# ============================================================


def run_feature_ablations() -> None:
    """
    Compare Random Forest performance across feature families.

    2025/26 is not evaluated.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - FEATURE FAMILY ABLATION STUDY"
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

    print(
        f"\nApproved features: "
        f"{len(all_features):,}"
    )

    # --------------------------------------------------------
    # FAMILY MAP
    # --------------------------------------------------------

    families = build_family_map(
        all_features
    )

    print(
        "\nFeature families:"
    )

    for family in [
        "elo",
        "ucl_form",
        "domestic",
        "squad",
        "transfer",
    ]:

        print(
            f"  {family:<10} "
            f"{len(families[family]):>4}"
        )

    if sum(
        len(
            features
        )
        for features in families.values()
    ) != len(
        all_features
    ):

        raise RuntimeError(
            "Feature family assignment "
            "does not cover all features."
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    dataframe = load_ml_data(
        all_features
    )

    (
        train,
        validation,
        final_test,
    ) = chronological_split(
        dataframe
    )

    print(
        "\nChronological split:"
    )

    print(
        f"  Train:      "
        f"{len(train):,}"
    )

    print(
        f"  Validation: "
        f"{len(validation):,}"
    )

    print(
        f"  Final test: "
        f"{len(final_test):,} "
        f"(untouched)"
    )

    # --------------------------------------------------------
    # SCENARIOS
    # --------------------------------------------------------

    scenarios = build_ablation_scenarios(
        all_features=all_features,
        families=families,
    )

    print(
        f"\nAblation scenarios: "
        f"{len(scenarios):,}"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
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

    TRAINED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # BASE PREDICTION FRAME
    # --------------------------------------------------------

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

    y_train = train[
        TARGET_COLUMN
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    metrics_rows = []

    confusion_matrices = {}

    trained_models = {}

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RUNNING ABLATIONS"
    )

    print(
        "=" * 75
    )

    for (
        scenario_name,
        feature_columns,
    ) in scenarios.items():

        if not feature_columns:

            raise RuntimeError(
                f"Ablation scenario "
                f"{scenario_name} "
                f"contains no features."
            )

        print(
            f"\nScenario: "
            f"{scenario_name}"
        )

        print(
            f"  Features: "
            f"{len(feature_columns):,}"
        )

        X_train = train[
            feature_columns
        ]

        X_validation = validation[
            feature_columns
        ]

        model = build_random_forest(
            feature_columns
        )

        model.fit(
            X_train,
            y_train,
        )

        (
            metrics,
            probabilities,
            predictions,
            confusion,
        ) = evaluate(
            scenario_name=scenario_name,
            model=model,
            feature_count=len(
                feature_columns
            ),
            X_validation=X_validation,
            y_validation=y_validation,
        )

        metrics_rows.append(
            metrics
        )

        confusion_matrices[
            scenario_name
        ] = confusion

        trained_models[
            scenario_name
        ] = model

        prediction_frame[
            f"{scenario_name}_prob_H"
        ] = probabilities[
            :,
            0
        ]

        prediction_frame[
            f"{scenario_name}_prob_D"
        ] = probabilities[
            :,
            1
        ]

        prediction_frame[
            f"{scenario_name}_prob_A"
        ] = probabilities[
            :,
            2
        ]

        prediction_frame[
            f"{scenario_name}_prediction"
        ] = [
            TARGET_NAMES[
                int(
                    value
                )
            ]
            for value in predictions
        ]

        print(
            f"  Accuracy:    "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Log Loss:    "
            f"{metrics['log_loss']:.4f}"
        )

        print(
            f"  Brier:       "
            f"{metrics['brier_score']:.4f}"
        )

        print(
            f"  Macro F1:    "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"  Draw Recall: "
            f"{metrics['draw_recall']:.4f}"
        )

    # --------------------------------------------------------
    # RANK
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
        "rank"
    ] = range(
        1,
        len(
            metrics_frame
        )
        + 1,
    )

    # --------------------------------------------------------
    # FULL MODEL DELTAS
    # --------------------------------------------------------

    full_row = metrics_frame[
        metrics_frame[
            "scenario"
        ]
        ==
        "all_features"
    ]

    if full_row.empty:

        raise RuntimeError(
            "all_features scenario missing."
        )

    full_log_loss = float(
        full_row.iloc[
            0
        ][
            "log_loss"
        ]
    )

    full_brier = float(
        full_row.iloc[
            0
        ][
            "brier_score"
        ]
    )

    full_accuracy = float(
        full_row.iloc[
            0
        ][
            "accuracy"
        ]
    )

    metrics_frame[
        "log_loss_delta_vs_all"
    ] = (
        metrics_frame[
            "log_loss"
        ]
        -
        full_log_loss
    )

    metrics_frame[
        "brier_delta_vs_all"
    ] = (
        metrics_frame[
            "brier_score"
        ]
        -
        full_brier
    )

    metrics_frame[
        "accuracy_delta_vs_all"
    ] = (
        metrics_frame[
            "accuracy"
        ]
        -
        full_accuracy
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FEATURE ABLATION RESULTS"
    )

    print(
        "=" * 75
    )

    display_columns = [
        "rank",
        "scenario",
        "feature_count",
        "accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
        "draw_recall",
        "predicted_draw",
        "log_loss_delta_vs_all",
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
    # PRINT TOP CONFUSION MATRICES
    # --------------------------------------------------------

    top_scenarios = (
        metrics_frame[
            "scenario"
        ]
        .head(
            5
        )
        .tolist()
    )

    for scenario_name in top_scenarios:

        print(
            "\n"
            + "-" * 75
        )

        print(
            f"CONFUSION MATRIX: "
            f"{scenario_name}"
        )

        print(
            "-" * 75
        )

        print(
            confusion_matrices[
                scenario_name
            ].to_string()
        )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_scenario = metrics_frame.iloc[
        0
    ][
        "scenario"
    ]

    best_model = trained_models[
        best_scenario
    ]

    joblib.dump(
        best_model,
        BEST_MODEL_FILE,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    metrics_frame.to_csv(
        ABLATION_METRICS_FILE,
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
        ABLATION_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    scenario_features = {
        scenario_name: feature_columns
        for (
            scenario_name,
            feature_columns,
        ) in scenarios.items()
    }

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "selection_metric": (
            "validation log loss"
        ),

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

        "feature_family_counts": {
            family: len(
                features
            )
            for (
                family,
                features,
            ) in families.items()
        },

        "scenario_features": (
            scenario_features
        ),

        "results": (
            metrics_frame.to_dict(
                orient="records"
            )
        ),

        "best_scenario": (
            best_scenario
        ),
    }

    with ABLATION_JSON_FILE.open(
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

    best_row = metrics_frame.iloc[
        0
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ABLATION STUDY SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nBest scenario: "
        f"{best_row['scenario']}"
    )

    print(
        f"Features: "
        f"{int(best_row['feature_count']):,}"
    )

    print(
        f"Validation Log Loss: "
        f"{best_row['log_loss']:.4f}"
    )

    print(
        f"Validation Brier: "
        f"{best_row['brier_score']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{best_row['accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_row['macro_f1']:.4f}"
    )

    print(
        f"Draw Recall: "
        f"{best_row['draw_recall']:.4f}"
    )

    print(
        "\nMetrics saved to:"
    )

    print(
        ABLATION_METRICS_FILE
    )

    print(
        "\nValidation predictions saved to:"
    )

    print(
        ABLATION_PREDICTIONS_FILE
    )

    print(
        "\nBest validation model saved to:"
    )

    print(
        BEST_MODEL_FILE
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FEATURE FAMILY ABLATION COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\n2025/26 remains untouched ✓"
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    run_feature_ablations()