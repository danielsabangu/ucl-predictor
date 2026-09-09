from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_METADATA_DIR,
    TRAINED_MODELS_DIR,
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

FINAL_EVALUATION_FILE = (
    MODEL_METADATA_DIR
    / "final_model_evaluation.json"
)

PRODUCTION_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "production_catboost.joblib"
)

PRODUCTION_METADATA_FILE = (
    MODEL_METADATA_DIR
    / "production_model.json"
)


# ============================================================
# LOCKED SPECIFICATION
# ============================================================

EXPECTED_ROWS = 2089

EXPECTED_MASTER_FEATURES = 411

EXPECTED_PRODUCTION_FEATURES = 290

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================


def require_file(
    path,
    description: str,
) -> None:
    """
    Fail early when a required project artifact is missing.
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


# ============================================================
# FEATURE FAMILY
# ============================================================


def determine_feature_family(
    feature: str,
) -> str:
    """
    Use the exact same feature-family policy established
    during model development.
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


def load_feature_manifest() -> dict:
    """
    Load leakage-audited master feature manifest.
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

    if len(
        features
    ) != EXPECTED_MASTER_FEATURES:

        raise RuntimeError(
            f"Expected {EXPECTED_MASTER_FEATURES} "
            f"master features, found "
            f"{len(features)}."
        )

    return manifest


# ============================================================
# VERIFY FINAL EVALUATION EXISTS
# ============================================================


def verify_final_evaluation() -> dict:
    """
    Production training is only permitted after the
    one-time held-out test has been completed.

    This prevents accidentally skipping final evaluation.
    """

    require_file(
        FINAL_EVALUATION_FILE,
        "Final model evaluation metadata",
    )

    with FINAL_EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    final_test = metadata.get(
        "final_test"
    )

    if not final_test:

        raise RuntimeError(
            "Final-test metadata missing."
        )

    if (
        final_test.get(
            "season"
        )
        !=
        2025
    ):

        raise RuntimeError(
            "Expected completed 2025/26 "
            "final evaluation."
        )

    return metadata


# ============================================================
# FINAL FEATURE POLICY
# ============================================================


def build_production_features(
    master_features: list[str],
) -> tuple[
    list[str],
    list[str],
]:
    """
    Locked production feature set:

        Elo
        UCL form
        domestic context
        squad/player strength

    Historical transfer features remain stored in the
    project but are excluded from the primary model.
    """

    production_features = []

    transfer_features = []

    for feature in master_features:

        if (
            determine_feature_family(
                feature
            )
            ==
            "transfer"
        ):

            transfer_features.append(
                feature
            )

        else:

            production_features.append(
                feature
            )

    if len(
        production_features
    ) != EXPECTED_PRODUCTION_FEATURES:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PRODUCTION_FEATURES} "
            f"production features, found "
            f"{len(production_features)}."
        )

    return (
        production_features,
        transfer_features,
    )


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================


def load_training_data(
    master_features: list[str],
) -> pd.DataFrame:
    """
    Load all historical UCL matches now available.

    After final evaluation, all historical seasons can be
    used to train the frozen deployment model.
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
        "target",
    }

    required.update(
        master_features
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
        "target"
    ] = pd.to_numeric(
        dataframe[
            "target"
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

    if len(
        dataframe
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} "
            f"historical matches, found "
            f"{len(dataframe):,}."
        )

    if dataframe[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate historical match IDs detected."
        )

    if not dataframe[
        "date"
    ].is_monotonic_increasing:

        raise RuntimeError(
            "Historical training data "
            "is not chronological."
        )

    target_values = set(
        dataframe[
            "target"
        ].unique()
    )

    if target_values != {
        0,
        1,
        2,
    }:

        raise RuntimeError(
            f"Unexpected target labels: "
            f"{target_values}"
        )

    return dataframe


# ============================================================
# BUILD LOCKED CATBOOST
# ============================================================


def build_production_model() -> Pipeline:
    """
    Exact model specification chosen before the final test.

    No parameter has been altered based on 2025/26 results.
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
# MAIN
# ============================================================


def train_production_model() -> None:
    """
    Train the frozen CatBoost specification on every
    completed historical UCL match.

    This model is for 2026/27 forward prediction.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - PRODUCTION MODEL TRAINING"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # VERIFY DEVELOPMENT IS COMPLETE
    # --------------------------------------------------------

    final_evaluation = verify_final_evaluation()

    print(
        "\nFinal held-out evaluation verified ✓"
    )

    print(
        "2025/26 is now historical training data."
    )

    # --------------------------------------------------------
    # MANIFEST / FEATURES
    # --------------------------------------------------------

    manifest = load_feature_manifest()

    master_features = manifest[
        "feature_columns"
    ]

    (
        feature_columns,
        transfer_features,
    ) = build_production_features(
        master_features
    )

    print(
        f"\nMaster features: "
        f"{len(master_features):,}"
    )

    print(
        f"Transfer features excluded: "
        f"{len(transfer_features):,}"
    )

    print(
        f"Production features: "
        f"{len(feature_columns):,}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    dataframe = load_training_data(
        master_features
    )

    print(
        f"\nHistorical matches: "
        f"{len(dataframe):,}"
    )

    print(
        f"Training seasons: "
        f"{dataframe['season'].min()} "
        f"to "
        f"{dataframe['season'].max()}"
    )

    print(
        f"Training dates: "
        f"{dataframe['date'].min().date()} "
        f"to "
        f"{dataframe['date'].max().date()}"
    )

    availability = (
        1
        -
        dataframe[
            feature_columns
        ].isna().mean().mean()
    )

    print(
        f"Feature availability: "
        f"{availability * 100:.2f}%"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    X = dataframe[
        feature_columns
    ]

    y = dataframe[
        "target"
    ]

    print(
        "\nTraining production CatBoost..."
    )

    model = build_production_model()

    model.fit(
        X,
        y,
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    classes = list(
        model.classes_
    )

    if classes != [
        0,
        1,
        2,
    ]:

        raise RuntimeError(
            f"Unexpected model classes: "
            f"{classes}"
        )

    print(
        "Model classes H/D/A = 0/1/2 ✓"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    TRAINED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        PRODUCTION_MODEL_FILE,
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "model_version": (
            "ucl_predictor_v1"
        ),

        "model_role": (
            "2026/27 forward H/D/A probability engine"
        ),

        "model": (
            "CatBoostClassifier"
        ),

        "training_rows": int(
            len(
                dataframe
            )
        ),

        "training_first_season": int(
            dataframe[
                "season"
            ].min()
        ),

        "training_last_season": int(
            dataframe[
                "season"
            ].max()
        ),

        "training_first_date": (
            dataframe[
                "date"
            ]
            .min()
            .date()
            .isoformat()
        ),

        "training_last_date": (
            dataframe[
                "date"
            ]
            .max()
            .date()
            .isoformat()
        ),

        "master_feature_count": (
            len(
                master_features
            )
        ),

        "production_feature_count": (
            len(
                feature_columns
            )
        ),

        "feature_columns": (
            feature_columns
        ),

        "excluded_transfer_feature_count": (
            len(
                transfer_features
            )
        ),

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

        "development_evidence": {
            "rolling_oof_catboost_log_loss": (
                0.9166
            ),

            "rolling_oof_catboost_brier": (
                0.5390
            ),

            "final_2025_26_log_loss": (
                final_evaluation[
                    "final_test"
                ][
                    "log_loss"
                ]
            ),

            "final_2025_26_brier": (
                final_evaluation[
                    "final_test"
                ][
                    "brier_score"
                ]
            ),

            "final_2025_26_accuracy": (
                final_evaluation[
                    "final_test"
                ][
                    "accuracy"
                ]
            ),
        },

        "locked_decisions": {
            "transfer_features_excluded": True,
            "probability_calibration": False,
            "ensemble": False,
            "dedicated_draw_layer": False,
        },

        "important_note": (
            "This model was trained after the final "
            "2025/26 evaluation using the exact frozen "
            "specification selected before that evaluation. "
            "No post-test tuning was performed."
        ),
    }

    with PRODUCTION_METADATA_FILE.open(
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
        "PRODUCTION MODEL TRAINED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nProduction model:"
    )

    print(
        PRODUCTION_MODEL_FILE
    )

    print(
        "\nProduction metadata:"
    )

    print(
        PRODUCTION_METADATA_FILE
    )

    print(
        "\nModel version: ucl_predictor_v1"
    )

    print(
        "\nNext phase:"
    )

    print(
        "Refresh current 2026/27 fixtures, "
        "domestic form, squads and availability."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    train_production_model()