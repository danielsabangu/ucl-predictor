from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    BASE_DIR,
    PREDICTIONS_DIR,
)


# ============================================================
# PATHS
# ============================================================

SOURCE_PREDICTIONS = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_complete.csv"
)

SOURCE_STATUS = (
    PREDICTIONS_DIR
    / "live_pipeline_status.json"
)

APP_DATA_DIR = (
    BASE_DIR
    / "data"
    / "app"
)

APP_PREDICTIONS = (
    APP_DATA_DIR
    / "current_predictions.csv"
)

APP_STATUS = (
    APP_DATA_DIR
    / "pipeline_status.json"
)

APP_METADATA = (
    APP_DATA_DIR
    / "snapshot_metadata.json"
)


# ============================================================
# REQUIRED PUBLIC COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "kickoff_utc",
    "home_team",
    "away_team",

    "home_probability",
    "draw_probability",
    "away_probability",

    "predicted_result",
    "predicted_selection",
    "max_probability",

    "confidence",
    "confidence_percent",
    "confidence_label",
    "confidence_caps",

    "model_top_two_margin",
    "model_decisiveness_profile",

    "elo_features_available",
    "ucl_form_features_available",
    "domestic_features_available",
    "squad_features_available",

    "overall_features_available",
    "overall_feature_availability",

    "predicted_home_goals",
    "predicted_away_goals",
    "predicted_scoreline",

    "predicted_scoreline_probability",

    "global_modal_scoreline",
    "global_modal_result",

    "score_home_lambda",
    "score_away_lambda",
    "score_total_lambda",

    "score_bridge_max_error_pp",
    "score_bridge_quality",
]


# ============================================================
# PREDICTION VALIDATION
# ============================================================


def validate_predictions(
    dataframe: pd.DataFrame,
) -> None:

    if dataframe.empty:

        raise RuntimeError(
            "Production prediction artifact is empty."
        )

    missing = (
        set(REQUIRED_COLUMNS)
        -
        set(dataframe.columns)
    )

    if missing:

        raise RuntimeError(
            "Production prediction artifact is missing "
            "required public columns:\n"
            f"{sorted(missing)}"
        )

    dataframe[
        "kickoff_utc"
    ] = pd.to_datetime(
        dataframe[
            "kickoff_utc"
        ],
        utc=True,
        errors="raise",
    )

    probability_columns = [
        "home_probability",
        "draw_probability",
        "away_probability",
    ]

    probabilities = dataframe[
        probability_columns
    ].astype(
        float
    )

    if (
        probabilities
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Public prediction probabilities contain NaNs."
        )

    if (
        (
            probabilities
            <
            0.0
        )
        .any()
        .any()
    ):

        raise RuntimeError(
            "Public prediction probabilities contain "
            "values below 0."
        )

    if (
        (
            probabilities
            >
            1.0
        )
        .any()
        .any()
    ):

        raise RuntimeError(
            "Public prediction probabilities contain "
            "values above 1."
        )

    sums = (
        probabilities
        .sum(
            axis=1
        )
    )

    max_error = float(
        (
            sums
            -
            1.0
        )
        .abs()
        .max()
    )

    if max_error > 1e-6:

        raise RuntimeError(
            "Public H/D/A probabilities do not sum to 1. "
            f"Maximum error: {max_error}"
        )

    confidence = dataframe[
        "confidence_percent"
    ].astype(
        float
    )

    if (
        (
            confidence
            <
            0.0
        ).any()
        or
        (
            confidence
            >
            100.0
        ).any()
    ):

        raise RuntimeError(
            "Evidence confidence is outside 0-100."
        )

    if dataframe[
        "predicted_scoreline"
    ].isna().any():

        raise RuntimeError(
            "Predicted scoreline is missing."
        )


# ============================================================
# SAFE STATUS
# ============================================================


def build_public_status(
    source_payload: dict,
) -> dict:

    return {
        "status": (
            str(
                source_payload.get(
                    "status",
                    "UNKNOWN",
                )
            )
            .upper()
        ),

        "production_ready": (
            bool(
                source_payload.get(
                    "production_ready",
                    False,
                )
            )
        ),

        "run_id": (
            source_payload.get(
                "run_id"
            )
        ),

        "started_at_utc": (
            source_payload.get(
                "started_at_utc"
            )
        ),

        "finished_at_utc": (
            source_payload.get(
                "finished_at_utc"
            )
        ),

        "duration_seconds": (
            source_payload.get(
                "duration_seconds"
            )
        ),

        "public_snapshot": True,
    }


# ============================================================
# PACKAGE
# ============================================================


def package_app_snapshot() -> None:

    print(
        "=" * 78
    )

    print(
        "UCL PREDICTOR - PACKAGE PUBLIC APP SNAPSHOT"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # SOURCE CHECKS
    # --------------------------------------------------------

    if not SOURCE_PREDICTIONS.exists():

        raise FileNotFoundError(
            "Production prediction artifact not found:\n"
            f"{SOURCE_PREDICTIONS}"
        )

    if not SOURCE_STATUS.exists():

        raise FileNotFoundError(
            "Production pipeline status not found:\n"
            f"{SOURCE_STATUS}"
        )

    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.read_csv(
        SOURCE_PREDICTIONS,
        low_memory=False,
    )

    validate_predictions(
        predictions
    )

    # Only publish columns the Streamlit app actually needs.
    public_predictions = predictions[
        REQUIRED_COLUMNS
    ].copy()

    public_predictions[
        "kickoff_utc"
    ] = pd.to_datetime(
        public_predictions[
            "kickoff_utc"
        ],
        utc=True,
        errors="raise",
    )

    public_predictions = (
        public_predictions
        .sort_values(
            [
                "kickoff_utc",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    with SOURCE_STATUS.open(
        "r",
        encoding="utf-8",
    ) as file:

        source_status = json.load(
            file
        )

    public_status = (
        build_public_status(
            source_status
        )
    )

    if (
        public_status[
            "status"
        ]
        !=
        "SUCCESS"
    ):

        raise RuntimeError(
            "Latest production pipeline status is not SUCCESS."
        )

    if not public_status[
        "production_ready"
    ]:

        raise RuntimeError(
            "Latest production pipeline is not marked "
            "production ready."
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    APP_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # WRITE SNAPSHOT
    # --------------------------------------------------------

    public_predictions.to_csv(
        APP_PREDICTIONS,
        index=False,
    )

    with APP_STATUS.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            public_status,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    kickoff_values = (
        public_predictions[
            "kickoff_utc"
        ]
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

        "fixture_count": (
            int(
                len(
                    public_predictions
                )
            )
        ),

        "first_kickoff_utc": (
            kickoff_values
            .min()
            .isoformat()
        ),

        "last_kickoff_utc": (
            kickoff_values
            .max()
            .isoformat()
        ),

        "prediction_columns": (
            int(
                len(
                    public_predictions.columns
                )
            )
        ),

        "source_run_id": (
            public_status.get(
                "run_id"
            )
        ),

        "probability_model": (
            "Frozen CatBoost"
        ),

        "scoreline_layer": (
            "Auxiliary Poisson bridge"
        ),

        "confidence_layer": (
            "Independent evidence-quality score"
        ),
    }

    with APP_METADATA.open(
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
    # FINAL VALIDATION
    # --------------------------------------------------------

    packaged_predictions = pd.read_csv(
        APP_PREDICTIONS,
        low_memory=False,
    )

    validate_predictions(
        packaged_predictions
    )

    with APP_STATUS.open(
        "r",
        encoding="utf-8",
    ) as file:

        packaged_status = json.load(
            file
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PUBLIC SNAPSHOT QUALITY CHECK"
    )

    print(
        "=" * 78
    )

    print(
        f"\nFixtures: "
        f"{len(packaged_predictions)}"
    )

    print(
        f"Public columns: "
        f"{len(packaged_predictions.columns)}"
    )

    print(
        f"Pipeline status: "
        f"{packaged_status['status']}"
    )

    print(
        f"Production ready: "
        f"{packaged_status['production_ready']}"
    )

    print(
        f"Run ID: "
        f"{packaged_status['run_id']}"
    )

    print(
        "\nPrediction snapshot:"
    )

    print(
        APP_PREDICTIONS
    )

    print(
        "\nPublic status:"
    )

    print(
        APP_STATUS
    )

    print(
        "\nSnapshot metadata:"
    )

    print(
        APP_METADATA
    )

    print(
        "\nSensitive production paths copied: NO ✓"
    )

    print(
        "Raw datasets copied: NO ✓"
    )

    print(
        "Model artifact copied: NO ✓"
    )

    print(
        "API credentials copied: NO ✓"
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PUBLIC APP SNAPSHOT READY ✓"
    )

    print(
        "=" * 78
    )


def main() -> None:

    package_app_snapshot()


if __name__ == "__main__":

    main()