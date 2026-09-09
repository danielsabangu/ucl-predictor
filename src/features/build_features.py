from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_METADATA_DIR,
)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_transfer_impact.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ml_features.csv"
)

FEATURE_MANIFEST_FILE = (
    MODEL_METADATA_DIR
    / "ml_feature_manifest.json"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_ROWS = 2089

TARGET_COLUMN = "result"

TARGET_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}


# ============================================================
# NON-MODEL METADATA
# ============================================================

METADATA_COLUMNS = [
    "match_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "result",
]


# ============================================================
# ABSOLUTE LEAKAGE BLOCKLIST
# ============================================================
#
# These fields describe the current match outcome itself or
# are post-match values.
#
# They must NEVER become predictors.
# ============================================================

EXACT_BLOCKED_COLUMNS = {
    # --------------------------------------------------------
    # CURRENT MATCH TARGET / OUTCOME
    # --------------------------------------------------------

    "result",
    "home_goals",
    "away_goals",
    "scoreline",

    # --------------------------------------------------------
    # IDENTIFIERS / DESCRIPTIVE METADATA
    # --------------------------------------------------------

    "match_id",
    "source_match_id",
    "date",
    "day",
    "season",
    "home_team",
    "away_team",
    "competition",
    "stage",
    "status",
    "source",
}


# ============================================================
# BLOCKED COLUMN PATTERNS
# ============================================================
#
# These patterns identify audit-only or explicitly post-match
# fields created during feature engineering.
# ============================================================

BLOCKED_SUBSTRINGS = [
    "_post",
    "latest_player_evidence_date",
    "latest_valuation_date_used",
    "transfer_latest_date_used",
]


# ============================================================
# APPROVED FEATURE FAMILIES
# ============================================================
#
# The final feature selection still requires numeric dtype,
# but every model feature should belong to one of these
# engineered pre-match families.
# ============================================================

FEATURE_FAMILY_MARKERS = {
    "elo": [
        "elo",
    ],

    "ucl_form": [
        "form",
        "ucl_",
        "momentum",
        "experience",
    ],

    "domestic": [
        "domestic",
        "known_matches",
        "known_match",
    ],

    "squad": [
        "squad",
    ],

    "transfer": [
        "transfer",
    ],
}


# ============================================================
# FILE CHECK
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:
    """
    Fail early if required input is unavailable.
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
# LOAD
# ============================================================


def load_feature_history() -> pd.DataFrame:
    """
    Load the complete historical engineered feature table.
    """

    print(
        "\nLoading engineered UCL feature history:"
    )

    print(
        INPUT_FILE
    )

    require_file(
        INPUT_FILE,
        "Transfer-impact feature dataset",
    )

    dataframe = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        TARGET_COLUMN,
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Input dataset missing required columns: "
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

    dataframe = dataframe.sort_values(
        by=[
            "date",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Rows loaded: "
        f"{len(dataframe):,}"
    )

    print(
        f"Columns loaded: "
        f"{len(dataframe.columns):,}"
    )

    return dataframe


# ============================================================
# TARGET
# ============================================================


def validate_target(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate H / D / A target labels.
    """

    values = set(
        dataframe[
            TARGET_COLUMN
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    expected = set(
        TARGET_MAP
    )

    if values != expected:

        raise RuntimeError(
            f"Unexpected result labels: "
            f"{sorted(values)}"
        )

    if dataframe[
        TARGET_COLUMN
    ].isna().any():

        raise RuntimeError(
            "Missing target labels detected."
        )


# ============================================================
# BLOCK RULES
# ============================================================


def is_blocked_column(
    column: str,
) -> bool:
    """
    Determine whether a column must never be a predictor.
    """

    if column in EXACT_BLOCKED_COLUMNS:

        return True

    lowered = column.lower()

    for pattern in BLOCKED_SUBSTRINGS:

        if pattern in lowered:

            return True

    return False


# ============================================================
# FEATURE FAMILY
# ============================================================


def determine_feature_family(
    column: str,
) -> str:
    """
    Assign feature to a high-level engineering family.
    """

    lowered = column.lower()

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
# SELECT MODEL FEATURES
# ============================================================


def select_model_features(
    dataframe: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
]:
    """
    Select numeric, pre-match features.

    No imputation occurs here.

    Missing values must later be handled inside each
    chronological training pipeline so imputation statistics
    are learned from training data only.
    """

    selected = []

    rejected = []

    for column in dataframe.columns:

        # ----------------------------------------------------
        # ABSOLUTE BLOCK
        # ----------------------------------------------------

        if is_blocked_column(
            column
        ):

            rejected.append(
                column
            )

            continue

        # ----------------------------------------------------
        # NUMERIC ONLY
        # ----------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            dataframe[
                column
            ]
        ):

            rejected.append(
                column
            )

            continue

        selected.append(
            column
        )

    return (
        selected,
        rejected,
    )


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================


def remove_constant_features(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    list[str],
    list[str],
]:
    """
    Remove columns containing <=1 unique non-null value.

    Constant variables cannot help prediction.
    """

    retained = []

    removed = []

    for column in feature_columns:

        unique_count = dataframe[
            column
        ].nunique(
            dropna=True
        )

        if unique_count <= 1:

            removed.append(
                column
            )

        else:

            retained.append(
                column
            )

    return (
        retained,
        removed,
    )


# ============================================================
# FEATURE QUALITY
# ============================================================


def build_feature_audit(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Produce feature-level completeness statistics.
    """

    rows = []

    total_rows = len(
        dataframe
    )

    for column in feature_columns:

        series = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )

        missing_count = int(
            series.isna().sum()
        )

        non_missing = series.dropna()

        rows.append(
            {
                "feature": column,

                "family": (
                    determine_feature_family(
                        column
                    )
                ),

                "dtype": str(
                    dataframe[
                        column
                    ].dtype
                ),

                "missing_count": (
                    missing_count
                ),

                "missing_percent": (
                    missing_count
                    /
                    total_rows
                    *
                    100
                ),

                "non_missing_count": int(
                    len(
                        non_missing
                    )
                ),

                "unique_values": int(
                    non_missing.nunique()
                ),

                "minimum": (
                    float(
                        non_missing.min()
                    )
                    if not non_missing.empty
                    else None
                ),

                "maximum": (
                    float(
                        non_missing.max()
                    )
                    if not non_missing.empty
                    else None
                ),

                "mean": (
                    float(
                        non_missing.mean()
                    )
                    if not non_missing.empty
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# INFINITY CHECK
# ============================================================


def check_infinite_values(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Infinity must never reach model training.
    """

    numeric = dataframe[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    infinity_mask = np.isinf(
        numeric.to_numpy(
            dtype=float
        )
    )

    infinity_count = int(
        infinity_mask.sum()
    )

    print(
        f"Infinite feature values: "
        f"{infinity_count:,}"
    )

    if infinity_count > 0:

        raise RuntimeError(
            "Infinite feature values detected."
        )


# ============================================================
# LEAKAGE AUDIT
# ============================================================


def leakage_audit(
    feature_columns: list[str],
) -> None:
    """
    Final explicit leakage check on selected feature names.
    """

    forbidden_exact = (
        set(
            feature_columns
        )
        &
        EXACT_BLOCKED_COLUMNS
    )

    if forbidden_exact:

        raise RuntimeError(
            "Blocked target/outcome columns entered "
            f"feature matrix: "
            f"{sorted(forbidden_exact)}"
        )

    forbidden_patterns = []

    for column in feature_columns:

        lowered = column.lower()

        for pattern in BLOCKED_SUBSTRINGS:

            if pattern in lowered:

                forbidden_patterns.append(
                    column
                )

                break

    if forbidden_patterns:

        raise RuntimeError(
            "Audit/post-match columns entered feature "
            f"matrix: {sorted(forbidden_patterns)}"
        )


# ============================================================
# BUILD OUTPUT
# ============================================================


def build_ml_dataframe(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Build chronological modeling table.

    Metadata and target remain available for splitting,
    evaluation and auditing.

    Only columns listed in the manifest may be supplied to
    model.fit().
    """

    output = pd.DataFrame(
        {
            "match_id": (
                dataframe[
                    "match_id"
                ]
            ),

            "date": (
                dataframe[
                    "date"
                ]
            ),

            "season": (
                dataframe[
                    "season"
                ]
            ),

            "home_team": (
                dataframe[
                    "home_team"
                ]
            ),

            "away_team": (
                dataframe[
                    "away_team"
                ]
            ),

            "result": (
                dataframe[
                    "result"
                ]
            ),

            "target": (
                dataframe[
                    "result"
                ].map(
                    TARGET_MAP
                )
            ),
        }
    )

    for column in feature_columns:

        output[
            column
        ] = dataframe[
            column
        ]

    return output


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    source: pd.DataFrame,
    output: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Validate the consolidated model dataset.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ML FEATURE MATRIX DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # ROWS
    # --------------------------------------------------------

    if len(
        output
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(output):,}."
        )

    print(
        f"\nMatch rows: "
        f"{len(output):,} ✓"
    )

    # --------------------------------------------------------
    # MATCH ID
    # --------------------------------------------------------

    if output[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate match IDs in ML dataset."
        )

    print(
        "Unique match IDs: valid ✓"
    )

    # --------------------------------------------------------
    # CHRONOLOGY
    # --------------------------------------------------------

    if not output[
        "date"
    ].is_monotonic_increasing:

        raise RuntimeError(
            "ML feature matrix is not chronological."
        )

    print(
        "Chronological ordering: valid ✓"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if output[
        "target"
    ].isna().any():

        raise RuntimeError(
            "Target encoding produced missing labels."
        )

    target_values = set(
        output[
            "target"
        ].unique()
    )

    if target_values != {
        0,
        1,
        2,
    }:

        raise RuntimeError(
            f"Invalid target values: "
            f"{target_values}"
        )

    print(
        "Target encoding H/D/A -> 0/1/2: valid ✓"
    )

    # --------------------------------------------------------
    # FEATURE COUNT
    # --------------------------------------------------------

    if len(
        feature_columns
    ) < 50:

        raise RuntimeError(
            "Unexpectedly small feature matrix."
        )

    print(
        f"Model feature columns: "
        f"{len(feature_columns):,} ✓"
    )

    # --------------------------------------------------------
    # LEAKAGE
    # --------------------------------------------------------

    leakage_audit(
        feature_columns
    )

    print(
        "Target/post-match column leakage check: passed ✓"
    )

    # --------------------------------------------------------
    # INFINITY
    # --------------------------------------------------------

    check_infinite_values(
        dataframe=output,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # NUMERIC MODEL FEATURES
    # --------------------------------------------------------

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            output[
                column
            ]
        )
    ]

    if non_numeric:

        raise RuntimeError(
            "Non-numeric model features found: "
            f"{non_numeric}"
        )

    print(
        "All model features numeric: valid ✓"
    )

    # --------------------------------------------------------
    # EARLY / LATE FEATURE AVAILABILITY
    # --------------------------------------------------------

    first_100 = output.head(
        100
    )[
        feature_columns
    ]

    last_100 = output.tail(
        100
    )[
        feature_columns
    ]

    early_available = (
        1
        -
        first_100.isna().mean().mean()
    )

    late_available = (
        1
        -
        last_100.isna().mean().mean()
    )

    print(
        "\nOverall feature availability:"
    )

    print(
        f"First 100 matches: "
        f"{early_available * 100:.2f}%"
    )

    print(
        f"Last 100 matches:  "
        f"{late_available * 100:.2f}%"
    )

    print(
        "\nMissing values remain intentionally."
    )

    print(
        "Imputation will be fitted using training data only ✓"
    )


# ============================================================
# FEATURE FAMILY SUMMARY
# ============================================================


def print_family_summary(
    audit: pd.DataFrame,
) -> None:
    """
    Summarize feature families and missingness.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "MODEL FEATURE FAMILY SUMMARY"
    )

    print(
        "=" * 75
    )

    summary = (
        audit
        .groupby(
            "family"
        )
        .agg(
            feature_count=(
                "feature",
                "count",
            ),

            mean_missing_percent=(
                "missing_percent",
                "mean",
            ),

            max_missing_percent=(
                "missing_percent",
                "max",
            ),
        )
        .sort_values(
            by="feature_count",
            ascending=False,
        )
    )

    print(
        "\n"
        + summary.to_string(
            float_format=lambda value: (
                f"{value:.2f}"
            )
        )
    )


# ============================================================
# SAVE MANIFEST
# ============================================================


def save_manifest(
    *,
    feature_columns: list[str],
    constant_removed: list[str],
    rejected_columns: list[str],
    audit: pd.DataFrame,
) -> None:
    """
    Save exact feature list used by future model pipelines.
    """

    MODEL_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    family_counts = (
        audit[
            "family"
        ]
        .value_counts()
        .to_dict()
    )

    missing_percent = {
        row.feature: float(
            row.missing_percent
        )
        for row in audit.itertuples(
            index=False
        )
    }

    manifest = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source_file": str(
            INPUT_FILE
        ),

        "output_file": str(
            OUTPUT_FILE
        ),

        "rows": EXPECTED_ROWS,

        "target_column": "target",

        "target_mapping": TARGET_MAP,

        "feature_count": len(
            feature_columns
        ),

        "feature_columns": (
            feature_columns
        ),

        "feature_family_counts": (
            family_counts
        ),

        "feature_missing_percent": (
            missing_percent
        ),

        "constant_features_removed": (
            constant_removed
        ),

        "rejected_source_columns": (
            rejected_columns
        ),

        "modeling_rules": {
            "chronological_split_required": True,

            "random_train_test_split_forbidden": True,

            "imputation_fit_on_training_only": True,

            "scaling_fit_on_training_only": True,

            "same_match_goals_forbidden": True,

            "same_match_result_forbidden": True,

            "post_match_elo_forbidden": True,

            "audit_date_columns_forbidden": True,

            "team_names_not_model_features": True,

            "raw_season_not_model_feature": True,
        },
    }

    with FEATURE_MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nFeature manifest saved to:"
    )

    print(
        FEATURE_MANIFEST_FILE
    )


# ============================================================
# SAVE ML MATRIX
# ============================================================


def save_ml_features(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save chronological feature matrix.
    """

    output = dataframe.copy()

    output[
        "date"
    ] = output[
        "date"
    ].dt.strftime(
        "%Y-%m-%d"
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nML feature matrix saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# DISPLAY TOP MISSING FEATURES
# ============================================================


def print_missingness(
    audit: pd.DataFrame,
) -> None:
    """
    Print most incomplete selected features.
    """

    display = (
        audit
        .sort_values(
            by="missing_percent",
            ascending=False,
        )
        .head(
            20
        )[
            [
                "feature",
                "family",
                "missing_count",
                "missing_percent",
            ]
        ]
    )

    print(
        "\n20 model features with highest missingness:"
    )

    print(
        display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================


def print_summary(
    *,
    output: pd.DataFrame,
    feature_columns: list[str],
    constant_removed: list[str],
) -> None:
    """
    Print overall modeling dataset summary.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ML FEATURE MATRIX SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nMatches: "
        f"{len(output):,}"
    )

    print(
        f"Model features: "
        f"{len(feature_columns):,}"
    )

    print(
        f"Constant features removed: "
        f"{len(constant_removed):,}"
    )

    print(
        f"Date range: "
        f"{output['date'].min().date()} "
        f"to "
        f"{output['date'].max().date()}"
    )

    print(
        "\nTarget distribution:"
    )

    target_summary = (
        output[
            "result"
        ]
        .value_counts()
        .reindex(
            [
                "H",
                "D",
                "A",
            ]
        )
    )

    print(
        target_summary.to_string()
    )

    total_missing = int(
        output[
            feature_columns
        ].isna().sum().sum()
    )

    cells = (
        len(
            output
        )
        *
        len(
            feature_columns
        )
    )

    missing_percentage = (
        total_missing
        /
        cells
        *
        100
    )

    print(
        f"\nTotal feature NaNs: "
        f"{total_missing:,}"
    )

    print(
        f"Overall feature missingness: "
        f"{missing_percentage:.2f}%"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No missing feature has been imputed yet."
    )

    print(
        "Future preprocessing will be fitted "
        "inside chronological training folds only."
    )


# ============================================================
# MAIN
# ============================================================


def build_ml_features() -> None:
    """
    Build final historical feature matrix for chronological
    model development.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - MASTER ML FEATURE MATRIX"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    source = load_feature_history()

    validate_target(
        source
    )

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    (
        selected_features,
        rejected_columns,
    ) = select_model_features(
        source
    )

    print(
        f"\nNumeric pre-match candidates: "
        f"{len(selected_features):,}"
    )

    (
        feature_columns,
        constant_removed,
    ) = remove_constant_features(
        dataframe=source,
        feature_columns=selected_features,
    )

    print(
        f"Constant features removed: "
        f"{len(constant_removed):,}"
    )

    print(
        f"Final model feature count: "
        f"{len(feature_columns):,}"
    )

    # --------------------------------------------------------
    # FEATURE AUDIT
    # --------------------------------------------------------

    audit = build_feature_audit(
        dataframe=source,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    output = build_ml_dataframe(
        dataframe=source,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    run_quality_checks(
        source=source,
        output=output,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # REPORTING
    # --------------------------------------------------------

    print_family_summary(
        audit
    )

    print_missingness(
        audit
    )

    print_summary(
        output=output,
        feature_columns=feature_columns,
        constant_removed=constant_removed,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_ml_features(
        output
    )

    save_manifest(
        feature_columns=feature_columns,
        constant_removed=constant_removed,
        rejected_columns=rejected_columns,
        audit=audit,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "MASTER ML FEATURE MATRIX CREATED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nThe dataset is now ready for "
        "chronological train / validation / test design."
    )

    print(
        "\nNo model should use columns outside "
        "ml_feature_manifest.json."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    build_ml_features()