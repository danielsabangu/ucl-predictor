from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    INTERIM_DATA_DIR,
    TRAINED_MODELS_DIR,
)

from src.ml import predict_live as v1
from src.ml import predict_live_fixed as live_fixed


# ============================================================
# PURPOSE
# ============================================================
#
# Research-only V1 vs V2 shadow inference.
#
# This module:
#   - does NOT replace the production model
#   - does NOT modify V1
#   - does NOT modify public prediction files
#   - does NOT write to prediction history
#   - does NOT publish anything
#   - does NOT re-run the 2025/26 confirmation
#
# Both models receive the exact same live 290-feature base.
#
# V1:
#   frozen 290 production features
#
# V2:
#   same frozen 290 features
#   + uefa_coefficient_diff
#   + uefa_strength_percentile_diff
#
# The shared 290-feature row is built using the corrected
# timezone-safe implementation from predict_live_fixed.py.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_V1_FEATURES = 290
EXPECTED_V2_FEATURES = 292

EXPECTED_UEFA_FIXTURE_ROWS = 144

CURRENT_SEASON = 2026
CURRENT_RANKING_YEAR = 2026

V1_MODEL_VERSION = "ucl_predictor_v1"

V2_MODEL_VERSION = (
    "ucl_predictor_v2_candidate"
)

V2_REQUIRED_STATUS = (
    "candidate_not_promoted"
)

LOCKED_UEFA_FEATURES = [
    "uefa_coefficient_diff",
    "uefa_strength_percentile_diff",
]

CLASS_TO_RESULT = {
    0: "H",
    1: "D",
    2: "A",
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[
    2
]

V1_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "production_catboost.joblib"
)

V1_METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "metadata"
    / "production_model.json"
)

V2_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "ucl_predictor_v2_candidate.joblib"
)

V2_METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "metadata"
    / "ucl_predictor_v2_candidate.json"
)

LIVE_UEFA_FILE = (
    INTERIM_DATA_DIR
    / "features"
    / "ucl_2026_2027_uefa_strength.csv"
)

SHADOW_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "research"
)

# The first V2 shadow snapshot was created before the immutable
# archive layout existed. These three legacy files are now treated
# as frozen research evidence and must never be overwritten.
LEGACY_SHADOW_COMPARISON_FILE = (
    SHADOW_OUTPUT_DIR
    / "v2_shadow_next_matchday_comparison.csv"
)

LEGACY_SHADOW_FEATURE_MATRIX_FILE = (
    SHADOW_OUTPUT_DIR
    / "v2_shadow_next_matchday_features.csv"
)

LEGACY_SHADOW_METADATA_FILE = (
    SHADOW_OUTPUT_DIR
    / "v2_shadow_next_matchday_metadata.json"
)

FROZEN_LEGACY_SHADOW_FILES = [
    LEGACY_SHADOW_COMPARISON_FILE,
    LEGACY_SHADOW_FEATURE_MATRIX_FILE,
    LEGACY_SHADOW_METADATA_FILE,
]

# Every new shadow run is written exactly once beneath a UTC fixture
# date. Existing dates are never replaced or revised in place.
SHADOW_ARCHIVE_ROOT = (
    SHADOW_OUTPUT_DIR
    / "shadow"
)

SHADOW_COMPARISON_FILENAME = "comparison.csv"
SHADOW_FEATURE_MATRIX_FILENAME = "features.csv"
SHADOW_METADATA_FILENAME = "metadata.json"


# ============================================================
# PROTECTED PRODUCTION FILES
# ============================================================

PROTECTED_FILES = [
    V1_MODEL_FILE,
    V1_METADATA_FILE,

    v1.OUTPUT_FILE,
    v1.FEATURE_MATRIX_FILE,
    v1.METADATA_FILE,

    PROJECT_ROOT
    / "data"
    / "app"
    / "current_predictions.csv",

    PROJECT_ROOT
    / "data"
    / "app"
    / "pipeline_status.json",

    PROJECT_ROOT
    / "data"
    / "app"
    / "snapshot_metadata.json",
]


# ============================================================
# BASIC HELPERS
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )


def relative_path(
    path: Path,
) -> str:
    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )

    except ValueError:
        return str(
            path
        )


def sha256_file(
    path: Path,
) -> str:
    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            hasher.update(
                block
            )

    return hasher.hexdigest()


def snapshot_files(
    paths: list[Path],
) -> dict[
    str,
    str | None,
]:
    snapshot = {}

    for path in paths:
        key = relative_path(
            path
        )

        if path.exists():
            snapshot[
                key
            ] = sha256_file(
                path
            )

        else:
            snapshot[
                key
            ] = None

    return snapshot


def snapshot_protected_files() -> dict[
    str,
    str | None,
]:
    return snapshot_files(
        PROTECTED_FILES
    )


def snapshot_frozen_legacy_shadow_files() -> dict[
    str,
    str | None,
]:
    return snapshot_files(
        FROZEN_LEGACY_SHADOW_FILES
    )


def verify_file_snapshot_unchanged(
    before: dict[
        str,
        str | None,
    ],
    after: dict[
        str,
        str | None,
    ],
    *,
    description: str,
) -> None:
    if before != after:
        changed = []

        all_paths = sorted(
            set(
                before
            )
            |
            set(
                after
            )
        )

        for path in all_paths:
            if (
                before.get(
                    path
                )
                !=
                after.get(
                    path
                )
            ):
                changed.append(
                    path
                )

        raise RuntimeError(
            f"{description} changed during "
            "shadow inference:\n"
            + "\n".join(
                changed
            )
        )


def verify_protected_files_unchanged(
    before: dict[
        str,
        str | None,
    ],
    after: dict[
        str,
        str | None,
    ],
) -> None:
    verify_file_snapshot_unchanged(
        before,
        after,
        description=(
            "Protected production files"
        ),
    )


def verify_frozen_legacy_shadow_files_unchanged(
    before: dict[
        str,
        str | None,
    ],
    after: dict[
        str,
        str | None,
    ],
) -> None:
    verify_file_snapshot_unchanged(
        before,
        after,
        description=(
            "Frozen legacy shadow files"
        ),
    )


def normalize_fixture_date_utc(
    value,
) -> str:
    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )

    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp.date().isoformat()


def fixture_date_utc_from_frame(
    dataframe: pd.DataFrame,
) -> str:
    if "date_utc" not in dataframe.columns:
        raise RuntimeError(
            "Shadow fixtures are missing "
            "date_utc."
        )

    if dataframe.empty:
        raise RuntimeError(
            "Shadow fixtures are empty."
        )

    dates = {
        normalize_fixture_date_utc(
            value
        )
        for value in dataframe[
            "date_utc"
        ]
    }

    if len(
        dates
    ) != 1:
        raise RuntimeError(
            "A shadow archive must contain "
            "exactly one UTC fixture date. "
            f"Found: {sorted(dates)}"
        )

    return next(
        iter(
            dates
        )
    )


def load_legacy_shadow_fixture_date() -> (
    str
    | None
):
    existence = [
        path.exists()
        for path in (
            FROZEN_LEGACY_SHADOW_FILES
        )
    ]

    if not any(
        existence
    ):
        return None

    if not all(
        existence
    ):
        missing = [
            relative_path(
                path
            )
            for path in (
                FROZEN_LEGACY_SHADOW_FILES
            )
            if not path.exists()
        ]

        raise RuntimeError(
            "The frozen legacy shadow snapshot "
            "is incomplete. Missing:\n"
            + "\n".join(
                missing
            )
        )

    with LEGACY_SHADOW_METADATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(
            file
        )

    fixture_date = metadata.get(
        "fixture_date_utc"
    )

    if not fixture_date:
        raise RuntimeError(
            "The frozen legacy shadow metadata "
            "does not contain fixture_date_utc."
        )

    normalized = normalize_fixture_date_utc(
        fixture_date
    )

    return normalized


def build_shadow_archive_paths(
    fixture_date_utc: str,
    *,
    archive_root: Path | None = None,
) -> dict[
    str,
    Path,
]:
    normalized_date = normalize_fixture_date_utc(
        fixture_date_utc
    )

    root = (
        SHADOW_ARCHIVE_ROOT
        if archive_root is None
        else Path(
            archive_root
        )
    )

    archive_dir = (
        root
        / normalized_date
    )

    return {
        "directory": archive_dir,
        "comparison": (
            archive_dir
            / SHADOW_COMPARISON_FILENAME
        ),
        "feature_matrix": (
            archive_dir
            / SHADOW_FEATURE_MATRIX_FILENAME
        ),
        "metadata": (
            archive_dir
            / SHADOW_METADATA_FILENAME
        ),
    }


def shadow_date_frozen_reason(
    fixture_date_utc: str,
    *,
    archive_root: Path | None = None,
    include_legacy: bool = True,
) -> str | None:
    normalized_date = normalize_fixture_date_utc(
        fixture_date_utc
    )

    if include_legacy:
        legacy_date = (
            load_legacy_shadow_fixture_date()
        )

        if (
            legacy_date is not None
            and
            legacy_date == normalized_date
        ):
            return (
                "A frozen legacy shadow snapshot "
                f"already exists for {normalized_date}."
            )

    paths = build_shadow_archive_paths(
        normalized_date,
        archive_root=archive_root,
    )

    archive_dir = paths[
        "directory"
    ]

    if archive_dir.exists():
        return (
            "An immutable shadow archive already "
            f"exists for {normalized_date}:\n"
            f"{archive_dir}"
        )

    existing_targets = [
        path
        for key, path in paths.items()
        if (
            key != "directory"
            and
            path.exists()
        )
    ]

    if existing_targets:
        return (
            "Immutable shadow archive targets "
            f"already exist for {normalized_date}:\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing_targets
            )
        )

    return None


def ensure_shadow_date_is_not_frozen(
    fixture_date_utc: str,
    *,
    archive_root: Path | None = None,
    include_legacy: bool = True,
) -> None:
    reason = shadow_date_frozen_reason(
        fixture_date_utc,
        archive_root=archive_root,
        include_legacy=include_legacy,
    )

    if reason is not None:
        raise RuntimeError(
            reason
            + "\nRefusing to overwrite frozen "
            "pre-match shadow evidence."
        )


def immutable_input_hashes() -> dict[
    str,
    str,
]:
    files = {
        "v1_model": V1_MODEL_FILE,
        "v1_metadata": V1_METADATA_FILE,
        "v2_model": V2_MODEL_FILE,
        "v2_metadata": V2_METADATA_FILE,
        "live_uefa_features": LIVE_UEFA_FILE,
    }

    hashes = {}

    for key, path in files.items():
        require_file(
            path,
            key.replace(
                "_",
                " "
            ),
        )

        hashes[
            key
        ] = sha256_file(
            path
        )

    return hashes


def normalize_source_match_id(
    value,
) -> str:
    if pd.isna(
        value
    ):
        raise RuntimeError(
            "Missing source_match_id."
        )

    text = str(
        value
    ).strip()

    if not text:
        raise RuntimeError(
            "Blank source_match_id."
        )

    try:
        numeric = float(
            text
        )

        if (
            np.isfinite(
                numeric
            )
            and
            numeric.is_integer()
        ):
            return str(
                int(
                    numeric
                )
            )

    except ValueError:
        pass

    return text


def probability_label(
    predicted_class: int,
    home_team: str,
    away_team: str,
) -> str:
    if predicted_class == 0:
        return home_team

    if predicted_class == 1:
        return "Draw"

    if predicted_class == 2:
        return away_team

    raise RuntimeError(
        f"Unknown prediction class: "
        f"{predicted_class}"
    )


# ============================================================
# MODEL FEATURE CONTRACT
# ============================================================


def recover_model_feature_names(
    model,
    *,
    expected_count: int,
    description: str,
) -> list[str]:
    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:
        if hasattr(
            model,
            "named_steps",
        ):
            for step in (
                model.named_steps.values()
            ):
                feature_names = getattr(
                    step,
                    "feature_names_in_",
                    None,
                )

                if feature_names is not None:
                    break

    if feature_names is None:
        raise RuntimeError(
            f"Could not recover feature names "
            f"from {description}."
        )

    feature_names = [
        str(
            feature
        )
        for feature in feature_names
    ]

    if (
        len(
            feature_names
        )
        !=
        expected_count
    ):
        raise RuntimeError(
            f"{description} expected "
            f"{expected_count} features, "
            f"found {len(feature_names)}."
        )

    if (
        len(
            set(
                feature_names
            )
        )
        !=
        len(
            feature_names
        )
    ):
        raise RuntimeError(
            f"{description} contains "
            "duplicate feature names."
        )

    return feature_names


def validate_model_classes(
    model,
    *,
    description: str,
) -> None:
    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None:
        raise RuntimeError(
            f"{description} exposes no classes_."
        )

    normalized = [
        int(
            value
        )
        for value in classes
    ]

    if normalized != [
        0,
        1,
        2,
    ]:
        raise RuntimeError(
            f"{description} classes are not "
            f"[0, 1, 2]: {normalized}"
        )


# ============================================================
# V2 MODEL
# ============================================================


def load_v2_candidate(
    v1_features: list[str],
):
    require_file(
        V2_MODEL_FILE,
        "V2 candidate model",
    )

    require_file(
        V2_METADATA_FILE,
        "V2 candidate metadata",
    )

    with V2_METADATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(
            file
        )

    if (
        metadata.get(
            "model_version"
        )
        !=
        V2_MODEL_VERSION
    ):
        raise RuntimeError(
            "Unexpected V2 model version in metadata."
        )

    if (
        metadata.get(
            "status"
        )
        !=
        V2_REQUIRED_STATUS
    ):
        raise RuntimeError(
            "V2 metadata does not have "
            "'candidate_not_promoted' status."
        )

    metadata_features = metadata.get(
        "feature_columns"
    )

    if not isinstance(
        metadata_features,
        list,
    ):
        raise RuntimeError(
            "V2 metadata has no valid "
            "feature_columns list."
        )

    metadata_features = [
        str(
            feature
        )
        for feature in metadata_features
    ]

    if (
        len(
            metadata_features
        )
        !=
        EXPECTED_V2_FEATURES
    ):
        raise RuntimeError(
            "V2 metadata feature count "
            "is not 292."
        )

    model = joblib.load(
        V2_MODEL_FILE
    )

    model_features = (
        recover_model_feature_names(
            model,
            expected_count=(
                EXPECTED_V2_FEATURES
            ),
            description=(
                "V2 candidate model"
            ),
        )
    )

    validate_model_classes(
        model,
        description=(
            "V2 candidate model"
        ),
    )

    if (
        metadata_features
        !=
        model_features
    ):
        raise RuntimeError(
            "V2 metadata feature order "
            "does not match V2 model."
        )

    if (
        model_features[
            :EXPECTED_V1_FEATURES
        ]
        !=
        v1_features
    ):
        raise RuntimeError(
            "The first 290 V2 features "
            "do not exactly match V1."
        )

    if (
        model_features[
            EXPECTED_V1_FEATURES:
        ]
        !=
        LOCKED_UEFA_FEATURES
    ):
        raise RuntimeError(
            "The final two V2 features "
            "are not the locked UEFA subset."
        )

    selected = metadata.get(
        "selected_uefa_features"
    )

    if (
        selected
        !=
        LOCKED_UEFA_FEATURES
    ):
        raise RuntimeError(
            "V2 metadata selected UEFA "
            "features do not match the lock."
        )

    return (
        model,
        model_features,
        metadata,
    )


# ============================================================
# LIVE UEFA FEATURE LAYER
# ============================================================


def load_live_uefa_features() -> pd.DataFrame:
    require_file(
        LIVE_UEFA_FILE,
        "Live UEFA strength features",
    )

    dataframe = pd.read_csv(
        LIVE_UEFA_FILE,
        low_memory=False,
    )

    required = {
        "source_match_id",
        "date_utc",
        "date",
        "season",
        "ranking_year",
        "home_team",
        "away_team",
        "home_uefa_coefficient",
        "away_uefa_coefficient",
        "uefa_coefficient_diff",
        "home_uefa_strength_percentile",
        "away_uefa_strength_percentile",
        "uefa_strength_percentile_diff",
        "home_uefa_country_floor_used",
        "away_uefa_country_floor_used",
        "home_uefa_source_type",
        "away_uefa_source_type",
        "home_uefa_resolution_method",
        "away_uefa_resolution_method",
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
            "Live UEFA file is missing "
            "required columns:\n"
            f"{sorted(missing)}"
        )

    if (
        len(
            dataframe
        )
        !=
        EXPECTED_UEFA_FIXTURE_ROWS
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_UEFA_FIXTURE_ROWS} "
            "live UEFA fixture rows, found "
            f"{len(dataframe)}."
        )

    dataframe[
        "source_match_id_normalized"
    ] = dataframe[
        "source_match_id"
    ].map(
        normalize_source_match_id
    )

    if dataframe[
        "source_match_id_normalized"
    ].duplicated().any():
        duplicates = dataframe.loc[
            dataframe[
                "source_match_id_normalized"
            ].duplicated(
                keep=False
            ),
            [
                "source_match_id",
                "date_utc",
                "home_team",
                "away_team",
            ],
        ]

        raise RuntimeError(
            "Duplicate source_match_id values "
            "in live UEFA file:\n"
            f"{duplicates.to_string(index=False)}"
        )

    dataframe[
        "date_utc"
    ] = pd.to_datetime(
        dataframe[
            "date_utc"
        ],
        errors="raise",
        utc=True,
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
        "ranking_year"
    ] = pd.to_numeric(
        dataframe[
            "ranking_year"
        ],
        errors="raise",
    ).astype(
        int
    )

    if not (
        dataframe[
            "season"
        ]
        ==
        CURRENT_SEASON
    ).all():
        raise RuntimeError(
            "Live UEFA file contains "
            "unexpected seasons."
        )

    if not (
        dataframe[
            "ranking_year"
        ]
        ==
        CURRENT_RANKING_YEAR
    ).all():
        raise RuntimeError(
            "Live UEFA file contains "
            "unexpected ranking years."
        )

    numeric_columns = [
        "home_uefa_coefficient",
        "away_uefa_coefficient",
        "uefa_coefficient_diff",
        "home_uefa_strength_percentile",
        "away_uefa_strength_percentile",
        "uefa_strength_percentile_diff",
    ]

    for column in numeric_columns:
        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="raise",
        )

        values = dataframe[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                "Non-finite live UEFA values "
                f"found in {column}."
            )

    for column in [
        "home_uefa_strength_percentile",
        "away_uefa_strength_percentile",
    ]:
        invalid = (
            (
                dataframe[
                    column
                ]
                <
                0.0
            )
            |
            (
                dataframe[
                    column
                ]
                >
                1.0
            )
        )

        if invalid.any():
            raise RuntimeError(
                f"{column} contains values "
                "outside [0, 1]."
            )

    coefficient_diff = (
        dataframe[
            "home_uefa_coefficient"
        ]
        -
        dataframe[
            "away_uefa_coefficient"
        ]
    )

    if not np.allclose(
        coefficient_diff.to_numpy(
            dtype=float
        ),
        dataframe[
            "uefa_coefficient_diff"
        ].to_numpy(
            dtype=float
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "uefa_coefficient_diff does not "
            "equal home minus away."
        )

    percentile_diff = (
        dataframe[
            "home_uefa_strength_percentile"
        ]
        -
        dataframe[
            "away_uefa_strength_percentile"
        ]
    )

    if not np.allclose(
        percentile_diff.to_numpy(
            dtype=float
        ),
        dataframe[
            "uefa_strength_percentile_diff"
        ].to_numpy(
            dtype=float
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "uefa_strength_percentile_diff "
            "does not equal home minus away."
        )

    resolution_columns = [
        "home_uefa_resolution_method",
        "away_uefa_resolution_method",
    ]

    for column in resolution_columns:
        contains_fuzzy = (
            dataframe[
                column
            ]
            .astype(
                str
            )
            .str.upper()
            .str.contains(
                "FUZZY",
                regex=False,
            )
        )

        if contains_fuzzy.any():
            raise RuntimeError(
                "Fuzzy UEFA resolution detected."
            )

    return dataframe


# ============================================================
# FIXTURE / UEFA IDENTITY
# ============================================================


def build_uefa_lookup(
    fixtures: pd.DataFrame,
    uefa: pd.DataFrame,
) -> dict[
    str,
    pd.Series,
]:
    required_fixture_columns = {
        "source_match_id",
        "date_utc",
        "home_team",
        "away_team",
    }

    missing = (
        required_fixture_columns
        -
        set(
            fixtures.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Selected live fixtures do not "
            "contain deterministic UEFA "
            "identity columns:\n"
            f"{sorted(missing)}"
        )

    lookup = {
        str(
            row[
                "source_match_id_normalized"
            ]
        ): row
        for _, row in (
            uefa.iterrows()
        )
    }

    matched = {}

    for fixture in (
        fixtures.to_dict(
            orient="records"
        )
    ):
        source_match_id = (
            normalize_source_match_id(
                fixture[
                    "source_match_id"
                ]
            )
        )

        if source_match_id not in lookup:
            raise RuntimeError(
                "No live UEFA row for "
                "selected fixture "
                f"source_match_id="
                f"{source_match_id}.\n"
                "Rerun:\n"
                "python -m "
                "src.features.live_uefa_strength"
            )

        row = lookup[
            source_match_id
        ]

        fixture_home = str(
            fixture[
                "home_team"
            ]
        )

        fixture_away = str(
            fixture[
                "away_team"
            ]
        )

        uefa_home = str(
            row[
                "home_team"
            ]
        )

        uefa_away = str(
            row[
                "away_team"
            ]
        )

        if (
            fixture_home
            !=
            uefa_home
            or
            fixture_away
            !=
            uefa_away
        ):
            raise RuntimeError(
                "Fixture / UEFA team identity "
                "mismatch for source_match_id "
                f"{source_match_id}.\n"
                f"Fixture: "
                f"{fixture_home} vs "
                f"{fixture_away}\n"
                f"UEFA: "
                f"{uefa_home} vs "
                f"{uefa_away}"
            )

        fixture_date = pd.Timestamp(
            fixture[
                "date_utc"
            ]
        )

        if fixture_date.tzinfo is None:
            fixture_date = (
                fixture_date.tz_localize(
                    "UTC"
                )
            )
        else:
            fixture_date = (
                fixture_date.tz_convert(
                    "UTC"
                )
            )

        uefa_date = pd.Timestamp(
            row[
                "date_utc"
            ]
        )

        if uefa_date.tzinfo is None:
            uefa_date = (
                uefa_date.tz_localize(
                    "UTC"
                )
            )
        else:
            uefa_date = (
                uefa_date.tz_convert(
                    "UTC"
                )
            )

        if fixture_date != uefa_date:
            raise RuntimeError(
                "Fixture / UEFA kickoff mismatch "
                "for source_match_id "
                f"{source_match_id}.\n"
                f"Fixture: {fixture_date}\n"
                f"UEFA: {uefa_date}"
            )

        matched[
            source_match_id
        ] = row

    if (
        len(
            matched
        )
        !=
        len(
            fixtures
        )
    ):
        raise RuntimeError(
            "Selected fixture / UEFA lookup "
            "cardinality mismatch."
        )

    return matched


# ============================================================
# GENERIC MODEL PREDICTION
# ============================================================


def predict_probabilities(
    *,
    model,
    feature_names: list[str],
    feature_row: dict,
    description: str,
) -> tuple[
    float,
    float,
    float,
    int,
]:
    matrix = pd.DataFrame(
        [
            feature_row
        ],
        columns=feature_names,
    )

    for column in (
        matrix.columns
    ):
        matrix[
            column
        ] = pd.to_numeric(
            matrix[
                column
            ],
            errors="coerce",
        )

    infinite_count = int(
        np.isinf(
            matrix.to_numpy(
                dtype=float
            )
        ).sum()
    )

    if infinite_count:
        raise RuntimeError(
            f"{description} feature row "
            "contains infinite values."
        )

    probabilities = model.predict_proba(
        matrix
    )[
        0
    ]

    classes = [
        int(
            value
        )
        for value in model.classes_
    ]

    by_class = {
        class_value: float(
            probability
        )
        for (
            class_value,
            probability,
        ) in zip(
            classes,
            probabilities,
        )
    }

    if set(
        by_class
    ) != {
        0,
        1,
        2,
    }:
        raise RuntimeError(
            f"{description} prediction "
            "did not return H/D/A."
        )

    ordered = np.array(
        [
            by_class[
                0
            ],
            by_class[
                1
            ],
            by_class[
                2
            ],
        ],
        dtype=float,
    )

    if not np.isfinite(
        ordered
    ).all():
        raise RuntimeError(
            f"{description} produced "
            "non-finite probabilities."
        )

    if (
        (
            ordered
            <
            0.0
        ).any()
        or
        (
            ordered
            >
            1.0
        ).any()
    ):
        raise RuntimeError(
            f"{description} produced "
            "a probability outside [0, 1]."
        )

    if not np.isclose(
        ordered.sum(),
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"{description} probabilities "
            "do not sum to 1."
        )

    predicted_class = int(
        np.argmax(
            ordered
        )
    )

    return (
        float(
            ordered[
                0
            ]
        ),
        float(
            ordered[
                1
            ]
        ),
        float(
            ordered[
                2
            ]
        ),
        predicted_class,
    )


# ============================================================
# SHADOW CONTEXT
# ============================================================


def prepare_shadow_context():
    (
        v1_model,
        v1_features,
    ) = (
        v1.load_production_model()
    )

    if (
        len(
            v1_features
        )
        !=
        EXPECTED_V1_FEATURES
    ):
        raise RuntimeError(
            "V1 production feature count "
            "is not 290."
        )

    validate_model_classes(
        v1_model,
        description=(
            "V1 production model"
        ),
    )

    families = (
        v1.reconstruct_feature_families(
            v1_features
        )
    )

    (
        v2_model,
        v2_features,
        v2_metadata,
    ) = load_v2_candidate(
        v1_features
    )

    fixtures = (
        v1.load_next_matchday_fixtures()
    )

    uefa = (
        load_live_uefa_features()
    )

    uefa_lookup = (
        build_uefa_lookup(
            fixtures,
            uefa,
        )
    )

    (
        base_replay,
        date_column,
    ) = (
        v1.load_live_ucl_replay()
    )

    (
        live_domestic,
        domestic_histories,
        domestic_dates,
    ) = (
        v1.load_live_domestic_histories()
    )

    squad_state = (
        v1.load_live_squad_state()
    )

    v1.verify_squad_reference_date(
        squad_state=squad_state,
        fixtures=fixtures,
    )

    squad_difference_mapping = (
        v1.infer_difference_feature_mapping(
            families[
                "squad"
            ]
        )
    )

    return {
        "v1_model": (
            v1_model
        ),
        "v1_features": (
            v1_features
        ),
        "families": (
            families
        ),
        "v2_model": (
            v2_model
        ),
        "v2_features": (
            v2_features
        ),
        "v2_metadata": (
            v2_metadata
        ),
        "fixtures": (
            fixtures
        ),
        "uefa": (
            uefa
        ),
        "uefa_lookup": (
            uefa_lookup
        ),
        "base_replay": (
            base_replay
        ),
        "date_column": (
            date_column
        ),
        "live_domestic": (
            live_domestic
        ),
        "domestic_histories": (
            domestic_histories
        ),
        "domestic_dates": (
            domestic_dates
        ),
        "squad_state": (
            squad_state
        ),
        "squad_difference_mapping": (
            squad_difference_mapping
        ),
    }


# ============================================================
# VALIDATION-ONLY GATE
# ============================================================


def run_validation_only() -> None:
    print(
        "=" * 90
    )

    print(
        "UCL PREDICTOR V2 - "
        "SHADOW INFERENCE CONTRACT"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "Mode: VALIDATION ONLY"
    )

    print(
        "Predictions generated: NO"
    )

    print(
        "Files written: NO"
    )

    print(
        "Production modified: NO"
    )

    context = (
        prepare_shadow_context()
    )

    fixtures = context[
        "fixtures"
    ]

    fixture_date_utc = (
        fixture_date_utc_from_frame(
            fixtures
        )
    )

    frozen_reason = (
        shadow_date_frozen_reason(
            fixture_date_utc
        )
    )

    print()
    print(
        "Immutable shadow archive:"
    )

    if frozen_reason is None:
        print(
            "  Current fixture date: AVAILABLE"
        )

    else:
        print(
            "  Current fixture date: FROZEN"
        )
        print(
            "  Rerun protection: ACTIVE"
        )

    print()
    print(
        "V1 feature count: "
        f"{len(context['v1_features'])}"
    )

    print(
        "V2 feature count: "
        f"{len(context['v2_features'])}"
    )

    print(
        "Locked UEFA additions: "
        f"{len(LOCKED_UEFA_FEATURES)}"
    )

    print()
    print(
        "Selected live fixtures: "
        f"{len(fixtures)}"
    )

    print(
        "Fixture date: "
        f"{fixtures['date_utc'].min().date()}"
    )

    print()
    print(
        "Fixture / UEFA identity:"
    )

    for fixture in (
        fixtures.itertuples(
            index=False
        )
    ):
        source_match_id = (
            normalize_source_match_id(
                fixture.source_match_id
            )
        )

        print(
            "  "
            f"{source_match_id} | "
            f"{fixture.home_team} vs "
            f"{fixture.away_team}"
        )

    print()
    print(
        "Live UCL replay rows: "
        f"{len(context['base_replay']):,}"
    )

    print(
        "Live domestic rows: "
        f"{len(context['live_domestic']):,}"
    )

    print(
        "Safe squad state teams: "
        f"{len(context['squad_state'])}"
    )

    print()
    print(
        "290 V1 feature contract: PASS"
    )

    print(
        "292 V2 feature contract: PASS"
    )

    print(
        "First 290 V2 features equal V1: PASS"
    )

    print(
        "Final two UEFA features exact: PASS"
    )

    print(
        "Deterministic source_match_id join: PASS"
    )

    print(
        "Fixture team identity parity: PASS"
    )

    print(
        "Fixture kickoff parity: PASS"
    )

    print(
        "Timezone-safe live builder selected: PASS"
    )

    print(
        "V2 candidate remains unpromoted: PASS"
    )

    print()
    print(
        "=" * 90
    )

    print(
        "V2 SHADOW CONTRACT: PASS"
    )

    print(
        "SAFE TO RUN SHADOW INFERENCE: YES"
    )

    print(
        "=" * 90
    )


# ============================================================
# OUTPUT QUALITY
# ============================================================


def validate_shadow_outputs(
    *,
    comparison: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    v1_features: list[str],
    v2_features: list[str],
) -> None:
    if comparison.empty:
        raise RuntimeError(
            "Shadow comparison is empty."
        )

    if (
        len(
            comparison
        )
        !=
        len(
            feature_matrix
        )
    ):
        raise RuntimeError(
            "Comparison / feature matrix "
            "row count mismatch."
        )

    if (
        list(
            feature_matrix.columns
        )
        !=
        v2_features
    ):
        raise RuntimeError(
            "Shadow V2 feature matrix "
            "order mismatch."
        )

    if (
        list(
            feature_matrix.columns[
                :EXPECTED_V1_FEATURES
            ]
        )
        !=
        v1_features
    ):
        raise RuntimeError(
            "Shadow V2 matrix does not "
            "preserve the V1 prefix."
        )

    if (
        list(
            feature_matrix.columns[
                EXPECTED_V1_FEATURES:
            ]
        )
        !=
        LOCKED_UEFA_FEATURES
    ):
        raise RuntimeError(
            "Shadow feature matrix has "
            "wrong UEFA suffix."
        )

    infinite_count = int(
        np.isinf(
            feature_matrix.to_numpy(
                dtype=float
            )
        ).sum()
    )

    if infinite_count:
        raise RuntimeError(
            "Infinite values found in "
            "shadow feature matrix."
        )

    probability_columns = [
        "v1_home_probability",
        "v1_draw_probability",
        "v1_away_probability",
        "v2_home_probability",
        "v2_draw_probability",
        "v2_away_probability",
    ]

    for column in probability_columns:
        values = comparison[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"Non-finite values in "
                f"{column}."
            )

        if (
            (
                values
                <
                0.0
            ).any()
            or
            (
                values
                >
                1.0
            ).any()
        ):
            raise RuntimeError(
                f"Invalid probability in "
                f"{column}."
            )

    v1_sums = (
        comparison[
            "v1_home_probability"
        ]
        +
        comparison[
            "v1_draw_probability"
        ]
        +
        comparison[
            "v1_away_probability"
        ]
    )

    v2_sums = (
        comparison[
            "v2_home_probability"
        ]
        +
        comparison[
            "v2_draw_probability"
        ]
        +
        comparison[
            "v2_away_probability"
        ]
    )

    if not np.allclose(
        v1_sums.to_numpy(
            dtype=float
        ),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            "At least one V1 probability "
            "row does not sum to 1."
        )

    if not np.allclose(
        v2_sums.to_numpy(
            dtype=float
        ),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            "At least one V2 probability "
            "row does not sum to 1."
        )


# ============================================================
# PRINT SHADOW COMPARISON
# ============================================================


def print_shadow_results(
    comparison: pd.DataFrame,
) -> None:
    print()
    print(
        "=" * 90
    )

    print(
        "UCL 2026/27 - "
        "V1 VS V2 SHADOW COMPARISON"
    )

    print(
        "=" * 90
    )

    for row in (
        comparison.itertuples(
            index=False
        )
    ):
        print()
        print(
            "-" * 90
        )

        print(
            f"{row.kickoff_utc}"
        )

        print(
            f"{row.home_team} "
            f"vs "
            f"{row.away_team}"
        )

        print()
        print(
            "UEFA:"
        )

        print(
            "  coefficient diff: "
            f"{row.uefa_coefficient_diff:+.3f}"
        )

        print(
            "  percentile diff:  "
            f"{row.uefa_strength_percentile_diff:+.6f}"
        )

        print()
        print(
            "V1:"
        )

        print(
            "  H "
            f"{row.v1_home_probability * 100:6.2f}%"
            " | D "
            f"{row.v1_draw_probability * 100:6.2f}%"
            " | A "
            f"{row.v1_away_probability * 100:6.2f}%"
        )

        print(
            "  Pick: "
            f"{row.v1_predicted_selection}"
        )

        print()
        print(
            "V2:"
        )

        print(
            "  H "
            f"{row.v2_home_probability * 100:6.2f}%"
            " | D "
            f"{row.v2_draw_probability * 100:6.2f}%"
            " | A "
            f"{row.v2_away_probability * 100:6.2f}%"
        )

        print(
            "  Pick: "
            f"{row.v2_predicted_selection}"
        )

        print()
        print(
            "V2 - V1 movement:"
        )

        print(
            "  H "
            f"{row.home_probability_delta_pp:+.2f} pp"
            " | D "
            f"{row.draw_probability_delta_pp:+.2f} pp"
            " | A "
            f"{row.away_probability_delta_pp:+.2f} pp"
        )

        print(
            "  Largest class movement: "
            f"{row.max_absolute_probability_delta_pp:.2f} pp"
        )

        print(
            "  Pick changed: "
            f"{'YES' if row.pick_changed else 'NO'}"
        )


# ============================================================
# ACTUAL SHADOW INFERENCE
# ============================================================


def run_shadow_inference() -> None:
    print(
        "=" * 90
    )

    print(
        "UCL PREDICTOR - "
        "V1 VS V2 SHADOW INFERENCE"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "Research-only execution."
    )

    print(
        "Production model promotion: NO"
    )

    print(
        "Public prediction write: NO"
    )

    print(
        "Prediction-history write: NO"
    )

    print(
        "2025 confirmation re-run: NO"
    )

    protected_before = (
        snapshot_protected_files()
    )

    frozen_legacy_before = (
        snapshot_frozen_legacy_shadow_files()
    )

    context = (
        prepare_shadow_context()
    )

    v1_model = context[
        "v1_model"
    ]

    v1_features = context[
        "v1_features"
    ]

    families = context[
        "families"
    ]

    v2_model = context[
        "v2_model"
    ]

    v2_features = context[
        "v2_features"
    ]

    fixtures = context[
        "fixtures"
    ]

    fixture_date_utc = (
        fixture_date_utc_from_frame(
            fixtures
        )
    )

    ensure_shadow_date_is_not_frozen(
        fixture_date_utc
    )

    archive_paths = (
        build_shadow_archive_paths(
            fixture_date_utc
        )
    )

    input_hashes = (
        immutable_input_hashes()
    )

    uefa_lookup = context[
        "uefa_lookup"
    ]

    base_replay = context[
        "base_replay"
    ]

    date_column = context[
        "date_column"
    ]

    domestic_histories = context[
        "domestic_histories"
    ]

    domestic_dates = context[
        "domestic_dates"
    ]

    squad_state = context[
        "squad_state"
    ]

    squad_difference_mapping = context[
        "squad_difference_mapping"
    ]

    print()
    print(
        "Selected fixture date: "
        f"{fixtures['date_utc'].min().date()}"
    )

    print(
        "Selected fixtures: "
        f"{len(fixtures)}"
    )

    comparison_rows = []
    v2_feature_rows = []

    for fixture_record in (
        fixtures.to_dict(
            orient="records"
        )
    ):
        fixture = pd.Series(
            fixture_record
        )

        source_match_id = (
            normalize_source_match_id(
                fixture[
                    "source_match_id"
                ]
            )
        )

        home_team = str(
            fixture[
                "home_team"
            ]
        )

        away_team = str(
            fixture[
                "away_team"
            ]
        )

        kickoff = pd.Timestamp(
            fixture[
                "date_utc"
            ]
        )

        print()
        print(
            "-" * 90
        )

        print(
            "Building shared live features: "
            f"{home_team} vs "
            f"{away_team}"
        )

        (
            v1_feature_row,
            availability,
        ) = (
            live_fixed.build_fixture_features(
                fixture=fixture,
                base_replay=base_replay,
                date_column=date_column,
                production_features=(
                    v1_features
                ),
                families=families,
                domestic_histories=(
                    domestic_histories
                ),
                domestic_dates=(
                    domestic_dates
                ),
                squad_state=squad_state,
                squad_difference_mapping=(
                    squad_difference_mapping
                ),
            )
        )

        if (
            list(
                v1_feature_row.keys()
            )
            !=
            v1_features
        ):
            raise RuntimeError(
                "Shared V1 feature row does "
                "not preserve exact V1 order."
            )

        (
            v1_home,
            v1_draw,
            v1_away,
            v1_class,
        ) = predict_probabilities(
            model=v1_model,
            feature_names=(
                v1_features
            ),
            feature_row=(
                v1_feature_row
            ),
            description=(
                "V1 production model"
            ),
        )

        uefa_row = uefa_lookup[
            source_match_id
        ]

        v2_feature_row = dict(
            v1_feature_row
        )

        for feature in (
            LOCKED_UEFA_FEATURES
        ):
            value = float(
                uefa_row[
                    feature
                ]
            )

            if not np.isfinite(
                value
            ):
                raise RuntimeError(
                    "Non-finite UEFA feature "
                    f"{feature} for "
                    f"{home_team} vs "
                    f"{away_team}."
                )

            v2_feature_row[
                feature
            ] = value

        if (
            list(
                v2_feature_row.keys()
            )
            !=
            v2_features
        ):
            raise RuntimeError(
                "V2 live feature row does "
                "not preserve exact 292-feature "
                "model order."
            )

        (
            v2_home,
            v2_draw,
            v2_away,
            v2_class,
        ) = predict_probabilities(
            model=v2_model,
            feature_names=(
                v2_features
            ),
            feature_row=(
                v2_feature_row
            ),
            description=(
                "V2 candidate model"
            ),
        )

        v2_feature_rows.append(
            v2_feature_row
        )

        home_delta_pp = (
            (
                v2_home
                -
                v1_home
            )
            *
            100.0
        )

        draw_delta_pp = (
            (
                v2_draw
                -
                v1_draw
            )
            *
            100.0
        )

        away_delta_pp = (
            (
                v2_away
                -
                v1_away
            )
            *
            100.0
        )

        max_delta_pp = max(
            abs(
                home_delta_pp
            ),
            abs(
                draw_delta_pp
            ),
            abs(
                away_delta_pp
            ),
        )

        mean_abs_delta_pp = float(
            np.mean(
                [
                    abs(
                        home_delta_pp
                    ),
                    abs(
                        draw_delta_pp
                    ),
                    abs(
                        away_delta_pp
                    ),
                ]
            )
        )

        comparison_rows.append(
            {
                "source_match_id": (
                    source_match_id
                ),

                "kickoff_utc": (
                    kickoff
                ),

                "home_team": (
                    home_team
                ),

                "away_team": (
                    away_team
                ),

                "home_uefa_coefficient": float(
                    uefa_row[
                        "home_uefa_coefficient"
                    ]
                ),

                "away_uefa_coefficient": float(
                    uefa_row[
                        "away_uefa_coefficient"
                    ]
                ),

                "uefa_coefficient_diff": float(
                    uefa_row[
                        "uefa_coefficient_diff"
                    ]
                ),

                "home_uefa_strength_percentile": float(
                    uefa_row[
                        "home_uefa_strength_percentile"
                    ]
                ),

                "away_uefa_strength_percentile": float(
                    uefa_row[
                        "away_uefa_strength_percentile"
                    ]
                ),

                "uefa_strength_percentile_diff": float(
                    uefa_row[
                        "uefa_strength_percentile_diff"
                    ]
                ),

                "home_uefa_source_type": str(
                    uefa_row[
                        "home_uefa_source_type"
                    ]
                ),

                "away_uefa_source_type": str(
                    uefa_row[
                        "away_uefa_source_type"
                    ]
                ),

                "home_uefa_resolution_method": str(
                    uefa_row[
                        "home_uefa_resolution_method"
                    ]
                ),

                "away_uefa_resolution_method": str(
                    uefa_row[
                        "away_uefa_resolution_method"
                    ]
                ),

                "home_uefa_country_floor_used": int(
                    uefa_row[
                        "home_uefa_country_floor_used"
                    ]
                ),

                "away_uefa_country_floor_used": int(
                    uefa_row[
                        "away_uefa_country_floor_used"
                    ]
                ),

                "v1_home_probability": (
                    v1_home
                ),

                "v1_draw_probability": (
                    v1_draw
                ),

                "v1_away_probability": (
                    v1_away
                ),

                "v1_predicted_class": (
                    v1_class
                ),

                "v1_predicted_result": (
                    CLASS_TO_RESULT[
                        v1_class
                    ]
                ),

                "v1_predicted_selection": (
                    probability_label(
                        v1_class,
                        home_team,
                        away_team,
                    )
                ),

                "v1_max_probability": max(
                    v1_home,
                    v1_draw,
                    v1_away,
                ),

                "v2_home_probability": (
                    v2_home
                ),

                "v2_draw_probability": (
                    v2_draw
                ),

                "v2_away_probability": (
                    v2_away
                ),

                "v2_predicted_class": (
                    v2_class
                ),

                "v2_predicted_result": (
                    CLASS_TO_RESULT[
                        v2_class
                    ]
                ),

                "v2_predicted_selection": (
                    probability_label(
                        v2_class,
                        home_team,
                        away_team,
                    )
                ),

                "v2_max_probability": max(
                    v2_home,
                    v2_draw,
                    v2_away,
                ),

                "home_probability_delta_pp": (
                    home_delta_pp
                ),

                "draw_probability_delta_pp": (
                    draw_delta_pp
                ),

                "away_probability_delta_pp": (
                    away_delta_pp
                ),

                "max_absolute_probability_delta_pp": (
                    max_delta_pp
                ),

                "mean_absolute_probability_delta_pp": (
                    mean_abs_delta_pp
                ),

                "pick_changed": (
                    v1_class
                    !=
                    v2_class
                ),

                "elo_features_available": (
                    availability[
                        "elo"
                    ][
                        "available"
                    ]
                ),

                "ucl_form_features_available": (
                    availability[
                        "ucl_form"
                    ][
                        "available"
                    ]
                ),

                "domestic_features_available": (
                    availability[
                        "domestic"
                    ][
                        "available"
                    ]
                ),

                "squad_features_available": (
                    availability[
                        "squad"
                    ][
                        "available"
                    ]
                ),

                "v1_features_available": (
                    availability[
                        "overall"
                    ][
                        "available"
                    ]
                ),

                "v1_feature_availability": (
                    availability[
                        "overall"
                    ][
                        "rate"
                    ]
                ),

                "v2_features_available": (
                    availability[
                        "overall"
                    ][
                        "available"
                    ]
                    +
                    len(
                        LOCKED_UEFA_FEATURES
                    )
                ),

                "v2_feature_availability": (
                    (
                        availability[
                            "overall"
                        ][
                            "available"
                        ]
                        +
                        len(
                            LOCKED_UEFA_FEATURES
                        )
                    )
                    /
                    EXPECTED_V2_FEATURES
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = (
        comparison.sort_values(
            by=[
                "kickoff_utc",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    feature_matrix = pd.DataFrame(
        v2_feature_rows,
        columns=v2_features,
    )

    validate_shadow_outputs(
        comparison=comparison,
        feature_matrix=(
            feature_matrix
        ),
        v1_features=v1_features,
        v2_features=v2_features,
    )

    pick_changes = int(
        comparison[
            "pick_changed"
        ].sum()
    )

    overall_mean_abs_delta = float(
        comparison[
            [
                "home_probability_delta_pp",
                "draw_probability_delta_pp",
                "away_probability_delta_pp",
            ]
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
        .mean()
    )

    largest_row_index = (
        comparison[
            "max_absolute_probability_delta_pp"
        ].idxmax()
    )

    largest_row = comparison.loc[
        largest_row_index
    ]

    max_class_shift_pp = float(
        largest_row[
            "max_absolute_probability_delta_pp"
        ]
    )

    SHADOW_ARCHIVE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_dir = archive_paths[
        "directory"
    ]

    # This is the hard immutability boundary. The directory is
    # created with exist_ok=False so an existing shadow date can
    # never be silently revised.
    archive_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    try:
        comparison.to_csv(
            archive_paths[
                "comparison"
            ],
            index=False,
            encoding="utf-8",
        )

        feature_matrix.to_csv(
            archive_paths[
                "feature_matrix"
            ],
            index=False,
            encoding="utf-8",
        )

        protected_after = (
            snapshot_protected_files()
        )

        frozen_legacy_after = (
            snapshot_frozen_legacy_shadow_files()
        )

        verify_protected_files_unchanged(
            protected_before,
            protected_after,
        )

        verify_frozen_legacy_shadow_files_unchanged(
            frozen_legacy_before,
            frozen_legacy_after,
        )

        comparison_sha256 = sha256_file(
            archive_paths[
                "comparison"
            ]
        )

        feature_matrix_sha256 = sha256_file(
            archive_paths[
                "feature_matrix"
            ]
        )

        metadata = {
            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            "status": (
                "shadow_research_only"
            ),

            "production_promoted": (
                False
            ),

            "public_predictions_modified": (
                False
            ),

            "prediction_history_modified": (
                False
            ),

            "comparison_season_re_run": (
                False
            ),

            "v1_model_version": (
                V1_MODEL_VERSION
            ),

            "v2_model_version": (
                V2_MODEL_VERSION
            ),

            "v2_model_status": (
                V2_REQUIRED_STATUS
            ),

            "v1_feature_count": (
                EXPECTED_V1_FEATURES
            ),

            "v2_feature_count": (
                EXPECTED_V2_FEATURES
            ),

            "selected_uefa_features": (
                LOCKED_UEFA_FEATURES
            ),

            "live_feature_builder": (
                "src.ml.predict_live_fixed."
                "build_fixture_features"
            ),

            "fixture_identity": (
                "source_match_id"
            ),

            "fixture_count": int(
                len(
                    comparison
                )
            ),

            "fixture_date_utc": (
                fixture_date_utc
            ),

            "pick_change_count": (
                pick_changes
            ),

            "pick_change_rate": (
                pick_changes
                /
                len(
                    comparison
                )
            ),

            "overall_mean_absolute_probability_delta_pp": (
                overall_mean_abs_delta
            ),

            "maximum_class_probability_delta_pp": (
                max_class_shift_pp
            ),

            "maximum_shift_fixture": {
                "source_match_id": str(
                    largest_row[
                        "source_match_id"
                    ]
                ),

                "home_team": str(
                    largest_row[
                        "home_team"
                    ]
                ),

                "away_team": str(
                    largest_row[
                        "away_team"
                    ]
                ),
            },

            "archive": {
                "archive_id": (
                    fixture_date_utc
                ),

                "immutable": True,

                "overwrite_allowed": False,

                "directory": (
                    relative_path(
                        archive_paths[
                            "directory"
                        ]
                    )
                ),

                "comparison_sha256": (
                    comparison_sha256
                ),

                "feature_matrix_sha256": (
                    feature_matrix_sha256
                ),
            },

            "input_hashes": (
                input_hashes
            ),

            "output_files": {
                "comparison": (
                    relative_path(
                        archive_paths[
                            "comparison"
                        ]
                    )
                ),

                "feature_matrix": (
                    relative_path(
                        archive_paths[
                            "feature_matrix"
                        ]
                    )
                ),

                "metadata": (
                    relative_path(
                        archive_paths[
                            "metadata"
                        ]
                    )
                ),
            },

            "protected_production_files": {
                "unchanged": True,

                "files_checked": sorted(
                    protected_before.keys()
                ),
            },

            "frozen_legacy_shadow_files": {
                "unchanged": True,

                "files_checked": sorted(
                    frozen_legacy_before.keys()
                ),
            },

            "rules": {
                "v1_model_modified": False,

                "v1_metadata_modified": False,

                "production_prediction_files_modified": False,

                "public_app_files_modified": False,

                "prediction_history_modified": False,

                "production_release_called": False,

                "v2_promoted": False,

                "same_290_live_features_used_for_both_models": True,

                "v2_only_adds_locked_uefa_features": True,

                "fuzzy_uefa_matching_used": False,

                "2025_confirmation_re_run": False,

                "raw_probabilities_only": True,

                "probability_is_not_confidence": True,

                "shadow_results_are_not_performance_evidence": True,

                "immutable_archive_write": True,

                "existing_snapshot_overwrite_allowed": False,

                "legacy_first_shadow_snapshot_protected": True,
            },
        }

        with archive_paths[
            "metadata"
        ].open(
            "x",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

    except Exception:
        # If this run fails after reserving a new archive directory,
        # remove only the directory created by this process. Existing
        # archives are never touched because the date guard and
        # exist_ok=False prevent entering this block for them.
        if archive_dir.exists():
            shutil.rmtree(
                archive_dir
            )

        raise

    print_shadow_results(
        comparison
    )

    print()
    print(
        "=" * 90
    )

    print(
        "SHADOW SUMMARY"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "Fixtures compared: "
        f"{len(comparison)}"
    )

    print(
        "V1 features: "
        f"{EXPECTED_V1_FEATURES}"
    )

    print(
        "V2 features: "
        f"{EXPECTED_V2_FEATURES}"
    )

    print(
        "Pick changes: "
        f"{pick_changes}/"
        f"{len(comparison)}"
    )

    print(
        "Mean absolute H/D/A movement: "
        f"{overall_mean_abs_delta:.3f} pp"
    )

    print(
        "Largest single-class movement: "
        f"{max_class_shift_pp:.3f} pp"
    )

    print(
        "Largest movement fixture: "
        f"{largest_row['home_team']} vs "
        f"{largest_row['away_team']}"
    )

    print()
    print(
        "Exact 290-feature V1 prefix: PASS"
    )

    print(
        "Exact two-feature UEFA suffix: PASS"
    )

    print(
        "H/D/A probability integrity: PASS"
    )

    print(
        "Deterministic UEFA fixture identity: PASS"
    )

    print(
        "Timezone-safe live feature builder: PASS"
    )

    print(
        "Protected production files unchanged: PASS"
    )

    print(
        "V2 production promotion: NO"
    )

    print()
    print(
        "Shadow comparison:"
    )

    print(
        archive_paths[
            "comparison"
        ]
    )

    print()
    print(
        "Exact 292-feature shadow matrix:"
    )

    print(
        archive_paths[
            "feature_matrix"
        ]
    )

    print()
    print(
        "Shadow metadata:"
    )

    print(
        archive_paths[
            "metadata"
        ]
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These are raw probability comparisons."
    )

    print(
        "A probability shift does NOT prove "
        "that V2 is better."
    )

    print(
        "Future verified outcomes are required "
        "for genuine shadow performance evidence."
    )

    print()
    print(
        "=" * 90
    )

    print(
        "V2 SHADOW INFERENCE COMPLETED"
    )

    print(
        "PRODUCTION REMAINS V1"
    )

    print(
        "=" * 90
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run research-only V1 vs V2 "
            "UCL shadow inference."
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the full shadow "
            "input contract without "
            "generating predictions or "
            "writing output files."
        ),
    )

    args = parser.parse_args()

    if args.validate_only:
        run_validation_only()
        return

    run_shadow_inference()


if __name__ == "__main__":
    main()