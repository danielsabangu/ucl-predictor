from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.history.evaluation import (
    multiclass_brier_score,
    multiclass_log_loss,
)
from src.history.resolver import canonical_team_key
from src.history.results_archive import load_results


# ============================================================
# PURPOSE
# ============================================================
#
# Forward-only V1 vs V2 shadow evaluation.
#
# This evaluator is deliberately separate from:
#   - production prediction generation
#   - production history evaluation
#   - model training
#   - V2 promotion
#   - the already-consumed 2025/26 confirmation
#
# Evaluation is allowed ONLY when every fixture inside a frozen
# pre-match shadow snapshot has an immutable verified FINISHED
# result.
#
# Partial shadow sets are never scored.
#
# Primary probability metrics:
#   1. mean log loss
#   2. mean multiclass Brier
#
# Secondary metrics:
#   - accuracy
#   - macro F1
#   - draw recall
#   - mean probability assigned to the actual result
#
# The metric definitions for log loss and Brier are imported
# directly from src.history.evaluation.
#
# This module does NOT promote V2.
# ============================================================


# ============================================================
# PATHS / SETTINGS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "app"
    / "history"
    / "results.jsonl"
)


# The first V1-vs-V2 forward snapshot was created before the
# immutable date-scoped shadow archive existed.
#
# It is intentionally preserved at these legacy paths.
LEGACY_SHADOW_DATE = "2026-10-13"

LEGACY_COMPARISON_FILE = (
    PROJECT_ROOT
    / "reports"
    / "research"
    / "v2_shadow_next_matchday_comparison.csv"
)

LEGACY_METADATA_FILE = (
    PROJECT_ROOT
    / "reports"
    / "research"
    / "v2_shadow_next_matchday_metadata.json"
)


# All later shadow sets live here:
#
# reports/research/shadow/YYYY-MM-DD/
#   comparison.csv
#   features.csv
#   metadata.json
SHADOW_ARCHIVE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "research"
    / "shadow"
)


# Evaluation artifacts are separate from frozen prediction
# archives.
#
# reports/research/shadow_evaluations/YYYY-MM-DD/
#   fixture_evaluations.csv
#   summary.json
SHADOW_EVALUATION_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "research"
    / "shadow_evaluations"
)


EVALUATION_SCHEMA_VERSION = 1

VALID_RESULTS = (
    "H",
    "D",
    "A",
)

MIN_RELIABLE_SAMPLE = 30


REQUIRED_COMPARISON_COLUMNS = {
    "source_match_id",
    "kickoff_utc",
    "home_team",
    "away_team",

    "v1_home_probability",
    "v1_draw_probability",
    "v1_away_probability",
    "v1_predicted_result",

    "v2_home_probability",
    "v2_draw_probability",
    "v2_away_probability",
    "v2_predicted_result",
}


# ============================================================
# DATA CONTAINERS
# ============================================================


@dataclass(
    frozen=True
)
class ShadowInputPaths:
    comparison: Path
    metadata: Path


@dataclass(
    frozen=True
)
class ShadowEvaluationPaths:
    directory: Path
    fixture_evaluations: Path
    summary: Path


@dataclass(
    frozen=True
)
class ShadowEvaluationRun:
    status: str

    shadow_date: str

    fixture_count: int
    resolved_count: int

    missing_source_match_ids: tuple[
        str,
        ...
    ]

    missing_fixtures: tuple[
        str,
        ...
    ]

    evaluation: (
        pd.DataFrame
        | None
    )

    summary: (
        dict[
            str,
            Any,
        ]
        | None
    )

    paths: ShadowEvaluationPaths


# ============================================================
# BASIC HELPERS
# ============================================================


def sha256_file(
    path: Path,
) -> str:
    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024
                *
                1024
            )

            if not block:
                break

            hasher.update(
                block
            )

    return hasher.hexdigest()


def normalize_source_match_id(
    value: Any,
) -> str:
    if (
        value is None
        or
        pd.isna(
            value
        )
    ):
        raise ValueError(
            "source_match_id cannot be missing."
        )

    text = str(
        value
    ).strip()

    if not text:
        raise ValueError(
            "source_match_id cannot be blank."
        )

    try:
        numeric = float(
            text
        )

    except ValueError:
        return text

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

    return text


def normalize_shadow_date(
    value: str,
) -> str:
    try:
        parsed = date.fromisoformat(
            str(
                value
            ).strip()
        )

    except ValueError as exc:
        raise ValueError(
            "shadow_date must use "
            "YYYY-MM-DD format."
        ) from exc

    return parsed.isoformat()


def utc_timestamp(
    value: Any,
    *,
    name: str,
) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(
            value
        )

    except Exception as exc:
        raise ValueError(
            f"Invalid {name}: "
            f"{value}"
        ) from exc

    if pd.isna(
        timestamp
    ):
        raise ValueError(
            f"{name} cannot be missing."
        )

    if timestamp.tzinfo is None:
        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )

    else:
        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return timestamp


# ============================================================
# SHADOW PATH RESOLUTION
# ============================================================


def resolve_shadow_input_paths(
    shadow_date: str,
) -> ShadowInputPaths:
    normalized_date = (
        normalize_shadow_date(
            shadow_date
        )
    )

    if (
        normalized_date
        ==
        LEGACY_SHADOW_DATE
    ):
        return ShadowInputPaths(
            comparison=(
                LEGACY_COMPARISON_FILE
            ),
            metadata=(
                LEGACY_METADATA_FILE
            ),
        )

    archive_directory = (
        SHADOW_ARCHIVE_ROOT
        /
        normalized_date
    )

    return ShadowInputPaths(
        comparison=(
            archive_directory
            /
            "comparison.csv"
        ),

        metadata=(
            archive_directory
            /
            "metadata.json"
        ),
    )


def evaluation_output_paths(
    shadow_date: str,
    *,
    root: Path = (
        SHADOW_EVALUATION_ROOT
    ),
) -> ShadowEvaluationPaths:
    normalized_date = (
        normalize_shadow_date(
            shadow_date
        )
    )

    directory = (
        Path(
            root
        )
        /
        normalized_date
    )

    return ShadowEvaluationPaths(
        directory=directory,

        fixture_evaluations=(
            directory
            /
            "fixture_evaluations.csv"
        ),

        summary=(
            directory
            /
            "summary.json"
        ),
    )


def assert_outputs_do_not_exist(
    paths: ShadowEvaluationPaths,
) -> None:
    existing = [
        path
        for path in (
            paths.fixture_evaluations,
            paths.summary,
        )
        if path.exists()
    ]

    if existing:
        rendered = "\n".join(
            str(
                path
            )
            for path in existing
        )

        raise FileExistsError(
            "Shadow evaluation artifacts "
            "are immutable and already exist:\n"
            f"{rendered}"
        )


# ============================================================
# SHADOW SNAPSHOT VALIDATION
# ============================================================


def load_shadow_snapshot(
    *,
    shadow_date: str,
    comparison_path: Path,
    metadata_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        Any,
    ],
]:
    normalized_date = (
        normalize_shadow_date(
            shadow_date
        )
    )

    if not comparison_path.exists():
        raise FileNotFoundError(
            "Shadow comparison is missing:\n"
            f"{comparison_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Shadow metadata is missing:\n"
            f"{metadata_path}"
        )

    comparison = pd.read_csv(
        comparison_path,
        low_memory=False,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(
            handle
        )

    if comparison.empty:
        raise ValueError(
            "Shadow comparison is empty."
        )

    missing_columns = (
        REQUIRED_COMPARISON_COLUMNS
        -
        set(
            comparison.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Shadow comparison is missing "
            "required columns:\n"
            f"{sorted(missing_columns)}"
        )

    comparison = comparison.copy()

    comparison[
        "source_match_id"
    ] = comparison[
        "source_match_id"
    ].map(
        normalize_source_match_id
    )

    if comparison[
        "source_match_id"
    ].duplicated().any():
        raise ValueError(
            "Shadow comparison contains "
            "duplicate source_match_id values."
        )

    comparison[
        "kickoff_utc"
    ] = pd.to_datetime(
        comparison[
            "kickoff_utc"
        ],
        utc=True,
        errors="raise",
    )

    observed_dates = {
        timestamp.date().isoformat()
        for timestamp in comparison[
            "kickoff_utc"
        ]
    }

    if (
        observed_dates
        !=
        {
            normalized_date
        }
    ):
        raise ValueError(
            "Shadow comparison kickoff dates "
            "do not match requested shadow date "
            f"{normalized_date}: "
            f"{sorted(observed_dates)}"
        )

    if (
        metadata.get(
            "status"
        )
        !=
        "shadow_research_only"
    ):
        raise ValueError(
            "Shadow metadata status is not "
            "shadow_research_only."
        )

    if (
        metadata.get(
            "production_promoted"
        )
        is not False
    ):
        raise ValueError(
            "Shadow metadata does not explicitly "
            "confirm production_promoted=false."
        )

    fixture_count = metadata.get(
        "fixture_count"
    )

    if fixture_count is None:
        raise ValueError(
            "Shadow metadata has no "
            "fixture_count."
        )

    if (
        int(
            fixture_count
        )
        !=
        len(
            comparison
        )
    ):
        raise ValueError(
            "Shadow metadata fixture_count "
            "does not match comparison rows."
        )

    metadata_date = str(
        metadata.get(
            "fixture_date_utc",
            "",
        )
    ).strip()

    if (
        metadata_date
        !=
        normalized_date
    ):
        raise ValueError(
            "Shadow metadata fixture_date_utc "
            "does not match requested shadow date."
        )

    created_at_raw = (
        metadata.get(
            "created_at_utc"
        )
    )

    if not created_at_raw:
        raise ValueError(
            "Shadow metadata has no "
            "created_at_utc timestamp."
        )

    created_at = utc_timestamp(
        created_at_raw,
        name="created_at_utc",
    )

    first_kickoff = comparison[
        "kickoff_utc"
    ].min()

    if (
        created_at
        >=
        first_kickoff
    ):
        raise ValueError(
            "Shadow snapshot was not created "
            "before the first fixture kickoff. "
            "It is not eligible for forward "
            "evaluation."
        )

    for prefix in (
        "v1",
        "v2",
    ):
        probability_columns = [
            f"{prefix}_home_probability",
            f"{prefix}_draw_probability",
            f"{prefix}_away_probability",
        ]

        probabilities = comparison[
            probability_columns
        ].apply(
            pd.to_numeric,
            errors="raise",
        )

        values = (
            probabilities.to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{prefix.upper()} "
                "probabilities contain "
                "non-finite values."
            )

        if (
            (
                values
                <
                0.0
            )
            |
            (
                values
                >
                1.0
            )
        ).any():
            raise ValueError(
                f"{prefix.upper()} "
                "probabilities fall outside "
                "[0, 1]."
            )

        if not np.allclose(
            values.sum(
                axis=1
            ),
            1.0,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"{prefix.upper()} "
                "probabilities do not sum "
                "to 1."
            )

        predicted = comparison[
            f"{prefix}_predicted_result"
        ].astype(
            str
        )

        if not predicted.isin(
            VALID_RESULTS
        ).all():
            raise ValueError(
                f"{prefix.upper()} "
                "predicted_result contains "
                "a non H/D/A value."
            )

        implied = np.array(
            VALID_RESULTS,
            dtype=object,
        )[
            np.argmax(
                values,
                axis=1,
            )
        ]

        if not np.array_equal(
            predicted.to_numpy(
                dtype=object
            ),
            implied,
        ):
            raise ValueError(
                f"{prefix.upper()} "
                "predicted_result does not "
                "match the highest probability."
            )

    return (
        comparison,
        metadata,
    )


# ============================================================
# VERIFIED RESULT MATCHING
# ============================================================


def result_records_to_dataframe(
    records: Iterable[
        Any
    ],
) -> pd.DataFrame:
    rows = []

    for record in records:
        rows.append(
            {
                "result_id": str(
                    record.result_id
                ),

                "source_match_id": (
                    normalize_source_match_id(
                        record.source_match_id
                    )
                ),

                "fixture_id": str(
                    record.fixture_id
                ),

                "home_team": str(
                    record.home_team
                ),

                "away_team": str(
                    record.away_team
                ),

                "kickoff_utc": (
                    utc_timestamp(
                        record.kickoff_utc,
                        name=(
                            "result kickoff_utc"
                        ),
                    )
                ),

                "status": str(
                    record.status
                ),

                "home_goals": int(
                    record.home_goals
                ),

                "away_goals": int(
                    record.away_goals
                ),

                "actual_result": str(
                    record.actual_result
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def match_verified_results(
    *,
    comparison: pd.DataFrame,
    result_records: Iterable[
        Any
    ],
) -> tuple[
    pd.DataFrame,
    list[
        str
    ],
    list[
        str
    ],
]:
    results = (
        result_records_to_dataframe(
            result_records
        )
    )

    if results.empty:
        missing_ids = (
            comparison[
                "source_match_id"
            ]
            .astype(
                str
            )
            .tolist()
        )

        missing_fixtures = (
            (
                comparison[
                    "home_team"
                ].astype(
                    str
                )
                +
                " vs "
                +
                comparison[
                    "away_team"
                ].astype(
                    str
                )
            )
            .tolist()
        )

        return (
            pd.DataFrame(),
            missing_ids,
            missing_fixtures,
        )

    if results[
        "source_match_id"
    ].duplicated().any():
        duplicates = sorted(
            results.loc[
                results[
                    "source_match_id"
                ].duplicated(
                    keep=False
                ),
                "source_match_id",
            ].unique()
        )

        raise ValueError(
            "Result archive contains "
            "duplicate provider match IDs:\n"
            f"{duplicates}"
        )

    results_by_id = (
        results.set_index(
            "source_match_id",
            drop=False,
        )
    )

    matched_rows = []
    missing_ids = []
    missing_fixtures = []

    for shadow in (
        comparison.to_dict(
            orient="records"
        )
    ):
        source_match_id = (
            normalize_source_match_id(
                shadow[
                    "source_match_id"
                ]
            )
        )

        if (
            source_match_id
            not in
            results_by_id.index
        ):
            missing_ids.append(
                source_match_id
            )

            missing_fixtures.append(
                f"{shadow['home_team']} "
                f"vs "
                f"{shadow['away_team']}"
            )

            continue

        result = results_by_id.loc[
            source_match_id
        ]

        if (
            str(
                result[
                    "status"
                ]
            ).upper()
            !=
            "FINISHED"
        ):
            raise ValueError(
                "Matched result record is not "
                "FINISHED for source_match_id="
                f"{source_match_id}."
            )

        actual_result = str(
            result[
                "actual_result"
            ]
        ).upper()

        if (
            actual_result
            not in
            VALID_RESULTS
        ):
            raise ValueError(
                "Matched result has invalid "
                "actual_result for "
                "source_match_id="
                f"{source_match_id}."
            )

        home_goals = int(
            result[
                "home_goals"
            ]
        )

        away_goals = int(
            result[
                "away_goals"
            ]
        )

        if home_goals > away_goals:
            calculated_result = "H"

        elif home_goals < away_goals:
            calculated_result = "A"

        else:
            calculated_result = "D"

        if (
            calculated_result
            !=
            actual_result
        ):
            raise ValueError(
                "Matched result label disagrees "
                "with score for source_match_id="
                f"{source_match_id}."
            )

        shadow_home = (
            canonical_team_key(
                str(
                    shadow[
                        "home_team"
                    ]
                )
            )
        )

        result_home = (
            canonical_team_key(
                str(
                    result[
                        "home_team"
                    ]
                )
            )
        )

        if (
            shadow_home
            !=
            result_home
        ):
            raise ValueError(
                "Home-team identity mismatch "
                "for source_match_id="
                f"{source_match_id}."
            )

        shadow_away = (
            canonical_team_key(
                str(
                    shadow[
                        "away_team"
                    ]
                )
            )
        )

        result_away = (
            canonical_team_key(
                str(
                    result[
                        "away_team"
                    ]
                )
            )
        )

        if (
            shadow_away
            !=
            result_away
        ):
            raise ValueError(
                "Away-team identity mismatch "
                "for source_match_id="
                f"{source_match_id}."
            )

        shadow_kickoff = (
            utc_timestamp(
                shadow[
                    "kickoff_utc"
                ],
                name=(
                    "shadow kickoff_utc"
                ),
            )
        )

        result_kickoff = (
            result[
                "kickoff_utc"
            ]
        )

        if (
            shadow_kickoff
            !=
            result_kickoff
        ):
            raise ValueError(
                "Kickoff identity mismatch "
                "for source_match_id="
                f"{source_match_id}."
            )

        matched_rows.append(
            {
                **shadow,

                "result_id": str(
                    result[
                        "result_id"
                    ]
                ),

                "fixture_id": str(
                    result[
                        "fixture_id"
                    ]
                ),

                "home_goals": (
                    home_goals
                ),

                "away_goals": (
                    away_goals
                ),

                "actual_result": (
                    actual_result
                ),
            }
        )

    matched = pd.DataFrame(
        matched_rows
    )

    if (
        len(
            matched
        )
        +
        len(
            missing_ids
        )
        !=
        len(
            comparison
        )
    ):
        raise RuntimeError(
            "Verified-result matching "
            "cardinality error."
        )

    return (
        matched,
        missing_ids,
        missing_fixtures,
    )


# ============================================================
# FIXTURE-LEVEL METRICS
# ============================================================


def model_fixture_metrics(
    row: pd.Series,
    *,
    prefix: str,
) -> dict[
    str,
    Any,
]:
    actual_result = str(
        row[
            "actual_result"
        ]
    )

    home_probability = float(
        row[
            f"{prefix}_home_probability"
        ]
    )

    draw_probability = float(
        row[
            f"{prefix}_draw_probability"
        ]
    )

    away_probability = float(
        row[
            f"{prefix}_away_probability"
        ]
    )

    probability_by_result = {
        "H": home_probability,
        "D": draw_probability,
        "A": away_probability,
    }

    actual_probability = (
        probability_by_result[
            actual_result
        ]
    )

    predicted_result = str(
        row[
            f"{prefix}_predicted_result"
        ]
    )

    return {
        "actual_result_probability": (
            actual_probability
        ),

        "pick_correct": (
            predicted_result
            ==
            actual_result
        ),

        "multiclass_brier": (
            multiclass_brier_score(
                home_probability=(
                    home_probability
                ),
                draw_probability=(
                    draw_probability
                ),
                away_probability=(
                    away_probability
                ),
                actual_result=(
                    actual_result
                ),
            )
        ),

        "log_loss": (
            multiclass_log_loss(
                home_probability=(
                    home_probability
                ),
                draw_probability=(
                    draw_probability
                ),
                away_probability=(
                    away_probability
                ),
                actual_result=(
                    actual_result
                ),
            )
        ),
    }


def winner_higher(
    delta: float,
) -> str:
    if delta > 1e-15:
        return "V2"

    if delta < -1e-15:
        return "V1"

    return "TIE"


def winner_lower(
    delta: float,
) -> str:
    if delta < -1e-15:
        return "V2"

    if delta > 1e-15:
        return "V1"

    return "TIE"


def build_fixture_evaluations(
    matched: pd.DataFrame,
) -> pd.DataFrame:
    if matched.empty:
        raise ValueError(
            "Cannot evaluate an empty "
            "matched result set."
        )

    rows = []

    for _, row in matched.iterrows():
        v1 = model_fixture_metrics(
            row,
            prefix="v1",
        )

        v2 = model_fixture_metrics(
            row,
            prefix="v2",
        )

        actual_probability_delta = (
            v2[
                "actual_result_probability"
            ]
            -
            v1[
                "actual_result_probability"
            ]
        )

        log_loss_delta = (
            v2[
                "log_loss"
            ]
            -
            v1[
                "log_loss"
            ]
        )

        brier_delta = (
            v2[
                "multiclass_brier"
            ]
            -
            v1[
                "multiclass_brier"
            ]
        )

        rows.append(
            {
                "source_match_id": str(
                    row[
                        "source_match_id"
                    ]
                ),

                "result_id": str(
                    row[
                        "result_id"
                    ]
                ),

                "fixture_id": str(
                    row[
                        "fixture_id"
                    ]
                ),

                "kickoff_utc": (
                    utc_timestamp(
                        row[
                            "kickoff_utc"
                        ],
                        name="kickoff_utc",
                    )
                    .isoformat()
                ),

                "home_team": str(
                    row[
                        "home_team"
                    ]
                ),

                "away_team": str(
                    row[
                        "away_team"
                    ]
                ),

                "home_goals": int(
                    row[
                        "home_goals"
                    ]
                ),

                "away_goals": int(
                    row[
                        "away_goals"
                    ]
                ),

                "actual_result": str(
                    row[
                        "actual_result"
                    ]
                ),

                "v1_predicted_result": str(
                    row[
                        "v1_predicted_result"
                    ]
                ),

                "v2_predicted_result": str(
                    row[
                        "v2_predicted_result"
                    ]
                ),

                "v1_actual_result_probability": (
                    float(
                        v1[
                            "actual_result_probability"
                        ]
                    )
                ),

                "v2_actual_result_probability": (
                    float(
                        v2[
                            "actual_result_probability"
                        ]
                    )
                ),

                "actual_result_probability_delta_v2_minus_v1": (
                    float(
                        actual_probability_delta
                    )
                ),

                "actual_result_probability_winner": (
                    winner_higher(
                        actual_probability_delta
                    )
                ),

                "v1_pick_correct": bool(
                    v1[
                        "pick_correct"
                    ]
                ),

                "v2_pick_correct": bool(
                    v2[
                        "pick_correct"
                    ]
                ),

                "v1_multiclass_brier": (
                    float(
                        v1[
                            "multiclass_brier"
                        ]
                    )
                ),

                "v2_multiclass_brier": (
                    float(
                        v2[
                            "multiclass_brier"
                        ]
                    )
                ),

                "brier_delta_v2_minus_v1": (
                    float(
                        brier_delta
                    )
                ),

                "brier_winner": (
                    winner_lower(
                        brier_delta
                    )
                ),

                "v1_log_loss": float(
                    v1[
                        "log_loss"
                    ]
                ),

                "v2_log_loss": float(
                    v2[
                        "log_loss"
                    ]
                ),

                "log_loss_delta_v2_minus_v1": (
                    float(
                        log_loss_delta
                    )
                ),

                "log_loss_winner": (
                    winner_lower(
                        log_loss_delta
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================


def classification_metrics(
    *,
    actual: list[
        str
    ],
    predicted: list[
        str
    ],
) -> dict[
    str,
    float
    | None,
]:
    if (
        len(
            actual
        )
        !=
        len(
            predicted
        )
    ):
        raise ValueError(
            "actual and predicted "
            "lengths differ."
        )

    if not actual:
        raise ValueError(
            "Cannot calculate classification "
            "metrics on an empty sample."
        )

    if any(
        label
        not in
        VALID_RESULTS
        for label in actual
    ):
        raise ValueError(
            "actual contains invalid "
            "result labels."
        )

    if any(
        label
        not in
        VALID_RESULTS
        for label in predicted
    ):
        raise ValueError(
            "predicted contains invalid "
            "result labels."
        )

    accuracy = (
        sum(
            actual_label
            ==
            predicted_label
            for (
                actual_label,
                predicted_label,
            ) in zip(
                actual,
                predicted,
            )
        )
        /
        len(
            actual
        )
    )

    f1_scores = []

    for label in VALID_RESULTS:
        true_positive = sum(
            (
                actual_label
                ==
                label
            )
            and
            (
                predicted_label
                ==
                label
            )
            for (
                actual_label,
                predicted_label,
            ) in zip(
                actual,
                predicted,
            )
        )

        false_positive = sum(
            (
                actual_label
                !=
                label
            )
            and
            (
                predicted_label
                ==
                label
            )
            for (
                actual_label,
                predicted_label,
            ) in zip(
                actual,
                predicted,
            )
        )

        false_negative = sum(
            (
                actual_label
                ==
                label
            )
            and
            (
                predicted_label
                !=
                label
            )
            for (
                actual_label,
                predicted_label,
            ) in zip(
                actual,
                predicted,
            )
        )

        denominator = (
            (
                2
                *
                true_positive
            )
            +
            false_positive
            +
            false_negative
        )

        if denominator:
            f1 = (
                (
                    2
                    *
                    true_positive
                )
                /
                denominator
            )

        else:
            f1 = 0.0

        f1_scores.append(
            float(
                f1
            )
        )

    draw_actual = sum(
        result
        ==
        "D"
        for result in actual
    )

    draw_true_positive = sum(
        (
            actual_label
            ==
            "D"
        )
        and
        (
            predicted_label
            ==
            "D"
        )
        for (
            actual_label,
            predicted_label,
        ) in zip(
            actual,
            predicted,
        )
    )

    if draw_actual:
        draw_recall = (
            draw_true_positive
            /
            draw_actual
        )

    else:
        draw_recall = None

    return {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            mean(
                f1_scores
            )
        ),

        "draw_recall": (
            float(
                draw_recall
            )
            if draw_recall is not None
            else None
        ),
    }


# ============================================================
# MODEL SUMMARY
# ============================================================


def model_summary(
    fixture_evaluations: pd.DataFrame,
    *,
    prefix: str,
) -> dict[
    str,
    Any,
]:
    actual = (
        fixture_evaluations[
            "actual_result"
        ]
        .astype(
            str
        )
        .tolist()
    )

    predicted = (
        fixture_evaluations[
            f"{prefix}_predicted_result"
        ]
        .astype(
            str
        )
        .tolist()
    )

    classification = (
        classification_metrics(
            actual=actual,
            predicted=predicted,
        )
    )

    return {
        "mean_log_loss": float(
            fixture_evaluations[
                f"{prefix}_log_loss"
            ].mean()
        ),

        "mean_multiclass_brier": float(
            fixture_evaluations[
                f"{prefix}_multiclass_brier"
            ].mean()
        ),

        "accuracy": (
            classification[
                "accuracy"
            ]
        ),

        "macro_f1": (
            classification[
                "macro_f1"
            ]
        ),

        "draw_recall": (
            classification[
                "draw_recall"
            ]
        ),

        "mean_actual_result_probability": float(
            fixture_evaluations[
                f"{prefix}_actual_result_probability"
            ].mean()
        ),

        "correct_picks": int(
            fixture_evaluations[
                f"{prefix}_pick_correct"
            ].sum()
        ),
    }


# ============================================================
# LOCKED SUMMARY
# ============================================================


def build_summary(
    *,
    shadow_date: str,
    fixture_evaluations: pd.DataFrame,
    comparison_hash: str,
    metadata_hash: str,
    results_hash: str,
) -> dict[
    str,
    Any,
]:
    sample_size = len(
        fixture_evaluations
    )

    if sample_size == 0:
        raise ValueError(
            "Cannot summarize an empty "
            "evaluation."
        )

    v1 = model_summary(
        fixture_evaluations,
        prefix="v1",
    )

    v2 = model_summary(
        fixture_evaluations,
        prefix="v2",
    )

    if (
        v1[
            "draw_recall"
        ]
        is not None
        and
        v2[
            "draw_recall"
        ]
        is not None
    ):
        draw_recall_delta = (
            v2[
                "draw_recall"
            ]
            -
            v1[
                "draw_recall"
            ]
        )

    else:
        draw_recall_delta = None

    deltas = {
        "mean_log_loss_v2_minus_v1": (
            v2[
                "mean_log_loss"
            ]
            -
            v1[
                "mean_log_loss"
            ]
        ),

        "mean_multiclass_brier_v2_minus_v1": (
            v2[
                "mean_multiclass_brier"
            ]
            -
            v1[
                "mean_multiclass_brier"
            ]
        ),

        "accuracy_v2_minus_v1": (
            v2[
                "accuracy"
            ]
            -
            v1[
                "accuracy"
            ]
        ),

        "macro_f1_v2_minus_v1": (
            v2[
                "macro_f1"
            ]
            -
            v1[
                "macro_f1"
            ]
        ),

        "draw_recall_v2_minus_v1": (
            draw_recall_delta
        ),

        "mean_actual_result_probability_v2_minus_v1": (
            v2[
                "mean_actual_result_probability"
            ]
            -
            v1[
                "mean_actual_result_probability"
            ]
        ),
    }

    return {
        "evaluation_schema_version": (
            EVALUATION_SCHEMA_VERSION
        ),

        "status": (
            "COMPLETE"
        ),

        "evaluation_type": (
            "V1_VS_V2_FORWARD_SHADOW"
        ),

        "shadow_date": (
            normalize_shadow_date(
                shadow_date
            )
        ),

        "sample_size": (
            sample_size
        ),

        "all_shadow_fixtures_resolved": (
            True
        ),

        "production_promoted": (
            False
        ),

        "promotion_decision_made": (
            False
        ),

        "primary_metrics": [
            "mean_log_loss",
            "mean_multiclass_brier",
        ],

        "metric_direction": {
            "mean_log_loss": (
                "LOWER_IS_BETTER"
            ),

            "mean_multiclass_brier": (
                "LOWER_IS_BETTER"
            ),

            "accuracy": (
                "HIGHER_IS_BETTER"
            ),

            "macro_f1": (
                "HIGHER_IS_BETTER"
            ),

            "draw_recall": (
                "HIGHER_IS_BETTER"
            ),

            "mean_actual_result_probability": (
                "HIGHER_IS_BETTER"
            ),
        },

        "v1": (
            v1
        ),

        "v2": (
            v2
        ),

        "v2_minus_v1": (
            deltas
        ),

        "fixture_level": {
            "v2_better_log_loss_count": int(
                (
                    fixture_evaluations[
                        "log_loss_winner"
                    ]
                    ==
                    "V2"
                ).sum()
            ),

            "v1_better_log_loss_count": int(
                (
                    fixture_evaluations[
                        "log_loss_winner"
                    ]
                    ==
                    "V1"
                ).sum()
            ),

            "log_loss_tie_count": int(
                (
                    fixture_evaluations[
                        "log_loss_winner"
                    ]
                    ==
                    "TIE"
                ).sum()
            ),

            "v2_higher_actual_probability_count": int(
                (
                    fixture_evaluations[
                        "actual_result_probability_winner"
                    ]
                    ==
                    "V2"
                ).sum()
            ),

            "v1_higher_actual_probability_count": int(
                (
                    fixture_evaluations[
                        "actual_result_probability_winner"
                    ]
                    ==
                    "V1"
                ).sum()
            ),

            "actual_probability_tie_count": int(
                (
                    fixture_evaluations[
                        "actual_result_probability_winner"
                    ]
                    ==
                    "TIE"
                ).sum()
            ),
        },

        "sample_warning": (
            (
                "INSUFFICIENT_SAMPLE_FOR_"
                "RELIABLE_PERFORMANCE_CLAIMS"
            )
            if (
                sample_size
                <
                MIN_RELIABLE_SAMPLE
            )
            else None
        ),

        "promotion_note": (
            "This evaluator records forward "
            "evidence only. It does not promote "
            "V2. A single shadow set is not a "
            "promotion decision."
        ),

        "input_sha256": {
            "shadow_comparison": (
                comparison_hash
            ),

            "shadow_metadata": (
                metadata_hash
            ),

            "verified_results_archive": (
                results_hash
            ),
        },

        "metric_definitions": {
            "mean_log_loss": (
                "Mean negative natural logarithm "
                "of the probability assigned to "
                "the verified H/D/A result. Lower "
                "is better. Uses the existing "
                "src.history.evaluation definition."
            ),

            "mean_multiclass_brier": (
                "Mean sum of squared H/D/A "
                "probability errors. Range 0 to 2. "
                "Lower is better. Uses the existing "
                "src.history.evaluation definition."
            ),

            "accuracy": (
                "Share of fixtures where the "
                "highest-probability H/D/A "
                "selection equals the verified "
                "result."
            ),

            "macro_f1": (
                "Unweighted mean of H, D and A "
                "F1 scores. All three classes are "
                "always included. A class with no "
                "precision/recall denominator "
                "contributes 0."
            ),

            "draw_recall": (
                "Share of actual draws predicted "
                "as draws. Null when the evaluated "
                "set contains no draws."
            ),

            "mean_actual_result_probability": (
                "Mean probability assigned to the "
                "verified H/D/A result. Higher is "
                "better."
            ),
        },
    }


# ============================================================
# EVALUATION RUNNER
# ============================================================


def evaluate_shadow_set(
    *,
    shadow_date: str,
    comparison_path: Path,
    metadata_path: Path,
    results_path: Path,
    output_paths: ShadowEvaluationPaths,
    write_artifacts: bool,
) -> ShadowEvaluationRun:
    normalized_date = (
        normalize_shadow_date(
            shadow_date
        )
    )

    comparison_hash_before = (
        sha256_file(
            comparison_path
        )
    )

    metadata_hash_before = (
        sha256_file(
            metadata_path
        )
    )

    comparison, _metadata = (
        load_shadow_snapshot(
            shadow_date=(
                normalized_date
            ),
            comparison_path=(
                comparison_path
            ),
            metadata_path=(
                metadata_path
            ),
        )
    )

    result_records = (
        load_results(
            results_path
        )
    )

    (
        matched,
        missing_ids,
        missing_fixtures,
    ) = match_verified_results(
        comparison=(
            comparison
        ),
        result_records=(
            result_records
        ),
    )

    # --------------------------------------------------------
    # CRITICAL FORWARD-EVALUATION GATE
    # --------------------------------------------------------
    #
    # If even one frozen fixture lacks a verified immutable
    # result:
    #
    #   - no metrics are calculated
    #   - no evaluation table is constructed
    #   - no files are written
    #
    # This prevents partial-result peeking.
    # --------------------------------------------------------

    if missing_ids:
        return ShadowEvaluationRun(
            status=(
                "WAITING_FOR_RESULTS"
            ),

            shadow_date=(
                normalized_date
            ),

            fixture_count=(
                len(
                    comparison
                )
            ),

            resolved_count=(
                len(
                    matched
                )
            ),

            missing_source_match_ids=(
                tuple(
                    missing_ids
                )
            ),

            missing_fixtures=(
                tuple(
                    missing_fixtures
                )
            ),

            evaluation=None,
            summary=None,
            paths=(
                output_paths
            ),
        )

    if (
        len(
            matched
        )
        !=
        len(
            comparison
        )
    ):
        raise RuntimeError(
            "Complete shadow result set "
            "failed cardinality check."
        )

    fixture_evaluations = (
        build_fixture_evaluations(
            matched
        )
    )

    if results_path.exists():
        results_hash = (
            sha256_file(
                results_path
            )
        )

    else:
        # This branch should only be reachable if a caller
        # injects result records during testing.
        results_hash = (
            hashlib.sha256(
                b""
            ).hexdigest()
        )

    summary = build_summary(
        shadow_date=(
            normalized_date
        ),

        fixture_evaluations=(
            fixture_evaluations
        ),

        comparison_hash=(
            comparison_hash_before
        ),

        metadata_hash=(
            metadata_hash_before
        ),

        results_hash=(
            results_hash
        ),
    )

    # Confirm frozen prediction evidence did not change during
    # evaluation.
    comparison_hash_after = (
        sha256_file(
            comparison_path
        )
    )

    metadata_hash_after = (
        sha256_file(
            metadata_path
        )
    )

    if (
        comparison_hash_after
        !=
        comparison_hash_before
    ):
        raise RuntimeError(
            "Shadow comparison changed "
            "during evaluation."
        )

    if (
        metadata_hash_after
        !=
        metadata_hash_before
    ):
        raise RuntimeError(
            "Shadow metadata changed "
            "during evaluation."
        )

    if write_artifacts:
        assert_outputs_do_not_exist(
            output_paths
        )

        output_paths.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        fixture_evaluations.to_csv(
            output_paths.fixture_evaluations,
            index=False,
            encoding="utf-8",
        )

        with output_paths.summary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                summary,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )

            handle.write(
                "\n"
            )

    return ShadowEvaluationRun(
        status=(
            "COMPLETE"
        ),

        shadow_date=(
            normalized_date
        ),

        fixture_count=(
            len(
                comparison
            )
        ),

        resolved_count=(
            len(
                matched
            )
        ),

        missing_source_match_ids=(
            tuple()
        ),

        missing_fixtures=(
            tuple()
        ),

        evaluation=(
            fixture_evaluations
        ),

        summary=(
            summary
        ),

        paths=(
            output_paths
        ),
    )


def run_for_date(
    *,
    shadow_date: str,
    results_path: Path = (
        DEFAULT_RESULTS_FILE
    ),
    output_root: Path = (
        SHADOW_EVALUATION_ROOT
    ),
    write_artifacts: bool = True,
) -> ShadowEvaluationRun:
    inputs = (
        resolve_shadow_input_paths(
            shadow_date
        )
    )

    outputs = (
        evaluation_output_paths(
            shadow_date,
            root=(
                output_root
            ),
        )
    )

    return evaluate_shadow_set(
        shadow_date=(
            shadow_date
        ),

        comparison_path=(
            inputs.comparison
        ),

        metadata_path=(
            inputs.metadata
        ),

        results_path=(
            Path(
                results_path
            )
        ),

        output_paths=(
            outputs
        ),

        write_artifacts=(
            write_artifacts
        ),
    )


# ============================================================
# TERMINAL REPORT
# ============================================================


def print_run(
    run: ShadowEvaluationRun,
) -> None:
    print(
        "=" * 90
    )

    print(
        "UCL PREDICTOR - "
        "V1 VS V2 FORWARD SHADOW EVALUATION"
    )

    print(
        "=" * 90
    )

    print()

    print(
        "Shadow date: "
        f"{run.shadow_date}"
    )

    print(
        "Frozen fixtures: "
        f"{run.fixture_count}"
    )

    print(
        "Verified results: "
        f"{run.resolved_count}"
    )

    print()

    if (
        run.status
        ==
        "WAITING_FOR_RESULTS"
    ):
        print(
            "SHADOW STATUS: "
            "WAITING_FOR_RESULTS"
        )

        print(
            "Metrics calculated: NO"
        )

        print(
            "Evaluation files written: NO"
        )

        print()

        print(
            "Missing fixtures:"
        )

        for fixture in (
            run.missing_fixtures
        ):
            print(
                "  - "
                f"{fixture}"
            )

        print()

        print(
            "Production promotion: NO"
        )

        return

    if run.summary is None:
        raise RuntimeError(
            "Complete run has no summary."
        )

    summary = run.summary

    print(
        "SHADOW STATUS: COMPLETE"
    )

    print(
        "All frozen fixtures resolved: YES"
    )

    print()

    print(
        "PRIMARY PROBABILITY METRICS"
    )

    print(
        "Log loss: "
        f"V1 "
        f"{summary['v1']['mean_log_loss']:.6f}"
        " | "
        f"V2 "
        f"{summary['v2']['mean_log_loss']:.6f}"
        " | "
        "V2-V1 "
        f"{summary['v2_minus_v1']['mean_log_loss_v2_minus_v1']:+.6f}"
    )

    print(
        "Brier:    "
        f"V1 "
        f"{summary['v1']['mean_multiclass_brier']:.6f}"
        " | "
        f"V2 "
        f"{summary['v2']['mean_multiclass_brier']:.6f}"
        " | "
        "V2-V1 "
        f"{summary['v2_minus_v1']['mean_multiclass_brier_v2_minus_v1']:+.6f}"
    )

    print()

    print(
        "SECONDARY METRICS"
    )

    print(
        "Accuracy: "
        f"V1 "
        f"{summary['v1']['accuracy']:.2%}"
        " | "
        f"V2 "
        f"{summary['v2']['accuracy']:.2%}"
    )

    print(
        "Macro F1: "
        f"V1 "
        f"{summary['v1']['macro_f1']:.6f}"
        " | "
        f"V2 "
        f"{summary['v2']['macro_f1']:.6f}"
    )

    v1_draw = (
        summary[
            "v1"
        ][
            "draw_recall"
        ]
    )

    v2_draw = (
        summary[
            "v2"
        ][
            "draw_recall"
        ]
    )

    if v1_draw is None:
        print(
            "Draw recall: "
            "N/A - no actual draws "
            "in this set"
        )

    else:
        print(
            "Draw recall: "
            f"V1 {v1_draw:.2%}"
            " | "
            f"V2 {v2_draw:.2%}"
        )

    print(
        "Mean actual-result probability: "
        f"V1 "
        f"{summary['v1']['mean_actual_result_probability']:.2%}"
        " | "
        f"V2 "
        f"{summary['v2']['mean_actual_result_probability']:.2%}"
    )

    print()

    if summary[
        "sample_warning"
    ]:
        print(
            "WARNING: sample is too small "
            "for reliable performance claims."
        )

    print(
        "Production promotion: NO"
    )

    print(
        "Promotion decision made: NO"
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen V1-vs-V2 UCL "
            "shadow snapshot only after every "
            "fixture has an immutable verified "
            "FINISHED result."
        )
    )

    parser.add_argument(
        "--shadow-date",
        required=True,
        help=(
            "Frozen shadow fixture date "
            "in YYYY-MM-DD format."
        ),
    )

    parser.add_argument(
        "--results",
        default=str(
            DEFAULT_RESULTS_FILE
        ),
        help=(
            "Immutable verified result "
            "archive JSONL."
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            SHADOW_EVALUATION_ROOT
        ),
        help=(
            "Root directory for immutable "
            "shadow evaluations."
        ),
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Run the result gate and, if "
            "complete, calculate metrics in "
            "memory without writing artifacts."
        ),
    )

    args = parser.parse_args()

    run = run_for_date(
        shadow_date=(
            args.shadow_date
        ),

        results_path=(
            Path(
                args.results
            )
        ),

        output_root=(
            Path(
                args.output_root
            )
        ),

        write_artifacts=(
            not args.no_write
        ),
    )

    print_run(
        run
    )


if __name__ == "__main__":
    main()