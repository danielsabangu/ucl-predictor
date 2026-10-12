from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BASE_DIR,
    FOOTBALL_DATA_API_KEY,
    INTERIM_DATA_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
)


# ============================================================
# PRODUCTION SETTINGS
# ============================================================

PIPELINE_NAME = "ucl_predictor_live_pipeline"
PIPELINE_VERSION = "1.0.0"
MODEL_VERSION = "ucl_predictor_v1"

EXPECTED_FEATURE_COUNT = 290
EXPECTED_HDA_CLASSES = 3

STATUS_FILE = (
    PREDICTIONS_DIR
    / "live_pipeline_status.json"
)

LATEST_LOG_FILE = (
    REPORTS_DIR
    / "pipeline"
    / "latest_live_pipeline.log"
)

LOG_ARCHIVE_DIR = (
    REPORTS_DIR
    / "pipeline"
    / "archive"
)


# ============================================================
# IMPORTANT OUTPUTS
# ============================================================

FIXTURES_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_fixtures_resolved.csv"
)

CURRENT_SQUAD_POOL_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "ucl_current_squad_pool.csv"
)

LIVE_UCL_METADATA_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "live_2026_27"
    / "ucl_live_state_metadata.json"
)

LIVE_DOMESTIC_METADATA_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "live_2026_27"
    / "ucl_live_domestic_metadata.json"
)

LIVE_SQUAD_METADATA_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "ucl_live_squad_state_metadata.json"
)

RAW_PREDICTIONS_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions.csv"
)

FEATURE_MATRIX_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_features.csv"
)

CONFIDENCE_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_with_confidence.csv"
)

COMPLETE_PREDICTIONS_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_complete.csv"
)


# ============================================================
# PIPELINE STEP
# ============================================================

@dataclass(frozen=True)
class PipelineStep:
    name: str
    module: str
    description: str


PIPELINE_STEPS = [
    PipelineStep(
        name="fixtures",
        module="src.data.collect_fixtures",
        description=(
            "Refresh current UEFA Champions League "
            "fixtures and verified results."
        ),
    ),

    PipelineStep(
        name="fixture_identity",
        module="src.utils.resolve_current_ucl_clubs",
        description=(
            "Resolve current UEFA Champions League "
            "club identities into the canonical fixture snapshot."
        ),
    ),

    PipelineStep(
        name="current_squads",
        module="src.data.collect_current_squads",
        description=(
            "Refresh current football-data.org "
            "squad snapshots."
        ),
    ),

    PipelineStep(
        name="current_squad_pool",
        module="src.data.build_current_squad_pool",
        description=(
            "Rebuild the unified current 36-club "
            "squad/value pool."
        ),
    ),

    PipelineStep(
        name="ucl_state",
        module="src.features.live_ucl_state",
        description=(
            "Replay finished 2026/27 UCL matches "
            "through Elo and UCL-form state."
        ),
    ),

    PipelineStep(
        name="domestic_state",
        module="src.data.refresh_live_domestic",
        description=(
            "Rebuild the safe live domestic match "
            "overlay from football-data.org."
        ),
    ),

    PipelineStep(
        name="squad_state",
        module="src.features.live_squad_state",
        description=(
            "Build the safe partial live squad state "
            "without fabricated usage evidence."
        ),
    ),

    PipelineStep(
        name="raw_prediction",
        module="src.ml.predict_live_fixed",
        description=(
            "Build the exact frozen 290-feature "
            "matrix and run production CatBoost."
        ),
    ),

    PipelineStep(
        name="confidence",
        module="src.ml.confidence_live",
        description=(
            "Calculate evidence confidence separately "
            "from model probability."
        ),
    ),

    PipelineStep(
        name="scoreline",
        module="src.ml.scoreline_live",
        description=(
            "Fit the auxiliary Poisson score bridge."
        ),
    ),
]


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:

    return datetime.now(
        timezone.utc
    )


def utc_iso() -> str:

    return utc_now().isoformat()


def run_identifier() -> str:

    return utc_now().strftime(
        "%Y%m%dT%H%M%SZ"
    )


# ============================================================
# FILE HELPERS
# ============================================================

def ensure_directories() -> None:

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LATEST_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def file_info(
    path: Path,
) -> dict:

    if not path.exists():

        return {
            "exists": False,
            "path": str(path),
            "size_bytes": None,
            "modified_at_utc": None,
        }

    modified = datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )

    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "modified_at_utc": (
            modified.isoformat()
        ),
    }


# ============================================================
# STATUS FILE
# ============================================================

def write_status(
    payload: dict,
) -> None:

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = STATUS_FILE.with_suffix(
        ".json.tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary.replace(
        STATUS_FILE
    )


# ============================================================
# PREFLIGHT
# ============================================================

def preflight() -> dict:

    print(
        "=" * 78
    )

    print(
        "UCL PREDICTOR - PRODUCTION PIPELINE PREFLIGHT"
    )

    print(
        "=" * 78
    )

    print(
        f"\nProject root:"
    )

    print(
        BASE_DIR
    )

    print(
        f"\nPython:"
    )

    print(
        sys.executable
    )

    if not FOOTBALL_DATA_API_KEY:

        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY is missing.\n"
            "Add it to .env locally or the deployment "
            "secret environment before running the pipeline."
        )

    print(
        "\nfootball-data.org credential: AVAILABLE ✓"
    )

    if not BASE_DIR.exists():

        raise RuntimeError(
            f"Project root does not exist:\n"
            f"{BASE_DIR}"
        )

    expected_modules = [
        BASE_DIR
        / Path(
            step.module.replace(
                ".",
                "/",
            )
            +
            ".py"
        )
        for step in PIPELINE_STEPS
    ]

    missing_modules = [
        path
        for path in expected_modules
        if not path.exists()
    ]

    if missing_modules:

        raise RuntimeError(
            "Production pipeline modules missing:\n"
            +
            "\n".join(
                str(path)
                for path in missing_modules
            )
        )

    print(
        f"Pipeline modules: "
        f"{len(expected_modules)}/{len(expected_modules)} ✓"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "src.data.collect_domestic is intentionally NOT "
        "part of this production pipeline."
    )

    print(
        "The live domestic layer uses "
        "src.data.refresh_live_domestic instead."
    )

    return {
        "football_data_api_key_available": True,
        "python_executable": str(
            sys.executable
        ),
        "project_root": str(
            BASE_DIR
        ),
        "step_count": len(
            PIPELINE_STEPS
        ),
    }


# ============================================================
# COMMAND RUNNER
# ============================================================

def run_module(
    *,
    step: PipelineStep,
    log_handle,
) -> dict:

    command = [
        sys.executable,
        "-m",
        step.module,
    ]

    started_at = utc_now()

    started_monotonic = (
        time.perf_counter()
    )

    separator = (
        "\n"
        +
        "=" * 78
        +
        "\n"
    )

    heading = (
        f"STEP: {step.name}\n"
        f"MODULE: {step.module}\n"
        f"{step.description}\n"
    )

    print(
        separator,
        end="",
    )

    print(
        heading,
        end="",
    )

    log_handle.write(
        separator
    )

    log_handle.write(
        heading
    )

    log_handle.flush()

    environment = os.environ.copy()

    environment[
        "PYTHONUNBUFFERED"
    ] = "1"

    process = subprocess.Popen(
        command,
        cwd=str(
            BASE_DIR
        ),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if process.stdout is None:

        raise RuntimeError(
            f"Could not capture output for "
            f"{step.module}."
        )

    for line in process.stdout:

        print(
            line,
            end="",
        )

        log_handle.write(
            line
        )

        log_handle.flush()

    return_code = process.wait()

    finished_at = utc_now()

    duration = (
        time.perf_counter()
        -
        started_monotonic
    )

    success = (
        return_code
        ==
        0
    )

    result = {
        "name": step.name,
        "module": step.module,
        "description": step.description,
        "command": command,
        "started_at_utc": (
            started_at.isoformat()
        ),
        "finished_at_utc": (
            finished_at.isoformat()
        ),
        "duration_seconds": round(
            duration,
            3,
        ),
        "return_code": int(
            return_code
        ),
        "success": bool(
            success
        ),
    }

    if success:

        print(
            f"\n✓ {step.name} completed "
            f"in {duration:.1f}s"
        )

    else:

        print(
            f"\n✗ {step.name} FAILED "
            f"with exit code {return_code}"
        )

    return result


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_core_files() -> dict:

    required_outputs = {
        "fixtures": (
            FIXTURES_FILE
        ),

        "current_squad_pool": (
            CURRENT_SQUAD_POOL_FILE
        ),

        "live_ucl_metadata": (
            LIVE_UCL_METADATA_FILE
        ),

        "live_domestic_metadata": (
            LIVE_DOMESTIC_METADATA_FILE
        ),

        "live_squad_metadata": (
            LIVE_SQUAD_METADATA_FILE
        ),

        "raw_predictions": (
            RAW_PREDICTIONS_FILE
        ),

        "feature_matrix": (
            FEATURE_MATRIX_FILE
        ),

        "confidence_predictions": (
            CONFIDENCE_FILE
        ),

        "complete_predictions": (
            COMPLETE_PREDICTIONS_FILE
        ),
    }

    files = {
        name: file_info(
            path
        )
        for (
            name,
            path,
        ) in required_outputs.items()
    }

    missing = [
        name
        for (
            name,
            info,
        ) in files.items()
        if not info[
            "exists"
        ]
    ]

    if missing:

        raise RuntimeError(
            "Production outputs missing after pipeline:\n"
            +
            "\n".join(
                missing
            )
        )

    return files


def validate_feature_matrix() -> dict:

    dataframe = pd.read_csv(
        FEATURE_MATRIX_FILE,
        low_memory=False,
    )

    rows = len(
        dataframe
    )

    columns = len(
        dataframe.columns
    )

    if rows < 1:

        raise RuntimeError(
            "Live feature matrix contains no fixtures."
        )

    if (
        columns
        !=
        EXPECTED_FEATURE_COUNT
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT} "
            f"production columns, found "
            f"{columns}."
        )

    numeric = dataframe.apply(
        pd.to_numeric,
        errors="coerce",
    )

    infinite_count = int(
        np.isinf(
            numeric.to_numpy(
                dtype=float
            )
        ).sum()
    )

    if infinite_count:

        raise RuntimeError(
            f"Feature matrix contains "
            f"{infinite_count} infinite values."
        )

    non_null = int(
        dataframe.notna().sum().sum()
    )

    total_cells = int(
        rows
        *
        columns
    )

    availability = (
        non_null
        /
        total_cells
    )

    return {
        "rows": int(
            rows
        ),
        "columns": int(
            columns
        ),
        "infinite_values": (
            infinite_count
        ),
        "non_null_values": (
            non_null
        ),
        "total_values": (
            total_cells
        ),
        "availability_rate": float(
            availability
        ),
    }


def validate_predictions() -> dict:

    dataframe = pd.read_csv(
        COMPLETE_PREDICTIONS_FILE,
        low_memory=False,
    )

    if dataframe.empty:

        raise RuntimeError(
            "Complete prediction output is empty."
        )

    required = {
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_probability",
        "draw_probability",
        "away_probability",
        "predicted_selection",
        "predicted_scoreline",
        "confidence",
        "confidence_percent",
        "confidence_label",
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
            "Complete prediction output is missing:\n"
            f"{sorted(missing)}"
        )

    probability_columns = [
        "home_probability",
        "draw_probability",
        "away_probability",
    ]

    probabilities = dataframe[
        probability_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    probability_sums = (
        probabilities.sum(
            axis=1
        )
    )

    sum_errors = int(
        (
            ~np.isclose(
                probability_sums,
                1.0,
                atol=1e-6,
            )
        ).sum()
    )

    if sum_errors:

        raise RuntimeError(
            f"{sum_errors} prediction rows have "
            "H/D/A probabilities that do not sum to 1."
        )

    invalid_probabilities = int(
        (
            (
                probabilities
                <
                0
            )
            |
            (
                probabilities
                >
                1
            )
        )
        .sum()
        .sum()
    )

    if invalid_probabilities:

        raise RuntimeError(
            "Prediction probabilities outside [0,1]."
        )

    confidence = pd.to_numeric(
        dataframe[
            "confidence"
        ],
        errors="raise",
    )

    invalid_confidence = int(
        (
            (
                confidence
                <
                0
            )
            |
            (
                confidence
                >
                1
            )
        ).sum()
    )

    if invalid_confidence:

        raise RuntimeError(
            "Confidence values outside [0,1]."
        )

    if dataframe[
        "predicted_scoreline"
    ].isna().any():

        raise RuntimeError(
            "One or more predicted scorelines are missing."
        )

    kickoffs = pd.to_datetime(
        dataframe[
            "kickoff_utc"
        ],
        errors="raise",
        utc=True,
    )

    return {
        "fixture_count": int(
            len(
                dataframe
            )
        ),

        "first_kickoff_utc": (
            kickoffs.min().isoformat()
        ),

        "last_kickoff_utc": (
            kickoffs.max().isoformat()
        ),

        "probability_sum_errors": (
            sum_errors
        ),

        "invalid_probability_values": (
            invalid_probabilities
        ),

        "invalid_confidence_values": (
            invalid_confidence
        ),

        "scorelines_complete": True,

        "teams": sorted(
            set(
                dataframe[
                    "home_team"
                ].astype(
                    str
                )
            )
            |
            set(
                dataframe[
                    "away_team"
                ].astype(
                    str
                )
            )
        ),
    }


# ============================================================
# PIPELINE VALIDATION
# ============================================================

def validate_pipeline() -> dict:

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        "PRODUCTION OUTPUT VALIDATION"
    )

    print(
        "=" * 78
    )

    files = (
        validate_core_files()
    )

    feature_matrix = (
        validate_feature_matrix()
    )

    predictions = (
        validate_predictions()
    )

    print(
        f"\nFeature rows: "
        f"{feature_matrix['rows']}"
    )

    print(
        f"Feature columns: "
        f"{feature_matrix['columns']}"
    )

    print(
        f"Feature availability: "
        f"{feature_matrix['availability_rate'] * 100:.2f}%"
    )

    print(
        f"Infinite values: "
        f"{feature_matrix['infinite_values']}"
    )

    print(
        f"\nPrediction fixtures: "
        f"{predictions['fixture_count']}"
    )

    print(
        f"First kickoff: "
        f"{predictions['first_kickoff_utc']}"
    )

    print(
        f"Last kickoff: "
        f"{predictions['last_kickoff_utc']}"
    )

    print(
        "\n290-feature architecture valid ✓"
    )

    print(
        "H/D/A probability integrity valid ✓"
    )

    print(
        "Evidence confidence valid ✓"
    )

    print(
        "Scoreline output complete ✓"
    )

    return {
        "files": files,
        "feature_matrix": (
            feature_matrix
        ),
        "predictions": (
            predictions
        ),
    }


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the full UCL Predictor live "
            "production refresh pipeline."
        )
    )

    parser.add_argument(
        "--start-at",
        choices=[
            step.name
            for step in PIPELINE_STEPS
        ],
        default=None,
        help=(
            "Resume the production pipeline from "
            "a named step."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    ensure_directories()

    run_id = (
        run_identifier()
    )

    archive_log = (
        LOG_ARCHIVE_DIR
        /
        f"live_pipeline_{run_id}.log"
    )

    started_at = utc_now()

    started_monotonic = (
        time.perf_counter()
    )

    status = {
        "pipeline": (
            PIPELINE_NAME
        ),

        "pipeline_version": (
            PIPELINE_VERSION
        ),

        "model_version": (
            MODEL_VERSION
        ),

        "run_id": (
            run_id
        ),

        "status": (
            "RUNNING"
        ),

        "production_ready": (
            False
        ),

        "started_at_utc": (
            started_at.isoformat()
        ),

        "finished_at_utc": (
            None
        ),

        "duration_seconds": (
            None
        ),

        "current_step": (
            "preflight"
        ),

        "steps": [],

        "validation": (
            None
        ),

        "error": (
            None
        ),

        "log_file": str(
            LATEST_LOG_FILE
        ),
    }

    write_status(
        status
    )

    try:

        preflight_result = (
            preflight()
        )

        status[
            "preflight"
        ] = preflight_result

        selected_steps = (
            PIPELINE_STEPS
        )

        if args.start_at:

            start_index = next(
                index
                for (
                    index,
                    step,
                ) in enumerate(
                    PIPELINE_STEPS
                )
                if (
                    step.name
                    ==
                    args.start_at
                )
            )

            selected_steps = (
                PIPELINE_STEPS[
                    start_index:
                ]
            )

        with archive_log.open(
            "w",
            encoding="utf-8",
        ) as archive_handle, (
            LATEST_LOG_FILE.open(
                "w",
                encoding="utf-8",
            )
        ) as latest_handle:

            class MultiLog:
                def write(
                    self,
                    text,
                ):

                    archive_handle.write(
                        text
                    )

                    latest_handle.write(
                        text
                    )

                def flush(
                    self,
                ):

                    archive_handle.flush()

                    latest_handle.flush()

            log_handle = (
                MultiLog()
            )

            for step in selected_steps:

                status[
                    "current_step"
                ] = step.name

                write_status(
                    status
                )

                result = run_module(
                    step=step,
                    log_handle=log_handle,
                )

                status[
                    "steps"
                ].append(
                    result
                )

                write_status(
                    status
                )

                if not result[
                    "success"
                ]:

                    raise RuntimeError(
                        f"Production step failed: "
                        f"{step.name}\n"
                        f"Module: {step.module}\n"
                        f"Exit code: "
                        f"{result['return_code']}"
                    )

        status[
            "current_step"
        ] = "validation"

        write_status(
            status
        )

        validation = (
            validate_pipeline()
        )

        finished_at = utc_now()

        duration = (
            time.perf_counter()
            -
            started_monotonic
        )

        status.update(
            {
                "status": (
                    "SUCCESS"
                ),

                "production_ready": (
                    True
                ),

                "current_step": (
                    None
                ),

                "finished_at_utc": (
                    finished_at.isoformat()
                ),

                "duration_seconds": round(
                    duration,
                    3,
                ),

                "validation": (
                    validation
                ),

                "error": (
                    None
                ),
            }
        )

        write_status(
            status
        )

        print(
            "\n"
            +
            "=" * 78
        )

        print(
            "UCL PREDICTOR PRODUCTION PIPELINE COMPLETE"
        )

        print(
            "=" * 78
        )

        print(
            f"\nRun ID: "
            f"{run_id}"
        )

        print(
            f"Duration: "
            f"{duration:.1f}s"
        )

        print(
            "\nStatus: SUCCESS"
        )

        print(
            "Production ready: YES ✓"
        )

        print(
            "\nApplication prediction artifact:"
        )

        print(
            COMPLETE_PREDICTIONS_FILE
        )

        print(
            "\nPipeline status:"
        )

        print(
            STATUS_FILE
        )

        print(
            "\nLatest pipeline log:"
        )

        print(
            LATEST_LOG_FILE
        )

    except Exception as error:

        finished_at = utc_now()

        duration = (
            time.perf_counter()
            -
            started_monotonic
        )

        status.update(
            {
                "status": (
                    "FAILED"
                ),

                "production_ready": (
                    False
                ),

                "finished_at_utc": (
                    finished_at.isoformat()
                ),

                "duration_seconds": round(
                    duration,
                    3,
                ),

                "error": {
                    "type": (
                        type(
                            error
                        ).__name__
                    ),

                    "message": (
                        str(
                            error
                        )
                    ),
                },
            }
        )

        write_status(
            status
        )

        print(
            "\n"
            +
            "=" * 78
        )

        print(
            "PRODUCTION PIPELINE FAILED"
        )

        print(
            "=" * 78
        )

        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )

        print(
            "\nStatus written to:"
        )

        print(
            STATUS_FILE
        )

        raise


if __name__ == "__main__":
    main()