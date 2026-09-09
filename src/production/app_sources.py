from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    BASE_DIR,
    PREDICTIONS_DIR,
)


# ============================================================
# MODES
# ============================================================

MODE_AUTO = "AUTO"
MODE_LOCAL = "LOCAL"
MODE_PUBLIC = "PUBLIC"

VALID_MODES = {
    MODE_AUTO,
    MODE_LOCAL,
    MODE_PUBLIC,
}


# ============================================================
# LOCAL PRODUCTION ARTIFACTS
# ============================================================

LOCAL_PREDICTIONS_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_complete.csv"
)

LOCAL_STATUS_FILE = (
    PREDICTIONS_DIR
    / "live_pipeline_status.json"
)


# ============================================================
# PUBLIC DEPLOYMENT SNAPSHOT
# ============================================================

PUBLIC_DATA_DIR = (
    BASE_DIR
    / "data"
    / "app"
)

PUBLIC_PREDICTIONS_FILE = (
    PUBLIC_DATA_DIR
    / "current_predictions.csv"
)

PUBLIC_STATUS_FILE = (
    PUBLIC_DATA_DIR
    / "pipeline_status.json"
)

PUBLIC_METADATA_FILE = (
    PUBLIC_DATA_DIR
    / "snapshot_metadata.json"
)


# ============================================================
# DATA SOURCE
# ============================================================


@dataclass(
    frozen=True
)
class AppDataSource:

    mode: str

    predictions_file: Path

    status_file: Path

    metadata_file: Path | None

    is_public_snapshot: bool


# ============================================================
# MODE
# ============================================================


def requested_mode() -> str:

    value = (
        os.getenv(
            "UCL_APP_MODE",
            MODE_AUTO,
        )
        .strip()
        .upper()
    )

    if value not in VALID_MODES:

        raise RuntimeError(
            "Invalid UCL_APP_MODE. "
            f"Expected one of "
            f"{sorted(VALID_MODES)}, "
            f"received {value!r}."
        )

    return value


# ============================================================
# AVAILABILITY
# ============================================================


def local_source_available() -> bool:

    return (
        LOCAL_PREDICTIONS_FILE.exists()
        and
        LOCAL_STATUS_FILE.exists()
    )


def public_source_available() -> bool:

    return (
        PUBLIC_PREDICTIONS_FILE.exists()
        and
        PUBLIC_STATUS_FILE.exists()
    )


# ============================================================
# BUILD SOURCES
# ============================================================


def local_source() -> AppDataSource:

    return AppDataSource(
        mode=MODE_LOCAL,
        predictions_file=(
            LOCAL_PREDICTIONS_FILE
        ),
        status_file=(
            LOCAL_STATUS_FILE
        ),
        metadata_file=None,
        is_public_snapshot=False,
    )


def public_source() -> AppDataSource:

    return AppDataSource(
        mode=MODE_PUBLIC,
        predictions_file=(
            PUBLIC_PREDICTIONS_FILE
        ),
        status_file=(
            PUBLIC_STATUS_FILE
        ),
        metadata_file=(
            PUBLIC_METADATA_FILE
        ),
        is_public_snapshot=True,
    )


# ============================================================
# RESOLVE
# ============================================================


def resolve_app_data_source() -> AppDataSource:

    mode = requested_mode()

    # --------------------------------------------------------
    # EXPLICIT LOCAL
    # --------------------------------------------------------

    if mode == MODE_LOCAL:

        if not local_source_available():

            raise RuntimeError(
                "UCL_APP_MODE=LOCAL was requested, "
                "but local production artifacts "
                "are unavailable.\n"
                f"Predictions: "
                f"{LOCAL_PREDICTIONS_FILE}\n"
                f"Status: "
                f"{LOCAL_STATUS_FILE}"
            )

        return local_source()

    # --------------------------------------------------------
    # EXPLICIT PUBLIC
    # --------------------------------------------------------

    if mode == MODE_PUBLIC:

        if not public_source_available():

            raise RuntimeError(
                "UCL_APP_MODE=PUBLIC was requested, "
                "but the public deployment snapshot "
                "is unavailable.\n"
                f"Predictions: "
                f"{PUBLIC_PREDICTIONS_FILE}\n"
                f"Status: "
                f"{PUBLIC_STATUS_FILE}"
            )

        return public_source()

    # --------------------------------------------------------
    # AUTO
    # --------------------------------------------------------

    if local_source_available():

        return local_source()

    if public_source_available():

        return public_source()

    raise RuntimeError(
        "No usable UCL Predictor application "
        "data source was found.\n\n"
        "Checked local production artifacts:\n"
        f"  {LOCAL_PREDICTIONS_FILE}\n"
        f"  {LOCAL_STATUS_FILE}\n\n"
        "Checked public deployment snapshot:\n"
        f"  {PUBLIC_PREDICTIONS_FILE}\n"
        f"  {PUBLIC_STATUS_FILE}"
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    print(
        "=" * 78
    )

    print(
        "UCL PREDICTOR - APPLICATION DATA SOURCE"
    )

    print(
        "=" * 78
    )

    print(
        f"\nRequested mode: "
        f"{requested_mode()}"
    )

    print(
        f"Local source available: "
        f"{local_source_available()}"
    )

    print(
        f"Public source available: "
        f"{public_source_available()}"
    )

    source = (
        resolve_app_data_source()
    )

    print(
        f"\nResolved mode: "
        f"{source.mode}"
    )

    print(
        f"Public snapshot: "
        f"{source.is_public_snapshot}"
    )

    print(
        "\nPredictions:"
    )

    print(
        source.predictions_file
    )

    print(
        "\nStatus:"
    )

    print(
        source.status_file
    )

    if source.metadata_file:

        print(
            "\nMetadata:"
        )

        print(
            source.metadata_file
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "APPLICATION DATA SOURCE READY ✓"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()