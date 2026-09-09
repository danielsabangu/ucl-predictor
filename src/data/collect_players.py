from __future__ import annotations

import gzip
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_DATA_DIR


# ============================================================
# SOURCE
# ============================================================

TRANSFERMARKT_ARCHIVE_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/"
    "data/transfermarkt-datasets.zip"
)

SOURCE_NAME = "dcaribou_transfermarkt_datasets"

SOURCE_NOTE = (
    "Transfermarkt-derived public football dataset. "
    "Individual published tables may have different freshness "
    "dates, so freshness is audited from the data itself."
)


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
)

ARCHIVE_FILE = (
    OUTPUT_DIR
    / "transfermarkt-datasets.zip"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "snapshot_metadata.json"
)


# ============================================================
# SETTINGS
# ============================================================

REQUEST_TIMEOUT = 60
MAX_RETRIES = 3

FORCE_REDOWNLOAD = False
FORCE_REEXTRACT = False

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
PANDAS_CHUNK_SIZE = 100_000

VALUATION_STALENESS_WARNING_DAYS = 120


# ============================================================
# FIRST PLAYER DATA LAYER
# ============================================================

REQUIRED_ASSETS = {
    "competitions.csv": {
        "required_columns": {
            "competition_id",
            "name",
        },
    },

    "clubs.csv": {
        "required_columns": {
            "club_id",
            "name",
        },
    },

    "players.csv": {
        "required_columns": {
            "player_id",
            "name",
            "current_club_id",
            "position",
        },
    },

    "player_valuations.csv": {
        "required_columns": {
            "player_id",
            "date",
            "market_value_in_eur",
        },
    },

    "transfers.csv": {
        "required_columns": {
            "player_id",
            "transfer_date",
            "from_club_id",
            "to_club_id",
        },
    },
}


# ============================================================
# DOWNLOAD
# ============================================================


def download_archive() -> Path:
    """
    Download the complete Transfermarkt archive.

    The existing 230 MB archive is reused automatically.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        ARCHIVE_FILE.exists()
        and ARCHIVE_FILE.stat().st_size > 0
        and not FORCE_REDOWNLOAD
    ):
        print(
            "\nUsing cached Transfermarkt archive:"
        )

        print(
            ARCHIVE_FILE
        )

        print(
            f"Archive size: "
            f"{ARCHIVE_FILE.stat().st_size / 1024 / 1024:,.1f} MB"
        )

        return ARCHIVE_FILE

    temporary_file = (
        ARCHIVE_FILE.with_suffix(
            ".zip.part"
        )
    )

    if temporary_file.exists():
        temporary_file.unlink()

    headers = {
        "User-Agent": "UCL-Predictor/1.0",
    }

    print(
        "\nDownloading Transfermarkt dataset archive:"
    )

    print(
        TRANSFERMARKT_ARCHIVE_URL
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            with requests.get(
                TRANSFERMARKT_ARCHIVE_URL,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                stream=True,
            ) as response:

                response.raise_for_status()

                total_bytes = int(
                    response.headers.get(
                        "content-length",
                        0,
                    )
                )

                downloaded_bytes = 0
                next_progress_mb = 100

                with temporary_file.open(
                    "wb"
                ) as destination:

                    for chunk in response.iter_content(
                        chunk_size=DOWNLOAD_CHUNK_SIZE
                    ):
                        if not chunk:
                            continue

                        destination.write(
                            chunk
                        )

                        downloaded_bytes += len(
                            chunk
                        )

                        downloaded_mb = (
                            downloaded_bytes
                            / 1024
                            / 1024
                        )

                        if (
                            downloaded_mb
                            >= next_progress_mb
                        ):
                            if total_bytes > 0:
                                percentage = (
                                    downloaded_bytes
                                    / total_bytes
                                    * 100
                                )

                                print(
                                    f"  Downloaded "
                                    f"{downloaded_mb:,.0f} MB "
                                    f"({percentage:.1f}%)"
                                )

                            else:
                                print(
                                    f"  Downloaded "
                                    f"{downloaded_mb:,.0f} MB"
                                )

                            next_progress_mb += 100

            temporary_file.replace(
                ARCHIVE_FILE
            )

            print(
                "\nArchive downloaded successfully."
            )

            print(
                f"Archive size: "
                f"{ARCHIVE_FILE.stat().st_size / 1024 / 1024:,.1f} MB"
            )

            return ARCHIVE_FILE

        except (
            requests.RequestException,
            OSError,
        ) as error:

            print(
                f"\nDownload attempt "
                f"{attempt}/{MAX_RETRIES} failed:"
            )

            print(
                error
            )

            if temporary_file.exists():
                temporary_file.unlink()

            if attempt < MAX_RETRIES:
                time.sleep(
                    3
                )

    raise RuntimeError(
        "Unable to download Transfermarkt dataset archive."
    )


# ============================================================
# ARCHIVE AUDIT
# ============================================================


def get_archive_file_members(
    archive: zipfile.ZipFile,
) -> list[str]:
    """
    Return all normal files stored inside the archive.
    """

    return [
        member
        for member in archive.namelist()
        if not member.endswith("/")
    ]


def print_archive_inventory(
    archive: zipfile.ZipFile,
) -> None:
    """
    Print relevant archive members.

    Useful when the upstream archive layout changes.
    """

    members = get_archive_file_members(
        archive
    )

    relevant = [
        member
        for member in members
        if (
            ".csv" in member.lower()
            or ".parquet" in member.lower()
            or ".duckdb" in member.lower()
        )
    ]

    print(
        "\nArchive inventory:"
    )

    print(
        f"  Files in archive: "
        f"{len(members):,}"
    )

    print(
        f"  Data-like files: "
        f"{len(relevant):,}"
    )

    print(
        "\nFirst data-like archive members:"
    )

    for member in relevant[:40]:
        print(
            f"  {member}"
        )


# ============================================================
# MEMBER LOOKUP
# ============================================================


def find_archive_member(
    archive: zipfile.ZipFile,
    output_filename: str,
) -> tuple[str, str]:
    """
    Locate either:

        competitions.csv

    or:

        competitions.csv.gz

    anywhere inside the archive.

    Returns:

        archive member path
        storage type
    """

    members = get_archive_file_members(
        archive
    )

    plain_name = output_filename

    gzip_name = (
        output_filename
        + ".gz"
    )

    plain_matches = [
        member
        for member in members
        if Path(
            member
        ).name == plain_name
    ]

    gzip_matches = [
        member
        for member in members
        if Path(
            member
        ).name == gzip_name
    ]

    if plain_matches:
        if len(
            plain_matches
        ) > 1:
            print(
                f"  Multiple {plain_name} "
                f"members found."
            )

        return (
            plain_matches[0],
            "csv",
        )

    if gzip_matches:
        if len(
            gzip_matches
        ) > 1:
            print(
                f"  Multiple {gzip_name} "
                f"members found."
            )

        return (
            gzip_matches[0],
            "csv.gz",
        )

    print_archive_inventory(
        archive
    )

    raise FileNotFoundError(
        f"Neither {plain_name} nor "
        f"{gzip_name} was found "
        f"inside the Transfermarkt archive."
    )


# ============================================================
# EXTRACT ONE ASSET
# ============================================================


def extract_archive_member(
    *,
    archive: zipfile.ZipFile,
    member: str,
    storage_type: str,
    destination: Path,
) -> None:
    """
    Extract a CSV.

    If the archive contains *.csv.gz, decompress it while
    extracting so our local project receives a normal *.csv.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with archive.open(
        member,
        "r",
    ) as source:

        if storage_type == "csv":
            with destination.open(
                "wb"
            ) as output:

                shutil.copyfileobj(
                    source,
                    output,
                )

            return

        if storage_type == "csv.gz":
            with gzip.GzipFile(
                fileobj=source,
                mode="rb",
            ) as decompressed:

                with destination.open(
                    "wb"
                ) as output:

                    shutil.copyfileobj(
                        decompressed,
                        output,
                    )

            return

    raise ValueError(
        f"Unsupported archive storage type: "
        f"{storage_type}"
    )


# ============================================================
# EXTRACT REQUIRED TABLES
# ============================================================


def extract_required_assets(
    archive_path: Path,
) -> dict[str, Path]:
    """
    Extract only the five tables required for the first
    squad/transfer feature layer.
    """

    extracted = {}

    print(
        "\n"
        + "=" * 75
    )

    print(
        "EXTRACTING PLAYER / VALUATION / TRANSFER TABLES"
    )

    print(
        "=" * 75
    )

    try:
        archive = zipfile.ZipFile(
            archive_path,
            "r",
        )

    except zipfile.BadZipFile as error:
        raise RuntimeError(
            "The cached Transfermarkt archive is not "
            "a valid ZIP file."
        ) from error

    with archive:
        for filename in REQUIRED_ASSETS:

            output_file = (
                OUTPUT_DIR
                / filename
            )

            print(
                f"\n{filename}"
            )

            if (
                output_file.exists()
                and output_file.stat().st_size > 0
                and not FORCE_REEXTRACT
            ):
                print(
                    "  Using cached extracted file."
                )

                extracted[
                    filename
                ] = output_file

                continue

            (
                member,
                storage_type,
            ) = find_archive_member(
                archive,
                filename,
            )

            print(
                f"  Archive member: "
                f"{member}"
            )

            print(
                f"  Storage format: "
                f"{storage_type}"
            )

            extract_archive_member(
                archive=archive,
                member=member,
                storage_type=storage_type,
                destination=output_file,
            )

            if (
                not output_file.exists()
                or output_file.stat().st_size == 0
            ):
                raise RuntimeError(
                    f"Extraction failed for "
                    f"{filename}."
                )

            print(
                f"  Extracted CSV: "
                f"{output_file}"
            )

            print(
                f"  Extracted size: "
                f"{output_file.stat().st_size / 1024 / 1024:,.2f} MB"
            )

            extracted[
                filename
            ] = output_file

    return extracted


# ============================================================
# HEADER VALIDATION
# ============================================================


def validate_columns(
    *,
    filename: str,
    file_path: Path,
) -> list[str]:
    """
    Validate required table columns without loading the
    complete file.
    """

    header = pd.read_csv(
        file_path,
        nrows=0,
        low_memory=False,
    )

    columns = list(
        header.columns
    )

    required_columns = (
        REQUIRED_ASSETS[
            filename
        ][
            "required_columns"
        ]
    )

    missing = (
        required_columns
        - set(
            columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{filename} is missing required "
            f"columns: {sorted(missing)}"
        )

    return columns


# ============================================================
# ROW COUNT
# ============================================================


def count_csv_rows(
    file_path: Path,
) -> int:
    """
    Count CSV rows safely using pandas chunks.
    """

    rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=PANDAS_CHUNK_SIZE,
        low_memory=False,
    ):
        rows += len(
            chunk
        )

    return rows


# ============================================================
# DATE RANGE
# ============================================================


def get_date_range(
    *,
    file_path: Path,
    date_column: str,
) -> tuple[
    pd.Timestamp | None,
    pd.Timestamp | None,
]:
    """
    Determine minimum and maximum dates using chunked reads.
    """

    minimum = None
    maximum = None

    for chunk in pd.read_csv(
        file_path,
        usecols=[
            date_column
        ],
        chunksize=PANDAS_CHUNK_SIZE,
        low_memory=False,
    ):
        dates = pd.to_datetime(
            chunk[
                date_column
            ],
            errors="coerce",
        ).dropna()

        if dates.empty:
            continue

        chunk_minimum = dates.min()
        chunk_maximum = dates.max()

        if (
            minimum is None
            or chunk_minimum < minimum
        ):
            minimum = chunk_minimum

        if (
            maximum is None
            or chunk_maximum > maximum
        ):
            maximum = chunk_maximum

    return (
        minimum,
        maximum,
    )


# ============================================================
# DATE FORMATTER
# ============================================================


def format_date(
    value: pd.Timestamp | None,
) -> str | None:
    """
    Convert Timestamp into YYYY-MM-DD.
    """

    if value is None:
        return None

    return (
        pd.Timestamp(
            value
        )
        .date()
        .isoformat()
    )


# ============================================================
# FRESHNESS AUDIT
# ============================================================


def valuation_freshness_warning(
    maximum_date: pd.Timestamp | None,
) -> None:
    """
    Warn if the historical valuation feed appears stale.

    We do not fail collection because older valuations remain
    extremely useful for historical training.
    """

    if maximum_date is None:
        print(
            "  WARNING: no valid valuation dates found."
        )

        return

    now = pd.Timestamp.now(
        tz="UTC"
    ).tz_localize(
        None
    )

    maximum_date = pd.Timestamp(
        maximum_date
    ).tz_localize(
        None
    )

    age_days = (
        now
        - maximum_date
    ).days

    print(
        f"  Latest valuation age: "
        f"{age_days:,} days"
    )

    if (
        age_days
        > VALUATION_STALENESS_WARNING_DAYS
    ):
        print(
            "  WARNING: player valuations are "
            "not sufficiently fresh for live/current "
            "2026/27 prediction features."
        )

        print(
            "  Historical valuation training use "
            "remains valid."
        )


# ============================================================
# QUALITY AUDIT
# ============================================================


def audit_assets(
    extracted: dict[str, Path],
) -> dict:
    """
    Validate schemas, row counts and temporal coverage.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    metadata = {
        "source": SOURCE_NAME,
        "source_url": (
            TRANSFERMARKT_ARCHIVE_URL
        ),
        "source_note": SOURCE_NOTE,
        "collected_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "assets": {},
    }

    for filename in REQUIRED_ASSETS:

        file_path = extracted[
            filename
        ]

        print(
            f"\n{filename}"
        )

        columns = validate_columns(
            filename=filename,
            file_path=file_path,
        )

        print(
            "  Required columns: valid ✓"
        )

        row_count = count_csv_rows(
            file_path
        )

        if row_count == 0:
            raise RuntimeError(
                f"{filename} contains zero rows."
            )

        print(
            f"  Rows: "
            f"{row_count:,}"
        )

        asset_metadata = {
            "path": str(
                file_path
            ),
            "rows": row_count,
            "columns": columns,
            "size_bytes": (
                file_path.stat().st_size
            ),
        }

        # ----------------------------------------------------
        # VALUATIONS
        # ----------------------------------------------------

        if (
            filename
            == "player_valuations.csv"
        ):
            (
                minimum,
                maximum,
            ) = get_date_range(
                file_path=file_path,
                date_column="date",
            )

            asset_metadata[
                "min_date"
            ] = format_date(
                minimum
            )

            asset_metadata[
                "max_date"
            ] = format_date(
                maximum
            )

            print(
                f"  Valuation date range: "
                f"{format_date(minimum)} "
                f"to "
                f"{format_date(maximum)}"
            )

            valuation_freshness_warning(
                maximum
            )

        # ----------------------------------------------------
        # TRANSFERS
        # ----------------------------------------------------

        if (
            filename
            == "transfers.csv"
        ):
            (
                minimum,
                maximum,
            ) = get_date_range(
                file_path=file_path,
                date_column="transfer_date",
            )

            asset_metadata[
                "min_date"
            ] = format_date(
                minimum
            )

            asset_metadata[
                "max_date"
            ] = format_date(
                maximum
            )

            print(
                f"  Transfer date range: "
                f"{format_date(minimum)} "
                f"to "
                f"{format_date(maximum)}"
            )

        metadata[
            "assets"
        ][
            filename
        ] = asset_metadata

    return metadata


# ============================================================
# METADATA
# ============================================================


def save_metadata(
    metadata: dict,
) -> None:
    """
    Save source provenance and collection information.
    """

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

    print(
        "\nSnapshot metadata saved to:"
    )

    print(
        METADATA_FILE
    )


# ============================================================
# PLAYER SAMPLE
# ============================================================


def print_player_sample(
    player_file: Path,
) -> None:
    """
    Show top player records for basic sanity checking.
    """

    header = pd.read_csv(
        player_file,
        nrows=0,
    )

    preferred_columns = [
        "player_id",
        "name",
        "current_club_id",
        "current_club_name",
        "position",
        "sub_position",
        "market_value_in_eur",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in header.columns
    ]

    players = pd.read_csv(
        player_file,
        usecols=columns,
        low_memory=False,
    )

    if (
        "market_value_in_eur"
        in players.columns
    ):
        players[
            "market_value_in_eur"
        ] = pd.to_numeric(
            players[
                "market_value_in_eur"
            ],
            errors="coerce",
        )

        players = players.sort_values(
            by="market_value_in_eur",
            ascending=False,
            na_position="last",
        )

    print(
        "\nTop 10 player rows:"
    )

    print(
        players.head(
            10
        ).to_string(
            index=False
        )
    )


# ============================================================
# TRANSFER SAMPLE
# ============================================================


def print_transfer_sample(
    transfer_file: Path,
) -> None:
    """
    Show recent transfers for basic sanity checking.
    """

    header = pd.read_csv(
        transfer_file,
        nrows=0,
    )

    preferred_columns = [
        "player_id",
        "player_name",
        "transfer_date",
        "from_club_id",
        "from_club_name",
        "to_club_id",
        "to_club_name",
        "transfer_fee",
        "market_value_in_eur",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in header.columns
    ]

    transfers = pd.read_csv(
        transfer_file,
        usecols=columns,
        low_memory=False,
    )

    if (
        "transfer_date"
        in transfers.columns
    ):
        transfers[
            "transfer_date"
        ] = pd.to_datetime(
            transfers[
                "transfer_date"
            ],
            errors="coerce",
        )

        transfers = transfers.sort_values(
            by="transfer_date",
            ascending=False,
            na_position="last",
        )

    print(
        "\n10 most recent transfer rows:"
    )

    print(
        transfers.head(
            10
        ).to_string(
            index=False
        )
    )


# ============================================================
# SAMPLES
# ============================================================


def print_samples(
    extracted: dict[str, Path],
) -> None:
    """
    Print small player and transfer samples.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PLAYER / TRANSFER DATA SAMPLE"
    )

    print(
        "=" * 75
    )

    print_player_sample(
        extracted[
            "players.csv"
        ]
    )

    print_transfer_sample(
        extracted[
            "transfers.csv"
        ]
    )


# ============================================================
# FINAL SUMMARY
# ============================================================


def print_final_summary(
    extracted: dict[str, Path],
) -> None:
    """
    Print local files now available.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PLAYER DATA COLLECTION COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nFiles ready:"
    )

    for filename in REQUIRED_ASSETS:

        print(
            f"  "
            f"{extracted[filename]}"
        )

    print(
        "\nArchive retained:"
    )

    print(
        f"  {ARCHIVE_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Historical valuations and transfers will "
        "be used chronologically."
    )

    print(
        "Current/live prediction features will not "
        "blindly use stale valuation records."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================


def collect_player_data() -> None:
    """
    Collect the initial historical player/squad dataset.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - PLAYER / VALUATION / TRANSFER COLLECTION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # ARCHIVE
    # --------------------------------------------------------

    archive_path = download_archive()

    # --------------------------------------------------------
    # EXTRACTION
    # --------------------------------------------------------

    extracted = extract_required_assets(
        archive_path
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    metadata = audit_assets(
        extracted
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    save_metadata(
        metadata
    )

    # --------------------------------------------------------
    # SAMPLES
    # --------------------------------------------------------

    print_samples(
        extracted
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_final_summary(
        extracted
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    collect_player_data()