from __future__ import annotations

import gzip
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
)


# ============================================================
# PATHS
# ============================================================

TRANSFERMARKT_RAW_DIR = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
)

ARCHIVE_FILE = (
    TRANSFERMARKT_RAW_DIR
    / "transfermarkt-datasets.zip"
)

CLUB_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)

OUTPUT_DIR = (
    INTERIM_DATA_DIR
    / "players"
)

APPEARANCES_OUTPUT_FILE = (
    OUTPUT_DIR
    / "transfermarkt_appearances_ucl_clubs.csv"
)

LINEUPS_OUTPUT_FILE = (
    OUTPUT_DIR
    / "transfermarkt_lineups_ucl_clubs.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "transfermarkt_player_history_metadata.json"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

FORCE_REBUILD = False

EXPECTED_UCL_CLUBS = 117


# ============================================================
# REQUIRED COLUMNS
# ============================================================

APPEARANCE_COLUMNS = [
    "appearance_id",
    "game_id",
    "player_id",
    "player_club_id",
    "player_current_club_id",
    "date",
    "player_name",
    "competition_id",
    "yellow_cards",
    "red_cards",
    "goals",
    "assists",
    "minutes_played",
]

LINEUP_COLUMNS = [
    "game_lineups_id",
    "game_id",
    "player_id",
    "club_id",
    "type",
    "player_name",
    "team_captain",
    "number",
    "position",
    "date",
]


# ============================================================
# LOAD FINAL CLUB MAP
# ============================================================


def load_ucl_club_map() -> pd.DataFrame:
    """
    Load the final 117-club UCL -> Transfermarkt map.
    """

    if not CLUB_MAP_FILE.exists():
        raise FileNotFoundError(
            f"Final club map not found:\n"
            f"{CLUB_MAP_FILE}"
        )

    mapping = pd.read_csv(
        CLUB_MAP_FILE,
        low_memory=False,
    )

    required = {
        "ucl_team",
        "club_id",
    }

    missing = (
        required
        - set(
            mapping.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Club map missing columns: "
            f"{sorted(missing)}"
        )

    mapping[
        "club_id"
    ] = pd.to_numeric(
        mapping[
            "club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    if len(
        mapping
    ) != EXPECTED_UCL_CLUBS:
        raise RuntimeError(
            f"Expected {EXPECTED_UCL_CLUBS} "
            f"mapped clubs, found {len(mapping)}."
        )

    if mapping[
        "ucl_team"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate UCL identities in club map."
        )

    if mapping[
        "club_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Transfermarkt club IDs "
            "in club map."
        )

    return mapping


# ============================================================
# FIND ARCHIVE MEMBER
# ============================================================


def find_archive_member(
    archive: zipfile.ZipFile,
    logical_filename: str,
) -> tuple[str, str]:
    """
    Find either:

        appearances.csv
        appearances.csv.gz

    or the equivalent for game_lineups.
    """

    members = [
        member
        for member in archive.namelist()
        if not member.endswith("/")
    ]

    plain_name = logical_filename

    gzip_name = (
        logical_filename
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
        return (
            plain_matches[0],
            "csv",
        )

    if gzip_matches:
        return (
            gzip_matches[0],
            "csv.gz",
        )

    raise FileNotFoundError(
        f"Neither {plain_name} nor "
        f"{gzip_name} exists in archive."
    )


# ============================================================
# STREAM CSV FROM ARCHIVE
# ============================================================


def build_csv_reader(
    *,
    archive: zipfile.ZipFile,
    member: str,
    storage_type: str,
    usecols: list[str],
):
    """
    Create a chunked pandas CSV reader directly from the ZIP.

    The full multi-million-row source table is never
    decompressed to disk.
    """

    raw_stream = archive.open(
        member,
        "r",
    )

    if storage_type == "csv.gz":

        data_stream = gzip.GzipFile(
            fileobj=raw_stream,
            mode="rb",
        )

    elif storage_type == "csv":

        data_stream = raw_stream

    else:

        raw_stream.close()

        raise ValueError(
            f"Unsupported storage type: "
            f"{storage_type}"
        )

    reader = pd.read_csv(
        data_stream,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    return (
        reader,
        data_stream,
        raw_stream,
    )


# ============================================================
# HEADER VALIDATION
# ============================================================


def validate_archive_table_header(
    *,
    archive: zipfile.ZipFile,
    member: str,
    storage_type: str,
    required_columns: list[str],
    table_name: str,
) -> None:
    """
    Validate source schema before filtering.
    """

    raw_stream = archive.open(
        member,
        "r",
    )

    try:

        if storage_type == "csv.gz":

            with gzip.GzipFile(
                fileobj=raw_stream,
                mode="rb",
            ) as data_stream:

                header = pd.read_csv(
                    data_stream,
                    nrows=0,
                    low_memory=False,
                )

        elif storage_type == "csv":

            header = pd.read_csv(
                raw_stream,
                nrows=0,
                low_memory=False,
            )

        else:

            raise ValueError(
                f"Unsupported storage type: "
                f"{storage_type}"
            )

    finally:

        raw_stream.close()

    missing = (
        set(
            required_columns
        )
        - set(
            header.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{table_name} missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# FILTER APPEARANCES
# ============================================================


def collect_appearances(
    *,
    archive: zipfile.ZipFile,
    club_ids: set[int],
) -> dict:
    """
    Filter Transfermarkt appearances to players representing
    our mapped UCL clubs.

    IMPORTANT:
    player_club_id means the club the player belonged to
    AT THE TIME OF THE GAME.

    We intentionally retain ALL competitions for those clubs.
    """

    (
        member,
        storage_type,
    ) = find_archive_member(
        archive,
        "appearances.csv",
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FILTERING TRANSFERMARKT APPEARANCES"
    )

    print(
        "=" * 75
    )

    print(
        f"\nArchive member: "
        f"{member}"
    )

    print(
        f"Storage format: "
        f"{storage_type}"
    )

    validate_archive_table_header(
        archive=archive,
        member=member,
        storage_type=storage_type,
        required_columns=APPEARANCE_COLUMNS,
        table_name="appearances",
    )

    print(
        "Required columns: valid ✓"
    )

    if (
        APPEARANCES_OUTPUT_FILE.exists()
        and not FORCE_REBUILD
    ):

        print(
            "\nUsing cached filtered appearances:"
        )

        print(
            APPEARANCES_OUTPUT_FILE
        )

        dataframe = pd.read_csv(
            APPEARANCES_OUTPUT_FILE,
            low_memory=False,
        )

        return audit_appearances(
            dataframe
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if APPEARANCES_OUTPUT_FILE.exists():
        APPEARANCES_OUTPUT_FILE.unlink()

    total_source_rows = 0
    retained_rows = 0

    first_write = True

    (
        reader,
        data_stream,
        raw_stream,
    ) = build_csv_reader(
        archive=archive,
        member=member,
        storage_type=storage_type,
        usecols=APPEARANCE_COLUMNS,
    )

    try:

        for chunk_number, chunk in enumerate(
            reader,
            start=1,
        ):

            total_source_rows += len(
                chunk
            )

            chunk[
                "player_club_id"
            ] = pd.to_numeric(
                chunk[
                    "player_club_id"
                ],
                errors="coerce",
            )

            mask = chunk[
                "player_club_id"
            ].isin(
                club_ids
            )

            filtered = chunk[
                mask
            ].copy()

            if filtered.empty:
                continue

            filtered[
                "player_club_id"
            ] = filtered[
                "player_club_id"
            ].astype(
                int
            )

            filtered.to_csv(
                APPEARANCES_OUTPUT_FILE,
                mode=(
                    "w"
                    if first_write
                    else "a"
                ),
                header=first_write,
                index=False,
                encoding="utf-8",
            )

            retained_rows += len(
                filtered
            )

            first_write = False

            if (
                chunk_number % 5
                ==
                0
            ):

                print(
                    f"  Processed "
                    f"{total_source_rows:,} source rows "
                    f"| retained "
                    f"{retained_rows:,}"
                )

    finally:

        try:
            data_stream.close()
        except Exception:
            pass

        try:
            raw_stream.close()
        except Exception:
            pass

    if not APPEARANCES_OUTPUT_FILE.exists():

        raise RuntimeError(
            "No relevant appearance rows were retained."
        )

    print(
        f"\nSource appearance rows processed: "
        f"{total_source_rows:,}"
    )

    print(
        f"Relevant appearance rows retained: "
        f"{retained_rows:,}"
    )

    print(
        f"Retention rate: "
        f"{retained_rows / total_source_rows * 100:.2f}%"
    )

    dataframe = pd.read_csv(
        APPEARANCES_OUTPUT_FILE,
        low_memory=False,
    )

    audit = audit_appearances(
        dataframe
    )

    audit[
        "source_rows"
    ] = total_source_rows

    audit[
        "retained_rows"
    ] = retained_rows

    return audit


# ============================================================
# APPEARANCE AUDIT
# ============================================================


def audit_appearances(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Validate filtered historical appearances.
    """

    df = dataframe.copy()

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

    df[
        "player_id"
    ] = pd.to_numeric(
        df[
            "player_id"
        ],
        errors="coerce",
    )

    df[
        "player_club_id"
    ] = pd.to_numeric(
        df[
            "player_club_id"
        ],
        errors="coerce",
    )

    df[
        "minutes_played"
    ] = pd.to_numeric(
        df[
            "minutes_played"
        ],
        errors="coerce",
    )

    missing_core = (
        df[
            [
                "game_id",
                "player_id",
                "player_club_id",
                "date",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    duplicate_ids = 0

    if (
        "appearance_id"
        in df.columns
    ):

        duplicate_ids = int(
            df[
                "appearance_id"
            ].duplicated().sum()
        )

    negative_minutes = int(
        (
            df[
                "minutes_played"
            ].dropna()
            <
            0
        ).sum()
    )

    clubs = (
        df[
            "player_club_id"
        ]
        .dropna()
        .astype(
            int
        )
        .nunique()
    )

    unique_players = (
        df[
            "player_id"
        ]
        .dropna()
        .astype(
            int
        )
        .nunique()
    )

    competitions = (
        df[
            "competition_id"
        ]
        .dropna()
        .astype(
            str
        )
        .nunique()
    )

    print(
        "\nAPPEARANCE DATA QUALITY CHECK"
    )

    print(
        "-" * 75
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique players: "
        f"{unique_players:,}"
    )

    print(
        f"Mapped club IDs represented: "
        f"{clubs:,}"
    )

    print(
        f"Competitions represented: "
        f"{competitions:,}"
    )

    print(
        f"Rows missing core values: "
        f"{missing_core:,}"
    )

    print(
        f"Duplicate appearance IDs: "
        f"{duplicate_ids:,}"
    )

    print(
        f"Negative minutes: "
        f"{negative_minutes:,}"
    )

    valid_dates = df[
        "date"
    ].dropna()

    if not valid_dates.empty:

        print(
            f"Date range: "
            f"{valid_dates.min().date()} "
            f"to "
            f"{valid_dates.max().date()}"
        )

    if missing_core > 0:

        raise RuntimeError(
            "Filtered appearances contain "
            "missing core values."
        )

    if duplicate_ids > 0:

        raise RuntimeError(
            "Duplicate appearance IDs detected."
        )

    if negative_minutes > 0:

        raise RuntimeError(
            "Negative appearance minutes detected."
        )

    print(
        "Appearance integrity checks: passed ✓"
    )

    return {
        "rows": int(
            len(df)
        ),
        "unique_players": int(
            unique_players
        ),
        "club_ids_represented": int(
            clubs
        ),
        "competitions_represented": int(
            competitions
        ),
        "min_date": (
            valid_dates.min().date().isoformat()
            if not valid_dates.empty
            else None
        ),
        "max_date": (
            valid_dates.max().date().isoformat()
            if not valid_dates.empty
            else None
        ),
    }


# ============================================================
# FILTER LINEUPS
# ============================================================


def collect_lineups(
    *,
    archive: zipfile.ZipFile,
    club_ids: set[int],
) -> dict:
    """
    Filter starting/bench lineup rows to our mapped UCL
    clubs across all competitions.
    """

    (
        member,
        storage_type,
    ) = find_archive_member(
        archive,
        "game_lineups.csv",
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FILTERING TRANSFERMARKT GAME LINEUPS"
    )

    print(
        "=" * 75
    )

    print(
        f"\nArchive member: "
        f"{member}"
    )

    print(
        f"Storage format: "
        f"{storage_type}"
    )

    validate_archive_table_header(
        archive=archive,
        member=member,
        storage_type=storage_type,
        required_columns=LINEUP_COLUMNS,
        table_name="game_lineups",
    )

    print(
        "Required columns: valid ✓"
    )

    if (
        LINEUPS_OUTPUT_FILE.exists()
        and not FORCE_REBUILD
    ):

        print(
            "\nUsing cached filtered lineups:"
        )

        print(
            LINEUPS_OUTPUT_FILE
        )

        dataframe = pd.read_csv(
            LINEUPS_OUTPUT_FILE,
            low_memory=False,
        )

        return audit_lineups(
            dataframe
        )

    if LINEUPS_OUTPUT_FILE.exists():
        LINEUPS_OUTPUT_FILE.unlink()

    total_source_rows = 0
    retained_rows = 0

    first_write = True

    (
        reader,
        data_stream,
        raw_stream,
    ) = build_csv_reader(
        archive=archive,
        member=member,
        storage_type=storage_type,
        usecols=LINEUP_COLUMNS,
    )

    try:

        for chunk_number, chunk in enumerate(
            reader,
            start=1,
        ):

            total_source_rows += len(
                chunk
            )

            chunk[
                "club_id"
            ] = pd.to_numeric(
                chunk[
                    "club_id"
                ],
                errors="coerce",
            )

            mask = chunk[
                "club_id"
            ].isin(
                club_ids
            )

            filtered = chunk[
                mask
            ].copy()

            if filtered.empty:
                continue

            filtered[
                "club_id"
            ] = filtered[
                "club_id"
            ].astype(
                int
            )

            filtered.to_csv(
                LINEUPS_OUTPUT_FILE,
                mode=(
                    "w"
                    if first_write
                    else "a"
                ),
                header=first_write,
                index=False,
                encoding="utf-8",
            )

            retained_rows += len(
                filtered
            )

            first_write = False

            if (
                chunk_number % 5
                ==
                0
            ):

                print(
                    f"  Processed "
                    f"{total_source_rows:,} source rows "
                    f"| retained "
                    f"{retained_rows:,}"
                )

    finally:

        try:
            data_stream.close()
        except Exception:
            pass

        try:
            raw_stream.close()
        except Exception:
            pass

    if not LINEUPS_OUTPUT_FILE.exists():

        raise RuntimeError(
            "No relevant lineup rows were retained."
        )

    print(
        f"\nSource lineup rows processed: "
        f"{total_source_rows:,}"
    )

    print(
        f"Relevant lineup rows retained: "
        f"{retained_rows:,}"
    )

    print(
        f"Retention rate: "
        f"{retained_rows / total_source_rows * 100:.2f}%"
    )

    dataframe = pd.read_csv(
        LINEUPS_OUTPUT_FILE,
        low_memory=False,
    )

    audit = audit_lineups(
        dataframe
    )

    audit[
        "source_rows"
    ] = total_source_rows

    audit[
        "retained_rows"
    ] = retained_rows

    return audit


# ============================================================
# LINEUP AUDIT
# ============================================================


def audit_lineups(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Validate historical lineup data.
    """

    df = dataframe.copy()

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

    df[
        "club_id"
    ] = pd.to_numeric(
        df[
            "club_id"
        ],
        errors="coerce",
    )

    df[
        "player_id"
    ] = pd.to_numeric(
        df[
            "player_id"
        ],
        errors="coerce",
    )

    missing_core = (
        df[
            [
                "game_id",
                "player_id",
                "club_id",
                "date",
                "type",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    duplicate_ids = int(
        df[
            "game_lineups_id"
        ].duplicated().sum()
    )

    clubs = (
        df[
            "club_id"
        ]
        .dropna()
        .astype(
            int
        )
        .nunique()
    )

    unique_players = (
        df[
            "player_id"
        ]
        .dropna()
        .astype(
            int
        )
        .nunique()
    )

    unique_games = (
        df[
            "game_id"
        ]
        .dropna()
        .nunique()
    )

    print(
        "\nLINEUP DATA QUALITY CHECK"
    )

    print(
        "-" * 75
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique games: "
        f"{unique_games:,}"
    )

    print(
        f"Unique players: "
        f"{unique_players:,}"
    )

    print(
        f"Mapped club IDs represented: "
        f"{clubs:,}"
    )

    print(
        f"Rows missing core values: "
        f"{missing_core:,}"
    )

    print(
        f"Duplicate lineup IDs: "
        f"{duplicate_ids:,}"
    )

    valid_dates = df[
        "date"
    ].dropna()

    if not valid_dates.empty:

        print(
            f"Date range: "
            f"{valid_dates.min().date()} "
            f"to "
            f"{valid_dates.max().date()}"
        )

    print(
        "\nLineup type distribution:"
    )

    type_counts = (
        df[
            "type"
        ]
        .fillna(
            "<missing>"
        )
        .astype(
            str
        )
        .value_counts()
    )

    print(
        type_counts.to_string()
    )

    print(
        "\nTop positions:"
    )

    position_counts = (
        df[
            "position"
        ]
        .fillna(
            "<missing>"
        )
        .astype(
            str
        )
        .value_counts()
        .head(
            15
        )
    )

    print(
        position_counts.to_string()
    )

    if missing_core > 0:

        raise RuntimeError(
            "Filtered lineups contain "
            "missing core values."
        )

    if duplicate_ids > 0:

        raise RuntimeError(
            "Duplicate lineup IDs detected."
        )

    print(
        "Lineup integrity checks: passed ✓"
    )

    return {
        "rows": int(
            len(df)
        ),
        "unique_games": int(
            unique_games
        ),
        "unique_players": int(
            unique_players
        ),
        "club_ids_represented": int(
            clubs
        ),
        "min_date": (
            valid_dates.min().date().isoformat()
            if not valid_dates.empty
            else None
        ),
        "max_date": (
            valid_dates.max().date().isoformat()
            if not valid_dates.empty
            else None
        ),
        "type_distribution": {
            str(
                key
            ): int(
                value
            )
            for (
                key,
                value,
            ) in type_counts.items()
        },
    }


# ============================================================
# CROSS-TABLE AUDIT
# ============================================================


def cross_table_audit(
    *,
    mapping: pd.DataFrame,
) -> dict:
    """
    Compare club/player coverage between appearances and
    lineups.
    """

    appearances = pd.read_csv(
        APPEARANCES_OUTPUT_FILE,
        usecols=[
            "player_id",
            "player_club_id",
            "game_id",
            "date",
        ],
        low_memory=False,
    )

    lineups = pd.read_csv(
        LINEUPS_OUTPUT_FILE,
        usecols=[
            "player_id",
            "club_id",
            "game_id",
            "date",
        ],
        low_memory=False,
    )

    appearance_clubs = set(
        pd.to_numeric(
            appearances[
                "player_club_id"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
    )

    lineup_clubs = set(
        pd.to_numeric(
            lineups[
                "club_id"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
    )

    mapped_clubs = set(
        mapping[
            "club_id"
        ].astype(
            int
        )
    )

    appearance_missing = (
        mapped_clubs
        - appearance_clubs
    )

    lineup_missing = (
        mapped_clubs
        - lineup_clubs
    )

    mapping_lookup = (
        mapping
        .set_index(
            "club_id"
        )[
            "ucl_team"
        ]
        .to_dict()
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PLAYER HISTORY CROSS-TABLE COVERAGE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nMapped UCL clubs: "
        f"{len(mapped_clubs):,}"
    )

    print(
        f"Clubs with appearances: "
        f"{len(appearance_clubs):,}"
    )

    print(
        f"Clubs with lineups: "
        f"{len(lineup_clubs):,}"
    )

    print(
        f"Mapped clubs without appearances: "
        f"{len(appearance_missing):,}"
    )

    if appearance_missing:

        for club_id in sorted(
            appearance_missing
        ):

            print(
                f"  "
                f"{mapping_lookup.get(club_id, '?')} "
                f"| club_id={club_id}"
            )

    print(
        f"\nMapped clubs without lineups: "
        f"{len(lineup_missing):,}"
    )

    if lineup_missing:

        for club_id in sorted(
            lineup_missing
        ):

            print(
                f"  "
                f"{mapping_lookup.get(club_id, '?')} "
                f"| club_id={club_id}"
            )

    appearance_players = set(
        pd.to_numeric(
            appearances[
                "player_id"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
    )

    lineup_players = set(
        pd.to_numeric(
            lineups[
                "player_id"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
    )

    player_overlap = (
        appearance_players
        & lineup_players
    )

    print(
        f"\nAppearance players: "
        f"{len(appearance_players):,}"
    )

    print(
        f"Lineup players: "
        f"{len(lineup_players):,}"
    )

    print(
        f"Players present in both tables: "
        f"{len(player_overlap):,}"
    )

    return {
        "mapped_clubs": len(
            mapped_clubs
        ),
        "appearance_clubs": len(
            appearance_clubs
        ),
        "lineup_clubs": len(
            lineup_clubs
        ),
        "appearance_missing_club_ids": sorted(
            appearance_missing
        ),
        "lineup_missing_club_ids": sorted(
            lineup_missing
        ),
        "appearance_players": len(
            appearance_players
        ),
        "lineup_players": len(
            lineup_players
        ),
        "player_overlap": len(
            player_overlap
        ),
    }


# ============================================================
# SAVE METADATA
# ============================================================


def save_metadata(
    *,
    mapping: pd.DataFrame,
    appearance_audit: dict,
    lineup_audit: dict,
    cross_audit: dict,
) -> None:
    """
    Save provenance and coverage metadata.
    """

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "mapped_ucl_clubs": int(
            len(
                mapping
            )
        ),

        "source_archive": str(
            ARCHIVE_FILE
        ),

        "appearance_filter": (
            "player_club_id in final 117 "
            "Transfermarkt UCL club IDs"
        ),

        "lineup_filter": (
            "club_id in final 117 "
            "Transfermarkt UCL club IDs"
        ),

        "competition_filter": (
            "none - all competitions retained "
            "for mapped clubs"
        ),

        "appearances": (
            appearance_audit
        ),

        "lineups": (
            lineup_audit
        ),

        "cross_table_coverage": (
            cross_audit
        ),
    }

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
        "\nPlayer-history metadata saved to:"
    )

    print(
        METADATA_FILE
    )


# ============================================================
# MAIN PIPELINE
# ============================================================


def collect_player_history() -> None:
    """
    Extract historical appearances and lineups for our
    canonical 117 UCL clubs.

    No future information is converted into ML features here.
    This step only creates historical source tables.

    Leakage-safe feature construction comes afterward.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - HISTORICAL PLAYER USAGE COLLECTION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # SOURCE CHECK
    # --------------------------------------------------------

    if not ARCHIVE_FILE.exists():

        raise FileNotFoundError(
            f"Transfermarkt archive not found:\n"
            f"{ARCHIVE_FILE}"
        )

    mapping = load_ucl_club_map()

    club_ids = set(
        mapping[
            "club_id"
        ].astype(
            int
        )
    )

    print(
        f"\nMapped UCL clubs loaded: "
        f"{len(mapping):,}"
    )

    print(
        f"Unique Transfermarkt club IDs: "
        f"{len(club_ids):,}"
    )

    if len(
        club_ids
    ) != EXPECTED_UCL_CLUBS:

        raise RuntimeError(
            "Expected exactly 117 unique "
            "Transfermarkt club IDs."
        )

    print(
        "Final club identity map: valid ✓"
    )

    # --------------------------------------------------------
    # ARCHIVE
    # --------------------------------------------------------

    try:

        archive = zipfile.ZipFile(
            ARCHIVE_FILE,
            "r",
        )

    except zipfile.BadZipFile as error:

        raise RuntimeError(
            "Transfermarkt archive is invalid."
        ) from error

    with archive:

        appearance_audit = collect_appearances(
            archive=archive,
            club_ids=club_ids,
        )

        lineup_audit = collect_lineups(
            archive=archive,
            club_ids=club_ids,
        )

    # --------------------------------------------------------
    # CROSS-TABLE CHECK
    # --------------------------------------------------------

    cross_audit = cross_table_audit(
        mapping=mapping
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    save_metadata(
        mapping=mapping,
        appearance_audit=appearance_audit,
        lineup_audit=lineup_audit,
        cross_audit=cross_audit,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "HISTORICAL PLAYER USAGE COLLECTION SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nFiltered appearances: "
        f"{appearance_audit['rows']:,}"
    )

    print(
        f"Appearance players: "
        f"{appearance_audit['unique_players']:,}"
    )

    print(
        f"Appearance clubs represented: "
        f"{appearance_audit['club_ids_represented']:,}"
    )

    print(
        f"\nFiltered lineup rows: "
        f"{lineup_audit['rows']:,}"
    )

    print(
        f"Lineup players: "
        f"{lineup_audit['unique_players']:,}"
    )

    print(
        f"Lineup clubs represented: "
        f"{lineup_audit['club_ids_represented']:,}"
    )

    print(
        "\nAppearances saved to:"
    )

    print(
        APPEARANCES_OUTPUT_FILE
    )

    print(
        "\nLineups saved to:"
    )

    print(
        LINEUPS_OUTPUT_FILE
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "HISTORICAL PLAYER USAGE COLLECTION COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are historical source records only."
    )

    print(
        "The next feature-engineering step will enforce:"
    )

    print(
        "    player information date < UCL match date"
    )

    print(
        "before any player or squad statistic is attached "
        "to a prediction row."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    collect_player_history()