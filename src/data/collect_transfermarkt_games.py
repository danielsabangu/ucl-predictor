from __future__ import annotations

import gzip
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR


# ============================================================
# PATHS
# ============================================================

TRANSFERMARKT_DIR = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
)

ARCHIVE_FILE = (
    TRANSFERMARKT_DIR
    / "transfermarkt-datasets.zip"
)

GAMES_FILE = (
    TRANSFERMARKT_DIR
    / "games.csv"
)

COMPETITIONS_FILE = (
    TRANSFERMARKT_DIR
    / "competitions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

FORCE_REEXTRACT = False

# Transfermarkt competition IDs:
#
# CL  = UEFA Champions League
# CLQ = UEFA Champions League Qualifying
#
# We deliberately use CL only because our master UCL dataset
# contains the competition proper rather than qualifiers.
UCL_COMPETITION_ID = "CL"


# ============================================================
# REQUIRED GAME COLUMNS
# ============================================================

REQUIRED_GAME_COLUMNS = {
    "game_id",
    "competition_id",
    "season",
    "date",
    "home_club_id",
    "away_club_id",
    "home_club_name",
    "away_club_name",
    "home_club_goals",
    "away_club_goals",
}


# ============================================================
# ARCHIVE MEMBER LOOKUP
# ============================================================


def find_games_member(
    archive: zipfile.ZipFile,
) -> tuple[str, str]:
    """
    Locate games.csv or games.csv.gz anywhere inside
    the cached Transfermarkt archive.
    """

    members = [
        member
        for member in archive.namelist()
        if not member.endswith("/")
    ]

    plain_matches = [
        member
        for member in members
        if Path(member).name == "games.csv"
    ]

    gzip_matches = [
        member
        for member in members
        if Path(member).name == "games.csv.gz"
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
        "Neither games.csv nor games.csv.gz "
        "was found inside the Transfermarkt archive."
    )


# ============================================================
# EXTRACT GAMES
# ============================================================


def extract_games() -> Path:
    """
    Extract Transfermarkt games.csv from the existing
    downloaded ZIP archive.
    """

    if not ARCHIVE_FILE.exists():
        raise FileNotFoundError(
            f"Transfermarkt archive not found:\n"
            f"{ARCHIVE_FILE}"
        )

    if (
        GAMES_FILE.exists()
        and GAMES_FILE.stat().st_size > 0
        and not FORCE_REEXTRACT
    ):
        print(
            "\nUsing cached Transfermarkt games file:"
        )

        print(
            GAMES_FILE
        )

        print(
            f"File size: "
            f"{GAMES_FILE.stat().st_size / 1024 / 1024:,.2f} MB"
        )

        return GAMES_FILE

    print(
        "\nOpening cached Transfermarkt archive:"
    )

    print(
        ARCHIVE_FILE
    )

    try:
        archive = zipfile.ZipFile(
            ARCHIVE_FILE,
            "r",
        )

    except zipfile.BadZipFile as error:
        raise RuntimeError(
            "Cached Transfermarkt archive is not "
            "a valid ZIP file."
        ) from error

    with archive:
        (
            member,
            storage_type,
        ) = find_games_member(
            archive
        )

        print(
            f"\nArchive member: "
            f"{member}"
        )

        print(
            f"Storage format: "
            f"{storage_type}"
        )

        with archive.open(
            member,
            "r",
        ) as source:

            if storage_type == "csv":

                with GAMES_FILE.open(
                    "wb"
                ) as destination:

                    shutil.copyfileobj(
                        source,
                        destination,
                    )

            elif storage_type == "csv.gz":

                with gzip.GzipFile(
                    fileobj=source,
                    mode="rb",
                ) as decompressed:

                    with GAMES_FILE.open(
                        "wb"
                    ) as destination:

                        shutil.copyfileobj(
                            decompressed,
                            destination,
                        )

            else:
                raise RuntimeError(
                    f"Unsupported storage format: "
                    f"{storage_type}"
                )

    if (
        not GAMES_FILE.exists()
        or GAMES_FILE.stat().st_size == 0
    ):
        raise RuntimeError(
            "games.csv extraction failed."
        )

    print(
        "\nGames file extracted successfully."
    )

    print(
        GAMES_FILE
    )

    print(
        f"Extracted size: "
        f"{GAMES_FILE.stat().st_size / 1024 / 1024:,.2f} MB"
    )

    return GAMES_FILE


# ============================================================
# HEADER CHECK
# ============================================================


def validate_header(
    file_path: Path,
) -> list[str]:
    """
    Validate fields needed for UCL club identification.
    """

    header = pd.read_csv(
        file_path,
        nrows=0,
        low_memory=False,
    )

    columns = list(
        header.columns
    )

    missing = (
        REQUIRED_GAME_COLUMNS
        - set(columns)
    )

    if missing:
        raise RuntimeError(
            "games.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    return columns


# ============================================================
# GAME DATA AUDIT
# ============================================================


def audit_games(
    file_path: Path,
) -> dict:
    """
    Audit games.csv using chunks.
    """

    total_rows = 0

    minimum_date = None
    maximum_date = None

    unique_game_ids = set()

    competition_counts: dict[
        str,
        int,
    ] = {}

    missing_core = 0
    duplicate_game_ids = 0

    usecols = [
        "game_id",
        "competition_id",
        "season",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_name",
        "away_club_name",
    ]

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        total_rows += len(
            chunk
        )

        chunk[
            "competition_id"
        ] = (
            chunk[
                "competition_id"
            ]
            .astype(str)
            .str.strip()
        )

        dates = pd.to_datetime(
            chunk[
                "date"
            ],
            errors="coerce",
        )

        valid_dates = dates.dropna()

        if not valid_dates.empty:

            chunk_min = valid_dates.min()
            chunk_max = valid_dates.max()

            if (
                minimum_date is None
                or chunk_min < minimum_date
            ):
                minimum_date = chunk_min

            if (
                maximum_date is None
                or chunk_max > maximum_date
            ):
                maximum_date = chunk_max

        missing_core += int(
            chunk[
                [
                    "game_id",
                    "competition_id",
                    "date",
                    "home_club_id",
                    "away_club_id",
                ]
            ]
            .isna()
            .any(
                axis=1
            )
            .sum()
        )

        for game_id in chunk[
            "game_id"
        ].dropna():

            game_id = int(
                game_id
            )

            if game_id in unique_game_ids:
                duplicate_game_ids += 1

            unique_game_ids.add(
                game_id
            )

        counts = (
            chunk[
                "competition_id"
            ]
            .value_counts()
        )

        for (
            competition_id,
            count,
        ) in counts.items():

            competition_counts[
                competition_id
            ] = (
                competition_counts.get(
                    competition_id,
                    0,
                )
                + int(count)
            )

    return {
        "total_rows": total_rows,
        "unique_game_ids": len(
            unique_game_ids
        ),
        "duplicate_game_ids": (
            duplicate_game_ids
        ),
        "missing_core_rows": (
            missing_core
        ),
        "minimum_date": minimum_date,
        "maximum_date": maximum_date,
        "competition_counts": (
            competition_counts
        ),
    }


# ============================================================
# LOAD COMPETITIONS
# ============================================================


def load_competitions() -> pd.DataFrame:
    """
    Load competition metadata.
    """

    if not COMPETITIONS_FILE.exists():
        raise FileNotFoundError(
            f"competitions.csv not found:\n"
            f"{COMPETITIONS_FILE}"
        )

    competitions = pd.read_csv(
        COMPETITIONS_FILE,
        low_memory=False,
    )

    required = {
        "competition_id",
        "name",
    }

    missing = (
        required
        - set(
            competitions.columns
        )
    )

    if missing:
        raise RuntimeError(
            "competitions.csv missing columns: "
            f"{sorted(missing)}"
        )

    competitions[
        "competition_id"
    ] = (
        competitions[
            "competition_id"
        ]
        .astype(str)
        .str.strip()
    )

    competitions[
        "name"
    ] = (
        competitions[
            "name"
        ]
        .astype(str)
        .str.strip()
    )

    return competitions


# ============================================================
# IDENTIFY UCL COMPETITION
# ============================================================


def get_champions_league_competition(
    competitions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the competition proper using Transfermarkt's
    stable competition ID: CL.

    This intentionally excludes CLQ qualifiers.
    """

    ucl = competitions[
        competitions[
            "competition_id"
        ]
        ==
        UCL_COMPETITION_ID
    ].copy()

    if not ucl.empty:
        return ucl

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------
    #
    # Upstream naming could theoretically change. If CL is
    # missing, inspect name / competition_code using a
    # normalized text search but explicitly exclude
    # qualifying.
    # --------------------------------------------------------

    searchable_columns = [
        column
        for column in [
            "name",
            "competition_code",
        ]
        if column in competitions.columns
    ]

    candidate_mask = pd.Series(
        False,
        index=competitions.index,
    )

    for column in searchable_columns:

        values = (
            competitions[
                column
            ]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(
                "-",
                " ",
                regex=False,
            )
            .str.replace(
                "_",
                " ",
                regex=False,
            )
        )

        candidate_mask = (
            candidate_mask
            |
            (
                values.str.contains(
                    "champions",
                    na=False,
                )
                &
                values.str.contains(
                    "league",
                    na=False,
                )
                &
                ~values.str.contains(
                    "qualif",
                    na=False,
                )
            )
        )

    return competitions[
        candidate_mask
    ].copy()


# ============================================================
# LOAD UCL GAMES
# ============================================================


def load_ucl_games(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load only competition_id == CL.

    CLQ qualifiers are excluded.
    """

    frames = []

    usecols = [
        "game_id",
        "competition_id",
        "season",
        "round",
        "date",
        "home_club_id",
        "home_club_name",
        "away_club_id",
        "away_club_name",
        "home_club_goals",
        "away_club_goals",
    ]

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        competition_ids = (
            chunk[
                "competition_id"
            ]
            .astype(str)
            .str.strip()
        )

        mask = (
            competition_ids
            ==
            UCL_COMPETITION_ID
        )

        if mask.any():

            frames.append(
                chunk[
                    mask
                ].copy()
            )

    if not frames:

        return pd.DataFrame(
            columns=usecols
        )

    ucl_games = pd.concat(
        frames,
        ignore_index=True,
    )

    ucl_games[
        "date"
    ] = pd.to_datetime(
        ucl_games[
            "date"
        ],
        errors="coerce",
    )

    ucl_games[
        "home_club_id"
    ] = pd.to_numeric(
        ucl_games[
            "home_club_id"
        ],
        errors="coerce",
    )

    ucl_games[
        "away_club_id"
    ] = pd.to_numeric(
        ucl_games[
            "away_club_id"
        ],
        errors="coerce",
    )

    ucl_games = ucl_games.sort_values(
        by=[
            "date",
            "game_id",
        ]
    ).reset_index(
        drop=True
    )

    return ucl_games


# ============================================================
# UCL CLUB TABLE
# ============================================================


def build_ucl_club_table(
    ucl_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create unique club_id / club_name pairs observed in
    Transfermarkt Champions League games.
    """

    home = (
        ucl_games[
            [
                "home_club_id",
                "home_club_name",
            ]
        ]
        .rename(
            columns={
                "home_club_id": (
                    "club_id"
                ),
                "home_club_name": (
                    "club_name"
                ),
            }
        )
    )

    away = (
        ucl_games[
            [
                "away_club_id",
                "away_club_name",
            ]
        ]
        .rename(
            columns={
                "away_club_id": (
                    "club_id"
                ),
                "away_club_name": (
                    "club_name"
                ),
            }
        )
    )

    clubs = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    clubs = clubs.dropna(
        subset=[
            "club_id"
        ]
    ).copy()

    clubs[
        "club_id"
    ] = clubs[
        "club_id"
    ].astype(
        int
    )

    clubs[
        "club_name"
    ] = (
        clubs[
            "club_name"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    clubs = clubs.drop_duplicates(
        subset=[
            "club_id",
            "club_name",
        ]
    )

    clubs = clubs.sort_values(
        by=[
            "club_name",
            "club_id",
        ]
    ).reset_index(
        drop=True
    )

    return clubs


# ============================================================
# MAIN
# ============================================================


def collect_transfermarkt_games() -> None:
    """
    Audit Transfermarkt historical games and isolate
    Champions League competition-proper matches.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - TRANSFERMARKT GAME COLLECTION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # EXTRACT / CACHE
    # --------------------------------------------------------

    games_file = extract_games()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    columns = validate_header(
        games_file
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT GAMES DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        "\nRequired columns: valid ✓"
    )

    print(
        f"Columns available: "
        f"{len(columns):,}"
    )

    # --------------------------------------------------------
    # FULL GAMES AUDIT
    # --------------------------------------------------------

    audit = audit_games(
        games_file
    )

    print(
        f"Rows: "
        f"{audit['total_rows']:,}"
    )

    print(
        f"Unique game IDs: "
        f"{audit['unique_game_ids']:,}"
    )

    print(
        f"Duplicate game IDs: "
        f"{audit['duplicate_game_ids']:,}"
    )

    print(
        f"Rows missing core IDs/date: "
        f"{audit['missing_core_rows']:,}"
    )

    if audit[
        "duplicate_game_ids"
    ] > 0:

        raise RuntimeError(
            "Duplicate game IDs detected."
        )

    if audit[
        "minimum_date"
    ] is not None:

        print(
            f"Date range: "
            f"{audit['minimum_date'].date()} "
            f"to "
            f"{audit['maximum_date'].date()}"
        )

    # --------------------------------------------------------
    # COMPETITION IDENTIFICATION
    # --------------------------------------------------------

    competitions = load_competitions()

    ucl_competition = (
        get_champions_league_competition(
            competitions
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CHAMPIONS LEAGUE COMPETITION IDENTIFICATION"
    )

    print(
        "=" * 75
    )

    print(
        f"\nExpected competition ID: "
        f"{UCL_COMPETITION_ID}"
    )

    if ucl_competition.empty:

        raise RuntimeError(
            "Unable to identify Transfermarkt "
            "Champions League competition."
        )

    display_columns = [
        column
        for column in [
            "competition_id",
            "competition_code",
            "name",
            "type",
            "sub_type",
            "country_name",
        ]
        if column in ucl_competition.columns
    ]

    print(
        "\nMatched competition:"
    )

    print(
        ucl_competition[
            display_columns
        ].to_string(
            index=False
        )
    )

    game_count = audit[
        "competition_counts"
    ].get(
        UCL_COMPETITION_ID,
        0,
    )

    print(
        f"\nGames with competition_id="
        f"{UCL_COMPETITION_ID}: "
        f"{game_count:,}"
    )

    qualifying_count = audit[
        "competition_counts"
    ].get(
        "CLQ",
        0,
    )

    print(
        f"Qualifying games with "
        f"competition_id=CLQ: "
        f"{qualifying_count:,}"
    )

    print(
        "CLQ is excluded from our UCL mapping ✓"
    )

    # --------------------------------------------------------
    # LOAD UCL GAMES
    # --------------------------------------------------------

    ucl_games = load_ucl_games(
        games_file
    )

    if ucl_games.empty:

        raise RuntimeError(
            "No competition_id=CL games found."
        )

    # --------------------------------------------------------
    # VALIDATE UCL GAME IDS
    # --------------------------------------------------------

    if ucl_games[
        "game_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate UCL game IDs detected."
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT UCL GAME SAMPLE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nUCL rows found: "
        f"{len(ucl_games):,}"
    )

    print(
        f"UCL unique game IDs: "
        f"{ucl_games['game_id'].nunique():,}"
    )

    print(
        f"UCL date range: "
        f"{ucl_games['date'].min().date()} "
        f"to "
        f"{ucl_games['date'].max().date()}"
    )

    # --------------------------------------------------------
    # UCL CLUB IDS
    # --------------------------------------------------------

    club_table = build_ucl_club_table(
        ucl_games
    )

    unique_club_ids = club_table[
        "club_id"
    ].nunique()

    print(
        f"Unique UCL club IDs in "
        f"Transfermarkt games: "
        f"{unique_club_ids:,}"
    )

    print(
        "\nFirst 20 UCL club identities:"
    )

    print(
        club_table.head(
            20
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # GAME EXAMPLES
    # --------------------------------------------------------

    display_game_columns = [
        "game_id",
        "season",
        "date",
        "round",
        "home_club_id",
        "home_club_name",
        "away_club_id",
        "away_club_name",
        "home_club_goals",
        "away_club_goals",
    ]

    print(
        "\nFirst 10 UCL games:"
    )

    print(
        ucl_games[
            display_game_columns
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nLast 10 UCL games:"
    )

    print(
        ucl_games[
            display_game_columns
        ]
        .tail(
            10
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT GAME COLLECTION COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nGames file:"
    )

    print(
        GAMES_FILE
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    collect_transfermarkt_games()