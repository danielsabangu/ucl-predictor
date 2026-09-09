from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

START_SEASON = 2010
LAST_COMPLETE_SEASON = 2025
CURRENT_SEASON = 2026

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# Reuse the 80 historical files already downloaded.
#
# Set True only if you deliberately want to re-download every
# historical file.
FORCE_HISTORICAL_REDOWNLOAD = False


# ============================================================
# EXPECTED HISTORICAL DATA
# ============================================================
#
# Big Five domestic leagues:
#
# Premier League:
#     380 matches per season
#
# La Liga:
#     380 matches per season
#
# Bundesliga:
#     306 matches per season
#
# Serie A:
#     380 matches per season
#
# Ligue 1:
#     Normally 380 until the move to 18 clubs.
#
#     2019/20 was ended early -> 279 matches.
#
#     2023/24 onward -> 306 matches.
# ============================================================

EXPECTED_HISTORICAL_TOTAL = 28893


# ============================================================
# DATA SOURCES
# ============================================================

GITHUB_DATASET_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "datasets/football-datasets/main/datasets"
)

OPENFOOTBALL_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "openfootball/football.json/master"
)


# ============================================================
# HISTORICAL BIG FIVE
# ============================================================

HISTORICAL_LEAGUES = {
    "premier-league": {
        "country": "England",
        "league": "Premier League",
        "league_code": "ENG1",
        "normal_matches": 380,
    },
    "la-liga": {
        "country": "Spain",
        "league": "La Liga",
        "league_code": "ESP1",
        "normal_matches": 380,
    },
    "bundesliga": {
        "country": "Germany",
        "league": "Bundesliga",
        "league_code": "GER1",
        "normal_matches": 306,
    },
    "serie-a": {
        "country": "Italy",
        "league": "Serie A",
        "league_code": "ITA1",
        "normal_matches": 380,
    },
    "ligue-1": {
        "country": "France",
        "league": "Ligue 1",
        "league_code": "FRA1",
        "normal_matches": 380,
    },
}


# ============================================================
# CURRENT OPENFOOTBALL LEAGUES
# ============================================================

CURRENT_LEAGUES = {
    "en.1.json": {
        "country": "England",
        "league": "Premier League",
        "league_code": "ENG1",
    },
    "es.1.json": {
        "country": "Spain",
        "league": "La Liga",
        "league_code": "ESP1",
    },
    "de.1.json": {
        "country": "Germany",
        "league": "Bundesliga",
        "league_code": "GER1",
    },
    "it.1.json": {
        "country": "Italy",
        "league": "Serie A",
        "league_code": "ITA1",
    },
    "fr.1.json": {
        "country": "France",
        "league": "Ligue 1",
        "league_code": "FRA1",
    },
    "nl.1.json": {
        "country": "Netherlands",
        "league": "Eredivisie",
        "league_code": "NED1",
    },
    "pt.1.json": {
        "country": "Portugal",
        "league": "Primeira Liga",
        "league_code": "POR1",
    },
}


# ============================================================
# OUTPUT PATHS
# ============================================================

RAW_HISTORICAL_DIR = (
    RAW_DATA_DIR
    / "matches"
    / "domestic"
    / "github_football_datasets"
)

RAW_CURRENT_DIR = (
    RAW_DATA_DIR
    / "matches"
    / "domestic"
    / "openfootball"
)

OUTPUT_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "domestic_matches.csv"
)


# ============================================================
# SEASON HELPERS
# ============================================================


def season_code(
    starting_year: int,
) -> str:
    """
    Convert season starting year:

    2010 -> 1011
    2025 -> 2526
    """

    first = str(
        starting_year
    )[-2:]

    second = str(
        starting_year + 1
    )[-2:]

    return (
        first
        + second
    )


def season_folder(
    starting_year: int,
) -> str:
    """
    Convert:

    2026 -> 2026-27
    """

    return (
        f"{starting_year}-"
        f"{str(starting_year + 1)[-2:]}"
    )


def season_label(
    starting_year: int,
) -> str:
    """
    Convert:

    2025 -> 2025/26
    """

    return (
        f"{starting_year}/"
        f"{str(starting_year + 1)[-2:]}"
    )


# ============================================================
# EXPECTED MATCH COUNTS
# ============================================================


def expected_historical_match_count(
    league_slug: str,
    season: int,
) -> int:
    """
    Return the expected number of matches for one complete
    historical domestic season.
    """

    if league_slug == "bundesliga":
        return 306

    if league_slug == "ligue-1":

        # COVID-shortened Ligue 1 season.
        if season == 2019:
            return 279

        # Ligue 1 changed from 20 to 18 clubs.
        if season >= 2023:
            return 306

        return 380

    return 380


# ============================================================
# RESULT
# ============================================================


def determine_result(
    home_goals: int,
    away_goals: int,
) -> str:
    """
    Return:

    H = home win
    D = draw
    A = away win
    """

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


# ============================================================
# HTTP
# ============================================================


def download_bytes(
    url: str,
) -> bytes | None:
    """
    Download a file with retries.
    """

    headers = {
        "User-Agent": (
            "UCL-Predictor/1.0"
        )
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as error:

            print(
                f"      Request error "
                f"({attempt}/{MAX_RETRIES}): "
                f"{error}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    2
                )

            continue

        if response.status_code == 200:

            return response.content

        if response.status_code == 404:

            print(
                "      File not available."
            )

            return None

        if response.status_code in {
            429,
            500,
            502,
            503,
            504,
        }:

            print(
                f"      HTTP "
                f"{response.status_code} "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    3
                )

            continue

        print(
            f"      Unexpected HTTP "
            f"{response.status_code}"
        )

        return None

    return None


# ============================================================
# URL BUILDERS
# ============================================================


def build_historical_url(
    league_slug: str,
    season: int,
) -> str:
    """
    Build historical GitHub mirror URL.
    """

    code = season_code(
        season
    )

    return (
        f"{GITHUB_DATASET_BASE_URL}/"
        f"{league_slug}/"
        f"season-{code}.csv"
    )


def build_current_url(
    filename: str,
) -> str:
    """
    Build current OpenFootball URL.
    """

    folder = season_folder(
        CURRENT_SEASON
    )

    return (
        f"{OPENFOOTBALL_BASE_URL}/"
        f"{folder}/"
        f"{filename}"
    )


# ============================================================
# RAW PATH HELPERS
# ============================================================


def historical_raw_path(
    *,
    league_slug: str,
    season: int,
) -> Path:
    """
    Return historical local raw path.
    """

    return (
        RAW_HISTORICAL_DIR
        / season_folder(
            season
        )
        / f"{league_slug}.csv"
    )


def current_raw_path(
    filename: str,
) -> Path:
    """
    Return current-season local raw path.
    """

    return (
        RAW_CURRENT_DIR
        / season_folder(
            CURRENT_SEASON
        )
        / filename
    )


# ============================================================
# HISTORICAL FILE ACQUISITION
# ============================================================


def get_historical_content(
    *,
    league_slug: str,
    season: int,
) -> bytes | None:
    """
    Reuse historical files already downloaded.

    This avoids making another 80 GitHub requests when
    the raw files are already present.
    """

    output_file = historical_raw_path(
        league_slug=league_slug,
        season=season,
    )

    if (
        output_file.exists()
        and not FORCE_HISTORICAL_REDOWNLOAD
    ):

        print(
            "      Using cached raw file."
        )

        return output_file.read_bytes()

    url = build_historical_url(
        league_slug,
        season,
    )

    print(
        f"      {url}"
    )

    content = download_bytes(
        url
    )

    if content is None:
        return None

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_bytes(
        content
    )

    print(
        f"      Raw saved: "
        f"{output_file}"
    )

    return content


# ============================================================
# CURRENT FILE ACQUISITION
# ============================================================


def get_current_content(
    *,
    filename: str,
) -> bytes | None:
    """
    Download current-season data fresh each run.

    Current season changes continuously, so unlike historical
    files we refresh these files.
    """

    url = build_current_url(
        filename
    )

    print(
        f"      {url}"
    )

    content = download_bytes(
        url
    )

    if content is None:
        return None

    output_file = current_raw_path(
        filename
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_bytes(
        content
    )

    print(
        f"      Raw saved: "
        f"{output_file}"
    )

    return content


# ============================================================
# HISTORICAL CSV READER
# ============================================================


def read_historical_csv(
    content: bytes,
) -> pd.DataFrame:
    """
    Read historical CSV.

    The GitHub mirror uses standardized ISO dates, but older
    text encodings can still vary.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:

        try:

            return pd.read_csv(
                io.BytesIO(
                    content
                ),
                encoding=encoding,
                on_bad_lines="skip",
            )

        except UnicodeDecodeError as error:

            last_error = error

    raise RuntimeError(
        "Unable to decode historical CSV."
    ) from last_error


# ============================================================
# HISTORICAL STANDARDIZATION
# ============================================================


def standardize_historical(
    *,
    dataframe: pd.DataFrame,
    season: int,
    league_slug: str,
) -> pd.DataFrame:
    """
    Standardize one historical league season.

    IMPORTANT FIX:

    The GitHub mirror uses ISO dates:

        YYYY-MM-DD

    We therefore explicitly parse using:

        format="%Y-%m-%d"

    We do NOT use dayfirst=True.
    """

    info = HISTORICAL_LEAGUES[
        league_slug
    ]

    df = dataframe.copy()

    required_columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    ]

    missing = (
        set(
            required_columns
        )
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Missing historical columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # EXPECTED RAW COMPLETED ROWS
    # --------------------------------------------------------

    raw_complete_rows = len(
        df.dropna(
            subset=required_columns
        )
    )

    # --------------------------------------------------------
    # SELECT CORE COLUMNS
    # --------------------------------------------------------

    df = df[
        required_columns
    ].copy()

    df = df.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        }
    )

    # --------------------------------------------------------
    # DATE FIX
    # --------------------------------------------------------

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        format="%Y-%m-%d",
        errors="coerce",
    )

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    df[
        "home_goals"
    ] = pd.to_numeric(
        df[
            "home_goals"
        ],
        errors="coerce",
    )

    df[
        "away_goals"
    ] = pd.to_numeric(
        df[
            "away_goals"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    # --------------------------------------------------------
    # PARSER LOSS CHECK
    # --------------------------------------------------------

    if len(df) != raw_complete_rows:

        raise RuntimeError(
            f"Parser lost historical rows. "
            f"Raw complete rows: {raw_complete_rows:,}. "
            f"Parsed rows: {len(df):,}."
        )

    # --------------------------------------------------------
    # INTEGER GOALS
    # --------------------------------------------------------

    df[
        "home_goals"
    ] = df[
        "home_goals"
    ].astype(
        int
    )

    df[
        "away_goals"
    ] = df[
        "away_goals"
    ].astype(
        int
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    df[
        "result"
    ] = df.apply(
        lambda row: determine_result(
            int(
                row[
                    "home_goals"
                ]
            ),
            int(
                row[
                    "away_goals"
                ]
            ),
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # SCORELINE
    # --------------------------------------------------------

    df[
        "scoreline"
    ] = (
        df[
            "home_goals"
        ].astype(
            str
        )
        +
        "-"
        +
        df[
            "away_goals"
        ].astype(
            str
        )
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    df[
        "season"
    ] = season

    df[
        "season_label"
    ] = season_label(
        season
    )

    df[
        "country"
    ] = info[
        "country"
    ]

    df[
        "league"
    ] = info[
        "league"
    ]

    df[
        "league_code"
    ] = info[
        "league_code"
    ]

    df[
        "competition_type"
    ] = "domestic_league"

    df[
        "source"
    ] = "github_football_datasets"

    # --------------------------------------------------------
    # EXPECTED SEASON COUNT
    # --------------------------------------------------------

    expected_count = (
        expected_historical_match_count(
            league_slug,
            season,
        )
    )

    if len(df) != expected_count:

        raise RuntimeError(
            f"{info['league']} "
            f"{season_label(season)} "
            f"expected {expected_count:,} matches "
            f"but parsed {len(df):,}."
        )

    return df[
        [
            "date",
            "season",
            "season_label",
            "country",
            "league",
            "league_code",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "scoreline",
            "competition_type",
            "source",
        ]
    ]


# ============================================================
# OPENFOOTBALL SCORE PARSER
# ============================================================


def extract_openfootball_full_time_score(
    match: dict,
) -> tuple[int, int] | None:
    """
    Return a completed full-time score if one exists.

    OpenFootball current files can contain different score
    shapes.

    Completed match example:

        "score": {
            "ht": [1, 0],
            "ft": [3, 0]
        }

    Unplayed / placeholder example:

        "score": [0, 0]

    A plain list must NOT be treated as a completed result.
    """

    score = match.get(
        "score"
    )

    if not isinstance(
        score,
        dict,
    ):

        return None

    full_time = score.get(
        "ft"
    )

    if not isinstance(
        full_time,
        (list, tuple),
    ):

        return None

    if len(
        full_time
    ) != 2:

        return None

    home_goals = full_time[
        0
    ]

    away_goals = full_time[
        1
    ]

    if (
        home_goals is None
        or away_goals is None
    ):

        return None

    try:

        return (
            int(
                home_goals
            ),
            int(
                away_goals
            ),
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# CURRENT JSON PARSER
# ============================================================


def parse_current_json(
    *,
    content: bytes,
    filename: str,
) -> tuple[pd.DataFrame, int, int]:
    """
    Parse one current OpenFootball league.

    Returns:

    dataframe
    completed_count
    skipped_nonfinal_count
    """

    info = CURRENT_LEAGUES[
        filename
    ]

    data = json.loads(
        content.decode(
            "utf-8"
        )
    )

    matches = data.get(
        "matches",
        []
    )

    if not isinstance(
        matches,
        list,
    ):

        raise RuntimeError(
            "OpenFootball 'matches' field "
            "is not a list."
        )

    rows = []

    skipped_nonfinal = 0

    for match in matches:

        if not isinstance(
            match,
            dict,
        ):

            skipped_nonfinal += 1
            continue

        final_score = (
            extract_openfootball_full_time_score(
                match
            )
        )

        if final_score is None:

            skipped_nonfinal += 1
            continue

        (
            home_goals,
            away_goals,
        ) = final_score

        date = pd.to_datetime(
            match.get(
                "date"
            ),
            format="%Y-%m-%d",
            errors="coerce",
        )

        if pd.isna(
            date
        ):

            continue

        home_team = match.get(
            "team1"
        )

        away_team = match.get(
            "team2"
        )

        if (
            not home_team
            or not away_team
        ):

            continue

        rows.append(
            {
                "date": date,
                "season": CURRENT_SEASON,
                "season_label": season_label(
                    CURRENT_SEASON
                ),
                "country": info[
                    "country"
                ],
                "league": info[
                    "league"
                ],
                "league_code": info[
                    "league_code"
                ],
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": determine_result(
                    home_goals,
                    away_goals,
                ),
                "scoreline": (
                    f"{home_goals}-"
                    f"{away_goals}"
                ),
                "competition_type": (
                    "domestic_league"
                ),
                "source": "openfootball",
            }
        )

    df = pd.DataFrame(
        rows
    )

    return (
        df,
        len(rows),
        skipped_nonfinal,
    )


# ============================================================
# HISTORICAL COLLECTION
# ============================================================


def collect_historical_big_five(
) -> list[pd.DataFrame]:
    """
    Load/download all Big Five league seasons from
    2010/11 through 2025/26.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "HISTORICAL BIG FIVE"
    )

    print(
        "=" * 75
    )

    frames = []

    successful = 0
    unavailable = 0
    failed = 0
    historical_rows = 0

    for season in range(
        START_SEASON,
        LAST_COMPLETE_SEASON + 1,
    ):

        print(
            "\n"
            + "-" * 75
        )

        print(
            f"SEASON "
            f"{season_label(season)}"
        )

        print(
            "-" * 75
        )

        for (
            league_slug,
            info,
        ) in HISTORICAL_LEAGUES.items():

            print(
                f"\n  "
                f"{info['country']} - "
                f"{info['league']}"
            )

            content = get_historical_content(
                league_slug=league_slug,
                season=season,
            )

            if content is None:

                unavailable += 1

                print(
                    "      No data."
                )

                continue

            try:

                raw_df = read_historical_csv(
                    content
                )

                clean_df = standardize_historical(
                    dataframe=raw_df,
                    season=season,
                    league_slug=league_slug,
                )

            except Exception as error:

                failed += 1

                print(
                    f"      Processing failed: "
                    f"{error}"
                )

                continue

            frames.append(
                clean_df
            )

            successful += 1

            historical_rows += len(
                clean_df
            )

            print(
                f"      Completed matches: "
                f"{len(clean_df):,} ✓"
            )

    print(
        "\nHistorical source summary:"
    )

    print(
        f"  Successful files: "
        f"{successful:,}"
    )

    print(
        f"  Unavailable files: "
        f"{unavailable:,}"
    )

    print(
        f"  Processing failures: "
        f"{failed:,}"
    )

    print(
        f"  Historical matches: "
        f"{historical_rows:,}"
    )

    # --------------------------------------------------------
    # STRICT HISTORICAL VALIDATION
    # --------------------------------------------------------

    expected_files = (
        (
            LAST_COMPLETE_SEASON
            - START_SEASON
            + 1
        )
        * len(
            HISTORICAL_LEAGUES
        )
    )

    if successful != expected_files:

        raise RuntimeError(
            f"Expected {expected_files} successful "
            f"historical league-season files, "
            f"received {successful}."
        )

    if unavailable != 0:

        raise RuntimeError(
            "One or more historical files "
            "were unavailable."
        )

    if failed != 0:

        raise RuntimeError(
            "One or more historical files "
            "failed processing."
        )

    if historical_rows != EXPECTED_HISTORICAL_TOTAL:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_HISTORICAL_TOTAL:,} "
            f"historical matches but found "
            f"{historical_rows:,}."
        )

    print(
        f"\nHistorical total validation: "
        f"{EXPECTED_HISTORICAL_TOTAL:,} ✓"
    )

    return frames


# ============================================================
# CURRENT COLLECTION
# ============================================================


def collect_current_season(
) -> list[pd.DataFrame]:
    """
    Collect completed 2026/27 domestic matches.

    OpenFootball files may also contain upcoming fixtures.

    Upcoming fixtures are intentionally skipped.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT DOMESTIC SEASON 2026/27"
    )

    print(
        "=" * 75
    )

    frames = []

    total_completed = 0
    total_skipped = 0

    for (
        filename,
        info,
    ) in CURRENT_LEAGUES.items():

        print(
            f"\n  "
            f"{info['country']} - "
            f"{info['league']}"
        )

        content = get_current_content(
            filename=filename
        )

        if content is None:

            print(
                "      Current file unavailable."
            )

            continue

        try:

            (
                df,
                completed_count,
                skipped_count,
            ) = parse_current_json(
                content=content,
                filename=filename,
            )

        except Exception as error:

            print(
                f"      Processing failed: "
                f"{error}"
            )

            continue

        total_completed += (
            completed_count
        )

        total_skipped += (
            skipped_count
        )

        if df.empty:

            print(
                "      Completed matches: 0"
            )

            print(
                f"      Upcoming/non-final skipped: "
                f"{skipped_count:,}"
            )

            continue

        frames.append(
            df
        )

        print(
            f"      Completed matches: "
            f"{completed_count:,}"
        )

        print(
            f"      Upcoming/non-final skipped: "
            f"{skipped_count:,}"
        )

    print(
        "\nCurrent-season summary:"
    )

    print(
        f"  Completed matches retained: "
        f"{total_completed:,}"
    )

    print(
        f"  Upcoming/non-final skipped: "
        f"{total_skipped:,}"
    )

    return frames


# ============================================================
# EXACT DUPLICATE CHECK
# ============================================================


def remove_exact_duplicates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove only rows that are completely identical in match
    identity and score.

    We do NOT silently remove conflicting duplicate matches.
    """

    df = dataframe.copy()

    exact_columns = [
        "date",
        "league_code",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "source",
    ]

    exact_duplicate_count = int(
        df.duplicated(
            subset=exact_columns,
            keep="first",
        ).sum()
    )

    if exact_duplicate_count > 0:

        print(
            f"\nRemoving exact duplicate rows: "
            f"{exact_duplicate_count:,}"
        )

        df = df.drop_duplicates(
            subset=exact_columns,
            keep="first",
        ).reset_index(
            drop=True
        )

    return df


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate combined historical + current domestic data.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DOMESTIC DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # MISSING CORE VALUES
    # --------------------------------------------------------

    core_columns = [
        "date",
        "season",
        "country",
        "league",
        "league_code",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "source",
    ]

    missing_core = (
        dataframe[
            core_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_core > 0:

        raise RuntimeError(
            f"Missing core values: "
            f"{missing_core}"
        )

    print(
        "\nMissing core values: 0 ✓"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    invalid_results = (
        ~dataframe[
            "result"
        ].isin(
            [
                "H",
                "D",
                "A",
            ]
        )
    ).sum()

    if invalid_results > 0:

        raise RuntimeError(
            "Invalid result labels detected."
        )

    print(
        "Result labels: valid ✓"
    )

    # --------------------------------------------------------
    # GOALS
    # --------------------------------------------------------

    negative_goals = (
        (
            dataframe[
                "home_goals"
            ] < 0
        )
        |
        (
            dataframe[
                "away_goals"
            ] < 0
        )
    ).sum()

    if negative_goals > 0:

        raise RuntimeError(
            "Negative goal values detected."
        )

    print(
        "Goal values: valid ✓"
    )

    # --------------------------------------------------------
    # SAME TEAM
    # --------------------------------------------------------

    same_team = (
        dataframe[
            "home_team"
        ]
        ==
        dataframe[
            "away_team"
        ]
    ).sum()

    if same_team > 0:

        raise RuntimeError(
            "Home team equals away team."
        )

    print(
        "Home team equals away team: 0 ✓"
    )

    # --------------------------------------------------------
    # LOGICAL DUPLICATES
    # --------------------------------------------------------

    duplicate_mask = (
        dataframe.duplicated(
            subset=[
                "date",
                "league_code",
                "home_team",
                "away_team",
            ],
            keep=False,
        )
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        print(
            "\nConflicting duplicate matches:"
        )

        print(
            dataframe.loc[
                duplicate_mask,
                [
                    "date",
                    "season",
                    "league",
                    "home_team",
                    "away_team",
                    "home_goals",
                    "away_goals",
                    "source",
                ],
            ]
            .head(
                40
            )
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Conflicting duplicate domestic "
            "matches detected."
        )

    print(
        "Duplicate matches: 0 ✓"
    )

    # --------------------------------------------------------
    # HISTORICAL TOTAL
    # --------------------------------------------------------

    historical = dataframe[
        dataframe[
            "season"
        ]
        <= LAST_COMPLETE_SEASON
    ]

    if len(
        historical
    ) != EXPECTED_HISTORICAL_TOTAL:

        raise RuntimeError(
            f"Historical total should be "
            f"{EXPECTED_HISTORICAL_TOTAL:,}, "
            f"found {len(historical):,}."
        )

    print(
        f"Historical Big Five total: "
        f"{EXPECTED_HISTORICAL_TOTAL:,} ✓"
    )

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    if dataframe[
        "date"
    ].is_monotonic_increasing:

        print(
            "Chronological ordering: valid ✓"
        )

    else:

        raise RuntimeError(
            "Domestic data is not chronologically sorted."
        )


# ============================================================
# MAIN PIPELINE
# ============================================================


def collect_domestic_matches(
) -> pd.DataFrame:
    """
    Build the domestic match dataset.

    Historical:
        Big Five
        2010/11 - 2025/26

    Current:
        2026/27
        England
        Spain
        Germany
        Italy
        France
        Netherlands
        Portugal
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - DOMESTIC MATCH COLLECTION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # HISTORICAL
    # --------------------------------------------------------

    historical_frames = (
        collect_historical_big_five()
    )

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    current_frames = (
        collect_current_season()
    )

    all_frames = (
        historical_frames
        +
        current_frames
    )

    if not all_frames:

        raise RuntimeError(
            "No domestic datasets were collected."
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    domestic = pd.concat(
        all_frames,
        ignore_index=True,
    )

    domestic[
        "date"
    ] = pd.to_datetime(
        domestic[
            "date"
        ],
        errors="raise",
    )

    # --------------------------------------------------------
    # EXACT DUPLICATES
    # --------------------------------------------------------

    domestic = (
        remove_exact_duplicates(
            domestic
        )
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    domestic = domestic.sort_values(
        by=[
            "date",
            "country",
            "league",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # CREATE DOMESTIC MATCH ID
    # --------------------------------------------------------

    domestic.insert(
        0,
        "domestic_match_id",
        range(
            1,
            len(domestic) + 1,
        ),
    )

    # --------------------------------------------------------
    # QUALITY CHECKS
    # --------------------------------------------------------

    run_quality_checks(
        domestic
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    domestic.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DOMESTIC DATA SUMMARY"
    )

    print(
        "=" * 75
    )

    historical = domestic[
        domestic[
            "season"
        ]
        <= LAST_COMPLETE_SEASON
    ]

    current = domestic[
        domestic[
            "season"
        ]
        == CURRENT_SEASON
    ]

    print(
        f"\nHistorical matches: "
        f"{len(historical):,}"
    )

    print(
        f"Current 2026/27 matches: "
        f"{len(current):,}"
    )

    print(
        f"Total domestic matches: "
        f"{len(domestic):,}"
    )

    print(
        f"Date range: "
        f"{domestic['date'].min().date()} "
        f"to "
        f"{domestic['date'].max().date()}"
    )

    unique_teams = set(
        domestic[
            "home_team"
        ]
    ).union(
        set(
            domestic[
                "away_team"
            ]
        )
    )

    print(
        f"Raw team names: "
        f"{len(unique_teams):,}"
    )

    print(
        "\nMatches by source:"
    )

    print(
        domestic[
            "source"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nMatches by league:"
    )

    print(
        domestic.groupby(
            [
                "country",
                "league",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    print(
        "\nHistorical matches by season:"
    )

    print(
        historical[
            "season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nCurrent matches by league:"
    )

    if current.empty:

        print(
            "No current completed matches retained."
        )

    else:

        print(
            current.groupby(
                [
                    "country",
                    "league",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
            .to_string()
        )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nFirst 10 matches:"
    )

    print(
        domestic.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\nLast 10 matches:"
    )

    print(
        domestic.tail(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DOMESTIC MATCH COLLECTION COMPLETED"
    )

    print(
        "=" * 75
    )

    return domestic


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    collect_domestic_matches()