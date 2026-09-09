from __future__ import annotations

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
)

from src.utils.team_names import (
    normalize_team_columns,
    get_team_names,
)


# ============================================================
# FILE PATHS
# ============================================================

HISTORICAL_FILE = (
    PROCESSED_DATA_DIR
    / "matches_clean.csv"
)

UCL_2022_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2022_2023.csv"
)

RECENT_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_recent_matches.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)


# ============================================================
# CURRENT EXPECTED DATASET SIZE
# ============================================================

EXPECTED_TOTAL_MATCHES = 2089

EXPECTED_RECENT_SEASON_COUNTS = {
    2022: 125,
    2023: 125,
    2024: 189,
    2025: 189,
}


# ============================================================
# COMMON HELPERS
# ============================================================


def determine_result(
    home_goals: int,
    away_goals: int,
) -> str:
    """
    Determine the match result from the home team's perspective.

    H = Home win
    D = Draw
    A = Away win
    """

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def build_scoreline(
    home_goals: int,
    away_goals: int,
) -> str:
    """
    Build a standard scoreline string.

    Example:
    2, 1 -> "2-1"
    """

    return f"{home_goals}-{away_goals}"


# ============================================================
# LOAD DATA
# ============================================================


def load_csv(
    file_path,
    source_name: str,
) -> pd.DataFrame:
    """
    Load a CSV and perform basic validation.
    """

    print(
        f"\nLoading {source_name}:"
    )

    print(
        file_path
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    return df


# ============================================================
# STANDARDIZE HISTORICAL KAGGLE DATA
# ============================================================


def standardize_historical(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the cleaned Kaggle dataset into the master schema.

    Kaggle does not provide match stage information in our
    cleaned table, so stage is marked as 'not_available'.
    """

    df = dataframe.copy()

    required_columns = {
        "match_id",
        "date",
        "day",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "scoreline",
        "competition",
        "source",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Historical dataset is missing columns: "
            f"{sorted(missing)}"
        )

    df = normalize_team_columns(
        df
    )

    df = df.rename(
        columns={
            "match_id": "source_match_id",
        }
    )

    df["stage"] = "not_available"

    df["status"] = "FINISHED"

    return df


# ============================================================
# STANDARDIZE API DATA
# ============================================================


def standardize_api_matches(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert API datasets into the master schema.
    """

    df = dataframe.copy()

    required_columns = {
        "api_match_id",
        "date",
        "day",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "scoreline",
        "competition",
        "source",
        "stage",
        "status",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "API dataset is missing columns: "
            f"{sorted(missing)}"
        )

    df = normalize_team_columns(
        df
    )

    df = df.rename(
        columns={
            "api_match_id": "source_match_id",
        }
    )

    return df


# ============================================================
# MASTER SCHEMA
# ============================================================


def enforce_master_schema(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert all source data into consistent column types
    and ordering.
    """

    df = dataframe.copy()

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df["day"] = (
        df["date"]
        .dt.day_name()
        .str[:3]
    )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [
        "season",
        "home_goals",
        "away_goals",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype(
            int
        )

    # Source IDs differ between providers,
    # so nullable integer is safer.
    df["source_match_id"] = pd.to_numeric(
        df["source_match_id"],
        errors="coerce",
    ).astype(
        "Int64"
    )

    # --------------------------------------------------------
    # REBUILD TARGET / SCORELINE
    # --------------------------------------------------------
    #
    # We calculate these ourselves from the goals instead
    # of trusting provider-specific result labels.
    # --------------------------------------------------------

    df["result"] = df.apply(
        lambda row: determine_result(
            int(row["home_goals"]),
            int(row["away_goals"]),
        ),
        axis=1,
    )

    df["scoreline"] = df.apply(
        lambda row: build_scoreline(
            int(row["home_goals"]),
            int(row["away_goals"]),
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # COMPETITION
    # --------------------------------------------------------

    df["competition"] = (
        "UEFA Champions League"
    )

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

    columns = [
        "source_match_id",
        "date",
        "day",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "scoreline",
        "competition",
        "stage",
        "status",
        "source",
    ]

    df = df[
        columns
    ]

    return df


# ============================================================
# DATA QUALITY CHECKS
# ============================================================


def run_quality_checks(
    dataframe: pd.DataFrame,
) -> None:
    """
    Run important consistency checks before saving
    the master match dataset.
    """

    df = dataframe

    print("\n" + "=" * 75)
    print("MASTER DATA QUALITY CHECK")
    print("=" * 75)

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "date",
            "home_team",
            "away_team",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        print(
            "\nDUPLICATE MATCHES FOUND:"
        )

        print(
            df.loc[
                duplicate_mask,
                [
                    "date",
                    "home_team",
                    "away_team",
                    "source",
                ],
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Master dataset contains duplicate matches."
        )

    print(
        "\nDuplicate matches: 0"
    )

    # --------------------------------------------------------
    # MISSING CORE VALUES
    # --------------------------------------------------------

    core_columns = [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "source",
    ]

    missing_core = (
        df[
            core_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_core > 0:

        raise RuntimeError(
            f"Missing core values found: {missing_core}"
        )

    print(
        "Missing core values: 0"
    )

    # --------------------------------------------------------
    # GOAL VALIDATION
    # --------------------------------------------------------

    negative_goals = (
        (
            df["home_goals"] < 0
        )
        |
        (
            df["away_goals"] < 0
        )
    ).sum()

    if negative_goals > 0:

        raise RuntimeError(
            "Negative goal values were detected."
        )

    print(
        "Negative goal values: 0"
    )

    # --------------------------------------------------------
    # RESULT VALIDATION
    # --------------------------------------------------------

    invalid_results = (
        ~df[
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
            "Invalid result labels were detected."
        )

    print(
        "Invalid result labels: 0"
    )

    # --------------------------------------------------------
    # HOME VS AWAY
    # --------------------------------------------------------

    same_team = (
        df["home_team"]
        ==
        df["away_team"]
    ).sum()

    if same_team > 0:

        raise RuntimeError(
            "Matches found where home_team equals away_team."
        )

    print(
        "Home team equals away team: 0"
    )

    # --------------------------------------------------------
    # ENCODING CHECK
    # --------------------------------------------------------

    suspicious_markers = (
        "Ã",
        "Â",
        "Å½",
        "Å ",
        "â€",
        "â€™",
    )

    suspicious_teams = set()

    for column in [
        "home_team",
        "away_team",
    ]:

        for team in df[
            column
        ].dropna().unique():

            if any(
                marker in str(team)
                for marker in suspicious_markers
            ):

                suspicious_teams.add(
                    team
                )

    if suspicious_teams:

        raise RuntimeError(
            "Suspicious team-name encoding remains: "
            f"{sorted(suspicious_teams)}"
        )

    print(
        "Obvious encoding problems: 0"
    )

    # --------------------------------------------------------
    # EXPECTED TOTAL
    # --------------------------------------------------------

    if len(df) != EXPECTED_TOTAL_MATCHES:

        raise RuntimeError(
            "Unexpected master match count. "
            f"Expected {EXPECTED_TOTAL_MATCHES:,}, "
            f"received {len(df):,}."
        )

    print(
        f"Expected total matches: "
        f"{EXPECTED_TOTAL_MATCHES:,} ✓"
    )

    # --------------------------------------------------------
    # RECENT SEASONS
    # --------------------------------------------------------

    season_counts = (
        df[
            "season"
        ]
        .value_counts()
        .sort_index()
    )

    for (
        season,
        expected_count,
    ) in EXPECTED_RECENT_SEASON_COUNTS.items():

        actual_count = int(
            season_counts.get(
                season,
                0,
            )
        )

        if actual_count != expected_count:

            raise RuntimeError(
                f"Season {season} expected "
                f"{expected_count} matches but found "
                f"{actual_count}."
            )

    print(
        "Recent season counts: correct ✓"
    )


# ============================================================
# BUILD MASTER DATASET
# ============================================================


def build_master_match_dataset() -> pd.DataFrame:
    """
    Build the complete normalized Champions League
    historical match database.

    Current coverage:
    2010/11 through 2025/26.
    """

    print("=" * 75)
    print("UCL PREDICTOR - MASTER MATCH DATASET")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD SOURCES
    # --------------------------------------------------------

    historical_raw = load_csv(
        HISTORICAL_FILE,
        "Kaggle 2010/11-2021/22",
    )

    season_2022_raw = load_csv(
        UCL_2022_FILE,
        "API-Football 2022/23",
    )

    recent_raw = load_csv(
        RECENT_FILE,
        "football-data.org 2023/24-2025/26",
    )

    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    historical = standardize_historical(
        historical_raw
    )

    season_2022 = standardize_api_matches(
        season_2022_raw
    )

    recent = standardize_api_matches(
        recent_raw
    )

    print("\n" + "=" * 75)
    print("SOURCE COUNTS")
    print("=" * 75)

    print(
        f"\nKaggle historical: "
        f"{len(historical):,}"
    )

    print(
        f"API-Football 2022/23: "
        f"{len(season_2022):,}"
    )

    print(
        f"football-data.org recent: "
        f"{len(recent):,}"
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    master = pd.concat(
        [
            historical,
            season_2022,
            recent,
        ],
        ignore_index=True,
        sort=False,
    )

    master = enforce_master_schema(
        master
    )

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    master = master.sort_values(
        by=[
            "date",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # CREATE OUR OWN PERMANENT MATCH ID
    # --------------------------------------------------------

    master.insert(
        0,
        "match_id",
        range(
            1,
            len(master) + 1,
        ),
    )

    # --------------------------------------------------------
    # QUALITY CHECKS
    # --------------------------------------------------------

    run_quality_checks(
        master
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("MASTER DATASET SUMMARY")
    print("=" * 75)

    print(
        f"\nMatches: "
        f"{len(master):,}"
    )

    print(
        f"Date range: "
        f"{master['date'].min().date()} "
        f"to "
        f"{master['date'].max().date()}"
    )

    print(
        f"Season range: "
        f"{master['season'].min()} "
        f"to "
        f"{master['season'].max()}"
    )

    unique_teams = get_team_names(
        master
    )

    print(
        f"Canonical clubs: "
        f"{len(unique_teams):,}"
    )

    print("\nMatches by source:")

    print(
        master[
            "source"
        ]
        .value_counts()
        .to_string()
    )

    print("\nMatches by season:")

    print(
        master[
            "season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nResult distribution:")

    result_counts = (
        master[
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
        result_counts.to_string()
    )

    print("\nResult percentages:")

    result_percentages = (
        result_counts
        / len(master)
        * 100
    ).round(
        2
    )

    print(
        result_percentages.to_string()
    )

    # --------------------------------------------------------
    # SAVE MASTER DATASET
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 75)

    print(
        "MASTER DATASET CREATED SUCCESSFULLY"
    )

    print("=" * 75)

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\nFirst 10 matches:")

    print(
        master.head(
            10
        ).to_string(
            index=False
        )
    )

    print("\nLast 10 matches:")

    print(
        master.tail(
            10
        ).to_string(
            index=False
        )
    )

    return master


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    build_master_match_dataset()