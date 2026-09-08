import unicodedata

import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


RAW_MATCH_FILE = RAW_DATA_DIR / "matches" / "ucl_kaggle" / "df.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "matches_clean.csv"


def fix_text_encoding(value: str) -> str:
    """
    Repair common UTF-8 / Latin-1 mojibake and normalize Unicode text.

    Example:
    AtlÃ©tico Madrid -> Atlético Madrid
    """

    if not isinstance(value, str):
        return value

    value = value.strip()

    bad_markers = ("Ã", "Â", "â")

    if any(marker in value for marker in bad_markers):
        try:
            value = value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    return unicodedata.normalize("NFC", value)


def determine_result(home_goals: int, away_goals: int) -> str:
    """
    Return the match result from the home team's perspective.

    H = Home win
    D = Draw
    A = Away win
    """

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def build_clean_match_dataset() -> pd.DataFrame:
    """
    Build a clean one-row-per-match historical Champions League dataset.

    The raw Kaggle dataset contains each match twice:
    once from the home team's perspective and once from the away team's
    perspective.

    We keep only the rows where home == 1 so that each real match appears once.
    """

    print("=" * 65)
    print("UCL PREDICTOR - CLEAN HISTORICAL MATCH DATA")
    print("=" * 65)

    print("\nLoading raw dataset:")
    print(RAW_MATCH_FILE)

    if not RAW_MATCH_FILE.exists():
        raise FileNotFoundError(
            f"Raw match file not found: {RAW_MATCH_FILE}"
        )

    df = pd.read_csv(RAW_MATCH_FILE)

    print(f"\nRaw rows: {len(df):,}")
    print(f"Raw columns: {len(df.columns):,}")

    required_columns = [
        "Date",
        "Day",
        "season",
        "team",
        "team_opp",
        "score",
        "score_opp",
        "home",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the raw dataset: "
            f"{missing_columns}"
        )

    # ---------------------------------------------------------
    # ONE ROW PER MATCH
    # ---------------------------------------------------------
    #
    # The raw dataset contains both:
    #
    # Lyon vs Schalke
    # Schalke vs Lyon
    #
    # Keeping only home == 1 gives us exactly one row per match.
    # ---------------------------------------------------------

    matches = df.loc[
        df["home"] == 1,
        [
            "Date",
            "Day",
            "season",
            "team",
            "team_opp",
            "score",
            "score_opp",
        ],
    ].copy()

    # Rename columns into clear match-level terminology.
    matches = matches.rename(
        columns={
            "Date": "date",
            "Day": "day",
            "team": "home_team",
            "team_opp": "away_team",
            "score": "home_goals",
            "score_opp": "away_goals",
        }
    )

    # ---------------------------------------------------------
    # CLEAN TEXT ENCODING
    # ---------------------------------------------------------

    matches["home_team"] = matches["home_team"].map(
        fix_text_encoding
    )

    matches["away_team"] = matches["away_team"].map(
        fix_text_encoding
    )

    matches["day"] = matches["day"].map(
        fix_text_encoding
    )

    # ---------------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------------

    matches["date"] = pd.to_datetime(
        matches["date"],
        errors="raise",
    )

    matches["season"] = pd.to_numeric(
        matches["season"],
        errors="raise",
    ).astype(int)

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="raise",
    ).astype(int)

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="raise",
    ).astype(int)

    # ---------------------------------------------------------
    # BUILD TARGET
    # ---------------------------------------------------------
    #
    # We deliberately recalculate the result ourselves rather
    # than trusting the original Kaggle result column.
    #
    # H = Home win
    # D = Draw
    # A = Away win
    # ---------------------------------------------------------

    matches["result"] = matches.apply(
        lambda row: determine_result(
            row["home_goals"],
            row["away_goals"],
        ),
        axis=1,
    )

    # Human-readable scoreline.
    matches["scoreline"] = (
        matches["home_goals"].astype(str)
        + "-"
        + matches["away_goals"].astype(str)
    )

    # Metadata identifying the competition and data source.
    matches["competition"] = "UEFA Champions League"
    matches["source"] = "kaggle_ucl"

    # ---------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # ---------------------------------------------------------

    matches = matches.sort_values(
        by=[
            "date",
            "home_team",
            "away_team",
        ]
    ).reset_index(drop=True)

    # Give each match an internal project ID.
    matches.insert(
        0,
        "match_id",
        range(1, len(matches) + 1),
    )

    # ---------------------------------------------------------
    # QUALITY CHECKS
    # ---------------------------------------------------------

    print("\nRunning quality checks...")

    expected_matches = len(
        df.loc[df["home"] == 1]
    )

    assert len(matches) == expected_matches, (
        "Unexpected match count after filtering."
    )

    assert matches["date"].notna().all(), (
        "Missing dates detected."
    )

    assert matches["home_team"].notna().all(), (
        "Missing home team names detected."
    )

    assert matches["away_team"].notna().all(), (
        "Missing away team names detected."
    )

    assert matches["home_goals"].notna().all(), (
        "Missing home goal values detected."
    )

    assert matches["away_goals"].notna().all(), (
        "Missing away goal values detected."
    )

    assert matches["result"].isin(
        ["H", "D", "A"]
    ).all(), (
        "Unexpected result values detected."
    )

    assert (
        matches["home_goals"] >= 0
    ).all(), (
        "Negative home goals detected."
    )

    assert (
        matches["away_goals"] >= 0
    ).all(), (
        "Negative away goals detected."
    )

    duplicate_count = matches.duplicated(
        subset=[
            "date",
            "home_team",
            "away_team",
        ]
    ).sum()

    if duplicate_count:
        print(
            f"WARNING: {duplicate_count} possible duplicate "
            f"matches found."
        )
    else:
        print("No duplicate matches found.")

    # Check for remaining obvious encoding problems.
    encoding_problem_mask = (
        matches["home_team"].str.contains(
            r"Ã|Â|â",
            regex=True,
            na=False,
        )
        |
        matches["away_team"].str.contains(
            r"Ã|Â|â",
            regex=True,
            na=False,
        )
    )

    encoding_problem_count = int(
        encoding_problem_mask.sum()
    )

    if encoding_problem_count:
        print(
            f"WARNING: {encoding_problem_count} matches still "
            f"contain possible encoding problems."
        )
    else:
        print("No obvious team-name encoding problems found.")

    # ---------------------------------------------------------
    # SAVE DATASET
    # ---------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\nDataset successfully created.")

    print(f"\nRows: {len(matches):,}")
    print(f"Columns: {len(matches.columns):,}")

    print(
        f"Date range: "
        f"{matches['date'].min().date()} "
        f"to "
        f"{matches['date'].max().date()}"
    )

    unique_teams = set(
        matches["home_team"]
    ).union(
        set(matches["away_team"])
    )

    print(f"Teams: {len(unique_teams):,}")

    print("\nResult distribution:")
    print(
        matches["result"]
        .value_counts()
        .reindex(["H", "D", "A"])
    )

    print("\nResult percentages:")
    print(
        (
            matches["result"]
            .value_counts(normalize=True)
            .reindex(["H", "D", "A"])
            .mul(100)
            .round(2)
        )
    )

    print("\nEncoding check example:")
    atletico_names = sorted(
        {
            team
            for team in unique_teams
            if "atl" in team.lower()
        }
    )

    print(atletico_names)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\nFirst 10 matches:")
    print(
        matches.head(10).to_string(
            index=False
        )
    )

    print("\nLast 10 matches:")
    print(
        matches.tail(10).to_string(
            index=False
        )
    )

    return matches


if __name__ == "__main__":
    build_clean_match_dataset()