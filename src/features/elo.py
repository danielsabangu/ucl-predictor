from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import PROCESSED_DATA_DIR


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_elo.csv"
)

LATEST_ELO_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_elo_latest.csv"
)


# ============================================================
# ELO CONFIGURATION
# ============================================================

INITIAL_ELO = 1500.0

K_FACTOR = 24.0

HOME_ADVANTAGE = 65.0

ELO_SCALE = 400.0

# At the beginning of a new season, ratings are partially
# pulled back toward the average.
#
# 0.75 means:
# retain 75% of the previous season's rating difference
# from 1500.
SEASON_CARRYOVER = 0.75


# ============================================================
# DATA CLASS
# ============================================================


@dataclass
class EloMatchResult:
    """
    Container for the Elo calculation of one match.
    """

    home_elo_pre: float
    away_elo_pre: float

    home_elo_expectancy: float
    away_elo_expectancy: float

    home_elo_post: float
    away_elo_post: float


# ============================================================
# ELO MATH
# ============================================================


def expected_score(
    rating_a: float,
    rating_b: float,
) -> float:
    """
    Calculate Elo expected score for team A.

    The result is between 0 and 1.

    Important:
    This is NOT yet our final probability of:

    Home win
    Draw
    Away win

    It is simply the Elo expected performance score.
    """

    exponent = (
        rating_b - rating_a
    ) / ELO_SCALE

    return 1.0 / (
        1.0
        + 10.0 ** exponent
    )


def actual_score_from_result(
    result: str,
) -> tuple[float, float]:
    """
    Convert H/D/A into Elo scores.

    Home win:
        home = 1
        away = 0

    Draw:
        home = 0.5
        away = 0.5

    Away win:
        home = 0
        away = 1
    """

    if result == "H":
        return 1.0, 0.0

    if result == "D":
        return 0.5, 0.5

    if result == "A":
        return 0.0, 1.0

    raise ValueError(
        f"Invalid match result: {result}"
    )


def update_ratings(
    home_rating: float,
    away_rating: float,
    result: str,
) -> EloMatchResult:
    """
    Calculate Elo values for one match.

    IMPORTANT:

    The pre-match values are calculated first.

    Only after that do we use the result to calculate
    the post-match ratings.
    """

    # --------------------------------------------------------
    # HOME ADVANTAGE
    # --------------------------------------------------------
    #
    # The home team's rating receives a temporary bonus when
    # calculating expected performance.
    #
    # We do NOT permanently add 65 points to the team's Elo.
    # --------------------------------------------------------

    adjusted_home_rating = (
        home_rating
        + HOME_ADVANTAGE
    )

    home_expectancy = expected_score(
        adjusted_home_rating,
        away_rating,
    )

    away_expectancy = (
        1.0 - home_expectancy
    )

    # --------------------------------------------------------
    # ACTUAL RESULT
    # --------------------------------------------------------

    (
        home_actual,
        away_actual,
    ) = actual_score_from_result(
        result
    )

    # --------------------------------------------------------
    # ELO CHANGE
    # --------------------------------------------------------

    home_change = (
        K_FACTOR
        * (
            home_actual
            - home_expectancy
        )
    )

    away_change = (
        K_FACTOR
        * (
            away_actual
            - away_expectancy
        )
    )

    home_rating_post = (
        home_rating
        + home_change
    )

    away_rating_post = (
        away_rating
        + away_change
    )

    return EloMatchResult(
        home_elo_pre=home_rating,
        away_elo_pre=away_rating,
        home_elo_expectancy=home_expectancy,
        away_elo_expectancy=away_expectancy,
        home_elo_post=home_rating_post,
        away_elo_post=away_rating_post,
    )


# ============================================================
# SEASON REGRESSION
# ============================================================


def regress_ratings_to_mean(
    ratings: dict[str, float],
) -> dict[str, float]:
    """
    Partially regress all existing Elo ratings toward 1500.

    This happens at the beginning of each new season.

    Why?

    Football teams change between seasons:

    - transfers
    - managers
    - injuries
    - promoted younger players
    - ageing players
    - tactical changes

    A club should therefore not carry 100% of its previous
    season rating forever.

    Later our squad-strength and transfer features will
    capture these changes more directly.
    """

    regressed = {}

    for team, rating in ratings.items():

        regressed_rating = (
            INITIAL_ELO
            + (
                rating
                - INITIAL_ELO
            )
            * SEASON_CARRYOVER
        )

        regressed[
            team
        ] = regressed_rating

    return regressed


# ============================================================
# BUILD ELO HISTORY
# ============================================================


def build_elo_features(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Build leakage-safe Elo features for every UCL match.

    Each row receives Elo values from immediately BEFORE
    that match.

    The result of that match is only used afterwards
    to update the ratings.
    """

    df = matches.copy()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    required_columns = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "result",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Elo input dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # DATE / SORTING
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df = df.sort_values(
        by=[
            "date",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # TEAM RATINGS
    # --------------------------------------------------------

    ratings: dict[str, float] = {}

    current_season = None

    # --------------------------------------------------------
    # FEATURE STORAGE
    # --------------------------------------------------------

    home_elo_pre_values = []
    away_elo_pre_values = []

    elo_difference_values = []
    adjusted_elo_difference_values = []

    home_expectancy_values = []
    away_expectancy_values = []

    home_elo_post_values = []
    away_elo_post_values = []

    # --------------------------------------------------------
    # PROCESS MATCHES CHRONOLOGICALLY
    # --------------------------------------------------------

    for row in df.itertuples(
        index=False
    ):

        season = int(
            row.season
        )

        home_team = row.home_team
        away_team = row.away_team

        # ----------------------------------------------------
        # NEW SEASON
        # ----------------------------------------------------

        if current_season is None:

            current_season = season

        elif season != current_season:

            ratings = regress_ratings_to_mean(
                ratings
            )

            current_season = season

        # ----------------------------------------------------
        # NEW TEAMS
        # ----------------------------------------------------

        if home_team not in ratings:

            ratings[
                home_team
            ] = INITIAL_ELO

        if away_team not in ratings:

            ratings[
                away_team
            ] = INITIAL_ELO

        home_rating = ratings[
            home_team
        ]

        away_rating = ratings[
            away_team
        ]

        # ----------------------------------------------------
        # PRE-MATCH DIFFERENCE
        # ----------------------------------------------------

        elo_difference = (
            home_rating
            - away_rating
        )

        adjusted_elo_difference = (
            home_rating
            + HOME_ADVANTAGE
            - away_rating
        )

        # ----------------------------------------------------
        # MATCH CALCULATION
        # ----------------------------------------------------

        elo_result = update_ratings(
            home_rating=home_rating,
            away_rating=away_rating,
            result=row.result,
        )

        # ----------------------------------------------------
        # SAVE PRE-MATCH FEATURES
        # ----------------------------------------------------

        home_elo_pre_values.append(
            elo_result.home_elo_pre
        )

        away_elo_pre_values.append(
            elo_result.away_elo_pre
        )

        elo_difference_values.append(
            elo_difference
        )

        adjusted_elo_difference_values.append(
            adjusted_elo_difference
        )

        home_expectancy_values.append(
            elo_result.home_elo_expectancy
        )

        away_expectancy_values.append(
            elo_result.away_elo_expectancy
        )

        # ----------------------------------------------------
        # SAVE POST-MATCH VALUES FOR AUDITING
        # ----------------------------------------------------
        #
        # These should NOT be used as features for predicting
        # the same match.
        # ----------------------------------------------------

        home_elo_post_values.append(
            elo_result.home_elo_post
        )

        away_elo_post_values.append(
            elo_result.away_elo_post
        )

        # ----------------------------------------------------
        # UPDATE RATINGS
        # ----------------------------------------------------
        #
        # This happens only AFTER storing the pre-match values.
        # ----------------------------------------------------

        ratings[
            home_team
        ] = elo_result.home_elo_post

        ratings[
            away_team
        ] = elo_result.away_elo_post

    # --------------------------------------------------------
    # ADD FEATURES
    # --------------------------------------------------------

    df[
        "home_elo_pre"
    ] = home_elo_pre_values

    df[
        "away_elo_pre"
    ] = away_elo_pre_values

    df[
        "elo_diff_pre"
    ] = elo_difference_values

    df[
        "elo_diff_home_adjusted"
    ] = adjusted_elo_difference_values

    df[
        "home_elo_expectancy"
    ] = home_expectancy_values

    df[
        "away_elo_expectancy"
    ] = away_expectancy_values

    df[
        "home_elo_post"
    ] = home_elo_post_values

    df[
        "away_elo_post"
    ] = away_elo_post_values

    return df, ratings


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_elo_quality_checks(
    dataframe: pd.DataFrame,
) -> None:
    """
    Check the generated Elo history.
    """

    print("\n" + "=" * 75)
    print("ELO DATA QUALITY CHECK")
    print("=" * 75)

    expected_rows = 2089

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    if len(dataframe) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows:,} rows "
            f"but received {len(dataframe):,}."
        )

    print(
        f"\nMatch rows: "
        f"{len(dataframe):,} ✓"
    )

    # --------------------------------------------------------
    # REQUIRED ELO COLUMNS
    # --------------------------------------------------------

    elo_columns = [
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff_pre",
        "elo_diff_home_adjusted",
        "home_elo_expectancy",
        "away_elo_expectancy",
        "home_elo_post",
        "away_elo_post",
    ]

    missing_values = (
        dataframe[
            elo_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values != 0:

        raise RuntimeError(
            f"Elo columns contain "
            f"{missing_values} missing values."
        )

    print(
        "Missing Elo values: 0 ✓"
    )

    # --------------------------------------------------------
    # EXPECTANCY RANGE
    # --------------------------------------------------------

    invalid_home_probability = (
        (
            dataframe[
                "home_elo_expectancy"
            ] < 0
        )
        |
        (
            dataframe[
                "home_elo_expectancy"
            ] > 1
        )
    ).sum()

    invalid_away_probability = (
        (
            dataframe[
                "away_elo_expectancy"
            ] < 0
        )
        |
        (
            dataframe[
                "away_elo_expectancy"
            ] > 1
        )
    ).sum()

    if (
        invalid_home_probability > 0
        or invalid_away_probability > 0
    ):

        raise RuntimeError(
            "Invalid Elo expectancy values detected."
        )

    print(
        "Elo expectancy range: valid ✓"
    )

    # --------------------------------------------------------
    # EXPECTANCY SUM
    # --------------------------------------------------------

    expectancy_sum = (
        dataframe[
            "home_elo_expectancy"
        ]
        +
        dataframe[
            "away_elo_expectancy"
        ]
    )

    maximum_error = (
        expectancy_sum
        - 1.0
    ).abs().max()

    if maximum_error > 1e-10:

        raise RuntimeError(
            "Home and away Elo expectancy "
            "do not sum to 1."
        )

    print(
        "Home + away expectancy = 1 ✓"
    )

    # --------------------------------------------------------
    # FIRST MATCH
    # --------------------------------------------------------

    first_match = dataframe.iloc[
        0
    ]

    if (
        abs(
            first_match[
                "home_elo_pre"
            ]
            - INITIAL_ELO
        )
        > 1e-10
    ):

        raise RuntimeError(
            "First home team's Elo should start at 1500."
        )

    if (
        abs(
            first_match[
                "away_elo_pre"
            ]
            - INITIAL_ELO
        )
        > 1e-10
    ):

        raise RuntimeError(
            "First away team's Elo should start at 1500."
        )

    print(
        "Initial Elo = 1500 ✓"
    )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------
    #
    # The first appearance of any club must have a pre-match
    # rating of 1500.
    # --------------------------------------------------------

    seen_teams = set()

    first_appearance_errors = []

    for row in dataframe.itertuples(
        index=False
    ):

        if row.home_team not in seen_teams:

            if (
                abs(
                    row.home_elo_pre
                    - INITIAL_ELO
                )
                > 1e-10
            ):

                first_appearance_errors.append(
                    row.home_team
                )

        if row.away_team not in seen_teams:

            if (
                abs(
                    row.away_elo_pre
                    - INITIAL_ELO
                )
                > 1e-10
            ):

                first_appearance_errors.append(
                    row.away_team
                )

        seen_teams.add(
            row.home_team
        )

        seen_teams.add(
            row.away_team
        )

    if first_appearance_errors:

        raise RuntimeError(
            "Unexpected Elo value on first team appearance: "
            f"{sorted(set(first_appearance_errors))}"
        )

    print(
        "First-appearance leakage check: passed ✓"
    )


# ============================================================
# LATEST RATINGS
# ============================================================


def build_latest_elo_table(
    ratings: dict[str, float],
) -> pd.DataFrame:
    """
    Convert the latest rating dictionary into a sorted table.

    This table will eventually help predict future fixtures.
    """

    rows = []

    for team, rating in ratings.items():

        rows.append(
            {
                "team": team,
                "ucl_elo": round(
                    rating,
                    2,
                ),
            }
        )

    latest = pd.DataFrame(
        rows
    )

    latest = latest.sort_values(
        by="ucl_elo",
        ascending=False,
    ).reset_index(
        drop=True
    )

    latest.insert(
        0,
        "rank",
        range(
            1,
            len(latest) + 1,
        ),
    )

    return latest


# ============================================================
# MAIN PIPELINE
# ============================================================


def generate_elo_dataset() -> pd.DataFrame:
    """
    Generate leakage-safe UCL Elo features.
    """

    print("=" * 75)
    print("UCL PREDICTOR - ELO FEATURE ENGINEERING")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD MASTER MATCHES
    # --------------------------------------------------------

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Master match dataset not found:\n"
            f"{INPUT_FILE}"
        )

    matches = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Matches loaded: "
        f"{len(matches):,}"
    )

    # --------------------------------------------------------
    # GENERATE FEATURES
    # --------------------------------------------------------

    elo_history, latest_ratings = (
        build_elo_features(
            matches
        )
    )

    # --------------------------------------------------------
    # QUALITY CHECKS
    # --------------------------------------------------------

    run_elo_quality_checks(
        elo_history
    )

    # --------------------------------------------------------
    # LATEST RATINGS
    # --------------------------------------------------------

    latest_table = build_latest_elo_table(
        latest_ratings
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    elo_history.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    latest_table.to_csv(
        LATEST_ELO_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("ELO SUMMARY")
    print("=" * 75)

    print(
        f"\nMatches processed: "
        f"{len(elo_history):,}"
    )

    print(
        f"Teams rated: "
        f"{len(latest_table):,}"
    )

    print(
        f"Initial Elo: "
        f"{INITIAL_ELO:.0f}"
    )

    print(
        f"K-factor: "
        f"{K_FACTOR:.0f}"
    )

    print(
        f"Home advantage: "
        f"{HOME_ADVANTAGE:.0f} Elo points"
    )

    print(
        f"Season carryover: "
        f"{SEASON_CARRYOVER:.2f}"
    )

    print("\nTop 15 latest UCL Elo ratings:")

    print(
        latest_table.head(
            15
        ).to_string(
            index=False
        )
    )

    print("\nElo history saved to:")

    print(
        OUTPUT_FILE
    )

    print("\nLatest Elo table saved to:")

    print(
        LATEST_ELO_FILE
    )

    print("\nFirst 5 match Elo values:")

    display_columns = [
        "date",
        "home_team",
        "away_team",
        "result",
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff_pre",
        "home_elo_expectancy",
        "home_elo_post",
        "away_elo_post",
    ]

    print(
        elo_history[
            display_columns
        ]
        .head(
            5
        )
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print("\nLast 5 match Elo values:")

    print(
        elo_history[
            display_columns
        ]
        .tail(
            5
        )
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)

    print(
        "ELO FEATURES CREATED SUCCESSFULLY"
    )

    print("=" * 75)

    return elo_history


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    generate_elo_dataset()