from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from src.config import PROCESSED_DATA_DIR


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_elo.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_form.csv"
)

LATEST_FORM_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_form_latest.csv"
)


# ============================================================
# FORM WINDOWS
# ============================================================

FORM_WINDOWS = [
    5,
    10,
]


# ============================================================
# MATCH HELPERS
# ============================================================


def points_from_result(
    result: str,
    perspective: str,
) -> int:
    """
    Return points earned by a team.

    perspective:
        "home"
        "away"
    """

    if perspective == "home":

        if result == "H":
            return 3

        if result == "D":
            return 1

        if result == "A":
            return 0

    if perspective == "away":

        if result == "A":
            return 3

        if result == "D":
            return 1

        if result == "H":
            return 0

    raise ValueError(
        f"Invalid result/perspective: "
        f"{result}, {perspective}"
    )


def result_flags(
    points: int,
) -> tuple[int, int, int]:
    """
    Convert points into win/draw/loss flags.
    """

    if points == 3:
        return 1, 0, 0

    if points == 1:
        return 0, 1, 0

    return 0, 0, 1


# ============================================================
# HISTORY RECORD
# ============================================================


def build_team_record(
    *,
    date: pd.Timestamp,
    season: int,
    venue: str,
    goals_for: int,
    goals_against: int,
    points: int,
) -> dict[str, Any]:
    """
    Build one historical team-performance record.
    """

    win, draw, loss = result_flags(
        points
    )

    return {
        "date": date,
        "season": season,
        "venue": venue,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": (
            goals_for
            - goals_against
        ),
        "points": points,
        "win": win,
        "draw": draw,
        "loss": loss,
        "clean_sheet": (
            1
            if goals_against == 0
            else 0
        ),
        "scored": (
            1
            if goals_for > 0
            else 0
        ),
    }


# ============================================================
# SUMMARY FUNCTIONS
# ============================================================


def safe_average(
    values: list[float],
) -> float:
    """
    Return NaN for an empty history.

    We intentionally preserve missing history instead of
    pretending a brand-new club has good or bad form.

    Later our ML preprocessing pipeline will handle these
    missing values properly.
    """

    if not values:
        return float("nan")

    return sum(values) / len(values)


def weighted_points_per_game(
    records: list[dict[str, Any]],
) -> float:
    """
    Calculate a recency-weighted PPG score.

    More recent matches receive more weight.

    Example with five matches:

    oldest -> weight 1
              weight 2
              weight 3
              weight 4
    newest -> weight 5

    Output remains between 0 and 3.
    """

    if not records:
        return float("nan")

    weights = list(
        range(
            1,
            len(records) + 1,
        )
    )

    weighted_points = sum(
        record["points"] * weight
        for record, weight
        in zip(records, weights)
    )

    total_weight = sum(
        weights
    )

    return (
        weighted_points
        / total_weight
    )


def summarize_records(
    records: list[dict[str, Any]],
    window: int | None = None,
) -> dict[str, float]:
    """
    Summarize historical matches.

    If window is provided, only the most recent N matches
    are considered.
    """

    if window is not None:

        records = records[
            -window:
        ]

    count = len(
        records
    )

    if count == 0:

        return {
            "matches": 0,
            "ppg": float("nan"),
            "win_rate": float("nan"),
            "draw_rate": float("nan"),
            "loss_rate": float("nan"),
            "gf_pg": float("nan"),
            "ga_pg": float("nan"),
            "gd_pg": float("nan"),
            "clean_sheet_rate": float("nan"),
            "scoring_rate": float("nan"),
        }

    points = [
        record["points"]
        for record in records
    ]

    wins = [
        record["win"]
        for record in records
    ]

    draws = [
        record["draw"]
        for record in records
    ]

    losses = [
        record["loss"]
        for record in records
    ]

    goals_for = [
        record["goals_for"]
        for record in records
    ]

    goals_against = [
        record["goals_against"]
        for record in records
    ]

    goal_difference = [
        record["goal_difference"]
        for record in records
    ]

    clean_sheets = [
        record["clean_sheet"]
        for record in records
    ]

    scoring = [
        record["scored"]
        for record in records
    ]

    return {
        "matches": count,
        "ppg": safe_average(
            points
        ),
        "win_rate": safe_average(
            wins
        ),
        "draw_rate": safe_average(
            draws
        ),
        "loss_rate": safe_average(
            losses
        ),
        "gf_pg": safe_average(
            goals_for
        ),
        "ga_pg": safe_average(
            goals_against
        ),
        "gd_pg": safe_average(
            goal_difference
        ),
        "clean_sheet_rate": safe_average(
            clean_sheets
        ),
        "scoring_rate": safe_average(
            scoring
        ),
    }


def venue_history(
    records: list[dict[str, Any]],
    venue: str,
) -> list[dict[str, Any]]:
    """
    Return only home or away history.
    """

    return [
        record
        for record in records
        if record["venue"] == venue
    ]


def season_history(
    records: list[dict[str, Any]],
    season: int,
) -> list[dict[str, Any]]:
    """
    Return matches already played in the current UCL season.
    """

    return [
        record
        for record in records
        if record["season"] == season
    ]


# ============================================================
# TEAM PRE-MATCH FEATURES
# ============================================================


def build_team_pre_match_features(
    *,
    team: str,
    season: int,
    match_date: pd.Timestamp,
    history: dict[str, list[dict[str, Any]]],
    last_match_dates: dict[str, pd.Timestamp],
    prefix: str,
    current_venue: str,
) -> dict[str, float]:
    """
    Build all historical features for one team BEFORE
    the current match is added to its history.

    This is the key leakage-prevention rule.
    """

    records = history[
        team
    ]

    features = {}

    # --------------------------------------------------------
    # TOTAL UCL EXPERIENCE
    # --------------------------------------------------------

    features[
        f"{prefix}_ucl_matches_before"
    ] = len(
        records
    )

    # --------------------------------------------------------
    # DAYS SINCE PREVIOUS UCL MATCH
    # --------------------------------------------------------

    if team in last_match_dates:

        days_since = (
            match_date
            - last_match_dates[team]
        ).days

        features[
            f"{prefix}_days_since_last_ucl"
        ] = float(
            days_since
        )

    else:

        features[
            f"{prefix}_days_since_last_ucl"
        ] = float("nan")

    # --------------------------------------------------------
    # LAST 5 / LAST 10
    # --------------------------------------------------------

    for window in FORM_WINDOWS:

        summary = summarize_records(
            records,
            window=window,
        )

        features[
            f"{prefix}_last{window}_matches"
        ] = summary["matches"]

        features[
            f"{prefix}_last{window}_ppg"
        ] = summary["ppg"]

        features[
            f"{prefix}_last{window}_win_rate"
        ] = summary["win_rate"]

        features[
            f"{prefix}_last{window}_draw_rate"
        ] = summary["draw_rate"]

        features[
            f"{prefix}_last{window}_loss_rate"
        ] = summary["loss_rate"]

        features[
            f"{prefix}_last{window}_gf_pg"
        ] = summary["gf_pg"]

        features[
            f"{prefix}_last{window}_ga_pg"
        ] = summary["ga_pg"]

        features[
            f"{prefix}_last{window}_gd_pg"
        ] = summary["gd_pg"]

        features[
            f"{prefix}_last{window}_clean_sheet_rate"
        ] = summary[
            "clean_sheet_rate"
        ]

        features[
            f"{prefix}_last{window}_scoring_rate"
        ] = summary[
            "scoring_rate"
        ]

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    recent_five = records[
        -5:
    ]

    features[
        f"{prefix}_momentum_last5"
    ] = weighted_points_per_game(
        recent_five
    )

    # --------------------------------------------------------
    # CURRENT-SEASON FORM
    # --------------------------------------------------------

    current_season_records = season_history(
        records,
        season,
    )

    season_summary = summarize_records(
        current_season_records
    )

    features[
        f"{prefix}_season_matches_before"
    ] = season_summary[
        "matches"
    ]

    features[
        f"{prefix}_season_ppg_pre"
    ] = season_summary[
        "ppg"
    ]

    features[
        f"{prefix}_season_win_rate_pre"
    ] = season_summary[
        "win_rate"
    ]

    features[
        f"{prefix}_season_gf_pg_pre"
    ] = season_summary[
        "gf_pg"
    ]

    features[
        f"{prefix}_season_ga_pg_pre"
    ] = season_summary[
        "ga_pg"
    ]

    features[
        f"{prefix}_season_gd_pg_pre"
    ] = season_summary[
        "gd_pg"
    ]

    # --------------------------------------------------------
    # VENUE-SPECIFIC FORM
    # --------------------------------------------------------

    venue_records = venue_history(
        records,
        current_venue,
    )

    venue_summary = summarize_records(
        venue_records,
        window=5,
    )

    features[
        f"{prefix}_{current_venue}_last5_matches"
    ] = venue_summary[
        "matches"
    ]

    features[
        f"{prefix}_{current_venue}_last5_ppg"
    ] = venue_summary[
        "ppg"
    ]

    features[
        f"{prefix}_{current_venue}_last5_win_rate"
    ] = venue_summary[
        "win_rate"
    ]

    features[
        f"{prefix}_{current_venue}_last5_gf_pg"
    ] = venue_summary[
        "gf_pg"
    ]

    features[
        f"{prefix}_{current_venue}_last5_ga_pg"
    ] = venue_summary[
        "ga_pg"
    ]

    features[
        f"{prefix}_{current_venue}_last5_gd_pg"
    ] = venue_summary[
        "gd_pg"
    ]

    return features


# ============================================================
# DIFFERENCE FEATURES
# ============================================================


def safe_difference(
    first: float,
    second: float,
) -> float:
    """
    Calculate a difference while preserving missing values.
    """

    if pd.isna(first) or pd.isna(second):

        return float("nan")

    return float(
        first - second
    )


def add_difference_features(
    row_features: dict[str, float],
) -> dict[str, float]:
    """
    Build home-minus-away comparison features.
    """

    row_features[
        "form_ppg_diff_last5"
    ] = safe_difference(
        row_features[
            "home_last5_ppg"
        ],
        row_features[
            "away_last5_ppg"
        ],
    )

    row_features[
        "form_ppg_diff_last10"
    ] = safe_difference(
        row_features[
            "home_last10_ppg"
        ],
        row_features[
            "away_last10_ppg"
        ],
    )

    row_features[
        "form_gd_diff_last5"
    ] = safe_difference(
        row_features[
            "home_last5_gd_pg"
        ],
        row_features[
            "away_last5_gd_pg"
        ],
    )

    row_features[
        "form_gd_diff_last10"
    ] = safe_difference(
        row_features[
            "home_last10_gd_pg"
        ],
        row_features[
            "away_last10_gd_pg"
        ],
    )

    row_features[
        "form_momentum_diff_last5"
    ] = safe_difference(
        row_features[
            "home_momentum_last5"
        ],
        row_features[
            "away_momentum_last5"
        ],
    )

    row_features[
        "season_ppg_diff_pre"
    ] = safe_difference(
        row_features[
            "home_season_ppg_pre"
        ],
        row_features[
            "away_season_ppg_pre"
        ],
    )

    row_features[
        "season_gd_diff_pre"
    ] = safe_difference(
        row_features[
            "home_season_gd_pg_pre"
        ],
        row_features[
            "away_season_gd_pg_pre"
        ],
    )

    row_features[
        "venue_ppg_diff_last5"
    ] = safe_difference(
        row_features[
            "home_home_last5_ppg"
        ],
        row_features[
            "away_away_last5_ppg"
        ],
    )

    row_features[
        "venue_gd_diff_last5"
    ] = safe_difference(
        row_features[
            "home_home_last5_gd_pg"
        ],
        row_features[
            "away_away_last5_gd_pg"
        ],
    )

    row_features[
        "ucl_experience_diff"
    ] = (
        row_features[
            "home_ucl_matches_before"
        ]
        -
        row_features[
            "away_ucl_matches_before"
        ]
    )

    row_features[
        "rest_days_diff"
    ] = safe_difference(
        row_features[
            "home_days_since_last_ucl"
        ],
        row_features[
            "away_days_since_last_ucl"
        ],
    )

    return row_features


# ============================================================
# BUILD FORM FEATURES
# ============================================================


def build_form_features(
    matches: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, list[dict[str, Any]]],
]:
    """
    Generate leakage-safe rolling UCL form features.

    Sequence:

    1. Read current match
    2. Calculate features from previous matches
    3. Store those pre-match features
    4. Read current result
    5. Add current match to history

    The current match therefore cannot influence its own
    prediction features.
    """

    df = matches.copy()

    required_columns = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Team-form input is missing columns: "
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
    # HISTORICAL STATE
    # --------------------------------------------------------

    history: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    last_match_dates: dict[
        str,
        pd.Timestamp,
    ] = {}

    feature_rows = []

    # --------------------------------------------------------
    # PROCESS CHRONOLOGICALLY
    # --------------------------------------------------------

    for row in df.itertuples(
        index=False
    ):

        match_date = pd.Timestamp(
            row.date
        )

        season = int(
            row.season
        )

        home_team = row.home_team
        away_team = row.away_team

        # ----------------------------------------------------
        # PRE-MATCH HOME FEATURES
        # ----------------------------------------------------

        home_features = (
            build_team_pre_match_features(
                team=home_team,
                season=season,
                match_date=match_date,
                history=history,
                last_match_dates=last_match_dates,
                prefix="home",
                current_venue="home",
            )
        )

        # ----------------------------------------------------
        # PRE-MATCH AWAY FEATURES
        # ----------------------------------------------------

        away_features = (
            build_team_pre_match_features(
                team=away_team,
                season=season,
                match_date=match_date,
                history=history,
                last_match_dates=last_match_dates,
                prefix="away",
                current_venue="away",
            )
        )

        row_features = {
            **home_features,
            **away_features,
        }

        row_features = (
            add_difference_features(
                row_features
            )
        )

        feature_rows.append(
            row_features
        )

        # ----------------------------------------------------
        # CURRENT MATCH RESULT
        # ----------------------------------------------------
        #
        # Only NOW do we look at this match's result to update
        # future history.
        # ----------------------------------------------------

        home_points = points_from_result(
            row.result,
            "home",
        )

        away_points = points_from_result(
            row.result,
            "away",
        )

        home_record = build_team_record(
            date=match_date,
            season=season,
            venue="home",
            goals_for=int(
                row.home_goals
            ),
            goals_against=int(
                row.away_goals
            ),
            points=home_points,
        )

        away_record = build_team_record(
            date=match_date,
            season=season,
            venue="away",
            goals_for=int(
                row.away_goals
            ),
            goals_against=int(
                row.home_goals
            ),
            points=away_points,
        )

        # ----------------------------------------------------
        # UPDATE HISTORY AFTER FEATURES HAVE BEEN STORED
        # ----------------------------------------------------

        history[
            home_team
        ].append(
            home_record
        )

        history[
            away_team
        ].append(
            away_record
        )

        last_match_dates[
            home_team
        ] = match_date

        last_match_dates[
            away_team
        ] = match_date

    # --------------------------------------------------------
    # ATTACH FEATURES
    # --------------------------------------------------------

    feature_dataframe = pd.DataFrame(
        feature_rows
    )

    df = pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            feature_dataframe.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return df, history


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_form_quality_checks(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate rolling features.
    """

    print("\n" + "=" * 75)
    print("TEAM FORM DATA QUALITY CHECK")
    print("=" * 75)

    expected_rows = 2089

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    if len(dataframe) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows:,} rows, "
            f"received {len(dataframe):,}."
        )

    print(
        f"\nMatch rows: "
        f"{len(dataframe):,} ✓"
    )

    # --------------------------------------------------------
    # WINDOW COUNTS
    # --------------------------------------------------------

    checks = {
        "home_last5_matches": 5,
        "away_last5_matches": 5,
        "home_last10_matches": 10,
        "away_last10_matches": 10,
        "home_home_last5_matches": 5,
        "away_away_last5_matches": 5,
    }

    for column, maximum in checks.items():

        invalid = (
            (
                dataframe[
                    column
                ] < 0
            )
            |
            (
                dataframe[
                    column
                ] > maximum
            )
        ).sum()

        if invalid > 0:

            raise RuntimeError(
                f"Invalid rolling match counts "
                f"in {column}."
            )

    print(
        "Rolling window counts: valid ✓"
    )

    # --------------------------------------------------------
    # PPG RANGE
    # --------------------------------------------------------

    ppg_columns = [
        "home_last5_ppg",
        "away_last5_ppg",
        "home_last10_ppg",
        "away_last10_ppg",
        "home_momentum_last5",
        "away_momentum_last5",
        "home_season_ppg_pre",
        "away_season_ppg_pre",
        "home_home_last5_ppg",
        "away_away_last5_ppg",
    ]

    for column in ppg_columns:

        values = dataframe[
            column
        ].dropna()

        invalid = (
            (
                values < 0
            )
            |
            (
                values > 3
            )
        ).sum()

        if invalid > 0:

            raise RuntimeError(
                f"Invalid PPG values "
                f"in {column}."
            )

    print(
        "PPG ranges: valid ✓"
    )

    # --------------------------------------------------------
    # RATE RANGE
    # --------------------------------------------------------

    rate_columns = [
        column
        for column in dataframe.columns
        if (
            column.endswith(
                "_rate"
            )
            or column.endswith(
                "_rate_pre"
            )
        )
    ]

    for column in rate_columns:

        values = dataframe[
            column
        ].dropna()

        invalid = (
            (
                values < 0
            )
            |
            (
                values > 1
            )
        ).sum()

        if invalid > 0:

            raise RuntimeError(
                f"Invalid rate values "
                f"in {column}."
            )

    print(
        "Win/draw/loss/rate ranges: valid ✓"
    )

    # --------------------------------------------------------
    # REST DAYS
    # --------------------------------------------------------

    for column in [
        "home_days_since_last_ucl",
        "away_days_since_last_ucl",
    ]:

        invalid = (
            dataframe[
                column
            ]
            .dropna()
            .lt(
                0
            )
            .sum()
        )

        if invalid > 0:

            raise RuntimeError(
                f"Negative rest days detected "
                f"in {column}."
            )

    print(
        "Days-since-last-match values: valid ✓"
    )

    # --------------------------------------------------------
    # FIRST APPEARANCE LEAKAGE TEST
    # --------------------------------------------------------

    seen_teams = set()

    leakage_errors = []

    for row in dataframe.itertuples(
        index=False
    ):

        if row.home_team not in seen_teams:

            if (
                row.home_ucl_matches_before
                != 0
            ):

                leakage_errors.append(
                    row.home_team
                )

            if (
                row.home_last5_matches
                != 0
            ):

                leakage_errors.append(
                    row.home_team
                )

        if row.away_team not in seen_teams:

            if (
                row.away_ucl_matches_before
                != 0
            ):

                leakage_errors.append(
                    row.away_team
                )

            if (
                row.away_last5_matches
                != 0
            ):

                leakage_errors.append(
                    row.away_team
                )

        seen_teams.add(
            row.home_team
        )

        seen_teams.add(
            row.away_team
        )

    if leakage_errors:

        raise RuntimeError(
            "First-appearance form leakage detected for: "
            f"{sorted(set(leakage_errors))}"
        )

    print(
        "First-appearance leakage check: passed ✓"
    )

    # --------------------------------------------------------
    # FIRST MATCH OF DATASET
    # --------------------------------------------------------

    first_match = dataframe.iloc[
        0
    ]

    if (
        first_match[
            "home_ucl_matches_before"
        ]
        != 0
    ):

        raise RuntimeError(
            "First match home team incorrectly has history."
        )

    if (
        first_match[
            "away_ucl_matches_before"
        ]
        != 0
    ):

        raise RuntimeError(
            "First match away team incorrectly has history."
        )

    print(
        "Initial-history check: passed ✓"
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------
    #
    # Missing rolling metrics are EXPECTED for new teams.
    #
    # We deliberately do not fill them here because doing so
    # with full-dataset statistics could create leakage.
    # --------------------------------------------------------

    form_columns = [
        column
        for column in dataframe.columns
        if (
            "last5" in column
            or "last10" in column
            or "season_" in column
            or "days_since" in column
        )
    ]

    missing_form_values = (
        dataframe[
            form_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Expected cold-start missing values: "
        f"{missing_form_values:,}"
    )

    print(
        "Cold-start NaNs preserved intentionally ✓"
    )


# ============================================================
# LATEST TEAM FORM TABLE
# ============================================================


def build_latest_form_table(
    history: dict[
        str,
        list[dict[str, Any]],
    ],
) -> pd.DataFrame:
    """
    Create latest UCL form for every team.

    This will later help generate future match predictions.
    """

    rows = []

    for team, records in history.items():

        last5 = summarize_records(
            records,
            window=5,
        )

        last10 = summarize_records(
            records,
            window=10,
        )

        home_last5 = summarize_records(
            venue_history(
                records,
                "home",
            ),
            window=5,
        )

        away_last5 = summarize_records(
            venue_history(
                records,
                "away",
            ),
            window=5,
        )

        last_match_date = max(
            record["date"]
            for record in records
        )

        rows.append(
            {
                "team": team,
                "ucl_matches": len(
                    records
                ),
                "last_match_date": (
                    last_match_date.date()
                ),
                "last5_ppg": last5[
                    "ppg"
                ],
                "last5_win_rate": last5[
                    "win_rate"
                ],
                "last5_gf_pg": last5[
                    "gf_pg"
                ],
                "last5_ga_pg": last5[
                    "ga_pg"
                ],
                "last5_gd_pg": last5[
                    "gd_pg"
                ],
                "last5_clean_sheet_rate": last5[
                    "clean_sheet_rate"
                ],
                "last10_ppg": last10[
                    "ppg"
                ],
                "last10_gf_pg": last10[
                    "gf_pg"
                ],
                "last10_ga_pg": last10[
                    "ga_pg"
                ],
                "home_last5_ppg": home_last5[
                    "ppg"
                ],
                "away_last5_ppg": away_last5[
                    "ppg"
                ],
                "momentum_last5": (
                    weighted_points_per_game(
                        records[
                            -5:
                        ]
                    )
                ),
            }
        )

    latest = pd.DataFrame(
        rows
    )

    latest = latest.sort_values(
        by=[
            "last5_ppg",
            "last5_gd_pg",
            "ucl_matches",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    return latest


# ============================================================
# MAIN PIPELINE
# ============================================================


def generate_team_form_dataset() -> pd.DataFrame:
    """
    Generate all leakage-safe rolling UCL form features.
    """

    print("=" * 75)
    print("UCL PREDICTOR - TEAM FORM FEATURE ENGINEERING")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Elo dataset not found:\n"
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
    # BUILD
    # --------------------------------------------------------

    form_history, team_history = (
        build_form_features(
            matches
        )
    )

    # --------------------------------------------------------
    # QUALITY CHECK
    # --------------------------------------------------------

    run_form_quality_checks(
        form_history
    )

    # --------------------------------------------------------
    # LATEST TABLE
    # --------------------------------------------------------

    latest_form = (
        build_latest_form_table(
            team_history
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    form_history.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    latest_form.to_csv(
        LATEST_FORM_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TEAM FORM SUMMARY")
    print("=" * 75)

    print(
        f"\nMatches processed: "
        f"{len(form_history):,}"
    )

    print(
        f"Teams tracked: "
        f"{len(team_history):,}"
    )

    form_feature_columns = [
        column
        for column in form_history.columns
        if column not in matches.columns
    ]

    print(
        f"New form features created: "
        f"{len(form_feature_columns):,}"
    )

    print("\nTop 15 teams by latest Last-5 UCL PPG:")

    latest_display = latest_form[
        [
            "team",
            "ucl_matches",
            "last5_ppg",
            "last5_win_rate",
            "last5_gf_pg",
            "last5_ga_pg",
            "last5_gd_pg",
            "momentum_last5",
        ]
    ].head(
        15
    )

    print(
        latest_display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print("\nForm history saved to:")

    print(
        OUTPUT_FILE
    )

    print("\nLatest form table saved to:")

    print(
        LATEST_FORM_FILE
    )

    # --------------------------------------------------------
    # EXAMPLE MATCHES
    # --------------------------------------------------------

    display_columns = [
        "date",
        "home_team",
        "away_team",
        "result",
        "home_elo_pre",
        "away_elo_pre",
        "home_last5_ppg",
        "away_last5_ppg",
        "home_last5_gd_pg",
        "away_last5_gd_pg",
        "home_momentum_last5",
        "away_momentum_last5",
        "home_days_since_last_ucl",
        "away_days_since_last_ucl",
    ]

    print("\nFirst 5 matches:")

    print(
        form_history[
            display_columns
        ]
        .head(
            5
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print("\nLast 5 matches:")

    print(
        form_history[
            display_columns
        ]
        .tail(
            5
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print("\n" + "=" * 75)

    print(
        "TEAM FORM FEATURES CREATED SUCCESSFULLY"
    )

    print("=" * 75)

    return form_history


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    generate_team_form_dataset()