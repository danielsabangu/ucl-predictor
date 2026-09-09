from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from typing import Any

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
)

from src.utils.team_names import (
    normalize_team_columns,
)


# ============================================================
# FILE PATHS
# ============================================================

UCL_INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_form.csv"
)

DOMESTIC_INPUT_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "domestic_matches.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_domestic_form.csv"
)

LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_domestic_form_latest.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FORM_WINDOWS = [
    5,
    10,
]

CONGESTION_WINDOWS = [
    14,
    30,
]

EXPECTED_UCL_MATCHES = 2089


# ============================================================
# RESULT HELPERS
# ============================================================


def points_from_result(
    result: str,
    perspective: str,
) -> int:
    """
    Return league points from one team's perspective.

    perspective:
        home
        away
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
    Convert points to win/draw/loss indicators.
    """

    if points == 3:
        return 1, 0, 0

    if points == 1:
        return 0, 1, 0

    return 0, 0, 1


# ============================================================
# TEAM RECORD
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
    Create one historical domestic performance record.
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
# STATISTICS
# ============================================================


def safe_average(
    values: list[float],
) -> float:
    """
    Preserve NaN when no historical information exists.
    """

    if not values:
        return float("nan")

    return (
        sum(values)
        / len(values)
    )


def weighted_points_per_game(
    records: list[dict[str, Any]],
) -> float:
    """
    Recency-weighted PPG.

    Oldest match receives weight 1.
    Newest match receives the largest weight.

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

    numerator = sum(
        record["points"] * weight
        for record, weight
        in zip(
            records,
            weights,
        )
    )

    denominator = sum(
        weights
    )

    return (
        numerator
        / denominator
    )


def summarize_records(
    records: list[dict[str, Any]],
    window: int | None = None,
) -> dict[str, float]:
    """
    Summarize a list of historical domestic performances.
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

    return {
        "matches": count,

        "ppg": safe_average(
            [
                record["points"]
                for record in records
            ]
        ),

        "win_rate": safe_average(
            [
                record["win"]
                for record in records
            ]
        ),

        "draw_rate": safe_average(
            [
                record["draw"]
                for record in records
            ]
        ),

        "loss_rate": safe_average(
            [
                record["loss"]
                for record in records
            ]
        ),

        "gf_pg": safe_average(
            [
                record["goals_for"]
                for record in records
            ]
        ),

        "ga_pg": safe_average(
            [
                record["goals_against"]
                for record in records
            ]
        ),

        "gd_pg": safe_average(
            [
                record["goal_difference"]
                for record in records
            ]
        ),

        "clean_sheet_rate": safe_average(
            [
                record["clean_sheet"]
                for record in records
            ]
        ),

        "scoring_rate": safe_average(
            [
                record["scored"]
                for record in records
            ]
        ),
    }


# ============================================================
# DOMESTIC HISTORY CONSTRUCTION
# ============================================================


def build_domestic_histories(
    domestic: pd.DataFrame,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[pd.Timestamp]],
]:
    """
    Convert domestic matches into team-perspective histories.

    Domestic team names are canonicalized before histories
    are built.
    """

    df = domestic.copy()

    required_columns = {
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "Domestic dataset is missing columns: "
            f"{sorted(missing)}"
        )

    df = normalize_team_columns(
        df
    )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="raise",
    )

    df = df.sort_values(
        by=[
            "date",
            "domestic_match_id",
        ]
    ).reset_index(
        drop=True
    )

    histories: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    for row in df.itertuples(
        index=False
    ):

        date = pd.Timestamp(
            row.date
        )

        season = int(
            row.season
        )

        home_points = points_from_result(
            row.result,
            "home",
        )

        away_points = points_from_result(
            row.result,
            "away",
        )

        histories[
            row.home_team
        ].append(
            build_team_record(
                date=date,
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
        )

        histories[
            row.away_team
        ].append(
            build_team_record(
                date=date,
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
        )

    dates = {}

    for (
        team,
        records,
    ) in histories.items():

        records.sort(
            key=lambda record: (
                record[
                    "date"
                ]
            )
        )

        dates[
            team
        ] = [
            record[
                "date"
            ]
            for record
            in records
        ]

    return (
        histories,
        dates,
    )


# ============================================================
# PRIOR HISTORY LOOKUP
# ============================================================


def prior_record_index(
    *,
    dates: list[pd.Timestamp],
    match_date: pd.Timestamp,
) -> int:
    """
    Return the number of historical records strictly before
    match_date.

    bisect_left ensures a match occurring on the current
    calendar date is NEVER used.
    """

    return bisect_left(
        dates,
        match_date,
    )


def count_dates_in_window(
    *,
    dates: list[pd.Timestamp],
    match_date: pd.Timestamp,
    days: int,
) -> int:
    """
    Count matches in:

        match_date - N days
        <= historical date
        < match_date
    """

    end_index = bisect_left(
        dates,
        match_date,
    )

    start_date = (
        match_date
        - pd.Timedelta(
            days=days
        )
    )

    start_index = bisect_left(
        dates,
        start_date,
    )

    return (
        end_index
        - start_index
    )


# ============================================================
# TEAM PRE-MATCH DOMESTIC FEATURES
# ============================================================


def build_team_domestic_features(
    *,
    team: str,
    season: int,
    match_date: pd.Timestamp,
    histories: dict[
        str,
        list[dict[str, Any]],
    ],
    dates_lookup: dict[
        str,
        list[pd.Timestamp],
    ],
    prefix: str,
    current_venue: str,
) -> tuple[
    dict[str, float],
    pd.Timestamp | None,
]:
    """
    Create strictly pre-match domestic features for one team.
    """

    all_records = histories.get(
        team,
        [],
    )

    all_dates = dates_lookup.get(
        team,
        [],
    )

    cutoff = prior_record_index(
        dates=all_dates,
        match_date=match_date,
    )

    records = all_records[
        :cutoff
    ]

    features = {}

    # --------------------------------------------------------
    # AVAILABILITY
    # --------------------------------------------------------

    features[
        f"{prefix}_domestic_history_available"
    ] = int(
        len(records) > 0
    )

    features[
        f"{prefix}_domestic_matches_before"
    ] = len(
        records
    )

    # --------------------------------------------------------
    # LAST DOMESTIC MATCH
    # --------------------------------------------------------

    if records:

        last_domestic_date = records[
            -1
        ][
            "date"
        ]

        days_since_domestic = (
            match_date
            - last_domestic_date
        ).days

        features[
            f"{prefix}_days_since_last_domestic"
        ] = float(
            days_since_domestic
        )

    else:

        last_domestic_date = None

        features[
            f"{prefix}_days_since_last_domestic"
        ] = float(
            "nan"
        )

    # --------------------------------------------------------
    # LAST 5 / LAST 10
    # --------------------------------------------------------

    for window in FORM_WINDOWS:

        summary = summarize_records(
            records,
            window=window,
        )

        base = (
            f"{prefix}_domestic_last"
            f"{window}"
        )

        features[
            f"{base}_matches"
        ] = summary[
            "matches"
        ]

        features[
            f"{base}_ppg"
        ] = summary[
            "ppg"
        ]

        features[
            f"{base}_win_rate"
        ] = summary[
            "win_rate"
        ]

        features[
            f"{base}_draw_rate"
        ] = summary[
            "draw_rate"
        ]

        features[
            f"{base}_loss_rate"
        ] = summary[
            "loss_rate"
        ]

        features[
            f"{base}_gf_pg"
        ] = summary[
            "gf_pg"
        ]

        features[
            f"{base}_ga_pg"
        ] = summary[
            "ga_pg"
        ]

        features[
            f"{base}_gd_pg"
        ] = summary[
            "gd_pg"
        ]

        features[
            f"{base}_clean_sheet_rate"
        ] = summary[
            "clean_sheet_rate"
        ]

        features[
            f"{base}_scoring_rate"
        ] = summary[
            "scoring_rate"
        ]

    # --------------------------------------------------------
    # RECENCY-WEIGHTED MOMENTUM
    # --------------------------------------------------------

    features[
        f"{prefix}_domestic_momentum_last5"
    ] = weighted_points_per_game(
        records[
            -5:
        ]
    )

    # --------------------------------------------------------
    # CURRENT DOMESTIC SEASON
    # --------------------------------------------------------

    current_season_records = [
        record
        for record
        in records
        if record[
            "season"
        ] == season
    ]

    season_summary = summarize_records(
        current_season_records
    )

    features[
        f"{prefix}_domestic_season_matches_before"
    ] = season_summary[
        "matches"
    ]

    features[
        f"{prefix}_domestic_season_ppg_pre"
    ] = season_summary[
        "ppg"
    ]

    features[
        f"{prefix}_domestic_season_win_rate_pre"
    ] = season_summary[
        "win_rate"
    ]

    features[
        f"{prefix}_domestic_season_gf_pg_pre"
    ] = season_summary[
        "gf_pg"
    ]

    features[
        f"{prefix}_domestic_season_ga_pg_pre"
    ] = season_summary[
        "ga_pg"
    ]

    features[
        f"{prefix}_domestic_season_gd_pg_pre"
    ] = season_summary[
        "gd_pg"
    ]

    # --------------------------------------------------------
    # VENUE-SPECIFIC FORM
    # --------------------------------------------------------

    venue_records = [
        record
        for record
        in records
        if record[
            "venue"
        ] == current_venue
    ]

    venue_summary = summarize_records(
        venue_records,
        window=5,
    )

    venue_base = (
        f"{prefix}_domestic_"
        f"{current_venue}_last5"
    )

    features[
        f"{venue_base}_matches"
    ] = venue_summary[
        "matches"
    ]

    features[
        f"{venue_base}_ppg"
    ] = venue_summary[
        "ppg"
    ]

    features[
        f"{venue_base}_win_rate"
    ] = venue_summary[
        "win_rate"
    ]

    features[
        f"{venue_base}_gf_pg"
    ] = venue_summary[
        "gf_pg"
    ]

    features[
        f"{venue_base}_ga_pg"
    ] = venue_summary[
        "ga_pg"
    ]

    features[
        f"{venue_base}_gd_pg"
    ] = venue_summary[
        "gd_pg"
    ]

    # --------------------------------------------------------
    # FIXTURE LOAD
    # --------------------------------------------------------

    for days in CONGESTION_WINDOWS:

        features[
            f"{prefix}_domestic_matches_last{days}d"
        ] = count_dates_in_window(
            dates=all_dates,
            match_date=match_date,
            days=days,
        )

    return (
        features,
        last_domestic_date,
    )


# ============================================================
# SAFE DIFFERENCE
# ============================================================


def safe_difference(
    first: float,
    second: float,
) -> float:
    """
    Preserve NaN when either side has no history.
    """

    if (
        pd.isna(
            first
        )
        or pd.isna(
            second
        )
    ):

        return float(
            "nan"
        )

    return float(
        first
        - second
    )


# ============================================================
# UCL DATE WINDOW HELPERS
# ============================================================


def count_prior_ucl_dates(
    *,
    dates: list[pd.Timestamp],
    match_date: pd.Timestamp,
    days: int,
) -> int:
    """
    Count prior UCL matches in a recent time window.
    """

    if not dates:
        return 0

    start_date = (
        match_date
        - pd.Timedelta(
            days=days
        )
    )

    return sum(
        1
        for date in dates
        if (
            start_date
            <= date
            < match_date
        )
    )


# ============================================================
# KNOWN MATCH CONTEXT
# ============================================================


def add_known_match_context(
    *,
    features: dict[str, float],
    prefix: str,
    match_date: pd.Timestamp,
    last_domestic_date: pd.Timestamp | None,
    prior_ucl_dates: list[pd.Timestamp],
) -> None:
    """
    Combine the two competitions currently available to us:

        domestic league
        Champions League

    This is the best known recent-match context currently
    available.

    It is deliberately NOT named "true_rest_days" because
    domestic cups are not yet in the dataset.
    """

    if prior_ucl_dates:

        last_ucl_date = prior_ucl_dates[
            -1
        ]

    else:

        last_ucl_date = None

    candidates = [
        date
        for date in [
            last_domestic_date,
            last_ucl_date,
        ]
        if date is not None
    ]

    if candidates:

        last_known_date = max(
            candidates
        )

        features[
            f"{prefix}_days_since_last_known_match"
        ] = float(
            (
                match_date
                - last_known_date
            ).days
        )

    else:

        features[
            f"{prefix}_days_since_last_known_match"
        ] = float(
            "nan"
        )

    for days in CONGESTION_WINDOWS:

        domestic_count = features[
            f"{prefix}_domestic_matches_last{days}d"
        ]

        ucl_count = count_prior_ucl_dates(
            dates=prior_ucl_dates,
            match_date=match_date,
            days=days,
        )

        features[
            f"{prefix}_ucl_matches_last{days}d"
        ] = ucl_count

        features[
            f"{prefix}_known_matches_last{days}d"
        ] = (
            domestic_count
            + ucl_count
        )


# ============================================================
# DIFFERENCE FEATURES
# ============================================================


def add_difference_features(
    features: dict[str, float],
) -> dict[str, float]:
    """
    Add home-minus-away domestic/context differences.
    """

    features[
        "domestic_ppg_diff_last5"
    ] = safe_difference(
        features[
            "home_domestic_last5_ppg"
        ],
        features[
            "away_domestic_last5_ppg"
        ],
    )

    features[
        "domestic_ppg_diff_last10"
    ] = safe_difference(
        features[
            "home_domestic_last10_ppg"
        ],
        features[
            "away_domestic_last10_ppg"
        ],
    )

    features[
        "domestic_gd_diff_last5"
    ] = safe_difference(
        features[
            "home_domestic_last5_gd_pg"
        ],
        features[
            "away_domestic_last5_gd_pg"
        ],
    )

    features[
        "domestic_gd_diff_last10"
    ] = safe_difference(
        features[
            "home_domestic_last10_gd_pg"
        ],
        features[
            "away_domestic_last10_gd_pg"
        ],
    )

    features[
        "domestic_momentum_diff_last5"
    ] = safe_difference(
        features[
            "home_domestic_momentum_last5"
        ],
        features[
            "away_domestic_momentum_last5"
        ],
    )

    features[
        "domestic_season_ppg_diff_pre"
    ] = safe_difference(
        features[
            "home_domestic_season_ppg_pre"
        ],
        features[
            "away_domestic_season_ppg_pre"
        ],
    )

    features[
        "domestic_season_gd_diff_pre"
    ] = safe_difference(
        features[
            "home_domestic_season_gd_pg_pre"
        ],
        features[
            "away_domestic_season_gd_pg_pre"
        ],
    )

    features[
        "domestic_venue_ppg_diff_last5"
    ] = safe_difference(
        features[
            "home_domestic_home_last5_ppg"
        ],
        features[
            "away_domestic_away_last5_ppg"
        ],
    )

    features[
        "domestic_venue_gd_diff_last5"
    ] = safe_difference(
        features[
            "home_domestic_home_last5_gd_pg"
        ],
        features[
            "away_domestic_away_last5_gd_pg"
        ],
    )

    features[
        "domestic_rest_days_diff"
    ] = safe_difference(
        features[
            "home_days_since_last_domestic"
        ],
        features[
            "away_days_since_last_domestic"
        ],
    )

    features[
        "known_rest_days_diff"
    ] = safe_difference(
        features[
            "home_days_since_last_known_match"
        ],
        features[
            "away_days_since_last_known_match"
        ],
    )

    features[
        "domestic_matches_last14d_diff"
    ] = (
        features[
            "home_domestic_matches_last14d"
        ]
        - features[
            "away_domestic_matches_last14d"
        ]
    )

    features[
        "domestic_matches_last30d_diff"
    ] = (
        features[
            "home_domestic_matches_last30d"
        ]
        - features[
            "away_domestic_matches_last30d"
        ]
    )

    features[
        "known_matches_last14d_diff"
    ] = (
        features[
            "home_known_matches_last14d"
        ]
        - features[
            "away_known_matches_last14d"
        ]
    )

    features[
        "known_matches_last30d_diff"
    ] = (
        features[
            "home_known_matches_last30d"
        ]
        - features[
            "away_known_matches_last30d"
        ]
    )

    features[
        "domestic_history_availability_diff"
    ] = (
        features[
            "home_domestic_history_available"
        ]
        - features[
            "away_domestic_history_available"
        ]
    )

    return features


# ============================================================
# BUILD DATASET
# ============================================================


def build_domestic_form_features(
    ucl: pd.DataFrame,
    domestic_histories: dict[
        str,
        list[dict[str, Any]],
    ],
    domestic_dates: dict[
        str,
        list[pd.Timestamp],
    ],
) -> pd.DataFrame:
    """
    Attach leakage-safe domestic features to UCL matches.
    """

    df = ucl.copy()

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "UCL form dataset is missing columns: "
            f"{sorted(missing)}"
        )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
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

    # Prior UCL dates are maintained separately because
    # combined known-match context must only see earlier UCL
    # fixtures.
    prior_ucl_dates: dict[
        str,
        list[pd.Timestamp],
    ] = defaultdict(
        list
    )

    feature_rows = []

    for row in df.itertuples(
        index=False
    ):

        match_date = pd.Timestamp(
            row.date
        )

        season = int(
            row.season
        )

        # ----------------------------------------------------
        # HOME DOMESTIC
        # ----------------------------------------------------

        (
            home_features,
            home_last_domestic_date,
        ) = build_team_domestic_features(
            team=row.home_team,
            season=season,
            match_date=match_date,
            histories=domestic_histories,
            dates_lookup=domestic_dates,
            prefix="home",
            current_venue="home",
        )

        # ----------------------------------------------------
        # AWAY DOMESTIC
        # ----------------------------------------------------

        (
            away_features,
            away_last_domestic_date,
        ) = build_team_domestic_features(
            team=row.away_team,
            season=season,
            match_date=match_date,
            histories=domestic_histories,
            dates_lookup=domestic_dates,
            prefix="away",
            current_venue="away",
        )

        features = {
            **home_features,
            **away_features,
        }

        # ----------------------------------------------------
        # COMBINED KNOWN MATCH CONTEXT
        # ----------------------------------------------------

        add_known_match_context(
            features=features,
            prefix="home",
            match_date=match_date,
            last_domestic_date=home_last_domestic_date,
            prior_ucl_dates=prior_ucl_dates[
                row.home_team
            ],
        )

        add_known_match_context(
            features=features,
            prefix="away",
            match_date=match_date,
            last_domestic_date=away_last_domestic_date,
            prior_ucl_dates=prior_ucl_dates[
                row.away_team
            ],
        )

        # ----------------------------------------------------
        # DIFFERENCES
        # ----------------------------------------------------

        features = add_difference_features(
            features
        )

        feature_rows.append(
            features
        )

        # ----------------------------------------------------
        # ONLY NOW ADD CURRENT UCL MATCH
        # ----------------------------------------------------
        #
        # This ordering is the critical leakage barrier.
        # ----------------------------------------------------

        prior_ucl_dates[
            row.home_team
        ].append(
            match_date
        )

        prior_ucl_dates[
            row.away_team
        ].append(
            match_date
        )

    features_df = pd.DataFrame(
        feature_rows
    )

    output = pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            features_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return output


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate domestic form feature engineering.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DOMESTIC FORM DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    if len(
        dataframe
    ) != EXPECTED_UCL_MATCHES:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_UCL_MATCHES:,} "
            f"UCL rows, received "
            f"{len(dataframe):,}."
        )

    print(
        f"\nMatch rows: "
        f"{len(dataframe):,} ✓"
    )

    # --------------------------------------------------------
    # UNIQUE MATCH IDS
    # --------------------------------------------------------

    if dataframe[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate match_id values detected."
        )

    print(
        "Unique match IDs: valid ✓"
    )

    # --------------------------------------------------------
    # CHRONOLOGY
    # --------------------------------------------------------

    if not dataframe[
        "date"
    ].is_monotonic_increasing:

        raise RuntimeError(
            "Dataset is not chronologically ordered."
        )

    print(
        "Chronological ordering: valid ✓"
    )

    # --------------------------------------------------------
    # ROLLING MATCH COUNTS
    # --------------------------------------------------------

    bounded_counts = {
        "home_domestic_last5_matches": 5,
        "away_domestic_last5_matches": 5,
        "home_domestic_last10_matches": 10,
        "away_domestic_last10_matches": 10,
        "home_domestic_home_last5_matches": 5,
        "away_domestic_away_last5_matches": 5,
    }

    for (
        column,
        maximum,
    ) in bounded_counts.items():

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
                f"Invalid rolling count "
                f"in {column}."
            )

    print(
        "Rolling domestic counts: valid ✓"
    )

    # --------------------------------------------------------
    # PPG
    # --------------------------------------------------------

    ppg_columns = [
        column
        for column
        in dataframe.columns
        if (
            "domestic" in column
            and (
                column.endswith(
                    "_ppg"
                )
                or column.endswith(
                    "_ppg_pre"
                )
            )
        )
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
        "Domestic PPG ranges: valid ✓"
    )

    # --------------------------------------------------------
    # RATES
    # --------------------------------------------------------

    rate_columns = [
        column
        for column
        in dataframe.columns
        if (
            "domestic" in column
            and (
                column.endswith(
                    "_rate"
                )
                or column.endswith(
                    "_rate_pre"
                )
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
        "Domestic rate ranges: valid ✓"
    )

    # --------------------------------------------------------
    # AVAILABILITY FLAGS
    # --------------------------------------------------------

    for column in [
        "home_domestic_history_available",
        "away_domestic_history_available",
    ]:

        invalid = (
            ~dataframe[
                column
            ].isin(
                [
                    0,
                    1,
                ]
            )
        ).sum()

        if invalid > 0:

            raise RuntimeError(
                f"Invalid availability flags "
                f"in {column}."
            )

    print(
        "Domestic availability flags: valid ✓"
    )

    # --------------------------------------------------------
    # LEAKAGE / REST SANITY
    # --------------------------------------------------------
    #
    # Because only STRICTLY EARLIER dates are allowed,
    # any available previous match must be at least one day
    # before the UCL fixture.
    # --------------------------------------------------------

    recency_columns = [
        "home_days_since_last_domestic",
        "away_days_since_last_domestic",
        "home_days_since_last_known_match",
        "away_days_since_last_known_match",
    ]

    for column in recency_columns:

        values = dataframe[
            column
        ].dropna()

        if (
            values <= 0
        ).any():

            raise RuntimeError(
                f"Possible same-day/future leakage "
                f"in {column}."
            )

    print(
        "Strict pre-match date leakage check: passed ✓"
    )

    # --------------------------------------------------------
    # MATCH LOAD
    # --------------------------------------------------------

    congestion_columns = [
        "home_domestic_matches_last14d",
        "away_domestic_matches_last14d",
        "home_domestic_matches_last30d",
        "away_domestic_matches_last30d",
        "home_known_matches_last14d",
        "away_known_matches_last14d",
        "home_known_matches_last30d",
        "away_known_matches_last30d",
    ]

    for column in congestion_columns:

        if (
            dataframe[
                column
            ] < 0
        ).any():

            raise RuntimeError(
                f"Negative congestion count "
                f"in {column}."
            )

    print(
        "Fixture-load counts: valid ✓"
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    home_available = (
        dataframe[
            "home_domestic_history_available"
        ]
        ==
        1
    )

    away_available = (
        dataframe[
            "away_domestic_history_available"
        ]
        ==
        1
    )

    both = (
        home_available
        & away_available
    ).sum()

    one = (
        home_available
        ^ away_available
    ).sum()

    neither = (
        ~home_available
        & ~away_available
    ).sum()

    print(
        "\nDomestic-history coverage:"
    )

    print(
        f"  Both teams available: "
        f"{both:,}"
    )

    print(
        f"  One team available:  "
        f"{one:,}"
    )

    print(
        f"  Neither available:   "
        f"{neither:,}"
    )

    total = len(
        dataframe
    )

    print(
        f"  Both-team coverage:  "
        f"{both / total * 100:.2f}%"
    )

    # --------------------------------------------------------
    # EXPECTED MISSINGNESS
    # --------------------------------------------------------

    new_columns = [
        column
        for column
        in dataframe.columns
        if (
            "domestic" in column
            or "known_match" in column
            or "known_rest" in column
            or "known_matches" in column
        )
    ]

    missing = (
        dataframe[
            new_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"\nExpected domestic cold-start/"
        f"coverage NaNs: "
        f"{missing:,}"
    )

    print(
        "Missing domestic history preserved intentionally ✓"
    )


# ============================================================
# LATEST DOMESTIC TABLE
# ============================================================


def build_latest_domestic_table(
    *,
    histories: dict[
        str,
        list[dict[str, Any]],
    ],
    ucl_teams: set[str],
) -> pd.DataFrame:
    """
    Build current domestic summary for UCL clubs.

    Clubs outside our domestic source coverage remain present
    with history_available = 0.
    """

    rows = []

    for team in sorted(
        ucl_teams
    ):

        records = histories.get(
            team,
            [],
        )

        if records:

            last5 = summarize_records(
                records,
                window=5,
            )

            last10 = summarize_records(
                records,
                window=10,
            )

            home_last5 = summarize_records(
                [
                    record
                    for record
                    in records
                    if record[
                        "venue"
                    ] == "home"
                ],
                window=5,
            )

            away_last5 = summarize_records(
                [
                    record
                    for record
                    in records
                    if record[
                        "venue"
                    ] == "away"
                ],
                window=5,
            )

            last_date = records[
                -1
            ][
                "date"
            ].date()

        else:

            last5 = summarize_records(
                []
            )

            last10 = summarize_records(
                []
            )

            home_last5 = summarize_records(
                []
            )

            away_last5 = summarize_records(
                []
            )

            last_date = pd.NaT

        rows.append(
            {
                "team": team,

                "domestic_history_available": int(
                    bool(
                        records
                    )
                ),

                "domestic_matches": len(
                    records
                ),

                "last_domestic_match_date": (
                    last_date
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
            "domestic_history_available",
            "last5_ppg",
            "last5_gd_pg",
            "domestic_matches",
        ],
        ascending=[
            False,
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


def generate_domestic_form_dataset(
) -> pd.DataFrame:
    """
    Generate leakage-safe domestic form features for every
    historical UCL match.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - DOMESTIC FORM FEATURE ENGINEERING"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    print(
        f"\nLoading UCL form dataset:\n"
        f"{UCL_INPUT_FILE}"
    )

    if not UCL_INPUT_FILE.exists():

        raise FileNotFoundError(
            f"UCL form dataset not found:\n"
            f"{UCL_INPUT_FILE}"
        )

    print(
        f"\nLoading domestic dataset:\n"
        f"{DOMESTIC_INPUT_FILE}"
    )

    if not DOMESTIC_INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Domestic dataset not found:\n"
            f"{DOMESTIC_INPUT_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    ucl = pd.read_csv(
        UCL_INPUT_FILE
    )

    domestic = pd.read_csv(
        DOMESTIC_INPUT_FILE
    )

    print(
        f"\nUCL matches loaded: "
        f"{len(ucl):,}"
    )

    print(
        f"Domestic matches loaded: "
        f"{len(domestic):,}"
    )

    # --------------------------------------------------------
    # NORMALIZE UCL TEAM NAMES AGAIN FOR SAFETY
    # --------------------------------------------------------

    ucl = normalize_team_columns(
        ucl
    )

    # --------------------------------------------------------
    # BUILD DOMESTIC HISTORIES
    # --------------------------------------------------------

    print(
        "\nBuilding canonical domestic team histories..."
    )

    (
        domestic_histories,
        domestic_dates,
    ) = build_domestic_histories(
        domestic
    )

    print(
        f"Domestic teams tracked after normalization: "
        f"{len(domestic_histories):,}"
    )

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    print(
        "\nAttaching strictly pre-match domestic features..."
    )

    output = build_domestic_form_features(
        ucl,
        domestic_histories,
        domestic_dates,
    )

    # --------------------------------------------------------
    # QUALITY CHECK
    # --------------------------------------------------------

    run_quality_checks(
        output
    )

    # --------------------------------------------------------
    # LATEST TABLE
    # --------------------------------------------------------

    ucl_teams = set(
        output[
            "home_team"
        ]
    ).union(
        set(
            output[
                "away_team"
            ]
        )
    )

    latest = build_latest_domestic_table(
        histories=domestic_histories,
        ucl_teams=ucl_teams,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    latest.to_csv(
        LATEST_FILE,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    original_columns = set(
        ucl.columns
    )

    new_columns = [
        column
        for column
        in output.columns
        if column not in original_columns
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DOMESTIC FORM SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nMatches processed: "
        f"{len(output):,}"
    )

    print(
        f"UCL teams: "
        f"{len(ucl_teams):,}"
    )

    print(
        f"New domestic/context features created: "
        f"{len(new_columns):,}"
    )

    print(
        "\nTop 15 UCL clubs by latest available "
        "domestic Last-5 PPG:"
    )

    display = latest[
        [
            "team",
            "domestic_history_available",
            "domestic_matches",
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
        display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print(
        "\nFeature history saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nLatest UCL-team domestic table saved to:"
    )

    print(
        LATEST_FILE
    )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    display_columns = [
        "date",
        "home_team",
        "away_team",
        "result",

        "home_domestic_last5_ppg",
        "away_domestic_last5_ppg",

        "home_domestic_last5_gd_pg",
        "away_domestic_last5_gd_pg",

        "home_domestic_momentum_last5",
        "away_domestic_momentum_last5",

        "home_days_since_last_domestic",
        "away_days_since_last_domestic",

        "home_days_since_last_known_match",
        "away_days_since_last_known_match",

        "home_known_matches_last14d",
        "away_known_matches_last14d",
    ]

    print(
        "\nFirst 5 UCL matches:"
    )

    print(
        output[
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

    print(
        "\nLast 5 UCL matches:"
    )

    print(
        output[
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

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DOMESTIC FORM FEATURES CREATED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    return output


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    generate_domestic_form_dataset()