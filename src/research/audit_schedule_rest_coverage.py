from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.team_names import (
    normalize_team_name,
    normalize_team_columns,
)


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

UCL_MATCHES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ucl_matches_master.csv"
)

DOMESTIC_MATCHES_PATH = (
    ROOT
    / "data"
    / "interim"
    / "matches"
    / "domestic_matches.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

TEAM_SIDE_AUDIT_PATH = (
    REPORT_DIR
    / "schedule_rest_team_side_audit.csv"
)

SEASON_COVERAGE_PATH = (
    REPORT_DIR
    / "schedule_rest_coverage_by_season.csv"
)

CLUB_COVERAGE_PATH = (
    REPORT_DIR
    / "schedule_rest_coverage_by_club.csv"
)

UNRESOLVED_PATH = (
    REPORT_DIR
    / "schedule_rest_unresolved_club_seasons.csv"
)

DOMESTIC_LEAGUE_COVERAGE_PATH = (
    REPORT_DIR
    / "schedule_rest_domestic_league_coverage.csv"
)

IDENTITY_RESOLUTION_PATH = (
    REPORT_DIR
    / "schedule_rest_identity_resolution.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "schedule_rest_coverage_summary.json"
)


# ============================================================
# CONTRACT
# ============================================================


EXPECTED_UCL_MATCHES = 2089

EXPECTED_TEAM_SIDES = (
    EXPECTED_UCL_MATCHES
    * 2
)

EXPECTED_FIRST_SEASON = 2010

EXPECTED_LAST_SEASON = 2025


WINDOWS = [
    7,
    14,
    30,
]


# ============================================================
# DATE HELPERS
# ============================================================


def previous_date(
    dates: list[pd.Timestamp],
    target: pd.Timestamp,
) -> pd.Timestamp | None:
    """
    Return the most recent fixture date strictly before target.
    """

    position = bisect_left(
        dates,
        target,
    )

    if position == 0:

        return None

    return dates[
        position - 1
    ]


def count_previous_matches(
    dates: list[pd.Timestamp],
    target: pd.Timestamp,
    days: int,
) -> int:
    """
    Count known fixtures in the pre-match window:

        target - days <= fixture_date < target

    Same-day fixtures are intentionally excluded because our
    historical domestic source is date-level rather than
    timestamp-level.
    """

    end = bisect_left(
        dates,
        target,
    )

    start_date = (
        target
        -
        pd.Timedelta(
            days=days
        )
    )

    start = bisect_left(
        dates,
        start_date,
    )

    return int(
        end
        -
        start
    )


def merge_dates(
    *date_lists: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """
    Sorted unique union of known competitive fixture dates.
    """

    values: set[pd.Timestamp] = set()

    for dates in date_lists:

        for date in dates:

            values.add(
                pd.Timestamp(
                    date
                )
            )

    return sorted(
        values
    )


# ============================================================
# LOAD UCL
# ============================================================


def load_ucl_matches() -> pd.DataFrame:
    if not UCL_MATCHES_PATH.exists():

        raise FileNotFoundError(
            UCL_MATCHES_PATH
        )

    dataframe = pd.read_csv(
        UCL_MATCHES_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "UCL master missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_UCL_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_UCL_MATCHES:,} "
            "historical UCL matches, found "
            f"{len(dataframe):,}."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    dataframe[
        "season"
    ] = pd.to_numeric(
        dataframe[
            "season"
        ],
        errors="raise",
    ).astype(
        int
    )

    if dataframe[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate UCL match_id values."
        )

    if (
        dataframe[
            "season"
        ].min()
        !=
        EXPECTED_FIRST_SEASON
    ):

        raise RuntimeError(
            "Unexpected first UCL season."
        )

    if (
        dataframe[
            "season"
        ].max()
        !=
        EXPECTED_LAST_SEASON
    ):

        raise RuntimeError(
            "Unexpected last UCL season."
        )

    # Match the existing project identity policy.
    dataframe = (
        normalize_team_columns(
            dataframe
        )
    )

    return (
        dataframe.sort_values(
            [
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LOAD DOMESTIC
# ============================================================


def load_domestic_matches() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not DOMESTIC_MATCHES_PATH.exists():

        raise FileNotFoundError(
            DOMESTIC_MATCHES_PATH
        )

    dataframe = pd.read_csv(
        DOMESTIC_MATCHES_PATH,
        low_memory=False,
    )

    required = {
        "domestic_match_id",
        "date",
        "season",
        "country",
        "league",
        "league_code",
        "home_team",
        "away_team",
        "competition_type",
        "source",
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Domestic match table missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    dataframe[
        "season"
    ] = pd.to_numeric(
        dataframe[
            "season"
        ],
        errors="raise",
    ).astype(
        int
    )

    if dataframe[
        "domestic_match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate domestic_match_id values."
        )

    # Preserve raw provider names for the identity audit.
    raw_identity = dataframe[
        [
            "domestic_match_id",
            "season",
            "league_code",
            "home_team",
            "away_team",
        ]
    ].copy()

    raw_identity = raw_identity.rename(
        columns={
            "home_team": (
                "raw_home_team"
            ),
            "away_team": (
                "raw_away_team"
            ),
        }
    )

    # Use the exact normalization policy already used by
    # src/features/domestic_form.py.
    dataframe = (
        normalize_team_columns(
            dataframe
        )
    )

    dataframe = dataframe.merge(
        raw_identity,
        on=[
            "domestic_match_id",
            "season",
            "league_code",
        ],
        how="left",
        validate="one_to_one",
    )

    return (
        dataframe.sort_values(
            [
                "date",
                "domestic_match_id",
            ]
        )
        .reset_index(
            drop=True
        ),
        raw_identity,
    )


# ============================================================
# DOMESTIC TEAM-SEASON INDEX
# ============================================================


def build_domestic_indexes(
    dataframe: pd.DataFrame,
) -> tuple[
    dict[
        tuple[int, str],
        list[pd.Timestamp],
    ],
    dict[
        tuple[int, str],
        dict,
    ],
]:
    """
    Build canonical domestic fixture histories by:

        (season, canonical club)

    There is no fuzzy matching here. Team identity has already
    been resolved by the project's explicit alias system.
    """

    dates_by_team_season = defaultdict(
        list
    )

    metadata_by_team_season: dict[
        tuple[int, str],
        dict,
    ] = {}

    for row in dataframe.itertuples(
        index=False
    ):

        for side in (
            "home",
            "away",
        ):

            canonical_team = str(
                getattr(
                    row,
                    f"{side}_team",
                )
            )

            raw_team = str(
                getattr(
                    row,
                    f"raw_{side}_team",
                )
            )

            season = int(
                row.season
            )

            index_key = (
                season,
                canonical_team,
            )

            dates_by_team_season[
                index_key
            ].append(
                pd.Timestamp(
                    row.date
                )
            )

            existing = (
                metadata_by_team_season.get(
                    index_key
                )
            )

            if existing is None:

                metadata_by_team_season[
                    index_key
                ] = {
                    "canonical_team": (
                        canonical_team
                    ),
                    "country": (
                        row.country
                    ),
                    "league": (
                        row.league
                    ),
                    "league_code": (
                        row.league_code
                    ),
                    "competition_type": (
                        row.competition_type
                    ),
                    "source": (
                        row.source
                    ),
                    "raw_source_names": {
                        raw_team
                    },
                }

            else:

                if (
                    existing[
                        "league_code"
                    ]
                    !=
                    row.league_code
                ):

                    raise RuntimeError(
                        "Canonical team appears in "
                        "multiple domestic leagues in "
                        "the same season:\n"
                        f"{index_key}\n"
                        f"{existing['league_code']}\n"
                        f"{row.league_code}"
                    )

                existing[
                    "raw_source_names"
                ].add(
                    raw_team
                )

    final_dates = {}

    for (
        key,
        dates,
    ) in dates_by_team_season.items():

        final_dates[
            key
        ] = sorted(
            set(
                dates
            )
        )

    for metadata in (
        metadata_by_team_season.values()
    ):

        metadata[
            "raw_source_names"
        ] = sorted(
            metadata[
                "raw_source_names"
            ]
        )

    return (
        final_dates,
        metadata_by_team_season,
    )


# ============================================================
# UCL DATE INDEX
# ============================================================


def build_ucl_index(
    dataframe: pd.DataFrame,
) -> dict[
    str,
    list[pd.Timestamp],
]:
    dates_by_team = defaultdict(
        list
    )

    for row in dataframe.itertuples(
        index=False
    ):

        for canonical_team in (
            row.home_team,
            row.away_team,
        ):

            dates_by_team[
                canonical_team
            ].append(
                pd.Timestamp(
                    row.date
                )
            )

    final_dates = {}

    for (
        team,
        dates,
    ) in dates_by_team.items():

        final_dates[
            team
        ] = sorted(
            set(
                dates
            )
        )

    return final_dates


# ============================================================
# TEAM-SIDE AUDIT
# ============================================================


def build_team_side_audit(
    *,
    ucl: pd.DataFrame,
    domestic_dates: dict,
    domestic_metadata: dict,
    ucl_dates: dict,
) -> pd.DataFrame:
    rows = []

    for match in ucl.itertuples(
        index=False
    ):

        match_date = pd.Timestamp(
            match.date
        )

        season = int(
            match.season
        )

        for side in (
            "home",
            "away",
        ):

            canonical_team = str(
                getattr(
                    match,
                    f"{side}_team",
                )
            )

            opponent = str(
                getattr(
                    match,
                    (
                        "away_team"
                        if side
                        ==
                        "home"
                        else
                        "home_team"
                    ),
                )
            )

            domestic_key = (
                season,
                canonical_team,
            )

            domestic_supported = (
                domestic_key
                in
                domestic_dates
            )

            domestic_history = (
                domestic_dates.get(
                    domestic_key,
                    [],
                )
            )

            ucl_history = (
                ucl_dates.get(
                    canonical_team,
                    [],
                )
            )

            prior_domestic = (
                previous_date(
                    domestic_history,
                    match_date,
                )
                if domestic_supported
                else None
            )

            prior_ucl = previous_date(
                ucl_history,
                match_date,
            )

            # We deliberately refuse to calculate combined
            # congestion for clubs without domestic coverage.
            #
            # UCL-only schedules would severely undercount
            # workload for those clubs.
            if domestic_supported:

                known_dates = merge_dates(
                    domestic_history,
                    ucl_history,
                )

                prior_known = previous_date(
                    known_dates,
                    match_date,
                )

            else:

                known_dates = []

                prior_known = None

            if prior_known is not None:

                if not (
                    prior_known
                    <
                    match_date
                ):

                    raise RuntimeError(
                        "Leakage detected: previous "
                        "known fixture is not strictly "
                        "before target UCL match."
                    )

                days_since_last_known = int(
                    (
                        match_date
                        -
                        prior_known
                    ).days
                )

            else:

                days_since_last_known = (
                    np.nan
                )

            if prior_domestic is not None:

                days_since_last_domestic = int(
                    (
                        match_date
                        -
                        prior_domestic
                    ).days
                )

            else:

                days_since_last_domestic = (
                    np.nan
                )

            if prior_ucl is not None:

                days_since_last_ucl = int(
                    (
                        match_date
                        -
                        prior_ucl
                    ).days
                )

            else:

                days_since_last_ucl = (
                    np.nan
                )

            metadata = (
                domestic_metadata.get(
                    domestic_key
                )
            )

            row = {
                "match_id": (
                    match.match_id
                ),
                "date": (
                    match_date
                ),
                "season": (
                    season
                ),
                "side": (
                    side.upper()
                ),
                "ucl_team": (
                    canonical_team
                ),
                "opponent": (
                    opponent
                ),

                "domestic_team_season_supported": (
                    domestic_supported
                ),

                "domestic_canonical_team": (
                    metadata[
                        "canonical_team"
                    ]
                    if metadata
                    else None
                ),

                "domestic_raw_source_names": (
                    " | ".join(
                        metadata[
                            "raw_source_names"
                        ]
                    )
                    if metadata
                    else None
                ),

                "country": (
                    metadata[
                        "country"
                    ]
                    if metadata
                    else None
                ),

                "league": (
                    metadata[
                        "league"
                    ]
                    if metadata
                    else None
                ),

                "league_code": (
                    metadata[
                        "league_code"
                    ]
                    if metadata
                    else None
                ),

                "domestic_source": (
                    metadata[
                        "source"
                    ]
                    if metadata
                    else None
                ),

                "prior_domestic_available": (
                    prior_domestic
                    is not None
                ),

                "prior_ucl_available": (
                    prior_ucl
                    is not None
                ),

                "candidate_schedule_available": (
                    prior_known
                    is not None
                ),

                "previous_domestic_date": (
                    prior_domestic
                ),

                "previous_ucl_date": (
                    prior_ucl
                ),

                "previous_known_match_date": (
                    prior_known
                ),

                "days_since_last_domestic": (
                    days_since_last_domestic
                ),

                "days_since_last_ucl": (
                    days_since_last_ucl
                ),

                "days_since_last_known_match": (
                    days_since_last_known
                ),
            }

            for window in WINDOWS:

                column = (
                    f"known_matches_last_"
                    f"{window}d"
                )

                if domestic_supported:

                    row[
                        column
                    ] = (
                        count_previous_matches(
                            known_dates,
                            match_date,
                            window,
                        )
                    )

                else:

                    row[
                        column
                    ] = np.nan

            rows.append(
                row
            )

    audit = pd.DataFrame(
        rows
    )

    if len(
        audit
    ) != EXPECTED_TEAM_SIDES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TEAM_SIDES:,} "
            "historical UCL team-sides, found "
            f"{len(audit):,}."
        )

    return audit


# ============================================================
# SEASON SUMMARY
# ============================================================


def build_season_summary(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        season,
        group,
    ) in audit.groupby(
        "season",
        sort=True,
    ):

        team_sides = int(
            len(
                group
            )
        )

        domestic_supported = int(
            group[
                "domestic_team_season_supported"
            ].sum()
        )

        candidate_schedule = int(
            group[
                "candidate_schedule_available"
            ].sum()
        )

        prior_domestic = int(
            group[
                "prior_domestic_available"
            ].sum()
        )

        prior_ucl = int(
            group[
                "prior_ucl_available"
            ].sum()
        )

        rows.append(
            {
                "season": int(
                    season
                ),
                "ucl_matches": int(
                    team_sides
                    /
                    2
                ),
                "team_sides": (
                    team_sides
                ),
                "unique_ucl_clubs": int(
                    group[
                        "ucl_team"
                    ].nunique()
                ),
                "domestic_supported_sides": (
                    domestic_supported
                ),
                "domestic_supported_rate": (
                    domestic_supported
                    /
                    team_sides
                ),
                "candidate_schedule_sides": (
                    candidate_schedule
                ),
                "candidate_schedule_rate": (
                    candidate_schedule
                    /
                    team_sides
                ),
                "prior_domestic_sides": (
                    prior_domestic
                ),
                "prior_ucl_sides": (
                    prior_ucl
                ),
                "unsupported_sides": int(
                    team_sides
                    -
                    domestic_supported
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CLUB SUMMARY
# ============================================================


def build_club_summary(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        club,
        group,
    ) in audit.groupby(
        "ucl_team",
        sort=True,
    ):

        sides = int(
            len(
                group
            )
        )

        supported = int(
            group[
                "domestic_team_season_supported"
            ].sum()
        )

        candidate = int(
            group[
                "candidate_schedule_available"
            ].sum()
        )

        countries = sorted(
            set(
                group[
                    "country"
                ]
                .dropna()
                .astype(
                    str
                )
            )
        )

        leagues = sorted(
            set(
                group[
                    "league"
                ]
                .dropna()
                .astype(
                    str
                )
            )
        )

        rows.append(
            {
                "ucl_team": (
                    club
                ),
                "team_sides": (
                    sides
                ),
                "first_season": int(
                    group[
                        "season"
                    ].min()
                ),
                "last_season": int(
                    group[
                        "season"
                    ].max()
                ),
                "domestic_supported_sides": (
                    supported
                ),
                "domestic_supported_rate": (
                    supported
                    /
                    sides
                ),
                "candidate_schedule_sides": (
                    candidate
                ),
                "candidate_schedule_rate": (
                    candidate
                    /
                    sides
                ),
                "countries": (
                    " | ".join(
                        countries
                    )
                ),
                "leagues": (
                    " | ".join(
                        leagues
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "domestic_supported_rate",
                "team_sides",
                "ucl_team",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# UNRESOLVED CLUB-SEASONS
# ============================================================


def build_unresolved_summary(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    unresolved = audit[
        ~audit[
            "domestic_team_season_supported"
        ]
    ].copy()

    if unresolved.empty:

        return pd.DataFrame(
            columns=[
                "season",
                "ucl_team",
                "team_sides",
                "ucl_matches",
                "prior_ucl_available_sides",
            ]
        )

    summary = (
        unresolved.groupby(
            [
                "season",
                "ucl_team",
            ],
            as_index=False,
        )
        .agg(
            team_sides=(
                "match_id",
                "size",
            ),
            ucl_matches=(
                "match_id",
                "nunique",
            ),
            prior_ucl_available_sides=(
                "prior_ucl_available",
                "sum",
            ),
        )
    )

    summary[
        "prior_ucl_available_sides"
    ] = summary[
        "prior_ucl_available_sides"
    ].astype(
        int
    )

    return (
        summary.sort_values(
            [
                "season",
                "team_sides",
                "ucl_team",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# IDENTITY RESOLUTION REPORT
# ============================================================


def build_identity_resolution_report(
    domestic: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    seen = set()

    for row in domestic.itertuples(
        index=False
    ):

        for side in (
            "home",
            "away",
        ):

            raw_team = str(
                getattr(
                    row,
                    f"raw_{side}_team",
                )
            )

            canonical_team = str(
                getattr(
                    row,
                    f"{side}_team",
                )
            )

            key = (
                int(
                    row.season
                ),
                row.league_code,
                raw_team,
                canonical_team,
            )

            if key in seen:

                continue

            seen.add(
                key
            )

            rows.append(
                {
                    "season": int(
                        row.season
                    ),
                    "league_code": (
                        row.league_code
                    ),
                    "country": (
                        row.country
                    ),
                    "league": (
                        row.league
                    ),
                    "raw_team_name": (
                        raw_team
                    ),
                    "canonical_team": (
                        canonical_team
                    ),
                    "name_changed": (
                        raw_team
                        !=
                        canonical_team
                    ),
                }
            )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "season",
                "league_code",
                "canonical_team",
                "raw_team_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# DOMESTIC LEAGUE REPORT
# ============================================================


def build_domestic_league_summary(
    domestic: pd.DataFrame,
) -> pd.DataFrame:
    historical = domestic[
        domestic[
            "season"
        ]
        <=
        EXPECTED_LAST_SEASON
    ].copy()

    rows = []

    for (
        season,
        league_code,
    ), group in historical.groupby(
        [
            "season",
            "league_code",
        ],
        sort=True,
    ):

        clubs = (
            set(
                group[
                    "home_team"
                ]
            )
            |
            set(
                group[
                    "away_team"
                ]
            )
        )

        rows.append(
            {
                "season": int(
                    season
                ),
                "country": (
                    group[
                        "country"
                    ].iloc[
                        0
                    ]
                ),
                "league": (
                    group[
                        "league"
                    ].iloc[
                        0
                    ]
                ),
                "league_code": (
                    league_code
                ),
                "matches": int(
                    len(
                        group
                    )
                ),
                "clubs": int(
                    len(
                        clubs
                    )
                ),
                "first_date": (
                    group[
                        "date"
                    ].min()
                ),
                "last_date": (
                    group[
                        "date"
                    ].max()
                ),
                "source": (
                    " | ".join(
                        sorted(
                            group[
                                "source"
                            ]
                            .astype(
                                str
                            )
                            .unique()
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# VALIDATION
# ============================================================


def validate_no_leakage(
    audit: pd.DataFrame,
) -> None:
    available = audit[
        audit[
            "candidate_schedule_available"
        ]
    ].copy()

    if not available.empty:

        violations = available[
            available[
                "previous_known_match_date"
            ]
            >=
            available[
                "date"
            ]
        ]

        if not violations.empty:

            raise RuntimeError(
                "Schedule leakage detected:\n"
                +
                violations.head(
                    20
                ).to_string(
                    index=False
                )
            )

    for column in [
        "days_since_last_known_match",
        "days_since_last_domestic",
        "days_since_last_ucl",
    ]:

        values = pd.to_numeric(
            audit[
                column
            ],
            errors="coerce",
        )

        if (
            values.dropna()
            <
            0
        ).any():

            raise RuntimeError(
                "Negative schedule interval "
                f"found in {column}."
            )


def percentage(
    numerator: int,
    denominator: int,
) -> str:
    if denominator == 0:

        return "0.00%"

    return (
        f"{numerator / denominator * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=" * 110
    )

    print(
        "UCL PREDICTOR V2 - CANDIDATE B"
    )

    print(
        "SCHEDULE / REST / CONGESTION COVERAGE AUDIT V2"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "This audit does NOT train a model."
    )

    print(
        "Project canonical team aliases are enabled."
    )

    print(
        "No fuzzy matching."
    )

    print(
        "Only fixtures strictly before the target "
        "UCL match may enter schedule history."
    )

    print(
        "Unsupported domestic leagues remain missing."
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "Loading and canonicalizing historical UCL matches..."
    )

    ucl = (
        load_ucl_matches()
    )

    print(
        "Loading and canonicalizing domestic matches..."
    )

    (
        domestic,
        _,
    ) = (
        load_domestic_matches()
    )

    historical_domestic = domestic[
        domestic[
            "season"
        ]
        <=
        EXPECTED_LAST_SEASON
    ].copy()

    print()

    print(
        f"Historical UCL matches: "
        f"{len(ucl):,}"
    )

    print(
        f"Historical UCL team-sides: "
        f"{len(ucl) * 2:,}"
    )

    print(
        f"Domestic rows through 2025/26: "
        f"{len(historical_domestic):,}"
    )

    print(
        f"Historical domestic date range: "
        f"{historical_domestic['date'].min().date()} "
        f"to "
        f"{historical_domestic['date'].max().date()}"
    )

    print()

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------

    print(
        "Building canonical domestic "
        "team-season fixture index..."
    )

    (
        domestic_dates,
        domestic_metadata,
    ) = (
        build_domestic_indexes(
            historical_domestic
        )
    )

    print(
        "Building canonical historical "
        "UCL fixture index..."
    )

    ucl_dates = (
        build_ucl_index(
            ucl
        )
    )

    print(
        "Auditing each UCL team-side..."
    )

    audit = (
        build_team_side_audit(
            ucl=ucl,
            domestic_dates=(
                domestic_dates
            ),
            domestic_metadata=(
                domestic_metadata
            ),
            ucl_dates=(
                ucl_dates
            ),
        )
    )

    validate_no_leakage(
        audit
    )

    season_summary = (
        build_season_summary(
            audit
        )
    )

    club_summary = (
        build_club_summary(
            audit
        )
    )

    unresolved = (
        build_unresolved_summary(
            audit
        )
    )

    league_summary = (
        build_domestic_league_summary(
            historical_domestic
        )
    )

    identity_resolution = (
        build_identity_resolution_report(
            historical_domestic
        )
    )

    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    total_sides = len(
        audit
    )

    supported_sides = int(
        audit[
            "domestic_team_season_supported"
        ].sum()
    )

    candidate_sides = int(
        audit[
            "candidate_schedule_available"
        ].sum()
    )

    prior_domestic_sides = int(
        audit[
            "prior_domestic_available"
        ].sum()
    )

    prior_ucl_sides = int(
        audit[
            "prior_ucl_available"
        ].sum()
    )

    unsupported_sides = int(
        total_sides
        -
        supported_sides
    )

    unique_ucl_clubs = int(
        audit[
            "ucl_team"
        ].nunique()
    )

    supported_unique_clubs = int(
        audit.loc[
            audit[
                "domestic_team_season_supported"
            ],
            "ucl_team",
        ].nunique()
    )

    unsupported_unique_clubs = int(
        audit.loc[
            ~audit[
                "domestic_team_season_supported"
            ],
            "ucl_team",
        ].nunique()
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "OVERALL COVERAGE AFTER PROJECT CANONICALIZATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"UCL matches: "
        f"{len(ucl):,}"
    )

    print(
        f"Team-sides: "
        f"{total_sides:,}"
    )

    print(
        f"Unique canonical UCL clubs: "
        f"{unique_ucl_clubs:,}"
    )

    print()

    print(
        "Same-season domestic league identity "
        "resolved:"
    )

    print(
        f"  {supported_sides:,}/"
        f"{total_sides:,} "
        f"({percentage(supported_sides, total_sides)})"
    )

    print()

    print(
        "Candidate-ready previous known fixture:"
    )

    print(
        f"  {candidate_sides:,}/"
        f"{total_sides:,} "
        f"({percentage(candidate_sides, total_sides)})"
    )

    print()

    print(
        "Previous domestic league match found:"
    )

    print(
        f"  {prior_domestic_sides:,}/"
        f"{total_sides:,} "
        f"({percentage(prior_domestic_sides, total_sides)})"
    )

    print()

    print(
        "Previous UCL match found:"
    )

    print(
        f"  {prior_ucl_sides:,}/"
        f"{total_sides:,} "
        f"({percentage(prior_ucl_sides, total_sides)})"
    )

    print()

    print(
        f"Unsupported team-sides: "
        f"{unsupported_sides:,}"
    )

    print(
        f"Clubs with at least one supported side: "
        f"{supported_unique_clubs:,}"
    )

    print(
        f"Clubs with at least one unsupported side: "
        f"{unsupported_unique_clubs:,}"
    )

    # --------------------------------------------------------
    # SEASON
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "COVERAGE BY SEASON"
    )

    print(
        "=" * 110
    )

    print()

    season_display = (
        season_summary.copy()
    )

    for column in [
        "domestic_supported_rate",
        "candidate_schedule_rate",
    ]:

        season_display[
            column
        ] = season_display[
            column
        ].map(
            lambda value: (
                f"{value * 100:.2f}%"
            )
        )

    print(
        season_display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # UNRESOLVED
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "GENUINELY UNRESOLVED / UNSUPPORTED "
        "DOMESTIC CLUB-SEASONS"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "These remain unresolved after the project's "
        "approved canonical alias system."
    )

    print()

    if unresolved.empty:

        print(
            "None."
        )

    else:

        print(
            f"Rows: "
            f"{len(unresolved):,}"
        )

        print()

        print(
            unresolved.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # LOWEST COVERAGE
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "LOWEST CLUB COVERAGE AFTER CANONICALIZATION"
    )

    print(
        "=" * 110
    )

    print()

    club_display = (
        club_summary.head(
            40
        )
        .copy()
    )

    for column in [
        "domestic_supported_rate",
        "candidate_schedule_rate",
    ]:

        club_display[
            column
        ] = club_display[
            column
        ].map(
            lambda value: (
                f"{value * 100:.2f}%"
            )
        )

    print(
        club_display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # IDENTITY EXAMPLES
    # --------------------------------------------------------

    changed_identity = (
        identity_resolution[
            identity_resolution[
                "name_changed"
            ]
        ]
        .copy()
    )

    print()
    print(
        "=" * 110
    )

    print(
        "DOMESTIC SOURCE NAMES CANONICALIZED"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Unique source-name/season mappings changed: "
        f"{len(changed_identity):,}"
    )

    print()

    if not changed_identity.empty:

        print(
            changed_identity.head(
                80
            ).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        TEAM_SIDE_AUDIT_PATH,
        index=False,
    )

    season_summary.to_csv(
        SEASON_COVERAGE_PATH,
        index=False,
    )

    club_summary.to_csv(
        CLUB_COVERAGE_PATH,
        index=False,
    )

    unresolved.to_csv(
        UNRESOLVED_PATH,
        index=False,
    )

    league_summary.to_csv(
        DOMESTIC_LEAGUE_COVERAGE_PATH,
        index=False,
    )

    identity_resolution.to_csv(
        IDENTITY_RESOLUTION_PATH,
        index=False,
    )

    summary = {
        "experiment": (
            "candidate_b_schedule_rest_coverage_audit_v2"
        ),
        "status": (
            "research_only"
        ),
        "ucl_matches": int(
            len(
                ucl
            )
        ),
        "team_sides": int(
            total_sides
        ),
        "unique_ucl_clubs": int(
            unique_ucl_clubs
        ),
        "domestic_rows_historical": int(
            len(
                historical_domestic
            )
        ),
        "domestic_supported_sides": int(
            supported_sides
        ),
        "domestic_supported_rate": float(
            supported_sides
            /
            total_sides
        ),
        "candidate_schedule_sides": int(
            candidate_sides
        ),
        "candidate_schedule_rate": float(
            candidate_sides
            /
            total_sides
        ),
        "prior_domestic_sides": int(
            prior_domestic_sides
        ),
        "prior_ucl_sides": int(
            prior_ucl_sides
        ),
        "unsupported_sides": int(
            unsupported_sides
        ),
        "supported_unique_clubs": int(
            supported_unique_clubs
        ),
        "unsupported_unique_clubs": int(
            unsupported_unique_clubs
        ),
        "identity_policy": (
            "src.utils.team_names.normalize_team_name"
        ),
        "fuzzy_matching": False,
        "domestic_scope": (
            "domestic_league_only"
        ),
        "known_schedule_definition": (
            "canonical_same_season_domestic_league_plus_prior_ucl"
        ),
        "unsupported_policy": (
            "leave_schedule_features_missing"
        ),
        "leakage_rule": (
            "fixture_date_strictly_before_target_ucl_date"
        ),
        "same_day_policy": (
            "excluded_due_to_date_level_historical_sources"
        ),
        "production_modified": False,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"{EXPECTED_UCL_MATCHES:,} "
        "historical UCL matches: PASS"
    )

    print(
        f"{EXPECTED_TEAM_SIDES:,} "
        "historical UCL team-sides: PASS"
    )

    print(
        "Existing project canonical team resolver used: PASS"
    )

    print(
        "Explicit deterministic aliases only: PASS"
    )

    print(
        "No fuzzy matching: PASS"
    )

    print(
        "All candidate fixtures strictly before "
        "target UCL date: PASS"
    )

    print(
        "Same-day fixtures excluded: PASS"
    )

    print(
        "Future fixtures excluded: PASS"
    )

    print(
        "Unsupported domestic clubs remain "
        "missing: PASS"
    )

    print(
        "Missing domestic schedule is not treated "
        "as extra rest: PASS"
    )

    print(
        "Domestic cup coverage is not claimed: PASS"
    )

    print(
        "Production feature matrix untouched: PASS"
    )

    print(
        "Production model untouched: PASS"
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 110
    )

    print()

    print(
        TEAM_SIDE_AUDIT_PATH
    )

    print(
        SEASON_COVERAGE_PATH
    )

    print(
        CLUB_COVERAGE_PATH
    )

    print(
        UNRESOLVED_PATH
    )

    print(
        DOMESTIC_LEAGUE_COVERAGE_PATH
    )

    print(
        IDENTITY_RESOLUTION_PATH
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Candidate B canonical coverage audit complete."
    )

    print(
        "No schedule/rest feature has entered "
        "the prediction model."
    )


if __name__ == "__main__":
    main()