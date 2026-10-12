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

RAW_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "matches"
    / "domestic"
    / "github_football_datasets"
)

UCL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ucl_matches_master.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "research"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

FEATURE_OUTPUT_PATH = (
    OUTPUT_DIR
    / "ucl_domestic_advanced_features.csv"
)

COVERAGE_OUTPUT_PATH = (
    REPORT_DIR
    / "domestic_advanced_feature_coverage.csv"
)

SEASON_OUTPUT_PATH = (
    REPORT_DIR
    / "domestic_advanced_feature_coverage_by_season.csv"
)

SUMMARY_OUTPUT_PATH = (
    REPORT_DIR
    / "domestic_advanced_feature_summary.json"
)


# ============================================================
# CONTRACT
# ============================================================


EXPECTED_UCL_MATCHES = 2089

FIRST_SEASON = 2010

LAST_SEASON = 2025

EXPECTED_RAW_FILES = 80


LEAGUES = [
    "premier-league",
    "la-liga",
    "bundesliga",
    "serie-a",
    "ligue-1",
]


RAW_COLUMNS = [
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
]


WINDOWS = [
    5,
    10,
]


BASE_METRICS = [
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "corners_for",
    "corners_against",
]


CANDIDATE_FEATURES = [
    f"{metric}_diff_last{window}"
    for window in WINDOWS
    for metric in BASE_METRICS
]


# ============================================================
# HELPERS
# ============================================================


def season_label(
    season: int,
) -> str:
    return (
        f"{season}-"
        f"{str(season + 1)[-2:]}"
    )


def to_numeric_or_nan(
    value,
) -> float:
    numeric = pd.to_numeric(
        pd.Series(
            [value]
        ),
        errors="coerce",
    ).iloc[0]

    if pd.isna(
        numeric
    ):
        return np.nan

    return float(
        numeric
    )


def mean_last_n(
    history: list[dict],
    metric: str,
    window: int,
) -> float:
    if not history:

        return np.nan

    selected = history[
        -window:
    ]

    values = [
        row[
            metric
        ]
        for row in selected
        if pd.notna(
            row[
                metric
            ]
        )
    ]

    if not values:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


# ============================================================
# LOAD UCL
# ============================================================


def load_ucl() -> pd.DataFrame:
    if not UCL_PATH.exists():

        raise FileNotFoundError(
            UCL_PATH
        )

    dataframe = pd.read_csv(
        UCL_PATH,
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
            "UCL matches, found "
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
# LOAD RAW DOMESTIC ADVANCED STATS
# ============================================================


def load_raw_domestic() -> pd.DataFrame:
    frames = []

    files_seen = 0

    for season in range(
        FIRST_SEASON,
        LAST_SEASON + 1,
    ):

        label = season_label(
            season
        )

        for league in LEAGUES:

            path = (
                RAW_ROOT
                / label
                / f"{league}.csv"
            )

            if not path.exists():

                raise FileNotFoundError(
                    path
                )

            files_seen += 1

            dataframe = pd.read_csv(
                path,
                low_memory=False,
            )

            required = {
                "Date",
                "HomeTeam",
                "AwayTeam",
                *RAW_COLUMNS,
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
                    f"{path} missing columns: "
                    f"{sorted(missing)}"
                )

            subset = dataframe[
                [
                    "Date",
                    "HomeTeam",
                    "AwayTeam",
                    *RAW_COLUMNS,
                ]
            ].copy()

            subset[
                "season"
            ] = season

            subset[
                "season_label"
            ] = label

            subset[
                "league_source"
            ] = league

            subset[
                "source_file"
            ] = str(
                path.relative_to(
                    ROOT
                )
            )

            frames.append(
                subset
            )

    if files_seen != EXPECTED_RAW_FILES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_RAW_FILES} files, "
            f"found {files_seen}."
        )

    domestic = pd.concat(
        frames,
        ignore_index=True,
    )

    domestic = domestic.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
        }
    )

    domestic[
        "date"
    ] = pd.to_datetime(
        domestic[
            "date"
        ],
        errors="raise",
    )

    for column in RAW_COLUMNS:

        domestic[
            column
        ] = pd.to_numeric(
            domestic[
                column
            ],
            errors="coerce",
        )

    domestic = (
        normalize_team_columns(
            domestic
        )
    )

    return (
        domestic.sort_values(
            [
                "date",
                "season",
                "league_source",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# BUILD TEAM-PERSPECTIVE HISTORIES
# ============================================================


def build_histories(
    domestic: pd.DataFrame,
) -> tuple[
    dict[str, list[dict]],
    dict[str, list[pd.Timestamp]],
]:
    histories: dict[
        str,
        list[dict],
    ] = defaultdict(
        list
    )

    dates: dict[
        str,
        list[pd.Timestamp],
    ] = defaultdict(
        list
    )

    skipped_incomplete = 0

    for row in domestic.itertuples(
        index=False
    ):

        values = {
            "HS": (
                float(
                    row.HS
                )
                if pd.notna(
                    row.HS
                )
                else np.nan
            ),
            "AS": (
                float(
                    row.AS
                )
                if pd.notna(
                    row.AS
                )
                else np.nan
            ),
            "HST": (
                float(
                    row.HST
                )
                if pd.notna(
                    row.HST
                )
                else np.nan
            ),
            "AST": (
                float(
                    row.AST
                )
                if pd.notna(
                    row.AST
                )
                else np.nan
            ),
            "HC": (
                float(
                    row.HC
                )
                if pd.notna(
                    row.HC
                )
                else np.nan
            ),
            "AC": (
                float(
                    row.AC
                )
                if pd.notna(
                    row.AC
                )
                else np.nan
            ),
        }

        if any(
            pd.isna(
                value
            )
            for value
            in values.values()
        ):

            skipped_incomplete += 1

            continue

        date = pd.Timestamp(
            row.date
        )

        home_team = str(
            row.home_team
        )

        away_team = str(
            row.away_team
        )

        home_record = {
            "date": date,
            "season": int(
                row.season
            ),
            "shots_for": (
                values[
                    "HS"
                ]
            ),
            "shots_against": (
                values[
                    "AS"
                ]
            ),
            "sot_for": (
                values[
                    "HST"
                ]
            ),
            "sot_against": (
                values[
                    "AST"
                ]
            ),
            "corners_for": (
                values[
                    "HC"
                ]
            ),
            "corners_against": (
                values[
                    "AC"
                ]
            ),
        }

        away_record = {
            "date": date,
            "season": int(
                row.season
            ),
            "shots_for": (
                values[
                    "AS"
                ]
            ),
            "shots_against": (
                values[
                    "HS"
                ]
            ),
            "sot_for": (
                values[
                    "AST"
                ]
            ),
            "sot_against": (
                values[
                    "HST"
                ]
            ),
            "corners_for": (
                values[
                    "AC"
                ]
            ),
            "corners_against": (
                values[
                    "HC"
                ]
            ),
        }

        histories[
            home_team
        ].append(
            home_record
        )

        histories[
            away_team
        ].append(
            away_record
        )

        dates[
            home_team
        ].append(
            date
        )

        dates[
            away_team
        ].append(
            date
        )

    for team in histories:

        order = np.argsort(
            np.array(
                dates[
                    team
                ],
                dtype="datetime64[ns]",
            )
        )

        histories[
            team
        ] = [
            histories[
                team
            ][
                index
            ]
            for index
            in order
        ]

        dates[
            team
        ] = [
            dates[
                team
            ][
                index
            ]
            for index
            in order
        ]

    print(
        f"Skipped domestic fixtures with "
        f"incomplete core statistics: "
        f"{skipped_incomplete}"
    )

    return (
        histories,
        dates,
    )


# ============================================================
# PRE-MATCH HISTORY LOOKUP
# ============================================================


def get_pre_match_history(
    *,
    team: str,
    match_date: pd.Timestamp,
    histories: dict[
        str,
        list[dict],
    ],
    dates: dict[
        str,
        list[pd.Timestamp],
    ],
) -> list[dict]:
    team_dates = (
        dates.get(
            team,
            [],
        )
    )

    team_history = (
        histories.get(
            team,
            [],
        )
    )

    if not team_dates:

        return []

    cutoff = bisect_left(
        team_dates,
        match_date,
    )

    history = team_history[
        :cutoff
    ]

    for record in history:

        if not (
            record[
                "date"
            ]
            <
            match_date
        ):

            raise RuntimeError(
                "Leakage detected in advanced "
                "domestic history."
            )

    return history


# ============================================================
# BUILD UCL FEATURE LAYER
# ============================================================


def build_feature_layer(
    *,
    ucl: pd.DataFrame,
    histories: dict[
        str,
        list[dict],
    ],
    dates: dict[
        str,
        list[pd.Timestamp],
    ],
) -> pd.DataFrame:
    rows = []

    for match in ucl.itertuples(
        index=False
    ):

        match_date = pd.Timestamp(
            match.date
        )

        home_team = str(
            match.home_team
        )

        away_team = str(
            match.away_team
        )

        home_history = (
            get_pre_match_history(
                team=home_team,
                match_date=match_date,
                histories=histories,
                dates=dates,
            )
        )

        away_history = (
            get_pre_match_history(
                team=away_team,
                match_date=match_date,
                histories=histories,
                dates=dates,
            )
        )

        row = {
            "match_id": (
                match.match_id
            ),
            "date": (
                match_date
            ),
            "season": int(
                match.season
            ),
            "home_team": (
                home_team
            ),
            "away_team": (
                away_team
            ),
            "home_advanced_history_matches": int(
                len(
                    home_history
                )
            ),
            "away_advanced_history_matches": int(
                len(
                    away_history
                )
            ),
            "home_advanced_history_available": bool(
                len(
                    home_history
                )
                > 0
            ),
            "away_advanced_history_available": bool(
                len(
                    away_history
                )
                > 0
            ),
        }

        for window in WINDOWS:

            for metric in BASE_METRICS:

                home_value = (
                    mean_last_n(
                        home_history,
                        metric,
                        window,
                    )
                )

                away_value = (
                    mean_last_n(
                        away_history,
                        metric,
                        window,
                    )
                )

                row[
                    f"home_{metric}_pg_last{window}"
                ] = (
                    home_value
                )

                row[
                    f"away_{metric}_pg_last{window}"
                ] = (
                    away_value
                )

                if (
                    pd.notna(
                        home_value
                    )
                    and
                    pd.notna(
                        away_value
                    )
                ):

                    row[
                        f"{metric}_diff_last{window}"
                    ] = (
                        home_value
                        -
                        away_value
                    )

                else:

                    row[
                        f"{metric}_diff_last{window}"
                    ] = (
                        np.nan
                    )

        row[
            "both_advanced_history_available"
        ] = bool(
            row[
                "home_advanced_history_available"
            ]
            and
            row[
                "away_advanced_history_available"
            ]
        )

        rows.append(
            row
        )

    dataframe = pd.DataFrame(
        rows
    )

    if len(
        dataframe
    ) != EXPECTED_UCL_MATCHES:

        raise RuntimeError(
            "Unexpected advanced-feature "
            "row count."
        )

    return dataframe


# ============================================================
# VALIDATE
# ============================================================


def validate_feature_layer(
    dataframe: pd.DataFrame,
) -> None:
    if dataframe[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate match_id in Candidate C layer."
        )

    both = dataframe[
        dataframe[
            "both_advanced_history_available"
        ]
    ]

    if not both.empty:

        missing_candidate = (
            both[
                CANDIDATE_FEATURES
            ]
            .isna()
            .any(
                axis=1
            )
        )

        if missing_candidate.any():

            examples = (
                both.loc[
                    missing_candidate,
                    [
                        "match_id",
                        "date",
                        "home_team",
                        "away_team",
                        "home_advanced_history_matches",
                        "away_advanced_history_matches",
                    ],
                ]
                .head(
                    20
                )
            )

            raise RuntimeError(
                "Both-history matches contain "
                "missing Candidate C features:\n"
                +
                examples.to_string(
                    index=False
                )
            )


# ============================================================
# COVERAGE REPORT
# ============================================================


def build_coverage_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for row in dataframe.itertuples(
        index=False
    ):

        home = bool(
            row.home_advanced_history_available
        )

        away = bool(
            row.away_advanced_history_available
        )

        if home and away:

            coverage_class = (
                "BOTH_SUPPORTED"
            )

        elif home or away:

            coverage_class = (
                "ONE_SIDE_SUPPORTED"
            )

        else:

            coverage_class = (
                "NEITHER_SUPPORTED"
            )

        rows.append(
            {
                "match_id": (
                    row.match_id
                ),
                "date": (
                    row.date
                ),
                "season": int(
                    row.season
                ),
                "home_team": (
                    row.home_team
                ),
                "away_team": (
                    row.away_team
                ),
                "home_history_matches": int(
                    row.home_advanced_history_matches
                ),
                "away_history_matches": int(
                    row.away_advanced_history_matches
                ),
                "coverage_class": (
                    coverage_class
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_season_summary(
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    table = (
        coverage.groupby(
            [
                "season",
                "coverage_class",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for column in [
        "BOTH_SUPPORTED",
        "ONE_SIDE_SUPPORTED",
        "NEITHER_SUPPORTED",
    ]:

        if column not in table.columns:

            table[
                column
            ] = 0

    table = table[
        [
            "BOTH_SUPPORTED",
            "ONE_SIDE_SUPPORTED",
            "NEITHER_SUPPORTED",
        ]
    ]

    table[
        "TOTAL"
    ] = table.sum(
        axis=1
    )

    table[
        "BOTH_RATE"
    ] = (
        table[
            "BOTH_SUPPORTED"
        ]
        /
        table[
            "TOTAL"
        ]
    )

    return (
        table.reset_index()
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=" * 110
    )

    print(
        "UCL PREDICTOR V2 - CANDIDATE C"
    )

    print(
        "BUILD DOMESTIC ADVANCED PERFORMANCE FEATURES"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Core signals:"
    )

    print(
        "  shots for / against"
    )

    print(
        "  shots on target for / against"
    )

    print(
        "  corners for / against"
    )

    print()

    print(
        "Windows:"
    )

    print(
        "  last 5 domestic matches"
    )

    print(
        "  last 10 domestic matches"
    )

    print()

    print(
        "Rules:"
    )

    print(
        "  strict fixture date < target UCL date"
    )

    print(
        "  no target-match statistics"
    )

    print(
        "  no future domestic fixtures"
    )

    print(
        "  no fuzzy team matching"
    )

    print(
        "  three incomplete raw fixtures are skipped"
    )

    print(
        "  production remains untouched"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "Loading UCL master..."
    )

    ucl = (
        load_ucl()
    )

    print(
        "Loading 80 historical Big Five files..."
    )

    domestic = (
        load_raw_domestic()
    )

    print(
        f"Domestic rows loaded: "
        f"{len(domestic):,}"
    )

    # --------------------------------------------------------
    # HISTORIES
    # --------------------------------------------------------

    print(
        "Building canonical team-perspective "
        "advanced-stat histories..."
    )

    (
        histories,
        dates,
    ) = build_histories(
        domestic
    )

    print(
        f"Canonical domestic teams tracked: "
        f"{len(histories):,}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    print(
        "Building strictly pre-match UCL feature layer..."
    )

    features = (
        build_feature_layer(
            ucl=ucl,
            histories=histories,
            dates=dates,
        )
    )

    validate_feature_layer(
        features
    )

    coverage = (
        build_coverage_report(
            features
        )
    )

    season_summary = (
        build_season_summary(
            coverage
        )
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    counts = (
        coverage[
            "coverage_class"
        ]
        .value_counts()
        .reindex(
            [
                "BOTH_SUPPORTED",
                "ONE_SIDE_SUPPORTED",
                "NEITHER_SUPPORTED",
            ],
            fill_value=0,
        )
    )

    both_count = int(
        counts[
            "BOTH_SUPPORTED"
        ]
    )

    total = int(
        len(
            coverage
        )
    )

    development = coverage[
        coverage[
            "season"
        ].between(
            2018,
            2024,
        )
    ]

    development_both = int(
        (
            development[
                "coverage_class"
            ]
            ==
            "BOTH_SUPPORTED"
        ).sum()
    )

    confirmation = coverage[
        coverage[
            "season"
        ]
        ==
        2025
    ]

    confirmation_both = int(
        (
            confirmation[
                "coverage_class"
            ]
            ==
            "BOTH_SUPPORTED"
        ).sum()
    )

    print()
    print(
        "=" * 110
    )

    print(
        "MATCH-LEVEL COVERAGE"
    )

    print(
        "=" * 110
    )

    print()

    for (
        label,
        count,
    ) in counts.items():

        print(
            f"{label:>20}: "
            f"{int(count):>4} / "
            f"{total} "
            f"({count / total * 100:.2f}%)"
        )

    print()

    print(
        "By season:"
    )

    display = (
        season_summary.copy()
    )

    display[
        "BOTH_RATE"
    ] = display[
        "BOTH_RATE"
    ].map(
        lambda value: (
            f"{value * 100:.2f}%"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print()

    print(
        "2018-2024 development:"
    )

    print(
        f"  both supported: "
        f"{development_both}/"
        f"{len(development)} "
        f"("
        f"{development_both / len(development) * 100:.2f}%"
        f")"
    )

    print()

    print(
        "2025/26 historical confirmation:"
    )

    print(
        f"  both supported: "
        f"{confirmation_both}/"
        f"{len(confirmation)} "
        f"("
        f"{confirmation_both / len(confirmation) * 100:.2f}%"
        f")"
    )

    # --------------------------------------------------------
    # FEATURE STATS
    # --------------------------------------------------------

    both_features = features[
        features[
            "both_advanced_history_available"
        ]
    ]

    print()
    print(
        "=" * 110
    )

    print(
        "CANDIDATE FEATURE SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Candidate features: "
        f"{len(CANDIDATE_FEATURES)}"
    )

    print()

    for feature in (
        CANDIDATE_FEATURES
    ):

        values = (
            both_features[
                feature
            ]
            .dropna()
        )

        print(
            f"{feature:<35} "
            f"n={len(values):>4} "
            f"mean={values.mean():>8.3f} "
            f"std={values.std():>8.3f} "
            f"min={values.min():>8.3f} "
            f"max={values.max():>8.3f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_OUTPUT_PATH,
        index=False,
    )

    season_summary.to_csv(
        SEASON_OUTPUT_PATH,
        index=False,
    )

    summary = {
        "experiment": (
            "candidate_c_domestic_advanced_features"
        ),
        "status": (
            "research_only"
        ),
        "ucl_matches": int(
            len(
                features
            )
        ),
        "raw_domestic_matches": int(
            len(
                domestic
            )
        ),
        "candidate_feature_count": int(
            len(
                CANDIDATE_FEATURES
            )
        ),
        "candidate_features": (
            CANDIDATE_FEATURES
        ),
        "both_supported_matches": (
            both_count
        ),
        "both_supported_rate": float(
            both_count
            /
            total
        ),
        "development_2018_2024_matches": int(
            len(
                development
            )
        ),
        "development_2018_2024_both_supported": (
            development_both
        ),
        "confirmation_2025_matches": int(
            len(
                confirmation
            )
        ),
        "confirmation_2025_both_supported": (
            confirmation_both
        ),
        "rolling_windows": (
            WINDOWS
        ),
        "metrics": (
            BASE_METRICS
        ),
        "raw_incomplete_fixture_policy": (
            "skip_fixture_from_advanced_history"
        ),
        "leakage_rule": (
            "domestic_fixture_date_strictly_before_target_ucl_date"
        ),
        "fuzzy_matching": False,
        "production_modified": False,
    }

    with SUMMARY_OUTPUT_PATH.open(
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
        "2,089 UCL feature rows: PASS"
    )

    print(
        "All rolling histories strictly pre-match: PASS"
    )

    print(
        "Current UCL match statistics used: NO"
    )

    print(
        "Future domestic fixtures used: NO"
    )

    print(
        "Incomplete raw fixtures imputed: NO"
    )

    print(
        "Fuzzy team matching: NO"
    )

    print(
        "Production feature matrix modified: NO"
    )

    print(
        "Production model modified: NO"
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
        FEATURE_OUTPUT_PATH
    )

    print(
        COVERAGE_OUTPUT_PATH
    )

    print(
        SEASON_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()