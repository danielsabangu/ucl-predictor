from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]


MASTER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ucl_matches_master.csv"
)


TM_GAMES_PATH = (
    ROOT
    / "data"
    / "raw"
    / "players"
    / "transfermarkt"
    / "games.csv"
)


TM_CLUB_MAP_PATH = (
    ROOT
    / "data"
    / "interim"
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)


OUTPUT_PATH = (
    ROOT
    / "data"
    / "research"
    / "ucl_tournament_context_features.csv"
)


TIE_AUDIT_PATH = (
    ROOT
    / "reports"
    / "research"
    / "tournament_context_tie_audit.csv"
)


COVERAGE_PATH = (
    ROOT
    / "reports"
    / "research"
    / "tournament_context_coverage_by_season.csv"
)


SUMMARY_PATH = (
    ROOT
    / "reports"
    / "research"
    / "tournament_context_feature_summary.json"
)


# ============================================================
# CONSTANTS
# ============================================================


UCL_COMPETITION_ID = "CL"


SUPPORTED_TM_SEASONS = list(
    range(
        2012,
        2026,
    )
)


DEVELOPMENT_SEASONS = list(
    range(
        2018,
        2025,
    )
)


CONFIRMATION_SEASON = 2025


EXPECTED_MASTER_ROWS = 2089


# Explicit, audited research-only source bridge.
#
# Master:
#   match_id 886
#   2017-09-28
#   CSKA Moscow vs Manchester United
#
# Transfermarkt:
#   game_id 2917382
#   2017-09-27
#   CSKA Moscow vs Manchester United
#
# Same home/away identities.
# Unique matching pair.
# Date differs by exactly one day.
#
# This does NOT mutate the historical master.
MASTER_TO_TM_GAME_ID_BRIDGES = {
    886: 2917382,
}


# ============================================================
# CANDIDATE FEATURE CONTRACT
# ============================================================


CANDIDATE_FEATURE_COLUMNS = [
    "tournament_is_knockout",
    "tournament_is_final",
    "tournament_is_first_leg",
    "tournament_is_second_leg",
    "tournament_round_depth",
    "aggregate_goal_diff_home_pre",
    "aggregate_abs_goal_diff_pre",
    "aggregate_state_home_pre",
]


# ============================================================
# ROUND PARSING
# ============================================================


def clean_round_text(
    value: object,
) -> str:
    if pd.isna(
        value
    ):

        return ""

    text = str(
        value
    )

    text = (
        text.replace(
            "\u2013",
            "-",
        )
        .replace(
            "\u2014",
            "-",
        )
        .strip()
        .lower()
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def classify_round(
    raw_round: object,
) -> dict:
    """
    Canonical tournament classification.

    round_depth:
        0 = group / league phase
        1 = knockout playoff / intermediate stage
        2 = round of 16
        3 = quarter-final
        4 = semi-final
        5 = final

    We deliberately do NOT use the raw Transfermarkt
    aggregate field anywhere.
    """

    text = clean_round_text(
        raw_round
    )

    first_leg = bool(
        (
            "1st leg"
            in text
        )
        or
        (
            "first leg"
            in text
        )
    )

    second_leg = bool(
        (
            "2nd leg"
            in text
        )
        or
        (
            "second leg"
            in text
        )
    )

    if (
        text.startswith(
            "group "
        )
        or
        text
        ==
        "group stage"
        or
        "league phase"
        in text
    ):

        return {
            "round_family": (
                "GROUP_OR_LEAGUE"
            ),
            "is_knockout": 0,
            "is_final": 0,
            "is_first_leg": 0,
            "is_second_leg": 0,
            "round_depth": 0,
        }

    if (
        "intermediate stage"
        in text
        or
        "knockout phase play-off"
        in text
        or
        "knockout phase playoff"
        in text
        or
        "knockout play-off"
        in text
        or
        "knockout playoff"
        in text
    ):

        return {
            "round_family": (
                "KNOCKOUT_PLAYOFF"
            ),
            "is_knockout": 1,
            "is_final": 0,
            "is_first_leg": int(
                first_leg
            ),
            "is_second_leg": int(
                second_leg
            ),
            "round_depth": 1,
        }

    if (
        "last 16"
        in text
        or
        "round of 16"
        in text
    ):

        return {
            "round_family": (
                "ROUND_OF_16"
            ),
            "is_knockout": 1,
            "is_final": 0,
            "is_first_leg": int(
                first_leg
            ),
            "is_second_leg": int(
                second_leg
            ),
            "round_depth": 2,
        }

    if (
        "quarter"
        in text
    ):

        return {
            "round_family": (
                "QUARTER_FINAL"
            ),
            "is_knockout": 1,
            "is_final": 0,
            "is_first_leg": int(
                first_leg
            ),
            "is_second_leg": int(
                second_leg
            ),
            "round_depth": 3,
        }

    if (
        "semi"
        in text
    ):

        return {
            "round_family": (
                "SEMI_FINAL"
            ),
            "is_knockout": 1,
            "is_final": 0,
            "is_first_leg": int(
                first_leg
            ),
            "is_second_leg": int(
                second_leg
            ),
            "round_depth": 4,
        }

    if (
        text
        ==
        "final"
        or
        text.endswith(
            " final"
        )
    ):

        return {
            "round_family": (
                "FINAL"
            ),
            "is_knockout": 1,
            "is_final": 1,
            "is_first_leg": 0,
            "is_second_leg": 0,
            "round_depth": 5,
        }

    return {
        "round_family": (
            "UNKNOWN"
        ),
        "is_knockout": None,
        "is_final": None,
        "is_first_leg": None,
        "is_second_leg": None,
        "round_depth": None,
    }


# ============================================================
# LOAD MASTER
# ============================================================


def load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():

        raise FileNotFoundError(
            MASTER_PATH
        )

    dataframe = pd.read_csv(
        MASTER_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "stage",
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
            "Master missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_MASTER_ROWS:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_MASTER_ROWS:,} "
            "master matches, found "
            f"{len(dataframe):,}."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "match_id"
    ] = pd.to_numeric(
        dataframe[
            "match_id"
        ],
        errors="raise",
    ).astype(
        int
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

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    ).dt.normalize()

    return dataframe


# ============================================================
# LOAD CLUB MAP
# ============================================================


def load_club_map() -> pd.DataFrame:
    if not TM_CLUB_MAP_PATH.exists():

        raise FileNotFoundError(
            TM_CLUB_MAP_PATH
        )

    dataframe = pd.read_csv(
        TM_CLUB_MAP_PATH,
        low_memory=False,
    )

    required = {
        "ucl_team",
        "club_id",
        "mapping_method",
        "mapping_confidence",
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
            "Transfermarkt club map missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "club_id"
    ] = pd.to_numeric(
        dataframe[
            "club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "mapping_confidence"
    ] = pd.to_numeric(
        dataframe[
            "mapping_confidence"
        ],
        errors="raise",
    )

    if dataframe[
        "club_id"
    ].duplicated().any():

        raise RuntimeError(
            "Transfermarkt club_id mapping "
            "is not unique."
        )

    if (
        dataframe[
            "mapping_confidence"
        ]
        <
        1.0
    ).any():

        raise RuntimeError(
            "Transfermarkt mapping contains "
            "confidence below 1.0."
        )

    return dataframe


# ============================================================
# LOAD TRANSFERMARKT UCL
# ============================================================


def load_transfermarkt_ucl(
    club_map: pd.DataFrame,
) -> pd.DataFrame:
    if not TM_GAMES_PATH.exists():

        raise FileNotFoundError(
            TM_GAMES_PATH
        )

    # IMPORTANT:
    # raw `aggregate` is intentionally NOT loaded.
    #
    # Current-match aggregate can contain information
    # unavailable before kickoff and is banned from
    # Candidate E.
    usecols = [
        "game_id",
        "competition_id",
        "season",
        "round",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
        "home_club_name",
        "away_club_name",
    ]

    dataframe = pd.read_csv(
        TM_GAMES_PATH,
        usecols=usecols,
        low_memory=False,
    )

    dataframe[
        "season"
    ] = pd.to_numeric(
        dataframe[
            "season"
        ],
        errors="coerce",
    )

    dataframe = dataframe[
        (
            dataframe[
                "competition_id"
            ]
            .astype(
                str
            )
            .str.strip()
            ==
            UCL_COMPETITION_ID
        )
        &
        dataframe[
            "season"
        ].isin(
            SUPPORTED_TM_SEASONS
        )
    ].copy()

    dataframe[
        "season"
    ] = dataframe[
        "season"
    ].astype(
        int
    )

    dataframe[
        "game_id"
    ] = pd.to_numeric(
        dataframe[
            "game_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "home_club_id"
    ] = pd.to_numeric(
        dataframe[
            "home_club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "away_club_id"
    ] = pd.to_numeric(
        dataframe[
            "away_club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    ).dt.normalize()

    if dataframe[
        "game_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Transfermarkt UCL game IDs."
        )

    id_to_team = (
        club_map.set_index(
            "club_id"
        )[
            "ucl_team"
        ]
        .to_dict()
    )

    dataframe[
        "home_team"
    ] = dataframe[
        "home_club_id"
    ].map(
        id_to_team
    )

    dataframe[
        "away_team"
    ] = dataframe[
        "away_club_id"
    ].map(
        id_to_team
    )

    unresolved = dataframe[
        dataframe[
            "home_team"
        ].isna()
        |
        dataframe[
            "away_team"
        ].isna()
    ]

    if not unresolved.empty:

        raise RuntimeError(
            "Unresolved Transfermarkt UCL "
            "club IDs remain."
        )

    return dataframe


# ============================================================
# CLASSIFY TRANSFERMARKT ROUNDS
# ============================================================


def add_round_classification(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        dataframe.copy()
    )

    parsed = result[
        "round"
    ].map(
        classify_round
    )

    parsed_frame = pd.DataFrame(
        parsed.tolist(),
        index=result.index,
    )

    result = pd.concat(
        [
            result,
            parsed_frame,
        ],
        axis=1,
    )

    unknown = result[
        result[
            "round_family"
        ]
        ==
        "UNKNOWN"
    ]

    if not unknown.empty:

        values = (
            unknown[
                "round"
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        raise RuntimeError(
            "Unclassified Transfermarkt UCL "
            f"round values: {values}"
        )

    return result


# ============================================================
# TIE IDENTITIES
# ============================================================


def add_tie_keys(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        dataframe.copy()
    )

    result[
        "tie_club_low"
    ] = result[
        [
            "home_club_id",
            "away_club_id",
        ]
    ].min(
        axis=1
    )

    result[
        "tie_club_high"
    ] = result[
        [
            "home_club_id",
            "away_club_id",
        ]
    ].max(
        axis=1
    )

    result[
        "tie_key"
    ] = (
        result[
            "season"
        ].astype(
            str
        )
        +
        "|"
        +
        result[
            "round_family"
        ].astype(
            str
        )
        +
        "|"
        +
        result[
            "tie_club_low"
        ].astype(
            str
        )
        +
        "|"
        +
        result[
            "tie_club_high"
        ].astype(
            str
        )
    )

    return result


# ============================================================
# PRE-MATCH AGGREGATE STATE
# ============================================================


def build_transfermarkt_context(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    result = (
        dataframe.copy()
    )

    result[
        "aggregate_goal_diff_home_pre"
    ] = 0.0

    result[
        "aggregate_abs_goal_diff_pre"
    ] = 0.0

    result[
        "aggregate_state_home_pre"
    ] = 0.0

    result[
        "paired_first_leg_game_id"
    ] = pd.NA

    result[
        "paired_first_leg_date"
    ] = pd.NaT

    first_legs = result[
        result[
            "is_first_leg"
        ]
        ==
        1
    ].copy()

    second_legs = result[
        result[
            "is_second_leg"
        ]
        ==
        1
    ].copy()

    # --------------------------------------------------------
    # FIRST-LEG UNIQUENESS
    # --------------------------------------------------------

    first_duplicates = (
        first_legs.duplicated(
            subset=[
                "tie_key",
            ],
            keep=False,
        )
    )

    if first_duplicates.any():

        examples = first_legs.loc[
            first_duplicates,
            [
                "game_id",
                "season",
                "date",
                "home_team",
                "away_team",
                "round",
                "tie_key",
            ],
        ]

        raise RuntimeError(
            "Multiple first legs found for "
            "the same tie:\n"
            +
            examples.to_string(
                index=False
            )
        )

    first_lookup = {
        row.tie_key: row
        for row
        in first_legs.itertuples(
            index=False
        )
    }

    tie_audit_rows = []

    # --------------------------------------------------------
    # SECOND LEGS
    # --------------------------------------------------------

    for index, second in (
        second_legs.iterrows()
    ):

        key = second[
            "tie_key"
        ]

        first = first_lookup.get(
            key
        )

        if first is None:

            raise RuntimeError(
                "Second leg has no corresponding "
                "first leg:\n"
                f"game_id={second['game_id']} "
                f"season={second['season']} "
                f"round={second['round']} "
                f"{second['home_team']} vs "
                f"{second['away_team']}"
            )

        first_date = pd.Timestamp(
            first.date
        )

        second_date = pd.Timestamp(
            second[
                "date"
            ]
        )

        if not (
            first_date
            <
            second_date
        ):

            raise RuntimeError(
                "First-leg date is not strictly "
                "before second-leg date for "
                f"tie {key}."
            )

        first_home_goals = (
            pd.to_numeric(
                pd.Series(
                    [
                        first.home_club_goals
                    ]
                ),
                errors="coerce",
            )
            .iloc[
                0
            ]
        )

        first_away_goals = (
            pd.to_numeric(
                pd.Series(
                    [
                        first.away_club_goals
                    ]
                ),
                errors="coerce",
            )
            .iloc[
                0
            ]
        )

        if (
            pd.isna(
                first_home_goals
            )
            or
            pd.isna(
                first_away_goals
            )
        ):

            raise RuntimeError(
                "First-leg score missing for "
                f"tie {key}."
            )

        current_home_id = int(
            second[
                "home_club_id"
            ]
        )

        if (
            current_home_id
            ==
            int(
                first.home_club_id
            )
        ):

            aggregate_diff = float(
                first_home_goals
                -
                first_away_goals
            )

        elif (
            current_home_id
            ==
            int(
                first.away_club_id
            )
        ):

            aggregate_diff = float(
                first_away_goals
                -
                first_home_goals
            )

        else:

            raise RuntimeError(
                "Tie club identity mismatch "
                f"for tie {key}."
            )

        aggregate_state = 0.0

        if (
            aggregate_diff
            >
            0
        ):

            aggregate_state = 1.0

        elif (
            aggregate_diff
            <
            0
        ):

            aggregate_state = -1.0

        result.loc[
            index,
            "aggregate_goal_diff_home_pre",
        ] = aggregate_diff

        result.loc[
            index,
            "aggregate_abs_goal_diff_pre",
        ] = abs(
            aggregate_diff
        )

        result.loc[
            index,
            "aggregate_state_home_pre",
        ] = aggregate_state

        result.loc[
            index,
            "paired_first_leg_game_id",
        ] = int(
            first.game_id
        )

        result.loc[
            index,
            "paired_first_leg_date",
        ] = first_date

        tie_audit_rows.append(
            {
                "season": int(
                    second[
                        "season"
                    ]
                ),
                "round_family": (
                    second[
                        "round_family"
                    ]
                ),
                "second_leg_game_id": int(
                    second[
                        "game_id"
                    ]
                ),
                "second_leg_date": (
                    second_date
                ),
                "second_leg_home_team": (
                    second[
                        "home_team"
                    ]
                ),
                "second_leg_away_team": (
                    second[
                        "away_team"
                    ]
                ),
                "first_leg_game_id": int(
                    first.game_id
                ),
                "first_leg_date": (
                    first_date
                ),
                "first_leg_home_team": (
                    first.home_team
                ),
                "first_leg_away_team": (
                    first.away_team
                ),
                "first_leg_home_goals": float(
                    first_home_goals
                ),
                "first_leg_away_goals": float(
                    first_away_goals
                ),
                "aggregate_goal_diff_home_pre": (
                    aggregate_diff
                ),
                "aggregate_state_home_pre": (
                    aggregate_state
                ),
                "strictly_prior_first_leg": True,
            }
        )

    tie_audit = pd.DataFrame(
        tie_audit_rows
    )

    return (
        result,
        tie_audit,
    )


# ============================================================
# ATTACH TO MASTER
# ============================================================


def attach_context_to_master(
    *,
    master: pd.DataFrame,
    tm_context: pd.DataFrame,
) -> pd.DataFrame:
    tm_columns = [
        "game_id",
        "season",
        "date",
        "home_team",
        "away_team",
        "round",
        "round_family",
        "is_knockout",
        "is_final",
        "is_first_leg",
        "is_second_leg",
        "round_depth",
        "aggregate_goal_diff_home_pre",
        "aggregate_abs_goal_diff_pre",
        "aggregate_state_home_pre",
        "paired_first_leg_game_id",
        "paired_first_leg_date",
    ]

    merge_source = tm_context[
        tm_columns
    ].copy()

    duplicate_identity = (
        merge_source.duplicated(
            subset=[
                "season",
                "date",
                "home_team",
                "away_team",
            ],
            keep=False,
        )
    )

    if duplicate_identity.any():

        raise RuntimeError(
            "Duplicate canonical Transfermarkt "
            "fixture identities detected."
        )

    result = master.merge(
        merge_source,
        on=[
            "season",
            "date",
            "home_team",
            "away_team",
        ],
        how="left",
        validate="one_to_one",
    )

    result[
        "context_source"
    ] = pd.NA

    result.loc[
        result[
            "game_id"
        ].notna(),
        "context_source",
    ] = (
        "transfermarkt_exact"
    )

    # --------------------------------------------------------
    # EXPLICIT AUDITED DATE BRIDGES
    # --------------------------------------------------------

    tm_by_game_id = (
        tm_context.set_index(
            "game_id",
            drop=False,
        )
    )

    for (
        master_match_id,
        tm_game_id,
    ) in (
        MASTER_TO_TM_GAME_ID_BRIDGES.items()
    ):

        master_mask = (
            result[
                "match_id"
            ]
            ==
            master_match_id
        )

        if (
            int(
                master_mask.sum()
            )
            !=
            1
        ):

            raise RuntimeError(
                "Bridge master match_id "
                f"{master_match_id} not unique."
            )

        if (
            tm_game_id
            not in
            tm_by_game_id.index
        ):

            raise RuntimeError(
                "Bridge Transfermarkt game_id "
                f"{tm_game_id} not found."
            )

        target = result.loc[
            master_mask
        ].iloc[
            0
        ]

        source = tm_by_game_id.loc[
            tm_game_id
        ]

        if isinstance(
            source,
            pd.DataFrame,
        ):

            raise RuntimeError(
                "Bridge Transfermarkt game_id "
                "is not unique."
            )

        if (
            target[
                "home_team"
            ]
            !=
            source[
                "home_team"
            ]
            or
            target[
                "away_team"
            ]
            !=
            source[
                "away_team"
            ]
            or
            int(
                target[
                    "season"
                ]
            )
            !=
            int(
                source[
                    "season"
                ]
            )
        ):

            raise RuntimeError(
                "Bridge identity validation "
                "failed for master match_id "
                f"{master_match_id}."
            )

        date_difference = abs(
            (
                pd.Timestamp(
                    target[
                        "date"
                    ]
                )
                -
                pd.Timestamp(
                    source[
                        "date"
                    ]
                )
            ).days
        )

        if (
            date_difference
            !=
            1
        ):

            raise RuntimeError(
                "Expected audited bridge date "
                "difference of exactly one day."
            )

        columns_to_copy = [
            "game_id",
            "round",
            "round_family",
            "is_knockout",
            "is_final",
            "is_first_leg",
            "is_second_leg",
            "round_depth",
            "aggregate_goal_diff_home_pre",
            "aggregate_abs_goal_diff_pre",
            "aggregate_state_home_pre",
            "paired_first_leg_game_id",
            "paired_first_leg_date",
        ]

        for column in (
            columns_to_copy
        ):

            result.loc[
                master_mask,
                column,
            ] = source[
                column
            ]

        result.loc[
            master_mask,
            "context_source",
        ] = (
            "transfermarkt_explicit_date_bridge"
        )

    # --------------------------------------------------------
    # 2025 FINAL FALLBACK
    # --------------------------------------------------------
    #
    # Transfermarkt snapshot does not contain the
    # PSG-Arsenal final.
    #
    # The historical master independently identifies
    # the fixture as FINAL.
    #
    # Final status is known before kickoff and requires
    # no previous-leg aggregate state.
    # --------------------------------------------------------

    fallback_mask = (
        result[
            "game_id"
        ].isna()
        &
        (
            result[
                "season"
            ]
            ==
            CONFIRMATION_SEASON
        )
        &
        (
            result[
                "stage"
            ]
            .fillna(
                ""
            )
            .astype(
                str
            )
            .str.strip()
            .str.upper()
            ==
            "FINAL"
        )
    )

    fallback_count = int(
        fallback_mask.sum()
    )

    if (
        fallback_count
        >
        1
    ):

        raise RuntimeError(
            "More than one 2025 final fallback "
            "row detected."
        )

    if (
        fallback_count
        ==
        1
    ):

        result.loc[
            fallback_mask,
            "round",
        ] = "FINAL"

        result.loc[
            fallback_mask,
            "round_family",
        ] = "FINAL"

        result.loc[
            fallback_mask,
            "is_knockout",
        ] = 1

        result.loc[
            fallback_mask,
            "is_final",
        ] = 1

        result.loc[
            fallback_mask,
            "is_first_leg",
        ] = 0

        result.loc[
            fallback_mask,
            "is_second_leg",
        ] = 0

        result.loc[
            fallback_mask,
            "round_depth",
        ] = 5

        result.loc[
            fallback_mask,
            "aggregate_goal_diff_home_pre",
        ] = 0.0

        result.loc[
            fallback_mask,
            "aggregate_abs_goal_diff_pre",
        ] = 0.0

        result.loc[
            fallback_mask,
            "aggregate_state_home_pre",
        ] = 0.0

        result.loc[
            fallback_mask,
            "context_source",
        ] = (
            "master_final_stage_fallback"
        )

    # --------------------------------------------------------
    # CANDIDATE NAMES
    # --------------------------------------------------------

    rename_map = {
        "is_knockout": (
            "tournament_is_knockout"
        ),
        "is_final": (
            "tournament_is_final"
        ),
        "is_first_leg": (
            "tournament_is_first_leg"
        ),
        "is_second_leg": (
            "tournament_is_second_leg"
        ),
        "round_depth": (
            "tournament_round_depth"
        ),
    }

    result = result.rename(
        columns=rename_map
    )

    result[
        "tournament_context_available"
    ] = (
        result[
            "context_source"
        ].notna()
    ).astype(
        int
    )

    return result


# ============================================================
# VALIDATION
# ============================================================


def validate_feature_layer(
    dataframe: pd.DataFrame,
) -> None:
    # --------------------------------------------------------
    # DEVELOPMENT MUST BE COMPLETE
    # --------------------------------------------------------

    development = dataframe[
        dataframe[
            "season"
        ].isin(
            DEVELOPMENT_SEASONS
        )
    ]

    development_available = int(
        development[
            "tournament_context_available"
        ].sum()
    )

    if (
        development_available
        !=
        len(
            development
        )
    ):

        missing = development[
            development[
                "tournament_context_available"
            ]
            ==
            0
        ][
            [
                "match_id",
                "date",
                "season",
                "home_team",
                "away_team",
            ]
        ]

        raise RuntimeError(
            "Candidate E development coverage "
            "is not 100%:\n"
            +
            missing.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 2012-2024 SHOULD NOW BE COMPLETE
    # --------------------------------------------------------

    supported_training = dataframe[
        dataframe[
            "season"
        ].between(
            2012,
            2024,
        )
    ]

    if (
        supported_training[
            "tournament_context_available"
        ]
        .ne(
            1
        )
        .any()
    ):

        missing = supported_training[
            supported_training[
                "tournament_context_available"
            ]
            !=
            1
        ][
            [
                "match_id",
                "date",
                "season",
                "home_team",
                "away_team",
            ]
        ]

        raise RuntimeError(
            "2012-2024 context coverage "
            "is not complete:\n"
            +
            missing.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SECOND-LEG FEATURES
    # --------------------------------------------------------

    second_legs = dataframe[
        dataframe[
            "tournament_is_second_leg"
        ]
        ==
        1
    ]

    if second_legs.empty:

        raise RuntimeError(
            "No second-leg matches found."
        )

    if (
        second_legs[
            "paired_first_leg_game_id"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Second-leg rows exist without "
            "paired first-leg evidence."
        )

    if (
        pd.to_datetime(
            second_legs[
                "paired_first_leg_date"
            ],
            errors="coerce",
        )
        >=
        pd.to_datetime(
            second_legs[
                "date"
            ],
            errors="coerce",
        )
    ).any():

        raise RuntimeError(
            "A paired first leg is not "
            "strictly before the second leg."
        )

    # --------------------------------------------------------
    # FIRST LEG / SECOND LEG MUTUAL EXCLUSION
    # --------------------------------------------------------

    invalid_leg_flags = dataframe[
        (
            dataframe[
                "tournament_is_first_leg"
            ]
            ==
            1
        )
        &
        (
            dataframe[
                "tournament_is_second_leg"
            ]
            ==
            1
        )
    ]

    if not invalid_leg_flags.empty:

        raise RuntimeError(
            "Rows marked both first and "
            "second leg."
        )

    # --------------------------------------------------------
    # FINAL CANNOT BE LEG
    # --------------------------------------------------------

    invalid_finals = dataframe[
        (
            dataframe[
                "tournament_is_final"
            ]
            ==
            1
        )
        &
        (
            (
                dataframe[
                    "tournament_is_first_leg"
                ]
                ==
                1
            )
            |
            (
                dataframe[
                    "tournament_is_second_leg"
                ]
                ==
                1
            )
        )
    ]

    if not invalid_finals.empty:

        raise RuntimeError(
            "Final incorrectly marked as "
            "first/second leg."
        )

    # --------------------------------------------------------
    # CANDIDATE FEATURE MISSINGNESS
    # --------------------------------------------------------

    candidate_missing = (
        supported_training[
            CANDIDATE_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    if (
        int(
            candidate_missing
        )
        !=
        0
    ):

        raise RuntimeError(
            "Candidate E contains missing feature "
            "values inside 2012-2024."
        )


# ============================================================
# COVERAGE REPORT
# ============================================================


def build_coverage_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    report = (
        dataframe.groupby(
            "season"
        )
        .agg(
            master_matches=(
                "match_id",
                "size",
            ),
            context_available=(
                "tournament_context_available",
                "sum",
            ),
            knockout_matches=(
                "tournament_is_knockout",
                "sum",
            ),
            first_legs=(
                "tournament_is_first_leg",
                "sum",
            ),
            second_legs=(
                "tournament_is_second_leg",
                "sum",
            ),
            finals=(
                "tournament_is_final",
                "sum",
            ),
        )
        .reset_index()
    )

    report[
        "context_rate"
    ] = (
        report[
            "context_available"
        ]
        /
        report[
            "master_matches"
        ]
    )

    return report


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=" * 118
    )

    print(
        "UCL PREDICTOR V2 - CANDIDATE E"
    )

    print(
        "BUILD TOURNAMENT / TIE CONTEXT FEATURES"
    )

    print(
        "=" * 118
    )

    print()

    print(
        "Research only."
    )

    print(
        "Frozen V1 model unchanged."
    )

    print(
        "Raw Transfermarkt aggregate column "
        "is NOT loaded."
    )

    print(
        "Second-leg aggregate state uses "
        "only the earlier completed first leg."
    )

    print(
        "No fuzzy matching."
    )

    print(
        "No 2025 model evaluation."
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print(
        "Loading historical master..."
    )

    master = (
        load_master()
    )

    print(
        f"Master matches: "
        f"{len(master):,}"
    )

    print(
        "Loading deterministic Transfermarkt "
        "club map..."
    )

    club_map = (
        load_club_map()
    )

    print(
        f"Club-map rows: "
        f"{len(club_map):,}"
    )

    print(
        "Loading Transfermarkt Champions "
        "League proper..."
    )

    tm = (
        load_transfermarkt_ucl(
            club_map
        )
    )

    print(
        f"Transfermarkt CL rows: "
        f"{len(tm):,}"
    )

    # --------------------------------------------------------
    # ROUND CLASSIFICATION
    # --------------------------------------------------------

    tm = (
        add_round_classification(
            tm
        )
    )

    tm = (
        add_tie_keys(
            tm
        )
    )

    print()
    print(
        "Transfermarkt round families:"
    )

    print(
        tm[
            "round_family"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # BUILD PRIOR-LEG FEATURES
    # --------------------------------------------------------

    (
        tm_context,
        tie_audit,
    ) = (
        build_transfermarkt_context(
            tm
        )
    )

    print()
    print(
        f"Transfermarkt first legs: "
        f"{int((tm_context['is_first_leg'] == 1).sum())}"
    )

    print(
        f"Transfermarkt second legs: "
        f"{int((tm_context['is_second_leg'] == 1).sum())}"
    )

    print(
        f"Second-leg ties with verified "
        f"strictly-prior first leg: "
        f"{len(tie_audit)}"
    )

    # --------------------------------------------------------
    # ATTACH
    # --------------------------------------------------------

    features = (
        attach_context_to_master(
            master=master,
            tm_context=tm_context,
        )
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_feature_layer(
        features
    )

    coverage = (
        build_coverage_report(
            features
        )
    )

    # --------------------------------------------------------
    # DISPLAY COVERAGE
    # --------------------------------------------------------

    print()
    print(
        "=" * 118
    )

    print(
        "CONTEXT COVERAGE BY SEASON"
    )

    print(
        "=" * 118
    )

    display = (
        coverage.copy()
    )

    display[
        "context_rate"
    ] = display[
        "context_rate"
    ].map(
        lambda value: (
            f"{value * 100:.2f}%"
        )
    )

    print()

    print(
        display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # RESEARCH WINDOWS
    # --------------------------------------------------------

    development = features[
        features[
            "season"
        ].isin(
            DEVELOPMENT_SEASONS
        )
    ]

    confirmation = features[
        features[
            "season"
        ]
        ==
        CONFIRMATION_SEASON
    ]

    print()
    print(
        "=" * 118
    )

    print(
        "RESEARCH WINDOWS"
    )

    print(
        "=" * 118
    )

    print()

    print(
        f"2018-2024 context coverage: "
        f"{int(development['tournament_context_available'].sum())}/"
        f"{len(development)} "
        f"("
        f"{development['tournament_context_available'].mean() * 100:.2f}%"
        f")"
    )

    print(
        f"2025 context coverage: "
        f"{int(confirmation['tournament_context_available'].sum())}/"
        f"{len(confirmation)} "
        f"("
        f"{confirmation['tournament_context_available'].mean() * 100:.2f}%"
        f")"
    )

    print()

    print(
        "2025 remains coverage-only."
    )

    # --------------------------------------------------------
    # CONTEXT SOURCES
    # --------------------------------------------------------

    print()
    print(
        "=" * 118
    )

    print(
        "CONTEXT SOURCE DISTRIBUTION"
    )

    print(
        "=" * 118
    )

    print()

    print(
        features[
            "context_source"
        ]
        .fillna(
            "UNAVAILABLE"
        )
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # SECOND LEG SAMPLE
    # --------------------------------------------------------

    print()
    print(
        "=" * 118
    )

    print(
        "SECOND-LEG PRE-MATCH AGGREGATE SAMPLE"
    )

    print(
        "=" * 118
    )

    sample_columns = [
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "round",
        "aggregate_goal_diff_home_pre",
        "aggregate_abs_goal_diff_pre",
        "aggregate_state_home_pre",
        "paired_first_leg_game_id",
        "paired_first_leg_date",
    ]

    sample = features[
        features[
            "tournament_is_second_leg"
        ]
        ==
        1
    ][
        sample_columns
    ].head(
        40
    )

    print()

    print(
        sample.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TIE_AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "context_source",
        "tournament_context_available",
        "round",
        "round_family",
        *CANDIDATE_FEATURE_COLUMNS,
        "paired_first_leg_game_id",
        "paired_first_leg_date",
    ]

    features[
        output_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    tie_audit.to_csv(
        TIE_AUDIT_PATH,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    supported_training = features[
        features[
            "season"
        ].between(
            2012,
            2024,
        )
    ]

    summary = {
        "candidate": (
            "E_tournament_context"
        ),
        "status": (
            "research_feature_layer_built"
        ),
        "production_modified": False,
        "model_trained": False,
        "confirmation_2025_model_evaluated": False,
        "fuzzy_matching_used": False,
        "raw_transfermarkt_aggregate_loaded": False,
        "raw_transfermarkt_aggregate_used": False,
        "current_match_result_used_as_candidate_feature": False,
        "prior_first_leg_score_used_for_second_leg_context": True,
        "prior_first_leg_must_be_strictly_before_second_leg": True,
        "master_rows": int(
            len(
                features
            )
        ),
        "transfermarkt_rows": int(
            len(
                tm_context
            )
        ),
        "candidate_feature_columns": (
            CANDIDATE_FEATURE_COLUMNS
        ),
        "explicit_date_bridges": {
            str(
                key
            ): int(
                value
            )
            for (
                key,
                value,
            )
            in (
                MASTER_TO_TM_GAME_ID_BRIDGES.items()
            )
        },
        "supported_training_2012_2024_rows": int(
            len(
                supported_training
            )
        ),
        "supported_training_2012_2024_context_rows": int(
            supported_training[
                "tournament_context_available"
            ].sum()
        ),
        "development_2018_2024_rows": int(
            len(
                development
            )
        ),
        "development_2018_2024_context_rows": int(
            development[
                "tournament_context_available"
            ].sum()
        ),
        "confirmation_2025_rows": int(
            len(
                confirmation
            )
        ),
        "confirmation_2025_context_rows": int(
            confirmation[
                "tournament_context_available"
            ].sum()
        ),
        "verified_second_leg_pairs": int(
            len(
                tie_audit
            )
        ),
        "context_source_counts": (
            features[
                "context_source"
            ]
            .fillna(
                "UNAVAILABLE"
            )
            .value_counts()
            .to_dict()
        ),
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
    # FINAL VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "=" * 118
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 118
    )

    print()

    print(
        "2012-2024 context complete: PASS"
    )

    print(
        "2018-2024 development coverage 100%: PASS"
    )

    print(
        "Deterministic club-ID mapping only: PASS"
    )

    print(
        "Explicit 2017 date bridge validated: PASS"
    )

    print(
        "2025 final stage-only fallback: PASS"
    )

    print(
        "Second-leg first-leg chronology: PASS"
    )

    print(
        "Raw aggregate loaded: NO"
    )

    print(
        "Raw aggregate used: NO"
    )

    print(
        "Current-match score used as "
        "candidate feature: NO"
    )

    print(
        "2025 model evaluation performed: NO"
    )

    print(
        "Production modified: NO"
    )

    print(
        "Model trained: NO"
    )

    print()

    print(
        "=" * 118
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 118
    )

    print()

    print(
        OUTPUT_PATH
    )

    print(
        TIE_AUDIT_PATH
    )

    print(
        COVERAGE_PATH
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()