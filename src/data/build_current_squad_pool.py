from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)


# ============================================================
# PATHS
# ============================================================

RESOLVED_CURRENT_PLAYERS_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "current_player_local_valuations_resolved.csv"
)

CURRENT_TEAM_SNAPSHOT_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "football_data_ucl_teams.csv"
)

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

EXTENDED_CLUB_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map_2026_27.csv"
)

TRANSFERMARKT_PLAYERS_FILE = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
    / "players.csv"
)

TRANSFERMARKT_VALUATIONS_FILE = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
    / "player_valuations.csv"
)

OUTPUT_DIR = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
)

PLAYER_POOL_FILE = (
    OUTPUT_DIR
    / "ucl_current_squad_pool.csv"
)

TEAM_AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_current_squad_pool_audit.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "ucl_current_squad_pool_metadata.json"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_UCL_CLUBS = 36

EXPECTED_FOOTBALL_DATA_SQUAD_CLUBS = 33

MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE = 20

EXPECTED_FALLBACK_CLUBS = {
    "Sabah FK",
    "Slavia Prague",
    "ŠK Slovan Bratislava",
}


# ============================================================
# COLUMN DETECTION
# ============================================================


def detect_column(
    *,
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
    required: bool = True,
) -> str | None:

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    if required:

        raise RuntimeError(
            f"Could not detect {description} column.\n"
            f"Available columns:\n"
            f"{list(dataframe.columns)}"
        )

    return None


# ============================================================
# LOAD CURRENT TEAM MAP
# ============================================================


def load_current_team_map() -> pd.DataFrame:

    if not CURRENT_TEAM_MAP_FILE.exists():

        raise FileNotFoundError(
            "Current UCL team map not found:\n"
            f"{CURRENT_TEAM_MAP_FILE}"
        )

    dataframe = pd.read_csv(
        CURRENT_TEAM_MAP_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "football_data_team_id",
        "identity_status",
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
            "Current team map missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_UCL_CLUBS:

        raise RuntimeError(
            f"Expected {EXPECTED_UCL_CLUBS} current clubs, "
            f"found {len(dataframe)}."
        )

    return dataframe


# ============================================================
# LOAD TEAM SNAPSHOT
# ============================================================


def load_team_snapshot() -> pd.DataFrame:

    if not CURRENT_TEAM_SNAPSHOT_FILE.exists():

        raise FileNotFoundError(
            "Football-data team snapshot not found:\n"
            f"{CURRENT_TEAM_SNAPSHOT_FILE}"
        )

    dataframe = pd.read_csv(
        CURRENT_TEAM_SNAPSHOT_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "squad_size",
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
            "Team snapshot missing columns: "
            f"{sorted(missing)}"
        )

    dataframe[
        "squad_size"
    ] = pd.to_numeric(
        dataframe[
            "squad_size"
        ],
        errors="coerce",
    ).fillna(
        0
    ).astype(
        int
    )

    return dataframe


# ============================================================
# LOAD EXTENDED TRANSFERMARKT MAP
# ============================================================


def load_extended_club_map() -> pd.DataFrame:

    if not EXTENDED_CLUB_MAP_FILE.exists():

        raise FileNotFoundError(
            "Extended Transfermarkt map not found:\n"
            f"{EXTENDED_CLUB_MAP_FILE}"
        )

    dataframe = pd.read_csv(
        EXTENDED_CLUB_MAP_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "transfermarkt_club_id",
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
            "Extended Transfermarkt map missing columns: "
            f"{sorted(missing)}"
        )

    dataframe[
        "transfermarkt_club_id"
    ] = pd.to_numeric(
        dataframe[
            "transfermarkt_club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    return dataframe


# ============================================================
# LOAD RESOLVED FOOTBALL-DATA PLAYERS
# ============================================================


def load_resolved_current_players() -> pd.DataFrame:

    if not RESOLVED_CURRENT_PLAYERS_FILE.exists():

        raise FileNotFoundError(
            "Resolved current-player valuation file "
            "not found:\n"
            f"{RESOLVED_CURRENT_PLAYERS_FILE}"
        )

    dataframe = pd.read_csv(
        RESOLVED_CURRENT_PLAYERS_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "football_data_team_id",
        "football_data_player_id",
        "player_name",
        "date_of_birth",
        "position",
        "nationality",
        "transfermarkt_club_id",
        "mapping_status",
        "transfermarkt_player_id",
        "usable_market_value_eur",
        "valuation_source",
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
            "Resolved player file missing columns: "
            f"{sorted(missing)}"
        )

    return dataframe


# ============================================================
# BUILD FOOTBALL-DATA PLAYER POOL
# ============================================================


def build_football_data_pool(
    players: pd.DataFrame,
) -> pd.DataFrame:

    output = pd.DataFrame()

    output[
        "canonical_team"
    ] = players[
        "canonical_team"
    ]

    output[
        "football_data_team_id"
    ] = players[
        "football_data_team_id"
    ]

    output[
        "transfermarkt_club_id"
    ] = players[
        "transfermarkt_club_id"
    ]

    output[
        "football_data_player_id"
    ] = players[
        "football_data_player_id"
    ]

    output[
        "transfermarkt_player_id"
    ] = players[
        "transfermarkt_player_id"
    ]

    output[
        "player_name"
    ] = players[
        "player_name"
    ]

    output[
        "date_of_birth"
    ] = players[
        "date_of_birth"
    ]

    output[
        "position"
    ] = players[
        "position"
    ]

    output[
        "nationality"
    ] = players[
        "nationality"
    ]

    output[
        "market_value_eur"
    ] = players[
        "usable_market_value_eur"
    ]

    output[
        "valuation_source"
    ] = players[
        "valuation_source"
    ]

    output[
        "player_identity_status"
    ] = players[
        "mapping_status"
    ]

    output[
        "squad_source"
    ] = (
        "football_data_current_squad"
    )

    output[
        "confirmed_current_squad"
    ] = 1

    output[
        "transfermarkt_fallback_pool"
    ] = 0

    output[
        "ucl_registration_confirmed"
    ] = 0

    return output


# ============================================================
# LOAD TRANSFERMARKT PLAYERS
# ============================================================


def load_transfermarkt_players() -> pd.DataFrame:

    if not TRANSFERMARKT_PLAYERS_FILE.exists():

        raise FileNotFoundError(
            "Transfermarkt players.csv not found:\n"
            f"{TRANSFERMARKT_PLAYERS_FILE}"
        )

    dataframe = pd.read_csv(
        TRANSFERMARKT_PLAYERS_FILE,
        low_memory=False,
    )

    player_id_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "player_id",
            "id",
        ],
        description="player ID",
    )

    name_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "name",
            "player_name",
        ],
        description="player name",
    )

    club_id_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "current_club_id",
            "club_id",
        ],
        description="current club ID",
    )

    dob_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "date_of_birth",
            "birth_date",
        ],
        description="date of birth",
        required=False,
    )

    position_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "position",
            "sub_position",
        ],
        description="position",
        required=False,
    )

    nationality_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "country_of_citizenship",
            "nationality",
            "country_of_birth",
        ],
        description="nationality",
        required=False,
    )

    snapshot_value_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "market_value_in_eur",
            "market_value",
        ],
        description="snapshot market value",
        required=False,
    )

    dataframe[
        "_player_id"
    ] = pd.to_numeric(
        dataframe[
            player_id_column
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "_name"
    ] = dataframe[
        name_column
    ].astype(
        str
    )

    dataframe[
        "_club_id"
    ] = pd.to_numeric(
        dataframe[
            club_id_column
        ],
        errors="coerce",
    )

    if dob_column is not None:

        dataframe[
            "_dob"
        ] = dataframe[
            dob_column
        ]

    else:

        dataframe[
            "_dob"
        ] = pd.NA

    if position_column is not None:

        dataframe[
            "_position"
        ] = dataframe[
            position_column
        ]

    else:

        dataframe[
            "_position"
        ] = pd.NA

    if nationality_column is not None:

        dataframe[
            "_nationality"
        ] = dataframe[
            nationality_column
        ]

    else:

        dataframe[
            "_nationality"
        ] = pd.NA

    if snapshot_value_column is not None:

        dataframe[
            "_snapshot_value"
        ] = pd.to_numeric(
            dataframe[
                snapshot_value_column
            ],
            errors="coerce",
        )

    else:

        dataframe[
            "_snapshot_value"
        ] = np.nan

    return dataframe


# ============================================================
# LOAD LATEST VALUATIONS
# ============================================================


def load_latest_valuations(
    player_ids: set[int],
) -> pd.DataFrame:

    if not TRANSFERMARKT_VALUATIONS_FILE.exists():

        raise FileNotFoundError(
            "Transfermarkt valuations file not found:\n"
            f"{TRANSFERMARKT_VALUATIONS_FILE}"
        )

    dataframe = pd.read_csv(
        TRANSFERMARKT_VALUATIONS_FILE,
        low_memory=False,
    )

    player_id_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "player_id",
        ],
        description="valuation player ID",
    )

    date_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "date",
            "valuation_date",
        ],
        description="valuation date",
    )

    value_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "market_value_in_eur",
            "market_value",
        ],
        description="valuation amount",
    )

    dataframe[
        "_player_id"
    ] = pd.to_numeric(
        dataframe[
            player_id_column
        ],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe[
            "_player_id"
        ].isin(
            player_ids
        )
    ].copy()

    dataframe[
        "_valuation_date"
    ] = pd.to_datetime(
        dataframe[
            date_column
        ],
        errors="coerce",
    )

    dataframe[
        "_market_value"
    ] = pd.to_numeric(
        dataframe[
            value_column
        ],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe[
            "_valuation_date"
        ].notna()
        &
        dataframe[
            "_market_value"
        ].notna()
        &
        (
            dataframe[
                "_market_value"
            ]
            >=
            0
        )
    ].copy()

    dataframe = dataframe.sort_values(
        by=[
            "_player_id",
            "_valuation_date",
        ]
    )

    latest = (
        dataframe
        .groupby(
            "_player_id",
            as_index=False,
        )
        .tail(
            1
        )
    )

    return latest[
        [
            "_player_id",
            "_valuation_date",
            "_market_value",
        ]
    ].rename(
        columns={
            "_player_id": (
                "transfermarkt_player_id"
            ),
            "_valuation_date": (
                "latest_valuation_date"
            ),
            "_market_value": (
                "latest_market_value_eur"
            ),
        }
    )


# ============================================================
# IDENTIFY FALLBACK CLUBS
# ============================================================


def identify_fallback_clubs(
    team_snapshot: pd.DataFrame,
) -> set[str]:
    """
    Identify clubs whose football-data squad response is too
    incomplete to be trusted as the primary current-squad source.

    A tiny partial squad is treated the same as a missing squad.
    The expected fallback set remains explicit so a new provider
    degradation cannot silently enter production.
    """

    required_columns = {
        "canonical_team",
        "squad_size",
    }

    missing_columns = (
        required_columns
        -
        set(team_snapshot.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Team snapshot is missing required fallback columns: "
            f"{sorted(missing_columns)}"
        )

    snapshot = team_snapshot[
        [
            "canonical_team",
            "squad_size",
        ]
    ].copy()

    snapshot[
        "squad_size"
    ] = pd.to_numeric(
        snapshot[
            "squad_size"
        ],
        errors="coerce",
    )

    if snapshot[
        "squad_size"
    ].isna().any():
        invalid = snapshot.loc[
            snapshot[
                "squad_size"
            ].isna(),
            "canonical_team",
        ].tolist()

        raise RuntimeError(
            "Invalid football-data squad_size values for: "
            f"{sorted(invalid)}"
        )

    if (
        snapshot[
            "canonical_team"
        ].duplicated()
        .any()
    ):
        duplicates = sorted(
            snapshot.loc[
                snapshot[
                    "canonical_team"
                ].duplicated(
                    keep=False
                ),
                "canonical_team",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Duplicate team rows in current team snapshot: "
            f"{duplicates}"
        )

    fallback = set(
        snapshot.loc[
            snapshot[
                "squad_size"
            ]
            <
            MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE,
            "canonical_team",
        ]
    )

    if (
        fallback
        !=
        EXPECTED_FALLBACK_CLUBS
    ):
        details = (
            snapshot.loc[
                snapshot[
                    "squad_size"
                ]
                <
                MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE,
                [
                    "canonical_team",
                    "squad_size",
                ],
            ]
            .sort_values(
                "canonical_team"
            )
            .to_dict(
                orient="records"
            )
        )

        raise RuntimeError(
            "\nUnexpected football-data incomplete-squad set.\n"
            f"Minimum trusted squad size: "
            f"{MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE}\n"
            f"Expected: "
            f"{sorted(EXPECTED_FALLBACK_CLUBS)}\n"
            f"Found:    "
            f"{sorted(fallback)}\n"
            f"Observed incomplete squads: "
            f"{details}"
        )

    return fallback


# ============================================================
# BUILD TRANSFERMARKT FALLBACK POOL
# ============================================================


def build_fallback_pool(
    *,
    fallback_clubs: set[str],
    current_team_map: pd.DataFrame,
    club_map: pd.DataFrame,
    transfermarkt_players: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for canonical_team in sorted(
        fallback_clubs
    ):

        current_team = current_team_map[
            current_team_map[
                "canonical_team"
            ]
            ==
            canonical_team
        ]

        if len(
            current_team
        ) != 1:

            raise RuntimeError(
                f"Could not resolve one current team row "
                f"for {canonical_team}."
            )

        tm_team = club_map[
            club_map[
                "canonical_team"
            ]
            ==
            canonical_team
        ]

        if len(
            tm_team
        ) != 1:

            raise RuntimeError(
                f"Could not resolve one Transfermarkt "
                f"club ID for {canonical_team}."
            )

        football_data_team_id = int(
            current_team.iloc[
                0
            ][
                "football_data_team_id"
            ]
        )

        transfermarkt_club_id = int(
            tm_team.iloc[
                0
            ][
                "transfermarkt_club_id"
            ]
        )

        players = transfermarkt_players[
            pd.to_numeric(
                transfermarkt_players[
                    "_club_id"
                ],
                errors="coerce",
            )
            ==
            transfermarkt_club_id
        ].copy()

        for (
            _,
            player,
        ) in players.iterrows():

            rows.append(
                {
                    "canonical_team": (
                        canonical_team
                    ),
                    "football_data_team_id": (
                        football_data_team_id
                    ),
                    "transfermarkt_club_id": (
                        transfermarkt_club_id
                    ),
                    "football_data_player_id": (
                        np.nan
                    ),
                    "transfermarkt_player_id": int(
                        player[
                            "_player_id"
                        ]
                    ),
                    "player_name": (
                        player[
                            "_name"
                        ]
                    ),
                    "date_of_birth": (
                        player[
                            "_dob"
                        ]
                    ),
                    "position": (
                        player[
                            "_position"
                        ]
                    ),
                    "nationality": (
                        player[
                            "_nationality"
                        ]
                    ),
                    "snapshot_market_value_eur": (
                        player[
                            "_snapshot_value"
                        ]
                    ),
                    "player_identity_status": (
                        "TRANSFERMARKT_ONLY"
                    ),
                    "squad_source": (
                        "transfermarkt_current_club_fallback"
                    ),
                    "confirmed_current_squad": 0,
                    "transfermarkt_fallback_pool": 1,
                    "ucl_registration_confirmed": 0,
                }
            )

    fallback = pd.DataFrame(
        rows
    )

    if fallback.empty:

        raise RuntimeError(
            "No fallback Transfermarkt players found."
        )

    return fallback


# ============================================================
# ADD FALLBACK VALUATIONS
# ============================================================


def add_fallback_valuations(
    fallback: pd.DataFrame,
) -> pd.DataFrame:

    player_ids = set(
        fallback[
            "transfermarkt_player_id"
        ].astype(
            int
        )
    )

    latest = load_latest_valuations(
        player_ids
    )

    output = fallback.merge(
        latest,
        on="transfermarkt_player_id",
        how="left",
        validate="one_to_one",
    )

    output[
        "market_value_eur"
    ] = output[
        "latest_market_value_eur"
    ]

    output[
        "valuation_source"
    ] = pd.NA

    history_mask = output[
        "latest_market_value_eur"
    ].notna()

    output.loc[
        history_mask,
        "valuation_source",
    ] = (
        "player_valuations_history_latest"
    )

    snapshot_mask = (
        output[
            "market_value_eur"
        ].isna()
        &
        output[
            "snapshot_market_value_eur"
        ].notna()
    )

    output.loc[
        snapshot_mask,
        "market_value_eur",
    ] = output.loc[
        snapshot_mask,
        "snapshot_market_value_eur",
    ]

    output.loc[
        snapshot_mask,
        "valuation_source",
    ] = "players_csv_snapshot"

    return output


# ============================================================
# STANDARDIZE FALLBACK COLUMNS
# ============================================================


def standardize_fallback(
    fallback: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "canonical_team",
        "football_data_team_id",
        "transfermarkt_club_id",
        "football_data_player_id",
        "transfermarkt_player_id",
        "player_name",
        "date_of_birth",
        "position",
        "nationality",
        "market_value_eur",
        "valuation_source",
        "player_identity_status",
        "squad_source",
        "confirmed_current_squad",
        "transfermarkt_fallback_pool",
        "ucl_registration_confirmed",
    ]

    return fallback[
        columns
    ].copy()


# ============================================================
# BUILD UNIFIED POOL
# ============================================================


def build_unified_pool(
    *,
    football_data_pool: pd.DataFrame,
    fallback_pool: pd.DataFrame,
    fallback_clubs: set[str],
) -> pd.DataFrame:
    """
    Build one current-player pool.

    Clubs classified as fallback-only must not retain partial
    football-data squad rows. This prevents duplicate/mixed squad
    sources when football-data returns an incomplete response.
    """

    required_columns = {
        "canonical_team",
        "squad_source",
        "player_name",
    }

    for (
        dataframe_name,
        dataframe,
    ) in [
        (
            "football_data_pool",
            football_data_pool,
        ),
        (
            "fallback_pool",
            fallback_pool,
        ),
    ]:
        missing_columns = (
            required_columns
            -
            set(dataframe.columns)
        )

        if missing_columns:
            raise RuntimeError(
                f"{dataframe_name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    primary_pool = (
        football_data_pool.loc[
            ~football_data_pool[
                "canonical_team"
            ].isin(
                fallback_clubs
            )
        ]
        .copy()
    )

    excluded_partial_rows = (
        len(football_data_pool)
        -
        len(primary_pool)
    )

    if excluded_partial_rows:
        print(
            "\nExcluded partial football-data rows "
            "for fallback-only clubs: "
            f"{excluded_partial_rows}"
        )

        excluded_counts = (
            football_data_pool.loc[
                football_data_pool[
                    "canonical_team"
                ].isin(
                    fallback_clubs
                )
            ]
            .groupby(
                "canonical_team"
            )
            .size()
            .sort_index()
        )

        for (
            canonical_team,
            row_count,
        ) in excluded_counts.items():
            print(
                f"  {canonical_team}: "
                f"{int(row_count)}"
            )

    primary_fallback_overlap = (
        set(
            primary_pool[
                "canonical_team"
            ]
        )
        &
        fallback_clubs
    )

    if primary_fallback_overlap:
        raise RuntimeError(
            "Fallback-only clubs remain in the "
            "football-data primary pool: "
            f"{sorted(primary_fallback_overlap)}"
        )

    unified = pd.concat(
        [
            primary_pool,
            fallback_pool,
        ],
        ignore_index=True,
    )

    mixed_source_clubs = (
        unified.groupby(
            "canonical_team"
        )[
            "squad_source"
        ]
        .nunique()
    )

    mixed_source_clubs = set(
        mixed_source_clubs[
            mixed_source_clubs
            >
            1
        ].index
    )

    if mixed_source_clubs:
        raise RuntimeError(
            "A club is represented by multiple current "
            "squad source types: "
            f"{sorted(mixed_source_clubs)}"
        )

    unified = unified.sort_values(
        by=[
            "canonical_team",
            "squad_source",
            "player_name",
        ]
    ).reset_index(
        drop=True
    )

    return unified


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    unified: pd.DataFrame,
    current_team_map: pd.DataFrame,
    fallback_clubs: set[str],
) -> pd.DataFrame:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT SQUAD POOL QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    expected_teams = set(
        current_team_map[
            "canonical_team"
        ]
    )

    represented_teams = set(
        unified[
            "canonical_team"
        ]
    )

    missing_teams = (
        expected_teams
        -
        represented_teams
    )

    extra_teams = (
        represented_teams
        -
        expected_teams
    )

    print(
        f"\nExpected UCL clubs: "
        f"{len(expected_teams)}"
    )

    print(
        f"Represented clubs: "
        f"{len(represented_teams)}"
    )

    print(
        f"Missing clubs: "
        f"{len(missing_teams)}"
    )

    print(
        f"Unexpected clubs: "
        f"{len(extra_teams)}"
    )

    if missing_teams:

        raise RuntimeError(
            f"Missing clubs from unified squad pool: "
            f"{sorted(missing_teams)}"
        )

    if extra_teams:

        raise RuntimeError(
            f"Unexpected clubs in unified squad pool: "
            f"{sorted(extra_teams)}"
        )

    # --------------------------------------------------------
    # SOURCE CHECK
    # --------------------------------------------------------

    football_data_teams = set(
        unified.loc[
            unified[
                "squad_source"
            ]
            ==
            "football_data_current_squad",
            "canonical_team",
        ]
    )

    fallback_teams = set(
        unified.loc[
            unified[
                "squad_source"
            ]
            ==
            "transfermarkt_current_club_fallback",
            "canonical_team",
        ]
    )

    print(
        f"\nFootball-data squad clubs: "
        f"{len(football_data_teams)}"
    )

    print(
        f"Transfermarkt fallback clubs: "
        f"{len(fallback_teams)}"
    )

    if (
        len(
            football_data_teams
        )
        !=
        EXPECTED_FOOTBALL_DATA_SQUAD_CLUBS
    ):

        raise RuntimeError(
            "Unexpected football-data squad club count."
        )

    if fallback_teams != fallback_clubs:

        raise RuntimeError(
            "Fallback club set changed unexpectedly."
        )

    # --------------------------------------------------------
    # PLAYER ID CONFLICT
    # --------------------------------------------------------

    matched_tm_ids = unified[
        unified[
            "transfermarkt_player_id"
        ].notna()
    ].copy()

    matched_tm_ids[
        "transfermarkt_player_id"
    ] = pd.to_numeric(
        matched_tm_ids[
            "transfermarkt_player_id"
        ],
        errors="coerce",
    )

    duplicated_tm = matched_tm_ids[
        matched_tm_ids[
            "transfermarkt_player_id"
        ].duplicated(
            keep=False
        )
    ]

    cross_team_conflicts = (
        duplicated_tm
        .groupby(
            "transfermarkt_player_id"
        )[
            "canonical_team"
        ]
        .nunique()
    )

    cross_team_conflicts = (
        cross_team_conflicts[
            cross_team_conflicts
            >
            1
        ]
    )

    print(
        f"Cross-team Transfermarkt player conflicts: "
        f"{len(cross_team_conflicts)}"
    )

    if len(
        cross_team_conflicts
    ):

        raise RuntimeError(
            "Transfermarkt player IDs appear in "
            "multiple current UCL clubs:\n"
            f"{list(cross_team_conflicts.index)}"
        )

    # --------------------------------------------------------
    # TEAM AUDIT
    # --------------------------------------------------------

    rows = []

    for team in sorted(
        expected_teams
    ):

        players = unified[
            unified[
                "canonical_team"
            ]
            ==
            team
        ]

        values = int(
            players[
                "market_value_eur"
            ].notna().sum()
        )

        confirmed = int(
            (
                players[
                    "confirmed_current_squad"
                ]
                ==
                1
            ).sum()
        )

        fallback = int(
            (
                players[
                    "transfermarkt_fallback_pool"
                ]
                ==
                1
            ).sum()
        )

        if confirmed:

            source_type = (
                "FOOTBALL_DATA_CURRENT_SQUAD"
            )

        elif fallback:

            source_type = (
                "TRANSFERMARKT_CURRENT_CLUB_FALLBACK"
            )

        else:

            source_type = (
                "UNKNOWN"
            )

        rows.append(
            {
                "canonical_team": (
                    team
                ),
                "player_pool_size": int(
                    len(
                        players
                    )
                ),
                "valued_players": (
                    values
                ),
                "valuation_coverage": (
                    values
                    /
                    len(
                        players
                    )
                    if len(
                        players
                    )
                    else 0.0
                ),
                "confirmed_squad_players": (
                    confirmed
                ),
                "fallback_pool_players": (
                    fallback
                ),
                "squad_source_type": (
                    source_type
                ),
            }
        )

    audit = pd.DataFrame(
        rows
    )

    audit = audit.sort_values(
        by=[
            "squad_source_type",
            "valuation_coverage",
            "canonical_team",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    total_values = int(
        unified[
            "market_value_eur"
        ].notna().sum()
    )

    print(
        f"\nUnified player rows: "
        f"{len(unified):,}"
    )

    print(
        f"Players with usable valuation: "
        f"{total_values:,}/{len(unified):,}"
    )

    print(
        f"Unified valuation coverage: "
        f"{total_values / len(unified) * 100:.2f}%"
    )

    return audit


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    unified: pd.DataFrame,
    audit: pd.DataFrame,
    fallback_clubs: set[str],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEAM_AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified.to_csv(
        PLAYER_POOL_FILE,
        index=False,
        encoding="utf-8",
    )

    audit.to_csv(
        TEAM_AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "season": (
            "2026/27"
        ),

        "ucl_clubs": (
            EXPECTED_UCL_CLUBS
        ),

        "football_data_squad_clubs": (
            EXPECTED_FOOTBALL_DATA_SQUAD_CLUBS
        ),

        "transfermarkt_fallback_clubs": sorted(
            fallback_clubs
        ),

        "player_rows": int(
            len(
                unified
            )
        ),

        "players_with_market_value": int(
            unified[
                "market_value_eur"
            ].notna().sum()
        ),

        "rules": {
            "football_data_squad_is_primary": True,

            "minimum_trusted_football_data_squad_size": (
                MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE
            ),

            "incomplete_football_data_squads_use_fallback": True,

            "partial_fallback_club_rows_excluded_from_primary_pool": True,

            "transfermarkt_current_club_is_fallback_only": True,

            "fallback_pool_does_not_imply_ucl_registration": True,

            "ucl_registration_confirmed_default": False,

            "historical_training_data_modified": False,

            "fuzzy_player_matching_used": False,
        },
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# SUMMARY
# ============================================================


def print_summary(
    *,
    unified: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 UNIFIED CURRENT SQUAD POOL SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nPlayer rows: "
        f"{len(unified):,}"
    )

    print(
        f"Clubs represented: "
        f"{unified['canonical_team'].nunique()}"
    )

    print(
        "\nSource distribution:"
    )

    print(
        unified[
            "squad_source"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTransfermarkt fallback clubs:"
    )

    fallback = audit[
        audit[
            "squad_source_type"
        ]
        ==
        "TRANSFERMARKT_CURRENT_CLUB_FALLBACK"
    ]

    print(
        fallback[
            [
                "canonical_team",
                "player_pool_size",
                "valued_players",
                "valuation_coverage",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nLowest valuation coverage:"
    )

    print(
        audit[
            [
                "canonical_team",
                "player_pool_size",
                "valued_players",
                "valuation_coverage",
                "squad_source_type",
            ]
        ]
        .sort_values(
            by=[
                "valuation_coverage",
                "canonical_team",
            ]
        )
        .head(
            15
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


# ============================================================
# MAIN
# ============================================================


def build_current_squad_pool() -> None:

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - BUILD CURRENT SQUAD POOL"
    )

    print(
        "=" * 75
    )

    current_team_map = (
        load_current_team_map()
    )

    team_snapshot = (
        load_team_snapshot()
    )

    club_map = (
        load_extended_club_map()
    )

    resolved_players = (
        load_resolved_current_players()
    )

    transfermarkt_players = (
        load_transfermarkt_players()
    )

    fallback_clubs = (
        identify_fallback_clubs(
            team_snapshot
        )
    )

    print(
        f"\nCurrent UCL clubs: "
        f"{len(current_team_map)}"
    )

    print(
        f"Resolved football-data players: "
        f"{len(resolved_players):,}"
    )

    print(
        f"Fallback clubs: "
        f"{len(fallback_clubs)}"
    )

    for club in sorted(
        fallback_clubs
    ):

        print(
            f"  {club}"
        )

    football_data_pool = (
        build_football_data_pool(
            resolved_players
        )
    )

    fallback_pool = (
        build_fallback_pool(
            fallback_clubs=fallback_clubs,
            current_team_map=current_team_map,
            club_map=club_map,
            transfermarkt_players=transfermarkt_players,
        )
    )

    fallback_pool = (
        add_fallback_valuations(
            fallback_pool
        )
    )

    fallback_pool = (
        standardize_fallback(
            fallback_pool
        )
    )

    unified = build_unified_pool(
            football_data_pool=football_data_pool,
            fallback_pool=fallback_pool,
            fallback_clubs=fallback_clubs,
        )

    audit = run_quality_checks(
        unified=unified,
        current_team_map=current_team_map,
        fallback_clubs=fallback_clubs,
    )

    save_outputs(
        unified=unified,
        audit=audit,
        fallback_clubs=fallback_clubs,
    )

    print_summary(
        unified=unified,
        audit=audit,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT SQUAD POOL BUILT SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nUnified current squad pool:"
    )

    print(
        PLAYER_POOL_FILE
    )

    print(
        "\nTeam coverage audit:"
    )

    print(
        TEAM_AUDIT_FILE
    )

    print(
        "\nMetadata:"
    )

    print(
        METADATA_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Transfermarkt fallback rows are current-club "
        "candidate pools, not confirmed UCL registrations."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    build_current_squad_pool()