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

CURRENT_PLAYER_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "current_player_transfermarkt_map.csv"
)

CANDIDATE_AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_unresolved_player_candidates.csv"
)

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
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

RESOLVED_PLAYER_MAP_FILE = (
    OUTPUT_DIR
    / "current_player_transfermarkt_map_resolved.csv"
)

RESOLVED_VALUATION_FILE = (
    OUTPUT_DIR
    / "current_player_local_valuations_resolved.csv"
)

RESOLVED_TEAM_AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_player_mapping_resolved_audit.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "current_player_gap_resolution_metadata.json"
)


# ============================================================
# APPROVAL POLICY
# ============================================================

APPROVED_CANDIDATE_CLASSES = {
    "A_EXACT_NAME_DOB_CURRENT_CLUB",
    "B_UNIQUE_DOB_CURRENT_CLUB_WITH_NAME_TOKEN",
}

EXPECTED_CURRENT_PLAYERS = 899


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
# LOAD CURRENT PLAYER MAP
# ============================================================


def load_current_player_map() -> pd.DataFrame:

    if not CURRENT_PLAYER_MAP_FILE.exists():

        raise FileNotFoundError(
            "Current player map not found:\n"
            f"{CURRENT_PLAYER_MAP_FILE}"
        )

    dataframe = pd.read_csv(
        CURRENT_PLAYER_MAP_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "football_data_player_id",
        "player_name",
        "date_of_birth",
        "transfermarkt_club_id",
        "mapping_status",
        "mapping_method",
        "candidate_count",
        "transfermarkt_player_id",
        "transfermarkt_player_name",
        "transfermarkt_current_club_id",
        "transfermarkt_current_club_name",
        "player_snapshot_market_value_eur",
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
            "Current player map missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_CURRENT_PLAYERS:

        raise RuntimeError(
            f"Expected {EXPECTED_CURRENT_PLAYERS} current "
            f"players, found {len(dataframe)}."
        )

    dataframe[
        "football_data_player_id"
    ] = pd.to_numeric(
        dataframe[
            "football_data_player_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "transfermarkt_club_id"
    ] = pd.to_numeric(
        dataframe[
            "transfermarkt_club_id"
        ],
        errors="coerce",
    )

    return dataframe


# ============================================================
# LOAD CANDIDATE AUDIT
# ============================================================


def load_candidate_audit() -> pd.DataFrame:

    if not CANDIDATE_AUDIT_FILE.exists():

        raise FileNotFoundError(
            "Player candidate audit not found:\n"
            f"{CANDIDATE_AUDIT_FILE}"
        )

    dataframe = pd.read_csv(
        CANDIDATE_AUDIT_FILE,
        low_memory=False,
    )

    required = {
        "football_data_player_id",
        "football_data_player_name",
        "candidate_transfermarkt_player_id",
        "candidate_transfermarkt_name",
        "candidate_current_club_id",
        "candidate_current_club_name",
        "same_current_club",
        "name_token_overlap_count",
        "candidate_class",
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
            "Candidate audit missing columns: "
            f"{sorted(missing)}"
        )

    dataframe[
        "football_data_player_id"
    ] = pd.to_numeric(
        dataframe[
            "football_data_player_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "candidate_transfermarkt_player_id"
    ] = pd.to_numeric(
        dataframe[
            "candidate_transfermarkt_player_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    return dataframe


# ============================================================
# LOAD CURRENT TEAMS
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

    return dataframe


# ============================================================
# LOAD TRANSFERMARKT PLAYER SNAPSHOT
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

    club_name_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "current_club_name",
            "club_name",
        ],
        description="current club name",
        required=False,
    )

    market_value_column = detect_column(
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
        "_current_club_id"
    ] = pd.to_numeric(
        dataframe[
            club_id_column
        ],
        errors="coerce",
    )

    if club_name_column is not None:

        dataframe[
            "_current_club_name"
        ] = dataframe[
            club_name_column
        ]

    else:

        dataframe[
            "_current_club_name"
        ] = pd.NA

    if market_value_column is not None:

        dataframe[
            "_snapshot_market_value"
        ] = pd.to_numeric(
            dataframe[
                market_value_column
            ],
            errors="coerce",
        )

    else:

        dataframe[
            "_snapshot_market_value"
        ] = np.nan

    duplicate_ids = int(
        dataframe[
            "_player_id"
        ].duplicated().sum()
    )

    if duplicate_ids:

        raise RuntimeError(
            "Duplicate Transfermarkt player IDs found "
            "inside players.csv."
        )

    return dataframe


# ============================================================
# CLEAR ONE MAPPING
# ============================================================


def clear_mapping(
    dataframe: pd.DataFrame,
    index,
    *,
    reason: str,
) -> None:

    dataframe.at[
        index,
        "mapping_status",
    ] = "NO_MATCH"

    dataframe.at[
        index,
        "mapping_method",
    ] = reason

    dataframe.at[
        index,
        "candidate_count",
    ] = 0

    dataframe.at[
        index,
        "transfermarkt_player_id",
    ] = np.nan

    dataframe.at[
        index,
        "transfermarkt_player_name",
    ] = pd.NA

    dataframe.at[
        index,
        "transfermarkt_current_club_id",
    ] = np.nan

    dataframe.at[
        index,
        "transfermarkt_current_club_name",
    ] = pd.NA

    dataframe.at[
        index,
        "player_snapshot_market_value_eur",
    ] = np.nan


# ============================================================
# RESOLVE SAFE DUPLICATE CONFLICTS
# ============================================================


def resolve_duplicate_conflicts(
    player_map: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
]:

    dataframe = player_map.copy()

    matched = dataframe[
        dataframe[
            "mapping_status"
        ]
        ==
        "MATCHED"
    ].copy()

    matched[
        "transfermarkt_player_id"
    ] = pd.to_numeric(
        matched[
            "transfermarkt_player_id"
        ],
        errors="coerce",
    )

    counts = (
        matched[
            "transfermarkt_player_id"
        ]
        .dropna()
        .value_counts()
    )

    duplicate_ids = list(
        counts[
            counts
            >
            1
        ].index
    )

    corrections = 0

    for transfermarkt_id in duplicate_ids:

        duplicate_group = dataframe[
            (
                dataframe[
                    "mapping_status"
                ]
                ==
                "MATCHED"
            )
            &
            (
                pd.to_numeric(
                    dataframe[
                        "transfermarkt_player_id"
                    ],
                    errors="coerce",
                )
                ==
                transfermarkt_id
            )
        ].copy()

        # ----------------------------------------------------
        # SAFE RESOLUTION RULE:
        #
        # Keep exactly one DOB-based mapping.
        #
        # Any competing row must:
        #   - lack DOB
        #   - have been mapped using EXACT_NAME_TEAM_ONLY
        #
        # This exactly covers the Pedrinho conflict while
        # refusing to silently resolve other duplicate types.
        # ----------------------------------------------------

        dob_based = duplicate_group[
            duplicate_group[
                "mapping_method"
            ]
            .fillna(
                ""
            )
            .str.contains(
                "DOB",
                regex=False,
            )
            &
            duplicate_group[
                "date_of_birth"
            ].notna()
        ]

        if len(
            dob_based
        ) != 1:

            raise RuntimeError(
                "\nUnsafe duplicate Transfermarkt mapping.\n"
                f"Transfermarkt player ID: "
                f"{int(transfermarkt_id)}\n"
                f"{duplicate_group.to_string(index=False)}"
            )

        keeper_index = (
            dob_based.index[
                0
            ]
        )

        removable_indices = [
            index
            for index in duplicate_group.index
            if index
            !=
            keeper_index
        ]

        for index in removable_indices:

            method = str(
                dataframe.at[
                    index,
                    "mapping_method",
                ]
            )

            dob = dataframe.at[
                index,
                "date_of_birth",
            ]

            if (
                method
                !=
                "EXACT_NAME_TEAM_ONLY"
                or
                pd.notna(
                    dob
                )
            ):

                raise RuntimeError(
                    "\nDuplicate mapping cannot be "
                    "resolved safely.\n"
                    f"Transfermarkt player ID: "
                    f"{int(transfermarkt_id)}\n"
                    f"{duplicate_group.to_string(index=False)}"
                )

            clear_mapping(
                dataframe,
                index,
                reason=(
                    "DUPLICATE_NAME_ONLY_MAPPING_REMOVED"
                ),
            )

            corrections += 1

    return (
        dataframe,
        corrections,
    )


# ============================================================
# APPROVE STRONG CANDIDATES
# ============================================================


def approve_strong_candidates(
    *,
    player_map: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    transfermarkt_players: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
]:

    dataframe = player_map.copy()

    approved = candidate_audit[
        candidate_audit[
            "candidate_class"
        ].isin(
            APPROVED_CANDIDATE_CLASSES
        )
    ].copy()

    # --------------------------------------------------------
    # EXTRA SAFETY CONDITIONS
    # --------------------------------------------------------

    approved = approved[
        pd.to_numeric(
            approved[
                "same_current_club"
            ],
            errors="coerce",
        )
        ==
        1
    ].copy()

    approved = approved[
        pd.to_numeric(
            approved[
                "name_token_overlap_count"
            ],
            errors="coerce",
        )
        >=
        1
    ].copy()

    counts = (
        approved[
            "football_data_player_id"
        ]
        .value_counts()
    )

    duplicate_candidate_players = list(
        counts[
            counts
            >
            1
        ].index
    )

    if duplicate_candidate_players:

        raise RuntimeError(
            "Some players have multiple approved strong "
            "candidates:\n"
            f"{duplicate_candidate_players}"
        )

    tm_lookup = (
        transfermarkt_players
        .set_index(
            "_player_id",
            drop=False,
        )
    )

    applied = 0

    for (
        _,
        candidate,
    ) in approved.iterrows():

        football_data_id = int(
            candidate[
                "football_data_player_id"
            ]
        )

        target_rows = dataframe[
            dataframe[
                "football_data_player_id"
            ]
            ==
            football_data_id
        ]

        if len(
            target_rows
        ) != 1:

            raise RuntimeError(
                f"Expected one current player row for "
                f"football-data player {football_data_id}, "
                f"found {len(target_rows)}."
            )

        index = target_rows.index[
            0
        ]

        if (
            dataframe.at[
                index,
                "mapping_status",
            ]
            ==
            "MATCHED"
        ):

            continue

        tm_player_id = int(
            candidate[
                "candidate_transfermarkt_player_id"
            ]
        )

        if tm_player_id not in tm_lookup.index:

            raise RuntimeError(
                f"Approved Transfermarkt player ID "
                f"{tm_player_id} not found in players.csv."
            )

        tm_player = tm_lookup.loc[
            tm_player_id
        ]

        if isinstance(
            tm_player,
            pd.DataFrame,
        ):

            raise RuntimeError(
                f"Transfermarkt player ID "
                f"{tm_player_id} is duplicated."
            )

        current_tm_club_id = int(
            dataframe.at[
                index,
                "transfermarkt_club_id",
            ]
        )

        candidate_tm_club_id = (
            int(
                tm_player[
                    "_current_club_id"
                ]
            )
            if pd.notna(
                tm_player[
                    "_current_club_id"
                ]
            )
            else None
        )

        if (
            candidate_tm_club_id
            !=
            current_tm_club_id
        ):

            raise RuntimeError(
                f"Current-club mismatch while approving "
                f"{dataframe.at[index, 'player_name']}."
            )

        dataframe.at[
            index,
            "mapping_status",
        ] = "MATCHED"

        dataframe.at[
            index,
            "mapping_method",
        ] = (
            "AUDIT_UNIQUE_DOB_CURRENT_CLUB_NAME_TOKEN"
        )

        dataframe.at[
            index,
            "candidate_count",
        ] = 1

        dataframe.at[
            index,
            "transfermarkt_player_id",
        ] = tm_player_id

        dataframe.at[
            index,
            "transfermarkt_player_name",
        ] = tm_player[
            "_name"
        ]

        dataframe.at[
            index,
            "transfermarkt_current_club_id",
        ] = candidate_tm_club_id

        dataframe.at[
            index,
            "transfermarkt_current_club_name",
        ] = tm_player[
            "_current_club_name"
        ]

        dataframe.at[
            index,
            "player_snapshot_market_value_eur",
        ] = tm_player[
            "_snapshot_market_value"
        ]

        applied += 1

    return (
        dataframe,
        applied,
    )


# ============================================================
# LOAD LATEST LOCAL VALUATIONS
# ============================================================


def load_latest_valuations(
    matched_player_ids: set[int],
) -> pd.DataFrame:

    if not TRANSFERMARKT_VALUATIONS_FILE.exists():

        raise FileNotFoundError(
            "Transfermarkt valuation history not found:\n"
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
        description="market value",
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
            matched_player_ids
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

    latest = latest[
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
                "latest_local_valuation_date"
            ),

            "_market_value": (
                "latest_local_market_value_eur"
            ),
        }
    )

    return latest


# ============================================================
# ATTACH VALUATIONS
# ============================================================


def attach_valuations(
    *,
    player_map: pd.DataFrame,
    latest_valuations: pd.DataFrame,
) -> pd.DataFrame:

    output = player_map.copy()

    output[
        "transfermarkt_player_id"
    ] = pd.to_numeric(
        output[
            "transfermarkt_player_id"
        ],
        errors="coerce",
    )

    output = output.merge(
        latest_valuations,
        on="transfermarkt_player_id",
        how="left",
        validate="many_to_one",
    )

    output[
        "usable_market_value_eur"
    ] = output[
        "latest_local_market_value_eur"
    ]

    output[
        "valuation_source"
    ] = pd.NA

    history_mask = output[
        "latest_local_market_value_eur"
    ].notna()

    output.loc[
        history_mask,
        "valuation_source",
    ] = (
        "player_valuations_history_latest"
    )

    fallback_mask = (
        output[
            "usable_market_value_eur"
        ].isna()
        &
        output[
            "player_snapshot_market_value_eur"
        ].notna()
    )

    output.loc[
        fallback_mask,
        "usable_market_value_eur",
    ] = output.loc[
        fallback_mask,
        "player_snapshot_market_value_eur",
    ]

    output.loc[
        fallback_mask,
        "valuation_source",
    ] = "players_csv_snapshot"

    return output


# ============================================================
# TEAM AUDIT
# ============================================================


def build_team_audit(
    *,
    current_teams: pd.DataFrame,
    valuation_map: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        _,
        team,
    ) in current_teams.iterrows():

        team_name = str(
            team[
                "canonical_team"
            ]
        )

        players = valuation_map[
            valuation_map[
                "canonical_team"
            ]
            ==
            team_name
        ]

        squad_size = len(
            players
        )

        mapped = int(
            (
                players[
                    "mapping_status"
                ]
                ==
                "MATCHED"
            ).sum()
        )

        valued = int(
            players[
                "usable_market_value_eur"
            ].notna().sum()
        )

        rows.append(
            {
                "canonical_team": (
                    team_name
                ),

                "identity_status": (
                    team[
                        "identity_status"
                    ]
                ),

                "squad_size": (
                    squad_size
                ),

                "mapped_players": (
                    mapped
                ),

                "mapping_coverage": (
                    mapped
                    /
                    squad_size
                    if squad_size
                    else 0.0
                ),

                "valued_players": (
                    valued
                ),

                "valuation_coverage": (
                    valued
                    /
                    squad_size
                    if squad_size
                    else 0.0
                ),

                "unresolved_players": (
                    squad_size
                    -
                    mapped
                ),

                "squad_missing": int(
                    squad_size
                    ==
                    0
                ),

                "squad_market_value_eur": (
                    float(
                        players[
                            "usable_market_value_eur"
                        ].sum()
                    )
                    if valued
                    else None
                ),
            }
        )

    audit = pd.DataFrame(
        rows
    )

    audit = audit.sort_values(
        by=[
            "squad_missing",
            "mapping_coverage",
            "valuation_coverage",
            "canonical_team",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return audit


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    original_map: pd.DataFrame,
    resolved_map: pd.DataFrame,
    valuation_map: pd.DataFrame,
    team_audit: pd.DataFrame,
    strong_candidates_applied: int,
    duplicate_corrections: int,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RESOLVED CURRENT PLAYER QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    original_matched = int(
        (
            original_map[
                "mapping_status"
            ]
            ==
            "MATCHED"
        ).sum()
    )

    resolved_matched = int(
        (
            resolved_map[
                "mapping_status"
            ]
            ==
            "MATCHED"
        ).sum()
    )

    unresolved = (
        len(
            resolved_map
        )
        -
        resolved_matched
    )

    valued = int(
        valuation_map[
            "usable_market_value_eur"
        ].notna().sum()
    )

    print(
        f"\nOriginal matched players: "
        f"{original_matched:,}"
    )

    print(
        f"Strong candidates applied: "
        f"{strong_candidates_applied:,}"
    )

    print(
        f"Unsafe duplicate mappings removed: "
        f"{duplicate_corrections:,}"
    )

    print(
        f"Final matched players: "
        f"{resolved_matched:,}"
    )

    print(
        f"Final unresolved players: "
        f"{unresolved:,}"
    )

    print(
        f"Players with usable valuation: "
        f"{valued:,}"
    )

    print(
        f"\nFinal player mapping coverage: "
        f"{resolved_matched / len(resolved_map) * 100:.2f}%"
    )

    print(
        f"Final valuation coverage: "
        f"{valued / len(resolved_map) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    matched = resolved_map[
        resolved_map[
            "mapping_status"
        ]
        ==
        "MATCHED"
    ].copy()

    matched[
        "transfermarkt_player_id"
    ] = pd.to_numeric(
        matched[
            "transfermarkt_player_id"
        ],
        errors="coerce",
    )

    duplicate_ids = int(
        matched[
            "transfermarkt_player_id"
        ]
        .dropna()
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate matched Transfermarkt IDs: "
        f"{duplicate_ids}"
    )

    if duplicate_ids:

        raise RuntimeError(
            "Duplicate Transfermarkt player IDs remain "
            "after resolution."
        )

    # --------------------------------------------------------
    # TEAM COVERAGE
    # --------------------------------------------------------

    print(
        "\nLowest team coverage after resolution:"
    )

    print(
        team_audit[
            [
                "canonical_team",
                "identity_status",
                "squad_size",
                "mapped_players",
                "mapping_coverage",
                "valued_players",
                "valuation_coverage",
                "squad_missing",
            ]
        ]
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
# SAVE
# ============================================================


def save_outputs(
    *,
    resolved_map: pd.DataFrame,
    valuation_map: pd.DataFrame,
    team_audit: pd.DataFrame,
    strong_candidates_applied: int,
    duplicate_corrections: int,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESOLVED_TEAM_AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_map.to_csv(
        RESOLVED_PLAYER_MAP_FILE,
        index=False,
        encoding="utf-8",
    )

    valuation_output = valuation_map.copy()

    if (
        "latest_local_valuation_date"
        in
        valuation_output.columns
    ):

        valuation_output[
            "latest_local_valuation_date"
        ] = pd.to_datetime(
            valuation_output[
                "latest_local_valuation_date"
            ],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d"
        )

    valuation_output.to_csv(
        RESOLVED_VALUATION_FILE,
        index=False,
        encoding="utf-8",
    )

    team_audit.to_csv(
        RESOLVED_TEAM_AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    matched = int(
        (
            resolved_map[
                "mapping_status"
            ]
            ==
            "MATCHED"
        ).sum()
    )

    valued = int(
        valuation_map[
            "usable_market_value_eur"
        ].notna().sum()
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "player_rows": int(
            len(
                resolved_map
            )
        ),

        "strong_candidate_classes_approved": sorted(
            APPROVED_CANDIDATE_CLASSES
        ),

        "strong_candidates_applied": int(
            strong_candidates_applied
        ),

        "duplicate_mappings_removed": int(
            duplicate_corrections
        ),

        "final_matched_players": (
            matched
        ),

        "final_mapping_coverage": (
            matched
            /
            len(
                resolved_map
            )
        ),

        "final_valued_players": (
            valued
        ),

        "final_valuation_coverage": (
            valued
            /
            len(
                valuation_map
            )
        ),

        "rules": {
            "fuzzy_matching_used": False,

            "unique_dob_current_club_required": True,

            "name_token_overlap_required": True,

            "weak_C_candidates_not_auto_approved": True,

            "duplicate_name_only_mapping_removed_when_dob_mapping_exists": True,

            "historical_training_data_modified": False,
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
    resolved_map: pd.DataFrame,
    valuation_map: pd.DataFrame,
    team_audit: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 RESOLVED PLAYER IDENTITY SUMMARY"
    )

    print(
        "=" * 75
    )

    matched = int(
        (
            resolved_map[
                "mapping_status"
            ]
            ==
            "MATCHED"
        ).sum()
    )

    valued = int(
        valuation_map[
            "usable_market_value_eur"
        ].notna().sum()
    )

    print(
        f"\nMapped players: "
        f"{matched:,}/{len(resolved_map):,}"
    )

    print(
        f"Players with usable valuation: "
        f"{valued:,}/{len(resolved_map):,}"
    )

    print(
        "\nMapping methods:"
    )

    print(
        resolved_map[
            "mapping_method"
        ]
        .fillna(
            "UNRESOLVED"
        )
        .value_counts()
        .to_string()
    )

    print(
        "\nNew 2026/27 clubs:"
    )

    new_clubs = team_audit[
        team_audit[
            "identity_status"
        ]
        ==
        "NEW_2026_27"
    ]

    print(
        new_clubs[
            [
                "canonical_team",
                "squad_size",
                "mapped_players",
                "mapping_coverage",
                "valued_players",
                "valuation_coverage",
                "squad_market_value_eur",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    unresolved = resolved_map[
        resolved_map[
            "mapping_status"
        ]
        !=
        "MATCHED"
    ]

    print(
        "\nFirst 30 remaining unresolved players:"
    )

    if unresolved.empty:

        print(
            "  None ✓"
        )

    else:

        print(
            unresolved[
                [
                    "canonical_team",
                    "football_data_player_id",
                    "player_name",
                    "date_of_birth",
                    "mapping_method",
                ]
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================


def resolve_current_player_gaps() -> None:

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - RESOLVE CURRENT PLAYER GAPS"
    )

    print(
        "=" * 75
    )

    original_map = (
        load_current_player_map()
    )

    candidate_audit = (
        load_candidate_audit()
    )

    current_teams = (
        load_current_team_map()
    )

    transfermarkt_players = (
        load_transfermarkt_players()
    )

    print(
        f"\nCurrent player rows: "
        f"{len(original_map):,}"
    )

    print(
        f"Candidate audit rows: "
        f"{len(candidate_audit):,}"
    )

    print(
        f"Transfermarkt players: "
        f"{len(transfermarkt_players):,}"
    )

    # --------------------------------------------------------
    # STEP 1
    # REMOVE ONLY SAFE DUPLICATE CONFLICTS
    # --------------------------------------------------------

    (
        cleaned_map,
        duplicate_corrections,
    ) = resolve_duplicate_conflicts(
        original_map
    )

    # --------------------------------------------------------
    # STEP 2
    # APPLY STRONG DETERMINISTIC CANDIDATES
    # --------------------------------------------------------

    (
        resolved_map,
        strong_candidates_applied,
    ) = approve_strong_candidates(
        player_map=cleaned_map,
        candidate_audit=candidate_audit,
        transfermarkt_players=transfermarkt_players,
    )

    # --------------------------------------------------------
    # STEP 3
    # RELOAD VALUATIONS FOR ALL MATCHED PLAYERS
    # --------------------------------------------------------

    matched_ids = set(
        pd.to_numeric(
            resolved_map.loc[
                resolved_map[
                    "mapping_status"
                ]
                ==
                "MATCHED",
                "transfermarkt_player_id",
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
    )

    latest_valuations = (
        load_latest_valuations(
            matched_ids
        )
    )

    valuation_map = attach_valuations(
        player_map=resolved_map,
        latest_valuations=latest_valuations,
    )

    team_audit = build_team_audit(
        current_teams=current_teams,
        valuation_map=valuation_map,
    )

    run_quality_checks(
        original_map=original_map,
        resolved_map=resolved_map,
        valuation_map=valuation_map,
        team_audit=team_audit,
        strong_candidates_applied=strong_candidates_applied,
        duplicate_corrections=duplicate_corrections,
    )

    save_outputs(
        resolved_map=resolved_map,
        valuation_map=valuation_map,
        team_audit=team_audit,
        strong_candidates_applied=strong_candidates_applied,
        duplicate_corrections=duplicate_corrections,
    )

    print_summary(
        resolved_map=resolved_map,
        valuation_map=valuation_map,
        team_audit=team_audit,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT PLAYER GAPS RESOLVED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nResolved player map:"
    )

    print(
        RESOLVED_PLAYER_MAP_FILE
    )

    print(
        "\nResolved valuation snapshot:"
    )

    print(
        RESOLVED_VALUATION_FILE
    )

    print(
        "\nResolved team audit:"
    )

    print(
        RESOLVED_TEAM_AUDIT_FILE
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
        "Weak candidates were not automatically approved."
    )

    print(
        "Historical training data remains unchanged."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    resolve_current_player_gaps()