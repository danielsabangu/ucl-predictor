from __future__ import annotations

import json
import re
import unicodedata
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

CURRENT_SQUADS_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "football_data_ucl_squads.csv"
)

CURRENT_UCL_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

CURRENT_TRANSFERMARKT_CLUB_MAP_FILE = (
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

PLAYER_MAP_FILE = (
    OUTPUT_DIR
    / "current_player_transfermarkt_map.csv"
)

VALUATION_OUTPUT_FILE = (
    OUTPUT_DIR
    / "current_player_local_valuations.csv"
)

AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_player_transfermarkt_mapping_audit.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "current_player_transfermarkt_mapping_metadata.json"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_UCL_CLUBS = 36

EXPECTED_EXTENDED_TRANSFERMARKT_CLUBS = 123


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(
    value,
) -> str:
    """
    Deterministic text normalization.

    This is NOT fuzzy matching.
    """

    if pd.isna(
        value
    ):

        return ""

    text = str(
        value
    ).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
    )

    replacements = {
        "ø": "o",
        "đ": "d",
        "ı": "i",
        "ł": "l",
        "ß": "ss",
        "æ": "ae",
        "œ": "oe",
    }

    for (
        source,
        target,
    ) in replacements.items():

        text = text.replace(
            source,
            target,
        )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# DATE NORMALIZATION
# ============================================================


def normalize_date(
    value,
):
    """
    Convert DOB to a comparable date value.
    """

    if pd.isna(
        value
    ):

        return pd.NaT

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(
        parsed
    ):

        return pd.NaT

    return parsed.normalize()


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
    """
    Detect a column across compatible Transfermarkt schemas.
    """

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
# LOAD CURRENT SQUADS
# ============================================================


def load_current_squads() -> pd.DataFrame:
    """
    Load football-data.org current roster snapshot.
    """

    if not CURRENT_SQUADS_FILE.exists():

        raise FileNotFoundError(
            "Current squad snapshot not found:\n"
            f"{CURRENT_SQUADS_FILE}"
        )

    dataframe = pd.read_csv(
        CURRENT_SQUADS_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "football_data_team_id",
        "football_data_player_id",
        "player_name",
        "first_name",
        "last_name",
        "date_of_birth",
        "position",
        "nationality",
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
            "Current squad file missing columns: "
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
        "normalized_player_name"
    ] = dataframe[
        "player_name"
    ].map(
        normalize_text
    )

    dataframe[
        "normalized_first_last"
    ] = (
        dataframe[
            "first_name"
        ].fillna(
            ""
        ).astype(
            str
        )
        +
        " "
        +
        dataframe[
            "last_name"
        ].fillna(
            ""
        ).astype(
            str
        )
    ).map(
        normalize_text
    )

    dataframe[
        "normalized_last_name"
    ] = dataframe[
        "last_name"
    ].map(
        normalize_text
    )

    dataframe[
        "normalized_dob"
    ] = dataframe[
        "date_of_birth"
    ].map(
        normalize_date
    )

    duplicate_player_ids = int(
        dataframe[
            "football_data_player_id"
        ].duplicated().sum()
    )

    if duplicate_player_ids:

        raise RuntimeError(
            "Duplicate football-data player IDs "
            "found in current squads."
        )

    return dataframe


# ============================================================
# LOAD CURRENT UCL TEAMS
# ============================================================


def load_current_ucl_teams() -> pd.DataFrame:
    """
    Load all 36 current UCL clubs.
    """

    if not CURRENT_UCL_TEAM_MAP_FILE.exists():

        raise FileNotFoundError(
            "Current UCL team map not found:\n"
            f"{CURRENT_UCL_TEAM_MAP_FILE}"
        )

    dataframe = pd.read_csv(
        CURRENT_UCL_TEAM_MAP_FILE,
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
            "Current UCL map missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_UCL_CLUBS:

        raise RuntimeError(
            f"Expected {EXPECTED_UCL_CLUBS} "
            f"current clubs, found "
            f"{len(dataframe)}."
        )

    return dataframe


# ============================================================
# LOAD CURRENT TRANSFERMARKT CLUB MAP
# ============================================================


def load_transfermarkt_club_map() -> pd.DataFrame:
    """
    Load the new 123-club Transfermarkt identity map.
    """

    if not CURRENT_TRANSFERMARKT_CLUB_MAP_FILE.exists():

        raise FileNotFoundError(
            "Extended Transfermarkt club map not found:\n"
            f"{CURRENT_TRANSFERMARKT_CLUB_MAP_FILE}"
        )

    dataframe = pd.read_csv(
        CURRENT_TRANSFERMARKT_CLUB_MAP_FILE,
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

    if len(
        dataframe
    ) != EXPECTED_EXTENDED_TRANSFERMARKT_CLUBS:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_EXTENDED_TRANSFERMARKT_CLUBS} "
            f"Transfermarkt clubs, found "
            f"{len(dataframe)}."
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
# ADD CURRENT TEAM TRANSFERMARKT IDS
# ============================================================


def attach_transfermarkt_team_ids(
    *,
    squads: pd.DataFrame,
    club_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach current club Transfermarkt ID to each roster player.
    """

    team_map = club_map[
        [
            "canonical_team",
            "transfermarkt_club_id",
        ]
    ].drop_duplicates(
        subset=[
            "canonical_team",
        ]
    )

    dataframe = squads.merge(
        team_map,
        on="canonical_team",
        how="left",
        validate="many_to_one",
    )

    missing_ids = int(
        dataframe[
            "transfermarkt_club_id"
        ].isna().sum()
    )

    if missing_ids:

        missing_teams = sorted(
            dataframe.loc[
                dataframe[
                    "transfermarkt_club_id"
                ].isna(),
                "canonical_team",
            ].unique()
        )

        raise RuntimeError(
            "Missing current Transfermarkt club IDs "
            f"for: {missing_teams}"
        )

    dataframe[
        "transfermarkt_club_id"
    ] = dataframe[
        "transfermarkt_club_id"
    ].astype(
        int
    )

    return dataframe


# ============================================================
# LOAD TRANSFERMARKT PLAYERS
# ============================================================


def load_transfermarkt_players() -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Load full local Transfermarkt player snapshot.
    """

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
        description="Transfermarkt player ID",
    )

    name_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "name",
            "player_name",
        ],
        description="Transfermarkt player name",
    )

    dob_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "date_of_birth",
            "birth_date",
        ],
        description="Transfermarkt date of birth",
    )

    current_club_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "current_club_id",
            "club_id",
        ],
        description="Transfermarkt current club ID",
    )

    first_name_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "first_name",
        ],
        description="first name",
        required=False,
    )

    last_name_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "last_name",
        ],
        description="last name",
        required=False,
    )

    snapshot_value_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "market_value_in_eur",
            "market_value",
        ],
        description="player snapshot market value",
        required=False,
    )

    current_club_name_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "current_club_name",
            "club_name",
        ],
        description="current club name",
        required=False,
    )

    dataframe[
        "_tm_player_id"
    ] = pd.to_numeric(
        dataframe[
            player_id_column
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "_tm_name"
    ] = dataframe[
        name_column
    ].astype(
        str
    )

    dataframe[
        "_tm_normalized_name"
    ] = dataframe[
        name_column
    ].map(
        normalize_text
    )

    dataframe[
        "_tm_dob"
    ] = dataframe[
        dob_column
    ].map(
        normalize_date
    )

    dataframe[
        "_tm_current_club_id"
    ] = pd.to_numeric(
        dataframe[
            current_club_column
        ],
        errors="coerce",
    )

    if (
        first_name_column is not None
        and
        last_name_column is not None
    ):

        dataframe[
            "_tm_first_last"
        ] = (
            dataframe[
                first_name_column
            ].fillna(
                ""
            ).astype(
                str
            )
            +
            " "
            +
            dataframe[
                last_name_column
            ].fillna(
                ""
            ).astype(
                str
            )
        ).map(
            normalize_text
        )

        dataframe[
            "_tm_last_name"
        ] = dataframe[
            last_name_column
        ].map(
            normalize_text
        )

    else:

        dataframe[
            "_tm_first_last"
        ] = ""

        dataframe[
            "_tm_last_name"
        ] = ""

    if snapshot_value_column is not None:

        dataframe[
            "_tm_snapshot_value"
        ] = pd.to_numeric(
            dataframe[
                snapshot_value_column
            ],
            errors="coerce",
        )

    else:

        dataframe[
            "_tm_snapshot_value"
        ] = np.nan

    if current_club_name_column is not None:

        dataframe[
            "_tm_current_club_name"
        ] = dataframe[
            current_club_name_column
        ]

    else:

        dataframe[
            "_tm_current_club_name"
        ] = pd.NA

    schema = {
        "player_id_column": (
            player_id_column
        ),

        "name_column": (
            name_column
        ),

        "dob_column": (
            dob_column
        ),

        "current_club_column": (
            current_club_column
        ),

        "snapshot_value_column": (
            snapshot_value_column
        ),

        "current_club_name_column": (
            current_club_name_column
        ),
    }

    return (
        dataframe,
        schema,
    )


# ============================================================
# MATCH HELPERS
# ============================================================


def filter_name_dob(
    *,
    players: pd.DataFrame,
    normalized_name: str,
    normalized_dob,
) -> pd.DataFrame:
    """
    Exact normalized name + exact DOB.
    """

    if (
        not normalized_name
        or
        pd.isna(
            normalized_dob
        )
    ):

        return players.iloc[
            0:0
        ]

    return players[
        (
            players[
                "_tm_normalized_name"
            ]
            ==
            normalized_name
        )
        &
        (
            players[
                "_tm_dob"
            ]
            ==
            normalized_dob
        )
    ]


def filter_first_last_dob(
    *,
    players: pd.DataFrame,
    normalized_first_last: str,
    normalized_dob,
) -> pd.DataFrame:
    """
    Exact first+last name + exact DOB.
    """

    if (
        not normalized_first_last
        or
        pd.isna(
            normalized_dob
        )
    ):

        return players.iloc[
            0:0
        ]

    return players[
        (
            players[
                "_tm_first_last"
            ]
            ==
            normalized_first_last
        )
        &
        (
            players[
                "_tm_dob"
            ]
            ==
            normalized_dob
        )
    ]


# ============================================================
# RESOLVE ONE PLAYER
# ============================================================


def resolve_player(
    *,
    current_player,
    transfermarkt_players: pd.DataFrame,
) -> dict:
    """
    Resolve one player using deterministic identity rules.

    No fuzzy matching.
    """

    current_team_tm_id = int(
        current_player.transfermarkt_club_id
    )

    normalized_name = str(
        current_player.normalized_player_name
    )

    normalized_first_last = str(
        current_player.normalized_first_last
    )

    normalized_last_name = str(
        current_player.normalized_last_name
    )

    normalized_dob = (
        current_player.normalized_dob
    )

    # --------------------------------------------------------
    # RULE 1
    # FULL NAME + DOB
    # --------------------------------------------------------

    candidates = filter_name_dob(
        players=transfermarkt_players,
        normalized_name=normalized_name,
        normalized_dob=normalized_dob,
    )

    if len(
        candidates
    ) == 1:

        candidate = candidates.iloc[
            0
        ]

        same_club = (
            pd.notna(
                candidate[
                    "_tm_current_club_id"
                ]
            )
            and
            int(
                candidate[
                    "_tm_current_club_id"
                ]
            )
            ==
            current_team_tm_id
        )

        return build_match_result(
            candidate=candidate,
            method=(
                "EXACT_NAME_DOB_TEAM"
                if same_club
                else
                "EXACT_NAME_DOB_GLOBAL"
            ),
            candidate_count=1,
        )

    if len(
        candidates
    ) > 1:

        club_candidates = candidates[
            pd.to_numeric(
                candidates[
                    "_tm_current_club_id"
                ],
                errors="coerce",
            )
            ==
            current_team_tm_id
        ]

        if len(
            club_candidates
        ) == 1:

            return build_match_result(
                candidate=club_candidates.iloc[
                    0
                ],
                method="EXACT_NAME_DOB_TEAM",
                candidate_count=len(
                    candidates
                ),
            )

    # --------------------------------------------------------
    # RULE 2
    # FIRST + LAST + DOB
    # --------------------------------------------------------

    candidates = filter_first_last_dob(
        players=transfermarkt_players,
        normalized_first_last=normalized_first_last,
        normalized_dob=normalized_dob,
    )

    if len(
        candidates
    ) == 1:

        candidate = candidates.iloc[
            0
        ]

        same_club = (
            pd.notna(
                candidate[
                    "_tm_current_club_id"
                ]
            )
            and
            int(
                candidate[
                    "_tm_current_club_id"
                ]
            )
            ==
            current_team_tm_id
        )

        return build_match_result(
            candidate=candidate,
            method=(
                "EXACT_FIRST_LAST_DOB_TEAM"
                if same_club
                else
                "EXACT_FIRST_LAST_DOB_GLOBAL"
            ),
            candidate_count=1,
        )

    if len(
        candidates
    ) > 1:

        club_candidates = candidates[
            pd.to_numeric(
                candidates[
                    "_tm_current_club_id"
                ],
                errors="coerce",
            )
            ==
            current_team_tm_id
        ]

        if len(
            club_candidates
        ) == 1:

            return build_match_result(
                candidate=club_candidates.iloc[
                    0
                ],
                method="EXACT_FIRST_LAST_DOB_TEAM",
                candidate_count=len(
                    candidates
                ),
            )

    # --------------------------------------------------------
    # RULE 3
    # NAME + CURRENT CLUB, ONLY IF UNIQUE
    # --------------------------------------------------------

    if normalized_name:

        club_name_candidates = transfermarkt_players[
            (
                transfermarkt_players[
                    "_tm_normalized_name"
                ]
                ==
                normalized_name
            )
            &
            (
                pd.to_numeric(
                    transfermarkt_players[
                        "_tm_current_club_id"
                    ],
                    errors="coerce",
                )
                ==
                current_team_tm_id
            )
        ]

        if len(
            club_name_candidates
        ) == 1:

            return build_match_result(
                candidate=club_name_candidates.iloc[
                    0
                ],
                method="EXACT_NAME_TEAM_ONLY",
                candidate_count=1,
            )

    # --------------------------------------------------------
    # RULE 4
    # LAST NAME + DOB + CURRENT CLUB
    # --------------------------------------------------------

    if (
        normalized_last_name
        and
        pd.notna(
            normalized_dob
        )
    ):

        last_name_candidates = transfermarkt_players[
            (
                transfermarkt_players[
                    "_tm_last_name"
                ]
                ==
                normalized_last_name
            )
            &
            (
                transfermarkt_players[
                    "_tm_dob"
                ]
                ==
                normalized_dob
            )
            &
            (
                pd.to_numeric(
                    transfermarkt_players[
                        "_tm_current_club_id"
                    ],
                    errors="coerce",
                )
                ==
                current_team_tm_id
            )
        ]

        if len(
            last_name_candidates
        ) == 1:

            return build_match_result(
                candidate=last_name_candidates.iloc[
                    0
                ],
                method="EXACT_LAST_DOB_TEAM",
                candidate_count=1,
            )

    # --------------------------------------------------------
    # UNRESOLVED / AMBIGUOUS
    # --------------------------------------------------------

    possible_name_dob = filter_name_dob(
        players=transfermarkt_players,
        normalized_name=normalized_name,
        normalized_dob=normalized_dob,
    )

    status = (
        "AMBIGUOUS"
        if len(
            possible_name_dob
        )
        >
        1
        else
        "NO_MATCH"
    )

    return {
        "mapping_status": (
            status
        ),

        "mapping_method": (
            None
        ),

        "candidate_count": int(
            len(
                possible_name_dob
            )
        ),

        "transfermarkt_player_id": (
            None
        ),

        "transfermarkt_player_name": (
            None
        ),

        "transfermarkt_current_club_id": (
            None
        ),

        "transfermarkt_current_club_name": (
            None
        ),

        "player_snapshot_market_value_eur": (
            None
        ),
    }


# ============================================================
# BUILD MATCH RESULT
# ============================================================


def build_match_result(
    *,
    candidate: pd.Series,
    method: str,
    candidate_count: int,
) -> dict:
    """
    Standard matched-player result.
    """

    return {
        "mapping_status": (
            "MATCHED"
        ),

        "mapping_method": (
            method
        ),

        "candidate_count": int(
            candidate_count
        ),

        "transfermarkt_player_id": int(
            candidate[
                "_tm_player_id"
            ]
        ),

        "transfermarkt_player_name": (
            candidate[
                "_tm_name"
            ]
        ),

        "transfermarkt_current_club_id": (
            int(
                candidate[
                    "_tm_current_club_id"
                ]
            )
            if pd.notna(
                candidate[
                    "_tm_current_club_id"
                ]
            )
            else None
        ),

        "transfermarkt_current_club_name": (
            candidate[
                "_tm_current_club_name"
            ]
        ),

        "player_snapshot_market_value_eur": (
            float(
                candidate[
                    "_tm_snapshot_value"
                ]
            )
            if pd.notna(
                candidate[
                    "_tm_snapshot_value"
                ]
            )
            else None
        ),
    }


# ============================================================
# MAP ALL PLAYERS
# ============================================================


def map_current_players(
    *,
    current_players: pd.DataFrame,
    transfermarkt_players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build deterministic football-data -> Transfermarkt map.
    """

    rows = []

    total = len(
        current_players
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "MAPPING CURRENT PLAYERS TO TRANSFERMARKT"
    )

    print(
        "=" * 75
    )

    for index, player in enumerate(
        current_players.itertuples(
            index=False
        ),
        start=1,
    ):

        result = resolve_player(
            current_player=player,
            transfermarkt_players=transfermarkt_players,
        )

        rows.append(
            {
                "canonical_team": (
                    player.canonical_team
                ),

                "football_data_team_id": (
                    player.football_data_team_id
                ),

                "transfermarkt_club_id": (
                    player.transfermarkt_club_id
                ),

                "football_data_player_id": (
                    player.football_data_player_id
                ),

                "player_name": (
                    player.player_name
                ),

                "first_name": (
                    player.first_name
                ),

                "last_name": (
                    player.last_name
                ),

                "date_of_birth": (
                    player.date_of_birth
                ),

                "position": (
                    player.position
                ),

                "nationality": (
                    player.nationality
                ),

                **result,
            }
        )

        if (
            index
            %
            100
            ==
            0
            or
            index
            ==
            total
        ):

            print(
                f"Processed "
                f"{index:,}/{total:,}"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOAD VALUATIONS
# ============================================================


def load_latest_valuations(
    matched_player_ids: set[int],
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Load latest locally available Transfermarkt valuation for
    every matched player.

    This is for LIVE 2026/27 use only.
    """

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
        "_market_value_eur"
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
            "_market_value_eur"
        ].notna()
        &
        (
            dataframe[
                "_market_value_eur"
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

    latest = dataframe.groupby(
        "_player_id",
        as_index=False,
    ).tail(
        1
    )

    latest = latest[
        [
            "_player_id",
            "_valuation_date",
            "_market_value_eur",
        ]
    ].rename(
        columns={
            "_player_id": (
                "transfermarkt_player_id"
            ),

            "_valuation_date": (
                "latest_local_valuation_date"
            ),

            "_market_value_eur": (
                "latest_local_market_value_eur"
            ),
        }
    )

    metadata = {
        "valuation_rows_scanned": int(
            len(
                dataframe
            )
        ),

        "latest_valuation_date": (
            latest[
                "latest_local_valuation_date"
            ].max()
            if not latest.empty
            else None
        ),
    }

    return (
        latest,
        metadata,
    )


# ============================================================
# ATTACH VALUATIONS
# ============================================================


def attach_valuations(
    *,
    player_map: pd.DataFrame,
    latest_valuations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add latest historical valuation and preserve snapshot value
    separately.

    No silent source substitution.
    """

    output = player_map.merge(
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
    ] = np.where(
        output[
            "latest_local_market_value_eur"
        ].notna(),
        "player_valuations_history_latest",
        pd.NA,
    )

    # --------------------------------------------------------
    # LIVE-ONLY FALLBACK
    # --------------------------------------------------------

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
# AUDIT
# ============================================================


def build_team_audit(
    *,
    current_teams: pd.DataFrame,
    valuation_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Team-level player mapping and valuation coverage.
    """

    rows = []

    for team in current_teams.itertuples(
        index=False
    ):

        team_name = str(
            team.canonical_team
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

        ambiguous = int(
            (
                players[
                    "mapping_status"
                ]
                ==
                "AMBIGUOUS"
            ).sum()
        )

        no_match = int(
            (
                players[
                    "mapping_status"
                ]
                ==
                "NO_MATCH"
            ).sum()
        )

        rows.append(
            {
                "canonical_team": (
                    team_name
                ),

                "identity_status": (
                    team.identity_status
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

                "ambiguous_players": (
                    ambiguous
                ),

                "unmatched_players": (
                    no_match
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
# QUALITY CHECK
# ============================================================


def run_quality_checks(
    *,
    player_map: pd.DataFrame,
    valuation_map: pd.DataFrame,
    team_audit: pd.DataFrame,
) -> None:
    """
    Validate mapping integrity.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT PLAYER / TRANSFERMARKT QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        f"\nCurrent roster players: "
        f"{len(player_map):,}"
    )

    matched = int(
        (
            player_map[
                "mapping_status"
            ]
            ==
            "MATCHED"
        ).sum()
    )

    ambiguous = int(
        (
            player_map[
                "mapping_status"
            ]
            ==
            "AMBIGUOUS"
        ).sum()
    )

    unmatched = int(
        (
            player_map[
                "mapping_status"
            ]
            ==
            "NO_MATCH"
        ).sum()
    )

    valued = int(
        valuation_map[
            "usable_market_value_eur"
        ].notna().sum()
    )

    print(
        f"Matched players: "
        f"{matched:,}"
    )

    print(
        f"Ambiguous players: "
        f"{ambiguous:,}"
    )

    print(
        f"Unmatched players: "
        f"{unmatched:,}"
    )

    print(
        f"Players with usable valuation: "
        f"{valued:,}"
    )

    if len(
        player_map
    ):

        print(
            f"Overall player mapping coverage: "
            f"{matched / len(player_map) * 100:.2f}%"
        )

        print(
            f"Overall valuation coverage: "
            f"{valued / len(player_map) * 100:.2f}%"
        )

    duplicate_tm_ids = (
        player_map[
            player_map[
                "mapping_status"
            ]
            ==
            "MATCHED"
        ][
            "transfermarkt_player_id"
        ]
        .dropna()
        .duplicated()
    )

    duplicate_count = int(
        duplicate_tm_ids.sum()
    )

    print(
        f"Duplicate matched Transfermarkt IDs: "
        f"{duplicate_count}"
    )

    if duplicate_count:

        duplicates = (
            player_map[
                player_map[
                    "mapping_status"
                ]
                ==
                "MATCHED"
            ]
            .loc[
                duplicate_tm_ids.values,
                [
                    "canonical_team",
                    "player_name",
                    "transfermarkt_player_id",
                    "transfermarkt_player_name",
                ],
            ]
        )

        print(
            "\nWARNING: duplicate mapped IDs:"
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

    missing_squad_teams = team_audit[
        team_audit[
            "squad_missing"
        ]
        ==
        1
    ]

    print(
        f"\nCurrent UCL clubs with no "
        f"football-data squad: "
        f"{len(missing_squad_teams)}"
    )

    if not missing_squad_teams.empty:

        print(
            missing_squad_teams[
                [
                    "canonical_team",
                    "identity_status",
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    player_map: pd.DataFrame,
    valuation_map: pd.DataFrame,
    team_audit: pd.DataFrame,
    transfermarkt_player_schema: dict,
    valuation_metadata: dict,
) -> None:
    """
    Save player identity and local valuation audit.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_map.to_csv(
        PLAYER_MAP_FILE,
        index=False,
        encoding="utf-8",
    )

    valuation_output = (
        valuation_map.copy()
    )

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
        VALUATION_OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    team_audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    matched = int(
        (
            player_map[
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

    latest_date = valuation_metadata[
        "latest_valuation_date"
    ]

    if pd.notna(
        latest_date
    ):

        latest_date_value = (
            pd.Timestamp(
                latest_date
            )
            .date()
            .isoformat()
        )

    else:

        latest_date_value = None

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "current_roster_source": (
            "football-data.org"
        ),

        "transfermarkt_player_source": (
            "local players.csv snapshot"
        ),

        "valuation_source": (
            "local player_valuations.csv"
        ),

        "current_player_rows": int(
            len(
                player_map
            )
        ),

        "matched_players": (
            matched
        ),

        "mapping_coverage": (
            matched
            /
            len(
                player_map
            )
            if len(
                player_map
            )
            else 0.0
        ),

        "valued_players": (
            valued
        ),

        "valuation_coverage": (
            valued
            /
            len(
                valuation_map
            )
            if len(
                valuation_map
            )
            else 0.0
        ),

        "latest_local_valuation_date": (
            latest_date_value
        ),

        "transfermarkt_player_schema": (
            transfermarkt_player_schema
        ),

        "rules": {
            "fuzzy_matching_used": False,

            "exact_name_and_dob_preferred": True,

            "current_club_used_only_for_live_identity_resolution": True,

            "historical_training_features_not_modified": True,

            "players_csv_market_value_used_only_as_live_fallback": True,

            "unmatched_players_preserved": True,

            "ambiguous_players_not_auto_resolved": True,
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
    player_map: pd.DataFrame,
    valuation_map: pd.DataFrame,
    team_audit: pd.DataFrame,
) -> None:
    """
    Print actionable mapping summary.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 PLAYER IDENTITY / VALUE SUMMARY"
    )

    print(
        "=" * 75
    )

    matched = int(
        (
            player_map[
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
        f"\nCurrent players: "
        f"{len(player_map):,}"
    )

    print(
        f"Mapped to Transfermarkt: "
        f"{matched:,}/{len(player_map):,}"
    )

    print(
        f"Players with usable value: "
        f"{valued:,}/{len(player_map):,}"
    )

    print(
        "\nMapping methods:"
    )

    print(
        player_map[
            "mapping_method"
        ]
        .fillna(
            "UNRESOLVED"
        )
        .value_counts()
        .to_string()
    )

    print(
        "\nLowest team coverage:"
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

    print(
        "\nNew 2026/27 clubs:"
    )

    print(
        team_audit[
            team_audit[
                "identity_status"
            ]
            ==
            "NEW_2026_27"
        ][
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

    unresolved = player_map[
        player_map[
            "mapping_status"
        ]
        !=
        "MATCHED"
    ]

    print(
        "\nFirst 30 unresolved players:"
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
                    "player_name",
                    "date_of_birth",
                    "mapping_status",
                    "candidate_count",
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


def map_current_players_to_transfermarkt() -> None:
    """
    Bridge the live football-data.org roster snapshot to the
    local Transfermarkt player and valuation history.

    No fuzzy player matching.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - CURRENT PLAYER TRANSFERMARKT MAPPING"
    )

    print(
        "=" * 75
    )

    current_squads = (
        load_current_squads()
    )

    current_teams = (
        load_current_ucl_teams()
    )

    club_map = (
        load_transfermarkt_club_map()
    )

    current_squads = (
        attach_transfermarkt_team_ids(
            squads=current_squads,
            club_map=club_map,
        )
    )

    (
        transfermarkt_players,
        transfermarkt_player_schema,
    ) = load_transfermarkt_players()

    print(
        f"\nCurrent roster players: "
        f"{len(current_squads):,}"
    )

    print(
        f"Transfermarkt players available: "
        f"{len(transfermarkt_players):,}"
    )

    player_map = map_current_players(
        current_players=current_squads,
        transfermarkt_players=transfermarkt_players,
    )

    matched_ids = set(
        pd.to_numeric(
            player_map.loc[
                player_map[
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

    (
        latest_valuations,
        valuation_metadata,
    ) = load_latest_valuations(
        matched_ids
    )

    valuation_map = attach_valuations(
        player_map=player_map,
        latest_valuations=latest_valuations,
    )

    team_audit = build_team_audit(
        current_teams=current_teams,
        valuation_map=valuation_map,
    )

    run_quality_checks(
        player_map=player_map,
        valuation_map=valuation_map,
        team_audit=team_audit,
    )

    save_outputs(
        player_map=player_map,
        valuation_map=valuation_map,
        team_audit=team_audit,
        transfermarkt_player_schema=transfermarkt_player_schema,
        valuation_metadata=valuation_metadata,
    )

    print_summary(
        player_map=player_map,
        valuation_map=valuation_map,
        team_audit=team_audit,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT PLAYER TRANSFERMARKT MAPPING COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nPlayer identity map:"
    )

    print(
        PLAYER_MAP_FILE
    )

    print(
        "\nPlayer valuation snapshot:"
    )

    print(
        VALUATION_OUTPUT_FILE
    )

    print(
        "\nTeam mapping/value audit:"
    )

    print(
        AUDIT_FILE
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
        "These are current/live player identities and "
        "latest locally available valuations."
    )

    print(
        "Historical training features remain unchanged."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    map_current_players_to_transfermarkt()