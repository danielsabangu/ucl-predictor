from __future__ import annotations

import re
import unicodedata

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

CURRENT_TEAM_SNAPSHOT_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "football_data_ucl_teams.csv"
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

OUTPUT_DIR = (
    REPORTS_DIR
    / "metrics"
)

UNRESOLVED_CANDIDATES_FILE = (
    OUTPUT_DIR
    / "ucl_2026_2027_unresolved_player_candidates.csv"
)

DUPLICATE_MAPPING_FILE = (
    OUTPUT_DIR
    / "ucl_2026_2027_duplicate_player_mapping_audit.csv"
)

MISSING_SQUAD_FILE = (
    OUTPUT_DIR
    / "ucl_2026_2027_missing_squad_transfermarkt_audit.csv"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(
    value,
) -> str:
    """
    Deterministic text normalization.

    Used only for candidate evidence.

    No fuzzy matching.
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

    return " ".join(
        text.split()
    )


def normalize_date(
    value,
):
    """
    Normalize date-of-birth values.
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


def name_tokens(
    value,
) -> set[str]:
    """
    Return deterministic normalized name tokens.
    """

    normalized = normalize_text(
        value
    )

    if not normalized:

        return set()

    return set(
        normalized.split()
    )


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
    Detect compatible Transfermarkt dataset columns.
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
# LOAD CURRENT PLAYER MAP
# ============================================================


def load_current_player_map() -> pd.DataFrame:
    """
    Load current football-data -> Transfermarkt mapping.
    """

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
        "transfermarkt_player_id",
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

    dataframe[
        "normalized_dob"
    ] = dataframe[
        "date_of_birth"
    ].map(
        normalize_date
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
# LOAD TRANSFERMARKT PLAYERS
# ============================================================


def load_transfermarkt_players() -> pd.DataFrame:
    """
    Load and normalize local Transfermarkt player snapshot.
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

    dob_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "date_of_birth",
            "birth_date",
        ],
        description="date of birth",
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
        "_normalized_name"
    ] = dataframe[
        name_column
    ].map(
        normalize_text
    )

    dataframe[
        "_dob"
    ] = dataframe[
        dob_column
    ].map(
        normalize_date
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

    if last_name_column is not None:

        dataframe[
            "_last_name"
        ] = dataframe[
            last_name_column
        ].map(
            normalize_text
        )

    else:

        dataframe[
            "_last_name"
        ] = ""

    if (
        first_name_column is not None
        and
        last_name_column is not None
    ):

        dataframe[
            "_first_last"
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

    else:

        dataframe[
            "_first_last"
        ] = ""

    return dataframe


# ============================================================
# UNRESOLVED CANDIDATE AUDIT
# ============================================================


def audit_unresolved_players(
    *,
    current_map: pd.DataFrame,
    transfermarkt_players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find exact-DOB evidence for currently unresolved players.

    No mappings are automatically approved.
    """

    unresolved = current_map[
        current_map[
            "mapping_status"
        ]
        !=
        "MATCHED"
    ].copy()

    rows = []

    for (
        _,
        player,
    ) in unresolved.iterrows():

        normalized_dob = player[
            "normalized_dob"
        ]

        if pd.isna(
            normalized_dob
        ):

            continue

        if pd.isna(
            player[
                "transfermarkt_club_id"
            ]
        ):

            continue

        current_team = str(
            player[
                "canonical_team"
            ]
        )

        current_player_name = str(
            player[
                "player_name"
            ]
        )

        team_tm_id = int(
            player[
                "transfermarkt_club_id"
            ]
        )

        current_normalized_name = normalize_text(
            current_player_name
        )

        current_tokens = name_tokens(
            current_player_name
        )

        # ----------------------------------------------------
        # EXACT DATE-OF-BIRTH CANDIDATES
        # ----------------------------------------------------

        dob_candidates = transfermarkt_players[
            transfermarkt_players[
                "_dob"
            ]
            ==
            normalized_dob
        ].copy()

        if dob_candidates.empty:

            continue

        same_club_mask = (
            pd.to_numeric(
                dob_candidates[
                    "_current_club_id"
                ],
                errors="coerce",
            )
            ==
            team_tm_id
        )

        same_club_count = int(
            same_club_mask.sum()
        )

        global_dob_count = int(
            len(
                dob_candidates
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use iterrows instead of itertuples here because
        # pandas renames namedtuple fields beginning with "_".
        #
        # Columns such as _current_club_id therefore cannot
        # safely be accessed from an itertuples namedtuple.
        # ----------------------------------------------------

        for (
            _,
            candidate,
        ) in dob_candidates.iterrows():

            raw_candidate_club_id = candidate[
                "_current_club_id"
            ]

            candidate_club_id = (
                int(
                    raw_candidate_club_id
                )
                if pd.notna(
                    raw_candidate_club_id
                )
                else None
            )

            same_club = (
                candidate_club_id
                ==
                team_tm_id
            )

            candidate_name = str(
                candidate[
                    "_name"
                ]
            )

            candidate_normalized_name = str(
                candidate[
                    "_normalized_name"
                ]
            )

            candidate_tokens = name_tokens(
                candidate_name
            )

            overlapping_tokens = sorted(
                current_tokens
                &
                candidate_tokens
            )

            token_overlap_count = len(
                overlapping_tokens
            )

            exact_normalized_name = int(
                current_normalized_name
                ==
                candidate_normalized_name
            )

            # ------------------------------------------------
            # REVIEW CLASS
            # ------------------------------------------------

            if (
                same_club
                and
                exact_normalized_name
            ):

                candidate_class = (
                    "A_EXACT_NAME_DOB_CURRENT_CLUB"
                )

                priority = 1

            elif (
                same_club
                and
                same_club_count
                ==
                1
                and
                token_overlap_count
                >=
                1
            ):

                candidate_class = (
                    "B_UNIQUE_DOB_CURRENT_CLUB_WITH_NAME_TOKEN"
                )

                priority = 2

            elif (
                same_club
                and
                same_club_count
                ==
                1
            ):

                candidate_class = (
                    "C_UNIQUE_DOB_CURRENT_CLUB"
                )

                priority = 3

            elif (
                global_dob_count
                ==
                1
                and
                token_overlap_count
                >=
                1
            ):

                candidate_class = (
                    "D_UNIQUE_GLOBAL_DOB_WITH_NAME_TOKEN"
                )

                priority = 4

            elif (
                global_dob_count
                ==
                1
            ):

                candidate_class = (
                    "E_UNIQUE_GLOBAL_DOB"
                )

                priority = 5

            else:

                candidate_class = (
                    "F_MULTIPLE_DOB_CANDIDATES"
                )

                priority = 6

            rows.append(
                {
                    "canonical_team": (
                        current_team
                    ),

                    "football_data_player_id": (
                        player[
                            "football_data_player_id"
                        ]
                    ),

                    "football_data_player_name": (
                        current_player_name
                    ),

                    "date_of_birth": (
                        player[
                            "date_of_birth"
                        ]
                    ),

                    "transfermarkt_club_id": (
                        team_tm_id
                    ),

                    "candidate_transfermarkt_player_id": int(
                        candidate[
                            "_player_id"
                        ]
                    ),

                    "candidate_transfermarkt_name": (
                        candidate_name
                    ),

                    "candidate_current_club_id": (
                        candidate_club_id
                    ),

                    "candidate_current_club_name": (
                        candidate[
                            "_current_club_name"
                        ]
                    ),

                    "same_current_club": int(
                        same_club
                    ),

                    "exact_normalized_name": (
                        exact_normalized_name
                    ),

                    "name_token_overlap_count": (
                        token_overlap_count
                    ),

                    "overlapping_name_tokens": (
                        "|".join(
                            overlapping_tokens
                        )
                    ),

                    "same_club_dob_candidate_count": (
                        same_club_count
                    ),

                    "global_dob_candidate_count": (
                        global_dob_count
                    ),

                    "candidate_class": (
                        candidate_class
                    ),

                    "priority": (
                        priority
                    ),
                }
            )

    output = pd.DataFrame(
        rows
    )

    if output.empty:

        return pd.DataFrame(
            columns=[
                "canonical_team",
                "football_data_player_id",
                "football_data_player_name",
                "date_of_birth",
                "transfermarkt_club_id",
                "candidate_transfermarkt_player_id",
                "candidate_transfermarkt_name",
                "candidate_current_club_id",
                "candidate_current_club_name",
                "same_current_club",
                "exact_normalized_name",
                "name_token_overlap_count",
                "overlapping_name_tokens",
                "same_club_dob_candidate_count",
                "global_dob_candidate_count",
                "candidate_class",
                "priority",
            ]
        )

    output = output.sort_values(
        by=[
            "priority",
            "canonical_team",
            "football_data_player_name",
            "candidate_transfermarkt_name",
        ]
    ).reset_index(
        drop=True
    )

    return output


# ============================================================
# DUPLICATE TRANSFERMARKT PLAYER ID AUDIT
# ============================================================


def audit_duplicate_mappings(
    current_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Show all current players sharing the same mapped
    Transfermarkt player ID.
    """

    matched = current_map[
        current_map[
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

    matched = matched[
        matched[
            "transfermarkt_player_id"
        ].notna()
    ].copy()

    counts = (
        matched[
            "transfermarkt_player_id"
        ]
        .value_counts()
    )

    duplicate_ids = set(
        counts[
            counts
            >
            1
        ].index
    )

    duplicates = matched[
        matched[
            "transfermarkt_player_id"
        ].isin(
            duplicate_ids
        )
    ].copy()

    if duplicates.empty:

        return pd.DataFrame(
            columns=[
                "canonical_team",
                "football_data_player_id",
                "player_name",
                "date_of_birth",
                "mapping_method",
                "transfermarkt_player_id",
                "transfermarkt_player_name",
                "transfermarkt_current_club_id",
                "transfermarkt_current_club_name",
                "mapping_count_for_tm_id",
            ]
        )

    duplicates[
        "mapping_count_for_tm_id"
    ] = duplicates[
        "transfermarkt_player_id"
    ].map(
        counts
    )

    desired_columns = [
        "canonical_team",
        "football_data_player_id",
        "player_name",
        "date_of_birth",
        "position",
        "nationality",
        "mapping_method",
        "transfermarkt_player_id",
        "transfermarkt_player_name",
        "transfermarkt_current_club_id",
        "transfermarkt_current_club_name",
        "mapping_count_for_tm_id",
    ]

    existing_columns = [
        column
        for column in desired_columns
        if column in duplicates.columns
    ]

    duplicates = duplicates[
        existing_columns
    ]

    duplicates = duplicates.sort_values(
        by=[
            "transfermarkt_player_id",
            "canonical_team",
            "player_name",
        ]
    ).reset_index(
        drop=True
    )

    return duplicates


# ============================================================
# LOAD TEAM SNAPSHOT
# ============================================================


def load_team_snapshot() -> pd.DataFrame:
    """
    Load football-data team snapshot.
    """

    if not CURRENT_TEAM_SNAPSHOT_FILE.exists():

        raise FileNotFoundError(
            "Current team snapshot not found:\n"
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
            "Current team snapshot missing columns: "
            f"{sorted(missing)}"
        )

    return dataframe


# ============================================================
# LOAD EXTENDED CLUB MAP
# ============================================================


def load_extended_club_map() -> pd.DataFrame:
    """
    Load 123-club Transfermarkt map.
    """

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
            "Extended club map missing columns: "
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
# MISSING FOOTBALL-DATA SQUAD AUDIT
# ============================================================


def audit_missing_squads(
    *,
    team_snapshot: pd.DataFrame,
    club_map: pd.DataFrame,
    transfermarkt_players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether local Transfermarkt players.csv can provide
    a current-club roster for teams where football-data
    returned no squad.
    """

    squad_size_numeric = pd.to_numeric(
        team_snapshot[
            "squad_size"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    missing = team_snapshot[
        squad_size_numeric
        ==
        0
    ].copy()

    rows = []

    for (
        _,
        team,
    ) in missing.iterrows():

        canonical_team = str(
            team[
                "canonical_team"
            ]
        )

        mapping = club_map[
            club_map[
                "canonical_team"
            ]
            ==
            canonical_team
        ]

        if len(
            mapping
        ) != 1:

            raise RuntimeError(
                f"Could not resolve exactly one "
                f"Transfermarkt club ID for "
                f"{canonical_team}."
            )

        club_id = int(
            mapping.iloc[
                0
            ][
                "transfermarkt_club_id"
            ]
        )

        local_players = transfermarkt_players[
            pd.to_numeric(
                transfermarkt_players[
                    "_current_club_id"
                ],
                errors="coerce",
            )
            ==
            club_id
        ].copy()

        player_ids = sorted(
            local_players[
                "_player_id"
            ].astype(
                int
            ).tolist()
        )

        player_names = sorted(
            local_players[
                "_name"
            ].astype(
                str
            ).tolist()
        )

        rows.append(
            {
                "canonical_team": (
                    canonical_team
                ),

                "transfermarkt_club_id": (
                    club_id
                ),

                "football_data_squad_size": 0,

                "local_transfermarkt_current_players": int(
                    len(
                        local_players
                    )
                ),

                "local_transfermarkt_player_ids": (
                    "|".join(
                        str(
                            value
                        )
                        for value in player_ids
                    )
                ),

                "local_transfermarkt_player_names": (
                    "|".join(
                        player_names
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT SUMMARY
# ============================================================


def print_summary(
    *,
    current_map: pd.DataFrame,
    candidates: pd.DataFrame,
    duplicates: pd.DataFrame,
    missing_squads: pd.DataFrame,
) -> None:
    """
    Print actionable live-player audit results.
    """

    unresolved = current_map[
        current_map[
            "mapping_status"
        ]
        !=
        "MATCHED"
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE PLAYER GAP AUDIT SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nCurrent unresolved players: "
        f"{len(unresolved)}"
    )

    if candidates.empty:

        print(
            "Players with exact-DOB candidate evidence: 0"
        )

    else:

        candidate_players = int(
            candidates[
                "football_data_player_id"
            ].nunique()
        )

        print(
            "Players with exact-DOB candidate evidence: "
            f"{candidate_players}"
        )

        print(
            "\nCandidate classes:"
        )

        print(
            candidates[
                "candidate_class"
            ]
            .value_counts()
            .to_string()
        )

        strong_classes = {
            "A_EXACT_NAME_DOB_CURRENT_CLUB",
            "B_UNIQUE_DOB_CURRENT_CLUB_WITH_NAME_TOKEN",
            "C_UNIQUE_DOB_CURRENT_CLUB",
            "D_UNIQUE_GLOBAL_DOB_WITH_NAME_TOKEN",
        }

        strong = candidates[
            candidates[
                "candidate_class"
            ].isin(
                strong_classes
            )
        ]

        strong_players = int(
            strong[
                "football_data_player_id"
            ].nunique()
        )

        print(
            "\nPlayers with strong deterministic "
            "review candidates: "
            f"{strong_players}"
        )

        print(
            "\nFirst 40 strongest candidates:"
        )

        print(
            candidates[
                [
                    "canonical_team",
                    "football_data_player_name",
                    "date_of_birth",
                    "candidate_transfermarkt_player_id",
                    "candidate_transfermarkt_name",
                    "candidate_current_club_name",
                    "same_current_club",
                    "name_token_overlap_count",
                    "candidate_class",
                ]
            ]
            .head(
                40
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DUPLICATE MAPPING CONFLICTS"
    )

    print(
        "=" * 75
    )

    if duplicates.empty:

        print(
            "\nNone ✓"
        )

    else:

        print(
            "\n"
            +
            duplicates.to_string(
                index=False
            )
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "MISSING FOOTBALL-DATA SQUAD AUDIT"
    )

    print(
        "=" * 75
    )

    if missing_squads.empty:

        print(
            "\nNone"
        )

    else:

        print(
            "\n"
            +
            missing_squads[
                [
                    "canonical_team",
                    "transfermarkt_club_id",
                    "football_data_squad_size",
                    "local_transfermarkt_current_players",
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_outputs(
    *,
    candidates: pd.DataFrame,
    duplicates: pd.DataFrame,
    missing_squads: pd.DataFrame,
) -> None:
    """
    Save all audit artifacts.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        UNRESOLVED_CANDIDATES_FILE,
        index=False,
        encoding="utf-8",
    )

    duplicates.to_csv(
        DUPLICATE_MAPPING_FILE,
        index=False,
        encoding="utf-8",
    )

    missing_squads.to_csv(
        MISSING_SQUAD_FILE,
        index=False,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================


def audit_current_player_gaps() -> None:
    """
    Audit:

        1. unresolved current players
        2. duplicate Transfermarkt mappings
        3. teams missing football-data squads

    This script performs no mutations.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - LIVE PLAYER GAP AUDIT"
    )

    print(
        "=" * 75
    )

    current_map = (
        load_current_player_map()
    )

    transfermarkt_players = (
        load_transfermarkt_players()
    )

    team_snapshot = (
        load_team_snapshot()
    )

    club_map = (
        load_extended_club_map()
    )

    print(
        f"\nCurrent player rows: "
        f"{len(current_map):,}"
    )

    print(
        f"Local Transfermarkt players: "
        f"{len(transfermarkt_players):,}"
    )

    candidates = audit_unresolved_players(
        current_map=current_map,
        transfermarkt_players=transfermarkt_players,
    )

    duplicates = audit_duplicate_mappings(
        current_map
    )

    missing_squads = audit_missing_squads(
        team_snapshot=team_snapshot,
        club_map=club_map,
        transfermarkt_players=transfermarkt_players,
    )

    save_outputs(
        candidates=candidates,
        duplicates=duplicates,
        missing_squads=missing_squads,
    )

    print_summary(
        current_map=current_map,
        candidates=candidates,
        duplicates=duplicates,
        missing_squads=missing_squads,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE PLAYER GAP AUDIT COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nUnresolved candidate audit:"
    )

    print(
        UNRESOLVED_CANDIDATES_FILE
    )

    print(
        "\nDuplicate mapping audit:"
    )

    print(
        DUPLICATE_MAPPING_FILE
    )

    print(
        "\nMissing squad audit:"
    )

    print(
        MISSING_SQUAD_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No player mappings were changed."
    )

    print(
        "This remains a review/audit step only."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    audit_current_player_gaps()