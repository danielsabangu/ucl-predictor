from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)


# ============================================================
# PATHS
# ============================================================

HISTORICAL_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

TRANSFERMARKT_CLUBS_FILE = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
    / "clubs.csv"
)

OUTPUT_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map_2026_27.csv"
)

AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_transfermarkt_extended_map_audit.csv"
)

METADATA_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map_2026_27_metadata.json"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_HISTORICAL_CLUBS = 117

EXPECTED_NEW_CLUBS = 6

EXPECTED_EXTENDED_CLUBS = 123


# ============================================================
# APPROVED NEW CLUB MAPPINGS
# ============================================================

APPROVED_NEW_MAPPINGS = {
    "Como 1907": {
        "transfermarkt_club_id": 1047,
        "expected_transfermarkt_name": "Como 1907",
        "mapping_source": "local_clubs_csv",
    },

    "Fenerbahçe": {
        "transfermarkt_club_id": 36,
        "expected_transfermarkt_name": "Fenerbahce",
        "mapping_source": "local_clubs_csv",
    },

    "LASK Linz": {
        "transfermarkt_club_id": 413,
        "expected_transfermarkt_name": "LASK",
        "mapping_source": "local_clubs_csv",
    },

    "Real Betis Balompié": {
        "transfermarkt_club_id": 150,
        "expected_transfermarkt_name": "Real Betis Balompié",
        "mapping_source": "local_clubs_csv",
    },

    "Sabah FK": {
        "transfermarkt_club_id": 63993,
        "expected_transfermarkt_name": "Sabah FK",
        "mapping_source": "external_transfermarkt_verified",
    },

    "Viking FK": {
        "transfermarkt_club_id": 239,
        "expected_transfermarkt_name": "Viking Fotballklubb",
        "mapping_source": "local_clubs_csv",
    },
}


# ============================================================
# LOAD HISTORICAL MAP
# ============================================================


def load_historical_map() -> pd.DataFrame:
    """
    Load the completed 117-club historical UCL /
    Transfermarkt map.
    """

    if not HISTORICAL_MAP_FILE.exists():

        raise FileNotFoundError(
            "Historical Transfermarkt map not found:\n"
            f"{HISTORICAL_MAP_FILE}"
        )

    dataframe = pd.read_csv(
        HISTORICAL_MAP_FILE,
        low_memory=False,
    )

    if len(
        dataframe
    ) != EXPECTED_HISTORICAL_CLUBS:

        raise RuntimeError(
            f"Expected {EXPECTED_HISTORICAL_CLUBS} "
            f"historical mappings, found "
            f"{len(dataframe)}."
        )

    return dataframe


# ============================================================
# DETECT HISTORICAL COLUMNS
# ============================================================


def detect_column(
    *,
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """
    Detect a required semantic column without assuming the
    exact legacy schema.
    """

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    raise RuntimeError(
        f"Could not identify {description} column.\n"
        f"Available columns:\n"
        f"{list(dataframe.columns)}"
    )


def detect_historical_columns(
    dataframe: pd.DataFrame,
) -> tuple[
    str,
    str,
    str | None,
]:
    """
    Detect canonical-team, Transfermarkt-ID and optional
    Transfermarkt-name columns.
    """

    team_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "canonical_team",
            "ucl_team",
            "team",
            "team_name",
            "canonical_name",
        ],
        description="canonical team",
    )

    id_column = detect_column(
        dataframe=dataframe,
        candidates=[
            "transfermarkt_club_id",
            "club_id",
            "tm_club_id",
        ],
        description="Transfermarkt club ID",
    )

    name_column = None

    for candidate in [
        "transfermarkt_name",
        "transfermarkt_club_name",
        "club_name",
        "tm_club_name",
    ]:

        if candidate in dataframe.columns:

            name_column = candidate
            break

    return (
        team_column,
        id_column,
        name_column,
    )


# ============================================================
# LOAD CURRENT UCL IDENTITIES
# ============================================================


def load_current_team_map() -> pd.DataFrame:
    """
    Load resolved 2026/27 UCL identities.
    """

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

    new_clubs = dataframe[
        dataframe[
            "identity_status"
        ]
        ==
        "NEW_2026_27"
    ].copy()

    found = set(
        new_clubs[
            "canonical_team"
        ]
    )

    expected = set(
        APPROVED_NEW_MAPPINGS
    )

    if found != expected:

        raise RuntimeError(
            "\nNew club set does not match approved "
            "Transfermarkt mappings.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(found)}"
        )

    return dataframe


# ============================================================
# LOAD LOCAL CLUB SNAPSHOT
# ============================================================


def load_local_transfermarkt_clubs() -> pd.DataFrame:
    """
    Load local Transfermarkt clubs.csv.

    Sabah FK is expected to be absent from this snapshot.
    """

    if not TRANSFERMARKT_CLUBS_FILE.exists():

        raise FileNotFoundError(
            "Transfermarkt clubs.csv not found:\n"
            f"{TRANSFERMARKT_CLUBS_FILE}"
        )

    dataframe = pd.read_csv(
        TRANSFERMARKT_CLUBS_FILE,
        low_memory=False,
    )

    required = {
        "club_id",
        "name",
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
            "Transfermarkt clubs.csv missing columns: "
            f"{sorted(missing)}"
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

    return dataframe


# ============================================================
# VERIFY APPROVED MAPPINGS
# ============================================================


def verify_new_mappings(
    local_clubs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Verify every approved mapping.

    Five are expected in the local snapshot.

    Sabah FK ID 63993 is externally verified and may not
    exist in the local snapshot.
    """

    rows = []

    for (
        canonical_team,
        mapping,
    ) in APPROVED_NEW_MAPPINGS.items():

        transfermarkt_id = int(
            mapping[
                "transfermarkt_club_id"
            ]
        )

        expected_name = mapping[
            "expected_transfermarkt_name"
        ]

        mapping_source = mapping[
            "mapping_source"
        ]

        local_match = local_clubs[
            local_clubs[
                "club_id"
            ]
            ==
            transfermarkt_id
        ]

        local_available = (
            not
            local_match.empty
        )

        if local_available:

            if len(
                local_match
            ) != 1:

                raise RuntimeError(
                    f"Transfermarkt club ID "
                    f"{transfermarkt_id} appears "
                    f"multiple times."
                )

            local_name = str(
                local_match.iloc[
                    0
                ][
                    "name"
                ]
            )

        else:

            local_name = None

        # ----------------------------------------------------
        # LOCAL MAPPINGS MUST EXIST
        # ----------------------------------------------------

        if (
            mapping_source
            ==
            "local_clubs_csv"
            and
            not local_available
        ):

            raise RuntimeError(
                f"{canonical_team} mapping expects "
                f"local Transfermarkt ID "
                f"{transfermarkt_id}, but it is "
                "not present in clubs.csv."
            )

        # ----------------------------------------------------
        # SABAH IS EXPECTED TO BE EXTERNAL
        # ----------------------------------------------------

        if (
            mapping_source
            ==
            "external_transfermarkt_verified"
            and
            canonical_team
            !=
            "Sabah FK"
        ):

            raise RuntimeError(
                "Unexpected external-only mapping."
            )

        rows.append(
            {
                "canonical_team": (
                    canonical_team
                ),

                "transfermarkt_club_id": (
                    transfermarkt_id
                ),

                "transfermarkt_name": (
                    local_name
                    if local_name is not None
                    else expected_name
                ),

                "expected_transfermarkt_name": (
                    expected_name
                ),

                "mapping_source": (
                    mapping_source
                ),

                "local_clubs_snapshot_available": int(
                    local_available
                ),

                "mapping_verified": 1,
            }
        )

    verification = pd.DataFrame(
        rows
    )

    return verification


# ============================================================
# STANDARDIZE HISTORICAL MAP
# ============================================================


def standardize_historical_map(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the 117-club historical map into the current
    standardized identity schema without modifying the
    original file.
    """

    (
        team_column,
        id_column,
        name_column,
    ) = detect_historical_columns(
        dataframe
    )

    output = pd.DataFrame()

    output[
        "canonical_team"
    ] = dataframe[
        team_column
    ].astype(
        str
    )

    output[
        "transfermarkt_club_id"
    ] = pd.to_numeric(
        dataframe[
            id_column
        ],
        errors="raise",
    ).astype(
        int
    )

    if name_column is not None:

        output[
            "transfermarkt_name"
        ] = dataframe[
            name_column
        ].astype(
            str
        )

    else:

        output[
            "transfermarkt_name"
        ] = pd.NA

    output[
        "identity_period"
    ] = "HISTORICAL_2010_2025"

    output[
        "mapping_source"
    ] = "validated_historical_map"

    output[
        "local_clubs_snapshot_available"
    ] = 1

    output[
        "mapping_verified"
    ] = 1

    return output


# ============================================================
# BUILD EXTENDED MAP
# ============================================================


def build_extended_map(
    *,
    historical_map: pd.DataFrame,
    verified_new: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build 117 + 6 = 123 club identity universe.
    """

    historical = (
        standardize_historical_map(
            historical_map
        )
    )

    new = verified_new.copy()

    new[
        "identity_period"
    ] = "NEW_2026_27"

    new = new[
        [
            "canonical_team",
            "transfermarkt_club_id",
            "transfermarkt_name",
            "identity_period",
            "mapping_source",
            "local_clubs_snapshot_available",
            "mapping_verified",
        ]
    ]

    extended = pd.concat(
        [
            historical,
            new,
        ],
        ignore_index=True,
    )

    extended = extended.sort_values(
        by=[
            "canonical_team",
        ]
    ).reset_index(
        drop=True
    )

    return extended


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    extended: pd.DataFrame,
) -> None:
    """
    Validate the new 123-club identity map.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "EXTENDED TRANSFERMARKT MAP QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        f"\nRows: "
        f"{len(extended)}"
    )

    if len(
        extended
    ) != EXPECTED_EXTENDED_CLUBS:

        raise RuntimeError(
            f"Expected {EXPECTED_EXTENDED_CLUBS} "
            f"clubs, found {len(extended)}."
        )

    duplicate_teams = int(
        extended[
            "canonical_team"
        ].duplicated().sum()
    )

    duplicate_ids = int(
        extended[
            "transfermarkt_club_id"
        ].duplicated().sum()
    )

    missing_teams = int(
        extended[
            "canonical_team"
        ].isna().sum()
    )

    missing_ids = int(
        extended[
            "transfermarkt_club_id"
        ].isna().sum()
    )

    unverified = int(
        (
            extended[
                "mapping_verified"
            ]
            !=
            1
        ).sum()
    )

    print(
        f"Duplicate canonical teams: "
        f"{duplicate_teams}"
    )

    print(
        f"Duplicate Transfermarkt IDs: "
        f"{duplicate_ids}"
    )

    print(
        f"Missing canonical teams: "
        f"{missing_teams}"
    )

    print(
        f"Missing Transfermarkt IDs: "
        f"{missing_ids}"
    )

    print(
        f"Unverified mappings: "
        f"{unverified}"
    )

    if duplicate_teams:

        raise RuntimeError(
            "Duplicate canonical team identities."
        )

    if duplicate_ids:

        raise RuntimeError(
            "Duplicate Transfermarkt club IDs."
        )

    if missing_teams:

        raise RuntimeError(
            "Missing canonical team."
        )

    if missing_ids:

        raise RuntimeError(
            "Missing Transfermarkt club ID."
        )

    if unverified:

        raise RuntimeError(
            "Unverified mapping detected."
        )

    historical_count = int(
        (
            extended[
                "identity_period"
            ]
            ==
            "HISTORICAL_2010_2025"
        ).sum()
    )

    new_count = int(
        (
            extended[
                "identity_period"
            ]
            ==
            "NEW_2026_27"
        ).sum()
    )

    external_only_count = int(
        (
            extended[
                "local_clubs_snapshot_available"
            ]
            ==
            0
        ).sum()
    )

    print(
        f"\nHistorical clubs: "
        f"{historical_count}"
    )

    print(
        f"New 2026/27 clubs: "
        f"{new_count}"
    )

    print(
        f"External-only club mappings: "
        f"{external_only_count}"
    )

    if (
        historical_count
        !=
        EXPECTED_HISTORICAL_CLUBS
    ):

        raise RuntimeError(
            "Historical mapping count changed."
        )

    if (
        new_count
        !=
        EXPECTED_NEW_CLUBS
    ):

        raise RuntimeError(
            "Unexpected new club count."
        )


# ============================================================
# PRINT NEW CLUBS
# ============================================================


def print_new_clubs(
    extended: pd.DataFrame,
) -> None:
    """
    Show exactly what was added.
    """

    new_clubs = extended[
        extended[
            "identity_period"
        ]
        ==
        "NEW_2026_27"
    ].copy()

    print(
        "\n"
        + "=" * 75
    )

    print(
        "NEW 2026/27 TRANSFERMARKT IDENTITIES"
    )

    print(
        "=" * 75
    )

    print(
        "\n"
        +
        new_clubs[
            [
                "canonical_team",
                "transfermarkt_club_id",
                "transfermarkt_name",
                "mapping_source",
                "local_clubs_snapshot_available",
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
    extended: pd.DataFrame,
    verified_new: pd.DataFrame,
) -> None:
    """
    Save current 123-club map and audit metadata.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extended.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    verified_new.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "historical_clubs": (
            EXPECTED_HISTORICAL_CLUBS
        ),

        "new_2026_27_clubs": (
            EXPECTED_NEW_CLUBS
        ),

        "extended_clubs": (
            EXPECTED_EXTENDED_CLUBS
        ),

        "approved_new_mappings": (
            APPROVED_NEW_MAPPINGS
        ),

        "external_only_mapping": {
            "canonical_team": (
                "Sabah FK"
            ),

            "transfermarkt_club_id": (
                63993
            ),

            "reason": (
                "Transfermarkt ID externally verified "
                "but absent from local clubs.csv snapshot."
            ),
        },

        "rules": {
            "fuzzy_matching_used": False,

            "historical_map_modified": False,

            "new_map_written_separately": True,

            "all_new_ids_manually_approved": True,
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
# MAIN
# ============================================================


def extend_current_transfermarkt_map() -> None:
    """
    Extend the validated historical Transfermarkt identity
    universe from 117 to 123 clubs for the 2026/27 live
    prediction pipeline.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - EXTEND CURRENT TRANSFERMARKT MAP"
    )

    print(
        "=" * 75
    )

    historical_map = (
        load_historical_map()
    )

    current_team_map = (
        load_current_team_map()
    )

    local_clubs = (
        load_local_transfermarkt_clubs()
    )

    print(
        f"\nHistorical mapped clubs: "
        f"{len(historical_map)}"
    )

    print(
        f"Current UCL clubs: "
        f"{len(current_team_map)}"
    )

    print(
        f"Local Transfermarkt clubs: "
        f"{len(local_clubs):,}"
    )

    verified_new = verify_new_mappings(
        local_clubs
    )

    extended = build_extended_map(
        historical_map=historical_map,
        verified_new=verified_new,
    )

    run_quality_checks(
        extended
    )

    save_outputs(
        extended=extended,
        verified_new=verified_new,
    )

    print_new_clubs(
        extended
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT TRANSFERMARKT MAP EXTENDED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nExtended map:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nNew-club verification audit:"
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
        "\nIdentity universe: "
        "117 historical + 6 new = 123 clubs ✓"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Sabah FK is mapped to Transfermarkt ID 63993 "
        "but is absent from the local Transfermarkt snapshot."
    )

    print(
        "Its current squad/player data must therefore "
        "come from the live refresh layer."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    extend_current_transfermarkt_map()