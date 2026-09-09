from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
)


# ============================================================
# PATHS
# ============================================================

TRANSFERMARKT_RAW_DIR = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
)

PLAYERS_FILE = (
    TRANSFERMARKT_RAW_DIR
    / "players.csv"
)

VALUATIONS_FILE = (
    TRANSFERMARKT_RAW_DIR
    / "player_valuations.csv"
)

APPEARANCES_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_appearances_ucl_clubs.csv"
)

LINEUPS_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_lineups_ucl_clubs.csv"
)

OUTPUT_DIR = (
    INTERIM_DATA_DIR
    / "players"
)

FILTERED_PROFILES_FILE = (
    OUTPUT_DIR
    / "transfermarkt_player_profiles_ucl_players.csv"
)

FILTERED_VALUATIONS_FILE = (
    OUTPUT_DIR
    / "transfermarkt_player_valuations_ucl_players.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "transfermarkt_player_valuation_metadata.json"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

FORCE_REBUILD = False


# ============================================================
# PLAYER PROFILE COLUMNS
# ============================================================

PROFILE_REQUIRED_COLUMNS = {
    "player_id",
    "name",
    "position",
}

PROFILE_PREFERRED_COLUMNS = [
    "player_id",
    "name",
    "current_club_id",
    "current_club_name",
    "country_of_citizenship",
    "country_of_birth",
    "date_of_birth",
    "position",
    "sub_position",
    "foot",
    "height_in_cm",
    "market_value_in_eur",
    "highest_market_value_in_eur",
    "contract_expiration_date",
    "last_season",
]


# ============================================================
# VALUATION COLUMNS
# ============================================================

VALUATION_REQUIRED_COLUMNS = {
    "date",
    "player_id",
    "market_value_in_eur",
}

VALUATION_PREFERRED_COLUMNS = [
    "date",
    "player_id",
    "current_club_name",
    "current_club_id",
    "market_value_in_eur",
    "player_club_domestic_competition_id",
]


# ============================================================
# FILE CHECK
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:
    """
    Fail early if a required source file is missing.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )

    if path.stat().st_size == 0:

        raise RuntimeError(
            f"{description} is empty:\n"
            f"{path}"
        )


# ============================================================
# RELEVANT PLAYER IDS
# ============================================================


def load_relevant_player_ids() -> dict:
    """
    Build the union of players found in our historical
    appearances and lineup tables.

    These are the only players whose profiles/valuations are
    required for our current UCL feature universe.
    """

    print(
        "\nLoading relevant historical player IDs..."
    )

    appearances = pd.read_csv(
        APPEARANCES_FILE,
        usecols=[
            "player_id",
        ],
        low_memory=False,
    )

    lineups = pd.read_csv(
        LINEUPS_FILE,
        usecols=[
            "player_id",
        ],
        low_memory=False,
    )

    appearance_ids = set(
        pd.to_numeric(
            appearances[
                "player_id"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
    )

    lineup_ids = set(
        pd.to_numeric(
            lineups[
                "player_id"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
    )

    union_ids = (
        appearance_ids
        | lineup_ids
    )

    overlap_ids = (
        appearance_ids
        & lineup_ids
    )

    appearance_only = (
        appearance_ids
        - lineup_ids
    )

    lineup_only = (
        lineup_ids
        - appearance_ids
    )

    print(
        f"Appearance players: "
        f"{len(appearance_ids):,}"
    )

    print(
        f"Lineup players: "
        f"{len(lineup_ids):,}"
    )

    print(
        f"Players in both: "
        f"{len(overlap_ids):,}"
    )

    print(
        f"Appearance-only players: "
        f"{len(appearance_only):,}"
    )

    print(
        f"Lineup-only players: "
        f"{len(lineup_only):,}"
    )

    print(
        f"Relevant player union: "
        f"{len(union_ids):,}"
    )

    return {
        "appearance_ids": appearance_ids,
        "lineup_ids": lineup_ids,
        "overlap_ids": overlap_ids,
        "appearance_only_ids": appearance_only,
        "lineup_only_ids": lineup_only,
        "relevant_ids": union_ids,
    }


# ============================================================
# FILTER PLAYER PROFILES
# ============================================================


def prepare_player_profiles(
    relevant_ids: set[int],
) -> pd.DataFrame:
    """
    Filter players.csv to players observed in our historical
    UCL-club appearances or lineup data.

    Static profile attributes such as DOB/position can later
    support age and positional features.

    Current club and current market value fields are retained
    for auditing only. They must NOT be used retroactively
    for historical UCL matches.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FILTERING PLAYER PROFILES"
    )

    print(
        "=" * 75
    )

    header = pd.read_csv(
        PLAYERS_FILE,
        nrows=0,
        low_memory=False,
    )

    missing = (
        PROFILE_REQUIRED_COLUMNS
        - set(
            header.columns
        )
    )

    if missing:

        raise RuntimeError(
            "players.csv missing required columns: "
            f"{sorted(missing)}"
        )

    usecols = [
        column
        for column in PROFILE_PREFERRED_COLUMNS
        if column in header.columns
    ]

    players = pd.read_csv(
        PLAYERS_FILE,
        usecols=usecols,
        low_memory=False,
    )

    players[
        "player_id"
    ] = pd.to_numeric(
        players[
            "player_id"
        ],
        errors="coerce",
    )

    players = players[
        players[
            "player_id"
        ].isin(
            relevant_ids
        )
    ].copy()

    players[
        "player_id"
    ] = players[
        "player_id"
    ].astype(
        int
    )

    players = players.sort_values(
        by="player_id"
    ).reset_index(
        drop=True
    )

    if players[
        "player_id"
    ].duplicated().any():

        duplicates = players[
            players[
                "player_id"
            ].duplicated(
                keep=False
            )
        ]

        print(
            duplicates.head(
                20
            ).to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate player IDs found in "
            "filtered profiles."
        )

    missing_profiles = (
        relevant_ids
        - set(
            players[
                "player_id"
            ]
        )
    )

    print(
        f"\nRelevant players requested: "
        f"{len(relevant_ids):,}"
    )

    print(
        f"Profiles found: "
        f"{len(players):,}"
    )

    print(
        f"Profiles missing: "
        f"{len(missing_profiles):,}"
    )

    if "date_of_birth" in players.columns:

        players[
            "date_of_birth"
        ] = pd.to_datetime(
            players[
                "date_of_birth"
            ],
            errors="coerce",
        )

        print(
            f"Missing date of birth: "
            f"{players['date_of_birth'].isna().sum():,}"
        )

    print(
        "\nPosition distribution:"
    )

    print(
        players[
            "position"
        ]
        .fillna(
            "<missing>"
        )
        .value_counts()
        .to_string()
    )

    players.to_csv(
        FILTERED_PROFILES_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nFiltered profiles saved to:"
    )

    print(
        FILTERED_PROFILES_FILE
    )

    return players


# ============================================================
# FILTER VALUATIONS
# ============================================================


def prepare_player_valuations(
    relevant_ids: set[int],
) -> pd.DataFrame:
    """
    Filter historical market valuations to relevant players.

    Historical membership is NOT inferred from
    current_club_id.

    Later, a player will belong to a club based on historical
    appearance/lineup evidence. The valuation lookup will use:

        valuation_date < UCL_match_date

    and select the latest available value for that player.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FILTERING HISTORICAL PLAYER VALUATIONS"
    )

    print(
        "=" * 75
    )

    header = pd.read_csv(
        VALUATIONS_FILE,
        nrows=0,
        low_memory=False,
    )

    missing = (
        VALUATION_REQUIRED_COLUMNS
        - set(
            header.columns
        )
    )

    if missing:

        raise RuntimeError(
            "player_valuations.csv missing columns: "
            f"{sorted(missing)}"
        )

    usecols = [
        column
        for column in VALUATION_PREFERRED_COLUMNS
        if column in header.columns
    ]

    frames = []

    source_rows = 0
    retained_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            VALUATIONS_FILE,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):

        source_rows += len(
            chunk
        )

        chunk[
            "player_id"
        ] = pd.to_numeric(
            chunk[
                "player_id"
            ],
            errors="coerce",
        )

        filtered = chunk[
            chunk[
                "player_id"
            ].isin(
                relevant_ids
            )
        ].copy()

        if filtered.empty:
            continue

        filtered[
            "player_id"
        ] = filtered[
            "player_id"
        ].astype(
            int
        )

        frames.append(
            filtered
        )

        retained_rows += len(
            filtered
        )

        if (
            chunk_number % 2
            ==
            0
        ):

            print(
                f"  Processed "
                f"{source_rows:,} valuation rows "
                f"| retained "
                f"{retained_rows:,}"
            )

    if not frames:

        raise RuntimeError(
            "No valuation rows found for "
            "relevant players."
        )

    valuations = pd.concat(
        frames,
        ignore_index=True,
    )

    valuations[
        "date"
    ] = pd.to_datetime(
        valuations[
            "date"
        ],
        errors="coerce",
    )

    valuations[
        "market_value_in_eur"
    ] = pd.to_numeric(
        valuations[
            "market_value_in_eur"
        ],
        errors="coerce",
    )

    valuations = valuations.sort_values(
        by=[
            "player_id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"\nSource valuation rows: "
        f"{source_rows:,}"
    )

    print(
        f"Relevant valuation rows: "
        f"{len(valuations):,}"
    )

    print(
        f"Retention rate: "
        f"{len(valuations) / source_rows * 100:.2f}%"
    )

    return valuations


# ============================================================
# VALUATION QUALITY CHECKS
# ============================================================


def audit_valuations(
    *,
    valuations: pd.DataFrame,
    relevant_ids: set[int],
    profiles: pd.DataFrame,
) -> dict:
    """
    Validate historical valuation data before feature
    engineering.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PLAYER VALUATION DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # CORE NULLS
    # --------------------------------------------------------

    core_missing = int(
        valuations[
            [
                "player_id",
                "date",
                "market_value_in_eur",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"\nRows missing player/date/value: "
        f"{core_missing:,}"
    )

    if core_missing > 0:

        raise RuntimeError(
            "Valuation rows contain missing "
            "core values."
        )

    # --------------------------------------------------------
    # UNIQUE PLAYER/DATE
    # --------------------------------------------------------

    duplicate_keys = int(
        valuations.duplicated(
            subset=[
                "player_id",
                "date",
            ]
        ).sum()
    )

    print(
        f"Duplicate player/date records: "
        f"{duplicate_keys:,}"
    )

    if duplicate_keys > 0:

        duplicates = valuations[
            valuations.duplicated(
                subset=[
                    "player_id",
                    "date",
                ],
                keep=False,
            )
        ].head(
            30
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate valuation player/date "
            "keys detected."
        )

    # --------------------------------------------------------
    # VALUE RANGE
    # --------------------------------------------------------

    negative_values = int(
        (
            valuations[
                "market_value_in_eur"
            ]
            <
            0
        ).sum()
    )

    print(
        f"Negative market values: "
        f"{negative_values:,}"
    )

    if negative_values > 0:

        raise RuntimeError(
            "Negative market values detected."
        )

    # --------------------------------------------------------
    # TEMPORAL RANGE
    # --------------------------------------------------------

    minimum_date = valuations[
        "date"
    ].min()

    maximum_date = valuations[
        "date"
    ].max()

    print(
        f"Valuation date range: "
        f"{minimum_date.date()} "
        f"to "
        f"{maximum_date.date()}"
    )

    # --------------------------------------------------------
    # PLAYER COVERAGE
    # --------------------------------------------------------

    valuation_players = set(
        valuations[
            "player_id"
        ].astype(
            int
        )
    )

    covered = (
        relevant_ids
        & valuation_players
    )

    missing = (
        relevant_ids
        - valuation_players
    )

    coverage = (
        len(
            covered
        )
        / len(
            relevant_ids
        )
        * 100
    )

    print(
        f"\nRelevant historical players: "
        f"{len(relevant_ids):,}"
    )

    print(
        f"Players with >=1 valuation: "
        f"{len(covered):,}"
    )

    print(
        f"Players without valuation history: "
        f"{len(missing):,}"
    )

    print(
        f"Player valuation coverage: "
        f"{coverage:.2f}%"
    )

    # --------------------------------------------------------
    # RECORD DENSITY
    # --------------------------------------------------------

    records_per_player = (
        valuations
        .groupby(
            "player_id"
        )
        .size()
    )

    print(
        "\nValuation records per covered player:"
    )

    print(
        f"  minimum: "
        f"{records_per_player.min():,}"
    )

    print(
        f"  median:  "
        f"{records_per_player.median():.1f}"
    )

    print(
        f"  mean:    "
        f"{records_per_player.mean():.2f}"
    )

    print(
        f"  maximum: "
        f"{records_per_player.max():,}"
    )

    # --------------------------------------------------------
    # MISSING PLAYER SAMPLE
    # --------------------------------------------------------

    if missing:

        profile_lookup = (
            profiles[
                [
                    "player_id",
                    "name",
                ]
            ]
            .set_index(
                "player_id"
            )[
                "name"
            ]
            .to_dict()
        )

        print(
            "\nFirst 25 relevant players "
            "without valuation history:"
        )

        for player_id in sorted(
            missing
        )[
            :25
        ]:

            print(
                f"  "
                f"{player_id:<10} "
                f"{profile_lookup.get(player_id, '<unknown>')}"
            )

    print(
        "\nValuation integrity checks: passed ✓"
    )

    return {
        "rows": int(
            len(
                valuations
            )
        ),

        "players_with_valuations": int(
            len(
                valuation_players
            )
        ),

        "relevant_players": int(
            len(
                relevant_ids
            )
        ),

        "players_missing_valuations": int(
            len(
                missing
            )
        ),

        "player_coverage_percent": float(
            coverage
        ),

        "min_date": (
            minimum_date
            .date()
            .isoformat()
        ),

        "max_date": (
            maximum_date
            .date()
            .isoformat()
        ),

        "records_per_player_min": int(
            records_per_player.min()
        ),

        "records_per_player_median": float(
            records_per_player.median()
        ),

        "records_per_player_mean": float(
            records_per_player.mean()
        ),

        "records_per_player_max": int(
            records_per_player.max()
        ),
    }


# ============================================================
# SAVE VALUATIONS
# ============================================================


def save_valuations(
    valuations: pd.DataFrame,
) -> None:
    """
    Save filtered valuation history.
    """

    output = valuations.copy()

    output[
        "date"
    ] = output[
        "date"
    ].dt.strftime(
        "%Y-%m-%d"
    )

    output.to_csv(
        FILTERED_VALUATIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nFiltered valuation history saved to:"
    )

    print(
        FILTERED_VALUATIONS_FILE
    )


# ============================================================
# SAVE METADATA
# ============================================================


def save_metadata(
    *,
    player_sets: dict,
    profiles: pd.DataFrame,
    valuation_audit: dict,
) -> None:
    """
    Save coverage/provenance information.
    """

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "historical_membership_rule": (
            "Club membership will be inferred from "
            "historical appearances / game lineups, "
            "not current_club_id."
        ),

        "valuation_join_rule": (
            "For a UCL match on date t, only valuation "
            "records with valuation_date < t may be used."
        ),

        "current_profile_fields_rule": (
            "players.csv current_club_id and "
            "market_value_in_eur are audit/current-snapshot "
            "fields and must not be retroactively attached "
            "to historical matches."
        ),

        "appearance_players": len(
            player_sets[
                "appearance_ids"
            ]
        ),

        "lineup_players": len(
            player_sets[
                "lineup_ids"
            ]
        ),

        "player_union": len(
            player_sets[
                "relevant_ids"
            ]
        ),

        "player_overlap": len(
            player_sets[
                "overlap_ids"
            ]
        ),

        "profiles_found": int(
            len(
                profiles
            )
        ),

        "valuation_audit": (
            valuation_audit
        ),
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

    print(
        "\nMetadata saved to:"
    )

    print(
        METADATA_FILE
    )


# ============================================================
# MAIN
# ============================================================


def prepare_player_valuation_history() -> None:
    """
    Prepare player profiles and historical market valuations
    for the player population observed at our 117 UCL clubs.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - PLAYER VALUATION PREPARATION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # REQUIRED FILES
    # --------------------------------------------------------

    require_file(
        PLAYERS_FILE,
        "Transfermarkt players.csv",
    )

    require_file(
        VALUATIONS_FILE,
        "Transfermarkt player_valuations.csv",
    )

    require_file(
        APPEARANCES_FILE,
        "Filtered appearances",
    )

    require_file(
        LINEUPS_FILE,
        "Filtered lineups",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # PLAYER POPULATION
    # --------------------------------------------------------

    player_sets = load_relevant_player_ids()

    relevant_ids = player_sets[
        "relevant_ids"
    ]

    # --------------------------------------------------------
    # PROFILES
    # --------------------------------------------------------

    profiles = prepare_player_profiles(
        relevant_ids
    )

    # --------------------------------------------------------
    # VALUATIONS
    # --------------------------------------------------------

    if (
        FILTERED_VALUATIONS_FILE.exists()
        and not FORCE_REBUILD
    ):

        print(
            "\nUsing cached filtered valuation history:"
        )

        print(
            FILTERED_VALUATIONS_FILE
        )

        valuations = pd.read_csv(
            FILTERED_VALUATIONS_FILE,
            low_memory=False,
        )

        valuations[
            "date"
        ] = pd.to_datetime(
            valuations[
                "date"
            ],
            errors="coerce",
        )

        valuations[
            "market_value_in_eur"
        ] = pd.to_numeric(
            valuations[
                "market_value_in_eur"
            ],
            errors="coerce",
        )

    else:

        valuations = prepare_player_valuations(
            relevant_ids
        )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    valuation_audit = audit_valuations(
        valuations=valuations,
        relevant_ids=relevant_ids,
        profiles=profiles,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if (
        not FILTERED_VALUATIONS_FILE.exists()
        or FORCE_REBUILD
    ):

        save_valuations(
            valuations
        )

    save_metadata(
        player_sets=player_sets,
        profiles=profiles,
        valuation_audit=valuation_audit,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PLAYER VALUATION PREPARATION SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nRelevant historical players: "
        f"{len(relevant_ids):,}"
    )

    print(
        f"Player profiles retained: "
        f"{len(profiles):,}"
    )

    print(
        f"Historical valuation rows retained: "
        f"{valuation_audit['rows']:,}"
    )

    print(
        f"Players with valuation history: "
        f"{valuation_audit['players_with_valuations']:,}"
    )

    print(
        f"Valuation player coverage: "
        f"{valuation_audit['player_coverage_percent']:.2f}%"
    )

    print(
        "\nProfiles:"
    )

    print(
        FILTERED_PROFILES_FILE
    )

    print(
        "\nValuations:"
    )

    print(
        FILTERED_VALUATIONS_FILE
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PLAYER VALUATION PREPARATION COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nLeakage rule for the next step:"
    )

    print(
        "    valuation_date < UCL_match_date"
    )

    print(
        "\nHistorical club membership will come from "
        "appearances/lineups, not current club fields."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    prepare_player_valuation_history()