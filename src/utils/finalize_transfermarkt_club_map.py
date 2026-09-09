from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)


# ============================================================
# PATHS
# ============================================================

UCL_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

TRANSFERMARKT_DIR = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
)

CLUBS_FILE = (
    TRANSFERMARKT_DIR
    / "clubs.csv"
)

GAMES_FILE = (
    TRANSFERMARKT_DIR
    / "games.csv"
)

PLAYERS_FILE = (
    TRANSFERMARKT_DIR
    / "players.csv"
)

TRANSFERS_FILE = (
    TRANSFERMARKT_DIR
    / "transfers.csv"
)

CURRENT_MAPPING_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)

GAME_AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "transfermarkt_game_club_audit.csv"
)

FINAL_REPORT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "transfermarkt_club_map_final_audit.csv"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_UCL_CLUBS = 117

EXPECTED_GAME_DERIVED_MATCHES = 9


# ============================================================
# FINAL EXPLICIT VERIFIED IDS
# ============================================================
#
# These clubs were not safely resolved by the general club
# name audit.
#
# IDs were verified independently against Transfermarkt.
#
# IMPORTANT:
#
# "Steaua" -> 301 is a dataset-continuity mapping to the
# Transfermarkt/FCSB historical entity used for the relevant
# European match history.
# ============================================================

VERIFIED_OVERRIDES = {
    "AEK Athens": {
        "club_id": 2441,
        "evidence_name": "AEK Athens",
        "verification_source": "transfermarkt_verified",
    },

    "APOEL": {
        "club_id": 2784,
        "evidence_name": "APOEL Nicosia",
        "verification_source": "transfermarkt_verified",
    },

    "FK Kairat": {
        "club_id": 10482,
        "evidence_name": "Kairat Almaty",
        "verification_source": "transfermarkt_verified",
    },

    "Hapoel Tel Aviv": {
        "club_id": 1017,
        "evidence_name": "Hapoel Tel Aviv",
        "verification_source": "transfermarkt_verified",
    },

    "Ludogorets": {
        "club_id": 31614,
        "evidence_name": "Ludogorets Razgrad",
        "verification_source": "transfermarkt_verified",
    },

    "MŠK Žilina": {
        "club_id": 1443,
        "evidence_name": "MSK Zilina",
        "verification_source": "transfermarkt_verified",
    },

    "Olympiacos": {
        "club_id": 683,
        "evidence_name": (
            "Olympiakos Syndesmos Filathlon Peiraios"
        ),
        "verification_source": (
            "transfermarkt_verified_and_ucl_games"
        ),
    },

    "Paphos FC": {
        "club_id": 45457,
        "evidence_name": "Pafos FC",
        "verification_source": "transfermarkt_verified",
    },

    "Steaua": {
        "club_id": 301,
        "evidence_name": "FCSB",
        "verification_source": (
            "transfermarkt_historical_continuity"
        ),
    },
}


# ============================================================
# LOAD UCL CLUB UNIVERSE
# ============================================================


def load_ucl_teams() -> list[str]:
    """
    Return the canonical 117-team UCL universe.
    """

    if not UCL_FILE.exists():
        raise FileNotFoundError(
            f"UCL master file not found:\n"
            f"{UCL_FILE}"
        )

    dataframe = pd.read_csv(
        UCL_FILE,
        low_memory=False,
    )

    required = {
        "home_team",
        "away_team",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"UCL master dataset missing columns: "
            f"{sorted(missing)}"
        )

    teams = set(
        dataframe[
            "home_team"
        ].dropna()
    )

    teams.update(
        dataframe[
            "away_team"
        ].dropna()
    )

    teams = sorted(
        teams
    )

    return teams


# ============================================================
# LOAD EXISTING 99-CLUB MAP
# ============================================================


def load_existing_mapping() -> pd.DataFrame:
    """
    Load the previously validated exact/manual map.
    """

    if not CURRENT_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Existing mapping not found:\n"
            f"{CURRENT_MAPPING_FILE}"
        )

    dataframe = pd.read_csv(
        CURRENT_MAPPING_FILE,
        low_memory=False,
    )

    required = {
        "ucl_team",
        "club_id",
        "transfermarkt_name",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Existing mapping missing columns: "
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

    # Preserve existing provenance where available.
    if (
        "mapping_method"
        not in dataframe.columns
    ):
        dataframe[
            "mapping_method"
        ] = "existing_validated_mapping"

    if (
        "mapping_source"
        not in dataframe.columns
    ):
        dataframe[
            "mapping_source"
        ] = "previous_resolution"

    if (
        "mapping_confidence"
        not in dataframe.columns
    ):
        dataframe[
            "mapping_confidence"
        ] = 1.0

    dataframe = dataframe[
        [
            "ucl_team",
            "club_id",
            "transfermarkt_name",
            "mapping_method",
            "mapping_source",
            "mapping_confidence",
        ]
    ].copy()

    return dataframe


# ============================================================
# LOAD GAME AUDIT
# ============================================================


def load_game_matches() -> pd.DataFrame:
    """
    Load exact canonical matches recovered from real
    Transfermarkt Champions League games.
    """

    if not GAME_AUDIT_FILE.exists():
        raise FileNotFoundError(
            f"Game club audit not found:\n"
            f"{GAME_AUDIT_FILE}"
        )

    audit = pd.read_csv(
        GAME_AUDIT_FILE,
        low_memory=False,
    )

    required = {
        "ucl_team",
        "status",
        "club_id",
        "transfermarkt_name",
    }

    missing = (
        required
        - set(
            audit.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Game audit missing columns: "
            f"{sorted(missing)}"
        )

    matched = audit[
        audit[
            "status"
        ]
        ==
        "MATCHED"
    ].copy()

    matched[
        "club_id"
    ] = pd.to_numeric(
        matched[
            "club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    matched[
        "mapping_method"
    ] = "ucl_game_exact_identity"

    matched[
        "mapping_source"
    ] = "transfermarkt_ucl_games"

    matched[
        "mapping_confidence"
    ] = 1.0

    matched = matched[
        [
            "ucl_team",
            "club_id",
            "transfermarkt_name",
            "mapping_method",
            "mapping_source",
            "mapping_confidence",
        ]
    ]

    return matched


# ============================================================
# BUILD VERIFIED OVERRIDE TABLE
# ============================================================


def build_verified_overrides() -> pd.DataFrame:
    """
    Convert final verified IDs into a dataframe.
    """

    rows = []

    for (
        ucl_team,
        data,
    ) in VERIFIED_OVERRIDES.items():

        rows.append(
            {
                "ucl_team": ucl_team,
                "club_id": int(
                    data[
                        "club_id"
                    ]
                ),
                "transfermarkt_name": (
                    data[
                        "evidence_name"
                    ]
                ),
                "mapping_method": (
                    "explicit_verified_club_id"
                ),
                "mapping_source": (
                    data[
                        "verification_source"
                    ]
                ),
                "mapping_confidence": 1.0,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOCAL ID EVIDENCE
# ============================================================


def add_evidence(
    evidence: dict[
        int,
        set[str],
    ],
    club_ids: pd.Series,
    source: str,
) -> None:
    """
    Record which local Transfermarkt tables contain each
    club ID.
    """

    values = pd.to_numeric(
        club_ids,
        errors="coerce",
    ).dropna()

    for club_id in values:

        evidence[
            int(
                club_id
            )
        ].add(
            source
        )


def build_local_id_evidence() -> dict[
    int,
    set[str],
]:
    """
    Build a union of club IDs observed throughout our local
    Transfermarkt snapshot.

    This is deliberately broader than clubs.csv because
    historical clubs may survive in games/transfers even when
    absent from the current club table.
    """

    evidence: dict[
        int,
        set[str],
    ] = defaultdict(
        set
    )

    # --------------------------------------------------------
    # CLUBS
    # --------------------------------------------------------

    if CLUBS_FILE.exists():

        clubs = pd.read_csv(
            CLUBS_FILE,
            usecols=[
                "club_id"
            ],
            low_memory=False,
        )

        add_evidence(
            evidence,
            clubs[
                "club_id"
            ],
            "clubs",
        )

    # --------------------------------------------------------
    # GAMES
    # --------------------------------------------------------

    if GAMES_FILE.exists():

        games = pd.read_csv(
            GAMES_FILE,
            usecols=[
                "home_club_id",
                "away_club_id",
            ],
            low_memory=False,
        )

        add_evidence(
            evidence,
            games[
                "home_club_id"
            ],
            "games",
        )

        add_evidence(
            evidence,
            games[
                "away_club_id"
            ],
            "games",
        )

    # --------------------------------------------------------
    # PLAYERS
    # --------------------------------------------------------

    if PLAYERS_FILE.exists():

        players = pd.read_csv(
            PLAYERS_FILE,
            usecols=[
                "current_club_id"
            ],
            low_memory=False,
        )

        add_evidence(
            evidence,
            players[
                "current_club_id"
            ],
            "players",
        )

    # --------------------------------------------------------
    # TRANSFERS
    # --------------------------------------------------------

    if TRANSFERS_FILE.exists():

        transfers = pd.read_csv(
            TRANSFERS_FILE,
            usecols=[
                "from_club_id",
                "to_club_id",
            ],
            low_memory=False,
        )

        add_evidence(
            evidence,
            transfers[
                "from_club_id"
            ],
            "transfers",
        )

        add_evidence(
            evidence,
            transfers[
                "to_club_id"
            ],
            "transfers",
        )

    return evidence


# ============================================================
# LOCAL CLUB NAME LOOKUP
# ============================================================


def build_local_name_lookup() -> dict[
    int,
    str,
]:
    """
    Find a useful local Transfermarkt name for each ID.

    Priority:
        clubs.csv
        games.csv

    Verified evidence names remain fallback values.
    """

    lookup = {}

    # --------------------------------------------------------
    # CLUB TABLE
    # --------------------------------------------------------

    if CLUBS_FILE.exists():

        clubs = pd.read_csv(
            CLUBS_FILE,
            usecols=[
                "club_id",
                "name",
            ],
            low_memory=False,
        )

        clubs[
            "club_id"
        ] = pd.to_numeric(
            clubs[
                "club_id"
            ],
            errors="coerce",
        )

        clubs = clubs.dropna(
            subset=[
                "club_id",
                "name",
            ]
        )

        for row in clubs.itertuples(
            index=False
        ):

            lookup[
                int(
                    row.club_id
                )
            ] = str(
                row.name
            ).strip()

    # --------------------------------------------------------
    # GAME TABLE
    # --------------------------------------------------------

    if GAMES_FILE.exists():

        games = pd.read_csv(
            GAMES_FILE,
            usecols=[
                "home_club_id",
                "home_club_name",
                "away_club_id",
                "away_club_name",
            ],
            low_memory=False,
        )

        for (
            id_column,
            name_column,
        ) in [
            (
                "home_club_id",
                "home_club_name",
            ),
            (
                "away_club_id",
                "away_club_name",
            ),
        ]:

            subset = games[
                [
                    id_column,
                    name_column,
                ]
            ].dropna()

            for row in subset.itertuples(
                index=False,
                name=None,
            ):

                club_id = pd.to_numeric(
                    row[
                        0
                    ],
                    errors="coerce",
                )

                if pd.isna(
                    club_id
                ):
                    continue

                club_id = int(
                    club_id
                )

                if club_id in lookup:
                    continue

                name = str(
                    row[
                        1
                    ]
                ).strip()

                if name:
                    lookup[
                        club_id
                    ] = name

    return lookup


# ============================================================
# COMBINE MAPPINGS
# ============================================================


def combine_mapping_layers(
    *,
    existing: pd.DataFrame,
    game_matches: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all three mapping layers.

    Existing validated mappings have priority.
    Game-derived mappings come second.
    Explicit verified IDs fill the final gaps.
    """

    frames = [
        existing.copy(),
        game_matches.copy(),
        overrides.copy(),
    ]

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # SAME UCL CLUB APPEARING IN MULTIPLE LAYERS
    # --------------------------------------------------------
    #
    # This is allowed only if every layer agrees on club_id.
    # --------------------------------------------------------

    conflicts = []

    for (
        ucl_team,
        group,
    ) in combined.groupby(
        "ucl_team"
    ):

        ids = set(
            group[
                "club_id"
            ].astype(
                int
            )
        )

        if len(
            ids
        ) > 1:

            conflicts.append(
                (
                    ucl_team,
                    sorted(
                        ids
                    ),
                )
            )

    if conflicts:

        print(
            "\nConflicting mappings detected:"
        )

        for (
            team,
            ids,
        ) in conflicts:

            print(
                f"  {team}: "
                f"{ids}"
            )

        raise RuntimeError(
            "Mapping layers disagree on club IDs."
        )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    priority = {
        "existing_validated_mapping": 0,
        "explicit_normalization": 0,
        "exact_normalization": 0,
        "ucl_game_exact_identity": 1,
        "explicit_verified_club_id": 2,
    }

    combined[
        "_priority"
    ] = (
        combined[
            "mapping_method"
        ]
        .map(
            priority
        )
        .fillna(
            0
        )
    )

    combined = (
        combined
        .sort_values(
            by=[
                "ucl_team",
                "_priority",
            ]
        )
        .drop_duplicates(
            subset=[
                "ucl_team"
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_priority"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return combined


# ============================================================
# ENRICH WITH LOCAL EVIDENCE
# ============================================================


def enrich_with_local_evidence(
    *,
    mapping: pd.DataFrame,
    evidence: dict[
        int,
        set[str],
    ],
    name_lookup: dict[
        int,
        str,
    ],
) -> pd.DataFrame:
    """
    Record whether mapped IDs exist anywhere in the local
    downloaded Transfermarkt snapshot.
    """

    output = mapping.copy()

    local_available = []

    local_sources = []

    resolved_names = []

    for row in output.itertuples(
        index=False
    ):

        club_id = int(
            row.club_id
        )

        sources = sorted(
            evidence.get(
                club_id,
                set(),
            )
        )

        local_available.append(
            int(
                bool(
                    sources
                )
            )
        )

        local_sources.append(
            "|".join(
                sources
            )
        )

        local_name = name_lookup.get(
            club_id
        )

        if local_name:
            resolved_names.append(
                local_name
            )
        else:
            resolved_names.append(
                row.transfermarkt_name
            )

    output[
        "transfermarkt_name"
    ] = resolved_names

    output[
        "local_id_observed"
    ] = local_available

    output[
        "local_evidence_sources"
    ] = local_sources

    return output


# ============================================================
# QUALITY CHECK
# ============================================================


def run_quality_checks(
    *,
    ucl_teams: list[str],
    mapping: pd.DataFrame,
) -> None:
    """
    Final identity integrity checks.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL TRANSFERMARKT CLUB MAP QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    if len(
        ucl_teams
    ) != EXPECTED_UCL_CLUBS:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_UCL_CLUBS} UCL clubs, "
            f"found {len(ucl_teams)}."
        )

    print(
        f"\nCanonical UCL clubs: "
        f"{len(ucl_teams):,} ✓"
    )

    if len(
        mapping
    ) != EXPECTED_UCL_CLUBS:

        missing = (
            set(
                ucl_teams
            )
            - set(
                mapping[
                    "ucl_team"
                ]
            )
        )

        raise RuntimeError(
            f"Final mapping has "
            f"{len(mapping):,} rows. "
            f"Missing teams: "
            f"{sorted(missing)}"
        )

    print(
        f"Mapped clubs: "
        f"{len(mapping):,} ✓"
    )

    if mapping[
        "ucl_team"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate canonical UCL team mappings."
        )

    print(
        "Unique UCL identities: valid ✓"
    )

    if mapping[
        "club_id"
    ].duplicated().any():

        duplicates = mapping[
            mapping[
                "club_id"
            ].duplicated(
                keep=False
            )
        ].sort_values(
            by="club_id"
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Transfermarkt club ID collision detected."
        )

    print(
        "Unique Transfermarkt club IDs: valid ✓"
    )

    mapped_teams = set(
        mapping[
            "ucl_team"
        ]
    )

    expected_teams = set(
        ucl_teams
    )

    if mapped_teams != expected_teams:

        raise RuntimeError(
            "Final mapped UCL universe does not match "
            "master match dataset."
        )

    print(
        "Mapped universe matches UCL master ✓"
    )

    if mapping[
        "club_id"
    ].isna().any():

        raise RuntimeError(
            "Missing Transfermarkt club IDs detected."
        )

    print(
        "Missing club IDs: 0 ✓"
    )


# ============================================================
# FINAL REPORT
# ============================================================


def build_final_report(
    *,
    ucl_teams: list[str],
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one audited row per UCL club.
    """

    report = mapping.copy()

    report[
        "identity_mapped"
    ] = 1

    report[
        "ucl_team"
    ] = pd.Categorical(
        report[
            "ucl_team"
        ],
        categories=sorted(
            ucl_teams
        ),
        ordered=True,
    )

    report = (
        report
        .sort_values(
            by="ucl_team"
        )
        .reset_index(
            drop=True
        )
    )

    report[
        "ucl_team"
    ] = report[
        "ucl_team"
    ].astype(
        str
    )

    return report


# ============================================================
# MAIN
# ============================================================


def finalize_transfermarkt_club_map() -> None:
    """
    Produce the final canonical UCL -> Transfermarkt ID map.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - FINAL TRANSFERMARKT CLUB MAPPING"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    ucl_teams = load_ucl_teams()

    existing = load_existing_mapping()

    game_matches = load_game_matches()

    overrides = build_verified_overrides()

    print(
        f"\nCanonical UCL clubs: "
        f"{len(ucl_teams):,}"
    )

    print(
        f"Existing validated mappings: "
        f"{len(existing):,}"
    )

    print(
        f"Game-derived exact mappings: "
        f"{len(game_matches):,}"
    )

    print(
        f"Final explicit verified IDs: "
        f"{len(overrides):,}"
    )

    if (
        len(
            game_matches
        )
        !=
        EXPECTED_GAME_DERIVED_MATCHES
    ):

        print(
            "\nWARNING:"
        )

        print(
            f"Expected "
            f"{EXPECTED_GAME_DERIVED_MATCHES} "
            f"game-derived matches but found "
            f"{len(game_matches)}."
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    mapping = combine_mapping_layers(
        existing=existing,
        game_matches=game_matches,
        overrides=overrides,
    )

    # --------------------------------------------------------
    # LOCAL SNAPSHOT EVIDENCE
    # --------------------------------------------------------

    print(
        "\nBuilding local Transfermarkt ID evidence..."
    )

    evidence = build_local_id_evidence()

    name_lookup = build_local_name_lookup()

    mapping = enrich_with_local_evidence(
        mapping=mapping,
        evidence=evidence,
        name_lookup=name_lookup,
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    run_quality_checks(
        ucl_teams=ucl_teams,
        mapping=mapping,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = build_final_report(
        ucl_teams=ucl_teams,
        mapping=mapping,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    CURRENT_MAPPING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping.to_csv(
        CURRENT_MAPPING_FILE,
        index=False,
        encoding="utf-8",
    )

    report.to_csv(
        FINAL_REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    local_count = int(
        mapping[
            "local_id_observed"
        ].sum()
    )

    external_only = mapping[
        mapping[
            "local_id_observed"
        ]
        ==
        0
    ].copy()

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL TRANSFERMARKT CLUB MAP SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nIdentity mappings: "
        f"{len(mapping):,} / "
        f"{len(ucl_teams):,}"
    )

    print(
        "Identity coverage: 100.00%"
    )

    print(
        f"\nIDs observed somewhere in local "
        f"Transfermarkt tables: "
        f"{local_count:,} / "
        f"{len(mapping):,}"
    )

    print(
        f"Local ID coverage: "
        f"{local_count / len(mapping) * 100:.2f}%"
    )

    print(
        f"Externally verified IDs not observed "
        f"locally: "
        f"{len(external_only):,}"
    )

    # --------------------------------------------------------
    # GAME-DERIVED ADDITIONS
    # --------------------------------------------------------

    print(
        "\nGame-derived mappings added:"
    )

    game_display = mapping[
        mapping[
            "mapping_method"
        ]
        ==
        "ucl_game_exact_identity"
    ][
        [
            "ucl_team",
            "club_id",
            "transfermarkt_name",
            "local_evidence_sources",
        ]
    ]

    if game_display.empty:

        print(
            "  None"
        )

    else:

        print(
            game_display.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # FINAL VERIFIED ADDITIONS
    # --------------------------------------------------------

    print(
        "\nFinal explicitly verified mappings:"
    )

    verified_display = mapping[
        mapping[
            "mapping_method"
        ]
        ==
        "explicit_verified_club_id"
    ][
        [
            "ucl_team",
            "club_id",
            "transfermarkt_name",
            "mapping_source",
            "local_id_observed",
            "local_evidence_sources",
        ]
    ]

    if verified_display.empty:

        print(
            "  None"
        )

    else:

        print(
            verified_display.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # EXTERNAL ONLY
    # --------------------------------------------------------

    if not external_only.empty:

        print(
            "\nIDs not found in the currently "
            "downloaded local snapshot:"
        )

        print(
            external_only[
                [
                    "ucl_team",
                    "club_id",
                    "transfermarkt_name",
                    "mapping_source",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            "\nThese identities remain valid, but "
            "player/squad features for them must carry "
            "an explicit data-availability flag."
        )

    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    print(
        "\nFinal mapping saved to:"
    )

    print(
        CURRENT_MAPPING_FILE
    )

    print(
        "\nFinal audit report saved to:"
    )

    print(
        FINAL_REPORT_FILE
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT CLUB IDENTITY LAYER COMPLETE"
    )

    print(
        "=" * 75
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    finalize_transfermarkt_club_map()