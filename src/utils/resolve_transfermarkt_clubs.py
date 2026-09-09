from __future__ import annotations

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

CLUBS_FILE = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
    / "clubs.csv"
)

AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "transfermarkt_club_identity_audit.csv"
)

MAPPING_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)

RESOLUTION_REPORT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "transfermarkt_club_resolution.csv"
)

UNRESOLVED_FILE = (
    REPORTS_DIR
    / "metrics"
    / "transfermarkt_clubs_unresolved.csv"
)


# ============================================================
# EXPLICIT VERIFIED OVERRIDES
# ============================================================
#
# These are NOT fuzzy matches.
#
# Every override uses a known Transfermarkt club ID.
#
# Some came directly from our local club audit.
# Others were verified against Transfermarkt.
#
# IMPORTANT:
# Before accepting an override, this script checks that the
# club_id actually exists inside OUR downloaded clubs.csv.
#
# If an ID is absent from our snapshot, it is NOT mapped.
# ============================================================

MANUAL_OVERRIDES = {
    # --------------------------------------------------------
    # DIRECTLY SUPPORTED BY LOCAL AUDIT
    # --------------------------------------------------------

    "Atlético Madrid": {
        "club_id": 13,
        "evidence_name": "Atlético de Madrid",
        "source": "local_audit",
    },

    "Legia Warsaw": {
        "club_id": 255,
        "evidence_name": "Legia Warszawa",
        "source": "local_audit",
    },

    "Austria Wien": {
        "club_id": 14,
        "evidence_name": "Austria Vienna",
        "source": "local_audit",
    },

    "CSKA Moscow": {
        "club_id": 2410,
        "evidence_name": "PFK CSKA Moskva",
        "source": "local_audit",
    },

    "Krasnodar": {
        "club_id": 16704,
        "evidence_name": "FK Krasnodar",
        "source": "local_audit",
    },

    "Lens": {
        "club_id": 826,
        "evidence_name": "RC Lens",
        "source": "local_audit",
    },

    "Lille": {
        "club_id": 1082,
        "evidence_name": "LOSC Lille",
        "source": "local_audit",
    },

    "Montpellier": {
        "club_id": 969,
        "evidence_name": "Montpellier HSC",
        "source": "local_audit",
    },

    "Nordsjælland": {
        "club_id": 2778,
        "evidence_name": "FC Nordsjaelland",
        "source": "local_audit",
    },

    "Rostov": {
        "club_id": 1083,
        "evidence_name": "FK Rostov",
        "source": "local_audit",
    },

    "Rubin Kazan": {
        "club_id": 2698,
        "evidence_name": "FC Rubin Kazan",
        "source": "local_audit",
    },

    "Slavia Prague": {
        "club_id": 62,
        "evidence_name": "SK Slavia Prague",
        "source": "local_audit",
    },

    "Sparta Prague": {
        "club_id": 197,
        "evidence_name": "AC Sparta Prague",
        "source": "local_audit",
    },

    "Spartak Moscow": {
        "club_id": 232,
        "evidence_name": "FK Spartak Moskva",
        "source": "local_audit",
    },

    "Twente": {
        "club_id": 317,
        "evidence_name": "FC Twente Enschede",
        "source": "local_audit",
    },

    "Union Berlin": {
        "club_id": 89,
        "evidence_name": "1.FC Union Berlin",
        "source": "local_audit",
    },

    "İstanbul Başakşehir": {
        "club_id": 6890,
        "evidence_name": "Basaksehir FK",
        "source": "local_audit_and_transfermarkt",
    },

    "Ajax": {
        "club_id": 610,
        "evidence_name": "Ajax Amsterdam",
        "source": "local_audit",
    },

    "Bodø/Glimt": {
        "club_id": 501,
        "evidence_name": "Fotballklubben Bodø/Glimt",
        "source": "local_audit",
    },

    "Lyon": {
        "club_id": 1041,
        "evidence_name": "Olympique Lyon",
        "source": "local_audit",
    },

    "Panathinaikos": {
        "club_id": 265,
        "evidence_name": "Panathinaikos Athlitikos Omilos",
        "source": "local_audit",
    },

    "Partizan": {
        "club_id": 669,
        "evidence_name": "FK Partizan Belgrade",
        "source": "local_audit",
    },

    "RB Salzburg": {
        "club_id": 409,
        "evidence_name": "Fußballclub Red Bull Salzburg",
        "source": "local_audit",
    },

    "Sturm Graz": {
        "club_id": 122,
        "evidence_name": "Sportklub Puntigamer Sturm Graz",
        "source": "local_audit",
    },

    # --------------------------------------------------------
    # VERIFIED TRANSFERMARKT IDS
    # --------------------------------------------------------

    "APOEL": {
        "club_id": 2784,
        "evidence_name": "APOEL Nicosia",
        "source": "transfermarkt_verified",
    },

    "Beşiktaş": {
        "club_id": 114,
        "evidence_name": "Besiktas JK",
        "source": "transfermarkt_verified",
    },

    "Bologna": {
        "club_id": 1025,
        "evidence_name": "Bologna FC 1909",
        "source": "transfermarkt_verified",
    },

    "Lazio": {
        "club_id": 398,
        "evidence_name": "SS Lazio",
        "source": "transfermarkt_verified",
    },

    "Lokomotiv Moscow": {
        "club_id": 932,
        "evidence_name": "Lokomotiv Moscow",
        "source": "transfermarkt_verified",
    },

    "Paphos FC": {
        "club_id": 45457,
        "evidence_name": "Pafos FC",
        "source": "transfermarkt_verified",
    },

    "Roma": {
        "club_id": 12,
        "evidence_name": "AS Roma",
        "source": "transfermarkt_verified",
    },

    "Ferencváros": {
        "club_id": 279,
        "evidence_name": "Ferencvárosi TC",
        "source": "transfermarkt_verified",
    },

    "Ludogorets": {
        "club_id": 31614,
        "evidence_name": "Ludogorets Razgrad",
        "source": "transfermarkt_verified",
    },

    "BATE Borisov": {
        "club_id": 713,
        "evidence_name": "BATE Borisov",
        "source": "transfermarkt_verified",
    },

    "Zenit": {
        "club_id": 964,
        "evidence_name": "Zenit St. Petersburg",
        "source": "transfermarkt_verified",
    },

    "Qarabağ": {
        "club_id": 10625,
        "evidence_name": "Qarabağ FK",
        "source": "transfermarkt_verified",
    },

    "Malmö": {
        "club_id": 496,
        "evidence_name": "Malmö FF",
        "source": "transfermarkt_verified",
    },
}


# ============================================================
# UCL CLUB UNIVERSE
# ============================================================


def get_ucl_teams() -> list[str]:
    """
    Load all canonical UCL teams.
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
            f"UCL dataset missing columns: "
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

    return sorted(
        teams
    )


# ============================================================
# LOAD CLUBS
# ============================================================


def load_transfermarkt_clubs() -> pd.DataFrame:
    """
    Load Transfermarkt club IDs.
    """

    if not CLUBS_FILE.exists():
        raise FileNotFoundError(
            f"Transfermarkt clubs file not found:\n"
            f"{CLUBS_FILE}"
        )

    clubs = pd.read_csv(
        CLUBS_FILE,
        low_memory=False,
    )

    required = {
        "club_id",
        "name",
    }

    missing = (
        required
        - set(
            clubs.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"clubs.csv missing columns: "
            f"{sorted(missing)}"
        )

    clubs = clubs[
        clubs[
            "club_id"
        ].notna()
        & clubs[
            "name"
        ].notna()
    ].copy()

    clubs[
        "club_id"
    ] = pd.to_numeric(
        clubs[
            "club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    if clubs[
        "club_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Transfermarkt club IDs found."
        )

    clubs[
        "name"
    ] = (
        clubs[
            "name"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    return clubs


# ============================================================
# LOAD EXISTING SAFE MAPPING
# ============================================================


def load_existing_mapping() -> pd.DataFrame:
    """
    Load mappings already accepted by the exact audit.
    """

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Existing safe mapping not found:\n"
            f"{MAPPING_FILE}"
        )

    mapping = pd.read_csv(
        MAPPING_FILE,
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
            mapping.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Existing mapping missing columns: "
            f"{sorted(missing)}"
        )

    mapping[
        "club_id"
    ] = pd.to_numeric(
        mapping[
            "club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    mapping = mapping[
        [
            "ucl_team",
            "club_id",
            "transfermarkt_name",
        ]
    ].copy()

    mapping[
        "mapping_method"
    ] = "exact_normalization"

    mapping[
        "mapping_confidence"
    ] = 1.0

    mapping[
        "mapping_source"
    ] = "automatic_exact_audit"

    return mapping


# ============================================================
# RESOLVE MANUAL OVERRIDES
# ============================================================


def resolve_overrides(
    *,
    clubs: pd.DataFrame,
    already_mapped: set[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Resolve explicit club IDs against our downloaded snapshot.

    A manual override is accepted ONLY when its club_id exists
    inside local clubs.csv.
    """

    club_lookup = (
        clubs
        .set_index(
            "club_id"
        )[
            "name"
        ]
        .to_dict()
    )

    resolved_rows = []
    failed_rows = []

    for (
        ucl_team,
        override,
    ) in MANUAL_OVERRIDES.items():

        club_id = int(
            override[
                "club_id"
            ]
        )

        if ucl_team in already_mapped:
            continue

        if club_id not in club_lookup:
            failed_rows.append(
                {
                    "ucl_team": ucl_team,
                    "club_id": club_id,
                    "expected_name": override[
                        "evidence_name"
                    ],
                    "status": (
                        "CLUB_ID_NOT_IN_LOCAL_SNAPSHOT"
                    ),
                    "source": override[
                        "source"
                    ],
                }
            )

            continue

        actual_name = club_lookup[
            club_id
        ]

        resolved_rows.append(
            {
                "ucl_team": ucl_team,
                "club_id": club_id,
                "transfermarkt_name": actual_name,
                "mapping_method": (
                    "explicit_club_id_override"
                ),
                "mapping_confidence": 1.0,
                "mapping_source": override[
                    "source"
                ],
                "evidence_name": override[
                    "evidence_name"
                ],
            }
        )

    resolved = pd.DataFrame(
        resolved_rows
    )

    failed = pd.DataFrame(
        failed_rows
    )

    return (
        resolved,
        failed,
    )


# ============================================================
# COMBINE MAPPINGS
# ============================================================


def combine_mappings(
    *,
    existing: pd.DataFrame,
    manual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge exact and explicit mappings.
    """

    existing = existing.copy()

    existing[
        "evidence_name"
    ] = existing[
        "transfermarkt_name"
    ]

    if manual.empty:
        combined = existing.copy()

    else:
        combined = pd.concat(
            [
                existing,
                manual,
            ],
            ignore_index=True,
        )

    if combined[
        "ucl_team"
    ].duplicated().any():

        duplicates = combined[
            combined[
                "ucl_team"
            ].duplicated(
                keep=False
            )
        ].sort_values(
            by="ucl_team"
        )

        print(
            "\nDuplicate UCL mappings:"
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "A UCL club is mapped more than once."
        )

    if combined[
        "club_id"
    ].duplicated().any():

        duplicates = combined[
            combined[
                "club_id"
            ].duplicated(
                keep=False
            )
        ].sort_values(
            by="club_id"
        )

        print(
            "\nTransfermarkt ID collisions:"
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "One Transfermarkt club ID maps "
            "to multiple UCL clubs."
        )

    combined = combined.sort_values(
        by="ucl_team"
    ).reset_index(
        drop=True
    )

    return combined


# ============================================================
# FULL RESOLUTION REPORT
# ============================================================


def build_resolution_report(
    *,
    ucl_teams: list[str],
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one row per canonical UCL club.
    """

    mapping_lookup = (
        mapping
        .set_index(
            "ucl_team"
        )
        .to_dict(
            orient="index"
        )
    )

    rows = []

    for team in ucl_teams:

        match = mapping_lookup.get(
            team
        )

        if match is None:
            rows.append(
                {
                    "ucl_team": team,
                    "mapped": 0,
                    "club_id": pd.NA,
                    "transfermarkt_name": pd.NA,
                    "mapping_method": pd.NA,
                    "mapping_source": pd.NA,
                    "mapping_confidence": pd.NA,
                }
            )

        else:
            rows.append(
                {
                    "ucl_team": team,
                    "mapped": 1,
                    "club_id": match[
                        "club_id"
                    ],
                    "transfermarkt_name": match[
                        "transfermarkt_name"
                    ],
                    "mapping_method": match[
                        "mapping_method"
                    ],
                    "mapping_source": match[
                        "mapping_source"
                    ],
                    "mapping_confidence": match[
                        "mapping_confidence"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# QUALITY CHECK
# ============================================================


def run_quality_checks(
    *,
    clubs: pd.DataFrame,
    ucl_teams: list[str],
    mapping: pd.DataFrame,
) -> None:
    """
    Validate the resolved mapping.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT RESOLVED MAPPING QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    if len(
        ucl_teams
    ) != 117:
        raise RuntimeError(
            f"Expected 117 canonical UCL clubs, "
            f"found {len(ucl_teams):,}."
        )

    print(
        f"\nCanonical UCL clubs: "
        f"{len(ucl_teams):,} ✓"
    )

    if mapping[
        "ucl_team"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate UCL mappings detected."
        )

    print(
        "Unique UCL mappings: valid ✓"
    )

    if mapping[
        "club_id"
    ].duplicated().any():
        raise RuntimeError(
            "Transfermarkt club ID collision detected."
        )

    print(
        "Unique Transfermarkt club IDs: valid ✓"
    )

    valid_ids = set(
        clubs[
            "club_id"
        ]
    )

    invalid_ids = set(
        mapping[
            "club_id"
        ]
    ) - valid_ids

    if invalid_ids:
        raise RuntimeError(
            f"Mapped IDs missing from clubs.csv: "
            f"{sorted(invalid_ids)}"
        )

    print(
        "All mapped IDs exist in local clubs.csv ✓"
    )

    invalid_teams = set(
        mapping[
            "ucl_team"
        ]
    ) - set(
        ucl_teams
    )

    if invalid_teams:
        raise RuntimeError(
            f"Mapped clubs outside UCL universe: "
            f"{sorted(invalid_teams)}"
        )

    print(
        "All mappings belong to UCL club universe ✓"
    )


# ============================================================
# MAIN
# ============================================================


def resolve_transfermarkt_clubs() -> None:
    """
    Combine exact normalization with manually verified
    Transfermarkt club IDs.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - TRANSFERMARKT CLUB RESOLUTION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    ucl_teams = get_ucl_teams()

    clubs = load_transfermarkt_clubs()

    existing = load_existing_mapping()

    print(
        f"\nCanonical UCL clubs: "
        f"{len(ucl_teams):,}"
    )

    print(
        f"Transfermarkt clubs: "
        f"{len(clubs):,}"
    )

    print(
        f"Existing exact mappings: "
        f"{len(existing):,}"
    )

    # --------------------------------------------------------
    # OVERRIDES
    # --------------------------------------------------------

    (
        manual,
        failed,
    ) = resolve_overrides(
        clubs=clubs,
        already_mapped=set(
            existing[
                "ucl_team"
            ]
        ),
    )

    print(
        f"Validated manual mappings: "
        f"{len(manual):,}"
    )

    print(
        f"Overrides absent from local snapshot: "
        f"{len(failed):,}"
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    mapping = combine_mappings(
        existing=existing,
        manual=manual,
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    run_quality_checks(
        clubs=clubs,
        ucl_teams=ucl_teams,
        mapping=mapping,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = build_resolution_report(
        ucl_teams=ucl_teams,
        mapping=mapping,
    )

    unresolved = report[
        report[
            "mapped"
        ]
        ==
        0
    ].copy()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    MAPPING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESOLUTION_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping.to_csv(
        MAPPING_FILE,
        index=False,
        encoding="utf-8",
    )

    report.to_csv(
        RESOLUTION_REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    unresolved.to_csv(
        UNRESOLVED_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    mapped = len(
        mapping
    )

    total = len(
        ucl_teams
    )

    coverage = (
        mapped
        / total
        * 100
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT CLUB RESOLUTION SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nExact mappings retained: "
        f"{len(existing):,}"
    )

    print(
        f"Manual verified mappings added: "
        f"{len(manual):,}"
    )

    print(
        f"Total mapped UCL clubs: "
        f"{mapped:,} / {total:,}"
    )

    print(
        f"Identity coverage: "
        f"{coverage:.2f}%"
    )

    print(
        f"Remaining unresolved clubs: "
        f"{len(unresolved):,}"
    )

    if not manual.empty:

        print(
            "\nManual mappings accepted:"
        )

        display_columns = [
            "ucl_team",
            "club_id",
            "transfermarkt_name",
            "mapping_source",
        ]

        print(
            manual[
                display_columns
            ]
            .sort_values(
                by="ucl_team"
            )
            .to_string(
                index=False
            )
        )

    if not failed.empty:

        print(
            "\nVerified IDs not present in "
            "our downloaded Transfermarkt snapshot:"
        )

        print(
            failed.to_string(
                index=False
            )
        )

    if not unresolved.empty:

        print(
            "\nRemaining unresolved UCL clubs:"
        )

        for team in unresolved[
            "ucl_team"
        ]:

            print(
                f"  {team}"
            )

    print(
        "\nFinal mapping saved to:"
    )

    print(
        MAPPING_FILE
    )

    print(
        "\nResolution report saved to:"
    )

    print(
        RESOLUTION_REPORT_FILE
    )

    print(
        "\nUnresolved list saved to:"
    )

    print(
        UNRESOLVED_FILE
    )

    print(
        "\nNo fuzzy suggestions were automatically accepted."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    resolve_transfermarkt_clubs()