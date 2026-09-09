from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)


# ============================================================
# PATHS
# ============================================================

FIXTURE_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_fixtures.csv"
)

HISTORICAL_MASTER_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

OUTPUT_FIXTURE_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_fixtures_resolved.csv"
)

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_team_identity_resolution.csv"
)

METADATA_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_identity_metadata.json"
)


# ============================================================
# EXPLICIT CURRENT-SEASON ALIASES
# ============================================================

CURRENT_TEAM_ALIASES = {
    "PAE AEK": "AEK Athens",
}


# ============================================================
# EXPECTED CURRENT SEASON
# ============================================================

EXPECTED_CURRENT_CLUBS = 36


# ============================================================
# LOAD FIXTURES
# ============================================================


def load_fixtures() -> pd.DataFrame:
    """
    Load current 2026/27 UCL fixture snapshot.
    """

    if not FIXTURE_FILE.exists():

        raise FileNotFoundError(
            "Current UCL fixture snapshot not found:\n"
            f"{FIXTURE_FILE}"
        )

    dataframe = pd.read_csv(
        FIXTURE_FILE,
        low_memory=False,
    )

    required = {
        "source_match_id",
        "raw_home_team",
        "home_team",
        "home_team_id",
        "raw_away_team",
        "away_team",
        "away_team_id",
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
            "Fixture file missing required columns: "
            f"{sorted(missing)}"
        )

    return dataframe


# ============================================================
# HISTORICAL IDENTITIES
# ============================================================


def load_historical_teams() -> set[str]:
    """
    Load canonical teams present in the historical
    2010-2025 UCL training universe.
    """

    if not HISTORICAL_MASTER_FILE.exists():

        raise FileNotFoundError(
            "Historical UCL master not found:\n"
            f"{HISTORICAL_MASTER_FILE}"
        )

    dataframe = pd.read_csv(
        HISTORICAL_MASTER_FILE,
        low_memory=False,
    )

    required = {
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
            "Historical master missing columns: "
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

    return teams


# ============================================================
# EXPLICIT RESOLUTION
# ============================================================


def resolve_team_name(
    name: str,
) -> str:
    """
    Resolve only explicitly approved aliases.

    No fuzzy matching.
    No automatic nearest-name matching.
    """

    return CURRENT_TEAM_ALIASES.get(
        name,
        name,
    )


# ============================================================
# BUILD CURRENT TEAM MAP
# ============================================================


def build_current_team_map(
    fixtures: pd.DataFrame,
    historical_teams: set[str],
) -> pd.DataFrame:
    """
    Build one identity row per current 2026/27 UCL club.
    """

    home = fixtures[
        [
            "raw_home_team",
            "home_team",
            "home_team_id",
        ]
    ].rename(
        columns={
            "raw_home_team": "raw_team",
            "home_team": "collector_team",
            "home_team_id": "football_data_team_id",
        }
    )

    away = fixtures[
        [
            "raw_away_team",
            "away_team",
            "away_team_id",
        ]
    ].rename(
        columns={
            "raw_away_team": "raw_team",
            "away_team": "collector_team",
            "away_team_id": "football_data_team_id",
        }
    )

    teams = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    teams = teams.drop_duplicates(
        subset=[
            "football_data_team_id",
        ]
    ).reset_index(
        drop=True
    )

    rows = []

    for row in teams.itertuples(
        index=False
    ):

        resolved_team = resolve_team_name(
            row.collector_team
        )

        alias_applied = (
            resolved_team
            !=
            row.collector_team
        )

        historical = (
            resolved_team
            in
            historical_teams
        )

        if historical:

            identity_status = (
                "HISTORICAL"
            )

        else:

            identity_status = (
                "NEW_2026_27"
            )

        rows.append(
            {
                "football_data_team_id": (
                    row.football_data_team_id
                ),

                "raw_team": (
                    row.raw_team
                ),

                "collector_team": (
                    row.collector_team
                ),

                "canonical_team": (
                    resolved_team
                ),

                "alias_applied": int(
                    alias_applied
                ),

                "historical_ucl_identity": int(
                    historical
                ),

                "identity_status": (
                    identity_status
                ),
            }
        )

    team_map = pd.DataFrame(
        rows
    )

    team_map = team_map.sort_values(
        by=[
            "identity_status",
            "canonical_team",
        ]
    ).reset_index(
        drop=True
    )

    return team_map


# ============================================================
# APPLY TEAM MAP TO FIXTURES
# ============================================================


def apply_team_map(
    fixtures: pd.DataFrame,
    team_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply resolved current-season canonical identities
    to every fixture.
    """

    home_map = team_map[
        [
            "football_data_team_id",
            "canonical_team",
            "identity_status",
        ]
    ].rename(
        columns={
            "football_data_team_id": (
                "home_team_id"
            ),

            "canonical_team": (
                "resolved_home_team"
            ),

            "identity_status": (
                "home_identity_status"
            ),
        }
    )

    away_map = team_map[
        [
            "football_data_team_id",
            "canonical_team",
            "identity_status",
        ]
    ].rename(
        columns={
            "football_data_team_id": (
                "away_team_id"
            ),

            "canonical_team": (
                "resolved_away_team"
            ),

            "identity_status": (
                "away_identity_status"
            ),
        }
    )

    resolved = fixtures.merge(
        home_map,
        on="home_team_id",
        how="left",
        validate="many_to_one",
    )

    resolved = resolved.merge(
        away_map,
        on="away_team_id",
        how="left",
        validate="many_to_one",
    )

    missing_resolutions = int(
        resolved[
            [
                "resolved_home_team",
                "resolved_away_team",
            ]
        ].isna().sum().sum()
    )

    if missing_resolutions:

        raise RuntimeError(
            "Some fixture teams were not resolved."
        )

    # Keep original collector values for audit.
    resolved[
        "collector_home_team"
    ] = resolved[
        "home_team"
    ]

    resolved[
        "collector_away_team"
    ] = resolved[
        "away_team"
    ]

    # From this point onward these become the canonical
    # current-season identities.
    resolved[
        "home_team"
    ] = resolved[
        "resolved_home_team"
    ]

    resolved[
        "away_team"
    ] = resolved[
        "resolved_away_team"
    ]

    resolved = resolved.drop(
        columns=[
            "resolved_home_team",
            "resolved_away_team",
        ]
    )

    return resolved


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    fixtures: pd.DataFrame,
    team_map: pd.DataFrame,
    historical_teams: set[str],
) -> None:
    """
    Validate current identity resolution.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT UCL TEAM IDENTITY QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    club_count = len(
        team_map
    )

    print(
        f"\nCurrent clubs: "
        f"{club_count}"
    )

    if (
        club_count
        !=
        EXPECTED_CURRENT_CLUBS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_CURRENT_CLUBS} "
            f"current clubs, found {club_count}."
        )

    duplicate_api_ids = int(
        team_map[
            "football_data_team_id"
        ].duplicated().sum()
    )

    print(
        f"Duplicate football-data team IDs: "
        f"{duplicate_api_ids}"
    )

    if duplicate_api_ids:

        raise RuntimeError(
            "Duplicate football-data team IDs."
        )

    duplicate_names = int(
        team_map[
            "canonical_team"
        ].duplicated().sum()
    )

    print(
        f"Duplicate resolved canonical teams: "
        f"{duplicate_names}"
    )

    if duplicate_names:

        raise RuntimeError(
            "Two current clubs resolve to "
            "the same canonical identity."
        )

    aliases = team_map[
        team_map[
            "alias_applied"
        ]
        ==
        1
    ]

    print(
        f"Explicit aliases applied: "
        f"{len(aliases)}"
    )

    historical_count = int(
        (
            team_map[
                "identity_status"
            ]
            ==
            "HISTORICAL"
        ).sum()
    )

    new_count = int(
        (
            team_map[
                "identity_status"
            ]
            ==
            "NEW_2026_27"
        ).sum()
    )

    print(
        f"Historical identities: "
        f"{historical_count}"
    )

    print(
        f"New 2026/27 identities: "
        f"{new_count}"
    )

    print(
        f"Historical coverage: "
        f"{historical_count / club_count * 100:.2f}%"
    )

    # --------------------------------------------------------
    # VERIFY HISTORICAL FLAGS
    # --------------------------------------------------------

    incorrectly_historical = team_map[
        (
            team_map[
                "identity_status"
            ]
            ==
            "HISTORICAL"
        )
        &
        ~team_map[
            "canonical_team"
        ].isin(
            historical_teams
        )
    ]

    if not incorrectly_historical.empty:

        raise RuntimeError(
            "Historical identity validation failed."
        )

    # --------------------------------------------------------
    # FIXTURE IDENTITY
    # --------------------------------------------------------

    missing_fixture_identity = int(
        fixtures[
            [
                "home_team",
                "away_team",
                "home_identity_status",
                "away_identity_status",
            ]
        ].isna().sum().sum()
    )

    print(
        f"Missing resolved fixture identities: "
        f"{missing_fixture_identity}"
    )

    if missing_fixture_identity:

        raise RuntimeError(
            "Missing resolved fixture identity."
        )

    same_team_matches = int(
        (
            fixtures[
                "home_team"
            ]
            ==
            fixtures[
                "away_team"
            ]
        ).sum()
    )

    print(
        f"Same-team fixtures after resolution: "
        f"{same_team_matches}"
    )

    if same_team_matches:

        raise RuntimeError(
            "Invalid same-team fixture after "
            "identity resolution."
        )


# ============================================================
# PRINT SUMMARY
# ============================================================


def print_summary(
    team_map: pd.DataFrame,
) -> None:
    """
    Print resolved and genuinely new identities.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 CURRENT TEAM IDENTITY SUMMARY"
    )

    print(
        "=" * 75
    )

    aliases = team_map[
        team_map[
            "alias_applied"
        ]
        ==
        1
    ]

    print(
        "\nExplicit aliases resolved:"
    )

    if aliases.empty:

        print(
            "  None"
        )

    else:

        print(
            aliases[
                [
                    "collector_team",
                    "canonical_team",
                    "football_data_team_id",
                ]
            ].to_string(
                index=False
            )
        )

    new_teams = team_map[
        team_map[
            "identity_status"
        ]
        ==
        "NEW_2026_27"
    ]

    print(
        "\nGenuinely new 2026/27 identities:"
    )

    if new_teams.empty:

        print(
            "  None"
        )

    else:

        print(
            new_teams[
                [
                    "canonical_team",
                    "football_data_team_id",
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
    fixtures: pd.DataFrame,
    team_map: pd.DataFrame,
) -> None:
    """
    Save resolved fixtures, team map and audit metadata.
    """

    OUTPUT_FIXTURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixtures.to_csv(
        OUTPUT_FIXTURE_FILE,
        index=False,
        encoding="utf-8",
    )

    team_map.to_csv(
        CURRENT_TEAM_MAP_FILE,
        index=False,
        encoding="utf-8",
    )

    team_map.to_csv(
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

        "season": (
            2026
        ),

        "clubs": int(
            len(
                team_map
            )
        ),

        "historical_clubs": int(
            (
                team_map[
                    "identity_status"
                ]
                ==
                "HISTORICAL"
            ).sum()
        ),

        "new_2026_27_clubs": int(
            (
                team_map[
                    "identity_status"
                ]
                ==
                "NEW_2026_27"
            ).sum()
        ),

        "explicit_aliases": (
            CURRENT_TEAM_ALIASES
        ),

        "rules": {
            "fuzzy_matching": False,

            "automatic_nearest_team_matching": False,

            "new_current_clubs_preserved_as_new": True,

            "historical_identity_requires_exact_resolved_match": True,
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


def resolve_current_ucl_clubs() -> None:
    """
    Resolve current 2026/27 club identities without
    contaminating genuinely new clubs with false history.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - CURRENT CLUB IDENTITY RESOLUTION"
    )

    print(
        "=" * 75
    )

    fixtures = load_fixtures()

    historical_teams = (
        load_historical_teams()
    )

    print(
        f"\nFixture rows: "
        f"{len(fixtures):,}"
    )

    print(
        f"Historical identities: "
        f"{len(historical_teams):,}"
    )

    team_map = build_current_team_map(
        fixtures,
        historical_teams,
    )

    resolved_fixtures = apply_team_map(
        fixtures,
        team_map,
    )

    run_quality_checks(
        fixtures=resolved_fixtures,
        team_map=team_map,
        historical_teams=historical_teams,
    )

    save_outputs(
        fixtures=resolved_fixtures,
        team_map=team_map,
    )

    print_summary(
        team_map
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT UCL CLUB IDENTITIES RESOLVED"
    )

    print(
        "=" * 75
    )

    print(
        "\nResolved fixture file:"
    )

    print(
        OUTPUT_FIXTURE_FILE
    )

    print(
        "\nCurrent team map:"
    )

    print(
        CURRENT_TEAM_MAP_FILE
    )

    print(
        "\nIdentity audit:"
    )

    print(
        AUDIT_FILE
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    resolve_current_ucl_clubs()