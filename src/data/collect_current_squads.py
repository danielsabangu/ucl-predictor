from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import (
    FOOTBALL_DATA_API_KEY,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.football-data.org/v4"

EXPECTED_UCL_CLUBS = 36

# Free football-data.org accounts allow 10 requests/minute.
# 6.5 seconds between calls keeps us safely below that limit.
REQUEST_DELAY_SECONDS = 6.5

MAX_RETRIES = 3


# ============================================================
# PATHS
# ============================================================

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

OUTPUT_DIR = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
)

PLAYER_OUTPUT_FILE = (
    OUTPUT_DIR
    / "football_data_ucl_squads.csv"
)

TEAM_OUTPUT_FILE = (
    OUTPUT_DIR
    / "football_data_ucl_teams.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "football_data_ucl_squad_metadata.json"
)

AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_squad_collection_audit.csv"
)


# ============================================================
# API CHECK
# ============================================================


def require_api_key() -> None:
    """
    Ensure football-data.org API key exists.
    """

    if not FOOTBALL_DATA_API_KEY:

        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY is missing.\n"
            "Check your .env file."
        )


# ============================================================
# LOAD CURRENT UCL CLUBS
# ============================================================


def load_current_ucl_clubs() -> pd.DataFrame:
    """
    Load the resolved 36-club 2026/27 UCL identity map.
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

    dataframe[
        "football_data_team_id"
    ] = pd.to_numeric(
        dataframe[
            "football_data_team_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    if len(
        dataframe
    ) != EXPECTED_UCL_CLUBS:

        raise RuntimeError(
            f"Expected {EXPECTED_UCL_CLUBS} "
            f"current UCL clubs, found "
            f"{len(dataframe)}."
        )

    duplicate_ids = int(
        dataframe[
            "football_data_team_id"
        ].duplicated().sum()
    )

    if duplicate_ids:

        raise RuntimeError(
            "Duplicate football-data team IDs "
            "found in current UCL map."
        )

    duplicate_names = int(
        dataframe[
            "canonical_team"
        ].duplicated().sum()
    )

    if duplicate_names:

        raise RuntimeError(
            "Duplicate canonical teams found "
            "in current UCL map."
        )

    return dataframe.sort_values(
        by="canonical_team"
    ).reset_index(
        drop=True
    )


# ============================================================
# REQUEST TEAM RESOURCE
# ============================================================


def fetch_team(
    *,
    team_id: int,
    team_name: str,
) -> dict:
    """
    Fetch one current football-data.org Team resource.

    Retries temporary failures and respects API throttling.
    """

    url = (
        f"{BASE_URL}/teams/"
        f"{team_id}"
    )

    headers = {
        "X-Auth-Token": (
            FOOTBALL_DATA_API_KEY
        ),
    }

    for attempt in range(
        1,
        MAX_RETRIES
        +
        1,
    ):

        response = requests.get(
            url,
            headers=headers,
            timeout=30,
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            return response.json()

        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        if response.status_code == 429:

            retry_after = response.headers.get(
                "Retry-After"
            )

            try:

                retry_seconds = float(
                    retry_after
                )

            except (
                TypeError,
                ValueError,
            ):

                retry_seconds = (
                    REQUEST_DELAY_SECONDS
                    *
                    2
                )

            print(
                f"    Rate limited. "
                f"Retrying after API reset..."
            )

            time.sleep(
                max(
                    retry_seconds,
                    REQUEST_DELAY_SECONDS,
                )
            )

            continue

        # ----------------------------------------------------
        # SERVER ERROR
        # ----------------------------------------------------

        if (
            500
            <=
            response.status_code
            <
            600
        ):

            if attempt < MAX_RETRIES:

                print(
                    f"    Server error "
                    f"{response.status_code}; "
                    f"retry {attempt}/{MAX_RETRIES}."
                )

                time.sleep(
                    REQUEST_DELAY_SECONDS
                )

                continue

        # ----------------------------------------------------
        # NON-RETRYABLE ERROR
        # ----------------------------------------------------

        try:

            payload = response.json()

        except Exception:

            payload = response.text

        raise RuntimeError(
            f"\nfootball-data.org request failed "
            f"for {team_name} ({team_id}).\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {payload}"
        )

    raise RuntimeError(
        f"Failed to retrieve {team_name} "
        f"after {MAX_RETRIES} attempts."
    )


# ============================================================
# CONTRACT PARSING
# ============================================================


def parse_contract(
    payload: dict | None,
) -> tuple[
    str | None,
    str | None,
]:
    """
    Extract contract start/end when available.
    """

    if not isinstance(
        payload,
        dict,
    ):

        return (
            None,
            None,
        )

    return (
        payload.get(
            "start"
        ),
        payload.get(
            "until"
        ),
    )


# ============================================================
# TEAM PARSING
# ============================================================


def parse_team(
    *,
    canonical_team: str,
    identity_status: str,
    payload: dict,
) -> dict:
    """
    Build one current-team summary row.
    """

    area = (
        payload.get(
            "area"
        )
        or
        {}
    )

    coach = (
        payload.get(
            "coach"
        )
        or
        {}
    )

    coach_contract_start, coach_contract_until = (
        parse_contract(
            coach.get(
                "contract"
            )
        )
    )

    squad = (
        payload.get(
            "squad"
        )
        or
        []
    )

    player_market_values = []

    for player in squad:

        market_value = player.get(
            "marketValue"
        )

        if market_value is not None:

            try:

                player_market_values.append(
                    float(
                        market_value
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

    return {
        "canonical_team": (
            canonical_team
        ),

        "identity_status": (
            identity_status
        ),

        "football_data_team_id": (
            payload.get(
                "id"
            )
        ),

        "football_data_name": (
            payload.get(
                "name"
            )
        ),

        "short_name": (
            payload.get(
                "shortName"
            )
        ),

        "tla": (
            payload.get(
                "tla"
            )
        ),

        "country": (
            area.get(
                "name"
            )
        ),

        "country_code": (
            area.get(
                "code"
            )
        ),

        "founded": (
            payload.get(
                "founded"
            )
        ),

        "venue": (
            payload.get(
                "venue"
            )
        ),

        "club_market_value": (
            payload.get(
                "marketValue"
            )
        ),

        "coach_id": (
            coach.get(
                "id"
            )
        ),

        "coach_name": (
            coach.get(
                "name"
            )
        ),

        "coach_date_of_birth": (
            coach.get(
                "dateOfBirth"
            )
        ),

        "coach_nationality": (
            coach.get(
                "nationality"
            )
        ),

        "coach_contract_start": (
            coach_contract_start
        ),

        "coach_contract_until": (
            coach_contract_until
        ),

        "squad_size": (
            len(
                squad
            )
        ),

        "players_with_market_value": (
            len(
                player_market_values
            )
        ),

        "player_market_value_coverage": (
            len(
                player_market_values
            )
            /
            len(
                squad
            )
            if squad
            else 0.0
        ),

        "summed_player_market_value": (
            sum(
                player_market_values
            )
            if player_market_values
            else None
        ),
    }


# ============================================================
# PLAYER PARSING
# ============================================================


def parse_players(
    *,
    canonical_team: str,
    identity_status: str,
    team_payload: dict,
) -> list[dict]:
    """
    Parse current squad players.
    """

    squad = (
        team_payload.get(
            "squad"
        )
        or
        []
    )

    rows = []

    for player in squad:

        contract_start, contract_until = (
            parse_contract(
                player.get(
                    "contract"
                )
            )
        )

        rows.append(
            {
                "canonical_team": (
                    canonical_team
                ),

                "identity_status": (
                    identity_status
                ),

                "football_data_team_id": (
                    team_payload.get(
                        "id"
                    )
                ),

                "football_data_team_name": (
                    team_payload.get(
                        "name"
                    )
                ),

                "football_data_player_id": (
                    player.get(
                        "id"
                    )
                ),

                "player_name": (
                    player.get(
                        "name"
                    )
                ),

                "first_name": (
                    player.get(
                        "firstName"
                    )
                ),

                "last_name": (
                    player.get(
                        "lastName"
                    )
                ),

                "position": (
                    player.get(
                        "position"
                    )
                ),

                "date_of_birth": (
                    player.get(
                        "dateOfBirth"
                    )
                ),

                "nationality": (
                    player.get(
                        "nationality"
                    )
                ),

                "shirt_number": (
                    player.get(
                        "shirtNumber"
                    )
                ),

                "market_value": (
                    player.get(
                        "marketValue"
                    )
                ),

                "contract_start": (
                    contract_start
                ),

                "contract_until": (
                    contract_until
                ),
            }
        )

    return rows


# ============================================================
# COLLECT ALL CURRENT SQUADS
# ============================================================


def collect_all_squads(
    current_clubs: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Collect current squad resources for all 36 UCL clubs.
    """

    team_rows = []

    player_rows = []

    total = len(
        current_clubs
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "COLLECTING CURRENT TEAM RESOURCES"
    )

    print(
        "=" * 75
    )

    for index, club in enumerate(
        current_clubs.itertuples(
            index=False
        ),
        start=1,
    ):

        canonical_team = str(
            club.canonical_team
        )

        team_id = int(
            club.football_data_team_id
        )

        identity_status = str(
            club.identity_status
        )

        print(
            f"\n[{index:02d}/{total}] "
            f"{canonical_team} "
            f"(football-data ID {team_id})"
        )

        payload = fetch_team(
            team_id=team_id,
            team_name=canonical_team,
        )

        returned_team_id = payload.get(
            "id"
        )

        if int(
            returned_team_id
        ) != team_id:

            raise RuntimeError(
                f"{canonical_team}: requested team ID "
                f"{team_id}, API returned "
                f"{returned_team_id}."
            )

        team_row = parse_team(
            canonical_team=canonical_team,
            identity_status=identity_status,
            payload=payload,
        )

        players = parse_players(
            canonical_team=canonical_team,
            identity_status=identity_status,
            team_payload=payload,
        )

        team_rows.append(
            team_row
        )

        player_rows.extend(
            players
        )

        print(
            f"    API name: "
            f"{payload.get('name')}"
        )

        print(
            f"    Squad players: "
            f"{len(players)}"
        )

        print(
            f"    Team market value: "
            f"{payload.get('marketValue')}"
        )

        player_values = sum(
            1
            for player in players
            if player[
                "market_value"
            ]
            is not None
        )

        print(
            f"    Players with market value: "
            f"{player_values}/{len(players)}"
        )

        # Stay below free-plan 10 requests/minute.
        if index < total:

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    teams = pd.DataFrame(
        team_rows
    )

    players = pd.DataFrame(
        player_rows
    )

    return (
        teams,
        players,
    )


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    current_clubs: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Audit current squad coverage and integrity.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT SQUAD DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # TEAM COVERAGE
    # --------------------------------------------------------

    collected_teams = set(
        teams[
            "canonical_team"
        ]
    )

    expected_teams = set(
        current_clubs[
            "canonical_team"
        ]
    )

    missing_teams = (
        expected_teams
        -
        collected_teams
    )

    extra_teams = (
        collected_teams
        -
        expected_teams
    )

    print(
        f"\nExpected UCL clubs: "
        f"{len(expected_teams)}"
    )

    print(
        f"Collected team resources: "
        f"{len(collected_teams)}"
    )

    print(
        f"Missing team resources: "
        f"{len(missing_teams)}"
    )

    print(
        f"Unexpected team resources: "
        f"{len(extra_teams)}"
    )

    if missing_teams:

        raise RuntimeError(
            f"Missing team resources: "
            f"{sorted(missing_teams)}"
        )

    if extra_teams:

        raise RuntimeError(
            f"Unexpected team resources: "
            f"{sorted(extra_teams)}"
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_team_ids = int(
        teams[
            "football_data_team_id"
        ].duplicated().sum()
    )

    print(
        f"Duplicate team IDs: "
        f"{duplicate_team_ids}"
    )

    if duplicate_team_ids:

        raise RuntimeError(
            "Duplicate current team IDs."
        )

    if players.empty:

        raise RuntimeError(
            "No current squad players collected."
        )

    # --------------------------------------------------------
    # PLAYER CORE FIELDS
    # --------------------------------------------------------

    missing_player_names = int(
        players[
            "player_name"
        ].isna().sum()
    )

    missing_player_ids = int(
        players[
            "football_data_player_id"
        ].isna().sum()
    )

    print(
        f"\nCurrent squad player rows: "
        f"{len(players):,}"
    )

    print(
        f"Missing player names: "
        f"{missing_player_names}"
    )

    print(
        f"Missing football-data player IDs: "
        f"{missing_player_ids}"
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    teams_with_squad = int(
        (
            teams[
                "squad_size"
            ]
            >
            0
        ).sum()
    )

    teams_without_squad = (
        len(
            teams
        )
        -
        teams_with_squad
    )

    player_value_rows = int(
        players[
            "market_value"
        ].notna().sum()
    )

    overall_market_value_coverage = (
        player_value_rows
        /
        len(
            players
        )
    )

    print(
        f"\nTeams with squad data: "
        f"{teams_with_squad}/{len(teams)}"
    )

    print(
        f"Teams without squad data: "
        f"{teams_without_squad}"
    )

    print(
        f"Players with market value: "
        f"{player_value_rows:,}/{len(players):,}"
    )

    print(
        f"Overall player market-value coverage: "
        f"{overall_market_value_coverage * 100:.2f}%"
    )

    # --------------------------------------------------------
    # AUDIT TABLE
    # --------------------------------------------------------

    audit_rows = []

    for team in teams.itertuples(
        index=False
    ):

        team_players = players[
            players[
                "canonical_team"
            ]
            ==
            team.canonical_team
        ]

        squad_size = len(
            team_players
        )

        values_available = int(
            team_players[
                "market_value"
            ].notna().sum()
        )

        audit_rows.append(
            {
                "canonical_team": (
                    team.canonical_team
                ),

                "identity_status": (
                    team.identity_status
                ),

                "football_data_team_id": (
                    team.football_data_team_id
                ),

                "football_data_name": (
                    team.football_data_name
                ),

                "squad_size": (
                    squad_size
                ),

                "players_with_market_value": (
                    values_available
                ),

                "market_value_coverage": (
                    values_available
                    /
                    squad_size
                    if squad_size
                    else 0.0
                ),

                "club_market_value": (
                    team.club_market_value
                ),

                "summed_player_market_value": (
                    team.summed_player_market_value
                ),

                "coach_name": (
                    team.coach_name
                ),

                "squad_available": int(
                    squad_size
                    >
                    0
                ),
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    audit = audit.sort_values(
        by=[
            "squad_available",
            "market_value_coverage",
            "canonical_team",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return audit


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    teams: pd.DataFrame,
    players: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """
    Save current squad snapshot and audit metadata.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    teams.to_csv(
        TEAM_OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    players.to_csv(
        PLAYER_OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    teams_with_squad = int(
        audit[
            "squad_available"
        ].sum()
    )

    total_players = int(
        len(
            players
        )
    )

    players_with_values = int(
        players[
            "market_value"
        ].notna().sum()
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source": (
            "football-data.org"
        ),

        "source_resource": (
            "/v4/teams/{id}"
        ),

        "season_context": (
            "2026/27 current season"
        ),

        "ucl_clubs_requested": (
            EXPECTED_UCL_CLUBS
        ),

        "ucl_clubs_collected": int(
            len(
                teams
            )
        ),

        "teams_with_nonempty_squad": (
            teams_with_squad
        ),

        "player_rows": (
            total_players
        ),

        "players_with_market_value": (
            players_with_values
        ),

        "player_market_value_coverage": (
            players_with_values
            /
            total_players
            if total_players
            else 0.0
        ),

        "request_delay_seconds": (
            REQUEST_DELAY_SECONDS
        ),

        "important_rules": {
            "current_squad_snapshot_only": True,

            "not_used_to_rewrite_historical_training_features": True,

            "missing_market_values_preserved": True,

            "no_player_identity_fuzzy_merge": True,

            "future_fixture_features_must_use_snapshot_available_before_prediction_time": True,
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
    teams: pd.DataFrame,
    players: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """
    Display live squad snapshot coverage.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 CURRENT SQUAD SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nTeams collected: "
        f"{len(teams)}"
    )

    print(
        f"Player rows: "
        f"{len(players):,}"
    )

    print(
        f"Teams with squad data: "
        f"{int(audit['squad_available'].sum())}"
        f"/{len(audit)}"
    )

    player_values = int(
        players[
            "market_value"
        ].notna().sum()
    )

    print(
        f"Players with market value: "
        f"{player_values:,}/{len(players):,}"
    )

    if len(
        players
    ):

        print(
            f"Overall market-value coverage: "
            f"{player_values / len(players) * 100:.2f}%"
        )

    print(
        "\nLowest squad/market-value coverage:"
    )

    print(
        audit[
            [
                "canonical_team",
                "identity_status",
                "squad_size",
                "players_with_market_value",
                "market_value_coverage",
                "club_market_value",
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

    new_clubs = audit[
        audit[
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
                "players_with_market_value",
                "market_value_coverage",
                "club_market_value",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


# ============================================================
# MAIN
# ============================================================


def collect_current_squads() -> None:
    """
    Collect fresh current-season squad information for every
    2026/27 Champions League club.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - 2026/27 CURRENT SQUAD COLLECTION"
    )

    print(
        "=" * 75
    )

    require_api_key()

    current_clubs = (
        load_current_ucl_clubs()
    )

    print(
        f"\nCurrent UCL clubs: "
        f"{len(current_clubs)}"
    )

    print(
        "Request policy: rate-safe football-data.org "
        "Team resource collection"
    )

    (
        teams,
        players,
    ) = collect_all_squads(
        current_clubs
    )

    audit = run_quality_checks(
        current_clubs=current_clubs,
        teams=teams,
        players=players,
    )

    save_outputs(
        teams=teams,
        players=players,
        audit=audit,
    )

    print_summary(
        teams=teams,
        players=players,
        audit=audit,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CURRENT UCL SQUADS COLLECTED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nTeam snapshot:"
    )

    print(
        TEAM_OUTPUT_FILE
    )

    print(
        "\nPlayer snapshot:"
    )

    print(
        PLAYER_OUTPUT_FILE
    )

    print(
        "\nCoverage audit:"
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


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    collect_current_squads()