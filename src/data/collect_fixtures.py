from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import (
    FOOTBALL_DATA_API_KEY,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

from src.utils import team_names


# ============================================================
# SETTINGS
# ============================================================

COMPETITION_CODE = "CL"
CURRENT_SEASON = 2026

BASE_URL = "https://api.football-data.org/v4"


# ============================================================
# PATHS
# ============================================================

FIXTURE_DIR = (
    INTERIM_DATA_DIR
    / "matches"
)

OUTPUT_FILE = (
    FIXTURE_DIR
    / "ucl_2026_2027_fixtures.csv"
)

METADATA_FILE = (
    FIXTURE_DIR
    / "ucl_2026_2027_fixtures_metadata.json"
)

HISTORICAL_MASTER_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_team_identity_audit.csv"
)


# ============================================================
# STATUS GROUPS
# ============================================================

FINISHED_STATUSES = {
    "FINISHED",
}

UPCOMING_STATUSES = {
    "TIMED",
    "SCHEDULED",
}

LIVE_STATUSES = {
    "LIVE",
    "IN_PLAY",
    "PAUSED",
}

INACTIVE_STATUSES = {
    "POSTPONED",
    "SUSPENDED",
    "CANCELLED",
}


# ============================================================
# TEAM NORMALIZATION ADAPTER
# ============================================================


def normalize_team_name(
    raw_name: str,
) -> str:
    """
    Use the existing project team-name normalization system.

    This adapter intentionally does NOT use fuzzy matching.

    It supports several common helper names so this live
    collector can work with the existing team_names.py
    implementation without changing that validated module.
    """

    candidate_functions = [
        "normalize_team_name",
        "canonicalize_team_name",
        "standardize_team_name",
        "get_canonical_team_name",
        "resolve_team_name",
    ]

    for function_name in candidate_functions:

        function = getattr(
            team_names,
            function_name,
            None,
        )

        if callable(
            function
        ):

            normalized = function(
                raw_name
            )

            if normalized is None:

                return raw_name

            return str(
                normalized
            )

    # --------------------------------------------------------
    # FALLBACK: EXPLICIT DICTIONARY-BASED ALIASES
    # --------------------------------------------------------

    candidate_mappings = [
        "TEAM_NAME_MAP",
        "TEAM_NAME_MAPPING",
        "TEAM_ALIASES",
        "ALIASES",
        "CANONICAL_TEAM_NAMES",
    ]

    for mapping_name in candidate_mappings:

        mapping = getattr(
            team_names,
            mapping_name,
            None,
        )

        if isinstance(
            mapping,
            dict,
        ):

            return str(
                mapping.get(
                    raw_name,
                    raw_name,
                )
            )

    raise RuntimeError(
        "\nNo supported team-name normalization helper "
        "was found in src.utils.team_names.\n\n"
        "Available public names in that module:\n"
        f"{sorted(name for name in dir(team_names) if not name.startswith('_'))}"
    )


# ============================================================
# API
# ============================================================


def require_api_key() -> None:
    """
    Ensure football-data.org credentials exist.
    """

    if not FOOTBALL_DATA_API_KEY:

        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY is missing.\n"
            "Check your .env file."
        )


def fetch_current_ucl_matches() -> dict:
    """
    Fetch all 2026/27 Champions League matches.
    """

    require_api_key()

    url = (
        f"{BASE_URL}/competitions/"
        f"{COMPETITION_CODE}/matches"
    )

    headers = {
        "X-Auth-Token": (
            FOOTBALL_DATA_API_KEY
        ),
    }

    params = {
        "season": (
            CURRENT_SEASON
        ),
    }

    print(
        "\nRequesting football-data.org:"
    )

    print(
        f"Competition: {COMPETITION_CODE}"
    )

    print(
        f"Season:      {CURRENT_SEASON}"
    )

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    if response.status_code != 200:

        try:

            error_payload = (
                response.json()
            )

        except Exception:

            error_payload = (
                response.text
            )

        raise RuntimeError(
            "football-data.org request failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {error_payload}"
        )

    payload = response.json()

    if "matches" not in payload:

        raise RuntimeError(
            "football-data.org response "
            "does not contain 'matches'."
        )

    return payload


# ============================================================
# SCORE EXTRACTION
# ============================================================


def extract_score(
    match: dict,
) -> tuple[
    float | None,
    float | None,
]:
    """
    Extract final full-time score when available.
    """

    score = (
        match.get(
            "score"
        )
        or
        {}
    )

    full_time = (
        score.get(
            "fullTime"
        )
        or
        {}
    )

    home_goals = (
        full_time.get(
            "home"
        )
    )

    away_goals = (
        full_time.get(
            "away"
        )
    )

    return (
        home_goals,
        away_goals,
    )


# ============================================================
# RESULT
# ============================================================


def determine_result(
    *,
    status: str,
    home_goals,
    away_goals,
) -> str | None:
    """
    H/D/A label only for finished matches.
    """

    if status not in FINISHED_STATUSES:

        return None

    if (
        home_goals is None
        or
        away_goals is None
    ):

        return None

    if home_goals > away_goals:

        return "H"

    if home_goals < away_goals:

        return "A"

    return "D"


# ============================================================
# MATCH PARSING
# ============================================================


def parse_matches(
    payload: dict,
) -> pd.DataFrame:
    """
    Normalize API response into the project's fixture schema.
    """

    rows = []

    for match in payload[
        "matches"
    ]:

        home = (
            match.get(
                "homeTeam"
            )
            or
            {}
        )

        away = (
            match.get(
                "awayTeam"
            )
            or
            {}
        )

        raw_home_name = (
            home.get(
                "name"
            )
        )

        raw_away_name = (
            away.get(
                "name"
            )
        )

        if not raw_home_name:

            raise RuntimeError(
                f"Match {match.get('id')} "
                "has no home-team name."
            )

        if not raw_away_name:

            raise RuntimeError(
                f"Match {match.get('id')} "
                "has no away-team name."
            )

        home_team = normalize_team_name(
            raw_home_name
        )

        away_team = normalize_team_name(
            raw_away_name
        )

        utc_date = pd.to_datetime(
            match.get(
                "utcDate"
            ),
            utc=True,
            errors="raise",
        )

        status = str(
            match.get(
                "status"
            )
            or
            ""
        ).upper()

        (
            home_goals,
            away_goals,
        ) = extract_score(
            match
        )

        result = determine_result(
            status=status,
            home_goals=home_goals,
            away_goals=away_goals,
        )

        season = (
            match.get(
                "season"
            )
            or
            {}
        )

        competition = (
            match.get(
                "competition"
            )
            or
            {}
        )

        rows.append(
            {
                "source_match_id": int(
                    match[
                        "id"
                    ]
                ),

                "date_utc": (
                    utc_date
                ),

                "date": (
                    utc_date
                    .tz_convert(
                        None
                    )
                    .normalize()
                ),

                "season": (
                    CURRENT_SEASON
                ),

                "competition": (
                    competition.get(
                        "name"
                    )
                ),

                "competition_code": (
                    competition.get(
                        "code"
                    )
                ),

                "stage": (
                    match.get(
                        "stage"
                    )
                ),

                "matchday": (
                    match.get(
                        "matchday"
                    )
                ),

                "status": (
                    status
                ),

                "raw_home_team": (
                    raw_home_name
                ),

                "home_team": (
                    home_team
                ),

                "home_team_id": (
                    home.get(
                        "id"
                    )
                ),

                "raw_away_team": (
                    raw_away_name
                ),

                "away_team": (
                    away_team
                ),

                "away_team_id": (
                    away.get(
                        "id"
                    )
                ),

                "home_goals": (
                    home_goals
                ),

                "away_goals": (
                    away_goals
                ),

                "result": (
                    result
                ),

                "winner": (
                    (
                        match.get(
                            "score"
                        )
                        or
                        {}
                    ).get(
                        "winner"
                    )
                ),

                "venue": (
                    match.get(
                        "venue"
                    )
                ),

                "last_updated": (
                    match.get(
                        "lastUpdated"
                    )
                ),

                "season_start_date": (
                    season.get(
                        "startDate"
                    )
                ),

                "season_end_date": (
                    season.get(
                        "endDate"
                    )
                ),

                "source": (
                    "football_data_org"
                ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:

        raise RuntimeError(
            "No Champions League matches "
            "were returned."
        )

    dataframe = dataframe.sort_values(
        by=[
            "date_utc",
            "source_match_id",
        ]
    ).reset_index(
        drop=True
    )

    return dataframe


# ============================================================
# HISTORICAL TEAM UNIVERSE
# ============================================================


def load_historical_teams() -> set[str]:
    """
    Load the historical canonical UCL identity universe.
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
        - set(
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
        ]
    )

    teams.update(
        dataframe[
            "away_team"
        ]
    )

    return teams


# ============================================================
# IDENTITY AUDIT
# ============================================================


def build_identity_audit(
    *,
    fixtures: pd.DataFrame,
    historical_teams: set[str],
) -> pd.DataFrame:
    """
    Audit current clubs against historical canonical clubs.

    No fuzzy matching is performed.
    """

    home = fixtures[
        [
            "home_team",
            "raw_home_team",
            "home_team_id",
        ]
    ].rename(
        columns={
            "home_team": (
                "canonical_team"
            ),

            "raw_home_team": (
                "raw_team"
            ),

            "home_team_id": (
                "football_data_team_id"
            ),
        }
    )

    away = fixtures[
        [
            "away_team",
            "raw_away_team",
            "away_team_id",
        ]
    ].rename(
        columns={
            "away_team": (
                "canonical_team"
            ),

            "raw_away_team": (
                "raw_team"
            ),

            "away_team_id": (
                "football_data_team_id"
            ),
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
            "canonical_team",
            "football_data_team_id",
        ]
    )

    rows = []

    for row in teams.itertuples(
        index=False
    ):

        known = (
            row.canonical_team
            in
            historical_teams
        )

        rows.append(
            {
                "raw_team": (
                    row.raw_team
                ),

                "canonical_team": (
                    row.canonical_team
                ),

                "football_data_team_id": (
                    row.football_data_team_id
                ),

                "in_historical_ucl_master": int(
                    known
                ),

                "identity_status": (
                    "KNOWN"
                    if known
                    else
                    "NEW_OR_UNRESOLVED"
                ),
            }
        )

    audit = pd.DataFrame(
        rows
    )

    audit = audit.sort_values(
        by=[
            "identity_status",
            "canonical_team",
        ]
    ).reset_index(
        drop=True
    )

    return audit


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    fixtures: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """
    Validate current fixture collection.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 UCL FIXTURE DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    duplicate_ids = int(
        fixtures[
            "source_match_id"
        ].duplicated().sum()
    )

    print(
        f"\nDuplicate source match IDs: "
        f"{duplicate_ids}"
    )

    if duplicate_ids:

        raise RuntimeError(
            "Duplicate fixture IDs detected."
        )

    missing_teams = int(
        fixtures[
            [
                "home_team",
                "away_team",
            ]
        ].isna().sum().sum()
    )

    print(
        f"Missing canonical team names: "
        f"{missing_teams}"
    )

    if missing_teams:

        raise RuntimeError(
            "Missing canonical team names."
        )

    same_team = int(
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
        f"Home team equals away team: "
        f"{same_team}"
    )

    if same_team:

        raise RuntimeError(
            "Invalid same-team fixture detected."
        )

    finished = fixtures[
        fixtures[
            "status"
        ]
        .isin(
            FINISHED_STATUSES
        )
    ]

    finished_missing_scores = int(
        finished[
            [
                "home_goals",
                "away_goals",
            ]
        ].isna().sum().sum()
    )

    print(
        f"Finished matches missing score values: "
        f"{finished_missing_scores}"
    )

    if finished_missing_scores:

        raise RuntimeError(
            "Finished match missing final score."
        )

    invalid_results = fixtures[
        fixtures[
            "result"
        ].notna()
        &
        ~fixtures[
            "result"
        ].isin(
            [
                "H",
                "D",
                "A",
            ]
        )
    ]

    print(
        f"Invalid result labels: "
        f"{len(invalid_results)}"
    )

    if not invalid_results.empty:

        raise RuntimeError(
            "Invalid result labels found."
        )

    total_clubs = len(
        audit
    )

    known_clubs = int(
        (
            audit[
                "identity_status"
            ]
            ==
            "KNOWN"
        ).sum()
    )

    unresolved_clubs = (
        total_clubs
        -
        known_clubs
    )

    print(
        "\nCurrent club identity coverage:"
    )

    print(
        f"2026/27 clubs: "
        f"{total_clubs}"
    )

    print(
        f"Already known: "
        f"{known_clubs}"
    )

    print(
        f"New/unresolved: "
        f"{unresolved_clubs}"
    )

    if total_clubs:

        print(
            f"Historical identity coverage: "
            f"{known_clubs / total_clubs * 100:.2f}%"
        )


# ============================================================
# SUMMARY
# ============================================================


def print_summary(
    *,
    fixtures: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """
    Display current-season fixture status.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 UCL FIXTURE SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nTotal matches returned: "
        f"{len(fixtures):,}"
    )

    print(
        f"Date range: "
        f"{fixtures['date'].min().date()} "
        f"to "
        f"{fixtures['date'].max().date()}"
    )

    print(
        "\nStatus distribution:"
    )

    print(
        fixtures[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nStage distribution:"
    )

    print(
        fixtures[
            "stage"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nCurrent 2026/27 clubs:"
    )

    current_clubs = sorted(
        set(
            fixtures[
                "home_team"
            ]
        )
        |
        set(
            fixtures[
                "away_team"
            ]
        )
    )

    for team in current_clubs:

        print(
            f"  {team}"
        )

    unresolved = audit[
        audit[
            "identity_status"
        ]
        ==
        "NEW_OR_UNRESOLVED"
    ]

    print(
        "\nNew/unresolved clubs requiring review:"
    )

    if unresolved.empty:

        print(
            "  None ✓"
        )

    else:

        print(
            unresolved[
                [
                    "raw_team",
                    "canonical_team",
                    "football_data_team_id",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nFirst 10 fixtures:"
    )

    print(
        fixtures[
            [
                "date_utc",
                "stage",
                "matchday",
                "status",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nNext 10 unfinished fixtures:"
    )

    upcoming = fixtures[
        ~fixtures[
            "status"
        ].isin(
            FINISHED_STATUSES
        )
    ]

    if upcoming.empty:

        print(
            "  None"
        )

    else:

        print(
            upcoming[
                [
                    "date_utc",
                    "stage",
                    "matchday",
                    "status",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(
                10
            )
            .to_string(
                index=False
            )
        )


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    fixtures: pd.DataFrame,
    audit: pd.DataFrame,
    payload: dict,
) -> None:
    """
    Save fixture snapshot, identity audit and metadata.
    """

    FIXTURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixture_output = (
        fixtures.copy()
    )

    fixture_output[
        "date_utc"
    ] = (
        fixture_output[
            "date_utc"
        ]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    fixture_output[
        "date"
    ] = (
        pd.to_datetime(
            fixture_output[
                "date"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    fixture_output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    audit.to_csv(
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

        "source": (
            "football-data.org"
        ),

        "competition_code": (
            COMPETITION_CODE
        ),

        "season_start_year": (
            CURRENT_SEASON
        ),

        "matches": int(
            len(
                fixtures
            )
        ),

        "clubs": int(
            len(
                audit
            )
        ),

        "known_historical_clubs": int(
            (
                audit[
                    "identity_status"
                ]
                ==
                "KNOWN"
            ).sum()
        ),

        "new_or_unresolved_clubs": int(
            (
                audit[
                    "identity_status"
                ]
                ==
                "NEW_OR_UNRESOLVED"
            ).sum()
        ),

        "api_result_set": (
            payload.get(
                "resultSet"
            )
        ),

        "important_rules": {
            "no_fuzzy_team_matching": True,

            "new_clubs_require_explicit_identity_resolution": True,

            "unfinished_matches_keep_goals_missing": True,

            "unfinished_matches_keep_result_missing": True,

            "production_prediction_features_must_use_only_information_before_fixture_time": True,
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


def collect_current_fixtures() -> None:
    """
    Collect and audit the live 2026/27 Champions League
    fixture snapshot.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - 2026/27 FIXTURE COLLECTION"
    )

    print(
        "=" * 75
    )

    payload = (
        fetch_current_ucl_matches()
    )

    fixtures = parse_matches(
        payload
    )

    historical_teams = (
        load_historical_teams()
    )

    print(
        f"\nHistorical UCL team identities: "
        f"{len(historical_teams)}"
    )

    audit = build_identity_audit(
        fixtures=fixtures,
        historical_teams=historical_teams,
    )

    run_quality_checks(
        fixtures=fixtures,
        audit=audit,
    )

    save_outputs(
        fixtures=fixtures,
        audit=audit,
        payload=payload,
    )

    print_summary(
        fixtures=fixtures,
        audit=audit,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 UCL FIXTURES COLLECTED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nFixture snapshot:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIdentity audit:"
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
    collect_current_fixtures()