from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.config import (
    FOOTBALL_DATA_API_KEY,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)

from src.features import domestic_form


# ============================================================
# SETTINGS
# ============================================================

CURRENT_SEASON = 2026
SEASON_LABEL = "2026/27"

EXPECTED_HISTORICAL_ROWS = 28893
EXPECTED_CURRENT_UCL_CLUBS = 36

LOCAL_TIMEZONE = ZoneInfo(
    "Africa/Johannesburg"
)

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 6.5

# Broad enough to capture the full beginning of 2026/27.
LIVE_DATE_FROM = "2026-07-01"


# ============================================================
# COMPETITIONS
# ============================================================

COMPETITIONS = {
    "PL": {
        "country": "England",
        "league": "Premier League",
        "league_code": "ENG1",
    },

    "PD": {
        "country": "Spain",
        "league": "La Liga",
        "league_code": "ESP1",
    },

    "BL1": {
        "country": "Germany",
        "league": "Bundesliga",
        "league_code": "GER1",
    },

    "SA": {
        "country": "Italy",
        "league": "Serie A",
        "league_code": "ITA1",
    },

    "FL1": {
        "country": "France",
        "league": "Ligue 1",
        "league_code": "FRA1",
    },

    "DED": {
        "country": "Netherlands",
        "league": "Eredivisie",
        "league_code": "NED1",
    },

    "PPL": {
        "country": "Portugal",
        "league": "Primeira Liga",
        "league_code": "POR1",
    },
}


# ============================================================
# PATHS
# ============================================================

BASE_DOMESTIC_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "domestic_matches.csv"
)

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

OUTPUT_DIR = (
    INTERIM_DATA_DIR
    / "matches"
    / "live_2026_27"
)

CURRENT_SEASON_PATCH_FILE = (
    OUTPUT_DIR
    / "football_data_domestic_2026_27.csv"
)

LIVE_DOMESTIC_FILE = (
    OUTPUT_DIR
    / "domestic_matches_live.csv"
)

LIVE_DOMESTIC_LATEST_FILE = (
    OUTPUT_DIR
    / "ucl_team_domestic_form_live_current_36.csv"
)

TEAM_AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_live_domestic_state_audit.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "ucl_live_domestic_metadata.json"
)


# ============================================================
# HELPERS
# ============================================================


def require_file(
    path,
    description: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )


def result_from_goals(
    home_goals: int,
    away_goals: int,
) -> str:

    if home_goals > away_goals:

        return "H"

    if home_goals < away_goals:

        return "A"

    return "D"


def current_local_date() -> str:

    return datetime.now(
        LOCAL_TIMEZONE
    ).date().isoformat()


# ============================================================
# HISTORICAL DOMESTIC CORE
# ============================================================


def load_historical_core() -> pd.DataFrame:
    """
    Keep only the frozen historical domestic rows.

    Any current 2026/27 OpenFootball rows in domestic_matches.csv
    are intentionally ignored because that source has now shown
    partial-download behaviour.
    """

    require_file(
        BASE_DOMESTIC_FILE,
        "Domestic match dataset",
    )

    dataframe = pd.read_csv(
        BASE_DOMESTIC_FILE,
        low_memory=False,
    )

    required = {
        "domestic_match_id",
        "date",
        "season",
        "season_label",
        "country",
        "league",
        "league_code",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "scoreline",
        "competition_type",
        "source",
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
            "Domestic dataset missing columns:\n"
            f"{sorted(missing)}"
        )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    dataframe[
        "season"
    ] = pd.to_numeric(
        dataframe[
            "season"
        ],
        errors="raise",
    ).astype(
        int
    )

    historical = dataframe[
        dataframe[
            "season"
        ]
        <
        CURRENT_SEASON
    ].copy()

    if (
        len(
            historical
        )
        !=
        EXPECTED_HISTORICAL_ROWS
    ):

        raise RuntimeError(
            f"Historical domestic core changed. "
            f"Expected "
            f"{EXPECTED_HISTORICAL_ROWS:,}, "
            f"found "
            f"{len(historical):,}."
        )

    historical = historical.sort_values(
        by=[
            "date",
            "domestic_match_id",
        ]
    ).reset_index(
        drop=True
    )

    return historical


# ============================================================
# CURRENT UCL TEAM IDENTITY MAP
# ============================================================


def load_current_team_identity() -> tuple[
    set[str],
    dict[int, str],
]:

    require_file(
        CURRENT_TEAM_MAP_FILE,
        "Current UCL team map",
    )

    dataframe = pd.read_csv(
        CURRENT_TEAM_MAP_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "football_data_team_id",
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
            "Current UCL team map missing columns:\n"
            f"{sorted(missing)}"
        )

    if (
        len(
            dataframe
        )
        !=
        EXPECTED_CURRENT_UCL_CLUBS
    ):

        raise RuntimeError(
            f"Expected 36 current UCL clubs, "
            f"found {len(dataframe)}."
        )

    canonical_teams = set(
        dataframe[
            "canonical_team"
        ].astype(
            str
        )
    )

    id_to_canonical = {}

    for row in dataframe.itertuples(
        index=False
    ):

        team_id = pd.to_numeric(
            row.football_data_team_id,
            errors="coerce",
        )

        if pd.isna(
            team_id
        ):

            continue

        team_id = int(
            team_id
        )

        if team_id in id_to_canonical:

            raise RuntimeError(
                "Duplicate football-data team ID "
                f"in current team map: {team_id}"
            )

        id_to_canonical[
            team_id
        ] = str(
            row.canonical_team
        )

    return (
        canonical_teams,
        id_to_canonical,
    )


# ============================================================
# API REQUEST
# ============================================================


def request_competition_matches(
    *,
    competition_code: str,
    date_from: str,
    date_to: str,
) -> list[dict]:

    if not FOOTBALL_DATA_API_KEY:

        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY is missing from .env."
        )

    url = (
        "https://api.football-data.org/v4/"
        f"competitions/{competition_code}/matches"
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

        "status": (
            "FINISHED"
        ),

        "dateFrom": (
            date_from
        ),

        "dateTo": (
            date_to
        ),
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:

                payload = response.json()

                return payload.get(
                    "matches",
                    []
                )

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

            if response.status_code == 429:

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                try:

                    delay = float(
                        retry_after
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    delay = (
                        REQUEST_DELAY_SECONDS
                        *
                        2
                    )

                print(
                    f"      Rate limited; "
                    f"sleeping {delay:.1f}s..."
                )

                time.sleep(
                    delay
                )

            else:

                time.sleep(
                    2
                    *
                    attempt
                )

        except requests.RequestException as error:

            last_error = str(
                error
            )

            time.sleep(
                2
                *
                attempt
            )

    raise RuntimeError(
        f"Failed to collect "
        f"{competition_code} after "
        f"{MAX_RETRIES} attempts.\n"
        f"{last_error}"
    )


# ============================================================
# TEAM NAME RESOLUTION
# ============================================================


def resolve_team_name(
    *,
    team_payload: dict,
    id_to_canonical: dict[int, str],
) -> tuple[str, str]:

    team_id = team_payload.get(
        "id"
    )

    raw_name = team_payload.get(
        "name"
    )

    if raw_name is None:

        raise RuntimeError(
            "Football-data team missing name."
        )

    if team_id is not None:

        try:

            numeric_id = int(
                team_id
            )

        except (
            TypeError,
            ValueError,
        ):

            numeric_id = None

        if (
            numeric_id is not None
            and
            numeric_id in id_to_canonical
        ):

            return (
                id_to_canonical[
                    numeric_id
                ],
                "CURRENT_UCL_CANONICAL_ID_MAP",
            )

    return (
        str(
            raw_name
        ),
        "FOOTBALL_DATA_RAW_NAME",
    )


# ============================================================
# PARSE CURRENT MATCH
# ============================================================


def parse_match(
    *,
    match: dict,
    competition_code: str,
    metadata: dict,
    id_to_canonical: dict[int, str],
) -> dict | None:

    if str(
        match.get(
            "status",
            ""
        )
    ) != "FINISHED":

        return None

    full_time = (
        match.get(
            "score",
            {}
        )
        .get(
            "fullTime",
            {}
        )
    )

    home_goals = full_time.get(
        "home"
    )

    away_goals = full_time.get(
        "away"
    )

    if (
        home_goals is None
        or
        away_goals is None
    ):

        return None

    home_team_payload = match.get(
        "homeTeam",
        {}
    )

    away_team_payload = match.get(
        "awayTeam",
        {}
    )

    (
        home_team,
        home_name_source,
    ) = resolve_team_name(
        team_payload=home_team_payload,
        id_to_canonical=id_to_canonical,
    )

    (
        away_team,
        away_name_source,
    ) = resolve_team_name(
        team_payload=away_team_payload,
        id_to_canonical=id_to_canonical,
    )

    utc_date = pd.to_datetime(
        match.get(
            "utcDate"
        ),
        errors="raise",
        utc=True,
    )

    home_goals = int(
        home_goals
    )

    away_goals = int(
        away_goals
    )

    return {
        "football_data_match_id": (
            match.get(
                "id"
            )
        ),

        "utc_date": (
            utc_date
        ),

        "date": (
            utc_date.tz_convert(
                None
            ).normalize()
        ),

        "season": (
            CURRENT_SEASON
        ),

        "season_label": (
            SEASON_LABEL
        ),

        "country": (
            metadata[
                "country"
            ]
        ),

        "league": (
            metadata[
                "league"
            ]
        ),

        "league_code": (
            metadata[
                "league_code"
            ]
        ),

        "football_data_competition_code": (
            competition_code
        ),

        "home_team": (
            home_team
        ),

        "away_team": (
            away_team
        ),

        "home_team_name_source": (
            home_name_source
        ),

        "away_team_name_source": (
            away_name_source
        ),

        "home_goals": (
            home_goals
        ),

        "away_goals": (
            away_goals
        ),

        "result": (
            result_from_goals(
                home_goals,
                away_goals,
            )
        ),

        "scoreline": (
            f"{home_goals}-{away_goals}"
        ),

        "competition_type": (
            "domestic_league"
        ),

        "source": (
            "football-data.org_live"
        ),
    }


# ============================================================
# COLLECT COMPLETE CURRENT SEASON
# ============================================================


def collect_current_season(
    *,
    date_from: str,
    date_to: str,
    id_to_canonical: dict[int, str],
) -> pd.DataFrame:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FOOTBALL-DATA.ORG 2026/27 DOMESTIC REBUILD"
    )

    print(
        "=" * 75
    )

    print(
        f"\nDate window: "
        f"{date_from} to {date_to}"
    )

    rows = []

    for index, (
        competition_code,
        metadata,
    ) in enumerate(
        COMPETITIONS.items()
    ):

        print(
            "\n"
            + "-"
            *
            75
        )

        print(
            f"{metadata['country']} - "
            f"{metadata['league']} "
            f"({competition_code})"
        )

        matches = request_competition_matches(
            competition_code=competition_code,
            date_from=date_from,
            date_to=date_to,
        )

        retained = 0

        for match in matches:

            parsed = parse_match(
                match=match,
                competition_code=competition_code,
                metadata=metadata,
                id_to_canonical=id_to_canonical,
            )

            if parsed is None:

                continue

            rows.append(
                parsed
            )

            retained += 1

        print(
            f"Finished matches retained: "
            f"{retained}"
        )

        if (
            index
            <
            len(
                COMPETITIONS
            )
            -
            1
        ):

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    current = pd.DataFrame(
        rows
    )

    if current.empty:

        raise RuntimeError(
            "Football-data current-season collection "
            "returned no finished matches."
        )

    duplicate_ids = int(
        current[
            "football_data_match_id"
        ]
        .dropna()
        .duplicated()
        .sum()
    )

    if duplicate_ids:

        raise RuntimeError(
            f"Duplicate football-data match IDs: "
            f"{duplicate_ids}"
        )

    duplicate_keys = int(
        current.duplicated(
            subset=[
                "date",
                "league_code",
                "home_team",
                "away_team",
            ]
        ).sum()
    )

    if duplicate_keys:

        raise RuntimeError(
            f"Duplicate current-season match keys: "
            f"{duplicate_keys}"
        )

    current = current.sort_values(
        by=[
            "utc_date",
            "league_code",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    return current


# ============================================================
# BUILD EXACT LIVE DOMESTIC SCHEMA
# ============================================================


def build_live_dataset(
    *,
    historical: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    base_columns = list(
        historical.columns
    )

    missing_for_merge = [
        column
        for column in base_columns
        if (
            column
            !=
            "domestic_match_id"
            and
            column
            not in
            current.columns
        )
    ]

    if missing_for_merge:

        raise RuntimeError(
            "Current football-data rows cannot reproduce "
            "base domestic schema. Missing:\n"
            f"{missing_for_merge}"
        )

    current_exact = current[
        [
            column
            for column in base_columns
            if column
            !=
            "domestic_match_id"
        ]
    ].copy()

    current_exact.insert(
        0,
        "domestic_match_id",
        range(
            len(
                historical
            )
            +
            1,
            len(
                historical
            )
            +
            len(
                current_exact
            )
            +
            1,
        ),
    )

    current_exact = current_exact[
        base_columns
    ]

    live = pd.concat(
        [
            historical,
            current_exact,
        ],
        ignore_index=True,
    )

    live[
        "date"
    ] = pd.to_datetime(
        live[
            "date"
        ],
        errors="raise",
    )

    live = live.sort_values(
        by=[
            "date",
            "domestic_match_id",
        ]
    ).reset_index(
        drop=True
    )

    return (
        live,
        current_exact,
    )


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    historical: pd.DataFrame,
    current: pd.DataFrame,
    live: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE DOMESTIC DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        f"\nHistorical rows: "
        f"{len(historical):,}"
    )

    print(
        f"Football-data 2026/27 rows: "
        f"{len(current):,}"
    )

    print(
        f"Live domestic rows: "
        f"{len(live):,}"
    )

    expected = (
        len(
            historical
        )
        +
        len(
            current
        )
    )

    if len(
        live
    ) != expected:

        raise RuntimeError(
            "Unexpected live domestic row count."
        )

    core_columns = [
        "date",
        "season",
        "country",
        "league",
        "league_code",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    ]

    missing_core = int(
        live[
            core_columns
        ].isna().sum().sum()
    )

    print(
        f"Missing core values: "
        f"{missing_core}"
    )

    if missing_core:

        raise RuntimeError(
            "Missing domestic core values."
        )

    invalid_results = (
        set(
            live[
                "result"
            ].astype(
                str
            )
        )
        -
        {
            "H",
            "D",
            "A",
        }
    )

    print(
        f"Invalid result labels: "
        f"{len(invalid_results)}"
    )

    if invalid_results:

        raise RuntimeError(
            f"Invalid result labels: "
            f"{sorted(invalid_results)}"
        )

    same_team = int(
        (
            live[
                "home_team"
            ]
            ==
            live[
                "away_team"
            ]
        ).sum()
    )

    print(
        f"Same-team matches: "
        f"{same_team}"
    )

    if same_team:

        raise RuntimeError(
            "Same-team domestic fixture detected."
        )

    duplicate_ids = int(
        live[
            "domestic_match_id"
        ].duplicated().sum()
    )

    print(
        f"Duplicate domestic IDs: "
        f"{duplicate_ids}"
    )

    if duplicate_ids:

        raise RuntimeError(
            "Duplicate domestic match IDs."
        )

    if not live[
        "date"
    ].is_monotonic_increasing:

        raise RuntimeError(
            "Domestic live dataset is not chronological."
        )

    current_counts = (
        current.groupby(
            [
                "country",
                "league",
            ]
        )
        .size()
    )

    print(
        "\nCurrent 2026/27 matches by league:"
    )

    print(
        current_counts.to_string()
    )

    print(
        f"\nHistorical latest date: "
        f"{historical['date'].max().date()}"
    )

    print(
        f"Live latest date: "
        f"{live['date'].max().date()}"
    )

    print(
        "\nHistorical 28,893-row core preserved ✓"
    )

    print(
        "Chronological ordering valid ✓"
    )


# ============================================================
# BUILD CURRENT DOMESTIC STATE
# ============================================================


def build_current_domestic_state(
    *,
    live: pd.DataFrame,
    current_ucl_teams: set[str],
) -> pd.DataFrame:

    print(
        "\nBuilding histories through existing "
        "src.features.domestic_form logic..."
    )

    (
        histories,
        dates_lookup,
    ) = (
        domestic_form.build_domestic_histories(
            live.copy()
        )
    )

    latest = (
        domestic_form.build_latest_domestic_table(
            histories=histories,
            ucl_teams=current_ucl_teams,
        )
    )

    if (
        "team"
        not in
        latest.columns
    ):

        raise RuntimeError(
            "Domestic latest-state output has no "
            "'team' column."
        )

    available_rows = set(
        latest[
            "team"
        ].astype(
            str
        )
    )

    missing_rows = (
        current_ucl_teams
        -
        available_rows
    )

    if missing_rows:

        raise RuntimeError(
            "Current UCL teams missing from domestic "
            "latest-state table:\n"
            f"{sorted(missing_rows)}"
        )

    latest = latest[
        latest[
            "team"
        ].isin(
            current_ucl_teams
        )
    ].copy()

    latest = latest.sort_values(
        "team"
    ).reset_index(
        drop=True
    )

    if (
        len(
            latest
        )
        !=
        EXPECTED_CURRENT_UCL_CLUBS
    ):

        raise RuntimeError(
            f"Expected 36 domestic-state rows, "
            f"found {len(latest)}."
        )

    return latest


# ============================================================
# TEAM AUDIT
# ============================================================


def build_team_audit(
    latest: pd.DataFrame,
) -> pd.DataFrame:

    audit = latest.copy()

    history_available = (
        pd.to_numeric(
            audit[
                "domestic_history_available"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    audit[
        "domestic_history_available"
    ] = history_available

    audit[
        "live_domestic_state_status"
    ] = history_available.map(
        {
            1: "AVAILABLE",
            0: "NO_SUPPORTED_DOMESTIC_HISTORY",
        }
    )

    return audit


# ============================================================
# SUMMARY
# ============================================================


def print_summary(
    *,
    current: pd.DataFrame,
    live: pd.DataFrame,
    latest: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 LIVE DOMESTIC STATE SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nCurrent-season matches rebuilt: "
        f"{len(current):,}"
    )

    print(
        f"Combined live domestic matches: "
        f"{len(live):,}"
    )

    print(
        f"Live latest date: "
        f"{live['date'].max().date()}"
    )

    print(
        "\n2026/27 matches by league:"
    )

    print(
        current.groupby(
            [
                "country",
                "league",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    history_available = (
        pd.to_numeric(
            latest[
                "domestic_history_available"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    available_count = int(
        history_available.sum()
    )

    print(
        f"\nCurrent UCL clubs with supported "
        f"domestic history: "
        f"{available_count}/36"
    )

    unavailable = latest[
        history_available
        ==
        0
    ]

    print(
        "Current UCL clubs without supported "
        f"domestic history: "
        f"{len(unavailable)}"
    )

    if not unavailable.empty:

        print(
            "\nUnsupported / unavailable domestic history:"
        )

        print(
            unavailable[
                [
                    "team",
                ]
            ].to_string(
                index=False
            )
        )

    display_columns = [
        column
        for column in [
            "team",
            "domestic_history_available",
            "domestic_matches",
            "last_domestic_match_date",
            "last5_ppg",
            "last5_win_rate",
            "last5_gf_pg",
            "last5_ga_pg",
            "last10_ppg",
            "momentum_last5",
        ]
        if column in latest.columns
    ]

    print(
        "\nCurrent domestic state:"
    )

    print(
        latest[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    current_raw: pd.DataFrame,
    current_exact: pd.DataFrame,
    live: pd.DataFrame,
    latest: pd.DataFrame,
    audit: pd.DataFrame,
    date_from: str,
    date_to: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEAM_AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_raw.to_csv(
        CURRENT_SEASON_PATCH_FILE,
        index=False,
        encoding="utf-8",
    )

    live.to_csv(
        LIVE_DOMESTIC_FILE,
        index=False,
        encoding="utf-8",
    )

    latest.to_csv(
        LIVE_DOMESTIC_LATEST_FILE,
        index=False,
        encoding="utf-8",
    )

    audit.to_csv(
        TEAM_AUDIT_FILE,
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
            SEASON_LABEL
        ),

        "date_from": (
            date_from
        ),

        "date_to": (
            date_to
        ),

        "historical_core_rows": (
            EXPECTED_HISTORICAL_ROWS
        ),

        "current_season_rows": int(
            len(
                current_exact
            )
        ),

        "live_combined_rows": int(
            len(
                live
            )
        ),

        "live_latest_date": (
            live[
                "date"
            ]
            .max()
            .date()
            .isoformat()
        ),

        "rules": {
            "openfootball_current_rows_ignored": True,

            "football_data_current_season_rebuilt_from_scratch": True,

            "historical_core_preserved": True,

            "base_domestic_file_overwritten": False,

            "only_finished_matches_used": True,

            "current_ucl_team_ids_canonicalized": True,

            "fuzzy_team_matching_used": False,

            "existing_domestic_form_logic_reused": True,

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
# MAIN
# ============================================================


def refresh_live_domestic() -> None:

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - REBUILD LIVE 2026/27 DOMESTIC STATE"
    )

    print(
        "=" * 75
    )

    historical = (
        load_historical_core()
    )

    (
        current_ucl_teams,
        id_to_canonical,
    ) = load_current_team_identity()

    date_from = (
        LIVE_DATE_FROM
    )

    date_to = (
        current_local_date()
    )

    print(
        f"\nHistorical core rows: "
        f"{len(historical):,}"
    )

    print(
        f"Historical latest date: "
        f"{historical['date'].max().date()}"
    )

    current_raw = (
        collect_current_season(
            date_from=date_from,
            date_to=date_to,
            id_to_canonical=id_to_canonical,
        )
    )

    print(
        f"\nTotal current-season finished "
        f"matches collected: "
        f"{len(current_raw):,}"
    )

    (
        live,
        current_exact,
    ) = build_live_dataset(
        historical=historical,
        current=current_raw,
    )

    run_quality_checks(
        historical=historical,
        current=current_exact,
        live=live,
    )

    latest = (
        build_current_domestic_state(
            live=live,
            current_ucl_teams=current_ucl_teams,
        )
    )

    audit = build_team_audit(
        latest
    )

    print_summary(
        current=current_exact,
        live=live,
        latest=latest,
    )

    save_outputs(
        current_raw=current_raw,
        current_exact=current_exact,
        live=live,
        latest=latest,
        audit=audit,
        date_from=date_from,
        date_to=date_to,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE DOMESTIC STATE REBUILT SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nCurrent-season football-data snapshot:"
    )

    print(
        CURRENT_SEASON_PATCH_FILE
    )

    print(
        "\nCombined live domestic dataset:"
    )

    print(
        LIVE_DOMESTIC_FILE
    )

    print(
        "\nCurrent 36-team domestic state:"
    )

    print(
        LIVE_DOMESTIC_LATEST_FILE
    )

    print(
        "\nCoverage audit:"
    )

    print(
        TEAM_AUDIT_FILE
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
        "domestic_matches.csv was NOT overwritten."
    )

    print(
        "Its partial 2026/27 rows were ignored."
    )

    print(
        "The live 2026/27 season was rebuilt from "
        "football-data.org."
    )


if __name__ == "__main__":
    refresh_live_domestic()