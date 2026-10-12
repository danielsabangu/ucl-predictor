from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.team_names import (
    normalize_team_name,
    normalize_team_columns,
)


# ============================================================
# PATHS / CONFIG
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    ROOT / ".env"
)


API_KEY = os.getenv(
    "API_FOOTBALL_KEY"
)

BASE_URL = (
    "https://v3.football.api-sports.io"
)

UCL_LEAGUE_ID = 2


UCL_MASTER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ucl_matches_master.csv"
)

CACHE_DIR = (
    ROOT
    / "data"
    / "research"
    / "api_football_injury_probe_cache"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

FIXTURE_AUDIT_PATH = (
    REPORT_DIR
    / "api_football_injury_probe_fixtures.csv"
)

INJURY_RECORDS_PATH = (
    REPORT_DIR
    / "api_football_injury_probe_records.csv"
)

SCHEMA_PATHS_PATH = (
    REPORT_DIR
    / "api_football_injury_probe_schema_paths.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "api_football_injury_probe_summary.json"
)


# ============================================================
# EXPERIMENT CONTRACT
# ============================================================


SEASONS = [
    2022,
    2023,
    2024,
]


PROBE_FIXTURES_PER_SEASON = 20


EXPECTED_UCL_TOTAL_MATCHES = 2089


PROVENANCE_KEY_TERMS = {
    "created",
    "created_at",
    "creation_date",
    "reported",
    "reported_at",
    "reported_date",
    "published",
    "published_at",
    "publication_date",
    "updated",
    "updated_at",
    "last_update",
    "last_updated",
    "modified",
    "modified_at",
    "recorded",
    "recorded_at",
    "observed",
    "observed_at",
}


# ============================================================
# API HELPERS
# ============================================================


def require_api_key() -> None:
    if not API_KEY:

        raise RuntimeError(
            "API_FOOTBALL_KEY is missing from .env."
        )


def cache_path(
    name: str,
) -> Path:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        CACHE_DIR
        / name
    )


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def save_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def api_get(
    *,
    endpoint: str,
    params: dict,
    cache_file: Path,
) -> tuple[
    dict,
    bool,
]:
    """
    Return:
        payload
        came_from_cache

    API keys are never written to cache or reports.
    """

    if cache_file.exists():

        return (
            load_json(
                cache_file
            ),
            True,
        )

    require_api_key()

    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        headers={
            "x-apisports-key": API_KEY,
        },
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    payload = (
        response.json()
    )

    errors = (
        payload.get(
            "errors",
            {},
        )
        or {}
    )

    if errors:

        raise RuntimeError(
            "API-Football error for "
            f"{endpoint} {params}: "
            f"{errors}"
        )

    save_json(
        cache_file,
        payload,
    )

    return (
        payload,
        False,
    )


# ============================================================
# GENERIC JSON SCHEMA INSPECTION
# ============================================================


def collect_key_paths(
    value: Any,
    prefix: str = "",
) -> set[str]:
    paths: set[str] = set()

    if isinstance(
        value,
        dict,
    ):

        for (
            key,
            child,
        ) in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            paths.add(
                path
            )

            paths.update(
                collect_key_paths(
                    child,
                    path,
                )
            )

    elif isinstance(
        value,
        list,
    ):

        for child in value:

            paths.update(
                collect_key_paths(
                    child,
                    prefix,
                )
            )

    return paths


def final_key(
    path: str,
) -> str:
    return (
        path.split(
            "."
        )[
            -1
        ]
        .strip()
        .lower()
    )


def is_fixture_timing_path(
    path: str,
) -> bool:
    normalized = (
        path.lower()
    )

    return (
        normalized
        ==
        "fixture.timestamp"
        or
        normalized
        ==
        "fixture.date"
        or
        normalized.endswith(
            ".fixture.timestamp"
        )
        or
        normalized.endswith(
            ".fixture.date"
        )
    )


def find_provenance_paths(
    schema_paths: set[str],
) -> list[str]:
    found = []

    for path in sorted(
        schema_paths
    ):

        key = final_key(
            path
        )

        if key not in PROVENANCE_KEY_TERMS:

            continue

        if is_fixture_timing_path(
            path
        ):

            continue

        found.append(
            path
        )

    return found


# ============================================================
# UCL MASTER
# ============================================================


def load_ucl_master() -> pd.DataFrame:
    if not UCL_MASTER_PATH.exists():

        raise FileNotFoundError(
            UCL_MASTER_PATH
        )

    dataframe = pd.read_csv(
        UCL_MASTER_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
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
            "UCL master missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_UCL_TOTAL_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_UCL_TOTAL_MATCHES:,} "
            "UCL matches, found "
            f"{len(dataframe):,}."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    ).dt.normalize()

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

    dataframe = (
        normalize_team_columns(
            dataframe
        )
    )

    return dataframe


# ============================================================
# API FIXTURES
# ============================================================


def fetch_season_fixtures(
    season: int,
) -> tuple[
    list[dict],
    bool,
]:
    path = cache_path(
        f"fixtures_ucl_{season}.json"
    )

    payload, cached = api_get(
        endpoint="fixtures",
        params={
            "league": UCL_LEAGUE_ID,
            "season": season,
        },
        cache_file=path,
    )

    rows = (
        payload.get(
            "response",
            [],
        )
        or []
    )

    return (
        rows,
        cached,
    )


def flatten_provider_fixtures(
    *,
    season: int,
    fixtures: list[dict],
) -> pd.DataFrame:
    rows = []

    for item in fixtures:

        fixture = (
            item.get(
                "fixture",
                {},
            )
            or {}
        )

        teams = (
            item.get(
                "teams",
                {},
            )
            or {}
        )

        league = (
            item.get(
                "league",
                {},
            )
            or {}
        )

        home = (
            teams.get(
                "home",
                {},
            )
            or {}
        )

        away = (
            teams.get(
                "away",
                {},
            )
            or {}
        )

        status = (
            fixture.get(
                "status",
                {},
            )
            or {}
        )

        fixture_id = (
            fixture.get(
                "id"
            )
        )

        fixture_date = (
            fixture.get(
                "date"
            )
        )

        home_name = (
            home.get(
                "name"
            )
        )

        away_name = (
            away.get(
                "name"
            )
        )

        if (
            fixture_id
            is None
            or
            fixture_date
            is None
            or
            home_name
            is None
            or
            away_name
            is None
        ):

            continue

        timestamp = pd.to_datetime(
            fixture_date,
            utc=True,
            errors="raise",
        )

        rows.append(
            {
                "season": int(
                    season
                ),
                "api_fixture_id": int(
                    fixture_id
                ),
                "api_fixture_timestamp_utc": (
                    timestamp
                ),
                "date": (
                    timestamp
                    .tz_convert(
                        None
                    )
                    .normalize()
                ),
                "api_home_team_id": (
                    home.get(
                        "id"
                    )
                ),
                "api_away_team_id": (
                    away.get(
                        "id"
                    )
                ),
                "api_home_team_raw": (
                    home_name
                ),
                "api_away_team_raw": (
                    away_name
                ),
                "home_team": (
                    normalize_team_name(
                        str(
                            home_name
                        )
                    )
                ),
                "away_team": (
                    normalize_team_name(
                        str(
                            away_name
                        )
                    )
                ),
                "status_short": (
                    status.get(
                        "short"
                    )
                ),
                "status_long": (
                    status.get(
                        "long"
                    )
                ),
                "round": (
                    league.get(
                        "round"
                    )
                ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:

        return dataframe

    if dataframe[
        "api_fixture_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate provider fixture IDs "
            f"for season {season}."
        )

    return (
        dataframe.sort_values(
            [
                "date",
                "api_fixture_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# EXACT PROVIDER -> MASTER MATCHING
# ============================================================


def match_to_ucl_master(
    *,
    master: pd.DataFrame,
    provider: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    historical = master[
        master[
            "season"
        ]
        ==
        season
    ][
        [
            "match_id",
            "date",
            "season",
            "home_team",
            "away_team",
        ]
    ].copy()

    provider_subset = provider[
        [
            "api_fixture_id",
            "api_fixture_timestamp_utc",
            "date",
            "season",
            "home_team",
            "away_team",
            "api_home_team_id",
            "api_away_team_id",
            "api_home_team_raw",
            "api_away_team_raw",
            "status_short",
            "status_long",
            "round",
        ]
    ].copy()

    duplicate_mask = (
        provider_subset.duplicated(
            subset=[
                "season",
                "date",
                "home_team",
                "away_team",
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():

        examples = (
            provider_subset.loc[
                duplicate_mask,
                [
                    "api_fixture_id",
                    "date",
                    "home_team",
                    "away_team",
                ],
            ]
            .head(
                20
            )
        )

        raise RuntimeError(
            "Provider fixture identity is not "
            "unique after canonicalization:\n"
            +
            examples.to_string(
                index=False
            )
        )

    merged = historical.merge(
        provider_subset,
        on=[
            "season",
            "date",
            "home_team",
            "away_team",
        ],
        how="left",
        validate="one_to_one",
    )

    merged[
        "api_fixture_matched"
    ] = (
        merged[
            "api_fixture_id"
        ].notna()
    )

    return merged


# ============================================================
# SMALL REPRESENTATIVE PROBE
# ============================================================


def select_probe_fixtures(
    matched: pd.DataFrame,
) -> pd.DataFrame:
    available = matched[
        matched[
            "api_fixture_matched"
        ]
    ].copy()

    available = (
        available.sort_values(
            [
                "date",
                "api_fixture_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if available.empty:

        return available

    count = min(
        PROBE_FIXTURES_PER_SEASON,
        len(
            available
        ),
    )

    positions = np.linspace(
        0,
        len(
            available
        )
        - 1,
        num=count,
        dtype=int,
    )

    positions = np.unique(
        positions
    )

    return (
        available.iloc[
            positions
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# INJURIES
# ============================================================


def fetch_season_injuries(
    season: int,
) -> tuple[
    list[dict],
    bool,
    dict,
]:
    """
    Free-plan-safe strategy.

    We intentionally DO NOT use the multi-fixture `ids`
    parameter because the current free API-Football plan
    rejects it.

    Officially supported query:
        /injuries?league=2&season=YYYY
    """

    path = cache_path(
        f"injuries_ucl_{season}_league_season.json"
    )

    payload, cached = api_get(
        endpoint="injuries",
        params={
            "league": UCL_LEAGUE_ID,
            "season": season,
        },
        cache_file=path,
    )

    rows = (
        payload.get(
            "response",
            [],
        )
        or []
    )

    paging = (
        payload.get(
            "paging",
            {},
        )
        or {}
    )

    return (
        rows,
        cached,
        paging,
    )


def flatten_injury_records(
    *,
    season: int,
    records: list[dict],
) -> pd.DataFrame:
    rows = []

    for item in records:

        player = (
            item.get(
                "player",
                {},
            )
            or {}
        )

        team = (
            item.get(
                "team",
                {},
            )
            or {}
        )

        fixture = (
            item.get(
                "fixture",
                {},
            )
            or {}
        )

        league = (
            item.get(
                "league",
                {},
            )
            or {}
        )

        fixture_id = (
            fixture.get(
                "id"
            )
        )

        if fixture_id is None:

            continue

        team_name = (
            team.get(
                "name"
            )
        )

        rows.append(
            {
                "probe_season": int(
                    season
                ),
                "api_fixture_id": int(
                    fixture_id
                ),
                "fixture_date": (
                    fixture.get(
                        "date"
                    )
                ),
                "fixture_timestamp": (
                    fixture.get(
                        "timestamp"
                    )
                ),
                "fixture_timezone": (
                    fixture.get(
                        "timezone"
                    )
                ),
                "league_id": (
                    league.get(
                        "id"
                    )
                ),
                "league_name": (
                    league.get(
                        "name"
                    )
                ),
                "league_season": (
                    league.get(
                        "season"
                    )
                ),
                "team_id": (
                    team.get(
                        "id"
                    )
                ),
                "team_name_raw": (
                    team_name
                ),
                "team_name_canonical": (
                    normalize_team_name(
                        str(
                            team_name
                        )
                    )
                    if team_name
                    is not None
                    else None
                ),
                "player_id": (
                    player.get(
                        "id"
                    )
                ),
                "player_name": (
                    player.get(
                        "name"
                    )
                ),
                "availability_type": (
                    player.get(
                        "type"
                    )
                ),
                "reason": (
                    player.get(
                        "reason"
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROBE FIXTURE COUNTS
# ============================================================


def attach_injury_counts(
    *,
    probe: pd.DataFrame,
    season_injuries: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        probe.copy()
    )

    integer_columns = [
        "injury_records_returned",
        "home_records",
        "away_records",
        "other_team_records",
        "missing_fixture_records",
        "questionable_records",
        "suspension_reason_records",
        "other_reason_records",
    ]

    for column in integer_columns:

        result[
            column
        ] = 0

    for index in result.index:

        fixture_id = int(
            result.loc[
                index,
                "api_fixture_id",
            ]
        )

        subset = season_injuries[
            season_injuries[
                "api_fixture_id"
            ]
            ==
            fixture_id
        ].copy()

        result.loc[
            index,
            "injury_records_returned",
        ] = int(
            len(
                subset
            )
        )

        if subset.empty:

            continue

        home_id = (
            result.loc[
                index,
                "api_home_team_id",
            ]
        )

        away_id = (
            result.loc[
                index,
                "api_away_team_id",
            ]
        )

        home_mask = (
            subset[
                "team_id"
            ]
            ==
            home_id
        )

        away_mask = (
            subset[
                "team_id"
            ]
            ==
            away_id
        )

        result.loc[
            index,
            "home_records",
        ] = int(
            home_mask.sum()
        )

        result.loc[
            index,
            "away_records",
        ] = int(
            away_mask.sum()
        )

        result.loc[
            index,
            "other_team_records",
        ] = int(
            (
                ~home_mask
                &
                ~away_mask
            ).sum()
        )

        availability = (
            subset[
                "availability_type"
            ]
            .fillna(
                ""
            )
            .astype(
                str
            )
            .str.strip()
            .str.lower()
        )

        reasons = (
            subset[
                "reason"
            ]
            .fillna(
                ""
            )
            .astype(
                str
            )
            .str.strip()
            .str.lower()
        )

        result.loc[
            index,
            "missing_fixture_records",
        ] = int(
            (
                availability
                ==
                "missing fixture"
            ).sum()
        )

        result.loc[
            index,
            "questionable_records",
        ] = int(
            (
                availability
                ==
                "questionable"
            ).sum()
        )

        suspension_mask = (
            reasons.str.contains(
                "suspend",
                regex=False,
            )
        )

        result.loc[
            index,
            "suspension_reason_records",
        ] = int(
            suspension_mask.sum()
        )

        result.loc[
            index,
            "other_reason_records",
        ] = int(
            (
                ~suspension_mask
            ).sum()
        )

    result[
        "zero_record_interpretation"
    ] = result[
        "injury_records_returned"
    ].map(
        lambda value: (
            "AMBIGUOUS_NO_RECORDS_RETURNED"
            if int(
                value
            )
            ==
            0
            else
            "RECORDS_RETURNED"
        )
    )

    return result


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=" * 110
    )

    print(
        "UCL PREDICTOR V2 - CANDIDATE D0"
    )

    print(
        "API-FOOTBALL PLAYER AVAILABILITY "
        "SNAPSHOT-SAFETY PROBE V2"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Determine whether historical injury / "
        "suspension records contain trustworthy "
        "record-level timing proving that the "
        "information existed before kickoff."
    )

    print()

    print(
        "Free-plan request strategy:"
    )

    print(
        "  fixtures: league + season"
    )

    print(
        "  injuries: league + season"
    )

    print(
        "  multi-fixture ids parameter: NOT USED"
    )

    print()

    print(
        "No model training."
    )

    print(
        "No 2025/26."
    )

    print(
        "No production changes."
    )

    print(
        "Cached responses are reused."
    )

    print()

    # --------------------------------------------------------
    # MASTER
    # --------------------------------------------------------

    print(
        "Loading canonical UCL master..."
    )

    master = (
        load_ucl_master()
    )

    master_probe = master[
        master[
            "season"
        ].isin(
            SEASONS
        )
    ].copy()

    print(
        f"Historical UCL matches in "
        f"2022-2024: "
        f"{len(master_probe):,}"
    )

    # --------------------------------------------------------
    # ACCUMULATORS
    # --------------------------------------------------------

    fixture_audits = []

    injury_frames = []

    season_summaries = []

    all_schema_paths: set[str] = set()

    total_new_api_calls = 0

    total_cached_calls = 0

    # --------------------------------------------------------
    # SEASONS
    # --------------------------------------------------------

    for season in SEASONS:

        print()
        print(
            "=" * 110
        )

        print(
            f"SEASON "
            f"{season}/"
            f"{str(season + 1)[-2:]}"
        )

        print(
            "=" * 110
        )

        # ----------------------------------------------------
        # FIXTURES
        # ----------------------------------------------------

        (
            fixture_rows,
            fixture_cached,
        ) = fetch_season_fixtures(
            season
        )

        if fixture_cached:

            total_cached_calls += 1

            print(
                "Fixtures source: CACHE"
            )

        else:

            total_new_api_calls += 1

            print(
                "Fixtures source: API"
            )

        provider = (
            flatten_provider_fixtures(
                season=season,
                fixtures=fixture_rows,
            )
        )

        matched = (
            match_to_ucl_master(
                master=master,
                provider=provider,
                season=season,
            )
        )

        matched_count = int(
            matched[
                "api_fixture_matched"
            ].sum()
        )

        historical_count = int(
            len(
                matched
            )
        )

        print(
            f"Master fixtures: "
            f"{historical_count}"
        )

        print(
            f"Exact provider fixture matches: "
            f"{matched_count}/"
            f"{historical_count} "
            f"("
            f"{matched_count / historical_count * 100:.2f}%"
            f")"
        )

        probe = (
            select_probe_fixtures(
                matched
            )
        )

        print(
            f"Probe fixtures selected: "
            f"{len(probe)}"
        )

        # ----------------------------------------------------
        # SEASON INJURIES
        # ----------------------------------------------------

        (
            raw_injury_rows,
            injury_cached,
            paging,
        ) = fetch_season_injuries(
            season
        )

        if injury_cached:

            total_cached_calls += 1

            print(
                "Season injuries source: CACHE"
            )

        else:

            total_new_api_calls += 1

            print(
                "Season injuries source: API"
            )

        print(
            f"Season injury records returned: "
            f"{len(raw_injury_rows)}"
        )

        print(
            "Injury response paging: "
            f"current={paging.get('current')} "
            f"total={paging.get('total')}"
        )

        for raw_record in raw_injury_rows:

            all_schema_paths.update(
                collect_key_paths(
                    raw_record
                )
            )

        season_injuries = (
            flatten_injury_records(
                season=season,
                records=raw_injury_rows,
            )
        )

        if not season_injuries.empty:

            injury_frames.append(
                season_injuries
            )

        # ----------------------------------------------------
        # FILTER TO PROBE FIXTURES LOCALLY
        # ----------------------------------------------------

        probe_ids = set(
            probe[
                "api_fixture_id"
            ]
            .astype(
                int
            )
            .tolist()
        )

        if season_injuries.empty:

            probe_injuries = (
                season_injuries.copy()
            )

        else:

            probe_injuries = (
                season_injuries[
                    season_injuries[
                        "api_fixture_id"
                    ].isin(
                        probe_ids
                    )
                ]
                .copy()
            )

        fixture_audit = (
            attach_injury_counts(
                probe=probe,
                season_injuries=season_injuries,
            )
        )

        fixture_audit[
            "probe_season"
        ] = season

        fixture_audits.append(
            fixture_audit
        )

        fixtures_with_records = int(
            fixture_audit[
                "injury_records_returned"
            ]
            .gt(
                0
            )
            .sum()
        )

        season_summaries.append(
            {
                "season": int(
                    season
                ),
                "master_matches": (
                    historical_count
                ),
                "provider_matches": int(
                    len(
                        provider
                    )
                ),
                "exact_matches": (
                    matched_count
                ),
                "probe_fixtures": int(
                    len(
                        probe
                    )
                ),
                "season_injury_records": int(
                    len(
                        season_injuries
                    )
                ),
                "probe_injury_records": int(
                    len(
                        probe_injuries
                    )
                ),
                "probe_fixtures_with_records": (
                    fixtures_with_records
                ),
                "paging_current": (
                    paging.get(
                        "current"
                    )
                ),
                "paging_total": (
                    paging.get(
                        "total"
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    fixture_audit = pd.concat(
        fixture_audits,
        ignore_index=True,
    )

    if injury_frames:

        injury_records = pd.concat(
            injury_frames,
            ignore_index=True,
        )

    else:

        injury_records = pd.DataFrame(
            columns=[
                "probe_season",
                "api_fixture_id",
                "fixture_date",
                "fixture_timestamp",
                "fixture_timezone",
                "league_id",
                "league_name",
                "league_season",
                "team_id",
                "team_name_raw",
                "team_name_canonical",
                "player_id",
                "player_name",
                "availability_type",
                "reason",
            ]
        )

    season_summary = pd.DataFrame(
        season_summaries
    )

    # --------------------------------------------------------
    # SCHEMA SAFETY
    # --------------------------------------------------------

    provenance_paths = (
        find_provenance_paths(
            all_schema_paths
        )
    )

    fixture_timestamp_paths = [
        path
        for path
        in sorted(
            all_schema_paths
        )
        if is_fixture_timing_path(
            path
        )
    ]

    has_record_provenance_timestamp = bool(
        provenance_paths
    )

    # IMPORTANT:
    # Merely detecting a timestamp-like field does not
    # automatically make the source safe. It would have
    # to be independently established as the timestamp at
    # which that availability record became known, and then
    # verified <= kickoff for every training observation.
    pre_match_snapshot_proven = False

    historical_training_safe = False

    # --------------------------------------------------------
    # CONTENT AUDIT
    # --------------------------------------------------------

    if injury_records.empty:

        type_counts = {}

        reason_counts = {}

        suspension_reason_records = 0

        records_with_player_id = 0

        records_with_team_id = 0

        unique_fixture_ids_with_records = 0

    else:

        type_counts = (
            injury_records[
                "availability_type"
            ]
            .fillna(
                "<NULL>"
            )
            .astype(
                str
            )
            .value_counts()
            .to_dict()
        )

        reason_counts = (
            injury_records[
                "reason"
            ]
            .fillna(
                "<NULL>"
            )
            .astype(
                str
            )
            .value_counts()
            .head(
                30
            )
            .to_dict()
        )

        suspension_reason_records = int(
            injury_records[
                "reason"
            ]
            .fillna(
                ""
            )
            .astype(
                str
            )
            .str.contains(
                "suspend",
                case=False,
                regex=False,
            )
            .sum()
        )

        records_with_player_id = int(
            injury_records[
                "player_id"
            ]
            .notna()
            .sum()
        )

        records_with_team_id = int(
            injury_records[
                "team_id"
            ]
            .notna()
            .sum()
        )

        unique_fixture_ids_with_records = int(
            injury_records[
                "api_fixture_id"
            ]
            .nunique()
        )

    # --------------------------------------------------------
    # DISPLAY MATCHING
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "FIXTURE / INJURY COVERAGE SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    print(
        season_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # RESPONSE CONTENT
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "INJURY RESPONSE CONTENT"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Probe fixtures total: "
        f"{len(fixture_audit)}"
    )

    print(
        f"All season injury records returned: "
        f"{len(injury_records)}"
    )

    print(
        f"Unique fixture IDs with records: "
        f"{unique_fixture_ids_with_records}"
    )

    print(
        f"Records with player ID: "
        f"{records_with_player_id}/"
        f"{len(injury_records)}"
    )

    print(
        f"Records with team ID: "
        f"{records_with_team_id}/"
        f"{len(injury_records)}"
    )

    print(
        f"Suspension-reason records: "
        f"{suspension_reason_records}"
    )

    print()

    print(
        "Availability type counts:"
    )

    if type_counts:

        for (
            label,
            count,
        ) in type_counts.items():

            print(
                f"  {label}: {count}"
            )

    else:

        print(
            "  NONE"
        )

    print()

    print(
        "Top reason counts:"
    )

    if reason_counts:

        for (
            label,
            count,
        ) in reason_counts.items():

            print(
                f"  {label}: {count}"
            )

    else:

        print(
            "  NONE"
        )

    # --------------------------------------------------------
    # SNAPSHOT SAFETY
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "SNAPSHOT-SAFETY AUDIT"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Fixture kickoff timestamp fields:"
    )

    if fixture_timestamp_paths:

        for path in (
            fixture_timestamp_paths
        ):

            print(
                f"  {path}"
            )

    else:

        print(
            "  NONE"
        )

    print()

    print(
        "Candidate record creation / publication / "
        "update timestamp fields:"
    )

    if provenance_paths:

        for path in (
            provenance_paths
        ):

            print(
                f"  {path}"
            )

    else:

        print(
            "  NONE"
        )

    print()

    print(
        "Record-level provenance timestamp present: "
        f"{'YES' if has_record_provenance_timestamp else 'NO'}"
    )

    print(
        "Can prove record existed before kickoff: "
        f"{'YES' if pre_match_snapshot_proven else 'NO'}"
    )

    print(
        "Safe for historical training under "
        "strict leakage policy: "
        f"{'YES' if historical_training_safe else 'NO'}"
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "CANDIDATE D0 DECISION"
    )

    print(
        "=" * 110
    )

    print()

    if historical_training_safe:

        decision = (
            "PROCEED_TO_FULL_PLAYER_AVAILABILITY_AUDIT"
        )

        print(
            f"Decision: {decision}"
        )

    else:

        decision = (
            "REJECT_API_INJURIES_FOR_HISTORICAL_TRAINING"
        )

        print(
            f"Decision: {decision}"
        )

        print()

        if not has_record_provenance_timestamp:

            print(
                "No record-level creation/publication/update "
                "timestamp was present in the historical "
                "injury response."
            )

        else:

            print(
                "Timestamp-like record fields were detected, "
                "but this audit has not established that they "
                "represent when the availability information "
                "became known."
            )

        print()

        print(
            "Fixture date/timestamp describes kickoff, "
            "not when the injury record was observed."
        )

        print()

        print(
            "Therefore these retrospective records cannot "
            "enter historical model training under our "
            "strict leakage policy."
        )

        print()

        print(
            "They may still be useful later for LIVE "
            "pre-match context because a live retrieval "
            "has its own observation timestamp."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixture_audit.to_csv(
        FIXTURE_AUDIT_PATH,
        index=False,
    )

    injury_records.to_csv(
        INJURY_RECORDS_PATH,
        index=False,
    )

    schema_rows = [
        {
            "schema_path": (
                path
            ),
            "is_fixture_timing": bool(
                is_fixture_timing_path(
                    path
                )
            ),
            "is_record_provenance_candidate": bool(
                path
                in
                provenance_paths
            ),
        }
        for path
        in sorted(
            all_schema_paths
        )
    ]

    schema_frame = pd.DataFrame(
        schema_rows
    )

    schema_frame.to_csv(
        SCHEMA_PATHS_PATH,
        index=False,
    )

    summary = {
        "experiment": (
            "candidate_d0_api_football_injury_snapshot_safety_v2"
        ),
        "status": (
            "research_only"
        ),
        "request_strategy": (
            "league_plus_season_free_plan_safe"
        ),
        "ids_parameter_used": False,
        "seasons": (
            SEASONS
        ),
        "probe_fixtures_per_season": (
            PROBE_FIXTURES_PER_SEASON
        ),
        "probe_fixture_count": int(
            len(
                fixture_audit
            )
        ),
        "all_season_injury_record_count": int(
            len(
                injury_records
            )
        ),
        "unique_fixture_ids_with_records": (
            unique_fixture_ids_with_records
        ),
        "availability_type_counts": (
            type_counts
        ),
        "top_reason_counts": (
            reason_counts
        ),
        "suspension_reason_records": (
            suspension_reason_records
        ),
        "records_with_player_id": (
            records_with_player_id
        ),
        "records_with_team_id": (
            records_with_team_id
        ),
        "fixture_timestamp_paths": (
            fixture_timestamp_paths
        ),
        "record_provenance_paths": (
            provenance_paths
        ),
        "record_provenance_timestamp_present": (
            has_record_provenance_timestamp
        ),
        "pre_match_snapshot_proven": (
            pre_match_snapshot_proven
        ),
        "historical_training_safe": (
            historical_training_safe
        ),
        "decision": (
            decision
        ),
        "new_api_calls": int(
            total_new_api_calls
        ),
        "cached_api_calls": int(
            total_cached_calls
        ),
        "zero_injury_records_interpretation": (
            "ambiguous_not_proof_of_full_availability"
        ),
        "fixture_kickoff_timestamp_counts_as_record_provenance": False,
        "production_modified": False,
        "model_trained": False,
        "confirmation_2025_used": False,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "API key written to reports/cache: NO"
    )

    print(
        "Restricted ids parameter used: NO"
    )

    print(
        "League + season injury query used: YES"
    )

    print(
        "Cached API responses reused: YES"
    )

    print(
        "No fuzzy team matching: PASS"
    )

    print(
        "Fixture timestamp treated as "
        "record provenance: NO"
    )

    print(
        "Zero injury rows treated as "
        "fully healthy squad: NO"
    )

    print(
        "2025/26 used: NO"
    )

    print(
        "Model trained: NO"
    )

    print(
        "Production modified: NO"
    )

    print()

    print(
        f"New API calls this run: "
        f"{total_new_api_calls}"
    )

    print(
        f"Cached API calls this run: "
        f"{total_cached_calls}"
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    print()
    print(
        "=" * 110
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 110
    )

    print()

    print(
        FIXTURE_AUDIT_PATH
    )

    print(
        INJURY_RECORDS_PATH
    )

    print(
        SCHEMA_PATHS_PATH
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()