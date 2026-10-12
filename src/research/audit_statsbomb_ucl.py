from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]

BASE_RAW_URL = (
    "https://raw.githubusercontent.com/"
    "hudl/open-data/master/data"
)

COMPETITIONS_URL = (
    f"{BASE_RAW_URL}/competitions.json"
)

UCL_COMPETITION_ID = 16

TARGET_START_YEAR = 2010
TARGET_END_YEAR = 2018

TIMEOUT_SECONDS = 30


def fetch_json(
    url: str,
) -> Any:
    response = requests.get(
        url,
        timeout=TIMEOUT_SECONDS,
        headers={
            "User-Agent": (
                "ucl-predictor-research/1.0"
            ),
        },
    )

    response.raise_for_status()

    return response.json()


def season_start_year(
    season_name: str,
) -> int | None:
    if not season_name:
        return None

    first_part = (
        str(season_name)
        .split("/")[0]
        .strip()
    )

    try:
        return int(first_part)
    except ValueError:
        return None


def stage_name(
    match: dict,
) -> str:
    competition_stage = (
        match.get(
            "competition_stage",
            {},
        )
        or {}
    )

    name = competition_stage.get(
        "name"
    )

    if name:
        return str(name)

    return "UNKNOWN"


def team_name(
    payload: dict | None,
) -> str | None:
    if not payload:
        return None

    value = payload.get(
        "home_team_name"
    )

    if value:
        return str(value)

    value = payload.get(
        "away_team_name"
    )

    if value:
        return str(value)

    value = payload.get(
        "name"
    )

    if value:
        return str(value)

    return None


def extract_match_team(
    match: dict,
    side: str,
) -> str | None:
    payload = (
        match.get(
            f"{side}_team",
            {},
        )
        or {}
    )

    direct_key = (
        f"{side}_team_name"
    )

    direct = payload.get(
        direct_key
    )

    if direct:
        return str(direct)

    name = payload.get(
        "name"
    )

    if name:
        return str(name)

    return None


def inspect_event_file(
    match_id: int,
) -> dict:
    url = (
        f"{BASE_RAW_URL}/events/"
        f"{match_id}.json"
    )

    try:
        events = fetch_json(
            url
        )
    except requests.HTTPError as exc:
        return {
            "event_file_available": False,
            "event_count_sample": None,
            "shot_count_sample": None,
            "statsbomb_xg_present": False,
            "event_error": str(exc),
        }
    except requests.RequestException as exc:
        return {
            "event_file_available": False,
            "event_count_sample": None,
            "shot_count_sample": None,
            "statsbomb_xg_present": False,
            "event_error": str(exc),
        }

    if not isinstance(
        events,
        list,
    ):
        return {
            "event_file_available": False,
            "event_count_sample": None,
            "shot_count_sample": None,
            "statsbomb_xg_present": False,
            "event_error": (
                "Unexpected event payload."
            ),
        }

    shot_events = [
        event
        for event in events
        if (
            event.get(
                "type",
                {},
            )
            or {}
        ).get(
            "name"
        )
        == "Shot"
    ]

    xg_present = any(
        (
            event.get(
                "shot",
                {},
            )
            or {}
        ).get(
            "statsbomb_xg"
        )
        is not None
        for event in shot_events
    )

    return {
        "event_file_available": True,
        "event_count_sample": len(
            events
        ),
        "shot_count_sample": len(
            shot_events
        ),
        "statsbomb_xg_present": (
            xg_present
        ),
        "event_error": None,
    }


def inspect_lineup_file(
    match_id: int,
) -> dict:
    url = (
        f"{BASE_RAW_URL}/lineups/"
        f"{match_id}.json"
    )

    try:
        lineups = fetch_json(
            url
        )
    except requests.HTTPError as exc:
        return {
            "lineup_file_available": False,
            "lineup_team_count_sample": None,
            "lineup_error": str(exc),
        }
    except requests.RequestException as exc:
        return {
            "lineup_file_available": False,
            "lineup_team_count_sample": None,
            "lineup_error": str(exc),
        }

    if not isinstance(
        lineups,
        list,
    ):
        return {
            "lineup_file_available": False,
            "lineup_team_count_sample": None,
            "lineup_error": (
                "Unexpected lineup payload."
            ),
        }

    return {
        "lineup_file_available": True,
        "lineup_team_count_sample": len(
            lineups
        ),
        "lineup_error": None,
    }


def main() -> None:
    print(
        "Loading StatsBomb competition metadata..."
    )

    competitions = fetch_json(
        COMPETITIONS_URL
    )

    ucl_seasons = []

    for competition in competitions:
        if (
            competition.get(
                "competition_id"
            )
            != UCL_COMPETITION_ID
        ):
            continue

        season_name = str(
            competition.get(
                "season_name",
                ""
            )
        )

        start_year = (
            season_start_year(
                season_name
            )
        )

        if start_year is None:
            continue

        if not (
            TARGET_START_YEAR
            <= start_year
            <= TARGET_END_YEAR
        ):
            continue

        ucl_seasons.append(
            competition
        )

    ucl_seasons.sort(
        key=lambda item: (
            season_start_year(
                str(
                    item.get(
                        "season_name",
                        ""
                    )
                )
            )
            or 0
        )
    )

    print()
    print(
        f"Target UCL seasons found: "
        f"{len(ucl_seasons)}"
    )
    print()

    summary_rows = []
    match_rows = []

    for competition in ucl_seasons:
        season_id = int(
            competition[
                "season_id"
            ]
        )

        season_name = str(
            competition[
                "season_name"
            ]
        )

        print(
            f"Auditing {season_name} "
            f"(season_id={season_id})..."
        )

        matches_url = (
            f"{BASE_RAW_URL}/matches/"
            f"{UCL_COMPETITION_ID}/"
            f"{season_id}.json"
        )

        try:
            matches = fetch_json(
                matches_url
            )
        except requests.RequestException as exc:
            summary_rows.append(
                {
                    "season_name": season_name,
                    "season_id": season_id,
                    "available": False,
                    "match_count": 0,
                    "unique_teams": 0,
                    "earliest_match": None,
                    "latest_match": None,
                    "stage_counts": None,
                    "event_file_available": False,
                    "statsbomb_xg_present": False,
                    "lineup_file_available": False,
                    "error": str(exc),
                }
            )

            continue

        if not isinstance(
            matches,
            list,
        ):
            summary_rows.append(
                {
                    "season_name": season_name,
                    "season_id": season_id,
                    "available": False,
                    "match_count": 0,
                    "unique_teams": 0,
                    "earliest_match": None,
                    "latest_match": None,
                    "stage_counts": None,
                    "event_file_available": False,
                    "statsbomb_xg_present": False,
                    "lineup_file_available": False,
                    "error": (
                        "Unexpected match payload."
                    ),
                }
            )

            continue

        stages = {}
        teams = set()
        dates = []

        for match in matches:
            match_id = match.get(
                "match_id"
            )

            home_team = (
                extract_match_team(
                    match,
                    "home",
                )
            )

            away_team = (
                extract_match_team(
                    match,
                    "away",
                )
            )

            match_date = match.get(
                "match_date"
            )

            stage = stage_name(
                match
            )

            stages[stage] = (
                stages.get(
                    stage,
                    0,
                )
                + 1
            )

            if home_team:
                teams.add(
                    home_team
                )

            if away_team:
                teams.add(
                    away_team
                )

            if match_date:
                dates.append(
                    str(
                        match_date
                    )
                )

            match_rows.append(
                {
                    "season_name": (
                        season_name
                    ),
                    "season_id": (
                        season_id
                    ),
                    "match_id": (
                        match_id
                    ),
                    "match_date": (
                        match_date
                    ),
                    "stage": stage,
                    "home_team": (
                        home_team
                    ),
                    "away_team": (
                        away_team
                    ),
                    "home_score": (
                        match.get(
                            "home_score"
                        )
                    ),
                    "away_score": (
                        match.get(
                            "away_score"
                        )
                    ),
                }
            )

        sample_event = {
            "event_file_available": False,
            "event_count_sample": None,
            "shot_count_sample": None,
            "statsbomb_xg_present": False,
            "event_error": None,
        }

        sample_lineup = {
            "lineup_file_available": False,
            "lineup_team_count_sample": None,
            "lineup_error": None,
        }

        if matches:
            sample_match_id = int(
                matches[0][
                    "match_id"
                ]
            )

            sample_event = (
                inspect_event_file(
                    sample_match_id
                )
            )

            sample_lineup = (
                inspect_lineup_file(
                    sample_match_id
                )
            )

        summary_rows.append(
            {
                "season_name": season_name,
                "season_id": season_id,
                "available": True,
                "match_count": len(
                    matches
                ),
                "unique_teams": len(
                    teams
                ),
                "earliest_match": (
                    min(dates)
                    if dates
                    else None
                ),
                "latest_match": (
                    max(dates)
                    if dates
                    else None
                ),
                "stage_counts": (
                    json.dumps(
                        stages,
                        sort_keys=True,
                    )
                ),
                "event_file_available": (
                    sample_event[
                        "event_file_available"
                    ]
                ),
                "event_count_sample": (
                    sample_event[
                        "event_count_sample"
                    ]
                ),
                "shot_count_sample": (
                    sample_event[
                        "shot_count_sample"
                    ]
                ),
                "statsbomb_xg_present": (
                    sample_event[
                        "statsbomb_xg_present"
                    ]
                ),
                "lineup_file_available": (
                    sample_lineup[
                        "lineup_file_available"
                    ]
                ),
                "lineup_team_count_sample": (
                    sample_lineup[
                        "lineup_team_count_sample"
                    ]
                ),
                "error": None,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    matches_frame = pd.DataFrame(
        match_rows
    )

    output_dir = (
        ROOT
        / "reports"
        / "research"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "statsbomb_ucl_coverage.csv"
    )

    matches_path = (
        output_dir
        / "statsbomb_ucl_matches.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    matches_frame.to_csv(
        matches_path,
        index=False,
    )

    print()
    print(
        "=" * 120
    )
    print(
        "STATSBOMB UCL COVERAGE AUDIT"
    )
    print(
        "=" * 120
    )
    print()

    if summary.empty:
        print(
            "No target Champions League seasons found."
        )
    else:
        display_columns = [
            "season_name",
            "season_id",
            "match_count",
            "unique_teams",
            "earliest_match",
            "latest_match",
            "event_file_available",
            "statsbomb_xg_present",
            "lineup_file_available",
        ]

        print(
            summary[
                display_columns
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "Stage coverage:"
        )
        print()

        for _, row in summary.iterrows():
            print(
                f"{row['season_name']}: "
                f"{row['stage_counts']}"
            )

        print()
        print(
            "Total StatsBomb UCL matches "
            f"in target period: "
            f"{int(summary['match_count'].sum())}"
        )

        xg_seasons = int(
            summary[
                "statsbomb_xg_present"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        lineup_seasons = int(
            summary[
                "lineup_file_available"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        print(
            "Seasons with sampled xG field: "
            f"{xg_seasons}/"
            f"{len(summary)}"
        )

        print(
            "Seasons with sampled lineup file: "
            f"{lineup_seasons}/"
            f"{len(summary)}"
        )

    print()
    print(
        "Saved:"
    )
    print(
        summary_path
    )
    print(
        matches_path
    )


if __name__ == "__main__":
    main()