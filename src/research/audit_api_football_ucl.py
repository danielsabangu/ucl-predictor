from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


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

SEASONS = list(
    range(2010, 2027)
)


def fetch_season_coverage(
    season: int,
) -> dict:
    if not API_KEY:
        raise RuntimeError(
            "API_FOOTBALL_KEY is missing from .env."
        )

    response = requests.get(
        f"{BASE_URL}/leagues",
        headers={
            "x-apisports-key": API_KEY,
        },
        params={
            "id": UCL_LEAGUE_ID,
            "season": season,
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    errors = (
        payload.get(
            "errors",
            {},
        )
        or {}
    )

    if errors:
        plan_error = errors.get(
            "plan"
        )

        if plan_error:
            return {
                "season": season,
                "available": False,
                "access_status": (
                    "PLAN_RESTRICTED"
                ),
                "access_message": str(
                    plan_error
                ),
                "start": None,
                "end": None,
                "events": None,
                "lineups": None,
                "fixture_statistics": None,
                "player_statistics": None,
                "standings": None,
                "players": None,
                "injuries": None,
                "predictions": None,
                "odds": None,
            }

        return {
            "season": season,
            "available": False,
            "access_status": "API_ERROR",
            "access_message": str(
                errors
            ),
            "start": None,
            "end": None,
            "events": None,
            "lineups": None,
            "fixture_statistics": None,
            "player_statistics": None,
            "standings": None,
            "players": None,
            "injuries": None,
            "predictions": None,
            "odds": None,
        }

    rows = (
        payload.get(
            "response",
            [],
        )
        or []
    )

    if not rows:
        return {
            "season": season,
            "available": False,
            "access_status": (
                "NO_RESPONSE_DATA"
            ),
            "access_message": None,
            "start": None,
            "end": None,
            "events": None,
            "lineups": None,
            "fixture_statistics": None,
            "player_statistics": None,
            "standings": None,
            "players": None,
            "injuries": None,
            "predictions": None,
            "odds": None,
        }

    season_info = []

    for row in rows:
        row_seasons = (
            row.get(
                "seasons",
                [],
            )
            or []
        )

        season_info.extend(
            row_seasons
        )

    matching = [
        item
        for item in season_info
        if item.get("year") == season
    ]

    if not matching:
        return {
            "season": season,
            "available": False,
            "access_status": (
                "SEASON_NOT_LISTED"
            ),
            "access_message": None,
            "start": None,
            "end": None,
            "events": None,
            "lineups": None,
            "fixture_statistics": None,
            "player_statistics": None,
            "standings": None,
            "players": None,
            "injuries": None,
            "predictions": None,
            "odds": None,
        }

    info = matching[0]

    coverage = (
        info.get(
            "coverage",
            {},
        )
        or {}
    )

    fixtures = (
        coverage.get(
            "fixtures",
            {},
        )
        or {}
    )

    return {
        "season": season,
        "available": True,
        "access_status": "AVAILABLE",
        "access_message": None,
        "start": info.get(
            "start"
        ),
        "end": info.get(
            "end"
        ),
        "events": fixtures.get(
            "events"
        ),
        "lineups": fixtures.get(
            "lineups"
        ),
        "fixture_statistics": (
            fixtures.get(
                "statistics_fixtures"
            )
        ),
        "player_statistics": (
            fixtures.get(
                "statistics_players"
            )
        ),
        "standings": coverage.get(
            "standings"
        ),
        "players": coverage.get(
            "players"
        ),
        "injuries": coverage.get(
            "injuries"
        ),
        "predictions": coverage.get(
            "predictions"
        ),
        "odds": coverage.get(
            "odds"
        ),
    }


def main() -> None:
    rows = []

    for season in SEASONS:
        season_label = (
            f"{season}/"
            f"{str(season + 1)[-2:]}"
        )

        print(
            f"Checking UCL {season_label}..."
        )

        try:
            result = (
                fetch_season_coverage(
                    season
                )
            )

        except requests.RequestException as exc:
            result = {
                "season": season,
                "available": False,
                "access_status": (
                    "REQUEST_ERROR"
                ),
                "access_message": str(
                    exc
                ),
                "start": None,
                "end": None,
                "events": None,
                "lineups": None,
                "fixture_statistics": None,
                "player_statistics": None,
                "standings": None,
                "players": None,
                "injuries": None,
                "predictions": None,
                "odds": None,
            }

        rows.append(
            result
        )

    frame = pd.DataFrame(
        rows
    )

    preferred_columns = [
        "season",
        "available",
        "access_status",
        "access_message",
        "start",
        "end",
        "events",
        "lineups",
        "fixture_statistics",
        "player_statistics",
        "standings",
        "players",
        "injuries",
        "predictions",
        "odds",
    ]

    frame = frame.reindex(
        columns=preferred_columns
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

    output_path = (
        output_dir
        / "api_football_ucl_coverage.csv"
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "API-FOOTBALL UCL HISTORICAL COVERAGE"
    )
    print(
        "=" * 100
    )
    print()

    display_columns = [
        "season",
        "available",
        "access_status",
        "events",
        "lineups",
        "fixture_statistics",
        "player_statistics",
        "injuries",
        "predictions",
        "odds",
    ]

    print(
        frame[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "ACCESS SUMMARY"
    )
    print(
        "=" * 100
    )
    print()

    status_counts = (
        frame[
            "access_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        status_counts.to_string()
    )

    available = frame[
        frame[
            "access_status"
        ]
        == "AVAILABLE"
    ]

    print()
    print(
        f"Available seasons: "
        f"{len(available)}/{len(frame)}"
    )

    if not available.empty:
        print()
        print(
            "Available season years:"
        )

        print(
            ", ".join(
                str(
                    int(season)
                )
                for season
                in available[
                    "season"
                ].tolist()
            )
        )

        print()
        print(
            "Advanced-data availability:"
        )

        for column in [
            "events",
            "lineups",
            "fixture_statistics",
            "player_statistics",
            "injuries",
            "predictions",
            "odds",
        ]:
            supported = (
                available[column]
                .fillna(False)
                .astype(bool)
                .sum()
            )

            print(
                f"  {column:<20}: "
                f"{supported}/"
                f"{len(available)} seasons"
            )

    restricted = frame[
        frame[
            "access_status"
        ]
        == "PLAN_RESTRICTED"
    ]

    if not restricted.empty:
        print()
        print(
            "Plan-restricted seasons:"
        )

        print(
            ", ".join(
                str(
                    int(season)
                )
                for season
                in restricted[
                    "season"
                ].tolist()
            )
        )

        messages = (
            restricted[
                "access_message"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if messages:
            print()
            print(
                "Provider plan message:"
            )

            for message in messages:
                print(
                    f"  {message}"
                )

    print()
    print(
        "=" * 100
    )
    print(
        "OUTPUT"
    )
    print(
        "=" * 100
    )
    print()
    print(
        f"Saved coverage report:"
    )
    print(
        output_path
    )


if __name__ == "__main__":
    main()