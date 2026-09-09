import requests

from src.config import (
    FOOTBALL_DATA_API_KEY,
    API_FOOTBALL_KEY,
)


FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"


def test_football_data_api() -> bool:
    """
    Test football-data.org authentication.
    """

    if not FOOTBALL_DATA_API_KEY:
        print("football-data.org key is missing.")
        return False

    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/CL"

    headers = {
        "X-Auth-Token": FOOTBALL_DATA_API_KEY,
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        print("\nfootball-data.org")
        print("-" * 50)
        print("Status: CONNECTED")
        print(f"Competition: {data.get('name')}")
        print(f"Code: {data.get('code')}")

        current_season = data.get("currentSeason", {})

        if current_season:
            print(
                f"Current season starts: "
                f"{current_season.get('startDate')}"
            )

        return True

    except requests.RequestException as error:
        print("\nfootball-data.org")
        print("-" * 50)
        print("Status: FAILED")
        print(error)

        return False


def test_api_football() -> bool:
    """
    Test API-Football authentication and UCL coverage.

    League ID 2 = UEFA Champions League.
    """

    if not API_FOOTBALL_KEY:
        print("API-Football key is missing.")
        return False

    url = f"{API_FOOTBALL_BASE_URL}/leagues"

    headers = {
        "x-apisports-key": API_FOOTBALL_KEY,
    }

    params = {
        "id": 2,
        "season": 2024,
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        print("\nAPI-Football")
        print("-" * 50)

        errors = data.get("errors")

        if errors:
            print("Status: API ERROR")
            print(errors)
            return False

        results = data.get("results", 0)

        if results == 0:
            print("Status: CONNECTED")
            print(
                "Authentication works, but no UCL 2026 "
                "league record was returned."
            )
            return True

        league_data = data["response"][0]

        league = league_data.get("league", {})
        country = league_data.get("country", {})

        print("Status: CONNECTED")
        print(f"League: {league.get('name')}")
        print(f"League ID: {league.get('id')}")
        print(f"Country/Area: {country.get('name')}")

        seasons = league_data.get("seasons", [])

        if seasons:
            season_data = seasons[0]

            print(
                f"Season: {season_data.get('year')}"
            )

            coverage = season_data.get(
                "coverage",
                {}
            )

            fixture_coverage = coverage.get(
                "fixtures",
                {}
            )

            print("\nCoverage")

            print(
                "Lineups:",
                fixture_coverage.get("lineups"),
            )

            print(
                "Fixture statistics:",
                fixture_coverage.get(
                    "statistics_fixtures"
                ),
            )

            print(
                "Player statistics:",
                fixture_coverage.get(
                    "statistics_players"
                ),
            )

            print(
                "Players:",
                coverage.get("players"),
            )

        remaining = response.headers.get(
            "x-ratelimit-requests-remaining"
        )

        if remaining is not None:
            print(
                f"\nAPI requests remaining today: "
                f"{remaining}"
            )

        return True

    except requests.RequestException as error:
        print("\nAPI-Football")
        print("-" * 50)
        print("Status: FAILED")
        print(error)

        return False


def test_all_apis() -> None:
    """
    Run both football API connectivity tests.
    """

    print("=" * 60)
    print("UCL PREDICTOR - API CONNECTION TEST")
    print("=" * 60)

    football_data_ok = test_football_data_api()

    api_football_ok = test_api_football()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        "football-data.org:",
        "OK" if football_data_ok else "FAILED",
    )

    print(
        "API-Football:",
        "OK" if api_football_ok else "FAILED",
    )


if __name__ == "__main__":
    test_all_apis()