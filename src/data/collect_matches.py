import json
import time
from pathlib import Path

import kagglehub
import pandas as pd
import requests

from src.config import (
    FOOTBALL_DATA_API_KEY,
    API_FOOTBALL_KEY,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

UCL_KAGGLE_DATASET = "raminvali/uefa-champions-league"

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_UCL_CODE = "CL"

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_UCL_ID = 2

RECENT_FOOTBALL_DATA_SEASONS = [
    2023,
    2024,
    2025,
]


# ============================================================
# COMMON HELPERS
# ============================================================


def determine_result(
    home_goals: int,
    away_goals: int,
) -> str:
    """
    Return the result from the home team's perspective.

    H = Home win
    D = Draw
    A = Away win
    """

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def season_label(season: int) -> str:
    """
    Convert a starting season year into a readable label.

    Example:
    2022 -> 2022/23
    """

    return f"{season}/{str(season + 1)[-2:]}"


def is_ucl_competition_proper(stage: str) -> bool:
    """
    Keep only Champions League competition-proper matches.

    Included:
    - Group-stage matches such as "Group A - 1"
    - Round of 16
    - Quarter-finals
    - Semi-finals
    - Final

    Excluded:
    - Preliminary Round
    - Qualifying rounds
    - Play-offs
    """

    if not isinstance(stage, str):
        return False

    stage = stage.strip().lower()

    # API-Football 2022/23 group-stage labels look like:
    # "Group A - 1"
    # "Group B - 2"
    # etc.
    if stage.startswith("group "):
        return True

    return stage in {
        "round of 16",
        "quarter-finals",
        "semi-finals",
        "final",
    }


# ============================================================
# KAGGLE HISTORICAL DATA
# ============================================================


def download_historical_ucl_data() -> Path:
    """
    Download the historical Champions League Kaggle dataset.

    This is currently our historical base covering approximately
    2010/11 through 2021/22.
    """

    output_dir = (
        RAW_DATA_DIR
        / "matches"
        / "ucl_kaggle"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("UCL PREDICTOR - HISTORICAL KAGGLE DATA")
    print("=" * 65)

    print("\nDownloading historical Champions League dataset...")
    print(f"Dataset: {UCL_KAGGLE_DATASET}")
    print(f"Destination: {output_dir}")

    downloaded_path = kagglehub.dataset_download(
        UCL_KAGGLE_DATASET,
        output_dir=str(output_dir),
    )

    print("\nDownload complete.")
    print(f"Dataset location: {downloaded_path}")

    return Path(downloaded_path)


# ============================================================
# FOOTBALL-DATA.ORG
# ============================================================


def get_football_data_headers() -> dict:
    """
    Return authentication headers for football-data.org.
    """

    if not FOOTBALL_DATA_API_KEY:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY is missing from .env"
        )

    return {
        "X-Auth-Token": FOOTBALL_DATA_API_KEY,
    }


def fetch_football_data_season(
    season: int,
) -> dict | None:
    """
    Download one Champions League season using football-data.org.

    Example:
    2024 = 2024/25
    """

    url = (
        f"{FOOTBALL_DATA_BASE_URL}"
        f"/competitions/{FOOTBALL_DATA_UCL_CODE}/matches"
    )

    params = {
        "season": season,
    }

    print(
        f"\nRequesting UCL {season_label(season)} "
        f"from football-data.org..."
    )

    try:
        response = requests.get(
            url,
            headers=get_football_data_headers(),
            params=params,
            timeout=30,
        )

    except requests.RequestException as error:
        print(f"Request failed: {error}")
        return None

    print(
        f"HTTP status: {response.status_code}"
    )

    if response.status_code == 403:
        print(
            f"Access denied for season {season_label(season)}."
        )
        return None

    if response.status_code == 429:
        print(
            "Rate limit reached."
        )
        return None

    if response.status_code != 200:
        print(
            f"Unexpected response:\n"
            f"{response.text[:500]}"
        )
        return None

    data = response.json()

    print(
        f"Matches returned: "
        f"{len(data.get('matches', [])):,}"
    )

    return data


def save_football_data_raw_json(
    season: int,
    data: dict,
) -> Path:
    """
    Save untouched football-data.org JSON.
    """

    output_dir = (
        RAW_DATA_DIR
        / "matches"
        / "football_data"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"ucl_{season}_{season + 1}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Raw JSON saved: {output_file}"
    )

    return output_file


def convert_football_data_season(
    season: int,
    data: dict,
) -> pd.DataFrame:
    """
    Convert football-data.org matches into our project structure.
    """

    rows = []

    matches = data.get(
        "matches",
        [],
    )

    for match in matches:

        status = match.get(
            "status"
        )

        if status != "FINISHED":
            continue

        score = match.get(
            "score",
            {},
        )

        full_time = score.get(
            "fullTime",
            {},
        )

        home_goals = full_time.get(
            "home"
        )

        away_goals = full_time.get(
            "away"
        )

        if (
            home_goals is None
            or away_goals is None
        ):
            continue

        utc_date = match.get(
            "utcDate"
        )

        if not utc_date:
            continue

        date = pd.to_datetime(
            utc_date,
            utc=True,
        )

        home_team = (
            match.get(
                "homeTeam",
                {},
            ).get(
                "name"
            )
        )

        away_team = (
            match.get(
                "awayTeam",
                {},
            ).get(
                "name"
            )
        )

        if not home_team or not away_team:
            continue

        home_goals = int(
            home_goals
        )

        away_goals = int(
            away_goals
        )

        rows.append(
            {
                "api_match_id": match.get("id"),
                "date": date.date().isoformat(),
                "day": date.day_name()[:3],
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": determine_result(
                    home_goals,
                    away_goals,
                ),
                "scoreline": (
                    f"{home_goals}-{away_goals}"
                ),
                "competition": "UEFA Champions League",
                "source": "football_data_org",
                "stage": match.get("stage"),
                "status": status,
            }
        )

    return pd.DataFrame(
        rows
    )


def download_recent_ucl_from_football_data() -> pd.DataFrame:
    """
    Download UCL seasons available through football-data.org.

    Currently used for:
    2023/24
    2024/25
    2025/26
    """

    print("=" * 65)
    print("UCL PREDICTOR - FOOTBALL-DATA.ORG")
    print("=" * 65)

    season_frames = []

    for index, season in enumerate(
        RECENT_FOOTBALL_DATA_SEASONS
    ):

        data = fetch_football_data_season(
            season
        )

        if data is None:
            continue

        save_football_data_raw_json(
            season,
            data,
        )

        season_df = convert_football_data_season(
            season,
            data,
        )

        print(
            f"Finished matches kept: "
            f"{len(season_df):,}"
        )

        if not season_df.empty:
            season_frames.append(
                season_df
            )

        if index < (
            len(RECENT_FOOTBALL_DATA_SEASONS) - 1
        ):
            time.sleep(7)

    if not season_frames:
        raise RuntimeError(
            "No football-data.org UCL seasons were downloaded."
        )

    combined = pd.concat(
        season_frames,
        ignore_index=True,
    )

    combined["date"] = pd.to_datetime(
        combined["date"]
    )

    combined = combined.sort_values(
        by=[
            "date",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    duplicates = combined.duplicated(
        subset=[
            "date",
            "home_team",
            "away_team",
        ]
    ).sum()

    print("\n" + "=" * 65)
    print("FOOTBALL-DATA.ORG SUMMARY")
    print("=" * 65)

    print(
        f"\nFinished matches: "
        f"{len(combined):,}"
    )

    print(
        f"Duplicates: "
        f"{duplicates:,}"
    )

    print(
        f"Date range: "
        f"{combined['date'].min().date()} "
        f"to "
        f"{combined['date'].max().date()}"
    )

    print("\nMatches by season:")

    print(
        combined["season"]
        .value_counts()
        .sort_index()
    )

    output_dir = (
        INTERIM_DATA_DIR
        / "matches"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "ucl_recent_matches.csv"
    )

    combined.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nSaved to:\n{output_file}"
    )

    return combined


# ============================================================
# API-FOOTBALL
# ============================================================


def get_api_football_headers() -> dict:
    """
    Return API-Football authentication headers.
    """

    if not API_FOOTBALL_KEY:
        raise RuntimeError(
            "API_FOOTBALL_KEY is missing from .env"
        )

    return {
        "x-apisports-key": API_FOOTBALL_KEY,
    }


def fetch_ucl_2022_from_api_football() -> pd.DataFrame:
    """
    Download the 2022/23 UEFA Champions League season
    from API-Football.

    API-Football league ID:
    UEFA Champions League = 2

    API-Football season:
    2022 = 2022/23

    Qualification and preliminary rounds are excluded.

    This leaves only:
    - Group stage
    - Round of 16
    - Quarter-finals
    - Semi-finals
    - Final
    """

    print("=" * 65)
    print("UCL PREDICTOR - API-FOOTBALL 2022/23")
    print("=" * 65)

    url = (
        f"{API_FOOTBALL_BASE_URL}/fixtures"
    )

    params = {
        "league": API_FOOTBALL_UCL_ID,
        "season": 2022,
    }

    print(
        "\nRequesting UCL 2022/23 from API-Football..."
    )

    try:
        response = requests.get(
            url,
            headers=get_api_football_headers(),
            params=params,
            timeout=30,
        )

    except requests.RequestException as error:
        raise RuntimeError(
            f"API-Football request failed: {error}"
        ) from error

    print(
        f"HTTP status: {response.status_code}"
    )

    response.raise_for_status()

    data = response.json()

    errors = data.get(
        "errors"
    )

    if errors:
        raise RuntimeError(
            f"API-Football returned errors: {errors}"
        )

    fixtures = data.get(
        "response",
        [],
    )

    print(
        f"Fixtures returned: "
        f"{len(fixtures):,}"
    )

    # --------------------------------------------------------
    # SAVE RAW RESPONSE
    # --------------------------------------------------------

    raw_dir = (
        RAW_DATA_DIR
        / "matches"
        / "api_football"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_file = (
        raw_dir
        / "ucl_2022_2023.json"
    )

    with open(
        raw_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Raw JSON saved: {raw_file}"
    )

    # --------------------------------------------------------
    # CONVERT FIXTURES
    # --------------------------------------------------------

    rows = []

    completed_statuses = {
        "FT",
        "AET",
        "PEN",
    }

    excluded_stage_count = 0

    included_stage_count = 0

    for item in fixtures:

        fixture = item.get(
            "fixture",
            {},
        )

        fixture_status = fixture.get(
            "status",
            {},
        )

        status_short = fixture_status.get(
            "short"
        )

        if status_short not in completed_statuses:
            continue

        league = item.get(
            "league",
            {},
        )

        stage = league.get(
            "round"
        )

        # ----------------------------------------------------
        # FILTER COMPETITION SCOPE
        # ----------------------------------------------------

        if not is_ucl_competition_proper(
            stage
        ):
            excluded_stage_count += 1
            continue

        included_stage_count += 1

        teams = item.get(
            "teams",
            {},
        )

        home_team = (
            teams.get(
                "home",
                {},
            ).get(
                "name"
            )
        )

        away_team = (
            teams.get(
                "away",
                {},
            ).get(
                "name"
            )
        )

        if not home_team or not away_team:
            continue

        # ----------------------------------------------------
        # SCORE HANDLING
        # ----------------------------------------------------

        score = item.get(
            "score",
            {},
        )

        full_time = score.get(
            "fulltime",
            {},
        )

        extra_time = score.get(
            "extratime",
            {},
        )

        # If a knockout match goes to extra time, use the
        # score after extra time but before penalties.
        if (
            status_short in {"AET", "PEN"}
            and extra_time.get("home") is not None
            and extra_time.get("away") is not None
        ):
            home_goals = extra_time.get(
                "home"
            )

            away_goals = extra_time.get(
                "away"
            )

        else:
            home_goals = full_time.get(
                "home"
            )

            away_goals = full_time.get(
                "away"
            )

        if (
            home_goals is None
            or away_goals is None
        ):
            continue

        home_goals = int(
            home_goals
        )

        away_goals = int(
            away_goals
        )

        raw_date = fixture.get(
            "date"
        )

        if not raw_date:
            continue

        date = pd.to_datetime(
            raw_date,
            utc=True,
        )

        rows.append(
            {
                "api_match_id": fixture.get("id"),
                "date": date.date().isoformat(),
                "day": date.day_name()[:3],
                "season": 2022,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": determine_result(
                    home_goals,
                    away_goals,
                ),
                "scoreline": (
                    f"{home_goals}-{away_goals}"
                ),
                "competition": "UEFA Champions League",
                "source": "api_football",
                "stage": stage,
                "status": status_short,
            }
        )

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        raise RuntimeError(
            "No completed Champions League proper matches "
            "were returned for 2022/23."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        by=[
            "date",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    duplicate_count = df.duplicated(
        subset=[
            "date",
            "home_team",
            "away_team",
        ]
    ).sum()

    # --------------------------------------------------------
    # DATA QUALITY CHECKS
    # --------------------------------------------------------

    if duplicate_count > 0:
        print(
            "\nWARNING: Duplicate matches were detected."
        )

    missing_scores = df[
        [
            "home_goals",
            "away_goals",
        ]
    ].isna().sum().sum()

    if missing_scores > 0:
        print(
            f"\nWARNING: Missing score values: "
            f"{missing_scores}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("2022/23 SUMMARY")
    print("=" * 65)

    print(
        f"\nFixtures returned by API: "
        f"{len(fixtures):,}"
    )

    print(
        f"Qualification/preliminary matches excluded: "
        f"{excluded_stage_count:,}"
    )

    print(
        f"Competition-proper fixtures identified: "
        f"{included_stage_count:,}"
    )

    print(
        f"Champions League proper matches kept: "
        f"{len(df):,}"
    )

    print(
        f"Duplicate matches: "
        f"{duplicate_count:,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    unique_teams = set(
        df["home_team"]
    ).union(
        set(df["away_team"])
    )

    print(
        f"Teams: "
        f"{len(unique_teams):,}"
    )

    print("\nMatches by stage:")

    print(
        df["stage"]
        .value_counts()
        .sort_index()
    )

    print("\nResult distribution:")

    print(
        df["result"]
        .value_counts()
        .reindex(
            ["H", "D", "A"]
        )
    )

    # --------------------------------------------------------
    # SAVE INTERIM CSV
    # --------------------------------------------------------

    interim_dir = (
        INTERIM_DATA_DIR
        / "matches"
    )

    interim_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        interim_dir
        / "ucl_2022_2023.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nSaved to:\n{output_file}"
    )

    print("\nFirst 10 rows:")

    print(
        df.head(
            10
        ).to_string(
            index=False
        )
    )

    print("\nLast 10 rows:")

    print(
        df.tail(
            10
        ).to_string(
            index=False
        )
    )

    return df


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    fetch_ucl_2022_from_api_football()