from __future__ import annotations

import json
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from src.config import (
    BASE_DIR,
    FOOTBALL_DATA_API_KEY,
    INTERIM_DATA_DIR,
)


# ============================================================
# PATHS
# ============================================================

TEAM_SNAPSHOT_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "football_data_ucl_teams.csv"
)

ASSETS_DIR = (
    BASE_DIR
    / "assets"
)

CRESTS_DIR = (
    ASSETS_DIR
    / "crests"
)

MANIFEST_FILE = (
    CRESTS_DIR
    / "manifest.json"
)


# ============================================================
# FOOTBALL-DATA
# ============================================================

FOOTBALL_DATA_BASE_URL = (
    "https://api.football-data.org/v4"
)

FOOTBALL_DATA_CREST_CDN = (
    "https://crests.football-data.org"
)


# ============================================================
# SETTINGS
# ============================================================

REQUEST_TIMEOUT_SECONDS = 25

# CDN requests can be made quickly.
CDN_REQUEST_DELAY_SECONDS = 0.05

# API fallback is intentionally slower / rate-safe.
API_REQUEST_DELAY_SECONDS = 7.0

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "UCLPredictor/1.0"
)

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".svg",
}


# ============================================================
# TEXT
# ============================================================


def slugify(
    value: str,
) -> str:

    value = (
        str(value)
        .strip()
        .lower()
    )

    replacements = {
        "á": "a",
        "à": "a",
        "ä": "a",
        "â": "a",
        "ã": "a",
        "å": "a",
        "é": "e",
        "è": "e",
        "ë": "e",
        "ê": "e",
        "í": "i",
        "ì": "i",
        "ï": "i",
        "î": "i",
        "ó": "o",
        "ò": "o",
        "ö": "o",
        "ô": "o",
        "õ": "o",
        "ø": "o",
        "ú": "u",
        "ù": "u",
        "ü": "u",
        "û": "u",
        "ç": "c",
        "ñ": "n",
        "š": "s",
        "ž": "z",
        "ć": "c",
        "č": "c",
        "đ": "d",
        "ý": "y",
        "æ": "ae",
        "œ": "oe",
    }

    for source, target in replacements.items():

        value = value.replace(
            source,
            target,
        )

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    return value.strip(
        "-"
    )


# ============================================================
# COLUMN HELPERS
# ============================================================


def detect_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    lower_map = {
        str(column).lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:

        candidate_lower = (
            candidate.lower()
        )

        if candidate_lower in lower_map:

            return lower_map[
                candidate_lower
            ]

    return None


def detect_team_column(
    dataframe: pd.DataFrame,
) -> str:

    column = detect_column(
        dataframe,
        [
            "canonical_team",
            "team",
            "team_name",
            "canonical_name",
            "name",
        ],
    )

    if column is None:

        raise RuntimeError(
            "Could not identify the canonical team "
            "column.\n"
            f"Columns found: "
            f"{list(dataframe.columns)}"
        )

    return column


def detect_team_id_column(
    dataframe: pd.DataFrame,
) -> str:

    column = detect_column(
        dataframe,
        [
            "football_data_team_id",
            "football_data_id",
            "team_id",
            "id",
        ],
    )

    if column is None:

        raise RuntimeError(
            "Could not identify the football-data "
            "team ID column.\n"
            f"Columns found: "
            f"{list(dataframe.columns)}"
        )

    return column


# ============================================================
# HTTP
# ============================================================


def request_bytes(
    url: str,
    *,
    api_key: str | None = None,
) -> tuple[
    bytes,
    str | None,
]:

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "image/avif,"
            "image/webp,"
            "image/apng,"
            "image/svg+xml,"
            "image/*,"
            "*/*"
        ),
    }

    if api_key:

        headers[
            "X-Auth-Token"
        ] = api_key

    request = Request(
        url,
        headers=headers,
    )

    with urlopen(
        request,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:

        content = (
            response.read()
        )

        content_type = (
            response.headers.get(
                "Content-Type"
            )
        )

    return (
        content,
        content_type,
    )


def request_json(
    url: str,
    *,
    api_key: str,
) -> dict:

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "X-Auth-Token": api_key,
        },
    )

    with urlopen(
        request,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:

        raw = (
            response.read()
        )

    return json.loads(
        raw.decode(
            "utf-8"
        )
    )


# ============================================================
# FILE TYPE
# ============================================================


def guess_extension(
    *,
    url: str,
    content_type: str | None,
) -> str:

    parsed = urlparse(
        url
    )

    suffix = (
        Path(
            parsed.path
        )
        .suffix
        .lower()
    )

    if suffix in SUPPORTED_EXTENSIONS:

        return suffix

    if content_type:

        normalized = (
            content_type
            .split(";")[0]
            .strip()
            .lower()
        )

        known = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/svg+xml": ".svg",
        }

        if normalized in known:

            return known[
                normalized
            ]

        guessed = (
            mimetypes.guess_extension(
                normalized
            )
        )

        if guessed:

            return guessed

    return ".img"


def looks_like_image(
    *,
    content: bytes,
    content_type: str | None,
) -> bool:

    if not content:

        return False

    if content_type:

        normalized = (
            content_type
            .lower()
            .strip()
        )

        if normalized.startswith(
            "image/"
        ):

            return True

    prefix = (
        content[:200]
        .lstrip()
        .lower()
    )

    # SVG
    if (
        prefix.startswith(
            b"<svg"
        )
        or
        b"<svg" in prefix
    ):

        return True

    # PNG
    if content.startswith(
        b"\x89PNG"
    ):

        return True

    # JPEG
    if content.startswith(
        b"\xff\xd8\xff"
    ):

        return True

    # WEBP
    if (
        len(content) >= 12
        and
        content[:4] == b"RIFF"
        and
        content[8:12] == b"WEBP"
    ):

        return True

    return False


# ============================================================
# CACHE
# ============================================================


def find_existing_crest(
    team: str,
) -> Path | None:

    slug = slugify(
        team
    )

    for path in CRESTS_DIR.glob(
        f"{slug}.*"
    ):

        if (
            path.name
            ==
            "manifest.json"
        ):

            continue

        if path.is_file():

            return path

    return None


def save_crest(
    *,
    team: str,
    url: str,
    content: bytes,
    content_type: str | None,
) -> Path:

    extension = guess_extension(
        url=url,
        content_type=content_type,
    )

    destination = (
        CRESTS_DIR
        /
        f"{slugify(team)}{extension}"
    )

    destination.write_bytes(
        content
    )

    return destination


# ============================================================
# CREST SOURCES
# ============================================================


def cdn_candidates(
    team_id: int,
) -> list[str]:

    # football-data.org commonly hosts badge assets here.
    # Try PNG first because it is simplest for browser/PDF use.
    return [
        (
            f"{FOOTBALL_DATA_CREST_CDN}"
            f"/{team_id}.png"
        ),
        (
            f"{FOOTBALL_DATA_CREST_CDN}"
            f"/{team_id}.svg"
        ),
    ]


def try_static_cdn(
    *,
    team: str,
    team_id: int,
) -> dict | None:

    for url in cdn_candidates(
        team_id
    ):

        try:

            content, content_type = (
                request_bytes(
                    url
                )
            )

            if not looks_like_image(
                content=content,
                content_type=content_type,
            ):

                continue

            destination = save_crest(
                team=team,
                url=url,
                content=content,
                content_type=content_type,
            )

            return {
                "source": (
                    "football_data_crest_cdn"
                ),
                "url": url,
                "filename": (
                    destination.name
                ),
                "content_type": (
                    content_type
                ),
                "size_bytes": (
                    len(content)
                ),
            }

        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
        ):

            continue

    return None


def get_api_crest_url(
    *,
    team_id: int,
) -> str | None:

    if not FOOTBALL_DATA_API_KEY:

        return None

    url = (
        f"{FOOTBALL_DATA_BASE_URL}"
        f"/teams/{team_id}"
    )

    payload = request_json(
        url,
        api_key=(
            FOOTBALL_DATA_API_KEY
        ),
    )

    crest = payload.get(
        "crest"
    )

    if not crest:

        return None

    return str(
        crest
    ).strip()


def try_api_fallback(
    *,
    team: str,
    team_id: int,
) -> dict | None:

    if not FOOTBALL_DATA_API_KEY:

        return None

    crest_url = get_api_crest_url(
        team_id=team_id,
    )

    if not crest_url:

        return None

    content, content_type = (
        request_bytes(
            crest_url
        )
    )

    if not looks_like_image(
        content=content,
        content_type=content_type,
    ):

        return None

    destination = save_crest(
        team=team,
        url=crest_url,
        content=content,
        content_type=content_type,
    )

    return {
        "source": (
            "football_data_team_endpoint"
        ),
        "url": (
            crest_url
        ),
        "filename": (
            destination.name
        ),
        "content_type": (
            content_type
        ),
        "size_bytes": (
            len(content)
        ),
    }


# ============================================================
# MAIN
# ============================================================


def prepare_crests() -> None:

    print(
        "=" * 78
    )

    print(
        "UCL PREDICTOR - PREPARE CLUB CRESTS"
    )

    print(
        "=" * 78
    )

    if not TEAM_SNAPSHOT_FILE.exists():

        raise FileNotFoundError(
            "Current football-data team snapshot "
            "does not exist:\n"
            f"{TEAM_SNAPSHOT_FILE}"
        )

    CRESTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    teams = pd.read_csv(
        TEAM_SNAPSHOT_FILE,
        low_memory=False,
    )

    team_column = (
        detect_team_column(
            teams
        )
    )

    team_id_column = (
        detect_team_id_column(
            teams
        )
    )

    print(
        f"\nTeam column: "
        f"{team_column}"
    )

    print(
        f"Team ID column: "
        f"{team_id_column}"
    )

    print(
        f"Rows loaded: "
        f"{len(teams)}"
    )

    teams = teams[
        [
            team_column,
            team_id_column,
        ]
    ].copy()

    teams = teams.dropna(
        subset=[
            team_column,
            team_id_column,
        ]
    )

    teams[
        team_column
    ] = (
        teams[
            team_column
        ]
        .astype(str)
        .str.strip()
    )

    teams[
        team_id_column
    ] = pd.to_numeric(
        teams[
            team_id_column
        ],
        errors="raise",
    ).astype(
        int
    )

    teams = (
        teams
        .drop_duplicates(
            subset=[
                team_column
            ],
            keep="first",
        )
        .sort_values(
            team_column
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Unique clubs: "
        f"{len(teams)}"
    )

    records = []

    downloaded_cdn = 0
    downloaded_api = 0
    reused = 0
    failed = 0

    for index, row in teams.iterrows():

        team = str(
            row[
                team_column
            ]
        ).strip()

        team_id = int(
            row[
                team_id_column
            ]
        )

        print(
            f"\n[{index + 1:02d}/{len(teams):02d}] "
            f"{team} "
            f"(football-data ID {team_id})"
        )

        # ----------------------------------------------------
        # ALREADY CACHED
        # ----------------------------------------------------

        existing = find_existing_crest(
            team
        )

        if existing is not None:

            print(
                f"    Cached: "
                f"{existing.name}"
            )

            reused += 1

            records.append(
                {
                    "team": team,
                    "football_data_team_id": (
                        team_id
                    ),
                    "filename": (
                        existing.name
                    ),
                    "status": (
                        "CACHED"
                    ),
                    "source": (
                        "local_cache"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # STATIC CDN FIRST
        # ----------------------------------------------------

        cdn_result = (
            try_static_cdn(
                team=team,
                team_id=team_id,
            )
        )

        if cdn_result is not None:

            print(
                f"    CDN: "
                f"{cdn_result['filename']}"
            )

            downloaded_cdn += 1

            records.append(
                {
                    "team": team,
                    "football_data_team_id": (
                        team_id
                    ),
                    "filename": (
                        cdn_result[
                            "filename"
                        ]
                    ),
                    "status": (
                        "DOWNLOADED"
                    ),
                    "source": (
                        cdn_result[
                            "source"
                        ]
                    ),
                    "crest_url": (
                        cdn_result[
                            "url"
                        ]
                    ),
                    "content_type": (
                        cdn_result[
                            "content_type"
                        ]
                    ),
                    "size_bytes": (
                        cdn_result[
                            "size_bytes"
                        ]
                    ),
                }
            )

            time.sleep(
                CDN_REQUEST_DELAY_SECONDS
            )

            continue

        # ----------------------------------------------------
        # API FALLBACK
        # ----------------------------------------------------

        print(
            "    Static CDN unavailable; "
            "trying Team API..."
        )

        try:

            api_result = (
                try_api_fallback(
                    team=team,
                    team_id=team_id,
                )
            )

            if api_result is None:

                raise RuntimeError(
                    "No usable crest URL returned."
                )

            print(
                f"    API: "
                f"{api_result['filename']}"
            )

            downloaded_api += 1

            records.append(
                {
                    "team": team,
                    "football_data_team_id": (
                        team_id
                    ),
                    "filename": (
                        api_result[
                            "filename"
                        ]
                    ),
                    "status": (
                        "DOWNLOADED"
                    ),
                    "source": (
                        api_result[
                            "source"
                        ]
                    ),
                    "crest_url": (
                        api_result[
                            "url"
                        ]
                    ),
                    "content_type": (
                        api_result[
                            "content_type"
                        ]
                    ),
                    "size_bytes": (
                        api_result[
                            "size_bytes"
                        ]
                    ),
                }
            )

        except Exception as error:

            print(
                f"    FAILED: "
                f"{error}"
            )

            failed += 1

            records.append(
                {
                    "team": team,
                    "football_data_team_id": (
                        team_id
                    ),
                    "filename": None,
                    "status": (
                        "FAILED"
                    ),
                    "source": None,
                    "error": (
                        str(error)
                    ),
                }
            )

        time.sleep(
            API_REQUEST_DELAY_SECONDS
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    usable = sum(
        1
        for record in records
        if record.get(
            "filename"
        )
    )

    manifest = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "source_team_snapshot": (
            str(
                TEAM_SNAPSHOT_FILE
            )
        ),
        "team_column": (
            team_column
        ),
        "team_id_column": (
            team_id_column
        ),
        "expected_teams": (
            len(teams)
        ),
        "usable_crests": (
            usable
        ),
        "downloaded_from_cdn": (
            downloaded_cdn
        ),
        "downloaded_from_api": (
            downloaded_api
        ),
        "cached": (
            reused
        ),
        "failed": (
            failed
        ),
        "teams": (
            records
        ),
    }

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        "CLUB CREST QUALITY CHECK"
    )

    print(
        "=" * 78
    )

    print(
        f"\nExpected clubs: "
        f"{len(teams)}"
    )

    print(
        f"Usable crests: "
        f"{usable}"
    )

    print(
        f"CDN downloads: "
        f"{downloaded_cdn}"
    )

    print(
        f"API downloads: "
        f"{downloaded_api}"
    )

    print(
        f"Cached: "
        f"{reused}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    coverage = (
        usable
        /
        len(teams)
        if len(teams)
        else 0.0
    )

    print(
        f"Crest coverage: "
        f"{coverage * 100:.2f}%"
    )

    if failed:

        print(
            "\nUnavailable crests:"
        )

        for record in records:

            if not record.get(
                "filename"
            ):

                print(
                    f"  {record['team']}"
                )

    print(
        "\nManifest:"
    )

    print(
        MANIFEST_FILE
    )

    print(
        "\nCrest directory:"
    )

    print(
        CRESTS_DIR
    )

    if usable == 0:

        raise RuntimeError(
            "No usable club crests were prepared."
        )

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        "CLUB CRESTS READY ✓"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    prepare_crests()