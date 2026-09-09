from __future__ import annotations

import base64
import html
import json
import mimetypes
from functools import lru_cache
from pathlib import Path

from src.config import BASE_DIR


# ============================================================
# PATHS
# ============================================================

CRESTS_DIR = (
    BASE_DIR
    / "assets"
    / "crests"
)

MANIFEST_FILE = (
    CRESTS_DIR
    / "manifest.json"
)


# ============================================================
# MANIFEST
# ============================================================


@lru_cache(
    maxsize=1
)
def load_crest_manifest() -> dict[str, str]:
    """
    Load the local club-crest manifest.

    Returns:
        {
            "Barcelona": "barcelona.png",
            "Liverpool": "liverpool.png",
            ...
        }

    Missing or invalid manifests return an empty dictionary
    so the Streamlit UI can safely fall back to initials.
    """

    if not MANIFEST_FILE.exists():

        return {}

    try:

        with MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(
                file
            )

    except Exception:

        return {}

    mapping: dict[str, str] = {}

    records = payload.get(
        "teams",
        [],
    )

    if not isinstance(
        records,
        list,
    ):

        return {}

    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            continue

        team = record.get(
            "team"
        )

        filename = record.get(
            "filename"
        )

        if not team:
            continue

        if not filename:
            continue

        mapping[
            str(
                team
            ).strip()
        ] = (
            str(
                filename
            ).strip()
        )

    return mapping


# ============================================================
# CREST LOOKUP
# ============================================================


def get_crest_path(
    team: str,
) -> Path | None:
    """
    Return a local crest path for a canonical team name.

    No fuzzy matching is used.
    Team identity must exactly match the manifest.
    """

    team = str(
        team
    ).strip()

    mapping = (
        load_crest_manifest()
    )

    filename = mapping.get(
        team
    )

    if not filename:

        return None

    path = (
        CRESTS_DIR
        / filename
    )

    if not path.exists():

        return None

    if not path.is_file():

        return None

    return path


# ============================================================
# MIME TYPES
# ============================================================


def detect_mime_type(
    path: Path,
) -> str:

    guessed, _ = (
        mimetypes.guess_type(
            str(
                path
            )
        )
    )

    if guessed:

        return guessed

    suffix = (
        path.suffix
        .lower()
    )

    manual_mapping = {
        ".svg": (
            "image/svg+xml"
        ),
        ".png": (
            "image/png"
        ),
        ".jpg": (
            "image/jpeg"
        ),
        ".jpeg": (
            "image/jpeg"
        ),
        ".webp": (
            "image/webp"
        ),
        ".gif": (
            "image/gif"
        ),
    }

    return manual_mapping.get(
        suffix,
        "application/octet-stream",
    )


# ============================================================
# DATA URI
# ============================================================


@lru_cache(
    maxsize=128
)
def file_to_data_uri(
    path_string: str,
) -> str:
    """
    Convert a local image to a browser-safe base64 data URI.

    This keeps the Streamlit receipt independent from remote
    image URLs and makes the crests deployment-safe.
    """

    path = Path(
        path_string
    )

    if not path.exists():

        return ""

    if not path.is_file():

        return ""

    try:

        raw = (
            path.read_bytes()
        )

    except Exception:

        return ""

    if not raw:

        return ""

    mime_type = (
        detect_mime_type(
            path
        )
    )

    encoded = (
        base64.b64encode(
            raw
        )
        .decode(
            "ascii"
        )
    )

    return (
        f"data:{mime_type};base64,"
        f"{encoded}"
    )


def get_crest_data_uri(
    team: str,
) -> str | None:
    """
    Get a base64 data URI for a club crest.

    Returns None when a crest is unavailable.
    """

    path = get_crest_path(
        team
    )

    if path is None:

        return None

    uri = file_to_data_uri(
        str(
            path
        )
    )

    if not uri:

        return None

    return uri


# ============================================================
# FALLBACK INITIALS
# ============================================================


def team_initials(
    team: str,
) -> str:
    """
    Generate a compact deterministic fallback label.

    Examples:
        Barcelona -> BAR
        Real Madrid -> RM
        Paris Saint-Germain -> PSG
    """

    value = (
        str(
            team
        )
        .replace(
            "-",
            " "
        )
        .strip()
    )

    words = [
        word
        for word in value.split()
        if word
    ]

    if not words:

        return "?"

    if len(
        words
    ) == 1:

        return (
            words[
                0
            ][:3]
            .upper()
        )

    initials = "".join(
        word[
            0
        ]
        for word in words[
            :3
        ]
    )

    return (
        initials.upper()
    )


# ============================================================
# HTML
# ============================================================


def crest_html(
    team: str,
    *,
    side: str = "",
) -> str:
    """
    Return receipt-ready club crest HTML.

    This is the function imported by app.py.

    If a local crest is unavailable, a thermal-style initials
    badge is returned instead. The app therefore remains usable
    even with incomplete crest coverage.
    """

    team = str(
        team
    ).strip()

    safe_team = (
        html.escape(
            team,
            quote=True,
        )
    )

    safe_side = (
        html.escape(
            str(
                side
            ),
            quote=True,
        )
    )

    uri = (
        get_crest_data_uri(
            team
        )
    )

    if uri:

        safe_uri = (
            html.escape(
                uri,
                quote=True,
            )
        )

        return (
            f'<div class="club-crest-wrap {safe_side}">'
            f'<img '
            f'class="club-crest" '
            f'src="{safe_uri}" '
            f'alt="{safe_team} crest">'
            f'</div>'
        )

    initials = (
        team_initials(
            team
        )
    )

    safe_initials = (
        html.escape(
            initials
        )
    )

    return (
        f'<div class="club-crest-wrap {safe_side}">'
        f'<div class="club-crest-fallback">'
        f'{safe_initials}'
        f'</div>'
        f'</div>'
    )


# ============================================================
# DEBUG / AUDIT
# ============================================================


def crest_available(
    team: str,
) -> bool:
    """
    Convenience helper for audits/tests.
    """

    return (
        get_crest_path(
            team
        )
        is not None
    )


def manifest_coverage() -> dict:
    """
    Return a lightweight crest cache summary.
    """

    mapping = (
        load_crest_manifest()
    )

    available = 0
    missing = []

    for team in sorted(
        mapping
    ):

        if crest_available(
            team
        ):

            available += 1

        else:

            missing.append(
                team
            )

    return {
        "manifest_entries": (
            len(
                mapping
            )
        ),
        "available_files": (
            available
        ),
        "missing_files": (
            missing
        ),
    }


# ============================================================
# CLI CHECK
# ============================================================


def main() -> None:

    print(
        "=" * 78
    )

    print(
        "UCL PREDICTOR - CLUB CREST HELPER"
    )

    print(
        "=" * 78
    )

    summary = (
        manifest_coverage()
    )

    print(
        f"\nManifest:"
    )

    print(
        MANIFEST_FILE
    )

    print(
        f"\nManifest entries: "
        f"{summary['manifest_entries']}"
    )

    print(
        f"Available crest files: "
        f"{summary['available_files']}"
    )

    if summary[
        "missing_files"
    ]:

        print(
            "\nManifest entries with missing files:"
        )

        for team in summary[
            "missing_files"
        ]:

            print(
                f"  {team}"
            )

    print(
        "\ncrest_html import/function available ✓"
    )


if __name__ == "__main__":

    main()