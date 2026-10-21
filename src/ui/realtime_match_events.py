
"""Authenticated football match-event adapter for UCL Predictor V2.

Fetches provider play-by-play records without interpreting their meaning.
The public Match Centre must not display these records until event coverage
and public redistribution permissions have been verified.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.ui.realtime_match_scores import (
    EVENTS_ENDPOINT,
    _canonical_team,
    _utc,
)


MAX_RESPONSE_BYTES = 2_000_000


def fetch_match_plays(
    api_key: str,
    event_id: str | int,
    *,
    page: int = 1,
    limit: int = 25,
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    """Fetch one page of verified provider play-by-play data.

    This function does not cache responses, infer incident types,
    change match scores or modify pre-match predictions.
    """

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Match-event API credentials are not configured")

    event_id = str(event_id)

    if not re.fullmatch(r"[1-9][0-9]{0,17}", event_id):
        raise ValueError("Invalid provider event ID")

    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("Invalid page number")

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("Event page size must be between 1 and 50")

    query = urlencode({"page": page, "limit": limit})
    url = f"{EVENTS_ENDPOINT}/{event_id}/plays?{query}"

    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Accept": "application/json",
            "User-Agent": "UCL-Predictor-V2/1.0",
        },
        method="GET",
    )

    with urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)

    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Match-event response exceeded its size limit")

    payload = json.loads(raw)

    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ValueError("Match-event provider returned an unsuccessful response")

    records = payload.get("data")

    if not isinstance(records, list):
        raise ValueError("Match-event provider did not return an event list")

    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Match-event response contains invalid records")

    return records


def match_fixture_event_id(
    fixture: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    kickoff_tolerance_minutes: int = 45,
) -> str | None:
    """Resolve a selected fixture to exactly one available provider event ID.

    Match ordered team names and UTC kickoff. Do not guess when records are
    missing, ambiguous, unavailable, or malformed. Never infer a match score.
    """
    if not isinstance(fixture, Mapping):
        return None
    if (
        isinstance(kickoff_tolerance_minutes, bool)
        or not isinstance(kickoff_tolerance_minutes, int)
        or kickoff_tolerance_minutes < 0
    ):
        raise ValueError("Kickoff tolerance must be a non-negative integer")

    try:
        home_name = _canonical_team(fixture["home_team"])
        away_name = _canonical_team(fixture["away_team"])
        kickoff = _utc(fixture["kickoff_utc"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return None

    if not home_name or not away_name:
        return None

    matches: list[str] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue

        home = event.get("homeTeam")
        away = event.get("awayTeam")
        if not isinstance(home, Mapping) or not isinstance(away, Mapping):
            continue
        if _canonical_team(home.get("name")) != home_name:
            continue
        if _canonical_team(away.get("name")) != away_name:
            continue

        try:
            event_kickoff = _utc(event.get("date"))
        except (ValueError, TypeError, OverflowError):
            continue
        if abs((event_kickoff - kickoff).total_seconds()) > kickoff_tolerance_minutes * 60:
            continue

        competition = event.get("competition")
        available = event.get("playByPlayAvailable")
        if available is None and isinstance(competition, Mapping):
            available = competition.get("playByPlayAvailable")
        if available is not True:
            continue

        event_id = event.get("id") or event.get("eventId")
        if isinstance(event_id, bool):
            continue
        identifier = str(event_id)
        if not re.fullmatch(r"[1-9][0-9]{0,17}", identifier):
            continue
        matches.append(identifier)

    return matches[0] if len(matches) == 1 else None
