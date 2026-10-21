"""Verified Realtime Sports API score adapter for UCL Predictor's Match Centre.

Only the authenticated official REST API supplies native scores. The free
HTML scoreboard widget is never scraped. No model or forecast values change.
This module deliberately uses only the Python standard library.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen


EVENTS_ENDPOINT = (
    "https://www.realtimesportsapi.com/api/v1/"
    "sports/soccer/leagues/uefa.champions/events"
)


@dataclass(frozen=True)
class ProviderScore:
    home_goals: int
    away_goals: int
    status: str
    minute: int | None = None


def fetch_champions_league_events(api_key: str, *, timeout: float = 8.0) -> list[dict[str, Any]]:
    """Read one league-wide scoreboard response with a server-only API key.

    Errors must be caught at the UI boundary; no request credentials are logged.
    One successful call consumes one quota unit, not one call per fixture.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Realtime Sports API credentials have not been configured")
    request = Request(
        EVENTS_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key.strip()}",
                 "Accept": "application/json", "User-Agent": "UCL-Predictor-V2/1.0"},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        # A deliberately bounded response prevents unexpectedly huge payloads.
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Score response exceeded the configured size limit")
    body = json.loads(raw)
    if not isinstance(body, dict) or body.get("success") is not True:
        raise ValueError("Score provider returned an unsuccessful response")
    rows = body.get("data")
    if not isinstance(rows, list):
        raise ValueError("Score provider did not return an event list")
    return [row for row in rows if isinstance(row, dict)]


def _canonical_team(value: object) -> str:
    name = str(value or "").casefold().strip()
    # NFKD alone does not replace ø and certain other common football letters.
    for source, target in (("ø", "o"), ("ł", "l"), ("æ", "ae"), ("ß", "ss")):
        name = name.replace(source, target)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(char for char in name if not unicodedata.combining(char))
    name = re.sub(r"[^a-z0-9]+", " ", name).strip()
    # Explicit aliases only. No fuzzy guesses between unrelated clubs.
    aliases = {
        "as roma": "roma", "a s roma": "roma", "fc porto": "porto",
        "sk slovan bratislava": "slovan bratislava",
        "s k slovan bratislava": "slovan bratislava",
        "bodo glimt": "bodo glimt", "bodo glimt fk": "bodo glimt",
        "fenerbahce sk": "fenerbahce", "fenerbahce": "fenerbahce",
        "paris sg": "paris saint germain", "psg": "paris saint germain",
        "vfb stuttgart": "vfb stuttgart", "stuttgart": "vfb stuttgart",
        "atletico de madrid": "atletico madrid",
        # Verified club-name variants used by official scoreboards.
        # Keep aliases explicit: no fuzzy team guesses or reversed fixtures.
        "como": "como 1907",
        "lille osc": "lille",
        "galatasaray istanbul": "galatasaray",
        "aston villa fc": "aston villa",
        "club brugge kv": "club brugge",
        "rc lens": "lens",
        "racing club de lens": "lens",
        "inter": "inter milan",
        "internazionale": "inter milan",
        "fc internazionale milano": "inter milan",
        "fc shakhtar donetsk": "shakhtar donetsk",
        "bayern munchen": "bayern munich",
        "fc bayern munchen": "bayern munich",
        "fc bayern munich": "bayern munich",
        "real betis balompie": "real betis",
        "feyenoord rotterdam": "feyenoord",
        "sporting lisbon": "sporting cp",
        "lask": "lask linz",
        "manchester united fc": "manchester united",
        "liverpool fc": "liverpool",
        "villarreal cf": "villarreal",
        "fc barcelona": "barcelona",
        "ssc napoli": "napoli",
        "sabah masazir": "sabah fk",
        "psv eindhoven": "psv",
    }
    return aliases.get(name, name)


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif hasattr(value, "to_pydatetime"):
        result = value.to_pydatetime()
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("Missing event kickoff time")
    if result.tzinfo is None:
        raise ValueError("Event kickoff time must include a timezone")
    return result.astimezone(timezone.utc)


def _valid_score(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= 99:
        return value
    if isinstance(value, str) and re.fullmatch(r"\d{1,2}", value.strip()):
        return int(value)
    return None


def _provider_score(event: Mapping[str, Any]) -> ProviderScore | None:
    status = event.get("status")
    home = event.get("homeTeam")
    away = event.get("awayTeam")
    if not all(isinstance(part, Mapping) for part in (status, home, away)):
        return None
    state = str(status.get("state", "")).lower()
    if state == "in":
        detail = str(status.get("detail") or "")
        half = bool(status.get("halftime")) or "half time" in detail.lower() or "halftime" in detail.lower()
        phase = "HALF_TIME" if half else "LIVE"
        minute = None
        if phase == "LIVE":
            # Soccer clocks may be '54:22', '54\u2032' or '54\''.
            clock = str(status.get("clock") or "")
            match = re.match(r"^\s*(\d{1,3})(?:[:\u2032']|$)", clock)
            if match:
                minute_candidate = int(match.group(1))
                minute = minute_candidate if 0 <= minute_candidate <= 130 else None
    elif state == "post" and status.get("completed") is True:
        phase, minute = "FULL_TIME", None
    else:
        return None  # Never show scheduled 0-0, postponements or unknown state.

    h = _valid_score(home.get("score"))
    a = _valid_score(away.get("score"))
    if h is None or a is None:
        return None
    return ProviderScore(home_goals=h, away_goals=a, status=phase, minute=minute)


def match_fixture_score(
    fixture: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    kickoff_tolerance_minutes: int = 45,
) -> ProviderScore | None:
    """Fail-closed match on ordered home/away club names AND UTC kickoff.

    Ambiguous multiple matches deliberately return no score rather than guess.
    """
    try:
        target_home = _canonical_team(fixture["home_team"])
        target_away = _canonical_team(fixture["away_team"])
        kickoff = _utc(fixture["kickoff_utc"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return None
    if not target_home or not target_away:
        return None
    matches: list[ProviderScore] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        home, away = event.get("homeTeam"), event.get("awayTeam")
        if not isinstance(home, Mapping) or not isinstance(away, Mapping):
            continue
        if _canonical_team(home.get("name")) != target_home:
            continue
        if _canonical_team(away.get("name")) != target_away:
            continue
        try:
            event_time = _utc(event.get("date"))
        except (ValueError, TypeError, OverflowError):
            continue
        if abs((event_time - kickoff).total_seconds()) > kickoff_tolerance_minutes * 60:
            continue
        score = _provider_score(event)
        if score is not None:
            matches.append(score)
    return matches[0] if len(matches) == 1 else None
