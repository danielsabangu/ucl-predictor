"""Provider-verified match events and broadcast presentation for UCL Predictor.

This module does not modify or infer forecasts, scores, event locations or players.
The source API is opt-in because a public right to republish API play data must
be confirmed separately from the provider's free HTML widget permission.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen


PLAYS_BASE = (
    "https://www.realtimesportsapi.com/api/v1/"
    "sports/soccer/leagues/uefa.champions/events"
)
ALLOWED_KINDS = {
    "goal": "GOAL",
    "own goal": "OWN GOAL",
    "penalty goal": "PENALTY GOAL",
    "yellow card": "YELLOW CARD",
    "second yellow card": "SECOND YELLOW",
    "red card": "RED CARD",
    "substitution": "SUBSTITUTION",
}


@dataclass(frozen=True)
class MatchMoment:
    kind: str
    minute: str
    description: str
    team: str = ""


def fetch_verified_plays(api_key: str, event_id: str, *, timeout: float = 8.0) -> list[dict[str, Any]]:
    """Fetch one event's official plays (one quota call); no secrets in URLs."""
    if not api_key or not api_key.strip():
        raise ValueError("Score provider credentials are not configured")
    event_id = str(event_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", event_id):
        raise ValueError("Invalid provider event identifier")
    url = f"{PLAYS_BASE}/{quote(event_id, safe='')}/plays?limit=100"
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
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("Play feed exceeded size limit")
    body = json.loads(raw)
    if not isinstance(body, dict) or body.get("success") is not True:
        raise ValueError("The play feed response was unsuccessful")
    rows = body.get("data")
    if not isinstance(rows, list):
        raise ValueError("The play feed did not contain a play list")
    return [x for x in rows if isinstance(x, dict)]


def _label(raw: object) -> str:
    if isinstance(raw, Mapping):
        return str(raw.get("text") or raw.get("displayName") or raw.get("name") or "").strip()
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _minute(raw: Mapping[str, Any]) -> str:
    clock = raw.get("clock")
    if isinstance(clock, Mapping):
        value = clock.get("displayValue") or clock.get("value")
    else:
        value = clock
    if value is not None:
        match = re.match(r"^\s*(\d{1,3})(?:[:\u2032']|$)", str(value))
        if match and int(match.group(1)) <= 130:
            return f"{int(match.group(1))}'"
    minute = raw.get("minute")
    if isinstance(minute, int) and not isinstance(minute, bool) and 0 <= minute <= 130:
        return f"{minute}'"
    return ""


def parse_match_moments(raw_plays: Sequence[Mapping[str, Any]]) -> list[MatchMoment]:
    """Keep only recognisable provider-labelled events. Never synthesize plays."""
    moments = []
    seen = set()
    for play in raw_plays[:250]:
        if not isinstance(play, Mapping):
            continue
        raw_kind = _label(play.get("type")).casefold().replace("_", " ").replace("-", " ")
        raw_kind = " ".join(raw_kind.split())
        kind = ALLOWED_KINDS.get(raw_kind)
        if not kind:
            # A scoring flag alone is not enough to claim a goal without a type.
            continue
        desc = str(play.get("text") or play.get("description") or "").strip()[:220]
        if not desc:
            # Provider has identified the event, but hasn't provided a narrative.
            desc = kind.title()
        team = _label(play.get("team"))[:70]
        minute = _minute(play)
        identity = str(play.get("id") or "").strip()
        unique = ("id", identity) if identity else ("body", kind, minute, desc, team)
        if unique in seen:
            continue
        seen.add(unique)
        moments.append(MatchMoment(kind, minute, desc, team))
    return moments


def event_timeline_html(moments: Sequence[MatchMoment], *, state: str) -> str:
    """Responsive verified timeline. Empty states are honest and informative."""
    messages = {
        "disabled": "Detailed live events are not enabled for public display yet.",
        "before_kickoff": "The event timeline will be available only when the provider publishes verified plays.",
        "unavailable": "No verified match events are available from the provider for this fixture.",
        "request_failed": "The event feed could not be loaded. Scores and pre-match forecasts remain separate.",
        "no_data": "The provider has no verified goals, cards or substitutions to show yet.",
    }
    style = """<style>
    .le-timeline{background:#0b1e34;border:1px solid #2d4665;border-radius:9px;
      padding:16px 18px;margin:12px 0;color:#e8f3ff;font:13px Inter,system-ui,sans-serif}
    .le-kicker{color:#7eb9f2;font-size:10px;font-weight:800;letter-spacing:.12em}
    .le-head{display:flex;align-items:center;justify-content:space-between;gap:9px;margin:7px 0 13px}
    .le-head strong{font-size:18px}.le-caption{color:#92abc7;font-size:11px;line-height:1.55}
    .le-moment{display:grid;grid-template-columns:48px 1fr;gap:13px;padding:12px 0;
      border-top:1px solid #243b55;align-items:start}
    .le-time{font-weight:850;color:#b9d9fb;font-variant-numeric:tabular-nums}
    .le-kind{display:inline-block;color:#b4d8fb;font-size:10px;font-weight:900;
      letter-spacing:.07em;margin-bottom:5px}
    .le-goal .le-kind{color:#80ebbf}.le-card .le-kind{color:#ffd78a}
    .le-moment p{margin:0;font-size:12px;line-height:1.5;color:#d8e8fa;overflow-wrap:anywhere}
    .le-team{color:#84accf;font-size:11px;margin-top:5px}
    .le-pulse{display:inline-block;width:7px;height:7px;border-radius:50%;
      background:#59d8aa;box-shadow:0 0 0 0 rgba(89,216,170,.45);
      animation:le-pulse 2.4s infinite}
    @keyframes le-pulse{50%{box-shadow:0 0 0 7px rgba(89,216,170,0)}}
    @media(prefers-reduced-motion:reduce){.le-pulse{animation:none}}
    </style>"""
    count = len(moments)
    title = f"{count} recorded moment{'s' if count != 1 else ''}" if count else "Event timeline"
    rows = []
    for moment in reversed(list(moments[-18:])):
        css = "le-goal" if "GOAL" in moment.kind else "le-card" if "CARD" in moment.kind or "YELLOW" in moment.kind else ""
        team = f'<div class="le-team">{escape(moment.team)}</div>' if moment.team else ""
        rows.append(
            f'<article class="le-moment {css}">'
            f'<div class="le-time">{escape(moment.minute or "—")}</div>'
            f'<div><div class="le-kind">{escape(moment.kind)}</div>'
            f'<p>{escape(moment.description)}</p>{team}</div></article>'
        )
    if not rows:
        rows.append(f'<p class="le-caption">{escape(messages.get(state, messages["unavailable"]))}</p>')
    return (
        style + '<section class="le-timeline">'
        '<div class="le-kicker">VERIFIED MATCH MOMENTS</div>'
        f'<div class="le-head"><strong>{escape(title)}</strong></div>'
        + "".join(rows) +
        '<p class="le-caption">Source: Realtime Sports API. Events can be delayed or corrected. '
        'No pitch positions or player movements are inferred.</p></section>'
    )


def pitch_scene_html(pitch_svg: str, *, status: str, minute: int | None = None,
                     moments: Sequence[MatchMoment] = ()) -> str:
    """Animated broadcast chrome around a STATIC pitch; no false event positions."""
    is_live = status == "LIVE"
    title = "MATCH LIVE" if is_live else "HALF-TIME" if status == "HALF_TIME" else (
        "FULL-TIME" if status == "FULL_TIME" else "MATCH CENTRE"
    )
    clock = f" · {minute}'" if is_live and isinstance(minute, int) else ""
    last_goal = next((m for m in reversed(moments) if "GOAL" in m.kind), None)
    note = (
        "Latest recorded goal: " + (f"{last_goal.minute} " if last_goal.minute else "")
        + last_goal.description if last_goal else "Verified event positions are not available"
    )
    pulse = '<span class="live-dot" aria-hidden="true"></span>' if is_live else ""
    return (
        """<html><head><style>
        html,body{margin:0;background:transparent;color:#e9f5ff;font:12px Inter,system-ui,sans-serif}
        .le-pitch{position:relative;width:100%;max-width:850px;margin:auto}
        .le-pitch svg{display:block;width:100%;height:auto;border-radius:8px}
        .le-status{display:flex;gap:8px;align-items:center;justify-content:center;
          font-size:11px;font-weight:800;letter-spacing:.08em;margin:2px 0 9px;color:#b9d7ee}
        .live-dot{width:9px;height:9px;background:#59deae;border-radius:50%;
          animation:live-beat 1.8s infinite}
        .le-note{text-align:center;font-size:10px;color:#9ebed5;line-height:1.5;margin:10px 0 0}
        @keyframes live-beat{50%{opacity:.35;transform:scale(.76)}}
        @media(prefers-reduced-motion:reduce){.live-dot{animation:none}}
        </style></head><body><div class="le-pitch"><div class="le-status">"""
        + pulse + escape(title + clock) + "</div>" + pitch_svg +
        '<div class="le-note">' + escape(note) + '</div></div></body></html>'
    )
