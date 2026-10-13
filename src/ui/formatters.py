from __future__ import annotations

import html
from zoneinfo import ZoneInfo

import pandas as pd


LOCAL_TZ = ZoneInfo(
    "Africa/Johannesburg"
)


def escape(
    value,
) -> str:
    return html.escape(
        str(value)
    )


def percentage(
    value,
) -> str:
    return (
        f"{float(value) * 100:.2f}%"
    )


def display_confidence_label(
    label: str,
) -> str:
    return (
        str(label)
        .replace(
            "_",
            " "
        )
    )


def display_profile(
    profile: str,
) -> str:
    return (
        str(profile)
        .replace(
            "_",
            " "
        )
    )


def match_identifier(
    row: pd.Series,
) -> str:
    kickoff = pd.Timestamp(
        row[
            "kickoff_utc"
        ]
    )

    return (
        f"{kickoff.strftime('%Y%m%d%H%M')}-"
        f"{row['home_team'][:3].upper()}-"
        f"{row['away_team'][:3].upper()}"
    )


def fixture_label(
    row,
) -> str:
    kickoff_local = (
        pd.Timestamp(
            row.kickoff_utc
        )
        .tz_convert(
            LOCAL_TZ
        )
    )

    return (
        f"{kickoff_local.strftime('%H:%M')}"
        f"  •  "
        f"{row.home_team} vs "
        f"{row.away_team}"
    )


def result_class_name(
    result: str,
) -> str:
    if result == "H":
        return "HOME"

    if result == "D":
        return "DRAW"

    return "AWAY"


def winning_cell_class(
    *,
    result: str,
    cell: str,
) -> str:
    if result == cell:
        return (
            "hda-cell winner"
        )

    return (
        "hda-cell"
    )


def confidence_explanation(
    row: pd.Series,
) -> str:
    profile = (
        display_profile(
            row[
                "model_decisiveness_profile"
            ]
        )
    )

    if profile == "CLEAR LEAN":
        text = (
            "The outcome distribution "
            "has a clear leader."
        )

    elif profile == "LEAN":
        text = (
            "The model has a lean, but "
            "alternatives remain material."
        )

    else:
        text = (
            "The outcome distribution "
            "is tight."
        )

    return (
        f"{text} Evidence confidence "
        "reflects data quality, not result "
        "probability."
    )


def health_css_class(
    severity: str,
) -> str:
    severity = (
        str(severity)
        .upper()
    )

    if severity == "GREEN":
        return "green"

    if severity == "AMBER":
        return "amber"

    return "red"


def format_health_finish_time(
    health: dict,
) -> str:
    value = health.get(
        "finished_at_utc"
    )

    if not value:
        return "UNKNOWN"

    try:
        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:
            timestamp = (
                timestamp
                .tz_localize(
                    "UTC"
                )
            )

        else:
            timestamp = (
                timestamp
                .tz_convert(
                    "UTC"
                )
            )

        timestamp = (
            timestamp
            .tz_convert(
                LOCAL_TZ
            )
        )

        return timestamp.strftime(
            "%H:%M SAST"
        )

    except Exception:
        return "UNKNOWN"


def format_next_kickoff_local(
    health: dict,
) -> str:
    value = health.get(
        "next_kickoff_utc"
    )

    if not value:
        return "NONE"

    try:
        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:
            timestamp = (
                timestamp
                .tz_localize(
                    "UTC"
                )
            )

        else:
            timestamp = (
                timestamp
                .tz_convert(
                    "UTC"
                )
            )

        timestamp = (
            timestamp
            .tz_convert(
                LOCAL_TZ
            )
        )

        return timestamp.strftime(
            "%H:%M SAST"
        )

    except Exception:
        return "UNKNOWN"