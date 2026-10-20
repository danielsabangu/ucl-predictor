"""Read an explicitly published future UCL fixture schedule; no network calls.

This is schedule data, not a prediction and not a live score provider.
The bundled CSV is a short-lived snapshot. Replace it with a refreshed,
verified schedule artifact when the next UCL matchday approaches.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOCAL_ZONE = 'Africa/Johannesburg'
REQUIRED = {'kickoff_utc', 'home_team', 'away_team'}
COLUMNS = ['kickoff_utc', 'home_team', 'away_team']


def load_future_fixtures(
    path: str | Path,
    *,
    after_matchday: pd.Timestamp | datetime,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return the closest later matchday from a verified schedule snapshot.

    Do not repeat matches already present on today's prediction board.
    Do not display past kickoffs as future fixtures. No prediction data is used.
    The function never falls back to synthesizing a schedule.
    """
    source = Path(path)
    empty = pd.DataFrame(columns=COLUMNS)
    if not source.is_file():
        return empty
    try:
        rows = pd.read_csv(source, dtype={'home_team': str, 'away_team': str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return empty
    if not REQUIRED.issubset(rows.columns):
        return empty

    rows = rows[COLUMNS].copy()
    rows['kickoff_utc'] = pd.to_datetime(rows['kickoff_utc'], utc=True, errors='coerce')
    rows = rows.dropna(subset=COLUMNS)
    rows = rows.loc[(rows['home_team'].str.strip() != '') &
                    (rows['away_team'].str.strip() != '')]
    if rows.empty:
        return empty

    boundary = pd.Timestamp(after_matchday)
    if boundary.tzinfo is None:
        raise ValueError('after_matchday must be timezone-aware')
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    if current.tzinfo is None:
        raise ValueError('now must be timezone-aware')

    matchday_date = boundary.tz_convert(LOCAL_ZONE).date()
    future_local_dates = rows['kickoff_utc'].dt.tz_convert(LOCAL_ZONE).dt.date
    rows = rows.loc[(future_local_dates > matchday_date) &
                    (rows['kickoff_utc'] > current)]
    if rows.empty:
        return empty

    nearest_day = rows['kickoff_utc'].dt.tz_convert(LOCAL_ZONE).dt.date.min()
    rows = rows.loc[rows['kickoff_utc'].dt.tz_convert(LOCAL_ZONE).dt.date == nearest_day]
    return (rows.sort_values(['kickoff_utc', 'home_team', 'away_team'])
                .drop_duplicates(subset=COLUMNS)
                .reset_index(drop=True))
