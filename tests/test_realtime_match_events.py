
"""Tests for the authenticated UCL match-event adapter."""

import json
from io import BytesIO
from unittest.mock import patch

import pytest

from src.ui.realtime_match_events import (
    fetch_match_plays,
    match_fixture_event_id,
)


def _response(data, success=True):
    payload = {"success": success, "data": data}
    return BytesIO(json.dumps(payload).encode("utf-8"))


def test_empty_pre_match_response():
    with patch(
        "src.ui.realtime_match_events.urlopen",
        return_value=_response([]),
    ) as mocked:
        result = fetch_match_plays("test-private-key", "401915395")

    assert result == []
    assert mocked.call_count == 1

    request = mocked.call_args.args[0]

    assert request.full_url.endswith(
        "/events/401915395/plays?page=1&limit=25"
    )
    assert request.get_header("Authorization") == "Bearer test-private-key"


def test_preserves_provider_event_data():
    plays = [
        {"id": "play-1", "text": "Provider event"},
        {"id": "play-2", "text": "Another event"},
    ]

    with patch(
        "src.ui.realtime_match_events.urlopen",
        return_value=_response(plays),
    ):
        result = fetch_match_plays("test-key", 401915395)

    assert result == plays


@pytest.mark.parametrize("event_id", [
    "",
    "invalid-id",
    "../other",
    "0",
    "-1",
])
def test_rejects_invalid_event_ids(event_id):
    with patch("src.ui.realtime_match_events.urlopen") as mocked:
        with pytest.raises(ValueError):
            fetch_match_plays("test-key", event_id)

    mocked.assert_not_called()


def test_rejects_invalid_provider_response():
    with patch(
        "src.ui.realtime_match_events.urlopen",
        return_value=_response({}, success=False),
    ):
        with pytest.raises(ValueError):
            fetch_match_plays("test-key", "401915395")


def test_rejects_malformed_records():
    with patch(
        "src.ui.realtime_match_events.urlopen",
        return_value=_response(["invalid-record"]),
    ):
        with pytest.raises(ValueError):
            fetch_match_plays("test-key", "401915395")


# These synthetic fixture records are test data, not actual match incidents.
def _sample_fixture():
    return {
        "home_team": "Como 1907",
        "away_team": "Manchester United",
        "kickoff_utc": "2026-10-21T16:45:00+00:00",
    }


def _sample_provider_event():
    return {
        "id": 401915395,
        "homeTeam": {"name": "Como"},
        "awayTeam": {"name": "Manchester United"},
        "date": "2026-10-21T16:45:00Z",
        "competition": {"playByPlayAvailable": True},
    }


def test_resolves_verified_fixture_event_id():
    assert match_fixture_event_id(
        _sample_fixture(), [_sample_provider_event()]
    ) == "401915395"


def test_accepts_top_level_availability_and_alternate_event_id():
    event = _sample_provider_event()
    event.pop("id")
    event.pop("competition")
    event["eventId"] = "401915395"
    event["playByPlayAvailable"] = True
    assert match_fixture_event_id(_sample_fixture(), [event]) == "401915395"


def test_rejects_reversed_fixture():
    event = _sample_provider_event()
    event["homeTeam"], event["awayTeam"] = event["awayTeam"], event["homeTeam"]
    assert match_fixture_event_id(_sample_fixture(), [event]) is None


def test_rejects_wrong_kickoff():
    event = _sample_provider_event()
    event["date"] = "2026-10-22T16:45:00Z"
    assert match_fixture_event_id(_sample_fixture(), [event]) is None


@pytest.mark.parametrize("availability", [False, None, "true", 1])
def test_rejects_unconfirmed_play_by_play(availability):
    event = _sample_provider_event()
    event["competition"]["playByPlayAvailable"] = availability
    assert match_fixture_event_id(_sample_fixture(), [event]) is None


def test_rejects_ambiguous_provider_matches():
    first = _sample_provider_event()
    second = _sample_provider_event()
    second["id"] = 999999999
    assert match_fixture_event_id(_sample_fixture(), [first, second]) is None


@pytest.mark.parametrize("bad_id", [True, "-2", "x", "1/2", None])
def test_rejects_invalid_provider_event_ids(bad_id):
    event = _sample_provider_event()
    event["id"] = bad_id
    assert match_fixture_event_id(_sample_fixture(), [event]) is None


def test_ignores_missing_or_malformed_provider_fields():
    assert match_fixture_event_id(
        _sample_fixture(), [{}, {"homeTeam": "Como"}, {**_sample_provider_event(), "date": None}]
    ) is None


def test_requires_timezone_aware_fixture_kickoff():
    fixture = _sample_fixture()
    fixture["kickoff_utc"] = "2026-10-21T16:45:00"
    assert match_fixture_event_id(fixture, [_sample_provider_event()]) is None


def test_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        match_fixture_event_id(_sample_fixture(), [_sample_provider_event()], kickoff_tolerance_minutes=-1)
