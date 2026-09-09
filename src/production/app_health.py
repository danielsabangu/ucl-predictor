from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.production.app_sources import (
    AppDataSource,
    resolve_app_data_source,
)


# ============================================================
# SETTINGS
# ============================================================

# Fully fresh production data.
FRESH_HOURS = 6.0

# After this age, forecasts are blocked until refreshed.
MAX_SERVE_HOURS = 24.0


# ============================================================
# TIME
# ============================================================


def utc_now() -> pd.Timestamp:

    return pd.Timestamp.now(
        tz="UTC"
    )


def normalize_utc_timestamp(
    value,
) -> pd.Timestamp | None:

    if value is None:

        return None

    try:

        timestamp = pd.Timestamp(
            value
        )

    except Exception:

        return None

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

    return timestamp


# ============================================================
# DISPLAY TIME
# ============================================================


def human_age(
    minutes: float | None,
) -> str:

    if minutes is None:

        return "UNKNOWN"

    if minutes < 1:

        return "<1 MIN"

    if minutes < 60:

        return (
            f"{int(minutes)} MIN"
        )

    hours = (
        minutes
        /
        60.0
    )

    if hours < 24:

        return (
            f"{hours:.1f} H"
        )

    days = (
        hours
        /
        24.0
    )

    return (
        f"{days:.1f} D"
    )


def human_countdown(
    minutes: float | None,
) -> str:

    if minutes is None:

        return "NONE"

    if minutes <= 0:

        return "STARTED"

    if minutes < 60:

        return (
            f"{int(minutes)} MIN"
        )

    hours = int(
        minutes
        //
        60
    )

    remaining_minutes = int(
        minutes
        %
        60
    )

    if remaining_minutes == 0:

        return (
            f"{hours}H"
        )

    return (
        f"{hours}H "
        f"{remaining_minutes}M"
    )


# ============================================================
# FILE MODIFICATION TIME
# ============================================================


def file_modified_utc(
    path: Path,
) -> pd.Timestamp | None:

    if not path.exists():

        return None

    try:

        return pd.Timestamp(
            datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            )
        )

    except Exception:

        return None


# ============================================================
# SOURCE
# ============================================================


def resolve_source_safe() -> tuple[
    AppDataSource | None,
    str | None,
]:

    try:

        source = (
            resolve_app_data_source()
        )

        return (
            source,
            None,
        )

    except Exception as error:

        return (
            None,
            str(
                error
            ),
        )


# ============================================================
# PIPELINE STATUS
# ============================================================


def load_pipeline_status(
    source: AppDataSource,
) -> dict:

    path = (
        source.status_file
    )

    if not path.exists():

        return {
            "available": False,
            "error": (
                "PIPELINE_STATUS_FILE_MISSING"
            ),
            "detail": None,
            "payload": None,
        }

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(
                file
            )

    except Exception as error:

        return {
            "available": False,
            "error": (
                "PIPELINE_STATUS_FILE_INVALID"
            ),
            "detail": str(
                error
            ),
            "payload": None,
        }

    if not isinstance(
        payload,
        dict,
    ):

        return {
            "available": False,
            "error": (
                "PIPELINE_STATUS_PAYLOAD_INVALID"
            ),
            "detail": (
                "Pipeline status JSON is not an object."
            ),
            "payload": None,
        }

    return {
        "available": True,
        "error": None,
        "detail": None,
        "payload": payload,
    }


# ============================================================
# PREDICTION ARTIFACT
# ============================================================


def load_prediction_schedule(
    source: AppDataSource,
) -> dict:

    path = (
        source.predictions_file
    )

    if not path.exists():

        return {
            "available": False,
            "error": (
                "PREDICTION_ARTIFACT_MISSING"
            ),
            "detail": None,
            "dataframe": None,
        }

    try:

        dataframe = pd.read_csv(
            path,
            usecols=[
                "kickoff_utc",
                "home_team",
                "away_team",
            ],
        )

    except Exception as error:

        return {
            "available": False,
            "error": (
                "PREDICTION_ARTIFACT_INVALID"
            ),
            "detail": str(
                error
            ),
            "dataframe": None,
        }

    if dataframe.empty:

        return {
            "available": False,
            "error": (
                "PREDICTION_ARTIFACT_EMPTY"
            ),
            "detail": None,
            "dataframe": dataframe,
        }

    try:

        dataframe[
            "kickoff_utc"
        ] = pd.to_datetime(
            dataframe[
                "kickoff_utc"
            ],
            errors="raise",
            utc=True,
        )

    except Exception as error:

        return {
            "available": False,
            "error": (
                "PREDICTION_KICKOFF_INVALID"
            ),
            "detail": str(
                error
            ),
            "dataframe": dataframe,
        }

    dataframe = (
        dataframe
        .sort_values(
            by=[
                "kickoff_utc",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return {
        "available": True,
        "error": None,
        "detail": None,
        "dataframe": dataframe,
    }


# ============================================================
# UPCOMING FIXTURES
# ============================================================


def filter_upcoming_predictions(
    dataframe: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:

    if now is None:

        now = utc_now()

    now = normalize_utc_timestamp(
        now
    )

    if now is None:

        raise ValueError(
            "Could not normalize current time."
        )

    output = dataframe.copy()

    output[
        "kickoff_utc"
    ] = pd.to_datetime(
        output[
            "kickoff_utc"
        ],
        errors="raise",
        utc=True,
    )

    output = output[
        output[
            "kickoff_utc"
        ]
        >
        now
    ].copy()

    return (
        output
        .sort_values(
            by=[
                "kickoff_utc",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HEALTH RESULT BASE
# ============================================================


def base_health(
    *,
    source: AppDataSource | None,
) -> dict:

    return {
        "state": (
            "UNKNOWN"
        ),

        "severity": (
            "RED"
        ),

        "production_ready": (
            False
        ),

        "can_serve_predictions": (
            False
        ),

        "refresh_recommended": (
            True
        ),

        "reason": (
            "UNKNOWN"
        ),

        "warnings": [],

        "pipeline_status": (
            "UNKNOWN"
        ),

        "run_id": (
            None
        ),

        "finished_at_utc": (
            None
        ),

        "pipeline_age_minutes": (
            None
        ),

        "pipeline_age_display": (
            "UNKNOWN"
        ),

        "artifact_age_minutes": (
            None
        ),

        "prediction_artifact_available": (
            False
        ),

        "fixture_count": (
            0
        ),

        "upcoming_fixture_count": (
            0
        ),

        "next_kickoff_utc": (
            None
        ),

        "next_home_team": (
            None
        ),

        "next_away_team": (
            None
        ),

        "minutes_to_next_kickoff": (
            None
        ),

        "next_kickoff_display": (
            "NONE"
        ),

        "source_mode": (
            source.mode
            if source is not None
            else
            "UNAVAILABLE"
        ),

        "is_public_snapshot": (
            source.is_public_snapshot
            if source is not None
            else
            False
        ),
    }


# ============================================================
# HEALTH EVALUATION
# ============================================================


def evaluate_app_health(
    *,
    now: pd.Timestamp | None = None,
    source: AppDataSource | None = None,
) -> dict:

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if now is None:

        now = utc_now()

    now = normalize_utc_timestamp(
        now
    )

    if now is None:

        raise ValueError(
            "Could not normalize application time."
        )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    if source is None:

        source, source_error = (
            resolve_source_safe()
        )

    else:

        source_error = None

    if source is None:

        health = base_health(
            source=None
        )

        health.update(
            {
                "state": (
                    "OFFLINE"
                ),

                "severity": (
                    "RED"
                ),

                "reason": (
                    "APP_DATA_SOURCE_UNAVAILABLE"
                ),

                "warnings": [
                    (
                        source_error
                        or
                        "APP_DATA_SOURCE_UNAVAILABLE"
                    )
                ],
            }
        )

        return health

    health = base_health(
        source=source
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    status_result = (
        load_pipeline_status(
            source
        )
    )

    prediction_result = (
        load_prediction_schedule(
            source
        )
    )

    # --------------------------------------------------------
    # STATUS INVALID
    # --------------------------------------------------------

    if not status_result[
        "available"
    ]:

        health.update(
            {
                "state": (
                    "OFFLINE"
                ),

                "severity": (
                    "RED"
                ),

                "reason": (
                    status_result[
                        "error"
                    ]
                ),

                "warnings": [
                    status_result[
                        "error"
                    ]
                ],

                "prediction_artifact_available": (
                    prediction_result[
                        "available"
                    ]
                ),
            }
        )

        return health

    payload = (
        status_result[
            "payload"
        ]
    )

    pipeline_status = (
        str(
            payload.get(
                "status",
                "UNKNOWN",
            )
        )
        .strip()
        .upper()
    )

    production_ready = bool(
        payload.get(
            "production_ready",
            False,
        )
    )

    run_id = payload.get(
        "run_id"
    )

    finished_at = (
        normalize_utc_timestamp(
            payload.get(
                "finished_at_utc"
            )
        )
    )

    health[
        "pipeline_status"
    ] = pipeline_status

    health[
        "production_ready"
    ] = production_ready

    health[
        "run_id"
    ] = run_id

    health[
        "finished_at_utc"
    ] = (
        finished_at.isoformat()
        if finished_at is not None
        else
        None
    )

    # --------------------------------------------------------
    # RUNNING
    # --------------------------------------------------------

    if pipeline_status == "RUNNING":

        health.update(
            {
                "state": (
                    "REFRESHING"
                ),

                "severity": (
                    "AMBER"
                ),

                "production_ready": (
                    False
                ),

                "can_serve_predictions": (
                    False
                ),

                "refresh_recommended": (
                    False
                ),

                "reason": (
                    "PIPELINE_RUNNING"
                ),

                "warnings": [
                    "PIPELINE_REFRESH_IN_PROGRESS"
                ],

                "pipeline_age_display": (
                    "REFRESHING"
                ),

                "next_kickoff_display": (
                    "REFRESHING"
                ),
            }
        )

        return health

    # --------------------------------------------------------
    # FAILED
    # --------------------------------------------------------

    if (
        pipeline_status
        !=
        "SUCCESS"
        or
        not production_ready
    ):

        error_payload = payload.get(
            "error"
        )

        if isinstance(
            error_payload,
            dict,
        ):

            error_message = (
                error_payload.get(
                    "message"
                )
                or
                "PIPELINE_FAILED"
            )

        else:

            error_message = (
                "PIPELINE_FAILED"
            )

        health.update(
            {
                "state": (
                    "FAILED"
                ),

                "severity": (
                    "RED"
                ),

                "production_ready": (
                    False
                ),

                "can_serve_predictions": (
                    False
                ),

                "refresh_recommended": (
                    True
                ),

                "reason": (
                    error_message
                ),

                "warnings": [
                    "LATEST_PRODUCTION_RUN_FAILED"
                ],

                "pipeline_age_display": (
                    "FAILED"
                ),

                "next_kickoff_display": (
                    "BLOCKED"
                ),
            }
        )

        return health

    # --------------------------------------------------------
    # PREDICTION FILE INVALID
    # --------------------------------------------------------

    if not prediction_result[
        "available"
    ]:

        health.update(
            {
                "state": (
                    "FAILED"
                ),

                "severity": (
                    "RED"
                ),

                "can_serve_predictions": (
                    False
                ),

                "refresh_recommended": (
                    True
                ),

                "reason": (
                    prediction_result[
                        "error"
                    ]
                ),

                "warnings": [
                    prediction_result[
                        "error"
                    ]
                ],
            }
        )

        return health

    predictions = (
        prediction_result[
            "dataframe"
        ]
    )

    fixture_count = int(
        len(
            predictions
        )
    )

    health[
        "prediction_artifact_available"
    ] = True

    health[
        "fixture_count"
    ] = fixture_count

    # --------------------------------------------------------
    # UPCOMING
    # --------------------------------------------------------

    upcoming = (
        filter_upcoming_predictions(
            predictions,
            now=now,
        )
    )

    upcoming_count = int(
        len(
            upcoming
        )
    )

    health[
        "upcoming_fixture_count"
    ] = upcoming_count

    # --------------------------------------------------------
    # PIPELINE AGE
    # --------------------------------------------------------

    warnings: list[str] = []

    if finished_at is not None:

        pipeline_age_minutes = float(
            (
                now
                -
                finished_at
            )
            .total_seconds()
            /
            60.0
        )

        pipeline_age_minutes = max(
            0.0,
            pipeline_age_minutes,
        )

    else:

        pipeline_age_minutes = None

        warnings.append(
            "PIPELINE_FINISH_TIME_UNKNOWN"
        )

    pipeline_age_hours = (
        pipeline_age_minutes
        /
        60.0
        if pipeline_age_minutes
        is not None
        else
        None
    )

    health[
        "pipeline_age_minutes"
    ] = pipeline_age_minutes

    health[
        "pipeline_age_display"
    ] = human_age(
        pipeline_age_minutes
    )

    # --------------------------------------------------------
    # ARTIFACT AGE
    # --------------------------------------------------------

    artifact_modified = (
        file_modified_utc(
            source.predictions_file
        )
    )

    if artifact_modified is not None:

        artifact_age_minutes = float(
            (
                now
                -
                artifact_modified
            )
            .total_seconds()
            /
            60.0
        )

        artifact_age_minutes = max(
            0.0,
            artifact_age_minutes,
        )

    else:

        artifact_age_minutes = None

    health[
        "artifact_age_minutes"
    ] = artifact_age_minutes

    # --------------------------------------------------------
    # NEXT FIXTURE
    # --------------------------------------------------------

    if upcoming_count > 0:

        next_fixture = (
            upcoming.iloc[
                0
            ]
        )

        next_kickoff = pd.Timestamp(
            next_fixture[
                "kickoff_utc"
            ]
        )

        minutes_to_next = float(
            (
                next_kickoff
                -
                now
            )
            .total_seconds()
            /
            60.0
        )

        next_home_team = str(
            next_fixture[
                "home_team"
            ]
        )

        next_away_team = str(
            next_fixture[
                "away_team"
            ]
        )

        health[
            "next_kickoff_utc"
        ] = next_kickoff.isoformat()

        health[
            "next_home_team"
        ] = next_home_team

        health[
            "next_away_team"
        ] = next_away_team

        health[
            "minutes_to_next_kickoff"
        ] = minutes_to_next

        health[
            "next_kickoff_display"
        ] = human_countdown(
            minutes_to_next
        )

    else:

        next_kickoff = None
        minutes_to_next = None

    # --------------------------------------------------------
    # NO UPCOMING FIXTURES
    # --------------------------------------------------------

    if upcoming_count == 0:

        warnings.append(
            "NO_UPCOMING_FIXTURES_IN_ARTIFACT"
        )

        health.update(
            {
                "state": (
                    "EXPIRED"
                ),

                "severity": (
                    "RED"
                ),

                "can_serve_predictions": (
                    False
                ),

                "refresh_recommended": (
                    True
                ),

                "reason": (
                    "NO_UPCOMING_FIXTURES"
                ),

                "warnings": (
                    warnings
                ),

                "next_kickoff_display": (
                    "NONE"
                ),
            }
        )

        return health

    # --------------------------------------------------------
    # HARD STALE
    # --------------------------------------------------------

    if (
        pipeline_age_hours is None
        or
        pipeline_age_hours
        >
        MAX_SERVE_HOURS
    ):

        warnings.append(
            "PRODUCTION_ARTIFACT_STALE"
        )

        health.update(
            {
                "state": (
                    "STALE"
                ),

                "severity": (
                    "RED"
                ),

                "can_serve_predictions": (
                    False
                ),

                "refresh_recommended": (
                    True
                ),

                "reason": (
                    "PIPELINE_TOO_OLD"
                ),

                "warnings": (
                    warnings
                ),
            }
        )

        return health

    # --------------------------------------------------------
    # HEALTHY / AGING
    # --------------------------------------------------------

    state = (
        "HEALTHY"
    )

    severity = (
        "GREEN"
    )

    refresh_recommended = (
        False
    )

    if (
        pipeline_age_hours
        is not None
        and
        pipeline_age_hours
        >
        FRESH_HOURS
    ):

        state = (
            "AGING"
        )

        severity = (
            "AMBER"
        )

        refresh_recommended = (
            True
        )

        warnings.append(
            "REFRESH_RECOMMENDED"
        )

    # --------------------------------------------------------
    # CLOSE TO KICKOFF
    # --------------------------------------------------------

    if (
        minutes_to_next
        is not None
        and
        minutes_to_next
        <=
        120
        and
        pipeline_age_hours
        is not None
        and
        pipeline_age_hours
        >
        2.0
    ):

        state = (
            "AGING"
        )

        severity = (
            "AMBER"
        )

        refresh_recommended = (
            True
        )

        if (
            "MATCH_CLOSE_REFRESH_RECOMMENDED"
            not in warnings
        ):

            warnings.append(
                "MATCH_CLOSE_REFRESH_RECOMMENDED"
            )

    health.update(
        {
            "state": (
                state
            ),

            "severity": (
                severity
            ),

            "production_ready": (
                True
            ),

            "can_serve_predictions": (
                True
            ),

            "refresh_recommended": (
                refresh_recommended
            ),

            "reason": (
                "PRODUCTION_ARTIFACT_VALID"
            ),

            "warnings": (
                warnings
            ),
        }
    )

    return health


# ============================================================
# CLI REPORT
# ============================================================


def print_health_report(
    health: dict,
) -> None:

    print(
        "=" * 78
    )

    print(
        "UCL PREDICTOR - APPLICATION HEALTH"
    )

    print(
        "=" * 78
    )

    print(
        f"\nData source:          "
        f"{health['source_mode']}"
    )

    print(
        f"Public snapshot:      "
        f"{health['is_public_snapshot']}"
    )

    print(
        f"\nState:                "
        f"{health['state']}"
    )

    print(
        f"Severity:             "
        f"{health['severity']}"
    )

    print(
        f"Pipeline status:      "
        f"{health['pipeline_status']}"
    )

    print(
        f"Production ready:     "
        f"{health['production_ready']}"
    )

    print(
        f"Serve predictions:    "
        f"{health['can_serve_predictions']}"
    )

    print(
        f"Refresh recommended:  "
        f"{health['refresh_recommended']}"
    )

    print(
        f"\nRun ID:               "
        f"{health['run_id']}"
    )

    print(
        f"Pipeline age:         "
        f"{health['pipeline_age_display']}"
    )

    print(
        f"\nFixtures in artifact: "
        f"{health['fixture_count']}"
    )

    print(
        f"Upcoming fixtures:    "
        f"{health['upcoming_fixture_count']}"
    )

    print(
        f"Next kickoff:         "
        f"{health['next_kickoff_utc']}"
    )

    print(
        f"Countdown:            "
        f"{health['next_kickoff_display']}"
    )

    print(
        f"Next fixture:         "
        f"{health['next_home_team']} "
        f"vs "
        f"{health['next_away_team']}"
    )

    print(
        "\nWarnings:"
    )

    warnings = health.get(
        "warnings",
        [],
    )

    if warnings:

        for warning in warnings:

            print(
                f"  - {warning}"
            )

    else:

        print(
            "  NONE"
        )

    print(
        "\n"
        + "=" * 78
    )

    if health[
        "can_serve_predictions"
    ]:

        print(
            "APPLICATION MAY SERVE PREDICTIONS ✓"
        )

    else:

        print(
            "APPLICATION MUST BLOCK PREDICTIONS ✗"
        )

    print(
        "=" * 78
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    health = (
        evaluate_app_health()
    )

    print_health_report(
        health
    )


if __name__ == "__main__":

    main()