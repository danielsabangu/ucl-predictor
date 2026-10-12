from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from .archive import load_records
from .result_schemas import (
    RESULT_SCHEMA_VERSION,
    ResultRecord,
)
from .results_archive import append_results
from .schemas import normalize_utc


DEFAULT_PREDICTION_ARCHIVE = Path(
    "data/app/history/predictions.jsonl"
)

DEFAULT_RESULT_ARCHIVE = Path(
    "data/app/history/results.jsonl"
)

DEFAULT_FIXTURE_SOURCE = Path(
    "data/interim/matches/"
    "ucl_2026_2027_fixtures_resolved.csv"
)


# Explicit deterministic aliases only.
#
# No fuzzy matching is allowed here.
TEAM_ALIASES = {
    "shakhtar": "shakhtar donetsk",
    "pae aek": "aek athens",
    "real betis": "real betis balompié",
    "viking": "viking fk",
    "sabah": "sabah fk",
    "slovan bratislava": "šk slovan bratislava",
    "fenerbahce": "fenerbahçe",
}


@dataclass(frozen=True)
class ResolutionSummary:
    finished_seen: int
    matched_finished: int
    unmatched_finished: int

    imported: int
    skipped: int
    total_results: int


def _clean_text(
    value,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.casefold() in {
        "nan",
        "none",
        "null",
        "nat",
    }:
        return None

    return text


def canonical_team_key(
    value: str,
) -> str:
    cleaned = _clean_text(value)

    if cleaned is None:
        raise ValueError(
            "Team name cannot be blank."
        )

    normalized = " ".join(
        unicodedata.normalize(
            "NFKC",
            cleaned,
        )
        .casefold()
        .split()
    )

    return TEAM_ALIASES.get(
        normalized,
        normalized,
    )


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _result_id(
    *,
    source_match_id: str,
    fixture_id: str,
) -> str:
    canonical = "|".join(
        [
            source_match_id,
            fixture_id,
        ]
    )

    return (
        "result_"
        + _sha256_text(canonical)[:24]
    )


def _parse_integer(
    value,
    *,
    name: str,
) -> int:
    cleaned = _clean_text(value)

    if cleaned is None:
        raise ValueError(
            f"{name} is missing."
        )

    numeric = float(cleaned)

    if not math.isfinite(numeric):
        raise ValueError(
            f"{name} must be finite."
        )

    if not numeric.is_integer():
        raise ValueError(
            f"{name} must represent an integer."
        )

    integer = int(numeric)

    if integer < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return integer


def _optional_integer(
    value,
) -> int | None:
    cleaned = _clean_text(value)

    if cleaned is None:
        return None

    numeric = float(cleaned)

    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
    ):
        raise ValueError(
            "Optional integer field contains "
            "an invalid value."
        )

    return int(numeric)


def _determine_result(
    *,
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def _load_fixture_rows(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Resolved fixture source is missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                "Resolved fixture CSV has no header."
            )

        return list(reader)


def _source_row_hash(
    *,
    row: dict[str, str],
    fixture_id: str,
) -> str:
    payload = {
        "fixture_id": fixture_id,
        "row": row,
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return _sha256_text(canonical)


def _fixture_key(
    *,
    kickoff_utc: str,
    home_team: str,
    away_team: str,
) -> tuple[str, str, str]:
    return (
        normalize_utc(kickoff_utc),
        canonical_team_key(home_team),
        canonical_team_key(away_team),
    )


def resolve_finished_results(
    *,
    prediction_archive: str | Path = (
        DEFAULT_PREDICTION_ARCHIVE
    ),
    fixture_source: str | Path = (
        DEFAULT_FIXTURE_SOURCE
    ),
    result_archive: str | Path = (
        DEFAULT_RESULT_ARCHIVE
    ),
) -> ResolutionSummary:
    prediction_archive = Path(
        prediction_archive
    )

    fixture_source = Path(
        fixture_source
    )

    result_archive = Path(
        result_archive
    )

    predictions = load_records(
        prediction_archive
    )

    if not predictions:
        raise ValueError(
            "Prediction archive is empty."
        )

    prediction_index: dict[
        tuple[str, str, str],
        list,
    ] = {}

    for prediction in predictions:
        key = _fixture_key(
            kickoff_utc=prediction.kickoff_utc,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
        )

        prediction_index.setdefault(
            key,
            [],
        ).append(prediction)

    fixture_rows = _load_fixture_rows(
        fixture_source
    )

    result_records: list[ResultRecord] = []

    finished_seen = 0
    matched_finished = 0
    unmatched_finished = 0

    for row in fixture_rows:
        status = (
            _clean_text(
                row.get("status")
            )
            or ""
        ).upper()

        if status != "FINISHED":
            continue

        finished_seen += 1

        kickoff_raw = _clean_text(
            row.get("date_utc")
        )

        home_team = _clean_text(
            row.get("home_team")
        )

        away_team = _clean_text(
            row.get("away_team")
        )

        if (
            kickoff_raw is None
            or home_team is None
            or away_team is None
        ):
            raise ValueError(
                "FINISHED fixture is missing "
                "identity fields."
            )

        kickoff_utc = normalize_utc(
            kickoff_raw
        )

        key = _fixture_key(
            kickoff_utc=kickoff_utc,
            home_team=home_team,
            away_team=away_team,
        )

        candidates = prediction_index.get(
            key,
            [],
        )

        if not candidates:
            unmatched_finished += 1
            continue

        fixture_ids = {
            prediction.fixture_id
            for prediction in candidates
        }

        if len(fixture_ids) != 1:
            raise ValueError(
                "A provider fixture matched multiple "
                "prediction fixture IDs."
            )

        fixture_id = next(
            iter(fixture_ids)
        )

        source_match_id = _clean_text(
            row.get("source_match_id")
        )

        if source_match_id is None:
            raise ValueError(
                "Matched FINISHED fixture is missing "
                "source_match_id."
            )

        home_goals = _parse_integer(
            row.get("home_goals"),
            name="home_goals",
        )

        away_goals = _parse_integer(
            row.get("away_goals"),
            name="away_goals",
        )

        calculated_result = (
            _determine_result(
                home_goals=home_goals,
                away_goals=away_goals,
            )
        )

        provider_result = _clean_text(
            row.get("result")
        )

        if provider_result is not None:
            provider_result = (
                provider_result.upper()
            )

            if (
                provider_result
                != calculated_result
            ):
                raise ValueError(
                    "Provider result label does not "
                    "agree with the final score for "
                    f"{home_team} vs {away_team}."
                )

        last_updated_raw = _clean_text(
            row.get("last_updated")
        )

        provider_last_updated_utc = (
            normalize_utc(last_updated_raw)
            if last_updated_raw is not None
            else None
        )

        provider_source = (
            _clean_text(
                row.get("source")
            )
            or "resolved_fixture_snapshot"
        )

        record = ResultRecord(
            schema_version=(
                RESULT_SCHEMA_VERSION
            ),

            result_id=_result_id(
                source_match_id=source_match_id,
                fixture_id=fixture_id,
            ),

            source_match_id=source_match_id,
            fixture_id=fixture_id,

            home_team=home_team,
            away_team=away_team,
            kickoff_utc=kickoff_utc,

            competition=_clean_text(
                row.get("competition")
            ),

            stage=_clean_text(
                row.get("stage")
            ),

            matchday=_optional_integer(
                row.get("matchday")
            ),

            status="FINISHED",

            home_goals=home_goals,
            away_goals=away_goals,
            actual_result=calculated_result,

            venue=_clean_text(
                row.get("venue")
            ),

            provider_source=provider_source,

            provider_last_updated_utc=(
                provider_last_updated_utc
            ),

            source_row_hash=_source_row_hash(
                row=row,
                fixture_id=fixture_id,
            ),
        )

        record.validate()

        result_records.append(
            record
        )

        matched_finished += 1

    write_result = append_results(
        result_archive,
        result_records,
    )

    return ResolutionSummary(
        finished_seen=finished_seen,
        matched_finished=matched_finished,
        unmatched_finished=unmatched_finished,

        imported=write_result.imported,
        skipped=write_result.skipped,
        total_results=write_result.total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve immutable prediction-history fixtures "
            "against verified FINISHED UCL results."
        )
    )

    parser.add_argument(
        "--predictions",
        default=str(
            DEFAULT_PREDICTION_ARCHIVE
        ),
    )

    parser.add_argument(
        "--fixtures",
        default=str(
            DEFAULT_FIXTURE_SOURCE
        ),
    )

    parser.add_argument(
        "--results",
        default=str(
            DEFAULT_RESULT_ARCHIVE
        ),
    )

    args = parser.parse_args()

    summary = resolve_finished_results(
        prediction_archive=args.predictions,
        fixture_source=args.fixtures,
        result_archive=args.results,
    )

    print(
        "Prediction result resolution complete."
    )

    print(
        f"Finished fixtures seen: "
        f"{summary.finished_seen}"
    )

    print(
        f"Matched to predictions: "
        f"{summary.matched_finished}"
    )

    print(
        f"Finished without archived prediction: "
        f"{summary.unmatched_finished}"
    )

    print(
        f"Imported results: "
        f"{summary.imported}"
    )

    print(
        f"Skipped existing results: "
        f"{summary.skipped}"
    )

    print(
        f"Total archived results: "
        f"{summary.total_results}"
    )


if __name__ == "__main__":
    main()